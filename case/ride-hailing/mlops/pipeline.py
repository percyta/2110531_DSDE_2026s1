"""MetroRide nightly pipeline, orchestrated by Prefect, with training tracked by MLflow.

    ingest -> bronze_to_silver -> silver_to_gold -> check_gold -> train_and_log -> promote_if_better

Run it from the mlops/ folder:

    python pipeline.py run 2026-06            # one flow run now
    python pipeline.py run 2026-05 --crash    # retry drill: bronze_to_silver fails once, then succeeds
    python pipeline.py backfill               # April to September, one flow run per month
    python pipeline.py serve                  # deployment: run every night at 02:00 Bangkok time
    python pipeline.py reset                  # delete the lakehouse (MLflow runs are kept)

The lakehouse functions are the phase-notebook steps in lakehouse.py; this file only adds the Prefect
decorators (tasks, retries, flow, schedule) and the MLflow training and promotion tasks.
"""
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prefect import flow, task
from prefect.cache_policies import NONE
from prefect.schedules import Cron
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import lakehouse as lh
import ml

RETRY_DELAY_SECONDS = 30
NIGHTLY_EXPERIMENT = "metroride-nightly"
_crash_drill = {"armed": False}       # set by the flow; the first bronze_to_silver attempt then fails


# --------------------------------------------------------------------------- data tasks
# cache_policy=NONE: tasks read the lakehouse, not their arguments, so a cached result would hide new data.

@task(retries=2, retry_delay_seconds=RETRY_DELAY_SECONDS, cache_policy=NONE)
def ingest(month: str) -> int:
    return lh.ingest_month(month)


@task(retries=2, retry_delay_seconds=RETRY_DELAY_SECONDS, cache_policy=NONE)
def bronze_to_silver() -> int:
    crash = _crash_drill["armed"]
    _crash_drill["armed"] = False                      # only the first attempt crashes
    return lh.transform_bronze_to_silver(crash_after_trips=crash)


@task(cache_policy=NONE)
def silver_to_gold() -> int:
    return lh.transform_silver_to_gold()


@task(cache_policy=NONE)
def check_gold(month: str) -> None:
    """Basic quality gate: fail the flow if gold does not add up to silver for this month."""
    check = lh.sql(f"""
        SELECT (SELECT count(*) FROM metroride.silver.trip_context
                WHERE strftime(service_date, '%Y-%m') = '{month}')                      AS silver_trips,
               (SELECT coalesce(sum(requests), 0) FROM metroride.gold.daily_service_performance
                WHERE strftime(service_date, '%Y-%m') = '{month}')                      AS gold_requests,
               (SELECT count(*) - count(DISTINCT (service_date, time_slot, is_rainy))
                FROM metroride.gold.daily_service_performance)                          AS duplicate_keys
    """).iloc[0]
    if check.silver_trips == 0 or check.silver_trips != check.gold_requests or check.duplicate_keys:
        raise ValueError(f"gold check failed for {month}: {check.to_dict()}")
    print(f"gold check OK for {month}: {int(check.gold_requests)} requests")


# --------------------------------------------------------------------------- MLflow tasks
@task(cache_policy=NONE)
def train_and_log(month: str) -> tuple[str, float] | None:
    """Train on the months before `month`, test on `month`, and record the run in MLflow."""
    labelled = lh.sql(f"""
        SELECT count(*) FILTER (strftime(service_date, '%Y-%m') < '{month}') AS n_train,
               count(*) FILTER (strftime(service_date, '%Y-%m') = '{month}') AS n_test
        FROM metroride.silver.trip_context
        WHERE status = 'completed' AND driver_arrival_wait_minutes IS NOT NULL""").iloc[0]
    if labelled.n_train == 0 or labelled.n_test == 0:          # April-May have no arrival events
        print(f"skip training: no labelled trips before {month} or in {month}")
        return None

    ml.setup_mlflow(NIGHTLY_EXPERIMENT)
    snap = ml.current_snapshot_id()                            # pin the data version
    data = ml.load_training_data(snap)
    test_month = int(month[5:7])
    train_df = data.filter(F.col("month") < test_month)
    test_df = data.filter(F.col("month") == test_month)

    numeric = ["accept_delay_minutes"]
    with mlflow.start_run(run_name=f"nightly {month} (snapshot {snap})"):
        mlflow.log_params({"categorical": ml.CATEGORICAL, "numeric": numeric,
                           "algorithm": "LinearRegression", "test_month": month})
        mlflow.set_tag("ducklake.snapshot_id", snap)
        model = Pipeline(stages=ml.build_stages(ml.CATEGORICAL, numeric)).fit(train_df)
        metrics = ml.regression_metrics(model.transform(test_df))
        mlflow.log_metrics(metrics)
        info = mlflow.spark.log_model(model, artifact_path="model")
    print(f"trained on snapshot {snap}: test MAE {metrics['mae']:.3f} on {month}")
    return info.model_uri, metrics["mae"]


@task(cache_policy=NONE)
def promote_if_better(model_uri: str, mae: float) -> bool:
    """Register the new model and move @champion to it, but only if its MAE beats the champion's.

    Simplification for class: the two MAEs may come from different test months.
    """
    client = MlflowClient()
    name = ml.REGISTERED_MODEL
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        best = client.get_run(champ.run_id).data.metrics["mae"]
    except MlflowException:                                    # no champion yet
        best = float("inf")
    if round(mae, 4) >= round(best, 4):                        # ignore float noise between refits
        print(f"keep the champion: MAE {mae:.3f} is not better than {best:.3f}")
        return False
    mv = mlflow.register_model(model_uri, name)
    client.set_registered_model_alias(name, "champion", mv.version)
    print(f"promoted {name} version {mv.version} to @champion (MAE {mae:.3f}, was {best:.3f})")
    return True


# --------------------------------------------------------------------------- flows
@flow(name="metroride-nightly", log_prints=True)
def nightly_pipeline(month: str = "2026-09", train: bool = True, simulate_crash: bool = False) -> dict:
    """Load one month into the lakehouse, then retrain and maybe promote the model."""
    _crash_drill["armed"] = simulate_crash
    try:
        added = {"bronze": ingest(month), "trip_context": bronze_to_silver(), "gold": silver_to_gold()}
        check_gold(month)
        if train:
            result = train_and_log(month)
            if result is not None:
                added["promoted"] = promote_if_better(*result)
        return added
    finally:
        lh.close_catalog()            # one writer at a time: release the catalog for other processes
        spark = SparkSession.getActiveSession()
        if spark is not None:
            spark.stop()              # one SparkSession per flow run


@flow(name="metroride-backfill", log_prints=True)
def backfill(months: list[str] = lh.MONTHS) -> None:
    """Run the nightly flow once per month, in order (each one is a subflow run)."""
    for month in months:
        nightly_pipeline(month)


# --------------------------------------------------------------------------- command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one flow run now")
    run.add_argument("month", help="e.g. 2026-06")
    run.add_argument("--crash", action="store_true", help="bronze_to_silver fails on its first attempt")
    run.add_argument("--no-train", action="store_true", help="skip training and promotion")
    sub.add_parser("backfill", help="run April to September in order")
    sub.add_parser("serve", help="deploy with a 02:00 Asia/Bangkok schedule and wait for runs")
    sub.add_parser("reset", help="delete the lakehouse")
    args = parser.parse_args()

    if args.command == "run":
        print(nightly_pipeline(args.month, train=not args.no_train, simulate_crash=args.crash))
    elif args.command == "backfill":
        backfill()
    elif args.command == "serve":
        nightly_pipeline.serve(
            name="nightly",
            schedule=Cron("0 2 * * *", timezone="Asia/Bangkok"),
            parameters={"month": "2026-09"},
        )
    elif args.command == "reset":
        lh.reset_lakehouse()
        print("lakehouse deleted:", lh.LAKE_DIR)
