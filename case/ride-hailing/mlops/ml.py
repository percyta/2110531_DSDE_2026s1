"""Spark ML helpers shared by tutorial_mlflow.ipynb and pipeline.py.

The model is the one from ml/tutorial_spark_ml.ipynb: predict driver_arrival_wait_minutes for a
completed trip, right after the driver accepts. The difference: the training data now comes from
the lakehouse table silver.trip_context, read at a fixed DuckLake snapshot.
"""
import os

import mlflow
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import lakehouse as lh

LABEL = "driver_arrival_wait_minutes"
CATEGORICAL = ["service_zone", "time_slot", "weekday"]
REGISTERED_MODEL = "metroride-arrival-wait"


# --------------------------------------------------------------------------- MLflow
def setup_mlflow(experiment: str) -> str:
    """Point MLflow at a tracking store and select an experiment.

    Default: a SQLite file next to this code (no server needed). To use a tracking server, set
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 before starting Python.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{lh.MLOPS_DIR / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite") and mlflow.get_experiment_by_name(experiment) is None:
        # keep model files next to the database, whatever the current folder is
        mlflow.create_experiment(experiment, artifact_location=(lh.MLOPS_DIR / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)
    return uri


# --------------------------------------------------------------------------- Spark
def get_spark() -> SparkSession:
    spark = (SparkSession.builder
             .master("local[*]")
             .appName("metroride-mlops")
             .config("spark.sql.session.timeZone", "Asia/Bangkok")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def to_spark(table, snapshot_id=None) -> DataFrame:
    """Read a DuckLake table (optionally at an older snapshot) into a Spark DataFrame.

    The same two-line bridge as phases/phase-1-batch/pyspark/tutorial_pyspark.ipynb.
    """
    scan = table.scan() if snapshot_id is None else table.scan().with_snapshot(snapshot_id)
    return get_spark().createDataFrame(scan.to_arrow())


def current_snapshot_id() -> int:
    """The snapshot id of silver.trip_context right now."""
    return lh.get_catalog().load_table(("silver", "trip_context")).current_snapshot().snapshot_id


def load_training_data(snapshot_id: int) -> DataFrame:
    """One row per completed trip with an arrival event: features + label.

    Every feature is known when the driver accepts. passenger_pickup_wait_minutes is kept only for the
    leakage exercise; it must never be a feature.
    """
    tc = lh.get_catalog().load_table(("silver", "trip_context"))
    return (to_spark(tc, snapshot_id)
            .filter((F.col("status") == "completed") & F.col(LABEL).isNotNull())
            .select(
                "trip_id", "service_date",
                F.month("service_date").alias("month"),
                "service_zone", "time_slot",
                F.date_format("requested_at", "EEE").alias("weekday"),
                ((F.unix_timestamp("accepted_at") - F.unix_timestamp("requested_at")) / 60)
                .alias("accept_delay_minutes"),
                F.col("is_rainy").cast("double").alias("is_rainy"),
                F.col("passenger_pickup_wait_minutes").cast("double").alias("passenger_pickup_wait_minutes"),
                F.col(LABEL).cast("double").alias(LABEL),
            )
            .cache())


def build_stages(categorical: list[str], numeric: list[str]) -> list:
    """The pipeline stages of the Spark ML tutorial: index + one-hot the categories, assemble, regress."""
    stages = []
    if categorical:
        stages += [
            StringIndexer(inputCols=categorical, outputCols=[c + "_idx" for c in categorical],
                          handleInvalid="keep"),
            OneHotEncoder(inputCols=[c + "_idx" for c in categorical],
                          outputCols=[c + "_ohe" for c in categorical]),
        ]
    stages += [
        VectorAssembler(inputCols=numeric + [c + "_ohe" for c in categorical], outputCol="features"),
        LinearRegression(featuresCol="features", labelCol=LABEL),
    ]
    return stages


def regression_metrics(predictions: DataFrame) -> dict[str, float]:
    return {m: RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName=m)
            .evaluate(predictions) for m in ["mae", "rmse", "r2"]}
