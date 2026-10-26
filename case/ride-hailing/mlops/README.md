# MetroRide MLOps: MLflow and Prefect

Sample code for the slide deck `mlflow_prefect.pptx`. It turns the phase notebooks and the Spark ML
tutorial into a tracked, scheduled pipeline:

```
nightly_pipeline(month)                                        Prefect flow
  ingest -> bronze_to_silver -> silver_to_gold -> check_gold   DuckLake lakehouse (lakehouse.py)
  -> train_and_log -> promote_if_better                        Spark ML + MLflow (ml.py)
```

| File | What it is | Slides |
|---|---|---|
| `lakehouse.py` | Compact phase 1–3 steps as functions: `ingest_month`, `transform_bronze_to_silver`, `transform_silver_to_gold`. Idempotent: file-hash and row guards, `NOT EXISTS` anti-joins, one transaction per write | Parts 1, 3 |
| `ml.py` | Spark helpers: `to_spark` (DuckLake → Spark at a snapshot), `load_training_data`, `build_stages`, `setup_mlflow` | Part 2 |
| `tutorial_mlflow.ipynb` | MLflow step by step: runs, snapshot tags, `search_runs`, leakage, a second experiment, registry and `@champion`, reproducing training rows, autologging | Part 2 |
| `pipeline.py` | The Prefect tasks and flows, the MLflow training and promotion tasks, and a command line (`run`, `backfill`, `serve`, `reset`) | Parts 3, 4 |
| `tutorial_prefect.ipynb` | Prefect step by step: one flow run, the crash-and-retry drill, re-run without duplicates, backfill with training, run history, scheduling | Parts 3, 4 |

The code reads the CSVs in `../phases/*/bronze/` and writes its own lakehouse to `mlops/lakehouse/`. It does not
touch the lakehouses of the phase notebooks.

## Setup

From this folder (`case/ride-hailing/mlops/`):

```
pip install -r requirements.txt        # or: uv pip install -r requirements.txt
```

* Spark needs **Java 17 or newer**.
* The first time pyducklake opens a catalog, DuckDB downloads the `ducklake` extension, so you need internet
  access once.

## Run order for the class

1. **MLflow** (Part 2): open `tutorial_mlflow.ipynb` and run it top to bottom (about 2 minutes). It builds the
   lakehouse if it is empty, logs runs to `mlflow.db`, and registers `metroride-arrival-wait@champion`.
   Browse the runs:

   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # http://127.0.0.1:5000
   ```

   To use a shared tracking server instead, start `mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.db`
   and set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` before starting Jupyter.

2. **Shut down the MLflow notebook's kernel.** The DuckLake catalog is one DuckDB file, and only one process
   can have it open (gotcha 1 in the slides).

3. **Prefect** (Parts 3–4): in a terminal, `prefect server start` (UI at http://127.0.0.1:4200), then
   `prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`. Open `tutorial_prefect.ipynb` and run it
   (about 3 minutes; the retry drill waits 30 seconds on purpose).

4. **Schedule** (Part 3, step 3), with the notebook kernel shut down:

   ```
   python pipeline.py serve                                              # 02:00 Asia/Bangkok every night
   prefect deployment run "metroride-nightly/nightly" -p month=2026-06   # from a second terminal
   ```

The command line works without notebooks:

```
python pipeline.py reset                  # delete the lakehouse (MLflow runs are kept)
python pipeline.py run 2026-04            # one month
python pipeline.py run 2026-05 --crash    # retry drill
python pipeline.py backfill               # April to September
```

To start completely fresh, delete `lakehouse/`, `mlflow.db` and `mlartifacts/`.

## Expected results

| Check | Value |
|---|---|
| Labelled trips (completed, with an arrival event) | 501 (train June–August 377, test September 124) |
| September MAE: baseline / zone + slot + weekday / + accept delay / LEAKY | 1.97 / 2.03 / 1.90 / 0.00 |
| September-only experiment: accept delay / + rain | 1.79 / 1.24 |
| Champion loaded by alias | MAE 1.90, same as before registering |
| Retry drill (May) | the retry adds 0 silver trips and 137 `trip_context` rows; a re-run adds nothing and creates no snapshot |
| Gold, September, dry / rainy | cancellation 4.17% / 11.11%, driver arrival wait 6.95 / 11.16 min (phase-3 README) |
| Nightly runs (July, August, September test months) | MAE about 1.81, 1.79, 1.90; September is not promoted |

Tested with Python 3.13, pyducklake 1.0.19, DuckDB 1.5, PySpark 4.2, MLflow 3.17 and Prefect 3.9.
`mlflow.spark.log_model` still takes `artifact_path=` (not `name=`) in MLflow 3.17.

## Simplifications for class

* All sources of a month are loaded together before the transforms run, so silver and gold only add rows for
  new trips and new dates. A real pipeline would also update rows when a late event arrives.
* `promote_if_better` compares MAEs measured on different test months (exercise 3 fixes this).
* `serve()` runs the flow in one long-running process; production setups use work pools and workers.
