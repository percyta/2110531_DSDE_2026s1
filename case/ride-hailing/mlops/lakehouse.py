"""MetroRide lakehouse steps as plain Python functions.

A compact version of the phase 1-3 notebooks, so that the MLflow notebook and the Prefect
flow can import the same steps:

    ingest_month(month)            CSV files of one month -> bronze   (file and row guards)
    transform_bronze_to_silver()   bronze -> silver.trips, driver_events, weather_observations,
                                   trip_context                        (NOT EXISTS anti-joins)
    transform_silver_to_gold()     silver.trip_context -> gold.daily_service_performance

Every step is idempotent: running it twice, or retrying it after a crash, adds nothing new.
Assumption for this class: all sources of a month arrive together, before the transforms run.
"""
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.csv as pacsv
from pyducklake import (
    Catalog, Schema, required, optional,
    StringType, IntegerType, TimestampTZType, DecimalType, DateType, BooleanType,
)

BKK = ZoneInfo("Asia/Bangkok")
MLOPS_DIR = Path(__file__).resolve().parent            # .../case/ride-hailing/mlops
PHASES_DIR = MLOPS_DIR.parent / "phases"               # .../case/ride-hailing/phases
LAKE_DIR = MLOPS_DIR / "lakehouse"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
AUTHOR = "mlops sample code"

TRIP_COLUMNS = ["trip_id", "driver_id", "service_zone", "requested_at", "accepted_at",
                "picked_up_at", "completed_at", "cancelled_at", "status", "fare_thb"]
EVENT_COLUMNS = ["event_id", "trip_id", "driver_id", "event_type", "event_at", "latitude", "longitude"]
WEATHER_COLUMNS = ["observed_at", "city", "time_slot", "rain_mm", "source_page"]

SOURCES = {   # file prefix -> (bronze table, raw columns, row-guard key or None)
    "trips": ("trips_raw", TRIP_COLUMNS, "trip_id"),
    "driver_events": ("driver_events_raw", EVENT_COLUMNS, None),   # bronze keeps repeated deliveries
    "weather": ("weather_raw", WEATHER_COLUMNS, None),
}

_catalog = None


# --------------------------------------------------------------------------- catalog
def get_catalog() -> Catalog:
    """Open the lakehouse (once per process) and create any missing tables.

    The catalog is a single DuckDB file: only ONE process may have it open at a time.
    Close it (close_catalog) or shut down the notebook kernel before another process uses it.
    """
    global _catalog
    if _catalog is None:
        LAKE_DIR.mkdir(exist_ok=True)
        _catalog = Catalog("metroride", str(LAKE_DIR / "metadata.ducklake"), data_path=str(LAKE_DIR / "data"))
        _catalog.set_option("data_inlining_row_limit", "0")
        _catalog.connection.execute("SET TimeZone = 'Asia/Bangkok'")
        _create_tables(_catalog)
    return _catalog


def close_catalog() -> None:
    global _catalog
    if _catalog is not None:
        _catalog.close()
        _catalog = None


def reset_lakehouse() -> None:
    """Delete the lakehouse folder and start empty."""
    close_catalog()
    if LAKE_DIR.exists():
        shutil.rmtree(LAKE_DIR)


def sql(query: str):
    """Run a SQL query on the lakehouse and return a pandas DataFrame."""
    return get_catalog().connection.sql(query).df()


def _meta():
    return [required("source_file", StringType()), required("source_row_number", IntegerType()),
            required("file_sha256", StringType()), required("ingested_at", TimestampTZType())]


def _create_tables(cat: Catalog) -> None:
    for ns in ["bronze", "silver", "gold"]:
        cat.create_namespace_if_not_exists(ns)
    for prefix, (table, columns, _) in SOURCES.items():
        cat.create_table_if_not_exists(("bronze", table),
                                       Schema.of(*[optional(c, StringType()) for c in columns], *_meta()))
    trips_fields = [
        required("trip_id", StringType()), required("driver_id", StringType()),
        required("service_zone", StringType()), required("status", StringType()),
        required("requested_at", TimestampTZType()), required("accepted_at", TimestampTZType()),
        optional("picked_up_at", TimestampTZType()), optional("completed_at", TimestampTZType()),
        optional("cancelled_at", TimestampTZType()), optional("fare_thb", DecimalType(10, 2)),
        required("service_date", DateType()), required("time_slot", StringType()),
    ]
    cat.create_table_if_not_exists(("silver", "trips"), Schema.of(*trips_fields))
    cat.create_table_if_not_exists(("silver", "driver_events"), Schema.of(
        required("event_id", StringType()), required("trip_id", StringType()),
        required("driver_id", StringType()), required("event_type", StringType()),
        required("event_at", TimestampTZType()),
        required("latitude", DecimalType(9, 5)), required("longitude", DecimalType(9, 5)),
    ))
    cat.create_table_if_not_exists(("silver", "weather_observations"), Schema.of(
        required("city", StringType()), required("service_date", DateType()),
        required("time_slot", StringType()), required("observed_at", TimestampTZType()),
        required("rain_mm", DecimalType(6, 1)), required("is_rainy", BooleanType()),
        required("source_page", StringType()),
    ))
    cat.create_table_if_not_exists(("silver", "trip_context"), Schema.of(
        *trips_fields,
        optional("driver_arrived_at", TimestampTZType()),
        optional("driver_arrival_wait_minutes", DecimalType(8, 2)),
        optional("passenger_pickup_wait_minutes", DecimalType(8, 2)),
        optional("rain_mm", DecimalType(6, 1)), optional("is_rainy", BooleanType()),
    ))
    cat.create_table_if_not_exists(("gold", "daily_service_performance"), Schema.of(
        required("service_date", DateType()), required("time_slot", StringType()),
        optional("is_rainy", BooleanType()),
        required("requests", IntegerType()), required("completed_trips", IntegerType()),
        required("cancelled_trips", IntegerType()), required("cancellation_rate_pct", DecimalType(5, 2)),
        required("gross_fare_thb", DecimalType(12, 2)),
        optional("average_passenger_pickup_wait_minutes", DecimalType(8, 2)),
        optional("average_driver_arrival_wait_minutes", DecimalType(8, 2)),
        optional("arrived_within_8_minutes_pct", DecimalType(5, 2)),
    ))


def _append(table_id, rows: pa.Table, message: str) -> int:
    """Append rows in ONE transaction: a crash inside it writes nothing."""
    if rows.num_rows == 0:
        return 0
    cat = get_catalog()
    target = cat.load_table(table_id).scan().with_limit(0).to_arrow().schema
    rows = rows.select(target.names).cast(target)            # pyducklake appends by position
    with cat.begin_transaction() as txn:
        cat.set_commit_message(message.format(n=rows.num_rows), author=AUTHOR)
        txn.load_table(table_id).append(rows)
    return rows.num_rows


# --------------------------------------------------------------------------- bronze
def source_files(month: str) -> dict[str, Path]:
    """The CSV files that exist for one month, e.g. {'trips': .../trips_2026-09.csv, ...}."""
    found = {}
    for prefix in SOURCES:
        matches = sorted(PHASES_DIR.glob(f"*/bronze/{prefix}_{month}.csv"))
        if matches:
            found[prefix] = matches[0]
    if "trips" not in found:
        raise FileNotFoundError(f"no trips file for {month} under {PHASES_DIR}")
    return found


def ingest_csv(csv_path: Path, table: str, columns: list[str], key: str | None) -> int:
    """Append one CSV to bronze.<table>. Returns the number of rows inserted."""
    cat, con = get_catalog(), get_catalog().connection
    file_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # Guard 1: the exact same file was ingested before
    if cat.load_table(("bronze", table)).scan(f"file_sha256 = '{file_hash}'").count() > 0:
        print(f"SKIP {csv_path.name}: same file already in bronze.{table}")
        return 0

    raw = pacsv.read_csv(csv_path, convert_options=pacsv.ConvertOptions(
        column_types={c: pa.string() for c in columns}, strings_can_be_null=True))
    if raw.column_names != columns:
        raise ValueError(f"unexpected header in {csv_path.name}: {raw.column_names}")
    n = raw.num_rows
    batch = (raw.append_column("source_file", pa.array([csv_path.name] * n, pa.string()))
                .append_column("source_row_number", pa.array(range(1, n + 1), pa.int32()))
                .append_column("file_sha256", pa.array([file_hash] * n, pa.string()))
                .append_column("ingested_at", pa.array([datetime.now(BKK)] * n,
                                                       pa.timestamp("us", tz="Asia/Bangkok"))))

    # Guard 2 (only for tables with a key): skip keys already in bronze, keep first copy in the file
    if key:
        con.register("incoming", batch)
        batch = con.sql(f"""
            SELECT * FROM incoming AS i
            WHERE NOT EXISTS (SELECT 1 FROM metroride.bronze.{table} AS b WHERE b.{key} = i.{key})
            QUALIFY row_number() OVER (PARTITION BY {key} ORDER BY source_row_number) = 1
            ORDER BY source_row_number""").to_arrow_table()
        con.unregister("incoming")

    added = _append(("bronze", table), batch, f"bronze: ingest {csv_path.name} ({{n}} rows)")
    print(f"OK   {csv_path.name}: {n} rows read, {added} appended to bronze.{table}")
    return added


def ingest_month(month: str) -> int:
    """Ingest every source file of one month. Returns the total number of bronze rows added."""
    total = 0
    for prefix, path in source_files(month).items():
        table, columns, key = SOURCES[prefix]
        total += ingest_csv(path, table, columns, key)
    return total


# --------------------------------------------------------------------------- silver
TRIPS_SQL = """
WITH new_rows AS (
    SELECT b.* FROM metroride.bronze.trips_raw AS b
    WHERE NOT EXISTS (SELECT 1 FROM metroride.silver.trips AS s WHERE s.trip_id = b.trip_id)
),
typed AS (
    SELECT trim(trip_id) AS trip_id, trim(driver_id) AS driver_id, trim(service_zone) AS service_zone,
           lower(trim(status)) AS status,
           TRY_CAST(requested_at AS TIMESTAMPTZ) AS requested_at,
           TRY_CAST(accepted_at  AS TIMESTAMPTZ) AS accepted_at,
           TRY_CAST(picked_up_at AS TIMESTAMPTZ) AS picked_up_at,
           TRY_CAST(completed_at AS TIMESTAMPTZ) AS completed_at,
           TRY_CAST(cancelled_at AS TIMESTAMPTZ) AS cancelled_at,
           TRY_CAST(fare_thb AS DECIMAL(10, 2))  AS fare_thb,
           ingested_at
    FROM new_rows
),
derived AS (
    SELECT *, CAST(timezone('Asia/Bangkok', requested_at) AS DATE) AS service_date,
           CASE hour(timezone('Asia/Bangkok', requested_at)) WHEN 8 THEN 'morning' WHEN 18 THEN 'evening' END
               AS time_slot
    FROM typed
)
SELECT * EXCLUDE (ingested_at) FROM derived
WHERE trip_id IS NOT NULL AND driver_id IS NOT NULL
  AND service_zone IN ('Siam', 'Asok', 'Ari', 'Silom')
  AND requested_at IS NOT NULL AND accepted_at IS NOT NULL AND requested_at <= accepted_at
  AND time_slot IS NOT NULL
  AND ((status = 'completed' AND picked_up_at IS NOT NULL AND completed_at IS NOT NULL
        AND accepted_at <= picked_up_at AND picked_up_at <= completed_at
        AND fare_thb IS NOT NULL AND fare_thb >= 0 AND cancelled_at IS NULL)
    OR (status = 'cancelled' AND cancelled_at IS NOT NULL AND accepted_at <= cancelled_at
        AND picked_up_at IS NULL AND completed_at IS NULL AND fare_thb IS NULL))
QUALIFY row_number() OVER (PARTITION BY trip_id ORDER BY ingested_at) = 1
"""

EVENTS_SQL = """
WITH typed AS (
    SELECT trim(event_id) AS event_id, trim(trip_id) AS trip_id, trim(driver_id) AS driver_id,
           trim(event_type) AS event_type, TRY_CAST(event_at AS TIMESTAMPTZ) AS event_at,
           TRY_CAST(latitude AS DECIMAL(9, 5)) AS latitude, TRY_CAST(longitude AS DECIMAL(9, 5)) AS longitude
    FROM metroride.bronze.driver_events_raw AS b
    WHERE NOT EXISTS (SELECT 1 FROM metroride.silver.driver_events AS s WHERE s.event_id = b.event_id)
),
distinct_events AS (SELECT DISTINCT * FROM typed),           -- identical repeated deliveries -> one row
unique_ids AS (                                               -- a reused id with different content is rejected
    SELECT * FROM distinct_events
    QUALIFY count(*) OVER (PARTITION BY event_id) = 1
)
SELECT e.* FROM unique_ids AS e
JOIN metroride.silver.trips AS t ON t.trip_id = e.trip_id AND t.driver_id = e.driver_id
WHERE e.event_at IS NOT NULL
  AND e.event_type IN ('accepted', 'gps_ping', 'arrived_at_pickup', 'cancelled')
  AND e.latitude BETWEEN -90 AND 90 AND e.longitude BETWEEN -180 AND 180
"""

WEATHER_SQL = """
WITH typed AS (
    SELECT trim(city) AS city, trim(time_slot) AS time_slot,
           TRY_CAST(observed_at AS TIMESTAMPTZ) AS observed_at,
           TRY_CAST(rain_mm AS DECIMAL(6, 1)) AS rain_mm, trim(source_page) AS source_page
    FROM metroride.bronze.weather_raw
)
SELECT city, CAST(timezone('Asia/Bangkok', observed_at) AS DATE) AS service_date, time_slot,
       observed_at, rain_mm, rain_mm >= 1.0 AS is_rainy, source_page
FROM typed AS w
WHERE observed_at IS NOT NULL AND rain_mm >= 0 AND time_slot IN ('morning', 'evening')
  AND NOT EXISTS (SELECT 1 FROM metroride.silver.weather_observations AS s
                  WHERE s.city = w.city AND s.time_slot = w.time_slot
                    AND s.service_date = CAST(timezone('Asia/Bangkok', w.observed_at) AS DATE))
QUALIFY row_number() OVER (PARTITION BY city, service_date, time_slot ORDER BY observed_at) = 1
"""

TRIP_CONTEXT_SQL = """
WITH arrivals AS (
    SELECT trip_id, min(event_at) AS driver_arrived_at
    FROM metroride.silver.driver_events WHERE event_type = 'arrived_at_pickup'
    GROUP BY trip_id
)
SELECT t.*,
       a.driver_arrived_at,
       CASE WHEN t.status = 'completed' AND a.driver_arrived_at IS NOT NULL THEN
           CAST(round(date_diff('second', t.requested_at, a.driver_arrived_at) / 60.0, 2) AS DECIMAL(8, 2))
       END AS driver_arrival_wait_minutes,
       CASE WHEN t.status = 'completed' THEN
           CAST(round(date_diff('second', t.requested_at, t.picked_up_at) / 60.0, 2) AS DECIMAL(8, 2))
       END AS passenger_pickup_wait_minutes,
       w.rain_mm, w.is_rainy
FROM metroride.silver.trips AS t
LEFT JOIN arrivals AS a ON a.trip_id = t.trip_id
LEFT JOIN metroride.silver.weather_observations AS w
       ON w.city = 'Bangkok' AND w.service_date = t.service_date AND w.time_slot = t.time_slot
WHERE NOT EXISTS (SELECT 1 FROM metroride.silver.trip_context AS c WHERE c.trip_id = t.trip_id)
"""


def transform_bronze_to_silver(crash_after_trips: bool = False) -> int:
    """Move new, valid bronze rows into the four silver tables. Returns rows added to trip_context.

    crash_after_trips=True raises an error after silver.trips is committed and before the other
    tables: a classroom drill showing that a retry only adds what is still missing.
    """
    con = get_catalog().connection
    n_trips = _append(("silver", "trips"), con.sql(TRIPS_SQL).to_arrow_table(), "silver: {n} trips")
    if crash_after_trips:
        raise RuntimeError(f"simulated crash after writing {n_trips} silver trips")
    n_events = _append(("silver", "driver_events"), con.sql(EVENTS_SQL).to_arrow_table(),
                       "silver: {n} driver events")
    n_weather = _append(("silver", "weather_observations"), con.sql(WEATHER_SQL).to_arrow_table(),
                        "silver: {n} weather observations")
    n_context = _append(("silver", "trip_context"), con.sql(TRIP_CONTEXT_SQL).to_arrow_table(),
                        "silver: {n} trip_context rows")
    print(f"silver: +{n_trips} trips, +{n_events} events, +{n_weather} weather, +{n_context} trip_context")
    return n_context


# --------------------------------------------------------------------------- gold
GOLD_SQL = """
SELECT service_date, time_slot, is_rainy,
       CAST(count(*) AS INTEGER)                                   AS requests,
       CAST(count(*) FILTER (status = 'completed') AS INTEGER)     AS completed_trips,
       CAST(count(*) FILTER (status = 'cancelled') AS INTEGER)     AS cancelled_trips,
       CAST(round(100.0 * count(*) FILTER (status = 'cancelled') / count(*), 2) AS DECIMAL(5, 2))
                                                                   AS cancellation_rate_pct,
       CAST(coalesce(sum(fare_thb) FILTER (status = 'completed'), 0) AS DECIMAL(12, 2)) AS gross_fare_thb,
       CAST(round(avg(passenger_pickup_wait_minutes) FILTER (status = 'completed'), 2) AS DECIMAL(8, 2))
                                                                   AS average_passenger_pickup_wait_minutes,
       CAST(round(avg(driver_arrival_wait_minutes) FILTER (status = 'completed'), 2) AS DECIMAL(8, 2))
                                                                   AS average_driver_arrival_wait_minutes,
       CAST(round(100.0 * count(*) FILTER (status = 'completed' AND driver_arrival_wait_minutes <= 8)
                  / nullif(count(driver_arrival_wait_minutes) FILTER (status = 'completed'), 0), 2)
            AS DECIMAL(5, 2))                                      AS arrived_within_8_minutes_pct
FROM metroride.silver.trip_context AS c
WHERE NOT EXISTS (SELECT 1 FROM metroride.gold.daily_service_performance AS g
                  WHERE g.service_date = c.service_date)
GROUP BY service_date, time_slot, is_rainy
ORDER BY service_date, time_slot
"""


def transform_silver_to_gold() -> int:
    """Add gold rows for service dates that are not in gold yet. Returns the number of rows added."""
    rows = get_catalog().connection.sql(GOLD_SQL).to_arrow_table()
    n = _append(("gold", "daily_service_performance"), rows, "gold: {n} daily rows")
    print(f"gold: +{n} rows")
    return n


# --------------------------------------------------------------------------- helpers
def row_counts():
    """Rows per table, for quick checks."""
    return sql("""
        SELECT 'bronze.trips_raw' AS table_name, count(*) AS n_rows FROM metroride.bronze.trips_raw
        UNION ALL SELECT 'bronze.driver_events_raw', count(*) FROM metroride.bronze.driver_events_raw
        UNION ALL SELECT 'bronze.weather_raw', count(*) FROM metroride.bronze.weather_raw
        UNION ALL SELECT 'silver.trips', count(*) FROM metroride.silver.trips
        UNION ALL SELECT 'silver.driver_events', count(*) FROM metroride.silver.driver_events
        UNION ALL SELECT 'silver.weather_observations', count(*) FROM metroride.silver.weather_observations
        UNION ALL SELECT 'silver.trip_context', count(*) FROM metroride.silver.trip_context
        UNION ALL SELECT 'gold.daily_service_performance', count(*) FROM metroride.gold.daily_service_performance
    """)


def build_all_months() -> None:
    """Load every month without an orchestrator (used by the MLflow notebook)."""
    for month in MONTHS:
        ingest_month(month)
        transform_bronze_to_silver()
        transform_silver_to_gold()
