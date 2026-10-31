"""
PAI Project 1: ETL + EDA Pipeline
Smog (PM2.5) forecasting for Pakistani cities: Lahore, Islamabad, Karachi, Peshawar

Steps
  1. EXTRACT    Hourly PM2.5 / PM10 from the Open-Meteo Air Quality API and hourly
                weather from the Open-Meteo Historical Weather API (Aug 2022 to present).
                Every API response is saved as raw JSON in data/raw/.
  2. LOAD RAW   Raw data is stored exactly as received in two tables:
                raw_weather and raw_air_quality.
  3. VALIDATE   Weather and air quality are merged by (city, timestamp) and every hourly
                record is checked with a Pydantic model. Records with impossible values
                go to the rejected_rows table with the reason.
  4. TRANSFORM  pandas: remove duplicates, rebuild a complete hourly time grid, fill
                short gaps (max 3 hours) by interpolation, drop hours with no PM2.5,
                add time features and an AQI category.
  5. LOAD CLEAN Cleaned data goes into the clean_hourly table. A summary of every step
                goes into pipeline_log.

Database: SQLite file at data/smog.db (lightweight, file-based, no server needed).

Install:  pip install -r requirements.txt
Run:      python pipeline.py
"""

import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# =============================================================== CONFIG
CITIES = {
    # city: (latitude, longitude)
    "Lahore": (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Peshawar": (34.0151, 71.5249),
}
START_DATE = date(2022, 8, 1)
END_DATE = date.today() - timedelta(days=7)  # historical weather archive lags a few days
CHUNK_DAYS = 180                             # request data in ~6-month pieces
TIMEZONE = "Asia/Karachi"                    # so hour-of-day patterns are in local time (PKT)
MAX_INTERPOLATE_HOURS = 3                    # only gaps of 3 hours or less are filled

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "smog.db"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m",        # °C
    "relative_humidity_2m",  # %
    "dew_point_2m",          # °C
    "surface_pressure",      # hPa
    "wind_speed_10m",        # km/h
    "wind_direction_10m",    # degrees
    "precipitation",         # mm
]
AIR_VARS = ["pm2_5", "pm10"]  # µg/m³ (modelled by CAMS, not ground sensors)
NUMERIC_COLS = AIR_VARS + WEATHER_VARS

# US EPA PM2.5 categories (24-hour breakpoints, 2024 revision), in µg/m³
AQI_BINS = [-0.1, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
              "Unhealthy", "Very Unhealthy", "Hazardous"]


# =============================================================== 1. EXTRACT
def date_chunks(start: date, end: date, days: int):
    """Split start..end into (chunk_start, chunk_end) pieces of `days` days."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_json(url: str, params: dict, retries: int = 4) -> dict:
    """GET a URL and return JSON, retrying on network errors and rate limits (HTTP 429)."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 429:
                wait = 30 * attempt
                print(f"    rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == retries:
                raise
            print(f"    request failed ({error}), retrying ...")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {url} after {retries} attempts")


def extract_source(source: str, url: str, variables: list[str]) -> pd.DataFrame:
    """Download hourly data for every city and date chunk from one Open-Meteo API.

    Each response is cached as JSON in data/raw/, so re-running the pipeline
    does not download the same data again.
    """
    frames = []
    for city, (lat, lon) in CITIES.items():
        for chunk_start, chunk_end in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
            raw_file = RAW_DIR / f"{source}_{city}_{chunk_start}_{chunk_end}.json"
            if raw_file.exists():
                payload = json.loads(raw_file.read_text())
            else:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": chunk_start.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "hourly": ",".join(variables),
                    "timezone": TIMEZONE,
                }
                payload = fetch_json(url, params)
                raw_file.write_text(json.dumps(payload))
                time.sleep(1)  # be polite to the free API

            chunk = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
            for var in variables:  # make sure every column exists even if the API skipped it
                if var not in chunk.columns:
                    chunk[var] = np.nan
            chunk.insert(0, "city", city)
            frames.append(chunk[["city", "timestamp"] + variables])
        print(f"  {source:<12} {city:<10} done")

    data = pd.concat(frames, ignore_index=True)
    data["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    return data


def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1] EXTRACT  {START_DATE} to {END_DATE}, cities: {', '.join(CITIES)}")
    weather = extract_source("weather", WEATHER_URL, WEATHER_VARS)
    air = extract_source("air_quality", AIR_URL, AIR_VARS)
    print(f"    weather rows: {len(weather):,}   air quality rows: {len(air):,}")
    return weather, air


# =============================================================== 2. LOAD RAW
def load_raw(conn: sqlite3.Connection, weather: pd.DataFrame, air: pd.DataFrame) -> None:
    """Store the data exactly as received from the APIs (no cleaning)."""
    weather.to_sql("raw_weather", conn, if_exists="replace", index=False)
    air.to_sql("raw_air_quality", conn, if_exists="replace", index=False)
    print("[2] LOAD RAW  tables raw_weather and raw_air_quality written")


# =============================================================== 3. VALIDATE (Pydantic)
class HourlyRecord(BaseModel):
    """One hour of merged weather + air quality data for one city.

    Values may be missing (None), because gaps are handled later by pandas.
    Values that are present must be physically possible.
    """
    city: str
    timestamp: datetime
    pm2_5: Optional[float] = Field(None, ge=0, le=1000)
    pm10: Optional[float] = Field(None, ge=0, le=2000)
    temperature_2m: Optional[float] = Field(None, ge=-30, le=60)
    relative_humidity_2m: Optional[float] = Field(None, ge=0, le=100)
    dew_point_2m: Optional[float] = Field(None, ge=-50, le=40)
    surface_pressure: Optional[float] = Field(None, ge=850, le=1100)
    wind_speed_10m: Optional[float] = Field(None, ge=0, le=200)
    wind_direction_10m: Optional[float] = Field(None, ge=0, le=360)
    precipitation: Optional[float] = Field(None, ge=0, le=300)

    @field_validator("city")
    @classmethod
    def city_must_be_known(cls, value: str) -> str:
        if value not in CITIES:
            raise ValueError(f"unknown city '{value}'")
        return value

    @model_validator(mode="after")
    def check_physical_consistency(self):
        # PM2.5 is part of PM10, so it can never be larger than PM10
        if self.pm2_5 is not None and self.pm10 is not None and self.pm2_5 > self.pm10 + 0.5:
            raise ValueError("pm2_5 is greater than pm10")
        # the dew point can never be above the air temperature
        if (self.dew_point_2m is not None and self.temperature_2m is not None
                and self.dew_point_2m > self.temperature_2m + 0.5):
            raise ValueError("dew point is above temperature")
        return self


def merge_sources(weather: pd.DataFrame, air: pd.DataFrame) -> pd.DataFrame:
    """Join weather and air quality on (city, timestamp). Outer join keeps hours missing in one source."""
    merged = pd.merge(
        weather.drop(columns="extracted_at"),
        air.drop(columns="extracted_at"),
        on=["city", "timestamp"],
        how="outer",
    )
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged[["city", "timestamp"] + NUMERIC_COLS]


def validate(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check every record with Pydantic. Returns (valid rows, rejected rows)."""
    # NaN -> None, so Pydantic treats missing values as missing instead of invalid
    records = merged.astype(object).where(merged.notna(), None).to_dict("records")

    valid, rejected = [], []
    for record in records:
        try:
            valid.append(HourlyRecord.model_validate(record).model_dump())
        except ValidationError as error:
            reasons = [
                {"field": ".".join(str(p) for p in e["loc"]) or "record", "message": e["msg"]}
                for e in error.errors()
            ]
            rejected.append({
                "city": record["city"],
                "timestamp": str(record["timestamp"]),
                "errors": json.dumps(reasons),
                "record_json": json.dumps(record, default=str),
            })

    valid_df = pd.DataFrame(valid, columns=["city", "timestamp"] + NUMERIC_COLS)
    rejected_df = pd.DataFrame(rejected, columns=["city", "timestamp", "errors", "record_json"])
    print(f"[3] VALIDATE  {len(records):,} records checked -> "
          f"{len(valid_df):,} valid, {len(rejected_df):,} rejected")
    return valid_df, rejected_df


# =============================================================== 4. TRANSFORM (pandas)
def season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def transform(valid: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = valid.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4a. duplicates: the same city + hour should appear only once
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="first")
    log["duplicates_removed"] = before - len(df)

    # 4b. complete hourly grid per city, then fill short gaps
    interp_cols = [c for c in NUMERIC_COLS if c != "wind_direction_10m"]
    frames, hours_added, values_filled = [], 0, 0
    for city, group in df.groupby("city"):
        group = group.set_index("timestamp").sort_index()
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="h")
        hours_added += len(full_index) - len(group)
        group = group.reindex(full_index)
        group["city"] = city

        missing_before = group[NUMERIC_COLS].isna()
        # linear-in-time interpolation, only for gaps of at most MAX_INTERPOLATE_HOURS
        group[interp_cols] = group[interp_cols].interpolate(
            method="time", limit=MAX_INTERPOLATE_HOURS, limit_area="inside")
        # wind direction is an angle (359° and 1° are neighbours), so carry it forward instead
        group["wind_direction_10m"] = group["wind_direction_10m"].ffill(limit=MAX_INTERPOLATE_HOURS)
        filled = missing_before & group[NUMERIC_COLS].notna()
        values_filled += int(filled.values.sum())
        group["is_interpolated"] = filled.any(axis=1).astype(int)

        frames.append(group.rename_axis("timestamp").reset_index())
    df = pd.concat(frames, ignore_index=True)
    log["missing_hours_added_to_grid"] = hours_added
    log["values_interpolated"] = values_filled

    # 4c. PM2.5 is our target, so hours where it is still missing are dropped
    before = len(df)
    df = df.dropna(subset=["pm2_5"])
    log["rows_dropped_no_pm25"] = before - len(df)
    log["weather_values_still_missing"] = int(df[WEATHER_VARS].isna().values.sum())

    # 4d. feature columns used in EDA (and later in Project 2)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["season"] = df["month"].map(season_of)
    df["is_smog_season"] = df["month"].isin([10, 11, 12, 1]).astype(int)
    df["aqi_category"] = pd.cut(df["pm2_5"], bins=AQI_BINS, labels=AQI_LABELS).astype(str)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].round(2)

    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    print(f"[4] TRANSFORM  {log['duplicates_removed']} duplicates removed, "
          f"{hours_added:,} missing hours added, {values_filled:,} values interpolated, "
          f"{log['rows_dropped_no_pm25']:,} rows dropped (no PM2.5)")
    return df


# =============================================================== 5. LOAD CLEAN
CLEAN_SCHEMA = """
CREATE TABLE clean_hourly (
    city                 TEXT    NOT NULL,
    timestamp            TEXT    NOT NULL,   -- local time (Asia/Karachi), 'YYYY-MM-DD HH:MM'
    pm2_5                REAL    NOT NULL,   -- µg/m³
    pm10                 REAL,               -- µg/m³
    temperature_2m       REAL,               -- °C
    relative_humidity_2m REAL,               -- %
    dew_point_2m         REAL,               -- °C
    surface_pressure     REAL,               -- hPa
    wind_speed_10m       REAL,               -- km/h
    wind_direction_10m   REAL,               -- degrees
    precipitation        REAL,               -- mm
    hour                 INTEGER,
    day_of_week          INTEGER,            -- 0 = Monday
    month                INTEGER,
    year                 INTEGER,
    season               TEXT,
    is_smog_season       INTEGER,            -- 1 for Oct, Nov, Dec, Jan
    aqi_category         TEXT,
    is_interpolated      INTEGER,            -- 1 if any value in this row was filled
    PRIMARY KEY (city, timestamp)
);
"""

REJECTED_SCHEMA = """
CREATE TABLE rejected_rows (
    city        TEXT,
    timestamp   TEXT,
    errors      TEXT,   -- JSON list of {field, message}
    record_json TEXT    -- the full record that failed
);
"""

CLEAN_COLUMNS = ["city", "timestamp"] + NUMERIC_COLS + [
    "hour", "day_of_week", "month", "year", "season",
    "is_smog_season", "aqi_category", "is_interpolated"]


def load_clean(conn: sqlite3.Connection, clean: pd.DataFrame, rejected: pd.DataFrame, log: dict) -> None:
    conn.execute("DROP TABLE IF EXISTS clean_hourly")
    conn.execute("DROP TABLE IF EXISTS rejected_rows")
    conn.execute(CLEAN_SCHEMA)
    conn.execute(REJECTED_SCHEMA)

    out = clean[CLEAN_COLUMNS].copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    out.to_sql("clean_hourly", conn, if_exists="append", index=False)
    rejected.to_sql("rejected_rows", conn, if_exists="append", index=False)

    log_df = pd.DataFrame({"step": list(log.keys()), "value": list(log.values())})
    log_df.insert(0, "run_at", datetime.now().isoformat(timespec="seconds"))
    log_df.to_sql("pipeline_log", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"[5] LOAD CLEAN  {len(out):,} rows -> clean_hourly, "
          f"{len(rejected):,} rows -> rejected_rows")


def verify(conn: sqlite3.Connection) -> None:
    """Read data back from the database to prove it was stored."""
    print("\nVERIFY: rows per city in clean_hourly")
    summary = pd.read_sql(
        """SELECT city,
                  COUNT(*)              AS rows,
                  MIN(timestamp)        AS first_hour,
                  MAX(timestamp)        AS last_hour,
                  ROUND(AVG(pm2_5), 1)  AS mean_pm2_5
           FROM clean_hourly GROUP BY city ORDER BY mean_pm2_5 DESC""", conn)
    print(summary.to_string(index=False))
    print("\nSELECT * FROM clean_hourly LIMIT 5")
    print(pd.read_sql("SELECT * FROM clean_hourly LIMIT 5", conn).to_string(index=False))


# =============================================================== MAIN
def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    weather, air = extract()

    with sqlite3.connect(DB_PATH) as conn:
        load_raw(conn, weather, air)

        log = {"raw_weather_rows": len(weather), "raw_air_quality_rows": len(air)}
        merged = merge_sources(weather, air)
        log["merged_rows"] = len(merged)

        valid, rejected = validate(merged)
        log["valid_rows"] = len(valid)
        log["rejected_rows"] = len(rejected)

        clean = transform(valid, log)
        log["clean_rows"] = len(clean)

        load_clean(conn, clean, rejected, log)
        verify(conn)

    print(f"\nDone. Database saved at {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
