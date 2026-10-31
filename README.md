# PAI Project 1: ETL + EDA Pipeline
**Smog (PM2.5) in Lahore, Islamabad, Karachi and Peshawar**

## Problem
Smog is a serious health issue in Pakistani cities, especially in winter, but people cannot easily get a reliable forecast of tomorrow's PM2.5 level, and ground-sensor coverage is limited.

## Data
Hourly data from August 2022 to the present (about 37,000 hours per city), local time (Asia/Karachi), from two free Open-Meteo APIs (no API key needed):
- **Air Quality API:** PM2.5 and PM10 (CAMS model estimates)
- **Historical Weather API:** temperature, humidity, dew point, surface pressure, wind speed and direction, precipitation

## How to run
```
pip install -r requirements.txt
python pipeline.py        # downloads data, validates, cleans, builds data/smog.db (a few minutes)
jupyter notebook eda.ipynb   # run all cells; figures are saved in figures/
streamlit run app.py         # web app (needs data/smog.db from the pipeline)
```
API responses are cached in `data/raw/`, so running the pipeline a second time does not download again.

## Pipeline (`pipeline.py`)
| Step | What happens | Tool |
|---|---|---|
| 1. Extract | Calls both APIs for each city in 6-month chunks, with retries for rate limits, saves raw JSON | requests |
| 2. Load raw | Stores data exactly as received | SQLite: `raw_weather`, `raw_air_quality` |
| 3. Validate | Merges the two sources by city + hour and checks every record: value ranges plus physical rules (PM2.5 ≤ PM10, dew point ≤ temperature). Failing records are saved with the reason | Pydantic → `rejected_rows` |
| 4. Transform | Removes duplicates, rebuilds a complete hourly grid, interpolates gaps of up to 3 hours, drops hours with no PM2.5, adds hour/month/season/smog-season/AQI-category columns | pandas |
| 5. Load clean | Writes the clean table with a defined schema and primary key (city, timestamp), plus a step-by-step log | SQLite: `clean_hourly`, `pipeline_log` |

**Why SQLite:** lightweight, a single file, built into Python, no server needed.

## EDA (`eda.ipynb`)
1. Data overview
2. Data quality checks: missing values, duplicates, impossible values, time gaps
3. Distributions (PM2.5, AQI categories, weather)
4. Seasonal and hourly patterns of PM2.5
5. Relationships between PM2.5 and weather, with city comparison
6. Autocorrelation / lag analysis and simple next-day baselines (for Project 2)

## Limitation
PM2.5 values are model estimates (CAMS), not ground-sensor measurements. They are used because sensor data in Pakistan is sparse, but they may under- or over-estimate local extremes.

## Web app (`app.py`)
A Streamlit app that reads `data/smog.db` and shows key numbers per city, the EDA charts (daily trend, seasonality, hour of day, weather, city comparison) and a summary of the ETL steps.

- Run locally: `streamlit run app.py`
- Live app: https://smog-cast-app-lp9ikz.streamlit.app
