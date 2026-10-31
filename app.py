"""Smog (PM2.5) in Pakistani cities: web app for PAI Project 1 (ETL + EDA).

Reads the clean data that pipeline.py stored in data/smog.db.
Run locally with:  streamlit run app.py
"""
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = "data/smog.db"
WHO_24H = 15  # WHO 24-hour PM2.5 guideline, µg/m³
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Smog in Pakistan", layout="wide")


@st.cache_data
def load_tables():
    con = sqlite3.connect(DB_PATH)
    clean = pd.read_sql("SELECT * FROM clean_hourly", con, parse_dates=["timestamp"])
    log = pd.read_sql("SELECT step, value FROM pipeline_log", con)
    n_rejected = pd.read_sql("SELECT COUNT(*) AS n FROM rejected_rows", con)["n"][0]
    con.close()
    return clean, log, int(n_rejected)


df, log, n_rejected = load_tables()
all_cities = sorted(df["city"].unique())

# ------------------------------------------------------------------ header
st.title(f"Smog (PM2.5) in {', '.join(all_cities[:-1])} and {all_cities[-1]}")
st.write(
    "Smog is a serious health issue in Pakistani cities, especially in winter, "
    "but ground-sensor coverage is limited. This app shows the result of an ETL "
    "pipeline and an exploratory data analysis (EDA) of hourly PM2.5 and weather data."
)

st.sidebar.header("Filter")
cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities)
if not cities:
    st.warning("Select at least one city in the sidebar.")
    st.stop()
data = df[df["city"].isin(cities)]

# ------------------------------------------------------------ key numbers
st.header("Key numbers")
cols = st.columns(len(cities))
for col, city in zip(cols, cities):
    sub = data[data["city"] == city]
    daily = sub.set_index("timestamp")["pm2_5"].resample("D").mean().dropna()
    with col:
        st.subheader(city)
        st.metric("Mean PM2.5", f"{sub['pm2_5'].mean():.1f} µg/m³")
        st.metric("Days above WHO 24-h guideline", f"{(daily > WHO_24H).mean() * 100:.0f}%")
        bad = sub["aqi_category"].isin(["Unhealthy", "Very Unhealthy", "Hazardous"])
        st.metric("Hours 'Unhealthy' or worse", f"{bad.mean() * 100:.1f}%")

# ------------------------------------------------------------------- EDA
st.header("Exploratory data analysis")
tabs = st.tabs(["Daily trend", "Seasonality", "Hour of day", "Weather", "City comparison"])

with tabs[0]:
    daily = (data.set_index("timestamp").groupby("city")["pm2_5"]
             .resample("D").mean().unstack(0))
    st.line_chart(daily, y_label="Daily mean PM2.5 (µg/m³)")

with tabs[1]:
    monthly = data.groupby(["month", "city"])["pm2_5"].mean().unstack()
    monthly.index = [MONTHS[m - 1] for m in monthly.index]
    st.bar_chart(monthly, y_label="Mean PM2.5 (µg/m³)")
    season = (data.groupby(["city", "is_smog_season"])["pm2_5"].mean()
              .unstack().rename(columns={0: "Rest of year", 1: "Smog season (Oct-Jan)"}).round(1))
    st.dataframe(season, use_container_width=True)

with tabs[2]:
    season_pick = st.radio("Season", ["Smog season (Oct-Jan)", "Rest of year"], horizontal=True)
    flag = 1 if season_pick.startswith("Smog") else 0
    hourly = (data[data["is_smog_season"] == flag]
              .groupby(["hour", "city"])["pm2_5"].mean().unstack())
    st.line_chart(hourly, x_label="Hour of day", y_label="Mean PM2.5 (µg/m³)")

with tabs[3]:
    weather = {
        "Wind speed (km/h)": "wind_speed_10m",
        "Temperature (°C)": "temperature_2m",
        "Humidity (%)": "relative_humidity_2m",
        "Surface pressure (hPa)": "surface_pressure",
    }
    label = st.selectbox("Weather variable", list(weather))
    var = weather[label]
    d = (data.set_index("timestamp").groupby("city")[["pm2_5", var]]
         .resample("D").mean().dropna().reset_index())
    st.scatter_chart(d, x=var, y="pm2_5", color="city",
                     x_label=label, y_label="Daily mean PM2.5 (µg/m³)")
    corr = (d.groupby("city")[[var, "pm2_5"]].corr(method="pearson")
            .xs("pm2_5", level=1)[var].round(2))
    st.write("Correlation with PM2.5 (daily means):")
    st.dataframe(corr.rename("correlation"), use_container_width=True)

with tabs[4]:
    summary = data.groupby("city")["pm2_5"].agg(["mean", "median", "max", "count"]).round(1)
    summary.columns = ["Mean", "Median", "Max", "Hours"]
    st.dataframe(summary, use_container_width=True)
    share = (data.groupby("city")["aqi_category"].value_counts(normalize=True)
             .unstack().fillna(0).mul(100).round(1))
    st.subheader("Share of hours in each AQI category (%)")
    st.dataframe(share, use_container_width=True)

# --------------------------------------------------------------- pipeline
st.header("How the data was prepared (ETL pipeline)")
st.markdown(
    "1. **Extract:** hourly PM2.5/PM10 (Open-Meteo Air Quality API) and weather "
    "(Open-Meteo Historical Weather API), saved as raw JSON.\n"
    "2. **Load raw:** stored unchanged in SQLite.\n"
    "3. **Validate:** every record checked with Pydantic (value ranges, PM2.5 ≤ PM10, "
    "dew point ≤ temperature). Failing records are kept with the reason.\n"
    "4. **Transform:** pandas removes duplicates, rebuilds a full hourly grid, "
    "interpolates gaps of up to 3 hours and adds time and AQI columns.\n"
    "5. **Load clean:** written to SQLite (`clean_hourly`) with a primary key (city, timestamp)."
)
c1, c2 = st.columns(2)
c1.metric("Clean hourly rows", f"{len(df):,}")
c2.metric("Rows rejected by validation", f"{n_rejected:,}")
with st.expander("Pipeline log"):
    st.dataframe(log, use_container_width=True)

st.caption(
    "Limitation: PM2.5 values are CAMS model estimates from Open-Meteo, not ground-sensor "
    "measurements, so they can under- or over-estimate local extremes."
)
