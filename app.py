"""Smog (PM2.5) in Pakistani cities: web app for PAI Project 1 (ETL + EDA).

Reads the clean data that pipeline.py stored in data/smog.db.
Run locally with:  streamlit run app.py
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

DB_PATH = Path(__file__).parent / "data" / "smog.db"
WHO_24H = 15  # WHO 24-hour PM2.5 guideline, µg/m³
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BAD_AQI = ["Unhealthy", "Very Unhealthy", "Hazardous"]

# One fixed colour per city (first three slots of a colour-blind-safe palette).
CITY_COLORS = {"Islamabad": "#2a78d6", "Karachi": "#1baf7a", "Lahore": "#eb6834"}
AQI_ORDER = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
             "Unhealthy", "Very Unhealthy", "Hazardous"]
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
AQI_DARK_TEXT = {"Good", "Moderate", "Unhealthy for Sensitive Groups"}
INK, MUTED, GRID, AXIS = "#1f2430", "#6b6a66", "#ecebe6", "#cfcdc6"

st.set_page_config(page_title="Smog in Pakistan", page_icon="🌫️", layout="wide")

CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1180px;}
footer{visibility:hidden;}
button[data-baseweb="tab"]{font-size:1rem;font-weight:600;}
.hero{background:linear-gradient(120deg,#16324f 0%,#2a78d6 100%);border-radius:18px;padding:30px 34px;color:#fff;margin-bottom:18px;}
.hero-kicker{font-size:.78rem;letter-spacing:.14em;opacity:.8;font-weight:600;}
.hero-title{font-size:2.05rem;font-weight:800;line-height:1.2;margin:6px 0 10px;}
.hero-sub{font-size:1.02rem;opacity:.93;max-width:780px;line-height:1.55;}
.chips{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px;}
.chip{background:rgba(255,255,255,.17);border-radius:999px;padding:5px 13px;font-size:.84rem;}
.glance{background:#fff;border:1px solid #e7e6e2;border-radius:14px;padding:14px 18px;color:#1f2430;margin-bottom:6px;font-size:1.02rem;line-height:1.5;box-shadow:0 1px 3px rgba(20,20,20,.05);}
.kpi{background:#fff;border:1px solid #e7e6e2;border-top:5px solid var(--c);border-radius:14px;padding:16px 18px 10px;color:#1f2430;box-shadow:0 1px 3px rgba(20,20,20,.05);}
.kpi-city{font-weight:700;font-size:1.05rem;}
.kpi-big{font-size:2.3rem;font-weight:800;line-height:1.15;margin-top:4px;}
.kpi-unit{font-size:.88rem;font-weight:500;color:#6b6a66;}
.kpi-x{display:inline-block;margin:4px 0 10px;background:#f4f3ef;border-radius:999px;padding:2px 11px;font-size:.82rem;color:#3d3c39;}
.kpi-row{display:flex;justify-content:space-between;font-size:.9rem;padding:7px 0;border-top:1px solid #f0efeb;}
.kpi-row span:first-child{color:#6b6a66;}
.note{background:#f3f7fd;border-left:4px solid #2a78d6;border-radius:10px;padding:12px 16px;color:#1f2430;margin:6px 0 12px;line-height:1.55;}
.insight{background:#effaf5;border-left:4px solid #1baf7a;border-radius:10px;padding:12px 16px;color:#1f2430;margin:10px 0 4px;line-height:1.55;}
.insight ul{margin:6px 0 0 0;padding-left:20px;}
.insight li{margin-bottom:3px;}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 4px;}
.pill{background:#fff;border:1px solid #e7e6e2;border-left:5px solid var(--c);border-radius:10px;padding:7px 13px;color:#1f2430;font-size:.92rem;}
.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:8px 0 14px;}
.step{background:#fff;border:1px solid #e7e6e2;border-radius:14px;padding:14px 15px;color:#1f2430;font-size:.9rem;line-height:1.45;box-shadow:0 1px 3px rgba(20,20,20,.05);}
.step-n{display:inline-block;width:26px;height:26px;border-radius:50%;background:#2a78d6;color:#fff;text-align:center;line-height:26px;font-weight:700;font-size:.85rem;margin-bottom:6px;}
.step-t{font-weight:700;font-size:1rem;margin-bottom:4px;}
.limit{background:#fff7e8;border:1px solid #f1dcae;border-radius:12px;padding:12px 16px;color:#4a3b12;font-size:.92rem;line-height:1.5;margin-top:10px;}
@media(max-width:900px){.steps{grid-template-columns:1fr;}.hero-title{font-size:1.6rem;}}
</style>
"""


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data
def load_tables():
    con = sqlite3.connect(DB_PATH)
    clean = pd.read_sql("SELECT * FROM clean_hourly", con, parse_dates=["timestamp"])
    log = pd.read_sql("SELECT step, value FROM pipeline_log", con)
    n_rejected = pd.read_sql("SELECT COUNT(*) AS n FROM rejected_rows", con)["n"][0]
    con.close()
    return clean, log, int(n_rejected)


# ------------------------------------------------------------ small helpers
def color_of(city):
    return CITY_COLORS.get(city, "#6250d6")


def corr_label(r):
    """Plain-English description of a correlation number."""
    if pd.isna(r) or abs(r) < 0.1:
        return "No clear link"
    strength = "Strong" if abs(r) >= 0.5 else "Moderate" if abs(r) >= 0.3 else "Weak"
    return f"{strength} {'negative' if r < 0 else 'positive'} link"


def note(text):
    html(f'<div class="note"><b>What this shows.</b> {text}</div>')


def insight(lines):
    if lines:
        items = "".join(f"<li>{line}</li>" for line in lines)
        html(f'<div class="insight"><b>Key takeaways for your selection</b><ul>{items}</ul></div>')


def table_view(frame, label="View the data behind this chart"):
    with st.expander(label):
        st.dataframe(frame, use_container_width=True)


def new_fig(width=10, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, ylabel=None, xlabel=None, legend=True, extra=None):
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles += extra or []
        labels += [h.get_label() for h in (extra or [])]
        if len(handles) > 1:
            ax.legend(handles, labels, frameon=False, loc="lower left", ncol=len(handles),
                      bbox_to_anchor=(0, 1.0), fontsize=9, labelcolor=INK, handlelength=1.6)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def who_line(ax, label=True):
    ax.axhline(WHO_24H, color="#8d8b85", lw=1.2, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(0.995, WHO_24H, f"WHO daily guideline ({WHO_24H})", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=MUTED,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9), zorder=5)


def shade_smog_season(ax, start, end):
    for year in range(start.year - 1, end.year + 1):
        a, b = pd.Timestamp(year, 10, 1), pd.Timestamp(year + 1, 1, 31)
        if b < start or a > end:
            continue
        ax.axvspan(max(a, start), min(b, end), color="#efede7", lw=0, zorder=0)


def city_stats(sub):
    daily = sub.set_index("timestamp")["pm2_5"].resample("D").mean().dropna()
    monthly = sub.groupby("month")["pm2_5"].mean()
    mean = sub["pm2_5"].mean()
    return {
        "mean": mean,
        "times": mean / WHO_24H,
        "days_above": (daily > WHO_24H).mean() * 100,
        "bad": sub["aqi_category"].isin(BAD_AQI).mean() * 100,
        "worst": MONTHS[int(monthly.idxmax()) - 1],
    }


# ------------------------------------------------------------------- data
if not DB_PATH.exists():
    st.error("Database not found. Run `python pipeline.py` first, then start the app "
             "from the smog_project1 folder.")
    st.stop()

df, log, n_rejected = load_tables()
all_cities = sorted(df["city"].unique())
html(CSS)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")
cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities)
if not cities:
    st.warning("Select at least one city in the sidebar.")
    st.stop()

first_day, last_day = df["timestamp"].min().date(), df["timestamp"].max().date()
picked = st.sidebar.date_input("Date range", value=(first_day, last_day),
                               min_value=first_day, max_value=last_day)
start_day, end_day = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2
                      else (first_day, last_day))

mask = (df["city"].isin(cities) & (df["timestamp"].dt.date >= start_day)
        & (df["timestamp"].dt.date <= end_day))
data = df[mask]
if data.empty:
    st.warning("No data for this selection. Try a wider date range.")
    st.stop()
cities = [c for c in all_cities if c in cities]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(data):,} hourly readings selected")
st.sidebar.caption("**How to read this page:** start with the key numbers, then open each tab. "
                   "Every chart has a short explanation and key takeaways underneath.")

# ------------------------------------------------------------------- hero
names = ", ".join(all_cities[:-1]) + f" and {all_cities[-1]}"
html(
    '<div class="hero">'
    '<div class="hero-kicker">PAI PROJECT 1 · ETL + EDA</div>'
    f'<div class="hero-title">Smog (PM2.5) in {names}</div>'
    '<div class="hero-sub">Smog is a serious health issue in Pakistani cities, especially in winter, '
    'but ground-sensor coverage is limited. This app turns an ETL pipeline and an exploratory '
    'data analysis of hourly air-quality and weather data into a few simple pictures.</div>'
    '<div class="chips">'
    f'<span class="chip">📅 {first_day:%b %Y} – {last_day:%b %Y}</span>'
    f'<span class="chip">🏙️ {len(all_cities)} cities</span>'
    f'<span class="chip">🕒 {len(df):,} hourly readings</span>'
    '<span class="chip">🛰️ CAMS model estimates</span>'
    '</div></div>'
)

stats = {c: city_stats(data[data["city"] == c]) for c in cities}
worst_city = max(stats, key=lambda c: stats[c]["mean"])
glance = (f"<b>At a glance:</b> {worst_city} has the most polluted air in this selection, averaging "
          f"<b>{stats[worst_city]['mean']:.0f} µg/m³</b>, about "
          f"<b>{stats[worst_city]['times']:.0f} times</b> the WHO daily guideline of {WHO_24H} µg/m³.")
if len(cities) > 1:
    best_city = min(stats, key=lambda c: stats[c]["mean"])
    glance += (f" The cleanest is {best_city} at {stats[best_city]['mean']:.0f} µg/m³, "
               f"and even that is {stats[best_city]['times']:.1f} times the guideline.")
html(f'<div class="glance">{glance}</div>')

# ------------------------------------------------------------ key numbers
st.subheader("Key numbers")
st.caption("PM2.5 means tiny particles in the air (under 2.5 micrometres) that can reach deep into "
           "the lungs. It is measured in µg/m³: the lower, the better.")
for col, city in zip(st.columns(len(cities)), cities):
    s = stats[city]
    with col:
        html(
            f'<div class="kpi" style="--c:{color_of(city)}">'
            f'<div class="kpi-city">{city}</div>'
            f'<div class="kpi-big">{s["mean"]:.1f} <span class="kpi-unit">µg/m³ average</span></div>'
            f'<div class="kpi-x">{s["times"]:.1f}× the WHO guideline</div>'
            f'<div class="kpi-row"><span>Days above WHO limit</span><b>{s["days_above"]:.0f}%</b></div>'
            f'<div class="kpi-row"><span>Hours "Unhealthy" or worse</span><b>{s["bad"]:.1f}%</b></div>'
            f'<div class="kpi-row"><span>Worst month</span><b>{s["worst"]}</b></div>'
            '</div>'
        )

# ------------------------------------------------------------------- EDA
st.subheader("Explore the data")
tabs = st.tabs(["📈 Daily trend", "🗓️ Seasonality", "🕒 Hour of day", "🌦️ Weather", "🏙️ City comparison"])

# ---- Daily trend
with tabs[0]:
    note("The average PM2.5 of every day, one line per city. Peaks are smog episodes. The grey bands "
         "mark the smog season (October to January), and the dashed line is the WHO daily guideline.")
    smoothing = st.radio("Smoothing", ["Daily", "7-day average", "30-day average"], index=1, horizontal=True)
    window = {"Daily": 1, "7-day average": 7, "30-day average": 30}[smoothing]
    daily = (data.set_index("timestamp").groupby("city")["pm2_5"].resample("D").mean().unstack(0))
    fig, ax = new_fig()
    shade_smog_season(ax, daily.index.min(), daily.index.max())
    for city in cities:
        s = daily[city].rolling(window, center=True, min_periods=1).mean()
        ax.plot(s.index, s.values, color=color_of(city), lw=1.8, label=city)
    who_line(ax)
    ax.set_xlim(daily.index.min(), daily.index.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    finish(fig, ax, ylabel="PM2.5 (µg/m³)",
           extra=[Patch(facecolor="#efede7", label="Smog season (Oct–Jan)")])
    lines = []
    for city in cities:
        s = daily[city].dropna()
        lines.append(f"<b>{city}:</b> worst day was {s.idxmax():%d %b %Y} at {s.max():.0f} µg/m³, "
                     f"{s.max() / WHO_24H:.0f} times the WHO guideline.")
    insight(lines)
    table_view(daily.round(1))

# ---- Seasonality
with tabs[1]:
    note("The average PM2.5 for each calendar month, using all years together. Tall bars are the "
         "months with the most pollution. Winter months stand out because cool, still air traps "
         "smoke and dust near the ground.")
    monthly = data.groupby(["month", "city"])["pm2_5"].mean().unstack().reindex(range(1, 13))
    fig, ax = new_fig()
    width = 0.8 / len(cities)
    x = np.arange(12)
    for i, city in enumerate(cities):
        ax.bar(x + (i - (len(cities) - 1) / 2) * width, monthly[city].values, width=width,
               color=color_of(city), edgecolor="white", linewidth=1.2, label=city)
    who_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    finish(fig, ax, ylabel="Average PM2.5 (µg/m³)")
    season = (data.groupby(["city", "is_smog_season"])["pm2_5"].mean().unstack()
              .rename(columns={0: "Rest of year", 1: "Smog season (Oct-Jan)"}).round(1))
    lines = []
    for city in cities:
        worst, best = monthly[city].idxmax(), monthly[city].idxmin()
        line = (f"<b>{city}:</b> worst month is {MONTHS[worst - 1]} ({monthly[city].max():.0f} µg/m³), "
                f"cleanest is {MONTHS[best - 1]} ({monthly[city].min():.0f} µg/m³).")
        if {"Rest of year", "Smog season (Oct-Jan)"} <= set(season.columns):
            smog, rest = season.loc[city, "Smog season (Oct-Jan)"], season.loc[city, "Rest of year"]
            if pd.notna(smog) and pd.notna(rest) and rest > 0:
                line += f" The smog season is {smog / rest:.1f}× higher than the rest of the year."
        lines.append(line)
    insight(lines)
    table_view(season, "View smog season vs rest of year as a table")

# ---- Hour of day
with tabs[2]:
    note("The average PM2.5 for each hour of the day (0 is midnight, 12 is noon). It shows when the air "
         "is cleanest and dirtiest. The dot marks each city's worst hour. Switch the season to compare.")
    season_pick = st.radio("Season", ["Smog season (Oct-Jan)", "Rest of year"], horizontal=True)
    flag = 1 if season_pick.startswith("Smog") else 0
    hourly = (data[data["is_smog_season"] == flag].groupby(["hour", "city"])["pm2_5"].mean().unstack())
    fig, ax = new_fig()
    for city in cities:
        if city in hourly.columns and hourly[city].notna().any():
            ax.plot(hourly.index, hourly[city], color=color_of(city), lw=2.2, label=city)
            peak = hourly[city].idxmax()
            ax.plot([peak], [hourly[city].max()], "o", ms=7, color=color_of(city),
                    markeredgecolor="white", markeredgewidth=1.8)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(0, 23)
    finish(fig, ax, ylabel="Average PM2.5 (µg/m³)")
    lines = []
    for city in cities:
        if city in hourly.columns and hourly[city].notna().any():
            lines.append(f"<b>{city}:</b> highest around {int(hourly[city].idxmax()):02d}:00 "
                         f"({hourly[city].max():.0f} µg/m³), lowest around "
                         f"{int(hourly[city].idxmin()):02d}:00 ({hourly[city].min():.0f} µg/m³).")
    lines.append("Peaks at certain hours are often linked to traffic, burning of waste and fuel, and "
                  "cool, still air near the ground that traps pollution.")
    insight(lines)
    table_view(hourly.round(1))

# ---- Weather
with tabs[3]:
    note("Pick a weather variable. The chart groups hours into ten bands of that weather (left to right) "
         "and shows the average PM2.5 in each band. A line that slopes down means this kind of weather "
         "goes with cleaner air. The boxes under the chart show the link as a correlation (r): near 0 means no link, "
         "near -1 or +1 means a strong one. A link does not prove one thing causes the other.")
    weather = {"Wind speed (km/h)": "wind_speed_10m", "Temperature (°C)": "temperature_2m",
               "Humidity (%)": "relative_humidity_2m", "Surface pressure (hPa)": "surface_pressure"}
    label = st.selectbox("Weather variable", list(weather))
    var = weather[label]
    lo, hi = data[var].quantile([0.01, 0.99])
    edges = np.linspace(lo, hi, 11)
    fig, ax = new_fig()
    binned = {}
    for city in cities:
        sub = data[data["city"] == city][[var, "pm2_5"]].dropna()
        g = sub.groupby(pd.cut(sub[var], edges, include_lowest=True), observed=True)["pm2_5"].agg(["mean", "count"])
        g = g[g["count"] >= 30]
        xs = [iv.mid for iv in g.index]
        binned[city] = pd.Series(g["mean"].values, index=np.round(xs, 1))
        ax.plot(xs, g["mean"].values, color=color_of(city), lw=2.2, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=1.4, label=city)
    finish(fig, ax, ylabel="Average PM2.5 (µg/m³)", xlabel=label)
    d = (data.set_index("timestamp").groupby("city")[["pm2_5", var]].resample("D").mean().dropna().reset_index())
    pills, lines = "", []
    name = label.split(" (")[0].lower()
    for city in cities:
        dc = d[d["city"] == city]
        r = dc[var].corr(dc["pm2_5"]) if len(dc) > 2 else np.nan
        pills += (f'<div class="pill" style="--c:{color_of(city)}"><b>{city}</b> · '
                  f'{corr_label(r)} (r = {r:.2f})</div>' if pd.notna(r) else
                  f'<div class="pill" style="--c:{color_of(city)}"><b>{city}</b> · not enough data</div>')
        if pd.notna(r) and abs(r) >= 0.1:
            lines.append(f"<b>{city}:</b> days with higher {name} tend to have "
                         f"{'cleaner' if r < 0 else 'dirtier'} air ({corr_label(r).lower()}).")
        else:
            lines.append(f"<b>{city}:</b> {name} alone does not explain PM2.5 here.")
    html(f'<div class="pills">{pills}</div>')
    insight(lines)
    table_view(pd.DataFrame(binned).rename_axis(label).round(1))

# ---- City comparison
with tabs[4]:
    note("Left: which city has the highest average PM2.5, compared with the WHO guideline. Right: what "
         "share of all hours the air was Good, Moderate, Unhealthy and so on (US EPA categories).")
    summary = data.groupby("city")["pm2_5"].agg(["mean", "median", "max", "count"]).round(1)
    summary.columns = ["Mean", "Median", "Max", "Hours"]
    share = (data.groupby("city")["aqi_category"].value_counts(normalize=True)
             .unstack().reindex(columns=AQI_ORDER).fillna(0).mul(100).round(1))
    left, right = st.columns(2)
    with left:
        order = summary["Mean"].sort_values().index.tolist()
        fig, ax = new_fig(5.6, 3.6)
        ax.grid(axis="x", color=GRID, lw=0.9)
        ax.grid(axis="y", visible=False)
        ax.barh(order, summary.loc[order, "Mean"], color=[color_of(c) for c in order],
                edgecolor="white", linewidth=1.2, height=0.62)
        for i, c in enumerate(order):
            ax.text(summary.loc[c, "Mean"] + summary["Mean"].max() * 0.015, i,
                    f"{summary.loc[c, 'Mean']:.1f}", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.axvline(WHO_24H, color="#8d8b85", lw=1.2, ls=(0, (4, 3)))
        ax.text(WHO_24H, len(order) - 0.45, f" WHO ({WHO_24H})", fontsize=8, color=MUTED, va="bottom")
        ax.set_xlim(0, summary["Mean"].max() * 1.15)
        ax.tick_params(axis="y", labelsize=10.5, colors=INK)
        finish(fig, ax, xlabel="Average PM2.5 (µg/m³)", legend=False)
    with right:
        fig, ax = new_fig(5.6, 3.6)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.9)
        order_s = [c for c in cities if c in share.index][::-1]
        left_edge = np.zeros(len(order_s))
        for cat, col in zip(AQI_ORDER, AQI_COLORS):
            vals = share.loc[order_s, cat].values
            ax.barh(order_s, vals, left=left_edge, color=col, edgecolor="white", linewidth=1.5,
                    height=0.62, label=cat)
            for i, v in enumerate(vals):
                if v >= 7:
                    ax.text(left_edge[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                            color=INK if cat in AQI_DARK_TEXT else "white", fontweight="bold")
            left_edge += vals
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", labelsize=10.5, colors=INK)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [l.replace("Unhealthy for Sensitive Groups", "Unhealthy (sensitive)") for l in labels],
                  frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8,
                  labelcolor=INK, handlelength=1.2)
        finish(fig, ax, xlabel="Share of hours (%)", legend=False)
    lines = []
    if len(summary) > 1:
        top, low = summary["Mean"].idxmax(), summary["Mean"].idxmin()
        lines.append(f"<b>{top}</b> has the highest average PM2.5 ({summary.loc[top, 'Mean']:.1f} µg/m³), "
                     f"about {summary.loc[top, 'Mean'] / summary.loc[low, 'Mean']:.1f}× {low} "
                     f"({summary.loc[low, 'Mean']:.1f} µg/m³), the lowest.")
    for city in cities:
        lines.append(f"<b>{city}:</b> the air was in the 'Good' category for {share.loc[city, 'Good']:.1f}% "
                     f"of hours and 'Unhealthy' or worse for {share.loc[city, BAD_AQI].sum():.1f}%.")
    insight(lines)
    table_view(summary, "View the summary table")

# --------------------------------------------------------------- pipeline
st.subheader("How the data was prepared (ETL pipeline)")
st.caption("The numbers above come from five automatic steps, run by pipeline.py.")
steps = [
    ("Extract", "Hourly PM2.5, PM10 (Air Quality API) and weather (Historical Weather API), saved as raw JSON."),
    ("Load raw", "Stored unchanged in a SQLite database, so the original data is never lost."),
    ("Validate", "Every record checked with Pydantic: value ranges, PM2.5 ≤ PM10, dew point ≤ temperature."),
    ("Transform", "pandas removes duplicates, fills gaps up to 3 hours, and adds time and AQI columns."),
    ("Load clean", "Clean table with a primary key (city, timestamp), ready for analysis."),
]
html('<div class="steps">' + "".join(
    f'<div class="step"><div class="step-n">{i}</div><div class="step-t">{t}</div>{d}</div>'
    for i, (t, d) in enumerate(steps, 1)) + '</div>')
m1, m2, m3 = st.columns(3)
m1.metric("Clean hourly rows", f"{len(df):,}")
m2.metric("Rows rejected by validation", f"{n_rejected:,}")
m3.metric("Cities", len(all_cities))
with st.expander("Pipeline log"):
    st.dataframe(log, use_container_width=True)

html('<div class="limit"><b>Limitation:</b> PM2.5 values are CAMS model estimates from Open-Meteo, '
     'not ground-sensor measurements, so they can under- or over-estimate local extremes.</div>')
