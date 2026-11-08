"""C1 delivery promise at checkout: the Streamlit app.

    streamlit run deployment/app.py        (run from the Phase 3 folder)

Tab 1 runs the 25,352 test orders (bought May to August 2018) on a simulated clock: each order is scored by the
saved model when it is bought and judged only when it is delivered. At the end, the totals equal Table D2 of the
notebook. Names, tables and numbers follow the submitted deck. Scoring lives in predict.py, the health checks in
monitor.py. Add ?date=2018-07-02 to the address to open at a date.
"""
import json
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit import config as st_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monitor  # noqa: E402
from predict import load_model, score  # noqa: E402

DATA = HERE / "app_data" / "test_orders.parquet"
OUTLINES = HERE / "app_data" / "brazil_states.geojson"   # built by build_map.py (click_that_hood, MIT)
MIN_STATE = 50            # fewer delivered orders than this: the state is greyed out on the map
GAP_MAX = 12              # the colour scale of the gap maps runs from -12 to +12 (percentage points, days)
VERSION = "v6 (8 Nov 2026)"
OURS, OLIST, GREY, INK, NAVY = "#156082", "#B5651D", "#5A6170", "#1F2430", "#0E2841"
REGIONS = ["North", "Northeast", "Centre-West", "Southeast", "South"]
RANGE_95 = "95% range 3.27 to 7.84"    # week-block bootstrap, Table D2 of the notebook
# Lowest and highest values in the 70,851 training orders, for the new-order checks
TRAIN_RANGE = {"distance_km": (0.0, 3387.26, "distance", "km"), "item_count": (1, 21, "number of items", ""),
               "order_item_value": (0.85, 13440.0, "value of goods", "R$"), "order_freight_value": (0.0, 1002.29, "freight", "R$"),
               "total_weight_kg": (0.002, 184.4, "total weight", "kg"), "total_volume_l": (0.288, 1476.0, "total volume", "litres")}
HOUR_BANDS = [(0, "night"), (6, "morning"), (12, "afternoon"), (18, "evening")]   # as built in the notebook
THEME = {"theme.base": "light", "theme.primaryColor": OURS, "theme.backgroundColor": "#F5F6F8",
         "theme.secondaryBackgroundColor": "#E9ECF1", "theme.textColor": "#1F2430"}
MAP_METRICS = {"Late rate gap": ("late_gap", "percentage points", "ML promise minus Olist's quotes. Blue: ML late less often."),
               "Promise gap": ("promise_gap", "days", "ML promise minus Olist's quotes. Blue: ML shorter."),
               "Late rate": ("late_ml", "%", "Late rate of the ML promise. Darker: late more often."),
               "Orders": ("n", "orders", "Delivered orders so far. Darker: more orders.")}
TOP_TITLES = {"Late rate gap": "Top states: fewer late than Olist (points)", "Promise gap": "Top states: longer than Olist (days)",
              "Late rate": "Top states: ML late rate (%)", "Orders": "Top states: delivered orders"}
STYLE = """<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: none}
.app-bar {background: #0E2841; color: #fff; border-radius: 10px; padding: 12px 20px; display: flex;
          justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap}
.app-bar .name {font-size: 1.3rem; font-weight: 700}
.app-bar .note {font-size: .9rem; opacity: .85}
.tiles {display: grid; gap: 10px; margin: 4px 0 10px}
@media (max-width: 1000px) {.tiles {grid-template-columns: repeat(2, minmax(0, 1fr)) !important}}
.tile {background: #fff; border: 1px solid #DDE1E8; border-radius: 10px; padding: 10px 14px}
.tile .label {font-size: .8rem; color: #5A6170}
.tile .value {font-size: 1.55rem; font-weight: 700; color: #0E2841; line-height: 1.25}
.tile .sub {font-size: .8rem; color: #5A6170}
.tile.good .value {color: #2E7D32}
.tile.warn .value {color: #B26A00}
.tile.bad .value {color: #C62828}
h3 {font-size: 1.2rem !important; padding-top: .4rem !important}
[data-stale="true"], .stale-element {opacity: 1 !important; transition: none !important}
[data-testid="stStatusWidget"] {visibility: hidden}
.chart-title {font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
/* bordered map and chart panels: white cards like the tiles, with tighter padding so legends fit on laptops */
[class*="st-key-chart_"], .st-key-map_panel {background: #fff; padding: 12px !important}
.st-key-map_panel [data-testid="stButtonGroup"] button {padding: 2px 9px; min-height: 2rem}
.st-key-map_panel [data-testid="stButtonGroup"] button p {font-size: .85rem}
/* the three action cards: every card as tall as the tallest */
.st-key-actions [data-testid="stHorizontalBlock"] {align-items: stretch}
.st-key-actions [data-testid="stColumn"] > div,
.st-key-actions [data-testid="stElementContainer"],
.st-key-actions [data-testid="stAlert"],
.st-key-actions [data-testid="stAlertContainer"] {height: 100%}
</style>"""

# The theme file can't sit at the team repository's root (it would restyle the team's other app), so set it here.
# The theme reaches the browser at the start of a run, hence the one rerun.
if any(st_config.get_option(key) != value for key, value in THEME.items()):
    for key, value in THEME.items():
        st_config.set_option(key, value)
    st.rerun()

st.set_page_config(page_title="Olist delivery promise", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_orders():
    """The test orders, scored now by the deployed model, with each order's outcome."""
    orders = pd.read_parquet(DATA)
    orders = orders.join(score(get_model(), orders))
    orders["late_ours"] = orders.delivery_days > orders.promise
    orders["late_olist"] = orders.delivery_days > orders.olist_days
    orders["error"] = orders.estimate - orders.delivery_days
    return orders


@st.cache_data
def daily_series(orders):
    """Day-by-day lines for the charts, over the whole run (the app shows the days already past)."""
    by_day = (orders.assign(day=orders.delivered.dt.normalize()).groupby("day")
              .agg(n=("order_id", "size"), ours=("late_ours", "sum"), olist=("late_olist", "sum")))
    by_day = by_day.reindex(pd.date_range(by_day.index.min(), by_day.index.max()), fill_value=0)
    week = by_day.rolling(7, min_periods=1).sum()
    week = week[week.n >= monitor.MIN_WEEK]
    late = pd.DataFrame({"ML promise": 100 * week.ours / week.n, "Olist's quotes": 100 * week.olist / week.n})
    # Medians over the last 7 days of deliveries: actual days, the ML estimate and both promises
    by_delivery = orders.sort_values("delivered")
    rows = {}
    for day in pd.date_range(orders.delivered.min().normalize(), orders.delivered.max().normalize()):
        upto = day + pd.Timedelta(days=1)
        window = by_delivery.iloc[by_delivery.delivered.searchsorted(upto - pd.Timedelta(days=7)):
                                  by_delivery.delivered.searchsorted(upto)]
        if len(window) >= monitor.MIN_WEEK:
            rows[day] = {"Actual": window.delivery_days.median(), "ML estimate": window.estimate.median(),
                         "ML promise": window.promise.median(), "Olist's quotes": window.olist_days.median()}
    return late, pd.DataFrame.from_dict(rows, orient="index")


@st.cache_data
def weekly_checks(orders):
    """The health checks at the end of each week of the run (the app shows the weeks already past)."""
    by_delivery = orders.sort_values("delivered").reset_index(drop=True)
    week_ends = pd.date_range(orders.purchased.min().normalize(), orders.delivered.max().normalize(), freq="W-SUN")
    return monitor.weekly_history(orders, by_delivery, get_model().log_margin, week_ends)


@st.cache_data
def equal_reliability_table(orders):
    return monitor.equal_reliability(orders)


@st.cache_data
def state_outlines():
    return json.loads(OUTLINES.read_text(encoding="utf-8"))["features"]


def pct(x):
    return f"{100 * x:.2f}%"


def lines(wide, upto, y_title, y_max, x_range, series=("ML promise", "Olist's quotes"), colours=(OURS, OLIST), dashed=()):
    """Lines up to the day before the clock, on fixed axes so the chart fills in."""
    data = (wide[wide.index < upto].rename_axis("Day").reset_index()
            .melt("Day", var_name="Series", value_name="Value"))
    legend = alt.Legend(orient="top", title=None, symbolStrokeWidth=2, labelFontSize=10, symbolSize=60, columnPadding=3, labelOffset=2)
    chart = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("Day:T", title=None, scale=alt.Scale(domain=x_range), axis=MONTHS),
        y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(domain=[0, y_max]), axis=y_axis()),
        color=alt.Color("Series:N", scale=alt.Scale(domain=list(series), range=list(colours)), legend=legend),
        strokeDash=alt.StrokeDash("Series:N", legend=legend, scale=alt.Scale(
            domain=list(series), range=[[6, 4] if s in dashed else [1, 0] for s in series])),
        tooltip=[alt.Tooltip("Day:T", format="%a %d %b %Y"), "Series:N", alt.Tooltip("Value:Q", format=".2f")])
    return no_new_orders(chart, x_range).properties(height=CHART_H, background="#FFFFFF")


def reference_line(value, label, domain, title):
    """A dashed horizontal line with a label, e.g. the margin in use; it also carries the chart's y-axis."""
    base = alt.Chart(pd.DataFrame({"y": [value]}))
    y = alt.Y("y:Q", title=title, scale=alt.Scale(domain=domain), axis=y_axis())
    return (base.mark_rule(strokeDash=[4, 4], color=GREY).encode(y=y)
            + base.mark_text(align="left", dx=4, dy=-6, color=GREY).encode(x=alt.value(0), y=y, text=alt.value(label)))


def empty_note(chart, data, domain):
    """Before the first weekly check, write a note in the middle of the empty plot."""
    if len(data):
        return chart
    note = alt.Chart(pd.DataFrame({"y": [sum(domain) / 2]})).mark_text(color=GREY, fontSize=12).encode(
        x=alt.value(150), y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=y_axis()),
        text=alt.value("First check about 7 June"))
    return chart + note


def chart_title(text):
    """One line per chart title, so the charts in a row always start level (a long title ends in '…', hover for all)."""
    st.markdown(f'<div class="chart-title" title="{text}">{text}</div>', unsafe_allow_html=True)


def no_new_orders(chart, x_range, at="top"):
    """Grey out the days after the last test purchase: from here on, only the slower parcels are still arriving.
    The note sits in the band's right-hand corner, clear of the lines (at the bottom where a line runs along the top)."""
    band = pd.DataFrame({"start": [LAST_PURCHASE], "end": [x_range[1]],
                         "note": [f"No new orders\nafter {LAST_PURCHASE:%d %b}"]})
    scale = alt.Scale(domain=x_range)
    shade = alt.Chart(band).mark_rect(color="#8A93A3", opacity=0.14).encode(
        x=alt.X("start:T", scale=scale, title=None), x2="end:T")
    label = alt.Chart(band).mark_text(align="right", baseline="top" if at == "top" else "bottom",
                                      dx=-4, dy=0 if at == "top" else -4,
                                      lineBreak="\n", fontSize=10, color=GREY).encode(
        x=alt.X("end:T", scale=scale, title=None),
        y=alt.value(4) if at == "top" else alt.datum(0), text="note:N")
    return alt.layer(shade, label, chart)


MONTHS = alt.Axis(format="%d %b", grid=False, tickCount={"interval": "month", "step": 1})


def week_x(x_range):
    return alt.X("Week ending:T", title=None, scale=alt.Scale(domain=x_range), axis=MONTHS)


CHART_H = 237                    # plot height of the four monitoring charts, so the grid ends level with the map panel
WEEK_TIP = alt.Tooltip("Week ending:T", format="%a %d %b %Y")
READINGS = ["Margin about right", "Points shorter", "Points longer"]
MARGIN_AXIS = [0.8, 2.6]


def y_axis():
    """The same y-axis width on every monitoring chart, so their plots line up."""
    return alt.Axis(minExtent=48, maxExtent=48)


def margin_chart(weeks, current, x_range):
    """The suggested margin at each week's check, coloured by its reading, against the margin in use."""
    data = weeks.dropna(subset=["Suggested margin"])
    x = week_x(x_range)
    line = alt.Chart(data).mark_line(strokeWidth=2, color=GREY).encode(
        x=x, y=alt.Y("Suggested margin:Q", title="Margin (×)", scale=alt.Scale(domain=MARGIN_AXIS), axis=y_axis()))
    points = alt.Chart(data).mark_point(filled=True, size=60).encode(
        x=x, y="Suggested margin:Q",
        color=alt.Color("Advice:N", legend=alt.Legend(orient="top", title=None, labelFontSize=10, symbolSize=50, columnPadding=4, labelOffset=3),
                        scale=alt.Scale(domain=READINGS, range=["#2E7D32", "#C77C02", "#C62828"])),
        shape=alt.Shape("Advice:N", legend=None, scale=alt.Scale(domain=READINGS, range=["circle", "triangle-down", "triangle-up"])),
        tooltip=[WEEK_TIP, alt.Tooltip("Suggested margin:Q", format=".2f"), "Advice:N"])
    chart = reference_line(current, f"In use × {current:.3f}", MARGIN_AXIS, "Margin (×)") + line + points
    return no_new_orders(empty_note(chart, data, MARGIN_AXIS), x_range).properties(height=CHART_H, background="#FFFFFF")


def error_chart(weeks, x_range):
    """The estimate's error at each week's check, against the retrain line."""
    data = weeks.dropna(subset=["MAE"]).assign(Series="Estimate error (MAE)")
    line = alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(
        x=week_x(x_range), y=alt.Y("MAE:Q", title="Days", scale=alt.Scale(domain=[0, 7]), axis=y_axis()),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Estimate error (MAE)"], range=[OURS]), legend=alt.Legend(
            orient="top", title=None, labelFontSize=10, symbolType="stroke", symbolStrokeWidth=2, symbolSize=80)),
        tooltip=[WEEK_TIP, alt.Tooltip("MAE:Q", format=".2f")])
    chart = reference_line(monitor.RETRAIN_MAE, f"Retrain above {monitor.RETRAIN_MAE:.2f}", [0, 7], "Days") + line
    return no_new_orders(empty_note(chart, data, [0, 7]), x_range, at="bottom").properties(height=CHART_H, background="#FFFFFF")


def state_stats(delivered):
    stats = (delivered.groupby("customer_state")
             .agg(n=("order_id", "size"), late_ml=("late_ours", "mean"), late_olist=("late_olist", "mean"),
                  promise_ml=("promise", "mean"), promise_olist=("olist_days", "mean"))
             .rename_axis("state").reset_index())
    stats[["late_ml", "late_olist"]] *= 100
    stats["late_gap"] = stats.late_ml - stats.late_olist
    stats["promise_gap"] = stats.promise_ml - stats.promise_olist
    return stats


def shade(value, low, high, stops):
    """A colour between the stops, for a value between low and high."""
    t = min(max((value - low) / (high - low), 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return [round(a + (b - a) * (t - i)) for a, b in zip(stops[i], stops[i + 1])] + [205]


def state_colour(row, field):
    if row["n"] < MIN_STATE:
        return [208, 211, 218, 205]
    if field in ("late_gap", "promise_gap"):
        return shade(row[field], -GAP_MAX, GAP_MAX, [(21, 96, 130), (236, 236, 236), (181, 101, 29)])
    light, dark = (232, 240, 245), (21, 96, 130)
    return shade(row[field], 0, 8, [light, dark]) if field == "late_ml" else shade(np.log10(max(row[field], 1)), 1, 4, [light, dark])


def state_map(stats, field):
    """Brazil on a real basemap, each state coloured by the chosen measure."""
    rows = stats.set_index("state").to_dict("index")
    features = []
    for feature in state_outlines():
        row = rows.get(feature["properties"]["state"], {"n": 0, "late_ml": 0, "late_olist": 0, "promise_ml": 0, "promise_olist": 0, field: 0})
        features.append({**feature, "properties": {
            "name": feature["properties"]["name"], "color": state_colour(row, field), "orders": f"{row['n']:,}",
            "late_ml": f"{row['late_ml']:.2f}", "late_olist": f"{row['late_olist']:.2f}",
            "promise_ml": f"{row['promise_ml']:.2f}", "promise_olist": f"{row['promise_olist']:.2f}"}})
    layer = pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": features}, pickable=True, stroked=True, filled=True,
                      get_fill_color="properties.color", get_line_color=[255, 255, 255], line_width_min_pixels=1)
    tooltip = {"html": "<b>{name}</b><br/>Orders delivered {orders}<br/>Late: ML {late_ml}%, Olist {late_olist}%<br/>"
                       "Promise: ML {promise_ml}, Olist {promise_olist} days",
               "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "13px"}}
    return pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=-15.8, longitude=-52.5, zoom=2.95),
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT, tooltip=tooltip)


def map_legend(field):
    """The colour scale of the chosen map measure, with the grey used for too few orders."""
    if field in ("late_gap", "promise_gap"):
        unit = "percentage points" if field == "late_gap" else "days"
        ramp, left, middle, right = ("#156082, #ECECEC, #B5651D", f"−{GAP_MAX} or less: ML better", "0", f"+{GAP_MAX} or more: Olist better")
    elif field == "late_ml":
        unit, ramp, left, middle, right = "% late", "#E8F0F5, #156082", "0%", "4%", "8% or more"
    else:
        unit, ramp, left, middle, right = "delivered orders", "#E8F0F5, #156082", "10", "100", "10,000 or more"
    st.markdown(
        f'<div style="font-size:.8rem;color:#5A6170">'
        f'<div style="height:12px;border-radius:6px;background:linear-gradient(to right, {ramp})"></div>'
        f'<div style="display:flex;justify-content:space-between"><span>{left}</span><span>{middle}</span><span>{right}</span></div>'
        f'<div style="margin-top:4px"><span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:#D0D3DA;vertical-align:middle"></span> Fewer than {MIN_STATE} delivered orders · scale in {unit}</div></div>',
        unsafe_allow_html=True)


def top_states(stats, field):
    """The eight states highest on the chosen measure, among those with enough orders."""
    names = {f["properties"]["state"]: f["properties"]["name"] for f in state_outlines()}
    data = stats[stats.n >= MIN_STATE].assign(State=lambda d: d.state.map(names),
                                              Value=lambda d: -d[field] if field == "late_gap" else d[field]).nlargest(8, "Value")
    number = ",.0f" if field == "n" else ".2f"
    bars = alt.Chart(data).mark_bar(color=OURS, size=16).encode(
        y=alt.Y("State:N", title=None, sort="-x"), x=alt.X("Value:Q", title=None, axis=None),
        tooltip=["State:N", alt.Tooltip("Value:Q", format=number)])
    return (bars + bars.mark_text(align="left", dx=4, color=GREY).encode(text=alt.Text("Value:Q", format=number))).properties(height=260)


def reliability_chart(table):
    """Average promise at each late rate, ML promise against Olist's quotes scaled to that rate."""
    data = (table.assign(Rate=[f"At {100 * r:.2f}% late" for r in table.rate]).drop(columns="rate")
            .melt("Rate", var_name="Series", value_name="Days"))
    series = ["ML promise", "Olist, scaled"]
    base = alt.Chart(data).encode(
        y=alt.Y("Rate:N", title=None, sort=None), yOffset=alt.YOffset("Series:N", sort=series),
        x=alt.X("Days:Q", title="Average promise (days)", scale=alt.Scale(domain=[0, 34])))
    bars = base.mark_bar(size=20).encode(
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=[OURS, "#8FB4C9"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Rate:N", "Series:N", alt.Tooltip("Days:Q", format=".2f")])
    labels = base.mark_text(align="left", dx=4, fontWeight="bold").encode(text=alt.Text("Days:Q", format=".2f"))
    # A fixed width (a chart in a hidden tab can't measure its column), small enough for a laptop with the sidebar open
    return (bars + labels).properties(width=560, height=300)


def weekly_table(weeks):
    """The weekly checks as the deck shows them, one row per completed week."""
    checked = weeks[weeks["Advice"].notna()]

    def same_direction(row):
        n = row["Weeks in a row"]
        return f"{n} week{'s' if n > 1 else ''}, {row['Advice'].split()[-1]}" if n else "None"

    return pd.DataFrame({
        "Week ending": checked["Week ending"].dt.strftime("%d %b").str.lstrip("0"),
        "Suggested margin": checked["Suggested margin"].map(lambda m: f"× {m:.2f}"),
        "Same direction for": checked.apply(same_direction, axis=1) if len(checked) else [],
        "Decision": np.where(checked["Weeks in a row"] >= monitor.REVIEW_WEEKS, "Review margin", f"Keep × {CURRENT:.3f}"),
    })


def build_steps(parts):
    """How a promise is built, step by step, from score()'s parts."""
    tiles([("Log OLS estimate", f"{parts.ols:.1f} days", "linear regression on log days", ""),
           ("Forest estimate", f"{parts.forest:.1f} days", "random forest", ""),
           ("Average, on the log scale", f"{parts.estimate:.1f} days", "the typical delivery time", ""),
           (f"Raised by × {CURRENT:.3f}, rounded up", f"{parts.promise:.0f} days", "the promise", "")])
    st.caption(f"(1 + {parts.estimate:.1f}) × {CURRENT:.3f} − 1 = {(1 + parts.estimate) * CURRENT - 1:.1f}, rounded up to "
               f"{parts.promise:.0f}. A promise is late when the parcel arrives after the promised day.")


def order_label(row):
    return f"{row.order_id[:10]}…  ·  bought {row.purchased:%d %b %Y}  ·  {row.seller_state} → {row.customer_state}"


def pick_random():
    pool = filtered_orders(ss.get("past_state", "All"), ss.get("past_month", "All"))
    ss.past_order = pool.order_id.sample(1).iloc[0]


def filtered_orders(state, month):
    pool = orders
    if state != "All":
        pool = pool[pool.customer_state == state]
    if month != "All":
        pool = pool[pool.purchased.dt.strftime("%Y-%m") == month]
    return pool


def tiles(items):
    """The headline strip, one column per item; each item is (label, value, note, tone)."""
    cards = "".join(f'<div class="tile {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
                    f'<div class="sub">{note}</div></div>' for label, value, note, tone in items)
    columns = f"grid-template-columns: repeat({len(items)}, minmax(0, 1fr))"
    st.markdown(f'<div class="tiles" style="{columns}">{cards}</div>', unsafe_allow_html=True)


orders = get_orders()
model = get_model()

# Start-up check: the deployed libraries must reproduce every promise saved when the data was built
check_ok = np.array_equal(orders.promise.to_numpy(), orders.promise_at_build.to_numpy())

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. **Operations command centre:** how promises are performing, and when the margin needs a "
        "review. Press Play, or jump to a date. Start with late deliveries against Olist, read the "
        "suggested margin, review it after 4 weeks in a row, and retrain only if error passes 6.21 days.\n"
        "2. **Test results:** the tested numbers, at Olist's quotes as issued and at equal reliability.\n"
        "3. **Score a past order:** pick any test order and see how its promise was built.\n"
        "4. **Try a new order:** start from a past order, change its details, and get a new promise.")
    st.header("About the model")
    st.markdown(
        "Ensemble of log-linear regression (Log OLS) and a random forest, plus a × 1.893 safety "
        "margin, rounded up to whole days. The margin makes the ML promise late no more often than Olist's "
        "quotes on validation (9.51% of orders).\n\n"
        "Trained on 70,851 delivered orders bought January 2017 to April 2018. Tested once on 25,352 "
        "orders bought May to August 2018.")
    if check_ok:
        st.success(f"Model loaded; start-up check passed: all {len(orders):,} test promises reproduced.")
    st.caption(f"{VERSION.split()[0]} · State outlines: click_that_hood (Code for America, MIT licence)")

st.markdown('<div class="app-bar"><span class="name">Olist Delivery Promise | Model Operations</span>'
            '<span class="note">Test period: May to August 2018</span></div>', unsafe_allow_html=True)
if not check_ok:
    st.error("Start-up check failed: the deployed model does not reproduce the saved promises. "
             "Check the library versions in deployment/requirements.txt.")
    st.stop()

tab_live, tab_test, tab_past, tab_new = st.tabs(["Operations command centre", "Test results", "Score a past order", "Try a new order"])

# ----------------------------------------------------------------------------------------------
# Tab 1: Operations command centre
# ----------------------------------------------------------------------------------------------
START = orders.purchased.min().normalize()
CLOCK_START = START                        # the clock starts before the first order, so every count starts at 0
END = orders.delivered.max().ceil("h")
LAST_PURCHASE = orders.purchased.max()
CURRENT = float(np.exp(model.log_margin))
by_delivery = orders.sort_values("delivered").reset_index(drop=True)
late_daily, days_daily = daily_series(orders)
all_weeks = weekly_checks(orders)
DAYS_MAX = float(np.ceil(days_daily.max().max() / 10) * 10)
LATE_MAX = float(np.ceil(late_daily.max().max() / 5) * 5)

ss = st.session_state
if "clock" not in ss:
    opened_at = st.query_params.get("date")
    ss.clock = pd.Timestamp(opened_at) + pd.Timedelta(hours=12) if opened_at else CLOCK_START
ss.setdefault("playing", False)
ss.setdefault("tick", time.monotonic())


def toggle_play():
    ss.playing = not ss.playing and ss.clock < END
    ss.tick = time.monotonic()


def restart():
    ss.clock, ss.playing = CLOCK_START, False


def jump_to_end():
    ss.clock, ss.playing = END, False


def jump_to_date():
    if ss.jump is not None:
        ss.clock = min(max(pd.Timestamp(ss.jump) + pd.Timedelta(hours=12), START), END)


def model_status(check, streak):
    """The one-glance verdict: label, note and tone."""
    if not check["ready"]:
        return "Burn-in", "First margin check about 7 June", ""
    if check["retrain"]:
        return "Retrain", f"Error above {monitor.RETRAIN_MAE:.2f} days", "bad"
    if check["refresh"] and streak >= monitor.REVIEW_WEEKS:
        return "Review margin", f"{streak} weeks in a row", "warn"
    return "Healthy", "No action needed", "good"


def action_cards(week, check, streak):
    """What to do now, in the deck's order: late deliveries, the margin, the model."""
    a = st.container(key="actions").columns(3)
    with a[0]:
        if not week["ready"]:
            st.info(f"**Late deliveries:** too few in the last 7 days to judge ({week['n']:,}).")
        elif week["alarm"]:
            st.warning(f"**Late deliveries:** ML promise {pct(week['ours'])} against Olist's {pct(week['olist'])} this week. "
                       "Early warning: promises may be too short.", icon="⚠️")
        else:
            st.success(f"**Late deliveries:** ML promise {pct(week['ours'])} against Olist's {pct(week['olist'])} this week.", icon="✅")
    with a[1]:
        if not check["ready"]:
            st.info("**Margin:** the first check needs orders bought at least 32 days ago.")
        else:
            direction = "longer" if check["longer"] else "shorter"
            suggested = f"Suggested × {check['margin']:.2f}; in use × {check['current']:.3f}."
            if not check["refresh"]:
                st.success(f"**Margin about right.** {suggested}", icon="✅")
            elif streak >= monitor.REVIEW_WEEKS:
                st.warning(f"**Review margin:** {streak} weeks in a row point {direction}. {suggested}", icon="⚠️")
            else:
                st.info(f"**Keep the margin:** points {direction}, week {streak} of {monitor.REVIEW_WEEKS}. {suggested}")
    with a[2]:
        if not check["ready"]:
            st.info("**Model:** checked on the same orders as the margin.")
        elif check["retrain"]:
            st.error(f"**Retrain:** error {check['mae']:.2f} days, above {monitor.RETRAIN_MAE:.2f}.", icon="🚨")
        else:
            st.success(f"**No retraining needed:** error {check['mae']:.2f} days, limit {monitor.RETRAIN_MAE:.2f}.", icon="✅")


def command_centre():
    # Advance the simulated clock by the wall time since the last run, at the chosen speed
    now = time.monotonic()
    if ss.playing:
        step = pd.Timedelta(days=ss.get("speed", 1.0) * (now - ss.tick)).floor("s")   # whole seconds:
        ss.clock = min(ss.clock + step, END)    # the data's timestamps cannot be compared with nanoseconds
        ss.playing = ss.clock < END
    ss.tick = now
    T = ss.clock = min(max(ss.clock, START), END).floor("s")
    if ss.playing != ss.refreshing:     # Play pressed, or the end reached: restart with the right refresh
        st.rerun()

    controls = st.columns([1, 1, 1, 2.2, 2, 2.6], vertical_alignment="bottom")
    controls[0].button("Pause" if ss.playing else "Play", on_click=toggle_play, type="primary", width="stretch")
    controls[1].button("Reset", on_click=restart, width="stretch")
    controls[2].button("Jump to end", on_click=jump_to_end, width="stretch")
    controls[3].select_slider("Speed (days per second)", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0, key="speed")
    controls[4].date_input("Jump to a date", value=None, min_value=START.date(), max_value=END.date(), key="jump", on_change=jump_to_date)
    controls[5].progress(max(0.0, (T - CLOCK_START) / (END - CLOCK_START)), text=f"**{T:%a %d %b %Y, %H:%M}**")

    placed = orders.iloc[: orders.purchased.searchsorted(T, side="right")]
    delivered = by_delivery.iloc[: by_delivery.delivered.searchsorted(T, side="right")]
    if placed.empty:
        st.info("No orders yet. Press Play to start the test months, or jump to a date.")

    week = monitor.late_now(by_delivery, T)
    check = monitor.margin_check(orders, T, model.log_margin)
    weeks = all_weeks[all_weeks["Week ending"] + pd.Timedelta(days=1) <= T]     # completed weeks only
    same_way = check["ready"] and len(weeks) and weeks["Advice"].iloc[-1] == monitor.advice(check)
    streak = int(weeks["Weeks in a row"].iloc[-1]) if same_way else 1
    status, status_note, tone = model_status(check, streak)
    margin_note = f"Suggested × {check['margin']:.2f}" if check["ready"] else "No suggestion yet"

    none_yet = "No orders yet" if placed.empty else "No deliveries yet"      # a rate or an average of nothing is not 0
    tiles([("Model status", status, status_note, tone),
           ("Late deliveries", pct(delivered.late_ours.mean()) if len(delivered) else "–",
            f"Olist's quotes {pct(delivered.late_olist.mean())}" if len(delivered) else none_yet, ""),
           ("Average promise", f"{placed.promise.mean():.2f} days" if len(placed) else "–",
            f"Olist's quotes {placed.olist_days.mean():.2f}" if len(placed) else "No orders yet", ""),
           ("Estimate error", f"{delivered.error.abs().mean():.2f} days" if len(delivered) else "–",
            f"Retrain above {monitor.RETRAIN_MAE:.2f}", ""),
           ("Margin in use", f"× {CURRENT:.3f}", margin_note, ""),
           ("Delivered", f"{len(delivered):,}", f"{len(placed) - len(delivered):,} in flight", "")])
    action_cards(week, check, streak)

    x_range = [START, END.normalize()]
    left, right = st.columns([4, 8])
    with left, st.container(border=True, key="map_panel"):
        chosen = st.segmented_control("Map shows", list(MAP_METRICS), default="Late rate gap", key="map_metric",
                                      label_visibility="collapsed") or "Late rate gap"
        field, unit, reading = MAP_METRICS[chosen]
        stats = state_stats(by_delivery.iloc[: by_delivery.delivered.searchsorted(T.normalize())])   # one update a day
        st.pydeck_chart(state_map(stats, field), height=430, width="stretch")
        map_legend(field)
        st.caption(f"{reading} Hover for a state's numbers.")
    with right:
        top = st.columns(2)
        bottom = st.columns(2)
        with top[0], st.container(border=True, key="chart_1"):
            chart_title("Late rate, last 7 days of deliveries")
            st.altair_chart(lines(late_daily, T.normalize(), "% late", LATE_MAX, x_range), width="stretch")
        with top[1], st.container(border=True, key="chart_2"):
            chart_title("Suggested margin against the margin in use")
            st.altair_chart(margin_chart(weeks, CURRENT, x_range), width="stretch")
        with bottom[0], st.container(border=True, key="chart_3"):
            chart_title("Estimate error against the retrain line")
            st.altair_chart(error_chart(weeks, x_range), width="stretch")
        with bottom[1], st.container(border=True, key="chart_4"):
            chart_title("Promises, estimate and actual (median days)")
            st.altair_chart(lines(days_daily, T.normalize(), "Days", DAYS_MAX, x_range,
                                  series=("Olist's quotes", "ML promise", "ML estimate", "Actual"),
                                  colours=(OLIST, OURS, OURS, GREY), dashed=("ML estimate",)), width="stretch")

    table_side, states_side = st.columns([7, 5])
    with table_side:
        st.subheader("Weekly model health")
        table = weekly_table(weeks)
        if len(table):
            st.dataframe(table, hide_index=True, width="stretch", height=min(35 * (len(table) + 1) + 3, 388))
        else:
            st.caption("The first weekly check comes once orders bought 32 days ago have arrived.")
        st.caption("Each week the app suggests a margin; the margin stays fixed until 4 weeks in a row point the same way. "
                   "These checks were added after the test, so the tested results stand. Checks run about a month past "
                   "the last purchase, because each one reads orders bought 32 to 60 days earlier; from October they cover "
                   "fewer orders.")
    with states_side:
        st.subheader(TOP_TITLES[chosen])
        st.altair_chart(top_states(stats, field), width="stretch")

    feed_side, region_side = st.columns([7, 5])
    with feed_side:
        st.subheader("Live feed")
        new = placed.tail(8).assign(when=lambda d: d.purchased, event="New order")
        done = delivered.tail(8).assign(when=lambda d: d.delivered, event="Delivered")
        feed = pd.concat([new, done]).sort_values("when", ascending=False).head(8)
        is_done = (feed.event == "Delivered").to_numpy()

        def result(late):
            return np.where(is_done, np.where(late, "Late", "On time"), "In flight")

        st.dataframe(pd.DataFrame({
            "Time": feed.when.dt.strftime("%d %b %H:%M"),
            "Event": feed.event,
            "Route (seller → customer)": feed.seller_state + " → " + feed.customer_state,
            "ML promise (days)": feed.promise.astype(int),
            "Olist's quote (days)": feed.olist_days,
            "Actual (days)": np.where(is_done, feed.delivery_days.astype(int).astype(str), "–"),
            "ML": result(feed.late_ours),
            "Olist": result(feed.late_olist),
        }), hide_index=True, width="stretch")
    with region_side:
        st.subheader("By customer region")
        regions = (delivered.groupby("region")
                   .agg(**{"Orders": ("order_id", "size"),
                           "Late, Olist (%)": ("late_olist", "mean"), "Late, ML (%)": ("late_ours", "mean"),
                           "Promise, Olist": ("olist_days", "mean"), "Promise, ML": ("promise", "mean")})
                   .reindex(REGIONS).dropna().rename_axis("Region"))
        regions[["Late, Olist (%)", "Late, ML (%)"]] *= 100
        st.dataframe(regions.round(2).reset_index(), hide_index=True, width="stretch",
                     column_config={"Region": st.column_config.TextColumn(width="medium")})

    if T > LAST_PURCHASE:
        st.caption(f"No new orders after {LAST_PURCHASE:%d %B %Y}, the last test purchase; the remaining parcels, "
                   "mostly the slow ones, are still arriving.")
    st.caption(f"The {len(orders):,} test orders on a simulated clock. Each order is scored by the deployed model when it is bought "
               "and judged only when it is delivered. At the end, the totals equal the tested results. Fewer orders bought in "
               "late August appear in the data, so the last weeks thin out.")


with tab_live:
    ss.refreshing = ss.playing
    st.fragment(command_centre, run_every=0.7 if ss.playing else None)()

with tab_past:
    st.subheader("Score a past order")
    st.caption("Any of the 25,352 test orders, scored now by the deployed model, with Olist's quote and the actual delivery beside it.")
    f = st.columns([2, 2, 1.3], vertical_alignment="bottom")
    state = f[0].selectbox("Customer state", ["All"] + sorted(orders.customer_state.unique()), key="past_state")
    month = f[1].selectbox("Purchase month", ["All", "2018-05", "2018-06", "2018-07", "2018-08"], key="past_month")
    f[2].button("Pick a random order", on_click=pick_random, width="stretch")
    pool = filtered_orders(state, month)
    shown = pool.head(500)
    if ss.get("past_order") in set(pool.order_id) and ss.past_order not in set(shown.order_id):
        shown = pd.concat([pool[pool.order_id == ss.past_order], shown])
    if ss.get("past_order") not in set(shown.order_id):
        ss.past_order = shown.order_id.iloc[0]
    labels = {row.order_id: order_label(row) for row in shown.itertuples()}
    chosen_id = st.selectbox(f"Order ({len(pool):,} match; the first 500 are listed, or pick at random)", list(labels),
                             format_func=labels.get, key="past_order")
    order = orders[orders.order_id == chosen_id].iloc[0]
    with st.container(border=True):
        st.markdown("**The order, as known at checkout**")
        known = st.columns(3)
        known[0].markdown(f"**Bought** {order.purchased:%a %d %b %Y, %H:%M}  \n**Customer** {order.customer_state}, "
                          f"{str(order.customer_city).title()}  \n**Seller** {order.seller_state}, {str(order.seller_city).title()}")
        known[1].markdown(f"**Distance** {order.distance_km:,.0f} km, {'same state' if order.same_state else 'crosses a state border'}  \n"
                          f"**Items** {order.item_count:.0f} from {order.seller_count:.0f} seller(s), {order.main_category}  \n"
                          f"**Goods / freight** R\\$ {order.order_item_value:,.2f} / R\\$ {order.order_freight_value:,.2f}")
        known[2].markdown(f"**Weight / volume** {order.total_weight_kg:,.2f} kg / {order.total_volume_l:,.2f} litres  \n"
                          f"**Seller's record** {order.seller_prior_orders:,.0f} orders, median handling "
                          f"{order.seller_prior_handling_days:.1f} days  \n**Peak season** {'Yes' if order.peak_season else 'No'}")
    ml_due = order.purchased.normalize() + pd.Timedelta(days=int(order.promise))
    tiles([("ML promise", f"{order.promise:.0f} days", f"Deliver by {ml_due:%a %d %b %Y}", ""),
           ("Olist's quote", f"{order.olist_days:.0f} days", f"Deliver by {order.olist_due:%a %d %b %Y}", ""),
           ("What happened", f"{order.delivery_days:.0f} days", f"Delivered {order.delivered:%a %d %b}: ML "
            f"{'late' if order.late_ours else 'on time'}, Olist {'late' if order.late_olist else 'on time'}",
            "bad" if order.late_ours else "good")])
    st.markdown("**How this promise was built**")
    build_steps(order)

with tab_new:
    st.subheader("Try a new order")
    base = orders[orders.order_id == ss.get("past_order", orders.order_id.iloc[0])].iloc[0]
    st.caption(f"Starts from the order chosen in 'Score a past order' ({base.order_id[:10]}…). Change its details and the deployed "
               "model gives a new promise. The seller and its record stay as in that order.")
    if ss.get("try_base") != base.order_id:     # a new starting order: reset the inputs to it
        ss.try_base = base.order_id
        ss.update({"try_state": base.customer_state, "try_distance": float(round(base.distance_km, 1)),
                   "try_date": base.purchased.date(), "try_time": base.purchased.time().replace(second=0, microsecond=0),
                   "try_items": int(base.item_count), "try_value": float(base.order_item_value),
                   "try_freight": float(base.order_freight_value), "try_category": base.main_category,
                   "try_weight": None if pd.isna(base.total_weight_kg) else float(base.total_weight_kg),
                   "try_volume": None if pd.isna(base.total_volume_l) else float(base.total_volume_l)})
    with st.container(border=True):
        where = st.columns(4)
        new_state = where[0].selectbox("Customer state", sorted(orders.customer_state.unique()), key="try_state")
        distance = where[1].number_input("Distance to the seller (km)", min_value=0.0, max_value=6000.0, step=10.0, key="try_distance")
        bought_on = where[2].date_input("Purchase date", key="try_date")
        bought_at = where[3].time_input("Purchase time", key="try_time")
        what = st.columns(4)
        items = what[0].number_input("Number of items", min_value=1, max_value=100, step=1, key="try_items")
        value = what[1].number_input("Value of goods (R$)", min_value=0.0, step=10.0, key="try_value")
        freight = what[2].number_input("Freight (R$)", min_value=0.0, step=1.0, key="try_freight")
        category = what[3].selectbox("Main product category", sorted(orders.main_category.dropna().unique()), key="try_category")
        size = st.columns(4)
        weight = size[0].number_input("Total weight (kg), blank if unknown", min_value=0.0, step=0.1, value=None, key="try_weight")
        volume = size[1].number_input("Total volume (litres), blank if unknown", min_value=0.0, step=0.5, value=None, key="try_volume")
    new = {"customer_state": new_state, "distance_km": distance, "item_count": items, "order_item_value": value,
           "order_freight_value": freight, "main_category": category,
           "total_weight_kg": np.nan if weight is None else weight, "total_volume_l": np.nan if volume is None else volume}
    stops = []
    if value <= 0:
        stops.append("The value of goods must be above zero.")
    if items < base.seller_count:
        stops.append(f"This order has {base.seller_count:.0f} sellers, so it needs at least {base.seller_count:.0f} items.")
    for message in stops:
        st.error(message, icon="🚫")
    if not stops:
        for field, (low, high, name, unit) in TRAIN_RANGE.items():
            amount, unit = new[field], unit.replace("$", "\\$")
            if not pd.isna(amount) and not low <= amount <= high:
                st.warning(f"The {name}, {amount:,} {unit}, is outside the training orders ({low:,} to {high:,} {unit}). "
                           "The promise may be less reliable.", icon="⚠️")
        if weight is None or volume is None:
            st.info("A missing weight or volume is filled with the training median, as in the notebook.")
        if new_state != base.customer_state and distance == round(base.distance_km, 1):
            st.info("The customer state changed but the distance didn't. Set the distance to the new customer too.")
        purchase = pd.Timestamp.combine(bought_on, bought_at)
        if purchase.month in (11, 12):
            st.info("Peak-season order: the training months hold only one peak season, so this effect is less certain.")
        row = base.copy()
        row.update(pd.Series(new))
        row["same_state"] = new_state == base.seller_state
        row["purchase_weekday"] = purchase.day_name()
        row["purchase_hour_band"] = [band for start, band in HOUR_BANDS if purchase.hour >= start][-1]
        row["peak_season"] = purchase.month in (11, 12)
        frame = pd.DataFrame([row[model.features]])
        for column in ("same_state", "peak_season", "seller_has_history"):
            frame[column] = frame[column].astype(bool)
        numeric = [c for c in model.features if c not in ("same_state", "peak_season", "seller_has_history", "customer_state",
                                                       "seller_state", "main_category", "purchase_weekday", "purchase_hour_band")]
        frame[numeric] = frame[numeric].astype(float)
        parts = score(model, frame).iloc[0]
        due = purchase.normalize() + pd.Timedelta(days=int(parts.promise))
        tiles([("ML promise", f"{parts.promise:.0f} days", f"Deliver by {due:%a %d %b %Y}", ""),
               ("Typical delivery time", f"{parts.estimate:.1f} days", "the estimate before the margin", ""),
               ("Starting order's promise", f"{base.promise:.0f} days", f"{parts.promise - base.promise:+.0f} days with your changes", "")])
        st.markdown("**How this promise was built**")
        build_steps(parts)

with tab_test:
    st.subheader("Ensemble model for the delivery promise: results")
    st.caption(f"Test period: May to August 2018, {len(orders):,} orders, scored once. Ensemble of log-linear "
               f"regression (Log OLS) and a random forest, plus a × {CURRENT:.3f} safety margin.")
    tiles([("Late deliveries", pct(orders.late_ours.mean()), f"Olist's quotes {pct(orders.late_olist.mean())}", ""),
           ("Average promise", f"{orders.promise.mean():.2f} days", f"Olist's quotes {orders.olist_days.mean():.2f}", ""),
           ("Estimate error", f"{orders.error.abs().mean():.2f} days", "Log OLS 4.35, forest 5.27", "")])
    st.caption("Late less than half as often, on a promise about 2 days longer: the margin was set on slower months. The "
               "test months were fast, where Log OLS does best; the ensemble led on validation.")
    st.subheader("Shorter promises at equal reliability")
    reliability = equal_reliability_table(orders)
    chart_side, reading_side = st.columns([3, 2])
    with chart_side:
        st.altair_chart(reliability_chart(reliability), width="content")
    with reading_side:
        gap = reliability["Olist, scaled"].iloc[0] - reliability["ML promise"].iloc[0]
        tiles([(f"Shorter at our tested {pct(orders.late_ours.mean())} late", f"{gap:.2f} days", RANGE_95, "")])
        st.markdown("**Rank orders by risk, then set the cushion.** Our promise is shorter at every late rate shown.\n\n"
                    "**The margin matters more than the model.** Set on slower months, it ran cautious.\n\n"
                    "**Choose the late rate deliberately.** That is a business decision.")
    st.caption("Olist's quotes stretched by one factor until as late as ours. Only the first late rate is the margin as "
               "tested; the other two are set on these months, to show what's achievable.")
