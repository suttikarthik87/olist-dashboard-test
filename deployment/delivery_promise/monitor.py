"""Model-health checks for the C1 app: when to refresh the margin and when to retrain.

A Phase 3 monitoring design (1 Nov 2026), added after the test months; not part of the tested model.
The margin was set so that our promise is late no more often than Olist's quote on the same orders
(9.51% on the validation folds). The checks ask whether that still holds on recent orders:

- late_now:     deliveries of the last 7 days, ours against Olist's: an early warning.
- margin_check: orders bought 32 to 60 days earlier, by when about 99% have arrived (95% of orders
                arrive within 32 days). A week is flagged when our late rate differs from Olist's on
                these orders by more than two paired standard errors; also gives the margin that
                would match Olist. The margin in use stays fixed: the app suggests, and asks for a
                review only after REVIEW_WEEKS flagged weeks in a row in the same direction. Orders still open count against us: late for ours, and late for Olist
                only once already past Olist's date.
                Retrain when the estimate's MAE on these orders is above the worst validation fold.

Every function takes the scored test orders and a moment T, and uses only what is known at T.
"""
import numpy as np
import pandas as pd

LAG_DAYS, WINDOW_DAYS = 32, 28    # margin check: orders bought 60 to 32 days before T
MIN_COHORT = 1000                 # fewer orders than this: too few to check the margin
MIN_WEEK = 200                    # fewer deliveries than this in 7 days: too few to judge
SE_BAND = 2.0                     # paired standard errors before a gap counts
REVIEW_WEEKS = 4                  # the margin in use stays fixed; review it after this many weeks in a row
RETRAIN_MAE = 6.21                # worst fold MAE of the ensemble on the validation folds (log row 23)


def paired_gap(late_ours, late_olist):
    """Our late rate minus Olist's on the same orders, and its standard error."""
    diff = np.asarray(late_ours, float) - np.asarray(late_olist, float)
    return diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))


def margin_to_match(log_estimate, days, still_open, late_rate):
    """The smallest log margin whose late rate on these orders is no higher than late_rate.

    A promise ceil(expm1(L + m)) is on time for an order delivered in d >= 1 days exactly when
    m > log(d) - L, so the margin is just above the right order statistic of log(d) - L.
    """
    r = np.where(still_open, np.inf, np.where(days <= 0, -np.inf, np.log(np.maximum(days, 1)) - log_estimate))
    allowed_late = int(np.floor(late_rate * len(r)))
    edge = np.sort(r)[len(r) - allowed_late - 1]
    return np.nextafter(edge, np.inf)


def scale_factor(days, quoted, late_rate):
    """The smallest single factor on the quotes, rounded up to whole days, late no more often than late_rate."""
    low, high = 0.2, 5.0
    for _ in range(60):
        middle = (low + high) / 2
        if (days > np.ceil(quoted * middle)).mean() <= late_rate:
            high = middle
        else:
            low = middle
    return high


def equal_reliability(orders):
    """Average promise, ML against Olist's quotes scaled, at the tested late rate, Olist's own, and the 9.51% of the folds."""
    days, quoted = orders.delivery_days.to_numpy(), orders.olist_days.to_numpy(float)
    log_estimate = np.log1p(orders.estimate.to_numpy())
    still_open = np.zeros(len(orders), bool)
    tested = orders.late_ours.mean()
    rows = []
    for rate in (tested, orders.late_olist.mean(), 0.0951):
        ml = orders.promise.mean() if rate == tested else np.ceil(
            np.expm1(log_estimate + margin_to_match(log_estimate, days, still_open, rate))).mean()
        rows.append({"rate": rate, "ML promise": ml, "Olist, scaled": np.ceil(quoted * scale_factor(days, quoted, rate)).mean()})
    return pd.DataFrame(rows)


def late_now(by_delivery, T):
    """Late rates of the deliveries in the 7 days up to T. by_delivery: orders sorted by delivery."""
    week = by_delivery.iloc[by_delivery.delivered.searchsorted(T - pd.Timedelta(days=7), side="right"):
                            by_delivery.delivered.searchsorted(T, side="right")]
    if len(week) < MIN_WEEK:
        return {"n": len(week), "ready": False}
    gap, se = paired_gap(week.late_ours, week.late_olist)
    return {"n": len(week), "ready": True, "ours": week.late_ours.mean(), "olist": week.late_olist.mean(),
            "alarm": gap > SE_BAND * se}


def margin_check(orders, T, log_margin):
    """Margin and accuracy check on the orders bought 60 to 32 days before T."""
    end = T - pd.Timedelta(days=LAG_DAYS)
    start = end - pd.Timedelta(days=WINDOW_DAYS)
    c = orders[(orders.purchased > start) & (orders.purchased <= end)]
    if len(c) < MIN_COHORT:
        return {"n": len(c), "ready": False}
    still_open = (c.delivered > T).to_numpy()
    days = c.delivery_days.to_numpy()
    elapsed = (T.normalize() - c.purchased.dt.normalize()).dt.days.to_numpy()
    late_ours = still_open | (days > c.promise.to_numpy())
    late_olist = np.where(still_open, elapsed > c.olist_days.to_numpy(), days > c.olist_days.to_numpy())
    gap, se = paired_gap(late_ours, late_olist)
    log_estimate = np.log1p(c.estimate.to_numpy())
    m = margin_to_match(log_estimate, days, still_open, late_olist.mean())
    mae = c.error[~still_open].abs().mean()
    return {"n": len(c), "ready": True, "first": c.purchased.min(), "last": c.purchased.max(),
            "delivered": 1 - still_open.mean(),
            "ours": late_ours.mean(), "olist": late_olist.mean(),
            "refresh": abs(gap) > SE_BAND * se, "longer": gap > 0,
            "margin": float(np.exp(m)), "current": float(np.exp(log_margin)),
            "promise_now": c.promise.mean(), "promise_new": np.ceil(np.expm1(log_estimate + m)).mean(),
            "olist_quote": c.olist_days.mean(), "mae": mae, "retrain": mae > RETRAIN_MAE}


def advice(check):
    """The margin check's reading in a few words."""
    if not check["refresh"]:
        return "Margin about right"
    return "Points longer" if check["longer"] else "Points shorter"


def weekly_history(orders, by_delivery, log_margin, week_ends):
    """Each check as it stood at the end of each week (Sunday 23:59), for the weekly charts."""
    rows = []
    for end in week_ends:
        T = end + pd.Timedelta(hours=23, minutes=59)
        week, check = late_now(by_delivery, T), margin_check(orders, T, log_margin)
        row = {"Week ending": end}
        if week["ready"]:
            row.update({"Late, ours (%)": 100 * week["ours"], "Late, Olist (%)": 100 * week["olist"]})
        if check["ready"]:
            row.update({"Suggested margin": check["margin"], "Advice": advice(check), "MAE": check["mae"]})
        rows.append(row)
    weeks = pd.DataFrame(rows)
    streak, previous, streaks = 0, None, []
    for reading in weeks["Advice"]:
        flagged = isinstance(reading, str) and reading != "Margin about right"
        streak = (streak + 1 if reading == previous else 1) if flagged else 0
        previous = reading
        streaks.append(streak)
    weeks["Weeks in a row"] = streaks
    return weeks
