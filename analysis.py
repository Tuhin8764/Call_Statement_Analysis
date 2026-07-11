"""
analysis.py
Core data-processing and analysis logic for the Call Statement Analyzer.
Contains NO Streamlit / plotting code on purpose, so it can be tested
and reused independently of the UI layer.
"""

import re
import datetime as dt
from math import log2

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────
# 1. LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────

def read_any(file, **kwargs):
    """Read a CSV or Excel file-like object / path into a DataFrame."""
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file, **kwargs)
    return pd.read_csv(file, **kwargs)


def _parse_time_value(val):
    """Parse a single time-like value (string, datetime.time, datetime.datetime)."""
    if pd.isna(val):
        return None
    if isinstance(val, dt.time):
        return val
    if isinstance(val, dt.datetime):
        return val.time()
    s = str(val).strip()
    # Normalize dot-separated times (22.25.14 -> 22:25:14)
    s = re.sub(r"(\d{1,2})\.(\d{1,2})\.(\d{1,2})", r"\1:\2:\3", s)
    try:
        return pd.to_datetime(s).time()
    except Exception:
        return None


def parse_date_column(series: pd.Series, date_format: str = "auto") -> pd.Series:
    """Parse a date column. date_format='auto' lets pandas infer per-row."""
    if date_format and date_format != "auto":
        return pd.to_datetime(series, format=date_format, errors="coerce")
    return pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")


def parse_time_column(series: pd.Series) -> pd.Series:
    """Parse a time column into python datetime.time objects (robust to . or : separators)."""
    return series.apply(_parse_time_value)


def clean_phone(num) -> str:
    """Normalize a phone number to its last 10 digits for reliable matching."""
    digits = re.sub(r"\D", "", str(num))
    return digits[-10:] if len(digits) >= 10 else digits


def build_call_dataframe(raw: pd.DataFrame, date_format: str = "auto") -> pd.DataFrame:
    """
    Take a raw 5-column dataframe (Sl no, Date, Time, Ph Number, Call time(sec))
    in that column order, and return a cleaned standard dataframe.
    """
    df = raw.copy()
    df = df.iloc[:, :5]
    df.columns = ["sl_no", "date", "time", "phone", "duration"]

    df["date"] = parse_date_column(df["date"], date_format)
    df["time_obj"] = parse_time_column(df["time"])
    df["hour"] = df["time_obj"].apply(lambda t: t.hour if t is not None else np.nan)
    df["minute"] = df["time_obj"].apply(lambda t: t.minute if t is not None else np.nan)

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["phone"] = df["phone"].astype(str).str.strip()
    df["phone_clean"] = df["phone"].apply(clean_phone)

    # drop rows where date or hour failed to parse - they're unusable
    before = len(df)
    df = df.dropna(subset=["date", "hour"]).copy()
    dropped = before - len(df)

    df["hour"] = df["hour"].astype(int)
    df["weekday_num"] = df["date"].dt.dayofweek
    df["weekday_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["weekday_num"] >= 5
    df["is_late"] = df["hour"].apply(lambda h: h >= 22 or h < 6)
    df["date_only"] = df["date"].dt.floor("D")
    df["month"] = df["date"].dt.to_period("M")
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["time_slot"] = df["hour"].apply(_time_slot)

    df.attrs["rows_dropped"] = dropped
    return df


def _time_slot(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning (5-12)"
    elif 12 <= hour < 17:
        return "Afternoon (12-17)"
    elif 17 <= hour < 22:
        return "Evening (17-22)"
    else:
        return "Late night (22-5)"


def load_contacts(file) -> pd.DataFrame:
    """Load a 2-column Name/Contacts file and return a cleaned name<->phone table."""
    contacts = read_any(file)
    contacts = contacts.iloc[:, :2]
    contacts.columns = ["name", "phone"]
    contacts["name"] = contacts["name"].astype(str).str.strip()
    contacts["phone_clean"] = contacts["phone"].apply(clean_phone)
    contacts = contacts.drop_duplicates(subset="phone_clean", keep="first")
    return contacts


def merge_names(df: pd.DataFrame, contacts: pd.DataFrame | None) -> pd.DataFrame:
    """Attach a `name` column to df using the contacts lookup. Falls back to phone number."""
    df = df.copy()
    if contacts is None or contacts.empty:
        df["name"] = df["phone"]
        df["matched"] = False
        return df

    phone_to_name = dict(zip(contacts["phone_clean"], contacts["name"]))
    df["name"] = df["phone_clean"].map(phone_to_name)
    df["matched"] = df["name"].notna()
    df["name"] = df["name"].fillna("Unknown (" + df["phone"].astype(str) + ")")
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. SMALL HELPERS
# ──────────────────────────────────────────────────────────────────────

def _norm_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(2)


def _safe_entropy_concentration(counts: pd.Series) -> float:
    """0 = perfectly spread out, 100 = perfectly concentrated in one bucket."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = (counts / total)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 100.0
    entropy = -sum(p * log2(p) for p in probs)
    max_entropy = log2(len(probs))
    if max_entropy == 0:
        return 100.0
    return round((1 - entropy / max_entropy) * 100, 1)


# ──────────────────────────────────────────────────────────────────────
# 3. OVERVIEW / STATISTICAL SUMMARY
# ──────────────────────────────────────────────────────────────────────

def overview_stats(df: pd.DataFrame) -> dict:
    return {
        "total_calls": len(df),
        "unique_contacts": df["name"].nunique(),
        "total_duration_sec": int(df["duration"].sum()),
        "avg_duration_sec": round(df["duration"].mean(), 1),
        "median_duration_sec": round(df["duration"].median(), 1),
        "longest_call_sec": int(df["duration"].max()),
        "shortest_call_sec": int(df["duration"].min()),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
        "days_covered": (df["date"].max() - df["date"].min()).days + 1,
        "weekday_calls": int((~df["is_weekend"]).sum()),
        "weekend_calls": int(df["is_weekend"].sum()),
        "late_night_calls": int(df["is_late"].sum()),
        "matched_pct": round(df["matched"].mean() * 100, 1) if "matched" in df else None,
    }


# ──────────────────────────────────────────────────────────────────────
# 4. TIME PATTERNS
# ──────────────────────────────────────────────────────────────────────

def hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("hour").agg(
        call_count=("name", "count"),
        avg_duration=("duration", "mean"),
    ).reindex(range(24), fill_value=0).reset_index()
    out["is_late"] = out["hour"].apply(lambda h: h >= 22 or h < 6)
    return out


def weekday_distribution(df: pd.DataFrame) -> pd.DataFrame:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = df.groupby("weekday_num").size().reindex(range(7), fill_value=0)
    out = out.reset_index()
    out.columns = ["weekday_num", "call_count"]
    out["weekday"] = names
    out["is_weekend"] = out["weekday_num"] >= 5
    return out


def time_slot_distribution(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Morning (5-12)", "Afternoon (12-17)", "Evening (17-22)", "Late night (22-5)"]
    out = df["time_slot"].value_counts().reindex(order, fill_value=0).reset_index()
    out.columns = ["time_slot", "call_count"]
    return out


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("date_only").size().reset_index()
    out.columns = ["date", "call_count"]
    out = out.sort_values("date").reset_index(drop=True)
    out["rolling_7d"] = out["call_count"].rolling(7, min_periods=1).mean()
    return out


# ──────────────────────────────────────────────────────────────────────
# 5. CONTACT ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def contact_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("name").agg(
        total_calls=("name", "count"),
        total_duration=("duration", "sum"),
        avg_duration=("duration", "mean"),
        late_pct=("is_late", "mean"),
        weekend_pct=("is_weekend", "mean"),
        first_call=("date", "min"),
        last_call=("date", "max"),
    ).reset_index()
    out["late_pct"] = (out["late_pct"] * 100).round(1)
    out["weekend_pct"] = (out["weekend_pct"] * 100).round(1)
    out["avg_duration"] = out["avg_duration"].round(1)
    return out.sort_values("total_calls", ascending=False).reset_index(drop=True)


def relationship_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto-style inner circle / close / casual contact tiers by share of call time."""
    cs = contact_summary(df).sort_values("total_duration", ascending=False).reset_index(drop=True)
    total = cs["total_duration"].sum()
    cs["cum_pct"] = (cs["total_duration"].cumsum() / total * 100).round(1) if total else 0

    def tier_of(p):
        if p <= 50:
            return "Inner circle"
        elif p <= 80:
            return "Close contact"
        return "Casual contact"

    cs["tier"] = cs["cum_pct"].apply(tier_of)
    return cs


def communication_dna(df: pd.DataFrame, name: str | None = None) -> dict:
    """6-axis communication fingerprint, for one contact or the whole dataset."""
    data = df if name is None else df[df["name"] == name]
    total = len(data)
    if total == 0:
        return {k: 0 for k in
                ["Frequency", "Duration", "Night calls", "Weekend", "Variety", "Consistency"]}

    max_calls_per_contact = df.groupby("name").size().max()
    freq_score = min(total / max(max_calls_per_contact, 1) * 100, 100)
    dur_score = min(data["duration"].mean() / 600 * 100, 100)
    night_score = data["is_late"].mean() * 100
    wknd_score = data["is_weekend"].mean() * 100

    if name is None:
        counts = data["name"].value_counts()
    else:
        counts = data["hour"].value_counts()
    probs = counts / counts.sum()
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    variety = min(entropy / log2(max(len(counts), 2)) * 100, 100)

    hourly = data.groupby("hour").size()
    consistency = min((hourly.max() / hourly.sum()) * 200, 100) if hourly.sum() else 0

    return {
        "Frequency": round(freq_score, 1),
        "Duration": round(dur_score, 1),
        "Night calls": round(night_score, 1),
        "Weekend": round(wknd_score, 1),
        "Variety": round(variety, 1),
        "Consistency": round(consistency, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# 6. PATTERN / ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────────────

def frequency_spikes(df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    cf = df["name"].value_counts().reset_index()
    cf.columns = ["name", "call_count"]
    mean, std = cf["call_count"].mean(), cf["call_count"].std()
    cf["z_score"] = 0.0 if std == 0 else ((cf["call_count"] - mean) / std).round(2)
    cf["is_spike"] = cf["z_score"] > z_thresh
    return cf


def outlier_calls(df: pd.DataFrame, pct: float = 0.99) -> tuple[pd.DataFrame, float]:
    threshold = df["duration"].quantile(pct)
    out = df[df["duration"] > threshold].sort_values("duration", ascending=False)
    return out, threshold


def call_gap_pattern(df: pd.DataFrame, name: str) -> dict:
    sub = df[df["name"] == name].sort_values("date").copy()
    if len(sub) < 3:
        return {"enough_data": False, "calls": len(sub)}

    sub["gap_days"] = sub["date"].diff().dt.total_seconds() / 86400
    sub = sub.dropna(subset=["gap_days"])

    return {
        "enough_data": True,
        "calls": len(sub) + 1,
        "avg_gap": round(sub["gap_days"].mean(), 1),
        "std_gap": round(sub["gap_days"].std(), 1) if len(sub) > 1 else 0.0,
        "median_gap": round(sub["gap_days"].median(), 1),
        "is_routine": (sub["gap_days"].std() < 2) if len(sub) > 1 else False,
        "timeline": sub[["date", "gap_days"]],
    }


def behavior_change_point(df: pd.DataFrame) -> dict:
    daily = df.groupby("date_only").agg(
        call_count=("name", "count"),
        unique_contacts=("name", "nunique"),
        late_night=("is_late", "sum"),
    ).reset_index().sort_values("date_only").reset_index(drop=True)

    daily["rolling_calls"] = daily["call_count"].rolling(7, min_periods=1).mean()
    daily["rolling_late"] = daily["late_night"].rolling(7, min_periods=1).mean()
    daily["call_diff"] = daily["rolling_calls"].diff().abs()

    if daily["call_diff"].notna().sum() == 0:
        change_date = daily["date_only"].iloc[0]
    else:
        change_date = daily.loc[daily["call_diff"].idxmax(), "date_only"]

    before = df[df["date_only"] < change_date]
    after = df[df["date_only"] >= change_date]

    return {
        "daily": daily,
        "change_date": change_date,
        "before_avg_calls": round(before.groupby("date_only").size().mean(), 1) if len(before) else 0,
        "after_avg_calls": round(after.groupby("date_only").size().mean(), 1) if len(after) else 0,
        "before_contacts": before["name"].nunique(),
        "after_contacts": after["name"].nunique(),
        "before_late": int(before["is_late"].sum()),
        "after_late": int(after["is_late"].sum()),
    }


def weekly_rhythm(df: pd.DataFrame, name: str) -> dict:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sub = df[df["name"] == name]
    heat = sub.groupby(["weekday_num", "hour"]).size().unstack(fill_value=0)
    heat = heat.reindex(range(7), fill_value=0)
    heat.index = day_names
    heat = heat.reindex(columns=range(24), fill_value=0)

    concentration = _safe_entropy_concentration(sub.groupby(["weekday_num", "hour"]).size())
    return {"heatmap": heat, "concentration": concentration, "is_routine": concentration > 60}


def anomaly_days(df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    daily = df.groupby("date_only").agg(
        calls=("name", "count"),
        duration=("duration", "sum"),
        unique_contacts=("name", "nunique"),
        late_calls=("is_late", "sum"),
    ).reset_index().sort_values("date_only").reset_index(drop=True)

    def safe_z(s):
        if s.std() == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return pd.Series(np.abs(stats.zscore(s)), index=s.index)

    for col in ["calls", "duration", "unique_contacts", "late_calls"]:
        daily[f"{col}_z"] = safe_z(daily[col])

    daily["anomaly"] = (
        (daily["calls_z"] > z_thresh) | (daily["duration_z"] > z_thresh) |
        (daily["unique_contacts_z"] > z_thresh) | (daily["late_calls_z"] > z_thresh)
    )
    return daily


def loyalty_shift(df: pd.DataFrame, top_n: int = 5) -> dict:
    midpoint = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
    top = df["name"].value_counts().head(top_n).index.tolist()

    monthly = df[df["name"].isin(top)].groupby(["month", "name"]).size().unstack(fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1).replace(0, np.nan), axis=0) * 100
    monthly_pct = monthly_pct.fillna(0)

    if len(monthly_pct) == 0:
        return {"monthly_pct": monthly_pct, "shift_detected": False}

    first_top = monthly_pct.iloc[0].idxmax()
    last_top = monthly_pct.iloc[-1].idxmax()
    shift_month = None
    for i in range(1, len(monthly_pct)):
        if monthly_pct.iloc[i].idxmax() != monthly_pct.iloc[i - 1].idxmax():
            shift_month = monthly_pct.index[i]
            break

    return {
        "monthly_pct": monthly_pct,
        "shift_detected": first_top != last_top,
        "first_top": first_top,
        "last_top": last_top,
        "shift_month": str(shift_month) if shift_month is not None else None,
        "midpoint": midpoint,
    }


# ──────────────────────────────────────────────────────────────────────
# 7. COMPOSITE IMPORTANCE / TRUST SCORE  (combines every signal above)
# ──────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "S1_frequency": 0.20,
    "S2_duration": 0.15,
    "S3_late_night": 0.20,
    "S4_weekend": 0.10,
    "S5_growth": 0.10,
    "S6_recency": 0.05,
    "S7_outlier": 0.05,
    "S8_rhythm": 0.10,
    "S9_loyalty": 0.05,
}

SCORE_COLS = list(WEIGHTS.keys())


def composite_score(df: pd.DataFrame) -> pd.DataFrame:
    midpoint = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
    q99 = df["duration"].quantile(0.99)
    first_half_total = max(len(df[df["date"] < midpoint]), 1)
    second_half_total = max(len(df[df["date"] >= midpoint]), 1)
    data_span_days = max((df["date"].max() - df["date"].min()).days, 1)

    records = []
    for person, grp in df.groupby("name"):
        grp = grp.sort_values("date")
        total = len(grp)

        avg_dur = grp["duration"].mean()
        late_ratio = grp["is_late"].mean() * 100
        wknd_ratio = grp["is_weekend"].mean() * 100

        first_n = (grp["date"] < midpoint).sum()
        second_n = (grp["date"] >= midpoint).sum()
        growth = max((second_n - first_n) / max(first_n, 1) * 100, 0)

        days_since_first = (df["date"].max() - grp["date"].min()).days
        recency_raw = max(0, data_span_days - days_since_first)

        outlier_count = (grp["duration"] > q99).sum()
        outlier_raw = outlier_count / total * 100

        concentration = _safe_entropy_concentration(
            grp.groupby(["weekday_num", "hour"]).size()
        )

        first_share = first_n / first_half_total * 100
        second_share = second_n / second_half_total * 100
        loyalty_growth = max(second_share - first_share, 0)

        records.append({
            "name": person,
            "total_calls": total,
            "avg_duration": round(avg_dur, 1),
            "late_pct": round(late_ratio, 1),
            "weekend_pct": round(wknd_ratio, 1),
            "_freq_raw": total,
            "_dur_raw": avg_dur,
            "_late_raw": late_ratio,
            "_wknd_raw": wknd_ratio,
            "_growth_raw": growth,
            "_recency_raw": recency_raw,
            "_outlier_raw": outlier_raw,
            "_rhythm_raw": concentration,
            "_loyalty_raw": loyalty_growth,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["S1_frequency"] = _norm_0_100(result["_freq_raw"])
    result["S2_duration"] = _norm_0_100(result["_dur_raw"])
    result["S3_late_night"] = _norm_0_100(result["_late_raw"])
    result["S4_weekend"] = _norm_0_100(result["_wknd_raw"])
    result["S5_growth"] = _norm_0_100(result["_growth_raw"])
    result["S6_recency"] = _norm_0_100(result["_recency_raw"])
    result["S7_outlier"] = _norm_0_100(result["_outlier_raw"])
    result["S8_rhythm"] = _norm_0_100(result["_rhythm_raw"])
    result["S9_loyalty"] = _norm_0_100(result["_loyalty_raw"])

    result["FINAL_SCORE"] = sum(result[c] * w for c, w in WEIGHTS.items()).round(1)

    def tier(s):
        if s >= 75:
            return "Critical"
        if s >= 55:
            return "High"
        if s >= 35:
            return "Medium"
        return "Low"

    result["tier"] = result["FINAL_SCORE"].apply(tier)
    result = result.sort_values("FINAL_SCORE", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result
