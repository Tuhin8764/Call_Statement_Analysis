"""
app.py
Call Statement Analyzer — Streamlit web application.

Run with:
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd

import analysis as A
import charts as C
import report as R

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLE
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Call Statement Analyzer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 3.2rem; padding-bottom: 2rem;}
    div[data-testid="stTabs"] {margin-top: 0.4rem;}
    div[data-testid="stTabs"] button {padding-top: 8px; padding-bottom: 8px;}
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 14px 16px; text-align: center;
    }
    .metric-card .label {font-size: 12px; color: var(--text-color); opacity: 0.65; margin-bottom: 4px;}
    .metric-card .value {font-size: 22px; font-weight: 700; color: var(--text-color);}
    .why-box {
        background: rgba(37, 99, 235, 0.12); border-left: 4px solid #2563EB; border-radius: 6px;
        padding: 12px 16px; margin: 10px 0 16px 0; font-size: 14px; color: var(--text-color);
    }
    .finding-box {
        background: rgba(234, 88, 12, 0.12); border-left: 4px solid #EA580C; border-radius: 6px;
        padding: 12px 16px; margin: 6px 0 18px 0; font-size: 14px; color: var(--text-color);
    }
    .tier-critical {color: #DC2626; font-weight: 700;}
    .tier-high {color: #EA580C; font-weight: 700;}
    .tier-medium {color: #D97706; font-weight: 700;}
    .tier-low {color: #2563EB; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


def explain(why: str, benefit: str):
    st.markdown(f"""
    <div class="why-box">
        <b>Why this analysis:</b> {why}<br>
        <b>Benefit:</b> {benefit}
    </div>
    """, unsafe_allow_html=True)


def finding(text: str):
    st.markdown(f"""<div class="finding-box">💡 <b>Key finding:</b> {text}</div>""",
                unsafe_allow_html=True)


def top3_caption(df_sorted: pd.DataFrame, name_col: str, value_col: str,
                  value_fmt: str = "{:.0f}", unit: str = ""):
    """Render a compact, always-visible Top 3 line (name + value) under a chart."""
    medals = ["🥇", "🥈", "🥉"]
    top3 = df_sorted.head(3)
    parts = []
    for i, (_, row) in enumerate(top3.iterrows()):
        val = value_fmt.format(row[value_col])
        parts.append(f"{medals[i]} **{row[name_col]}** ({val}{unit})")
    if parts:
        st.markdown("**Top 3:** " + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts))


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA LOADING
# ════════════════════════════════════════════════════════════════════

st.sidebar.title("📞 Call Statement Analyzer")
st.sidebar.caption("Upload your call record file to begin.")

call_file = st.sidebar.file_uploader(
    "Call records (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Expected columns in order: Sl no., Date, Time, Ph Number, Call time(sec)",
)

date_fmt_choice = st.sidebar.selectbox(
    "Date format in file",
    ["Auto-detect", "DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD"],
    index=0,
)
date_fmt_map = {
    "Auto-detect": "auto",
    "DD-MM-YYYY": "%d-%m-%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
}

st.sidebar.markdown("---")
st.sidebar.subheader("Contact names (optional)")
contacts_file = st.sidebar.file_uploader(
    "Contacts file (Name + Phone, Excel/CSV)",
    type=["csv", "xlsx", "xls"],
    help="Two columns: Name, Contacts. Calls will be shown with names instead of raw numbers.",
)

st.sidebar.markdown("---")
use_demo = st.sidebar.button("🎲 Try with demo data instead", width='stretch')

st.sidebar.markdown("---")
st.sidebar.caption(
    "All processing happens locally in this app session. "
    "No data is uploaded anywhere else."
)


@st.cache_data(show_spinner=False)
def _process(call_bytes, call_name, date_fmt, contacts_bytes, contacts_name):
    call_io = io.BytesIO(call_bytes)
    call_io.name = call_name
    raw = A.read_any(call_io)
    df = A.build_call_dataframe(raw, date_format=date_fmt)

    contacts = None
    if contacts_bytes is not None:
        c_io = io.BytesIO(contacts_bytes)
        c_io.name = contacts_name
        contacts = A.load_contacts(c_io)

    df = A.merge_names(df, contacts)
    return df


df = None
load_error = None

if use_demo:
    st.session_state["use_demo"] = True
if call_file is not None:
    st.session_state["use_demo"] = False

try:
    if st.session_state.get("use_demo"):
        import numpy as np
        rng = np.random.default_rng(7)
        names = [f"Contact {chr(65+i)}" for i in range(18)]
        phone_pool = [f"9{rng.integers(100000000, 999999999)}" for _ in names]

        # A few contacts dominate (Pareto-style), most are occasional
        raw_w = np.array([40, 28, 16] + [1] * (len(names) - 3), dtype=float)
        contact_p = raw_w / raw_w.sum()

        rows = []
        start = pd.Timestamp("2026-01-01")
        for i in range(700):
            day_offset = int(rng.integers(0, 150))
            date = start + pd.Timedelta(days=day_offset)
            weights = np.array([1,1,1,1,1,1,2,3,4,5,5,5,5,5,5,6,6,7,8,9,9,8,5,3], dtype=float)
            hour = int(rng.choice(24, p=weights / weights.sum()))
            minute, second = int(rng.integers(0, 60)), int(rng.integers(0, 60))
            phone = phone_pool[rng.choice(len(phone_pool), p=contact_p)]
            duration = int(rng.exponential(120)) + 5
            rows.append([i + 1, date.strftime("%d-%m-%Y"), f"{hour:02d}.{minute:02d}.{second:02d}",
                         phone, duration])
        demo_raw = pd.DataFrame(rows, columns=["Sl no.", "Date", "Time", "Ph Number", "Call time(sec)"])
        df = A.build_call_dataframe(demo_raw, date_format="auto")
        name_map = dict(zip(phone_pool, names))
        df["name"] = df["phone"].map(name_map)
        df["matched"] = True

    elif call_file is not None:
        contacts_bytes = contacts_file.getvalue() if contacts_file is not None else None
        contacts_name = contacts_file.name if contacts_file is not None else None
        df = _process(call_file.getvalue(), call_file.name,
                       date_fmt_map[date_fmt_choice], contacts_bytes, contacts_name)

except Exception as e:
    load_error = str(e)


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════

if load_error:
    st.error(f"Could not process the uploaded file: {load_error}")
    st.info("Check that your columns are in the order: Sl no., Date, Time, Ph Number, "
            "Call time(sec), and try a different date format from the sidebar.")
    st.stop()

if df is None:
    st.title("📞 Call Statement Analyzer")
    st.markdown(
        "Upload a call record file from the sidebar to get a full statistical, "
        "behavioral and pattern-based breakdown — or click **Try with demo data** "
        "to explore the app right away."
    )
    st.markdown("##### Expected file format")
    st.dataframe(pd.DataFrame({
        "Sl no.": [1, 2], "Date": ["21-01-2026", "22-01-2026"],
        "Time": ["18:57:21", "09:14:02"], "Ph Number": ["9876543210", "9123456780"],
        "Call time(sec)": [78, 245],
    }), width='stretch', hide_index=True)
    st.stop()

if df.attrs.get("rows_dropped", 0) > 0:
    st.warning(f"⚠️ {df.attrs['rows_dropped']} row(s) could not be parsed (bad date/time "
               f"format) and were excluded from analysis.")

if "matched" in df and not df["matched"].all() and df["matched"].any():
    match_pct = df["matched"].mean() * 100
    st.info(f"ℹ️ {match_pct:.1f}% of calls matched a name from your contacts file. "
            f"Unmatched numbers are shown as 'Unknown (number)'.")

stats = A.overview_stats(df)
cs = A.contact_summary(df)
rt = A.relationship_tiers(df)
all_names = cs["name"].tolist()
top_name_default = all_names[0]

tab_overview, tab_time, tab_contacts, tab_patterns, tab_score, tab_report = st.tabs(
    ["📊 Overview", "⏰ Time Patterns", "👥 Contacts", "🔍 Pattern Detection",
     "⚠️ Importance Score", "📄 Download Report"]
)

# ──────────────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ──────────────────────────────────────────────────────────────────────
with tab_overview:
    st.header("Overview & Statistical Summary")
    explain(
        "A statistical summary establishes the baseline scale of the dataset before "
        "diving into deeper behavioral patterns.",
        "Gives an immediate, honest sense of volume and time span, so every later "
        "percentage or score can be interpreted in context.",
    )

    metric_cards([
        ("Total calls", f"{stats['total_calls']:,}"),
        ("Unique contacts", stats["unique_contacts"]),
        ("Avg duration", f"{stats['avg_duration_sec']:.0f}s"),
        ("Longest call", f"{stats['longest_call_sec']}s"),
        ("Days covered", stats["days_covered"]),
    ])
    st.markdown("")
    finding(
        f"The dataset spans <b>{stats['days_covered']} days</b> "
        f"({stats['date_start'].date()} → {stats['date_end'].date()}) with "
        f"<b>{stats['total_calls']:,} calls</b> to <b>{stats['unique_contacts']} contacts</b>, "
        f"averaging {stats['avg_duration_sec']:.0f} seconds per call."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(C.weekday_weekend_donut(stats["weekday_calls"], stats["weekend_calls"]),
                         width='stretch')
    with col2:
        st.plotly_chart(C.time_slot_bar(A.time_slot_distribution(df)), width='stretch')

# ──────────────────────────────────────────────────────────────────────
# TAB: TIME PATTERNS
# ──────────────────────────────────────────────────────────────────────
with tab_time:
    st.header("Time-Based Patterns")

    st.subheader("Calls by hour of day")
    explain(
        "Calling activity is rarely uniform across the day. Mapping it by hour reveals "
        "exactly when communication happens.",
        "Separates routine daytime contact from late-night or off-hours communication, "
        "which often carries different significance.",
    )
    hourly = A.hourly_distribution(df)
    st.plotly_chart(C.hourly_bar(hourly), width='stretch')
    peak_hour = int(hourly.loc[hourly["call_count"].idxmax(), "hour"])
    late_pct = round(df["is_late"].mean() * 100, 1)
    finding(f"The busiest hour is <b>{peak_hour:02d}:00</b>, and "
            f"<b>{late_pct}%</b> of all calls happen between 10 PM and 6 AM.")

    st.divider()
    st.subheader("Calls by day of week")
    explain(
        "People have natural weekly rhythms. Knowing which day carries the most "
        "activity highlights routine vs irregular behavior.",
        "A sudden change in the busiest day over time can flag a shift in schedule, "
        "routine, or relationships.",
    )
    wd = A.weekday_distribution(df)
    st.plotly_chart(C.weekday_bar(wd), width='stretch')
    busiest_day = wd.loc[wd["call_count"].idxmax(), "weekday"]
    finding(f"<b>{busiest_day}</b> is the most active day of the week in this dataset.")

    st.divider()
    st.subheader("Daily call volume trend")
    explain(
        "Tracking daily volume over the full period shows whether communication is "
        "stable, growing, or spiking around specific dates.",
        "Long unexplained spikes are easy to spot visually and can be cross-referenced "
        "against real-world events or dates of interest.",
    )
    trend = A.daily_trend(df)
    st.plotly_chart(C.daily_trend_line(trend), width='stretch')
    peak_row = trend.loc[trend["call_count"].idxmax()]
    finding(f"The single busiest day was <b>{peak_row['date'].date()}</b> with "
            f"<b>{int(peak_row['call_count'])} calls</b>.")

# ──────────────────────────────────────────────────────────────────────
# TAB: CONTACTS
# ──────────────────────────────────────────────────────────────────────
with tab_contacts:
    st.header("Contact Analysis")

    st.subheader("Top contacts")
    explain(
        "Raw call counts and total talk-time immediately show who dominates the "
        "communication pattern.",
        "Separates habitual, high-frequency contacts from occasional or one-off numbers.",
    )
    metric_choice = st.radio("Rank by", ["Number of calls", "Total duration"],
                              horizontal=True, label_visibility="collapsed")
    metric_col = "total_calls" if metric_choice == "Number of calls" else "total_duration"
    st.plotly_chart(C.top_contacts_bar(cs, top_n=15, metric=metric_col), width='stretch')
    finding(f"<b>{top_name_default}</b> is the most frequently called contact, with "
            f"<b>{int(cs.iloc[0]['total_calls'])} calls</b> totaling "
            f"{cs.iloc[0]['total_duration']/60:.0f} minutes.")
    top3_caption(cs.sort_values(metric_col, ascending=False), "name", metric_col,
                 unit=" calls" if metric_col == "total_calls" else "s")

    st.divider()
    st.subheader("Relationship map — who really matters")
    explain(
        "Not every contact deserves equal attention — a small number of people "
        "typically account for the majority of call time (the Pareto principle).",
        "Strips away noise from rarely-called numbers and focuses attention on who "
        "genuinely matters in this person's communication life.",
    )
    st.plotly_chart(C.relationship_bubble(rt), width='stretch')
    tier_counts = rt["tier"].value_counts()
    finding(f"<b>{tier_counts.get('Inner circle', 0)} contact(s)</b> form the inner circle, "
            f"accounting for roughly the first 50% of all call time. "
            f"{tier_counts.get('Close contact', 0)} are close contacts, and "
            f"{tier_counts.get('Casual contact', 0)} are casual/occasional contacts.")
    top3_caption(rt.sort_values("total_duration", ascending=False).assign(
        total_minutes=lambda d: (d["total_duration"] / 60).round(0)),
        "name", "total_minutes", unit=" min")

    st.divider()
    st.subheader("Communication DNA fingerprint")
    explain(
        "Builds a 6-axis communication profile (frequency, duration, night calls, "
        "weekend activity, variety and consistency) for one contact, compared to the "
        "dataset's overall average.",
        "Reveals whether a specific contact's communication style stands out from "
        "what's typical in this dataset — a useful way to spot a contact who doesn't "
        "fit the normal pattern.",
    )
    dna_name = st.selectbox("Select a contact", all_names, index=0, key="dna_select")
    contact_dna = A.communication_dna(df, dna_name)
    baseline_dna = A.communication_dna(df, None)
    st.plotly_chart(C.communication_dna_radar(contact_dna, baseline_dna, dna_name),
                     width='stretch')
    biggest_gap = max(contact_dna, key=lambda k: abs(contact_dna[k] - baseline_dna[k]))
    finding(f"<b>{dna_name}</b> differs most from the dataset average on "
            f"<b>{biggest_gap}</b> ({contact_dna[biggest_gap]:.0f} vs "
            f"{baseline_dna[biggest_gap]:.0f} average).")

# ──────────────────────────────────────────────────────────────────────
# TAB: PATTERN DETECTION
# ──────────────────────────────────────────────────────────────────────
with tab_patterns:
    st.header("Pattern & Anomaly Detection")

    st.subheader("Frequency spike detection")
    explain(
        "A statistical z-score test flags contacts called far more often than the "
        "typical contact in this dataset.",
        "Quickly separates ordinary contacts from ones receiving unusually concentrated "
        "attention, without manually scanning every row.",
    )
    spikes = A.frequency_spikes(df)
    st.plotly_chart(C.frequency_spike_chart(spikes), width='stretch')
    n_spikes = int(spikes["is_spike"].sum())
    finding(f"<b>{n_spikes} contact(s)</b> show a statistically significant frequency "
            f"spike (more than 2 standard deviations above the average contact).")
    top3_caption(spikes.sort_values("call_count", ascending=False), "name", "call_count",
                 unit=" calls")

    st.divider()
    st.subheader("Outlier call duration")
    explain(
        "A small number of unusually long calls can carry more behavioral signal than "
        "hundreds of short, routine ones.",
        "Surfaces conversations that stand out by length, which often indicates deeper "
        "or more significant exchanges.",
    )
    outliers, threshold = A.outlier_calls(df)
    st.plotly_chart(C.outlier_scatter(df, outliers, threshold), width='stretch')
    finding(f"<b>{len(outliers)} call(s)</b> exceed the 99th percentile duration of "
            f"<b>{threshold/60:.1f} minutes</b>. The single longest call lasted "
            f"{df['duration'].max()/60:.1f} minutes.")
    if len(outliers) > 0:
        top3_caption(outliers.assign(duration_min=lambda d: (d["duration"] / 60).round(1))
                     .sort_values("duration_min", ascending=False),
                     "name", "duration_min", value_fmt="{:.1f}", unit=" min")
    with st.expander("View outlier call details"):
        st.dataframe(outliers[["date", "name", "duration"]].assign(
            duration_min=lambda d: (d["duration"] / 60).round(1)
        )[["date", "name", "duration_min"]].rename(columns={"duration_min": "duration (min)"}),
            width='stretch', hide_index=True)

    st.divider()
    st.subheader("Call gap pattern (regularity)")
    explain(
        "Checks whether calls to a specific contact happen at suspiciously regular "
        "intervals, which can indicate a scheduled routine rather than casual contact.",
        "A very regular gap (low variation) suggests a fixed habit or schedule; a wide, "
        "irregular gap suggests casual or reactive contact.",
    )
    gap_name = st.selectbox("Select a contact", all_names, index=0, key="gap_select")
    gap_info = A.call_gap_pattern(df, gap_name)
    if not gap_info["enough_data"]:
        st.info(f"{gap_name} has only {gap_info['calls']} call(s) — not enough data "
                f"to detect a gap pattern (minimum 3 needed).")
    else:
        st.plotly_chart(C.call_gap_line(gap_info, gap_name), width='stretch')
        routine_text = "a very regular, routine-like schedule" if gap_info["is_routine"] \
            else "no strong fixed schedule"
        finding(f"<b>{gap_name}</b> is called every <b>{gap_info['avg_gap']} days</b> on "
                f"average (std dev: {gap_info['std_gap']} days) — this suggests "
                f"{routine_text}.")

    st.divider()
    st.subheader("Behavior change point")
    explain(
        "Communication habits are not static. This finds the date where overall "
        "behavior shifted most sharply, using a 7-day rolling-average comparison.",
        "Pinpoints exactly when a routine changed, which can be cross-checked against "
        "real-world events or relationship milestones.",
    )
    bcp = A.behavior_change_point(df)
    st.plotly_chart(C.change_point_chart(bcp), width='stretch')
    finding(f"The largest behavioral shift occurred around "
            f"<b>{bcp['change_date'].date()}</b>. Before: {bcp['before_avg_calls']} "
            f"calls/day across {bcp['before_contacts']} contacts. After: "
            f"{bcp['after_avg_calls']} calls/day across {bcp['after_contacts']} contacts.")

    st.divider()
    st.subheader("Recurring weekly rhythm")
    explain(
        "Plots one contact's calls by day-of-week and hour to reveal whether contact "
        "follows a fixed routine or happens at random times.",
        "A highly concentrated, repeating time slot suggests a scheduled habit; a "
        "scattered pattern suggests casual or reactive contact.",
    )
    rhythm_name = st.selectbox("Select a contact", all_names, index=0, key="rhythm_select")
    wr = A.weekly_rhythm(df, rhythm_name)
    st.plotly_chart(C.weekly_rhythm_heatmap(wr["heatmap"], rhythm_name), width='stretch')
    routine_word = "a clear recurring routine" if wr["is_routine"] else "no strong fixed routine"
    finding(f"<b>{rhythm_name}</b> has a routine concentration score of "
            f"<b>{wr['concentration']}/100</b> — indicating {routine_word}.")

    st.divider()
    st.subheader("Statistical anomaly days")
    explain(
        "Z-score anomaly detection flags individual days where call count, total "
        "duration, contact variety, or late-night activity deviated sharply from the "
        "dataset's own normal range.",
        "Removes guesswork from spotting 'unusual days' — the flagging is based on "
        "statistics, not a visual guess.",
    )
    ad = A.anomaly_days(df)
    st.plotly_chart(C.anomaly_line(ad), width='stretch')
    n_anom = int(ad["anomaly"].sum())
    finding(f"<b>{n_anom} day(s)</b> out of {len(ad)} total days were flagged as "
            f"statistically anomalous.")
    if n_anom > 0:
        anom_top3 = ad[ad["anomaly"]].sort_values("calls", ascending=False).assign(
            day_label=lambda d: d["date_only"].dt.strftime("%d %b %Y"))
        top3_caption(anom_top3, "day_label", "calls", unit=" calls")
        with st.expander("View anomalous days"):
            st.dataframe(ad[ad["anomaly"]][["date_only", "calls", "duration", "unique_contacts",
                                            "late_calls"]],
                         width='stretch', hide_index=True)

    st.divider()
    st.subheader("Loyalty shift detection")
    explain(
        "Tracks each top contact's share of total monthly calls to see whether one "
        "contact is gradually being replaced by another over time.",
        "A clear crossover — one contact's share falling as another's rises — is one "
        "of the strongest signals that attention has shifted from one relationship "
        "to another.",
    )
    ls = A.loyalty_shift(df)
    if len(ls["monthly_pct"]) > 1:
        st.plotly_chart(C.loyalty_shift_chart(ls["monthly_pct"]), width='stretch')
        if ls["shift_detected"]:
            finding(f"A loyalty shift was detected: <b>{ls['first_top']}</b> was the top "
                    f"contact early on, but <b>{ls['last_top']}</b> became the top contact "
                    f"later" + (f" (around {ls['shift_month']})" if ls["shift_month"] else "") + ".")
        else:
            finding(f"No loyalty shift detected — <b>{ls['first_top']}</b> remains the top "
                    f"contact throughout the entire period.")
    else:
        st.info("Not enough months of data to evaluate a loyalty shift.")

# ──────────────────────────────────────────────────────────────────────
# TAB: IMPORTANCE SCORE
# ──────────────────────────────────────────────────────────────────────
with tab_score:
    st.header("Composite Importance Score")
    explain(
        "No single metric tells the whole story — a contact might be called often but "
        "briefly, or rarely but for hours late at night. This combines frequency, "
        "duration, late-night ratio, weekend ratio, growth, recency, outliers, weekly "
        "rhythm and loyalty shift into one weighted 0–100 score per contact.",
        "Produces one clear ranked list instead of nine separate ones, making it "
        "immediately obvious which contacts stand out overall — useful for triage when "
        "a dataset has many contacts.",
    )

    comp = A.composite_score(df)
    top_comp = comp.iloc[0]

    metric_cards([
        ("#1 contact", top_comp["name"]),
        ("Score", f"{top_comp['FINAL_SCORE']}/100"),
        ("Tier", top_comp["tier"]),
        ("Critical tier contacts", int((comp["tier"] == "Critical").sum())),
    ])
    st.markdown("")

    st.plotly_chart(C.importance_rank_bar(comp), width='stretch')
    finding(f"<b>{top_comp['name']}</b> ranks #1 with a composite score of "
            f"<b>{top_comp['FINAL_SCORE']}/100</b> (tier: {top_comp['tier']}), based on "
            f"{int(top_comp['total_calls'])} calls averaging {top_comp['avg_duration']}s, "
            f"with {top_comp['late_pct']}% at night and {top_comp['weekend_pct']}% on weekends.")
    top3_caption(comp, "name", "FINAL_SCORE", value_fmt="{:.1f}", unit="/100")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(C.importance_radar(top_comp, A.SCORE_COLS), width='stretch')
    with col2:
        st.plotly_chart(C.tier_donut(comp), width='stretch')

    st.plotly_chart(C.importance_heatmap(comp, A.SCORE_COLS), width='stretch')

    with st.expander("View full ranking table"):
        display_cols = ["rank", "name", "total_calls", "avg_duration", "late_pct",
                         "weekend_pct", "FINAL_SCORE", "tier"]
        st.dataframe(comp[display_cols], width='stretch', hide_index=True)

    with st.expander("How are the weights chosen?"):
        st.markdown("""
        | Dimension | What it captures | Weight |
        |---|---|---|
        | Frequency | How often this contact is called | 20% |
        | Duration | Average call length | 15% |
        | Late night | % of calls between 10 PM–6 AM | 20% |
        | Weekend | % of calls on Sat/Sun | 10% |
        | Growth | Increase in calls, 2nd half vs 1st half of period | 10% |
        | Recency | How recently this contact first appeared | 5% |
        | Outlier | Share of this contact's calls that are unusually long | 5% |
        | Rhythm | How concentrated/routine the calling pattern is | 10% |
        | Loyalty | Growth in this contact's share of total monthly calls | 5% |

        You can treat this as a starting point — what matters most depends on context.
        """)

# ──────────────────────────────────────────────────────────────────────
# TAB: DOWNLOAD REPORT
# ──────────────────────────────────────────────────────────────────────
with tab_report:
    st.header("Download Full Report")
    st.markdown(
        "Generate a complete PDF report covering every analysis in this app — "
        "overview statistics, time patterns, contact analysis, pattern/anomaly "
        "detection, and the composite importance score — each with a clear "
        "explanation of **why** the analysis was done, its **benefit**, and the "
        "**specific finding** for this dataset."
    )

    if st.button("📄 Generate PDF report", type="primary", width='stretch'):
        with st.spinner("Building your report... this can take a few seconds."):
            pdf_bytes = R.build_pdf_report(df)
        st.session_state["pdf_bytes"] = pdf_bytes
        st.success("Report ready below.")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Download report (PDF)",
            data=st.session_state["pdf_bytes"],
            file_name="call_statement_analysis_report.pdf",
            mime="application/pdf",
            width='stretch',
        )
