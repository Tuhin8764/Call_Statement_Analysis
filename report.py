"""
report.py
Builds a downloadable PDF report summarizing every analysis.
Uses matplotlib (static images) for chart rendering inside the PDF,
and fpdf2 for page layout - keeps this independent from the Plotly UI layer.
"""

import io
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

import analysis as A

BLUE = "#2563EB"
CORAL = "#E85D30"
RED = "#DC2626"
ORANGE = "#EA580C"
AMBER = "#D97706"
TIER_COLORS = {"Critical": RED, "High": ORANGE, "Medium": AMBER, "Low": BLUE}


def _fig_to_png_bytes(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _top3_text(df_sorted: pd.DataFrame, name_col: str, value_col: str,
                fmt: str = "{:.0f}", unit: str = "") -> str:
    """Build a 'Top 3: 1) X (val)  2) Y (val)  3) Z (val)' line for the PDF body text."""
    top3 = df_sorted.head(3)
    parts = []
    for i, (_, row) in enumerate(top3.iterrows(), start=1):
        parts.append(f"{i}) {row[name_col]} ({fmt.format(row[value_col])}{unit})")
    return "Top 3: " + "   ".join(parts)


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, "Call statement analysis report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"Generated {dt.datetime.now().strftime('%d %b %Y, %H:%M')}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 30, 60)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(37, 99, 235)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def why_box(self, why, benefit, finding):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(37, 99, 235)
        self.cell(0, 6, "Why this analysis:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, why)
        self.ln(1)

        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(37, 99, 235)
        self.cell(0, 6, "Benefit:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, benefit)
        self.ln(1)

        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(200, 80, 20)
        self.cell(0, 6, "Key finding:", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, finding)
        self.ln(3)

    def add_image(self, png_buf, w=170):
        self.image(png_buf, w=w)
        self.ln(3)

    def metric_row(self, items):
        """items: list of (label, value) tuples rendered as a row of boxes."""
        n = len(items)
        col_w = (self.w - self.l_margin - self.r_margin) / n
        y0 = self.get_y()
        for i, (label, value) in enumerate(items):
            x = self.l_margin + i * col_w
            self.set_xy(x, y0)
            self.set_fill_color(241, 245, 249)
            self.rect(x + 1, y0, col_w - 2, 18, style="F")
            self.set_xy(x + 2, y0 + 2)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(100, 100, 100)
            self.cell(col_w - 4, 4, label)
            self.set_xy(x + 2, y0 + 7)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(20, 30, 60)
            self.cell(col_w - 4, 8, str(value))
        self.set_y(y0 + 22)


# ──────────────────────────────────────────────────────────────────────
# Matplotlib chart helpers (static versions for the PDF only)
# ──────────────────────────────────────────────────────────────────────

def _mpl_bar(x, y, colors, title, xlabel, ylabel, figsize=(7, 3), rot=0):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", rotation=rot, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.margins(y=0.15)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _mpl_donut(values, labels, colors, title, figsize=(4.2, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
           wedgeprops=dict(width=0.55, edgecolor="white"))
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _mpl_line(x, ys, labels, colors, title, xlabel, ylabel, figsize=(7, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    for y, label, color in zip(ys, labels, colors):
        ax.plot(x, y, color=color, linewidth=1.8, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _mpl_heatmap(z, xlabels, ylabels, title, figsize=(7, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(z, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    vmax = np.nanmax(z) if np.nanmax(z) > 0 else 1
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            if val == 0:
                continue
            color = "white" if val > vmax * 0.55 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=6.5, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _mpl_hbar(labels, values, colors, title, xlabel, figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=3)
    ax.margins(x=0.12)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ──────────────────────────────────────────────────────────────────────
# Main report builder
# ──────────────────────────────────────────────────────────────────────

def build_pdf_report(df: pd.DataFrame) -> bytes:
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)

    stats = A.overview_stats(df)
    cs = A.contact_summary(df)
    rt = A.relationship_tiers(df)
    top_name = cs.iloc[0]["name"]
    comp = A.composite_score(df)
    top_comp = comp.iloc[0]

    # ── COVER PAGE ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_y(60)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 30, 60)
    pdf.cell(0, 14, "Call Statement Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, f"Period: {stats['date_start'].date()} to {stats['date_end'].date()}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total calls analyzed: {stats['total_calls']:,}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Unique contacts: {stats['unique_contacts']}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 6,
        "This report summarizes statistical, behavioral and pattern-based findings "
        "derived automatically from the uploaded call statement. Each section explains "
        "why the analysis was performed, what it is useful for, and what was found in "
        "this specific dataset.", align="C")

    # ── SECTION 1: OVERVIEW ─────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1. Overview & Statistical Summary")
    pdf.metric_row([
        ("Total calls", f"{stats['total_calls']:,}"),
        ("Unique contacts", stats["unique_contacts"]),
        ("Avg duration", f"{stats['avg_duration_sec']:.0f}s"),
    ])
    pdf.metric_row([
        ("Longest call", f"{stats['longest_call_sec']}s"),
        ("Days covered", stats["days_covered"]),
        ("Late-night calls", stats["late_night_calls"]),
    ])
    pdf.ln(4)
    pdf.why_box(
        why="A statistical summary establishes the baseline scale of the dataset before "
            "diving into deeper behavioral patterns.",
        benefit="Gives an immediate, honest sense of volume and time span, so every "
                "later percentage or score can be interpreted in context.",
        finding=f"The dataset spans {stats['days_covered']} days with {stats['total_calls']:,} calls "
                f"to {stats['unique_contacts']} unique contacts, averaging "
                f"{stats['avg_duration_sec']:.0f} seconds per call.",
    )

    img = _mpl_donut(
        [stats["weekday_calls"], stats["weekend_calls"]], ["Weekday", "Weekend"],
        [BLUE, CORAL], "Weekday vs weekend calls",
    )
    pdf.add_image(img, w=90)

    # ── SECTION 2: TIME PATTERNS ────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2. Time-Based Patterns")

    pdf.sub_title("2.1 Calls by hour of day")
    hourly = A.hourly_distribution(df)
    colors = [CORAL if l else BLUE for l in hourly["is_late"]]
    img = _mpl_bar([f"{h:02d}" for h in hourly["hour"]], hourly["call_count"], colors,
                    "Calls by hour of day", "Hour", "Calls", rot=0)
    pdf.add_image(img)
    peak_hour = int(hourly.loc[hourly["call_count"].idxmax(), "hour"])
    late_night_pct = round(df["is_late"].mean() * 100, 1)
    pdf.why_box(
        why="Calling activity is rarely uniform across the day. Mapping it by hour reveals "
            "when communication actually happens.",
        benefit="Helps separate routine daytime contact (work, family, errands) from "
                "late-night or off-hours communication, which often carries different "
                "significance.",
        finding=f"The busiest hour is {peak_hour:02d}:00, and {late_night_pct}% of all calls "
                f"happen between 10 PM and 6 AM.",
    )

    pdf.sub_title("2.2 Calls by day of week")
    wd = A.weekday_distribution(df)
    colors = [CORAL if w else BLUE for w in wd["is_weekend"]]
    img = _mpl_bar(wd["weekday"], wd["call_count"], colors, "Calls by day of week", "Day", "Calls")
    pdf.add_image(img, w=110)
    busiest_day = wd.loc[wd["call_count"].idxmax(), "weekday"]
    pdf.why_box(
        why="People have natural weekly rhythms. Knowing which day carries the most "
            "activity highlights routine vs irregular behavior.",
        benefit="A sudden change in the busiest day over time can flag a shift in "
                "schedule, routine, or relationships.",
        finding=f"{busiest_day} is the most active day of the week in this dataset.",
    )

    pdf.sub_title("2.3 Daily call volume trend")
    trend = A.daily_trend(df)
    img = _mpl_line(trend["date"], [trend["call_count"], trend["rolling_7d"]],
                     ["Daily calls", "7-day average"], [BLUE, CORAL],
                     "Daily call volume trend", "Date", "Calls")
    pdf.add_image(img)
    peak_row = trend.loc[trend["call_count"].idxmax()]
    pdf.why_box(
        why="Tracking daily volume over the full period shows whether communication "
            "is stable, growing, or spiking around specific dates.",
        benefit="Long unexplained spikes are easy to spot visually and can be "
                "cross-referenced against real-world events or dates of interest.",
        finding=f"The single busiest day was {peak_row['date'].date()} with "
                f"{int(peak_row['call_count'])} calls.",
    )

    # ── SECTION 3: CONTACT ANALYSIS ─────────────────────────────────
    pdf.add_page()
    pdf.section_title("3. Contact Analysis")

    pdf.sub_title("3.1 Top contacts by call frequency")
    top10 = cs.head(10).sort_values("total_calls")
    img = _mpl_hbar(top10["name"], top10["total_calls"], BLUE,
                     "Top 10 contacts by number of calls", "Calls")
    pdf.add_image(img, w=140)
    pdf.body_text(_top3_text(cs.sort_values("total_calls", ascending=False),
                              "name", "total_calls", unit=" calls"))
    pdf.why_box(
        why="Raw call counts immediately show who dominates the communication pattern.",
        benefit="Useful for separating habitual, high-frequency contacts (family, close "
                "friends, work) from occasional or one-off numbers.",
        finding=f"{top_name} is the most frequently called contact, with "
                f"{int(cs.iloc[0]['total_calls'])} total calls.",
    )

    pdf.sub_title("3.2 Relationship tiers (Pareto distribution)")
    tier_counts = rt["tier"].value_counts()
    pdf.body_text(
        f"Inner circle: {tier_counts.get('Inner circle', 0)} contact(s) account for the "
        f"first 50% of total call time. Close contacts: {tier_counts.get('Close contact', 0)}. "
        f"Casual contacts: {tier_counts.get('Casual contact', 0)}."
    )
    pdf.why_box(
        why="Not every contact deserves equal attention - a small number of people "
            "typically account for the majority of call time (the Pareto principle).",
        benefit="Strips away noise from rarely-called numbers and focuses attention on "
                "who genuinely matters in this person's communication life.",
        finding=f"{tier_counts.get('Inner circle', 0)} contact(s) make up the inner circle, "
                f"consuming roughly half of all call time in the dataset.",
    )

    # ── SECTION 4: PATTERN & ANOMALY DETECTION ──────────────────────
    pdf.add_page()
    pdf.section_title("4. Pattern & Anomaly Detection")

    pdf.sub_title("4.1 Frequency spike detection")
    spikes = A.frequency_spikes(df)
    n_spikes = int(spikes["is_spike"].sum())
    top_spike = spikes.head(12)
    colors = [RED if s else BLUE for s in top_spike["is_spike"]]
    img = _mpl_bar(top_spike["name"], top_spike["call_count"], colors,
                    "Frequency spikes (red = statistically unusual)", "", "Calls", rot=70,
                    figsize=(7, 3.6))
    pdf.add_image(img)
    pdf.body_text(_top3_text(spikes.sort_values("call_count", ascending=False),
                              "name", "call_count", unit=" calls"))
    pdf.why_box(
        why="A statistical z-score test flags contacts called far more often than the "
            "typical contact in this dataset.",
        benefit="Quickly separates ordinary contacts from ones receiving unusually "
                "concentrated attention, without manually scanning every row.",
        finding=f"{n_spikes} contact(s) show statistically significant frequency spikes "
                f"(more than 2 standard deviations above average).",
    )

    pdf.sub_title("4.2 Outlier call duration")
    outliers, threshold = A.outlier_calls(df)
    if len(outliers) > 0:
        outliers_min = outliers.assign(duration_min=(outliers["duration"] / 60).round(1))
        pdf.body_text(_top3_text(outliers_min.sort_values("duration_min", ascending=False),
                                  "name", "duration_min", fmt="{:.1f}", unit=" min"))
    pdf.why_box(
        why="A small number of unusually long calls can carry more behavioral signal "
            "than hundreds of short, routine ones.",
        benefit="Surfaces conversations that stand out by length, which often indicates "
                "deeper or more significant exchanges.",
        finding=f"{len(outliers)} call(s) exceed the 99th percentile duration of "
                f"{threshold/60:.1f} minutes. The single longest call lasted "
                f"{df['duration'].max()/60:.1f} minutes.",
    )

    pdf.sub_title("4.3 Behavior change point")
    bcp = A.behavior_change_point(df)
    img = _mpl_line(bcp["daily"]["date_only"],
                     [bcp["daily"]["rolling_calls"], bcp["daily"]["rolling_late"]],
                     ["7-day avg calls", "7-day avg late-night"], [BLUE, CORAL],
                     "Behavior change point over time", "Date", "Calls")
    pdf.add_image(img)
    pdf.why_box(
        why="Communication habits are not static. This analysis finds the date where "
            "behavior shifted most sharply using a rolling-average comparison.",
        benefit="Pinpoints exactly when a routine changed, which can be cross-checked "
                "against real-world events or relationship milestones.",
        finding=f"The largest behavioral shift occurred around {bcp['change_date'].date()}. "
                f"Before: {bcp['before_avg_calls']} calls/day, {bcp['before_contacts']} contacts. "
                f"After: {bcp['after_avg_calls']} calls/day, {bcp['after_contacts']} contacts.",
    )

    pdf.sub_title("4.4 Weekly call rhythm (top contact)")
    wr = A.weekly_rhythm(df, top_name)
    heat = wr["heatmap"]
    step = 2
    cols = list(range(0, 24, step))
    img = _mpl_heatmap(heat.iloc[:, cols].values, [f"{h:02d}" for h in cols], heat.index,
                        f"Weekly rhythm - {top_name}")
    pdf.add_image(img, w=140)
    pdf.why_box(
        why="Plotting one contact's calls by day-of-week and hour reveals whether "
            "contact follows a fixed routine or happens randomly.",
        benefit="A highly concentrated, repeating time slot suggests a scheduled habit; "
                "a scattered pattern suggests casual or reactive contact.",
        finding=f"{top_name} has a routine concentration score of {wr['concentration']}/100 "
                f"({'a clear recurring routine' if wr['is_routine'] else 'no strong fixed routine'}).",
    )

    pdf.sub_title("4.5 Statistical anomaly days")
    ad = A.anomaly_days(df)
    n_anom = int(ad["anomaly"].sum())
    pdf.why_box(
        why="Z-score anomaly detection flags individual days where call count, total "
            "duration, contact variety, or late-night activity deviated sharply from "
            "the dataset's own normal range.",
        benefit="Removes guesswork from spotting 'unusual days' - the flagging is "
                "based on statistics, not a visual guess.",
        finding=f"{n_anom} day(s) out of {len(ad)} were flagged as statistically anomalous.",
    )

    # ── SECTION 5: IMPORTANCE / TRUST SCORE ─────────────────────────
    pdf.add_page()
    pdf.section_title("5. Composite Importance Score")
    pdf.body_text(
        "This score combines every signal above - frequency, duration, late-night "
        "ratio, weekend ratio, growth, recency, outliers, rhythm and loyalty shift - "
        "into a single 0-100 score per contact, so the most behaviorally significant "
        "contacts rise to the top automatically."
    )

    top15 = comp.head(15).sort_values("FINAL_SCORE")
    colors = [TIER_COLORS[t] for t in top15["tier"]]
    img = _mpl_hbar([f"#{r} {n}" for r, n in zip(top15["rank"], top15["name"])],
                     top15["FINAL_SCORE"], colors,
                     "Top 15 contacts - composite importance score", "Score (0-100)",
                     figsize=(7, 5))
    pdf.add_image(img, w=150)
    pdf.body_text(_top3_text(comp, "name", "FINAL_SCORE", fmt="{:.1f}", unit="/100"))

    pdf.why_box(
        why="No single metric tells the whole story - a contact might be called often "
            "but briefly, or rarely but for hours late at night. Combining every "
            "dimension avoids being misled by any one signal.",
        benefit="Produces one clear ranked list instead of nine separate ones, making "
                "it immediately obvious who stands out overall.",
        finding=f"{top_comp['name']} ranks #1 with a composite score of "
                f"{top_comp['FINAL_SCORE']}/100 (tier: {top_comp['tier']}), based on "
                f"{int(top_comp['total_calls'])} calls averaging {top_comp['avg_duration']}s, "
                f"with {top_comp['late_pct']}% occurring at night and "
                f"{top_comp['weekend_pct']}% on weekends.",
    )

    tier_counts = comp["tier"].value_counts()
    pdf.sub_title("Tier distribution across all contacts")
    pdf.body_text(
        f"Critical: {tier_counts.get('Critical', 0)}   |   "
        f"High: {tier_counts.get('High', 0)}   |   "
        f"Medium: {tier_counts.get('Medium', 0)}   |   "
        f"Low: {tier_counts.get('Low', 0)}"
    )

    # ── SECTION 6: FULL DATA TABLE ───────────────────────────────────
    pdf.add_page()
    pdf.section_title("6. Full Contact Ranking Table")
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(37, 99, 235)
    pdf.set_text_color(255, 255, 255)
    headers = ["#", "Name", "Calls", "Avg dur(s)", "Late%", "Wknd%", "Score", "Tier"]
    widths = [10, 55, 18, 22, 16, 16, 18, 25]
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=0, fill=True, align="C")
    pdf.ln(7)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(40, 40, 40)
    for i, row in comp.head(30).iterrows():
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 247, 250)
        vals = [str(int(row["rank"])), str(row["name"])[:32], str(int(row["total_calls"])),
                f"{row['avg_duration']:.0f}", f"{row['late_pct']:.0f}", f"{row['weekend_pct']:.0f}",
                f"{row['FINAL_SCORE']:.1f}", row["tier"]]
        for v, w in zip(vals, widths):
            pdf.cell(w, 6.5, v, border=0, fill=fill, align="C" if v != vals[1] else "L")
        pdf.ln(6.5)

    return bytes(pdf.output())
