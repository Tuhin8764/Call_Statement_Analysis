"""
charts.py
Builds interactive Plotly figures from the dataframes produced by analysis.py.
Every figure returned here is hoverable, zoomable, and pannable in Streamlit,
shows its data labels directly on the chart (not just on hover), and legend
items are clickable to toggle series on/off.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#2563EB"
CORAL = "#E85D30"
RED = "#DC2626"
ORANGE = "#EA580C"
AMBER = "#D97706"
GREEN = "#16A34A"
GRAY = "#94A3B8"

TIER_COLORS = {"Critical": RED, "High": ORANGE, "Medium": AMBER, "Low": BLUE}
MEDALS = ["🥇", "🥈", "🥉"]

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    margin=dict(l=70, r=40, t=70, b=60),
    font=dict(size=13),
    hovermode="closest",
)


def _apply_layout(fig, title=None, height=400):
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, pad=dict(b=12)))
    return fig


def _medal_labels(names, n=3):
    """Prefix the top-n entries with a medal emoji so winners are visible on the chart itself."""
    out = []
    for i, n_ in enumerate(names):
        out.append(f"{MEDALS[i]} {n_}" if i < n else str(n_))
    return out


# ──────────────────────────────────────────────────────────────────────
# Overview
# ──────────────────────────────────────────────────────────────────────

def weekday_weekend_donut(weekday_calls: int, weekend_calls: int):
    total = max(weekday_calls + weekend_calls, 1)
    fig = go.Figure(data=[go.Pie(
        labels=["Weekday", "Weekend"],
        values=[weekday_calls, weekend_calls],
        hole=0.55,
        marker=dict(colors=[BLUE, CORAL], line=dict(color="white", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{label}: %{value} calls (%{percent})<extra></extra>",
    )])
    fig.add_annotation(text=f"<b>{total}</b><br>calls", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=14))
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, "Weekday vs weekend calls", height=400)


# ──────────────────────────────────────────────────────────────────────
# Time patterns
# ──────────────────────────────────────────────────────────────────────

def hourly_bar(hourly_df: pd.DataFrame):
    colors = [CORAL if late else BLUE for late in hourly_df["is_late"]]
    fig = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in hourly_df["hour"]],
        y=hourly_df["call_count"],
        marker_color=colors,
        text=hourly_df["call_count"],
        textposition="outside",
        textfont=dict(size=10),
        cliponaxis=False,
        customdata=hourly_df["avg_duration"].round(0),
        hovertemplate="Hour %{x}<br>Calls: %{y}<br>Avg duration: %{customdata}s<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Number of calls")
    fig.update_yaxes(range=[0, hourly_df["call_count"].max() * 1.22])
    return _apply_layout(fig, "Calls by hour of day (late-night hours in coral)")


def weekday_bar(weekday_df: pd.DataFrame):
    colors = [CORAL if w else BLUE for w in weekday_df["is_weekend"]]
    fig = go.Figure(go.Bar(
        x=weekday_df["weekday"], y=weekday_df["call_count"],
        marker_color=colors,
        text=weekday_df["call_count"], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Calls: %{y}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Day of week", yaxis_title="Number of calls")
    fig.update_yaxes(range=[0, weekday_df["call_count"].max() * 1.22])
    return _apply_layout(fig, "Calls by day of week", height=380)


def time_slot_bar(slot_df: pd.DataFrame):
    colors = [BLUE, GREEN, AMBER, CORAL]
    fig = go.Figure(go.Bar(
        x=slot_df["time_slot"], y=slot_df["call_count"],
        marker_color=colors[: len(slot_df)],
        text=slot_df["call_count"], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Calls: %{y}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Time slot", yaxis_title="Number of calls")
    fig.update_yaxes(range=[0, slot_df["call_count"].max() * 1.22])
    return _apply_layout(fig, "Calls by time slot", height=380)


def daily_trend_line(trend_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["date"], y=trend_df["call_count"], mode="lines",
        line=dict(color=BLUE, width=1), opacity=0.5, name="Daily calls",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["date"], y=trend_df["rolling_7d"], mode="lines",
        line=dict(color=CORAL, width=2.5), name="7-day average",
    ))
    peak = trend_df.loc[trend_df["call_count"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[peak["date"]], y=[peak["call_count"]], mode="markers+text",
        marker=dict(color=RED, size=12, symbol="star"),
        text=[f"Peak: {int(peak['call_count'])} calls<br>{peak['date'].date()}"],
        textposition="top center", textfont=dict(size=10, color=RED),
        name="Peak day",
        hovertemplate=f"Peak day<br>{peak['date'].date()}<br>%{{y}} calls<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of calls",
                       legend=dict(orientation="h", y=1.15))
    return _apply_layout(fig, "Daily call volume trend", height=440)


# ──────────────────────────────────────────────────────────────────────
# Contact analysis
# ──────────────────────────────────────────────────────────────────────

def top_contacts_bar(cs_df: pd.DataFrame, top_n=15, metric="total_calls"):
    top = cs_df.head(top_n).sort_values(metric)
    label = "Total calls" if metric == "total_calls" else "Total duration (sec)"
    # rank order is ascending in the plot (bottom=lowest); medals belong to the
    # 3 largest values, i.e. the last 3 rows of `top`
    y_labels = list(top["name"])
    for i, idx in enumerate(reversed(top.index[-3:])):
        pos = top.index.get_loc(idx)
        y_labels[pos] = f"{MEDALS[i]} {y_labels[pos]}"

    fig = go.Figure(go.Bar(
        x=top[metric], y=y_labels, orientation="h",
        marker_color=BLUE,
        text=top[metric], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>" + label + ": %{x}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=label, yaxis_title="")
    fig.update_xaxes(range=[0, top[metric].max() * 1.18])
    return _apply_layout(fig, f"Top {top_n} contacts by {label.lower()}",
                          height=max(380, 30 * len(top) + 110))


def relationship_bubble(rt_df: pd.DataFrame, top_n=25):
    top = rt_df.head(top_n)
    tier_colors = {"Inner circle": RED, "Close contact": AMBER, "Casual contact": BLUE}
    top3_names = set(top.sort_values("total_duration", ascending=False).head(3)["name"])

    fig = go.Figure()
    for tier_name, color in tier_colors.items():
        sub = top[top["tier"] == tier_name]
        if sub.empty:
            continue
        labels = [n if n in top3_names else "" for n in sub["name"]]
        fig.add_trace(go.Scatter(
            x=sub["total_calls"], y=sub["avg_duration"], mode="markers+text",
            marker=dict(size=np.clip(sub["total_duration"] / sub["total_duration"].max() * 50, 8, 50),
                        color=color, opacity=0.75, line=dict(color="white", width=1)),
            name=tier_name, text=labels, textposition="top center", textfont=dict(size=10),
            customdata=sub["name"],
            hovertemplate="%{customdata}<br>Calls: %{x}<br>Avg duration: %{y:.0f}s<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Number of calls", yaxis_title="Average duration (sec)",
                       legend=dict(orientation="h", y=1.12))
    return _apply_layout(fig, "Relationship map — who really matters (top 3 labeled)", height=460)


def communication_dna_radar(contact_dna: dict, baseline_dna: dict, contact_label: str):
    categories = list(contact_dna.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(baseline_dna.values()) + [list(baseline_dna.values())[0]],
        theta=categories + [categories[0]],
        fill="toself", name="Dataset average",
        line=dict(color=GRAY, dash="dash"), opacity=0.5,
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(contact_dna.values()) + [list(contact_dna.values())[0]],
        theta=categories + [categories[0]],
        fill="toself", name=contact_label,
        line=dict(color=RED, width=2),
        text=[f"{v:.0f}" for v in contact_dna.values()] + [f"{list(contact_dna.values())[0]:.0f}"],
        mode="lines+markers+text", textposition="top center", textfont=dict(size=9, color=RED),
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 105])),
                       legend=dict(orientation="h", y=1.18))
    return _apply_layout(fig, f"Communication DNA — {contact_label} vs dataset average", height=460)


# ──────────────────────────────────────────────────────────────────────
# Pattern / anomaly detection
# ──────────────────────────────────────────────────────────────────────

def frequency_spike_chart(spike_df: pd.DataFrame, top_n=20):
    top = spike_df.head(top_n)
    colors = [RED if s else BLUE for s in top["is_spike"]]
    x_labels = _medal_labels(top["name"], n=min(3, len(top)))
    fig = go.Figure(go.Bar(
        x=x_labels, y=top["call_count"], marker_color=colors,
        text=top["call_count"], textposition="outside", cliponaxis=False,
        customdata=top["z_score"],
        hovertemplate="%{x}<br>Calls: %{y}<br>Z-score: %{customdata:.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="", yaxis_title="Number of calls", xaxis_tickangle=-40)
    fig.update_yaxes(range=[0, top["call_count"].max() * 1.2])
    return _apply_layout(fig, "Frequency spikes (red = statistically unusual, top 3 medaled)", height=460)


def outlier_scatter(df: pd.DataFrame, outliers: pd.DataFrame, threshold: float):
    normal = df[df["duration"] <= threshold]
    top3 = outliers.nlargest(min(3, len(outliers)), "duration")
    rest = outliers.drop(top3.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["duration"] / 60, mode="markers",
        marker=dict(color=BLUE, size=5, opacity=0.3), name="Normal calls",
        text=normal["name"],
        hovertemplate="%{text}<br>%{x}<br>%{y:.1f} min<extra></extra>",
    ))
    if len(rest):
        fig.add_trace(go.Scatter(
            x=rest["date"], y=rest["duration"] / 60, mode="markers",
            marker=dict(color=CORAL, size=10, line=dict(color="white", width=1)),
            name="Outlier calls (top 1%)", text=rest["name"],
            hovertemplate="%{text}<br>%{x}<br>%{y:.1f} min<extra></extra>",
        ))
    if len(top3):
        labels = [f"{MEDALS[i]} {n} ({d/60:.0f}m)" for i, (n, d) in
                  enumerate(zip(top3["name"], top3["duration"]))]
        fig.add_trace(go.Scatter(
            x=top3["date"], y=top3["duration"] / 60, mode="markers+text",
            marker=dict(color=RED, size=13, line=dict(color="white", width=1.5)),
            text=labels, textposition="top center", textfont=dict(size=10, color=RED),
            name="Top 3 longest calls",
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.add_hline(y=threshold / 60, line_dash="dash", line_color=RED,
                  annotation_text=f"99th pct ({threshold/60:.1f} min)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Duration (min)",
                       legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(range=[0, max(outliers["duration"].max() / 60 * 1.18, threshold / 60 * 1.3)])
    return _apply_layout(fig, "Outlier calls — unusually long duration (top 3 labeled)", height=480)


def call_gap_line(gap_info: dict, name: str):
    tl = gap_info["timeline"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tl["date"], y=tl["gap_days"], mode="lines+markers+text",
        line=dict(color=AMBER), marker=dict(size=7),
        text=tl["gap_days"].round(1), textposition="top center", textfont=dict(size=9),
        hovertemplate="%{x}<br>Gap: %{y:.1f} days<extra></extra>",
    ))
    fig.add_hline(y=gap_info["avg_gap"], line_dash="dash", line_color=GRAY,
                  annotation_text=f"Average gap: {gap_info['avg_gap']} days")
    fig.update_layout(xaxis_title="Call date", yaxis_title="Days since previous call")
    fig.update_yaxes(range=[0, tl["gap_days"].max() * 1.25])
    return _apply_layout(fig, f"Days between calls — {name}", height=420)


def change_point_chart(bcp: dict):
    daily = bcp["daily"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date_only"], y=daily["rolling_calls"], mode="lines",
        line=dict(color=BLUE, width=2), name="7-day avg calls",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date_only"], y=daily["rolling_late"], mode="lines",
        line=dict(color=CORAL, width=2), name="7-day avg late-night calls",
    ))
    fig.add_vline(x=bcp["change_date"], line_dash="dash", line_color="gray",
                  annotation_text=f"Change point: {bcp['change_date'].date()}",
                  annotation_font=dict(size=11))
    fig.update_layout(xaxis_title="Date", yaxis_title="Calls (7-day average)",
                       legend=dict(orientation="h", y=1.15))
    return _apply_layout(fig, "Behavior change point over time", height=440)


def weekly_rhythm_heatmap(heat_df: pd.DataFrame, name: str):
    z = heat_df.values
    text = np.where(z == 0, "", z.astype(int).astype(str))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in heat_df.columns], y=heat_df.index,
        colorscale="YlOrRd",
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="black"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Calls: %{z}<extra></extra>",
        colorbar=dict(title="Calls"),
    ))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Day of week")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return _apply_layout(fig, f"Weekly call rhythm — {name} (numbers = call count)", height=420)


def anomaly_line(anomaly_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomaly_df["date_only"], y=anomaly_df["calls"], mode="lines",
        line=dict(color=BLUE, width=1.3), name="Daily calls",
    ))
    anom = anomaly_df[anomaly_df["anomaly"]]
    if len(anom):
        labels = [f"{d.strftime('%d %b')} ({c})" for d, c in zip(anom["date_only"], anom["calls"])]
        fig.add_trace(go.Scatter(
            x=anom["date_only"], y=anom["calls"], mode="markers+text",
            marker=dict(color=RED, size=10, symbol="x"), name="Anomalous day",
            text=labels, textposition="top center", textfont=dict(size=9, color=RED),
            hovertemplate="%{x}<br>Calls: %{y}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of calls",
                       legend=dict(orientation="h", y=1.15))
    return _apply_layout(fig, "Daily call volume — statistically anomalous days flagged", height=440)


def loyalty_shift_chart(monthly_pct: pd.DataFrame):
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for i, col in enumerate(monthly_pct.columns):
        fig.add_trace(go.Scatter(
            x=[str(m) for m in monthly_pct.index], y=monthly_pct[col],
            mode="lines+markers+text", name=str(col),
            text=monthly_pct[col].round(0).astype(int).astype(str) + "%",
            textposition="top center", textfont=dict(size=9),
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(xaxis_title="Month", yaxis_title="Share of monthly calls (%)",
                       legend=dict(orientation="h", y=1.18))
    fig.update_yaxes(range=[0, 108])
    return _apply_layout(fig, "Contact share of calls per month — loyalty shift", height=460)


# ──────────────────────────────────────────────────────────────────────
# Composite importance score dashboard
# ──────────────────────────────────────────────────────────────────────

def importance_rank_bar(comp_df: pd.DataFrame, top_n=15):
    top = comp_df.head(top_n).sort_values("FINAL_SCORE")
    colors = [TIER_COLORS[t] for t in top["tier"]]
    y_labels = [f"#{r} {n}" for r, n in zip(top["rank"], top["name"])]
    for i, idx in enumerate(reversed(top.index[-3:])):
        pos = top.index.get_loc(idx)
        y_labels[pos] = f"{MEDALS[i]} {y_labels[pos]}"

    fig = go.Figure(go.Bar(
        x=top["FINAL_SCORE"], y=y_labels,
        orientation="h", marker_color=colors,
        text=top["FINAL_SCORE"], textposition="outside", cliponaxis=False,
        customdata=top["tier"],
        hovertemplate="%{y}<br>Score: %{x}<br>Tier: %{customdata}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Composite importance score (0-100)", yaxis_title="",
                       xaxis_range=[0, 112])
    return _apply_layout(fig, f"Top {top_n} contacts — composite importance score",
                          height=max(400, 30 * len(top) + 120))


def importance_radar(top_row: pd.Series, score_cols: list):
    labels = ["Frequency", "Duration", "Late night", "Weekend",
              "Growth", "Recency", "Outlier", "Rhythm", "Loyalty"]
    values = [top_row[c] for c in score_cols]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself", line=dict(color=RED, width=2),
        text=[f"{v:.0f}" for v in values] + [f"{values[0]:.0f}"],
        mode="lines+markers+text", textposition="top center", textfont=dict(size=9, color=RED),
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 105])))
    return _apply_layout(fig, f"Score breakdown — #{int(top_row['rank'])} {top_row['name']}", height=420)


def importance_heatmap(comp_df: pd.DataFrame, score_cols: list, top_n=10):
    top = comp_df.head(top_n)
    short_labels = ["Freq", "Dur", "Late", "Wknd", "Grow", "Recn", "Outl", "Rhyt", "Loya"]
    z = top[score_cols].values
    y_labels = [f"#{r} {n}" for r, n in zip(top["rank"], top["name"])]
    for i, idx in enumerate(top.index[:3]):
        pos = top.index.get_loc(idx)
        y_labels[pos] = f"{MEDALS[i]} {y_labels[pos]}"

    fig = go.Figure(go.Heatmap(
        z=z, x=short_labels, y=y_labels, colorscale="RdYlGn", zmin=0, zmax=100,
        text=z.round(0), texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y}<br>%{x}: %{z:.0f}<extra></extra>",
        colorbar=dict(title="Score"),
    ))
    fig.update_layout(xaxis_title="", yaxis_title="")
    return _apply_layout(fig, f"Sub-score breakdown — top {top_n} contacts",
                          height=max(400, 34 * len(top) + 120))


def tier_donut(comp_df: pd.DataFrame):
    order = ["Critical", "High", "Medium", "Low"]
    counts = comp_df["tier"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker=dict(colors=[TIER_COLORS[t] for t in order], line=dict(color="white", width=2)),
        textinfo="label+value", textposition="outside",
    ))
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, "Contacts by importance tier", height=400)
