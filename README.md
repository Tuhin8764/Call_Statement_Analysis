# Call Statement Analyzer

A Streamlit web application that turns a raw call statement (Sl no., Date, Time,
Ph Number, Call time(sec)) into a full interactive analysis — time patterns,
contact analysis, pattern/anomaly detection, and a composite importance score —
with a one-click downloadable PDF report.

## Features

- **Clean, tabbed UI** — Overview, Time Patterns, Contacts, Pattern Detection,
  Importance Score, and Download Report.
- **Every chart is interactive** (built with Plotly) — hover for exact values,
  zoom/pan, click legend items to toggle series on/off.
- **Name lookup** — upload an optional 2-column contacts file (Name, Contacts)
  and every chart/table will show names instead of raw phone numbers.
- **Explained analysis** — every section has a "Why this analysis" / "Benefit"
  box plus an auto-generated "Key finding" pulled from your actual data.
- **Composite importance score** — combines 9 behavioral signals (frequency,
  duration, late-night ratio, weekend ratio, growth, recency, outliers,
  weekly rhythm, loyalty shift) into one ranked 0–100 score per contact.
- **Downloadable PDF report** — a full multi-page report with charts and
  written findings for every analysis, ready to save or share.
- **Demo data button** — try the whole app instantly without uploading anything.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Expected file formats

**Call records** (CSV or Excel) — 5 columns in this exact order:

| Sl no. | Date | Time | Ph Number | Call time(sec) |
|---|---|---|---|---|
| 1 | 21-01-2026 | 18:57:21 | 1234567890 | 78 |

- Date format can be auto-detected, or you can pick the exact format from the
  sidebar if auto-detect gets it wrong.
- Time can use `:` or `.` as the separator (e.g. `18.57.21` works too).

**Contacts file** (optional, CSV or Excel) — 2 columns in this order:

| Name | Contacts |
|---|---|
| John Doe | 1234567890 |

Phone numbers are matched on their last 10 digits, so formatting differences
(spaces, dashes, +91 prefixes) are handled automatically.

## Project structure

```
app.py          → Streamlit UI (tabs, charts, explanations, report button)
analysis.py     → All data loading/cleaning/analysis logic (no UI code)
charts.py       → Interactive Plotly chart builders
report.py       → PDF report generator (matplotlib + fpdf2)
requirements.txt
```

## Notes

- The PDF report regenerates fresh each time you click "Generate PDF report",
  reflecting whatever file is currently loaded.
