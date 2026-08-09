"""
04_generate_report.py
----------------------
Assembles the final PDF case-study report (Vahan_Case_Study_Report.pdf)
from the cohort SQL output, the trained model metrics, and the charts
produced by the earlier scripts.

Run:
    python 04_generate_report.py
"""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
SQL_DIR = ROOT / "sql"
REPORT_PATH = OUT_DIR / "Vahan_Case_Study_Report.pdf"

NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
LIGHT_BLUE = colors.HexColor("#eff6ff")
GRAY = colors.HexColor("#475569")
GREEN = colors.HexColor("#16a34a")
RED = colors.HexColor("#dc2626")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
cohort_df = pd.read_csv(OUT_DIR / "cohort_aggregated_output.csv")
with open(OUT_DIR / "model_metrics.json") as f:
    metrics = json.load(f)
feat_df = pd.read_csv(OUT_DIR / "feature_importance.csv")
sql_text = (SQL_DIR / "02_cohort_analysis.sql").read_text()

total_leads = int(cohort_df["uploaded_leads"].sum())
total_ft = int(cohort_df["ft_after_upload"].sum())
overall_ft_rate = total_ft / metrics["n_total"] * 100  # use full dataset denominator

top3 = cohort_df.sort_values("ft_rate_per_uploaded_pct", ascending=False).head(3)

# ------------------------------------------------------------------
# Styles
# ------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, textColor=NAVY,
                           spaceAfter=4, spaceBefore=0, leading=22))
styles.add(ParagraphStyle("H1Sub", parent=styles["Normal"], fontSize=10.5, textColor=BLUE,
                           spaceAfter=14, leading=13))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, textColor=NAVY,
                           spaceBefore=14, spaceAfter=6, leading=16))
styles.add(ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, textColor=BLUE,
                           spaceBefore=10, spaceAfter=4, leading=13))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#1e293b"),
                           leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle("BulletBody", parent=styles["Body"], leftIndent=14, bulletIndent=4, spaceAfter=4))
styles.add(ParagraphStyle("Small", parent=styles["Normal"], fontSize=7.8, textColor=GRAY, leading=10))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=GRAY,
                           leading=10, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("CodeBlock", parent=styles["Code"], fontSize=6.9, leading=9,
                           backColor=colors.HexColor("#0f172a"), textColor=colors.HexColor("#e2e8f0"),
                           leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=4, borderPadding=6))
styles.add(ParagraphStyle("MetricLabel", parent=styles["Normal"], fontSize=8.5, textColor=GRAY,
                           alignment=TA_CENTER, leading=10))
styles.add(ParagraphStyle("MetricValue", parent=styles["Normal"], fontSize=17, textColor=NAVY,
                           alignment=TA_CENTER, leading=20, fontName="Helvetica-Bold"))

PAGE_W, PAGE_H = LETTER
CONTENT_W = PAGE_W - 1.3 * inch


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
    canvas.line(0.65 * inch, 0.6 * inch, PAGE_W - 0.65 * inch, 0.6 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(0.65 * inch, 0.42 * inch, "Vahan | Lead-Source Cohort Performance — Case Study")
    canvas.drawRightString(PAGE_W - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def section_title(title, subtitle=None):
    els = [Paragraph(title, styles["H1"])]
    if subtitle:
        els.append(Paragraph(subtitle, styles["H1Sub"]))
    els.append(HRFlowable(width=CONTENT_W, thickness=1.4, color=BLUE, spaceAfter=12))
    return els


def metric_box(label, value, color=NAVY, width=1.55 * inch):
    t = Table([[Paragraph(value, ParagraphStyle("v", parent=styles["MetricValue"], textColor=color))],
               [Paragraph(label, styles["MetricLabel"])]],
              colWidths=[width])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def df_to_table(df, col_widths, header_bg=NAVY, font_size=7.6, align_right_from=1, pad=4):
    data = [list(df.columns)] + df.values.tolist()
    data = [[Paragraph(f"<b>{c}</b>", ParagraphStyle("th", parent=styles["Small"], fontSize=font_size,
                                                       textColor=colors.white)) for c in data[0]]] + \
            [[Paragraph(str(c), ParagraphStyle("td", parent=styles["Small"], fontSize=font_size,
                                                textColor=colors.HexColor("#1e293b"))) for c in row]
             for row in data[1:]]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


story = []

# ============================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ============================================================
story += section_title("Vahan — Lead-Source Cohort Performance",
                        "Product Analytics Case Study &nbsp;|&nbsp; Executive Summary")

story.append(Paragraph(
    f"Vahan sources candidate leads from 16 distinct lead-source cohorts and calls them to drive "
    f"conversion to <b>First Trip (FT)</b> — the point at which a candidate actually completes their first "
    f"ride/trip after onboarding. Across <b>{metrics['n_total']:,}</b> leads uploaded between "
    f"18 Jul and 6 Aug 2026, <b>{total_ft}</b> converted to FT, an overall conversion rate of "
    f"<b>{overall_ft_rate:.3f}%</b>. FT is a rare event, so the cohort and modeling approach in this "
    f"report are built specifically to make rare-event comparisons fair and rare-event prediction usable.",
    styles["Body"]))

story.append(Spacer(1, 6))

metrics_row = Table([[
    metric_box("Total Leads", f"{metrics['n_total']:,}"),
    metric_box("Total FT", f"{total_ft}"),
    metric_box("Overall FT Rate", f"{overall_ft_rate:.3f}%", color=BLUE),
    metric_box("Cohorts Analyzed", f"{len(cohort_df)}"),
]], colWidths=[1.5 * inch] * 4, hAlign="LEFT")
metrics_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
story.append(metrics_row)
story.append(Spacer(1, 14))

story.append(Paragraph("Top 3 Performing Cohorts (by FT rate on uploaded leads)", styles["H2"]))
top3_display = top3[["lead_source", "uploaded_leads", "ft_after_upload", "ft_rate_per_uploaded_pct"]].copy()
top3_display.columns = ["Lead Source (Cohort)", "Uploaded Leads", "FT Count", "FT Rate %"]
top3_display["FT Rate %"] = top3_display["FT Rate %"].map(lambda x: f"{x:.3f}%")
story.append(df_to_table(top3_display, col_widths=[2.9 * inch, 1.15 * inch, 0.9 * inch, 1.05 * inch],
                          header_bg=BLUE, font_size=8.3))
story.append(Spacer(1, 14))

story.append(Paragraph("Key Findings", styles["H2"]))
findings = [
    "<b>Warm, human-vetted sources dominate.</b> The top 2 cohorts — a &gt;7-day-old single-referral "
    "batch (0.93% FT) and a re-targeted 2W (two-wheeler) Khanna batch (0.91% FT) — convert 3–12&times; "
    "better than paid/marketplace sources like OLX (0.08% FT), despite OLX supplying the single largest "
    "volume of leads (5,182).",
    "<b>Most drop-off happens before a conversation even starts.</b> Only ~45–55% of attempted calls "
    "actually connect across every cohort, and several large cohorts (e.g. OLX) are barely attempted at "
    "all (18.6%) — dialing discipline, not just lead quality, is limiting conversion.",
    "<b>FT is an extreme rare event (0.30% overall).</b> Any comparison across cohorts or any model built "
    "on this data must be evaluated with imbalance-aware metrics (PR-AUC, recall) rather than accuracy alone.",
    "<b>Whether a candidate was ever attempted, and which cohort they came from, are the strongest "
    "signals of FT</b> in the predictive model — more so than whether the call actually connected, "
    "suggesting outreach persistence and source quality matter more than a single successful call.",
]
for f in findings:
    story.append(Paragraph(f"&bull;&nbsp; {f}", styles["BulletBody"]))

story.append(PageBreak())

# ============================================================
# PAGE 2 — COHORT ANALYSIS
# ============================================================
story += section_title("Cohort Analysis", "Which lead sources perform best, and how we measured it")

story.append(Paragraph("Methodology", styles["H2"]))
story.append(Paragraph(
    "Raw data is one row per uploaded lead (candidate_phone &times; upload_date), tagged with its "
    "lead_source. This is exactly the grain the case study defines as a \"cohort,\" so leads were "
    "grouped by <b>lead_source</b> and rolled up with SQL (see Page 3) into one row per cohort, carrying "
    "forward funnel counts (Attempted, Connected, Interested, Onboarded, FT) and rate metrics. Cohorts "
    "with fewer than 20 uploaded leads were excluded from ranking as too small to compare reliably.",
    styles["Body"]))

story.append(Paragraph("Cohort Performance Table", styles["H2"]))
tbl_df = cohort_df.sort_values("ft_rate_per_uploaded_pct", ascending=False).copy()
tbl_df = tbl_df[["lead_source", "uploaded_leads", "attempted_pct", "attempt_to_connect_pct",
                  "ft_after_upload", "ft_rate_per_uploaded_pct", "ft_rate_per_attempted_pct"]]
tbl_df.columns = ["Lead Source", "Leads", "Attempt %", "Connect %", "FT #", "FT/Upload %", "FT/Attempt %"]
for c in ["Attempt %", "Connect %"]:
    tbl_df[c] = tbl_df[c].map(lambda x: f"{x:.1f}")
for c in ["FT/Upload %", "FT/Attempt %"]:
    tbl_df[c] = tbl_df[c].map(lambda x: f"{x:.3f}")
story.append(df_to_table(tbl_df, col_widths=[2.05 * inch, 0.62 * inch, 0.68 * inch, 0.68 * inch,
                                              0.5 * inch, 0.78 * inch, 0.78 * inch], font_size=7.0, pad=2.0))
story.append(Spacer(1, 6))

story.append(Image(str(CHART_DIR / "cohort_ft_rate.png"), width=5.6 * inch, height=3.13 * inch))
story.append(Paragraph("Figure 1. FT conversion rate by lead-source cohort (% of leads uploaded). "
                        "Top-3 cohorts highlighted in dark blue.", styles["Caption"]))

story.append(KeepTogether([
    Paragraph("Why FT-per-Uploaded-Lead is the Right Metric", styles["H2"]),
    Paragraph(
        "FT count on its own is biased by cohort size, and FT-per-attempted-lead rewards cohorts that are "
        "under-dialed (a cohort that is barely called can post an inflated rate off very few attempts — see "
        "Khanna AI, only 27.7% attempted). <b>FT per uploaded lead</b> holds the denominator constant "
        "(total leads sourced) across every cohort, so it fairly reflects true end-to-end source quality "
        "rather than calling effort, and it is the metric decision-makers actually care about: given 1,000 "
        "leads from source X, how many will become drivers?",
        styles["Body"]),
]))


# ============================================================
# PAGE 3 — SQL ANALYSIS
# ============================================================
story += section_title("SQL Analysis", "Cohort aggregation query and output")

story.append(Paragraph("Query", styles["H2"]))
story.append(Paragraph(
    "The raw lead-level table (<font face='Courier'>raw_leads</font>) is aggregated to one row per "
    "<font face='Courier'>lead_source</font>, computing funnel volumes and conversion rates "
    "(fully-commented version in <font face='Courier'>sql/02_cohort_analysis.sql</font>):",
    styles["Body"]))

sql_display_text = """SELECT
    lead_source,
    SUM(uploaded_leads)                                            AS uploaded_leads,
    SUM(attempted)                                                 AS attempted,
    SUM(connected)                                                 AS connected,
    SUM(interested)                                                AS interested,
    SUM(ob_after_upload)                                           AS onboarded,
    SUM(ft_after_upload)                                           AS ft_after_upload,
    SUM(ft_after_first_attempt)                                    AS ft_after_first_attempt,
    ROUND(100.0 * SUM(attempted)  / NULLIF(SUM(uploaded_leads),0), 2) AS attempted_pct,
    ROUND(100.0 * SUM(connected)  / NULLIF(SUM(attempted),0),      2) AS attempt_to_connect_pct,
    ROUND(100.0 * SUM(interested) / NULLIF(SUM(connected),0),      2) AS connect_to_interested_pct,
    -- primary ranking metric: overall FT conversion on total leads uploaded
    ROUND(100.0 * SUM(ft_after_upload) / NULLIF(SUM(uploaded_leads),0), 3) AS ft_rate_per_uploaded_pct,
    -- secondary metric: FT conversion on leads actually worked (attempted)
    ROUND(100.0 * SUM(ft_after_upload) / NULLIF(SUM(attempted),0), 3)      AS ft_rate_per_attempted_pct,
    ROUND(AVG(upload_to_first_attempt_p50_hrs), 1)                 AS avg_hrs_to_first_attempt
FROM raw_leads
GROUP BY lead_source
HAVING SUM(uploaded_leads) >= 20   -- drop micro cohorts too small to rank reliably
ORDER BY ft_rate_per_uploaded_pct DESC;"""
story.append(Paragraph(sql_display_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["CodeBlock"]))

story.append(Paragraph("Aggregated Output (Top 6 Cohorts)", styles["H2"]))
out_df = cohort_df.sort_values("ft_rate_per_uploaded_pct", ascending=False).head(6).copy()
out_df = out_df[["lead_source", "uploaded_leads", "attempted", "connected", "interested",
                  "onboarded", "ft_after_upload", "ft_rate_per_uploaded_pct"]]
out_df.columns = ["lead_source", "uploaded", "attempted", "connected", "interested", "onboarded", "ft", "ft_rate_%"]
out_df["ft_rate_%"] = out_df["ft_rate_%"].map(lambda x: f"{x:.3f}")
story.append(df_to_table(out_df, col_widths=[1.75 * inch, 0.62 * inch, 0.68 * inch, 0.68 * inch,
                                              0.68 * inch, 0.68 * inch, 0.45 * inch, 0.65 * inch], font_size=7.3, pad=2.6))
story.append(Spacer(1, 6))

story.append(Paragraph("Explanation", styles["H2"]))
story.append(Paragraph(
    "The query filters out cohorts with fewer than 20 uploaded leads (<font face='Courier'>HAVING</font> "
    "clause) so single-digit-lead batches don't produce noisy 0%/100% rates. It reports both the primary "
    "ranking metric (FT per uploaded lead) and a secondary operational metric (FT per attempted lead), "
    "plus median hours-to-first-attempt to flag cohorts where slow follow-up may be suppressing "
    "conversion (e.g. the 2W3W-Khanna cohorts averaged 125–175 hours to first attempt with zero FTs).",
    styles["Body"]))

story.append(PageBreak())

# ============================================================
# PAGE 4 — ML APPROACH
# ============================================================
story += section_title("Machine Learning Approach", "Predicting First-Trip conversion")

story.append(Paragraph("Target Variable", styles["H2"]))
story.append(Paragraph(
    "<font face='Courier'><b>FT_after_upload</b></font> (binary): 1 if the candidate completed their "
    f"First Trip at any point after being uploaded, 0 otherwise. Positive rate in the full dataset is "
    f"{metrics['positive_rate_pct']:.3f}% ({metrics['n_positive']} of {metrics['n_total']:,} leads) — "
    "a substantial class imbalance that shapes every modeling choice below.",
    styles["Body"]))

story.append(Paragraph("Features Used", styles["H2"]))
feat_list = [
    "<b>lead_source_grp</b> — cohort identity (top cohorts kept distinct, rare ones grouped as \"Other\")",
    "<b>attempted, connected, tag_filled, interested</b> — funnel-stage flags, each causally prior to FT",
    "<b>attempt_per_lead</b> — number of call attempts made on the lead",
    "<b>upload_to_first_attempt_p50_hrs</b> — speed of follow-up after upload",
    "<b>upload_dow</b> — day of week the lead was uploaded (captures batch/operational effects)",
]
for f in feat_list:
    story.append(Paragraph(f"&bull;&nbsp; {f}", styles["BulletBody"]))
story.append(Paragraph(
    "<b>Deliberately excluded:</b> OB_after_upload, OB_after_first_attempt, FT_after_first_attempt, and "
    "all pre-computed *_pct ratio columns. OB (onboarding) is the funnel step immediately before FT and "
    "is a near-perfect proxy for it (see Recommendations & Limitations, Target Leakage) — including it would make the model "
    "trivially \"accurate\" while learning nothing actionable.",
    styles["Body"]))

story.append(Paragraph("Train / Test Split", styles["H2"]))
story.append(Paragraph(
    f"A stratified 75/25 split was used to preserve the rare positive class in both sets: "
    f"<b>{metrics['n_train']:,} training rows</b> (40 FT positives) and "
    f"<b>{metrics['n_test']:,} test rows</b> (14 FT positives). Stratification is essential here — "
    "a random split without it risks a test set with zero or very few positives, making evaluation "
    "metrics meaningless.",
    styles["Body"]))

story.append(Paragraph("Model: Logistic Regression with Class Weighting", styles["H2"]))
story.append(Paragraph(
    "A logistic regression classifier was trained with <font face='Courier'>class_weight='balanced'</font>, "
    "which re-weights the loss function inversely proportional to class frequency. This penalizes "
    "misclassifying the rare FT=1 class much more heavily than the abundant FT=0 class, pushing the "
    "decision boundary to actually try to catch positives rather than trivially predicting \"no FT\" "
    "for everyone (which would already be 99.7% \"accurate\" and completely useless).",
    styles["Body"]))

story.append(Paragraph("Why This Model", styles["H2"]))
why_list = [
    "<b>Interpretability</b> — the case study explicitly asks for \"factors that influence FT\"; logistic "
    "regression coefficients map directly to odds ratios per feature, which stakeholders can act on.",
    "<b>Appropriate for very few positives</b> — with only 54 positive examples in the whole dataset, "
    "complex models (gradient boosting, random forests) risk overfitting noise; a regularized linear "
    "model generalizes more reliably at this sample size.",
    "<b>Native class-imbalance handling</b> — class weighting avoids the need to synthetically oversample "
    "(e.g. SMOTE) a target with only 54 real examples, which risks fabricating unrealistic data.",
    "<b>Fast, auditable, reproducible</b> — no random-seed-sensitive ensembling; results are stable and "
    "easy to explain to a non-technical audience.",
]
for f in why_list:
    story.append(Paragraph(f"&bull;&nbsp; {f}", styles["BulletBody"]))

story.append(PageBreak())

# ============================================================
# PAGE 5 — MODEL RESULTS
# ============================================================
story += section_title("Model Results", "Performance on the held-out test set")

img_row = Table([[
    Image(str(CHART_DIR / "confusion_matrix.png"), width=2.55 * inch, height=2.5 * inch),
    Image(str(CHART_DIR / "roc_curve.png"), width=2.55 * inch, height=2.5 * inch),
]], colWidths=[3.15 * inch, 3.15 * inch])
img_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(img_row)
story.append(Paragraph("Figure 2. Confusion matrix (left) and ROC curve (right) on the test set "
                        f"(n={metrics['n_test']:,}).", styles["Caption"]))

metrics_row2 = Table([[
    metric_box("Precision", f"{metrics['precision']:.3f}", color=RED),
    metric_box("Recall", f"{metrics['recall']:.3f}", color=GREEN),
    metric_box("F1 Score", f"{metrics['f1_score']:.3f}"),
    metric_box("ROC-AUC", f"{metrics['roc_auc']:.3f}", color=BLUE),
    metric_box("PR-AUC", f"{metrics['pr_auc']:.3f}", color=BLUE, width=1.3 * inch),
]], colWidths=[1.28 * inch] * 4 + [1.3 * inch], hAlign="LEFT")
story.append(metrics_row2)
story.append(Spacer(1, 8))

story.append(Paragraph(
    f"The model correctly identifies <b>{metrics['confusion_matrix'][1][1]} of "
    f"{metrics['confusion_matrix'][1][0] + metrics['confusion_matrix'][1][1]} true FT converters "
    f"(recall = {metrics['recall']:.1%})</b>, at the cost of many false positives "
    f"(precision = {metrics['precision']:.1%}) — an expected and acceptable trade-off given "
    "class-balanced weighting on a ~0.3% positive rate: for a low-cost intervention like a follow-up "
    "call, high recall (catching almost 2 out of 3 real converters) is worth far more than precision. "
    f"ROC-AUC of {metrics['roc_auc']:.2f} shows real separation between classes; PR-AUC of "
    f"{metrics['pr_auc']:.3f} is modest in absolute terms but roughly "
    f"{metrics['pr_auc']/metrics['positive_rate_pct']*100:.0f}&times; better than the "
    f"{metrics['positive_rate_pct']:.2f}% random-guess baseline for this metric, which is the correct "
    "reference point under this level of imbalance.",
    styles["Body"]))

story.append(Paragraph("Driver Interpretation", styles["H2"]))
story.append(Image(str(CHART_DIR / "feature_importance.png"), width=5.0 * inch, height=3.48 * inch))
story.append(Paragraph("Figure 3. Logistic regression coefficients (log-odds). Green = increases FT "
                        "odds, red = decreases FT odds.", styles["Caption"]))

story.append(Paragraph(
    f"The strongest positive drivers are <b>attempted</b> (odds &times;187.1), "
    f"<b>Single Referral &gt; 7 days</b> cohort (odds &times;33.6), and <b>AI Connected but not "
    f"Connected by TC — Set 1</b> cohort (odds &times;11.6). Simply being <b>attempted at all</b> is the "
    "single largest lift — unsurprising, since FT cannot happen without outreach, but it quantifies how "
    "much value is left on the table in low-attempt cohorts (e.g. OLX at 18.6%). Cohort identity adds "
    "large lift independently of attempt, confirming the Page 2 ranking. Counter-intuitively, "
    "<b>connected</b> alone has a negative coefficient once cohort and attempt are controlled for — see "
    "next page for why this is association, not causation.",
    styles["Body"]))

story.append(PageBreak())

# ============================================================
# PAGE 6 — RECOMMENDATIONS & LIMITATIONS
# ============================================================

story.append(Paragraph("Recommendations & Limitations", styles["H1"]))
story.append(HRFlowable(width=CONTENT_W, thickness=1.4, color=BLUE, spaceAfter=12))

story.append(Paragraph("What Vahan Should Do", styles["H2"]))
recs = [
    "<b>Reallocate sourcing spend toward referral and retargeted-2W channels</b> (Single Referral, "
    "Khanna 2W) and away from high-volume/low-quality sources like OLX, or renegotiate OLX lead pricing "
    "given its 0.08% FT rate versus 0.9%+ for the top cohorts.",
    "<b>Fix dialing discipline before fixing sourcing.</b> Cohorts with near-0% attempt rates (Khanna AI "
    "at 27.7%, OLX at 18.6%) are leaving conversions on the table regardless of source quality — closing "
    "this gap is operationally cheaper than acquiring new leads.",
    "<b>Prioritize speed-to-first-attempt.</b> The zero-FT 2W3W-Khanna cohorts also had the slowest "
    "follow-up (125–175 hours); tightening SLA on first call may recover otherwise-viable leads.",
    "<b>Use the model as a triage/prioritization tool, not a gatekeeper.</b> Given low precision, route "
    "high-scored leads to be called first/more persistently rather than excluding low-scored leads "
    "entirely — the cost of an extra call is low relative to the value of a missed convert.",
]
for r in recs:
    story.append(Paragraph(f"&bull;&nbsp; {r}", styles["BulletBody"]))

story.append(Paragraph("Class Imbalance", styles["H2"]))
story.append(Paragraph(
    f"FT positives are only {metrics['positive_rate_pct']:.2f}% of leads. Accuracy is meaningless here "
    "(a model predicting \"no FT\" for everyone would score 99.7%); this report uses recall, precision, "
    "F1, ROC-AUC and PR-AUC together, and leans on PR-AUC as the more honest metric under severe "
    "imbalance. class_weight='balanced' was used instead of oversampling to avoid fabricating synthetic "
    "positive examples from only 54 real ones.",
    styles["Body"]))

story.append(Paragraph("Target Leakage Considerations", styles["H2"]))
story.append(Paragraph(
    "OB_after_upload (onboarding) is the funnel step directly preceding FT and is a near-deterministic "
    "predictor of it (41% of onboarded candidates went on to FT, versus 0.03% of non-onboarded "
    "candidates) — including it, or FT_after_first_attempt / the derived *_pct columns, would leak "
    "target information and produce an artificially strong but operationally useless model. All such "
    "fields were excluded from training.",
    styles["Body"]))

story.append(Paragraph("Association vs. Causation", styles["H2"]))
story.append(Paragraph(
    "Logistic regression coefficients describe association within this observational data, not proven "
    "causal effects. The negative coefficient on \"connected\" almost certainly reflects confounding "
    "(e.g. candidates who connect but aren't interested are still counted as connected) rather than "
    "connecting actually hurting conversion — it should prompt a follow-up experiment (e.g. call-quality "
    "audit), not a policy of avoiding connections.",
    styles["Body"]))

story.append(Paragraph("Assumptions", styles["H2"]))
assumptions = [
    "FT_after_upload (not FT_after_first_attempt) was treated as the primary conversion metric, since it "
    "captures the full post-upload journey rather than only the first call.",
    "Cohorts with under 20 uploaded leads were treated as statistically unreliable and excluded from "
    "ranking, though they remain in the raw dataset.",
    "candidate_phone values are already hashed/anonymized in the source data; no further PII handling "
    "was required.",
    "The observation window (18 Jul – 6 Aug 2026) is assumed representative; cohorts uploaded very "
    "recently may still convert after the data snapshot was taken (right-censoring).",
]
for a in assumptions:
    story.append(Paragraph(f"&bull;&nbsp; {a}", styles["BulletBody"]))

# ------------------------------------------------------------------
doc = SimpleDocTemplate(
    str(REPORT_PATH), pagesize=LETTER,
    leftMargin=0.65 * inch, rightMargin=0.65 * inch,
    topMargin=0.6 * inch, bottomMargin=0.75 * inch,
    title="Vahan Case Study — Lead-Source Cohort Performance",
    author="Vahan Product Analytics Internship",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Saved -> {REPORT_PATH}")
