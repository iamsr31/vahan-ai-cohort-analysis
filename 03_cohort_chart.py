"""
03_cohort_chart.py
-------------------
Generates the cohort-performance bar chart used on Page 2 of the report
(FT rate per uploaded lead, by lead-source cohort).

Run:
    python 03_cohort_chart.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(OUT_DIR / "cohort_aggregated_output.csv")
    df = df.sort_values("ft_rate_per_uploaded_pct", ascending=True)

    colors = ["#2563eb" if i >= len(df) - 3 else "#93c5fd" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(df["lead_source"], df["ft_rate_per_uploaded_pct"], color=colors)
    ax.set_xlabel("FT conversion rate (% of uploaded leads)")
    ax.set_title("Cohort (Lead-Source) Performance — FT Conversion Rate")
    for bar, val in zip(bars, df["ft_rate_per_uploaded_pct"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}%",
                 va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "cohort_ft_rate.png", dpi=150)
    plt.close()
    print(f"Saved -> {CHART_DIR / 'cohort_ft_rate.png'}")


if __name__ == "__main__":
    main()
