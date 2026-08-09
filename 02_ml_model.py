"""
02_ml_model.py
--------------
Builds a Logistic Regression model to identify the factors that influence
the probability of a candidate completing their First Trip (FT_after_upload).

Design choices (see report Page 4 for full reasoning):
  - Target        : FT_after_upload (1 = candidate completed First Trip)
  - Leakage guard : OB_after_upload / OB_after_first_attempt / FT_after_first_attempt
                    and all the pre-computed *_pct ratio columns are EXCLUDED.
                    OB (onboarding) is the funnel step immediately before FT and
                    is a near-perfect proxy for the target (see report Page 6),
                    so it cannot be used as a predictive feature.
  - Features      : lead_source (top cohorts + "Other"), attempted, connected,
                    tag_filled, interested, attempt_per_lead,
                    upload_to_first_attempt_p50_hrs, day-of-week of upload
  - Imbalance     : FT is rare (~0.3% positive). Uses class_weight='balanced'
                    rather than oversampling, and reports PR-AUC alongside
                    ROC-AUC because ROC-AUC is optimistic under heavy imbalance.

Run:
    python 02_ml_model.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
MODEL_DIR = OUT_DIR / "model"
CHART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "ft_after_upload"

# ------------------------------------------------------------------
# 1. Load + prepare data
# ------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "raw_leads.csv")
    df = df.rename(columns={
        "Uploaded Leads": "uploaded_leads",
        "Attempted": "attempted",
        "Connected": "connected",
        "Attempt per Lead": "attempt_per_lead",
        "tag_filled": "tag_filled",
        "Interested": "interested",
        "OB_after_upload": "ob_after_upload",
        "OB_after_first_attempt": "ob_after_first_attempt",
        "FT_after_upload": "ft_after_upload",
        "FT_after_first_attempt": "ft_after_first_attempt",
        "upload_to_first_attempt_P50 (hrs)": "upload_to_first_attempt_p50_hrs",
    })
    df["candidate_phone"] = df["candidate_phone"].astype(str)
    df = df.dropna(subset=["candidate_phone"])
    df = df[df["candidate_phone"] != "nan"].copy()

    # collapse rare lead sources into "Other" (keep cohorts with >=100 leads distinct)
    counts = df["lead_source"].value_counts()
    keep = counts[counts >= 100].index
    df["lead_source_grp"] = np.where(df["lead_source"].isin(keep), df["lead_source"], "Other")

    # day of week upload happened (captures batch/operational timing effects)
    df["upload_date"] = pd.to_datetime(df["upload_date"], errors="coerce")
    df["upload_dow"] = df["upload_date"].dt.dayofweek.astype("Int64").astype(str)

    return df


FEATURES_NUM = ["attempt_per_lead", "upload_to_first_attempt_p50_hrs"]
FEATURES_BIN = ["attempted", "connected", "tag_filled", "interested"]
FEATURES_CAT = ["lead_source_grp", "upload_dow"]


def build_pipeline() -> Pipeline:
    numeric_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(transformers=[
        ("num", numeric_tf, FEATURES_NUM),
        ("bin", "passthrough", FEATURES_BIN),
        ("cat", categorical_tf, FEATURES_CAT),
    ])
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("prep", preprocess), ("clf", model)])


def get_feature_names(pipeline: Pipeline) -> list:
    prep = pipeline.named_steps["prep"]
    num_names = FEATURES_NUM
    bin_names = FEATURES_BIN
    cat_names = list(prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(FEATURES_CAT))
    return num_names + bin_names + cat_names


def main():
    df = load_data()
    X = df[FEATURES_NUM + FEATURES_BIN + FEATURES_CAT]
    y = df[TARGET].astype(int)

    print(f"Total leads: {len(df):,}  |  FT positives: {y.sum():,} ({y.mean()*100:.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum()} positives)  |  "
          f"Test: {len(X_test):,} rows ({y_test.sum()} positives)")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\nConfusion matrix [[TN FP] [FN TP]]:\n", cm)
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}  PR-AUC: {pr_auc:.3f}")
    print("\n", classification_report(y_test, y_pred, digits=3, zero_division=0))

    # ---------------- feature importance (coefficients) ----------------
    feature_names = get_feature_names(pipe)
    coefs = pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df = coef_df.sort_values("coefficient", ascending=False)
    coef_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("\nTop positive drivers:\n", coef_df.head(8).to_string(index=False))
    print("\nTop negative drivers:\n", coef_df.tail(8).to_string(index=False))

    # ---------------- save metrics ----------------
    metrics = {
        "n_total": int(len(df)),
        "n_positive": int(y.sum()),
        "positive_rate_pct": round(float(y.mean() * 100), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
    }
    with open(OUT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- charts ----------------
    # confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No FT", "Pred: FT"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No FT", "Actual: FT"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, label=f"PR-AUC = {pr_auc:.3f}", color="#16a34a")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "pr_curve.png", dpi=150)
    plt.close()

    # feature importance chart (top 8 pos / top 8 neg)
    top = pd.concat([coef_df.head(8), coef_df.tail(8)]).drop_duplicates()
    top = top.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_xlabel("Logistic Regression Coefficient (log-odds)")
    ax.set_title("Top Drivers of First-Trip (FT) Conversion")
    fig.subplots_adjust(left=0.42, right=0.95, top=0.92, bottom=0.1)
    plt.savefig(CHART_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # save model
    with open(MODEL_DIR / "logistic_regression_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print(f"\nSaved model, metrics, and charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
