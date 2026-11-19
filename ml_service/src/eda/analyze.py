"""Exploratory Data Analysis (EDA) pipeline for IEEE-CIS Fraud Detection.

Computes genuine measured statistics on the merged IEEE-CIS dataset:
  - Class imbalance and fraud incidence rate
  - Missingness rates per feature
  - Numeric distributions (TransactionAmt, dist1, TransactionDT)
  - Categorical cardinality and fraud rates (ProductCD, card4, card6, DeviceType, P_emaildomain)
  - Top numeric feature correlations with isFraud
  - Publication-ready PNG charts and machine-readable eda_summary.json

Run:
    cd ml_service
    python -m src.eda.analyze [--nrows 100000]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data.loader import TARGET_COLUMN, load_ieee_cis_split

logger = logging.getLogger(__name__)


def run_eda(
    data_dir: Path,
    output_dir: Path,
    nrows: int | None = None,
) -> dict[str, Any]:
    """Execute full EDA on merged IEEE-CIS training data and write charts + JSON report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_ieee_cis_split(data_dir=data_dir, split="train", nrows=nrows)

    total_rows = int(len(df))
    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    legit_count = int((df[TARGET_COLUMN] == 0).sum())
    fraud_rate = float(fraud_count / max(1, total_rows))
    imbalance_ratio = float(legit_count / max(1, fraud_count))

    # 1. Missing values profile
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    columns_with_missing = int((missing_pct > 0).sum())
    columns_over_50pct_missing = int((missing_pct > 50.0).sum())
    top_missing = [
        {"feature": col, "missing_pct": round(float(pct), 2)}
        for col, pct in missing_pct.head(30).items()
    ]

    # 2. Numeric amount distribution split by target
    fraud_amt = df.loc[df[TARGET_COLUMN] == 1, "TransactionAmt"].dropna()
    legit_amt = df.loc[df[TARGET_COLUMN] == 0, "TransactionAmt"].dropna()

    def _quantiles(series: pd.Series) -> dict[str, float]:
        if series.empty:
            return {}
        return {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "p95": round(float(series.quantile(0.95)), 2),
            "max": round(float(series.max()), 2),
        }

    # 3. Categorical fraud rates
    categorical_candidates = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain"]
    categorical_summary: dict[str, Any] = {}
    for col in categorical_candidates:
        if col not in df.columns:
            continue
        series = df[col].fillna("MISSING").astype(str)
        grouped = (
            pd.DataFrame({"value": series, TARGET_COLUMN: df[TARGET_COLUMN]})
            .groupby("value", as_index=False)
            .agg(count=(TARGET_COLUMN, "size"), fraud_rate=(TARGET_COLUMN, "mean"))
            .sort_values("count", ascending=False)
        )
        categorical_summary[col] = {
            "unique_values": int(series.nunique()),
            "top_categories": [
                {
                    "value": str(row["value"]),
                    "count": int(row["count"]),
                    "fraud_rate_pct": round(float(row["fraud_rate"] * 100.0), 3),
                }
                for _, row in grouped.head(10).iterrows()
            ],
        }

    # 4. Numeric feature correlations with isFraud
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("TransactionID", TARGET_COLUMN)
    ]
    corr_rows: list[dict[str, Any]] = []
    target_series = df[TARGET_COLUMN].astype(float)
    for col in numeric_cols:
        col_series = df[col]
        if col_series.nunique(dropna=True) < 2:
            continue
        corr_val = col_series.corr(target_series)
        if pd.notna(corr_val):
            corr_rows.append({"feature": col, "pearson_corr": round(float(corr_val), 5)})

    corr_rows.sort(key=lambda r: abs(r["pearson_corr"]), reverse=True)
    top_correlations = corr_rows[:25]

    # 5. Generate EDA Charts
    sns.set_theme(style="whitegrid")

    # Chart 1: Class Imbalance
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Legitimate (isFraud=0)", "Fraud (isFraud=1)"]
    counts = [legit_count, fraud_count]
    bars = ax.bar(labels, counts, color=["#1D4ED8", "#DC2626"], width=0.55)
    ax.set_title("IEEE-CIS Class Distribution (Severe Imbalance)")
    ax.set_ylabel("Transaction Count")
    for bar, count in zip(bars, counts):
        pct = count / max(1, total_rows) * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    chart_imbalance = charts_dir / "class_imbalance.png"
    fig.savefig(chart_imbalance, dpi=150)
    plt.close(fig)

    # Chart 2: Top 30 Missingness Features
    top_miss_series = missing_pct.head(30)
    fig, ax = plt.subplots(figsize=(9, 7))
    top_miss_series.iloc[::-1].plot(kind="barh", ax=ax, color="#475569")
    ax.set_title("Top 30 Features by Missing Value Percentage")
    ax.set_xlabel("Missing Values (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    chart_missing = charts_dir / "missingness_top30.png"
    fig.savefig(chart_missing, dpi=150)
    plt.close(fig)

    # Chart 3: Log Transaction Amount Distribution by Fraud Label
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        np.log1p(legit_amt.clip(lower=0)),
        bins=60,
        density=True,
        alpha=0.6,
        color="#1D4ED8",
        label="Legitimate",
    )
    ax.hist(
        np.log1p(fraud_amt.clip(lower=0)),
        bins=60,
        density=True,
        alpha=0.6,
        color="#DC2626",
        label="Fraud",
    )
    ax.set_title("Log(1 + TransactionAmt) Density by Fraud Label")
    ax.set_xlabel("log1p(TransactionAmt)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    chart_amount = charts_dir / "amount_distribution_by_fraud.png"
    fig.savefig(chart_amount, dpi=150)
    plt.close(fig)

    # Chart 4: Fraud Rate by Hour of Day (TransactionDT modulo 86400)
    hour_of_day = ((df["TransactionDT"] // 3600) % 24).astype(int)
    hourly = (
        pd.DataFrame({"hour": hour_of_day, TARGET_COLUMN: df[TARGET_COLUMN]})
        .groupby("hour", as_index=False)
        .agg(volume=(TARGET_COLUMN, "size"), fraud_rate=(TARGET_COLUMN, "mean"))
    )
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(hourly["hour"], hourly["volume"], color="#CBD5E1", label="Tx Volume")
    ax1.set_xlabel("Hour of Day (TransactionDT mod 24h)")
    ax1.set_ylabel("Transaction Volume", color="#475569")
    ax2 = ax1.twinx()
    ax2.plot(
        hourly["hour"],
        hourly["fraud_rate"] * 100.0,
        color="#DC2626",
        marker="o",
        linewidth=2,
        label="Fraud Rate (%)",
    )
    ax2.set_ylabel("Fraud Rate (%)", color="#DC2626")
    ax1.set_title("Transaction Volume vs. Fraud Rate by Hour of Day")
    fig.tight_layout()
    chart_temporal = charts_dir / "temporal_fraud_rate.png"
    fig.savefig(chart_temporal, dpi=150)
    plt.close(fig)

    # Chart 5: Top 20 Absolute Correlations with isFraud
    if top_correlations:
        top20 = top_correlations[:20][::-1]
        fig, ax = plt.subplots(figsize=(8.5, 6))
        colors = ["#DC2626" if r["pearson_corr"] > 0 else "#1D4ED8" for r in top20]
        ax.barh([r["feature"] for r in top20], [r["pearson_corr"] for r in top20], color=colors)
        ax.axvline(0.0, color="#0F172A", linewidth=0.8)
        ax.set_title("Top 20 Numeric Feature Correlations with isFraud")
        ax.set_xlabel("Pearson Correlation")
        fig.tight_layout()
        chart_corr = charts_dir / "top_feature_correlations.png"
        fig.savefig(chart_corr, dpi=150)
        plt.close(fig)
    else:
        chart_corr = None

    summary: dict[str, Any] = {
        "dataset": "IEEE-CIS Fraud Detection",
        "rows_analyzed": total_rows,
        "column_count": int(df.shape[1]),
        "identity_join_match_rate_pct": round(float(df["has_identity_record"].mean() * 100.0), 2),
        "class_balance": {
            "legitimate_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_rate_pct": round(fraud_rate * 100.0, 4),
            "imbalance_ratio_legit_per_fraud": round(imbalance_ratio, 2),
        },
        "missing_values": {
            "columns_with_any_missing": columns_with_missing,
            "columns_over_50pct_missing": columns_over_50pct_missing,
            "top_30_missing": top_missing,
        },
        "transaction_amount": {
            "legitimate": _quantiles(legit_amt),
            "fraud": _quantiles(fraud_amt),
        },
        "time_span": {
            "min_transaction_dt": float(df["TransactionDT"].min()),
            "max_transaction_dt": float(df["TransactionDT"].max()),
            "span_days": round(float((df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400.0), 2),
        },
        "categorical_profiles": categorical_summary,
        "top_correlations_with_target": top_correlations,
        "generated_charts": [
            str(p)
            for p in (chart_imbalance, chart_missing, chart_amount, chart_temporal, chart_corr)
            if p is not None
        ],
    }

    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("EDA summary written to %s", summary_path)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run IEEE-CIS Fraud Detection EDA pipeline.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with raw IEEE-CIS CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for EDA charts & JSON summary.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick exploratory run.")
    args = parser.parse_args()

    cfg = load_config()
    summary = run_eda(
        data_dir=args.data_dir or cfg.data_dir,
        output_dir=args.output_dir or cfg.eda_output_dir,
        nrows=args.nrows,
    )
    print(json.dumps(summary["class_balance"], indent=2))


if __name__ == "__main__":
    main()
