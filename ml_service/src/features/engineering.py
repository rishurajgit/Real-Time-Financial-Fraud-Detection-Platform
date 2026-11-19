"""Leakage-free transaction feature engineering & reproducible preprocessing.

Computes behavioral features strictly from prior transactions (causal order
sorted by TransactionDT) so training and online Feast serving use identical
definitions with zero target or future leakage.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# Engineered behavioral features shared across Phase 1 training, Phase 2 FastAPI, and Phase 3 Feast
ENGINEERED_BEHAVIORAL_FEATURES: tuple[str, ...] = (
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "account_tx_count_prior",
    "account_amt_mean_prior",
    "account_amt_std_prior",
    "amount_vs_user_avg",
    "amount_zscore_deviation",
    "time_since_prev_tx_sec",
    "device_change_flag",
    "location_change_flag",
    "tx_hour_of_day",
    "tx_day_of_week",
    "amt_log1p",
    "amt_decimal_cents",
    "has_identity_record",
)

RISK_ENCODED_CATEGORIES: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "DeviceType",
)

CORE_RAW_NUMERIC_CANDIDATES: tuple[str, ...] = (
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_19",
    "id_20",
)

CORE_RAW_CATEGORICAL_CANDIDATES: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)


def _safe_str(val: Any) -> str:
    if val is None or val != val:  # NaN check without requiring numpy/pandas import
        return "UNKNOWN"
    text = str(val).strip()
    return text if text and text.lower() != "nan" else "UNKNOWN"


def build_entity_keys_row(
    card1: Any,
    addr1: Any,
    p_email: Any,
    device_info: Any,
    device_type: Any,
    addr2: Any,
) -> tuple[str, str, str]:
    """Derive deterministic account, device, and location keys from raw IEEE-CIS attributes."""
    account_key = f"c1:{_safe_str(card1)}|a1:{_safe_str(addr1)}|em:{_safe_str(p_email)}"
    device_key = f"{_safe_str(device_type)}|{_safe_str(device_info)}"
    location_key = f"{_safe_str(addr1)}|{_safe_str(addr2)}"
    return account_key, device_key, location_key


def compute_causal_behavioral_records(
    records: list[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Pure-Python causal behavioral feature calculator (strictly ordered by TransactionDT).

    For each transaction row t, only strictly preceding transactions (< t) of the
    same derived account_key contribute to velocity counts, running mean/std,
    time_since_prev_tx_sec, device_change_flag, and location_change_flag.
    """
    import math

    ordered_indices = sorted(
        range(len(records)),
        key=lambda i: (float(records[i]["TransactionDT"]), int(records[i].get("TransactionID", i))),
    )

    # Rolling per-account state
    window_1h: dict[str, deque[float]] = {}
    window_24h: dict[str, deque[float]] = {}
    window_7d: dict[str, deque[float]] = {}
    amt_count: dict[str, int] = {}
    amt_mean: dict[str, float] = {}
    amt_m2: dict[str, float] = {}  # Welford running sum of squared deviations
    last_dt: dict[str, float] = {}
    last_device: dict[str, str] = {}
    last_location: dict[str, str] = {}

    results: list[dict[str, float | int | str]] = [{} for _ in range(len(records))]

    for idx in ordered_indices:
        row = records[idx]
        dt = float(row["TransactionDT"])
        amt = max(0.0, float(row["TransactionAmt"]))

        account_key, device_key, location_key = build_entity_keys_row(
            card1=row.get("card1"),
            addr1=row.get("addr1"),
            p_email=row.get("P_emaildomain"),
            device_info=row.get("DeviceInfo"),
            device_type=row.get("DeviceType"),
            addr2=row.get("addr2"),
        )

        # Evict expired timestamps BEFORE recording current row (strictly causal)
        q1 = window_1h.setdefault(account_key, deque())
        while q1 and dt - q1[0] > 3600.0:
            q1.popleft()
        tx_count_1h = len(q1)

        q24 = window_24h.setdefault(account_key, deque())
        while q24 and dt - q24[0] > 86400.0:
            q24.popleft()
        tx_count_24h = len(q24)

        q7d = window_7d.setdefault(account_key, deque())
        while q7d and dt - q7d[0] > 604800.0:
            q7d.popleft()
        tx_count_7d = len(q7d)

        prior_n = amt_count.get(account_key, 0)
        prior_mean = amt_mean.get(account_key, 0.0)
        prior_std = math.sqrt(amt_m2[account_key] / (prior_n - 1)) if prior_n >= 2 else 0.0

        if prior_n > 0:
            amount_vs_user_avg = amt / max(prior_mean, 1.0)
            amount_zscore = (amt - prior_mean) / max(prior_std, 1.0)
            time_since_prev = max(0.0, dt - last_dt[account_key])
            prev_dev = last_device[account_key]
            prev_loc = last_location[account_key]
            device_change = 1 if (device_key != "UNKNOWN|UNKNOWN" and prev_dev != "UNKNOWN|UNKNOWN" and device_key != prev_dev) else 0
            location_change = 1 if (location_key != "UNKNOWN|UNKNOWN" and prev_loc != "UNKNOWN|UNKNOWN" and location_key != prev_loc) else 0
        else:
            amount_vs_user_avg = 1.0
            amount_zscore = 0.0
            time_since_prev = -1.0
            device_change = 0
            location_change = 0

        results[idx] = {
            "account_key": account_key,
            "device_key": device_key,
            "location_key": location_key,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "tx_count_7d": tx_count_7d,
            "account_tx_count_prior": prior_n,
            "account_amt_mean_prior": round(prior_mean, 4),
            "account_amt_std_prior": round(prior_std, 4),
            "amount_vs_user_avg": round(amount_vs_user_avg, 4),
            "amount_zscore_deviation": round(amount_zscore, 4),
            "time_since_prev_tx_sec": round(time_since_prev, 2),
            "device_change_flag": device_change,
            "location_change_flag": location_change,
            "tx_hour_of_day": int((dt // 3600) % 24),
            "tx_day_of_week": int((dt // 86400) % 7),
            "amt_log1p": round(math.log1p(amt), 6),
            "amt_decimal_cents": round(amt - math.floor(amt), 4),
            "has_identity_record": int(row.get("has_identity_record", 0)),
        }

        # Update rolling state AFTER emitting current row's features
        q1.append(dt)
        q24.append(dt)
        q7d.append(dt)
        new_n = prior_n + 1
        delta = amt - prior_mean
        new_mean = prior_mean + delta / new_n
        delta2 = amt - new_mean
        amt_count[account_key] = new_n
        amt_mean[account_key] = new_mean
        amt_m2[account_key] = amt_m2.get(account_key, 0.0) + delta * delta2
        last_dt[account_key] = dt
        if device_key != "UNKNOWN|UNKNOWN":
            last_device[account_key] = device_key
        elif account_key not in last_device:
            last_device[account_key] = device_key
        if location_key != "UNKNOWN|UNKNOWN":
            last_location[account_key] = location_key
        elif account_key not in last_location:
            last_location[account_key] = location_key

    return results


def engineer_transaction_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Attach leakage-free behavioral features to a merged IEEE-CIS DataFrame."""
    import pandas as pd

    required = ["TransactionDT", "TransactionAmt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns for feature engineering: {missing}")

    ordered = df.sort_values(
        by=["TransactionDT", "TransactionID"] if "TransactionID" in df.columns else ["TransactionDT"],
        ascending=True,
    ).reset_index(drop=True)

    cols_needed = [
        c
        for c in (
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
            "addr1",
            "addr2",
            "P_emaildomain",
            "DeviceInfo",
            "DeviceType",
            "has_identity_record",
        )
        if c in ordered.columns
    ]
    records = ordered[cols_needed].to_dict(orient="records")
    computed = compute_causal_behavioral_records(records)
    feat_df = pd.DataFrame(computed)

    out = ordered.copy()
    for col in feat_df.columns:
        out[col] = feat_df[col].values
    return out


def _normalize_category_value(value: Any) -> str:
    if value is None or value != value:  # handles float('nan') / None
        return "MISSING"
    text = str(value)
    return "MISSING" if text.lower() == "nan" else text


def compute_oof_smoothed_risk_rates(
    categories: Sequence[Any],
    targets: Sequence[int | float],
    n_splits: int = 5,
    smoothing_strength: float = 50.0,
    random_state: int = 42,
) -> tuple[list[float], dict[str, float], float]:
    """Compute leakage-safe out-of-fold Bayesian-smoothed target encodings.

    Fold assignments depend strictly on (len(targets), n_splits, random_state)
    and NEVER on target labels. Consequently, for every training row i:
      - Fold membership of row i is invariant to targets[i].
      - Both the out-of-fold global prior and out-of-fold category statistics
        exclude row i's fold entirely.
      - Flipping targets[i] cannot change encoded_oof[i].

    Also returns the full-training-set mapping (full_risk_map, full_global_rate)
    for validation and production inference via transform().
    """
    n = len(categories)
    if n != len(targets):
        raise ValueError("categories and targets must have identical length.")
    if n == 0:
        return [], {}, 0.0

    norm_cats = [_normalize_category_value(c) for c in categories]
    y_vals = [float(t) for t in targets]

    # Full-training mapping for validation & inference (strictly fit on training split)
    full_total_sum = sum(y_vals)
    full_global_rate = full_total_sum / n
    full_counts: dict[str, int] = {}
    full_sums: dict[str, float] = {}
    for cat, y_val in zip(norm_cats, y_vals):
        full_counts[cat] = full_counts.get(cat, 0) + 1
        full_sums[cat] = full_sums.get(cat, 0.0) + y_val

    full_risk_map = {
        cat: (full_sums[cat] + smoothing_strength * full_global_rate)
        / (full_counts[cat] + smoothing_strength)
        for cat in full_counts
    }

    if n < 2:
        return [0.0], full_risk_map, full_global_rate

    effective_splits = max(2, min(int(n_splits), n))

    # Deterministic, label-independent K-fold partition
    indices = list(range(n))
    rng = random.Random(random_state)
    rng.shuffle(indices)
    fold_assignments = [0] * n
    for rank, idx in enumerate(indices):
        fold_assignments[idx] = rank % effective_splits

    # Pre-aggregate per-fold totals and per-(fold, category) sums/counts in O(n)
    fold_sizes = [0] * effective_splits
    fold_y_sums = [0.0] * effective_splits
    fold_cat_counts: list[dict[str, int]] = [{} for _ in range(effective_splits)]
    fold_cat_sums: list[dict[str, float]] = [{} for _ in range(effective_splits)]

    for idx in range(n):
        f = fold_assignments[idx]
        cat = norm_cats[idx]
        y_val = y_vals[idx]
        fold_sizes[f] += 1
        fold_y_sums[f] += y_val
        fold_cat_counts[f][cat] = fold_cat_counts[f].get(cat, 0) + 1
        fold_cat_sums[f][cat] = fold_cat_sums[f].get(cat, 0.0) + y_val

    oof_encoded = [0.0] * n
    for idx in range(n):
        f = fold_assignments[idx]
        cat = norm_cats[idx]
        oof_n = n - fold_sizes[f]
        oof_total_y = full_total_sum - fold_y_sums[f]
        oof_global_prior = oof_total_y / oof_n if oof_n > 0 else 0.0

        oof_cat_count = full_counts[cat] - fold_cat_counts[f].get(cat, 0)
        oof_cat_sum = full_sums[cat] - fold_cat_sums[f].get(cat, 0.0)
        oof_encoded[idx] = (oof_cat_sum + smoothing_strength * oof_global_prior) / (
            oof_cat_count + smoothing_strength
        )

    return oof_encoded, full_risk_map, full_global_rate


@dataclass
class FraudFeaturePreprocessor:
    """Reproducible training/serving preprocessor fit strictly on training data.

    Prevents target leakage by:
      - Fitting high-missingness column drop masks on train only
      - Computing out-of-fold (OOF) Bayesian-smoothed category risk rates during
        fit_transform() so no training row's label leaks into its own feature
      - Applying full training-fitted risk-rate mappings during transform() for
        validation and online inference
      - Fitting numeric median imputers and categorical vocabularies on train only
    """

    max_missing_fraction: float = 0.95
    smoothing_strength: float = 50.0
    oof_n_splits: int = 5
    random_state: int = 42
    numeric_columns_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    risk_rate_maps_: dict[str, dict[str, float]] = field(default_factory=dict)
    global_fraud_rate_: float = 0.0
    numeric_medians_: dict[str, float] = field(default_factory=dict)
    category_vocab_: dict[str, dict[str, int]] = field(default_factory=dict)
    output_feature_names_: list[str] = field(default_factory=list)

    def fit(self, df: "pd.DataFrame", y: "pd.Series") -> "FraudFeaturePreprocessor":
        import numpy as np

        self.global_fraud_rate_ = float(y.mean()) if len(y) > 0 else 0.0

        # Select engineered + core raw numeric features present and below missingness cap
        candidate_num = list(ENGINEERED_BEHAVIORAL_FEATURES) + [
            c for c in CORE_RAW_NUMERIC_CANDIDATES if c in df.columns
        ]
        # Include V-features with reasonable completeness if present (cap to top 40 lowest-missing)
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if v_cols:
            v_missing = df[v_cols].isna().mean().sort_values()
            selected_v = v_missing[v_missing <= 0.50].head(40).index.tolist()
            candidate_num.extend(selected_v)

        seen: set[str] = set()
        self.numeric_columns_ = []
        for col in candidate_num:
            if col in df.columns and col not in seen:
                if float(df[col].isna().mean()) <= self.max_missing_fraction:
                    seen.add(col)
                    self.numeric_columns_.append(col)

        self.categorical_columns_ = [
            c
            for c in CORE_RAW_CATEGORICAL_CANDIDATES
            if c in df.columns and float(df[c].isna().mean()) <= self.max_missing_fraction
        ]

        # Numeric medians fit strictly on training fold
        self.numeric_medians_ = {}
        for col in self.numeric_columns_:
            med = df[col].median()
            self.numeric_medians_[col] = float(med) if np.isfinite(med) else 0.0

        # Categorical ordinal vocabularies fit strictly on training fold (0 reserved for unseen/missing)
        self.category_vocab_ = {}
        for col in self.categorical_columns_:
            values = df[col].fillna("MISSING").astype(str)
            counts = values.value_counts()
            # Keep categories occurring at least 5 times to prevent high-cardinality memorization
            frequent = counts[counts >= 5].index.tolist()
            self.category_vocab_[col] = {val: i + 1 for i, val in enumerate(frequent)}

        # Full-training Bayesian-smoothed category risk rates (used ONLY for validation/inference transform)
        self.risk_rate_maps_ = {}
        y_list = y.tolist()
        for col in RISK_ENCODED_CATEGORIES:
            if col not in df.columns:
                continue
            _, full_map, global_rate = compute_oof_smoothed_risk_rates(
                categories=df[col].tolist(),
                targets=y_list,
                n_splits=self.oof_n_splits,
                smoothing_strength=self.smoothing_strength,
                random_state=self.random_state,
            )
            self.global_fraud_rate_ = global_rate
            self.risk_rate_maps_[col] = full_map

        self.output_feature_names_ = (
            list(self.numeric_columns_)
            + [f"{c}_risk_rate" for c in self.risk_rate_maps_.keys()]
            + [f"{c}_ord" for c in self.categorical_columns_]
        )
        return self

    def transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Transform validation or online inference rows using training-fitted mappings only."""
        import numpy as np
        import pandas as pd

        if not self.output_feature_names_:
            raise RuntimeError("FraudFeaturePreprocessor must be fit before calling transform().")

        cols_out: dict[str, Any] = {}

        for col in self.numeric_columns_:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_medians_[col])
                cols_out[col] = series.astype(np.float32).values
            else:
                cols_out[col] = np.full(len(df), self.numeric_medians_[col], dtype=np.float32)

        for col, risk_map in self.risk_rate_maps_.items():
            if col in df.columns:
                vals = df[col].fillna("MISSING").astype(str)
                cols_out[f"{col}_risk_rate"] = (
                    vals.map(risk_map).fillna(self.global_fraud_rate_).astype(np.float32).values
                )
            else:
                cols_out[f"{col}_risk_rate"] = np.full(len(df), self.global_fraud_rate_, dtype=np.float32)

        for col in self.categorical_columns_:
            vocab = self.category_vocab_[col]
            if col in df.columns:
                vals = df[col].fillna("MISSING").astype(str)
                cols_out[f"{col}_ord"] = vals.map(vocab).fillna(0).astype(np.int32).values
            else:
                cols_out[f"{col}_ord"] = np.zeros(len(df), dtype=np.int32)

        return pd.DataFrame(cols_out, index=df.index)[self.output_feature_names_]

    def fit_transform(self, df: "pd.DataFrame", y: "pd.Series") -> "pd.DataFrame":
        """Fit full mappings for inference and return leakage-safe OOF-encoded training features."""
        import numpy as np

        self.fit(df, y)
        out = self.transform(df)

        y_list = y.tolist()
        for col in self.risk_rate_maps_.keys():
            oof_vals, _, _ = compute_oof_smoothed_risk_rates(
                categories=df[col].tolist(),
                targets=y_list,
                n_splits=self.oof_n_splits,
                smoothing_strength=self.smoothing_strength,
                random_state=self.random_state,
            )
            out[f"{col}_risk_rate"] = np.asarray(oof_vals, dtype=np.float32)

        return out[self.output_feature_names_]
