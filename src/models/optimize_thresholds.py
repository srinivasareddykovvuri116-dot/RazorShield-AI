"""
RazorShield AI - Risk Threshold Optimization

Finds practical fraud-risk thresholds for:

    ALLOW
    REVIEW
    BLOCK

The optimization is performed on the temporal-safe test set.

Important:
    We do NOT choose thresholds using accuracy alone.
    Fraud detection and false-positive cost are both considered.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "temporal_features.csv"
)

TARGET_PATH = (
    PROCESSED_DATA_DIR
    / "temporal_target.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "temporal_baseline_logistic_regression.joblib"
)

SCALER_PATH = (
    MODEL_DIR
    / "temporal_baseline_scaler.joblib"
)

RESULT_PATH = (
    PROCESSED_DATA_DIR
    / "threshold_analysis.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("= - optimize_thresholds.py:79" * 70)
    print("RAZORSHIELD AI  THRESHOLD OPTIMIZATION - optimize_thresholds.py:80")
    print("= - optimize_thresholds.py:81" * 70)
    print()

    X = pd.read_csv(
        FEATURE_PATH
    )

    y = pd.read_csv(
        TARGET_PATH
    )["is_fraud"]

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"Features: {len(X):,}"
    )

    print(
        f"Fraud cases: {int(y.sum()):,}"
    )

    print()

    return X, y, model, scaler


# ============================================================
# TEST SET
# ============================================================

def create_test_set(X, y):

    print("= - optimize_thresholds.py:119" * 70)
    print("1. RECREATING TEST SET - optimize_thresholds.py:120")
    print("= - optimize_thresholds.py:121" * 70)
    print()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"Test transactions: {len(X_test):,}"
    )

    print(
        f"Test fraud cases: {int(y_test.sum()):,}"
    )

    print()

    return X_test, y_test


# ============================================================
# PREDICTIONS
# ============================================================

def get_probabilities(
    model,
    scaler,
    X_test,
):

    print("= - optimize_thresholds.py:160" * 70)
    print("2. GENERATING FRAUD PROBABILITIES - optimize_thresholds.py:161")
    print("= - optimize_thresholds.py:162" * 70)
    print()

    X_scaled = scaler.transform(
        X_test
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    print(
        "[PASS] Fraud probabilities generated."
    )

    print()

    return probabilities


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    y_test,
    probabilities,
):

    print("= - optimize_thresholds.py:191" * 70)
    print("3. THRESHOLD ANALYSIS - optimize_thresholds.py:192")
    print("= - optimize_thresholds.py:193" * 70)
    print()

    thresholds = np.arange(
        0.05,
        1.00,
        0.05,
    )

    results = []

    total_fraud = int(
        y_test.sum()
    )

    total_legitimate = int(
        (y_test == 0).sum()
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tp = int(
            (
                (predictions == 1)
                &
                (y_test.to_numpy() == 1)
            ).sum()
        )

        fp = int(
            (
                (predictions == 1)
                &
                (y_test.to_numpy() == 0)
            ).sum()
        )

        fn = int(
            (
                (predictions == 0)
                &
                (y_test.to_numpy() == 1)
            ).sum()
        )

        tn = int(
            (
                (predictions == 0)
                &
                (y_test.to_numpy() == 0)
            ).sum()
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        fraud_capture_rate = (
            tp / total_fraud
        )

        false_positive_rate = (
            fp / total_legitimate
        )

        results.append({

            "threshold":
                round(
                    float(threshold),
                    2,
                ),

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fraud_capture_rate":
                fraud_capture_rate,

            "false_positive_rate":
                false_positive_rate,

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "true_negatives":
                tn,
        })

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "fraud_capture_rate",
                "false_positive_rate",
            ]
        ].to_string(
            index=False,
            formatters={
                "precision":
                    "{:.3f}".format,

                "recall":
                    "{:.3f}".format,

                "f1":
                    "{:.3f}".format,

                "fraud_capture_rate":
                    "{:.3f}".format,

                "false_positive_rate":
                    "{:.3f}".format,
            },
        )
    )

    print()

    return results_df


# ============================================================
# CHOOSE BALANCED THRESHOLD
# ============================================================

def choose_balanced_threshold(
    results_df,
):

    print("= - optimize_thresholds.py:360" * 70)
    print("4. BALANCED THRESHOLD - optimize_thresholds.py:361")
    print("= - optimize_thresholds.py:362" * 70)
    print()

    # --------------------------------------------------------
    # We want:
    #
    #   high recall
    #   high precision
    #   high F1
    #
    # But we also want to avoid excessive false positives.
    #
    # Constraint:
    #   capture at least 85% of fraud
    #   false-positive rate <= 5%
    # --------------------------------------------------------

    candidates = results_df[
        (
            results_df[
                "fraud_capture_rate"
            ] >= 0.85
        )
        &
        (
            results_df[
                "false_positive_rate"
            ] <= 0.05
        )
    ].copy()

    if candidates.empty:

        print(
            "[WARNING] No threshold satisfied "
            "both constraints."
        )

        # Fall back to best F1.
        best = results_df.loc[
            results_df["f1"].idxmax()
        ]

        reason = (
            "Best F1 fallback"
        )

    else:

        # Choose highest F1 among candidates.
        best = candidates.loc[
            candidates["f1"].idxmax()
        ]

        reason = (
            "Best F1 under business constraints"
        )

    threshold = float(
        best["threshold"]
    )

    print(
        f"Recommended threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Reason: {reason}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    print(
        f"Fraud capture: "
        f"{best['fraud_capture_rate']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{best['false_positive_rate']:.4f}"
    )

    print()

    return threshold


# ============================================================
# THREE-LEVEL RISK POLICY
# ============================================================

def create_risk_policy(
    results_df,
    block_threshold,
):

    print("= - optimize_thresholds.py:472" * 70)
    print("5. RISK POLICY - optimize_thresholds.py:473")
    print("= - optimize_thresholds.py:474" * 70)
    print()

    # --------------------------------------------------------
    # We define:
    #
    # BLOCK:
    #     optimized threshold and above
    #
    # REVIEW:
    #     middle range
    #
    # ALLOW:
    #     lower-risk transactions
    #
    # For the initial policy, review threshold is
    # half of the block threshold.
    #
    # This is a starting policy, not a final business rule.
    # --------------------------------------------------------

    review_threshold = round(
        block_threshold * 0.5,
        2,
    )

    if review_threshold < 0.10:

        review_threshold = 0.10

    print(
        f"ALLOW  : score < {review_threshold:.2f}"
    )

    print(
        f"REVIEW : {review_threshold:.2f} "
        f"≤ score < {block_threshold:.2f}"
    )

    print(
        f"BLOCK  : score ≥ {block_threshold:.2f}"
    )

    print()

    return review_threshold


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
):

    results_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print(
        f"Threshold analysis saved to:"
    )

    print(
        RESULT_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    X, y, model, scaler = load_data()

    X_test, y_test = create_test_set(
        X,
        y,
    )

    probabilities = get_probabilities(
        model,
        scaler,
        X_test,
    )

    results_df = evaluate_thresholds(
        y_test,
        probabilities,
    )

    block_threshold = (
        choose_balanced_threshold(
            results_df
        )
    )

    review_threshold = (
        create_risk_policy(
            results_df,
            block_threshold,
        )
    )

    save_results(
        results_df
    )

    print("= - optimize_thresholds.py:587" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETE - optimize_thresholds.py:588")
    print("= - optimize_thresholds.py:589" * 70)
    print()

    print(
        "Recommended initial policy:"
    )

    print(
        f"  ALLOW  < {review_threshold:.2f}"
    )

    print(
        f"  REVIEW {review_threshold:.2f}"
        f" - {block_threshold:.2f}"
    )

    print(
        f"  BLOCK  >= {block_threshold:.2f}"
    )

    print()


if __name__ == "__main__":

    main()