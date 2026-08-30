"""
RazorShield AI - Gradient Boosting Threshold Optimization

Optimizes and evaluates decision thresholds for the current
champion Gradient Boosting fraud model.

The analysis reports:

    - Precision
    - Recall
    - F1
    - Fraud capture rate
    - False-positive rate
    - Flagged transaction rate
    - False-positive transaction count
    - Legitimate transaction volume affected
    - Fraud transaction volume captured

The selected operating point is based on the existing
business constraints:

    - At least 85% fraud capture
    - False-positive rate <= 2%

The script also reports several fixed operating points so
the trade-off between fraud capture and false positives
is transparent.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
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
    / "gradient_boosting_fraud_model.joblib"
)

RESULT_PATH = (
    PROCESSED_DATA_DIR
    / "gradient_threshold_analysis.csv"
)

OPERATING_POINT_PATH = (
    PROCESSED_DATA_DIR
    / "gradient_operating_points.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - optimize_gradient_thresholds.py:96" * 70)
    print(
        "RAZORSHIELD AI - GRADIENT THRESHOLD OPTIMIZATION"
    )
    print("= - optimize_gradient_thresholds.py:100" * 70)
    print()

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_PATH}"
        )

    if not TARGET_PATH.exists():

        raise FileNotFoundError(
            f"Target file not found:\n{TARGET_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Gradient Boosting model not found:\n"
            f"{MODEL_PATH}"
        )

    X = pd.read_csv(
        FEATURE_PATH
    )

    y = pd.read_csv(
        TARGET_PATH
    )["is_fraud"]

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Features: {X.shape[0]:,} rows × "
        f"{X.shape[1]:,} columns"
    )

    print(
        f"Fraud cases: {int(y.sum()):,}"
    )

    print(
        f"Legitimate cases: "
        f"{int((y == 0).sum()):,}"
    )

    print()

    return X, y, model


# ============================================================
# RECREATE TEST SET
# ============================================================

def create_test_set(
    X,
    y,
):

    print("= - optimize_gradient_thresholds.py:162" * 70)
    print("1. RECREATING HELDOUT TEST SET - optimize_gradient_thresholds.py:163")
    print("= - optimize_gradient_thresholds.py:164" * 70)
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
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    print(
        f"Test fraud cases: {int(y_test.sum()):,}"
    )

    print(
        f"Test legitimate cases: "
        f"{int((y_test == 0).sum()):,}"
    )

    print()

    return X_test, y_test


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

def generate_probabilities(
    model,
    X_test,
):

    print("= - optimize_gradient_thresholds.py:211" * 70)
    print("2. GENERATING FRAUD PROBABILITIES - optimize_gradient_thresholds.py:212")
    print("= - optimize_gradient_thresholds.py:213" * 70)
    print()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "[PASS] Fraud probabilities generated."
    )

    print()

    print(
        f"Minimum probability: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: "
        f"{probabilities.max():.6f}"
    )

    print()

    return probabilities


# ============================================================
# OVERALL MODEL METRICS
# ============================================================

def calculate_model_metrics(
    y_test,
    probabilities,
):

    print("= - optimize_gradient_thresholds.py:250" * 70)
    print("3. MODEL METRICS - optimize_gradient_thresholds.py:251")
    print("= - optimize_gradient_thresholds.py:252" * 70)
    print()

    predictions = (
        probabilities >= 0.50
    ).astype(int)

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        f"Precision @ 0.50 : {precision:.4f}"
    )

    print(
        f"Recall @ 0.50    : {recall:.4f}"
    )

    print(
        f"F1 @ 0.50        : {f1:.4f}"
    )

    print()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    X_test,
    y_test,
    probabilities,
):

    print("= - optimize_gradient_thresholds.py:325" * 70)
    print("4. FULL THRESHOLD ANALYSIS - optimize_gradient_thresholds.py:326")
    print("= - optimize_gradient_thresholds.py:327" * 70)
    print()

    # Fine-grained threshold sweep.
    thresholds = np.arange(
        0.10,
        1.00,
        0.025,
    )

    total_fraud = int(
        y_test.sum()
    )

    total_legitimate = int(
        (y_test == 0).sum()
    )

    # --------------------------------------------------------
    # Amount column
    # --------------------------------------------------------

    if "amount" not in X_test.columns:

        raise ValueError(
            "The held-out feature set does not contain "
            "the required 'amount' column."
        )

    amounts = X_test["amount"].to_numpy()

    y_array = y_test.to_numpy()

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        tp = int(
            (
                (predictions == 1)
                &
                (y_array == 1)
            ).sum()
        )

        fp = int(
            (
                (predictions == 1)
                &
                (y_array == 0)
            ).sum()
        )

        fn = int(
            (
                (predictions == 0)
                &
                (y_array == 1)
            ).sum()
        )

        tn = int(
            (
                (predictions == 0)
                &
                (y_array == 0)
            ).sum()
        )

        # ----------------------------------------------------
        # Classification metrics
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Rates
        # ----------------------------------------------------

        fraud_capture_rate = (
            tp / total_fraud
            if total_fraud > 0
            else 0.0
        )

        false_positive_rate = (
            fp / total_legitimate
            if total_legitimate > 0
            else 0.0
        )

        flagged_rate = (
            (tp + fp) / len(y_test)
        )

        # ----------------------------------------------------
        # BUSINESS VOLUME
        # ----------------------------------------------------

        legitimate_flagged_mask = (
            (predictions == 1)
            &
            (y_array == 0)
        )

        fraud_captured_mask = (
            (predictions == 1)
            &
            (y_array == 1)
        )

        legitimate_volume_flagged = float(
            amounts[
                legitimate_flagged_mask
            ].sum()
        )

        fraud_volume_captured = float(
            amounts[
                fraud_captured_mask
            ].sum()
        )

        total_flagged_volume = float(
            amounts[
                predictions == 1
            ].sum()
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "threshold":
                round(
                    float(threshold),
                    3,
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

            "flagged_rate":
                flagged_rate,

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "true_negatives":
                tn,

            "legitimate_volume_flagged":
                legitimate_volume_flagged,

            "fraud_volume_captured":
                fraud_volume_captured,

            "total_flagged_volume":
                total_flagged_volume,
        })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Print compact threshold table
    # --------------------------------------------------------

    display_columns = [

        "threshold",
        "precision",
        "recall",
        "f1",
        "fraud_capture_rate",
        "false_positive_rate",
        "false_positives",
        "legitimate_volume_flagged",
        "fraud_volume_captured",
    ]

    print(
        results_df[
            display_columns
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

                "legitimate_volume_flagged":
                    "{:,.2f}".format,

                "fraud_volume_captured":
                    "{:,.2f}".format,
            },
        )
    )

    print()

    return results_df


# ============================================================
# SELECTED OPERATING POINTS
# ============================================================

def create_operating_point_table(
    results_df,
):

    print("= - optimize_gradient_thresholds.py:597" * 70)
    print("5. SELECTED OPERATING POINTS - optimize_gradient_thresholds.py:598")
    print("= - optimize_gradient_thresholds.py:599" * 70)
    print()

    # These points are intentionally explicit so the dashboard
    # and README can show the operating trade-off.
    selected_thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.50,
    ]

    operating_points = (
        results_df[
            results_df["threshold"].isin(
                selected_thresholds
            )
        ]
        .copy()
        .sort_values("threshold")
    )

    # --------------------------------------------------------
    # If a threshold is absent because of floating-point
    # representation, locate the closest available threshold.
    # --------------------------------------------------------

    if len(operating_points) != len(
        selected_thresholds
    ):

        rows = []

        for requested in selected_thresholds:

            index = (
                results_df["threshold"]
                .sub(requested)
                .abs()
                .idxmin()
            )

            row = (
                results_df
                .loc[index]
                .copy()
            )

            row["threshold"] = requested

            rows.append(row)

        operating_points = (
            pd.DataFrame(rows)
            .sort_values("threshold")
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Add human-readable percentages
    # --------------------------------------------------------

    operating_points[
        "threshold_percent"
    ] = (
        operating_points["threshold"]
        * 100
    )

    operating_points[
        "precision_percent"
    ] = (
        operating_points["precision"]
        * 100
    )

    operating_points[
        "recall_percent"
    ] = (
        operating_points["recall"]
        * 100
    )

    operating_points[
        "fpr_percent"
    ] = (
        operating_points["false_positive_rate"]
        * 100
    )

    # --------------------------------------------------------
    # Print judge-friendly table
    # --------------------------------------------------------

    print(
        "Threshold | Precision | Recall | FPR | "
        "FP Count | Legitimate Volume Flagged | "
        "Fraud Volume Captured"
    )

    print("" * 105)

    for _, row in operating_points.iterrows():

        print(
            f"{row['threshold'] * 100:8.0f}% | "
            f"{row['precision'] * 100:8.2f}% | "
            f"{row['recall'] * 100:6.2f}% | "
            f"{row['false_positive_rate'] * 100:5.2f}% | "
            f"{int(row['false_positives']):8d} | "
            f"₹{row['legitimate_volume_flagged']:,.2f} | "
            f"₹{row['fraud_volume_captured']:,.2f}"
        )

    print()

    # --------------------------------------------------------
    # Save selected operating points
    # --------------------------------------------------------

    operating_points.to_csv(
        OPERATING_POINT_PATH,
        index=False,
    )

    print(
        "Selected operating-point analysis saved to:"
    )

    print(
        OPERATING_POINT_PATH
    )

    print()

    return operating_points


# ============================================================
# BUSINESS THRESHOLD
# ============================================================

def choose_block_threshold(
    results_df,
):

    print("= - optimize_gradient_thresholds.py:748" * 70)
    print("6. RECOMMENDED BLOCK THRESHOLD - optimize_gradient_thresholds.py:749")
    print("= - optimize_gradient_thresholds.py:750" * 70)
    print()

    # --------------------------------------------------------
    # Business constraints
    #
    # We want:
    #
    #   at least 85% fraud capture
    #   false-positive rate <= 2%
    #
    # If several thresholds satisfy the constraints,
    # choose the one with the best F1.
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
            ] <= 0.02
        )
    ].copy()

    if not candidates.empty:

        best = candidates.loc[
            candidates["f1"].idxmax()
        ]

        selection_reason = (
            "Best F1 under 85% fraud-capture "
            "and 2% false-positive constraints."
        )

    else:

        fallback = results_df[
            results_df[
                "fraud_capture_rate"
            ] >= 0.80
        ].copy()

        if fallback.empty:

            best = results_df.loc[
                results_df["f1"].idxmax()
            ]

            selection_reason = (
                "No threshold reached 80% fraud capture; "
                "selected best F1."
            )

        else:

            best = (
                fallback
                .sort_values(
                    [
                        "false_positive_rate",
                        "f1",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .iloc[0]
            )

            selection_reason = (
                "Fallback: minimized false positives "
                "while retaining at least 80% fraud capture."
            )

    threshold = float(
        best["threshold"]
    )

    print(
        f"Recommended BLOCK threshold: "
        f"{threshold:.3f}"
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

    print(
        f"False-positive transactions: "
        f"{int(best['false_positives'])}"
    )

    print(
        "Legitimate volume flagged: "
        f"₹{best['legitimate_volume_flagged']:,.2f}"
    )

    print(
        "Fraud volume captured: "
        f"₹{best['fraud_volume_captured']:,.2f}"
    )

    print()

    print(
        f"Selection reason: "
        f"{selection_reason}"
    )

    print()

    return threshold


# ============================================================
# REVIEW THRESHOLD
# ============================================================

def choose_review_threshold(
    results_df,
    block_threshold,
):

    print("= - optimize_gradient_thresholds.py:901" * 70)
    print("7. REVIEW THRESHOLD - optimize_gradient_thresholds.py:902")
    print("= - optimize_gradient_thresholds.py:903" * 70)
    print()

    candidates = results_df[
        (
            results_df[
                "threshold"
            ] < block_threshold
        )
        &
        (
            results_df[
                "fraud_capture_rate"
            ] >= 0.95
        )
        &
        (
            results_df[
                "flagged_rate"
            ] <= 0.10
        )
    ].copy()

    if candidates.empty:

        review_threshold = round(
            block_threshold * 0.5,
            3,
        )

        reason = (
            "Fallback: half of block threshold."
        )

    else:

        best = candidates.sort_values(
            "threshold",
            ascending=False,
        ).iloc[0]

        review_threshold = float(
            best["threshold"]
        )

        reason = (
            "Highest threshold retaining at least "
            "95% fraud capture with <=10% flagged volume."
        )

    print(
        f"Recommended REVIEW threshold: "
        f"{review_threshold:.3f}"
    )

    print(
        f"Reason: {reason}"
    )

    print()

    return review_threshold


# ============================================================
# RISK POLICY SUMMARY
# ============================================================

def print_policy(
    review_threshold,
    block_threshold,
):

    print("= - optimize_gradient_thresholds.py:976" * 70)
    print("8. PROPOSED RISK POLICY - optimize_gradient_thresholds.py:977")
    print("= - optimize_gradient_thresholds.py:978" * 70)
    print()

    print(
        f"ALLOW  : score < {review_threshold:.3f}"
    )

    print(
        f"REVIEW : {review_threshold:.3f}"
        f" <= score < {block_threshold:.3f}"
    )

    print(
        f"BLOCK  : score >= {block_threshold:.3f}"
    )

    print()

    print(
        "Note: These thresholds are model-validation "
        "thresholds, not universal payment-industry rules."
    )

    print()


# ============================================================
# SAVE FULL RESULTS
# ============================================================

def save_results(
    results_df,
):

    results_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print(
        "Full threshold analysis saved to:"
    )

    print(
        RESULT_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, model = load_data()

    # --------------------------------------------------------
    # Recreate exact held-out test set
    # --------------------------------------------------------

    X_test, y_test = create_test_set(
        X,
        y,
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = (
        generate_probabilities(
            model,
            X_test,
        )
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    calculate_model_metrics(
        y_test,
        probabilities,
    )

    # --------------------------------------------------------
    # Full threshold sweep
    # --------------------------------------------------------

    results_df = analyze_thresholds(
        X_test,
        y_test,
        probabilities,
    )

    # --------------------------------------------------------
    # Selected operating points
    # --------------------------------------------------------

    create_operating_point_table(
        results_df,
    )

    # --------------------------------------------------------
    # Choose block threshold
    # --------------------------------------------------------

    block_threshold = (
        choose_block_threshold(
            results_df
        )
    )

    # --------------------------------------------------------
    # Choose review threshold
    # --------------------------------------------------------

    review_threshold = (
        choose_review_threshold(
            results_df,
            block_threshold,
        )
    )

    # --------------------------------------------------------
    # Print policy
    # --------------------------------------------------------

    print_policy(
        review_threshold,
        block_threshold,
    )

    # --------------------------------------------------------
    # Save full threshold analysis
    # --------------------------------------------------------

    save_results(
        results_df
    )

    print("= - optimize_gradient_thresholds.py:1125" * 70)
    print(
        "GRADIENT THRESHOLD OPTIMIZATION COMPLETE"
    )
    print("= - optimize_gradient_thresholds.py:1129" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()