"""
RazorShield AI - Calibration Analysis

Evaluates whether the Gradient Boosting model's predicted
fraud probabilities are reasonably aligned with observed
fraud rates on the same held-out test set used for model
evaluation.

This is an evaluation-only script.

It does NOT retrain or modify the champion model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
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

CALIBRATION_CSV_PATH = (
    PROCESSED_DATA_DIR
    / "calibration_analysis.csv"
)

CALIBRATION_CURVE_PATH = (
    PROCESSED_DATA_DIR
    / "calibration_curve.png"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - calibration_analysis.py:74" * 70)
    print("RAZORSHIELD AI  CALIBRATION ANALYSIS - calibration_analysis.py:75")
    print("= - calibration_analysis.py:76" * 70)
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
# RECREATE HELD-OUT TEST SET
# ============================================================

def create_test_set(
    X,
    y,
):

    print("= - calibration_analysis.py:138" * 70)
    print("1. RECREATING HELDOUT TEST SET - calibration_analysis.py:139")
    print("= - calibration_analysis.py:140" * 70)
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

    print("= - calibration_analysis.py:187" * 70)
    print("2. GENERATING MODEL PROBABILITIES - calibration_analysis.py:188")
    print("= - calibration_analysis.py:189" * 70)
    print()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

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
# CALIBRATION BUCKETS
# ============================================================

def calculate_calibration_buckets(
    y_test,
    probabilities,
):

    print("= - calibration_analysis.py:237" * 70)
    print("3. CALIBRATION BUCKET ANALYSIS - calibration_analysis.py:238")

    print("= - calibration_analysis.py:240" * 70)
    print()

    y_array = np.asarray(
        y_test,
        dtype=int,
    )

    # --------------------------------------------------------
    # Fixed probability buckets
    #
    # These buckets are easier for judges to interpret than
    # automatically generated bins.
    # --------------------------------------------------------

    bins = np.array(
        [
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.35,
            0.50,
            0.75,
            1.01,
        ]
    )

    labels = [
        "0-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "20-25%",
        "25-35%",
        "35-50%",
        "50-75%",
        "75-100%",
    ]

    bucket_ids = pd.cut(
        probabilities,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    rows = []

    for label in labels:

        mask = (
            bucket_ids == label
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        bucket_probabilities = (
            probabilities[mask]
        )

        bucket_actual = (
            y_array[mask]
        )

        mean_predicted = float(
            bucket_probabilities.mean()
        )

        observed_fraud_rate = float(
            bucket_actual.mean()
        )

        fraud_count = int(
            bucket_actual.sum()
        )

        rows.append(
            {
                "probability_bucket": label,
                "transaction_count": count,
                "fraud_count": fraud_count,
                "mean_predicted_probability":
                    mean_predicted,
                "observed_fraud_rate":
                    observed_fraud_rate,
                "calibration_gap":
                    abs(
                        mean_predicted
                        - observed_fraud_rate
                    ),
            }
        )

    calibration_df = pd.DataFrame(
        rows
    )

    print(
        "Probability | Transactions | Fraud | "
        "Mean Predicted | Observed Fraud | Gap"
    )

    print("" * 85)

    for _, row in calibration_df.iterrows():

        print(
            f"{row['probability_bucket']:>10} | "
            f"{int(row['transaction_count']):12d} | "
            f"{int(row['fraud_count']):5d} | "
            f"{row['mean_predicted_probability'] * 100:13.2f}% | "
            f"{row['observed_fraud_rate'] * 100:14.2f}% | "
            f"{row['calibration_gap'] * 100:5.2f}%"
        )

    print()

    return calibration_df


# ============================================================
# BRIER SCORE
# ============================================================

def calculate_brier_score(
    y_test,
    probabilities,
):

    print("= - calibration_analysis.py:378" * 70)
    print("4. BRIER SCORE - calibration_analysis.py:379")
    print("= - calibration_analysis.py:380" * 70)
    print()

    score = brier_score_loss(
        y_test,
        probabilities,
    )

    print(
        f"Brier score: {score:.6f}"
    )

    print()

    return score


# ============================================================
# CALIBRATION ERROR
# ============================================================

def calculate_calibration_error(
    calibration_df,
):

    if calibration_df.empty:

        return 0.0

    # --------------------------------------------------------
    # Weighted mean absolute calibration gap.
    #
    # Larger buckets contribute proportionally more because
    # they contain more transactions.
    # --------------------------------------------------------

    total_transactions = (
        calibration_df[
            "transaction_count"
        ].sum()
    )

    if total_transactions == 0:

        return 0.0

    weighted_error = (
        (
            calibration_df[
                "calibration_gap"
            ]
            *
            calibration_df[
                "transaction_count"
            ]
        ).sum()
        / total_transactions
    )

    return float(
        weighted_error
    )


# ============================================================
# CALIBRATION CURVE
# ============================================================

def create_calibration_curve(
    y_test,
    probabilities,
):

    print("= - calibration_analysis.py:453" * 70)
    print("5. CALIBRATION CURVE - calibration_analysis.py:454")
    print("= - calibration_analysis.py:455" * 70)
    print()

    import matplotlib.pyplot as plt

    fraction_of_positives, mean_predicted = (
        calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )
    )

    figure = plt.figure(
        figsize=(8, 6)
    )

    axis = figure.add_subplot(
        111
    )

    axis.plot(
        mean_predicted,
        fraction_of_positives,
        marker="o",
        label="Gradient Boosting",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.set_xlabel(
        "Mean predicted fraud probability"
    )

    axis.set_ylabel(
        "Observed fraud rate"
    )

    axis.set_title(
        "RazorShield AI - Calibration Curve"
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        CALIBRATION_CURVE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Calibration curve saved to:"
    )

    print(
        CALIBRATION_CURVE_PATH
    )

    print()

    return (
        mean_predicted,
        fraction_of_positives,
    )


# ============================================================
# INTERPRETATION
# ============================================================

def print_interpretation(
    brier_score,
    calibration_error,
):

    print("= - calibration_analysis.py:557" * 70)
    print("6. CALIBRATION INTERPRETATION - calibration_analysis.py:558")
    print("= - calibration_analysis.py:559" * 70)
    print()

    print(
        f"Weighted calibration error: "
        f"{calibration_error * 100:.2f}%"
    )

    print(
        f"Brier score: "
        f"{brier_score:.6f}"
    )

    print()

    # --------------------------------------------------------
    # Do not claim that the model is "perfectly calibrated".
    # This is a descriptive sanity check.
    # --------------------------------------------------------

    if calibration_error <= 0.05:

        print(
            "[INFO] Predicted probabilities show "
            "relatively small average bucket-level "
            "deviation from observed fraud rates."
        )

    elif calibration_error <= 0.10:

        print(
            "[INFO] Predicted probabilities show "
            "moderate bucket-level calibration error."
        )

    else:

        print(
            "[INFO] Predicted probabilities show "
            "substantial bucket-level calibration error."
        )

    print()

    print(
        "Important: calibration is evaluated on the "
        "held-out test set and does not change the "
        "model's risk decisions."
    )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_calibration_results(
    calibration_df,
    brier_score,
    calibration_error,
):

    output_df = (
        calibration_df.copy()
    )

    # Add summary metrics to every row so the CSV
    # remains self-contained for downstream use.

    output_df[
        "brier_score"
    ] = brier_score

    output_df[
        "weighted_calibration_error"
    ] = calibration_error

    output_df.to_csv(
        CALIBRATION_CSV_PATH,
        index=False,
    )

    print("= - calibration_analysis.py:642" * 70)
    print("7. SAVING CALIBRATION RESULTS - calibration_analysis.py:643")
    print("= - calibration_analysis.py:644" * 70)
    print()

    print(
        "Calibration analysis saved to:"
    )

    print(
        CALIBRATION_CSV_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load existing champion model
    # --------------------------------------------------------

    X, y, model = load_data()

    # --------------------------------------------------------
    # Recreate the exact held-out test set
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
    # Calibration buckets
    # --------------------------------------------------------

    calibration_df = (
        calculate_calibration_buckets(
            y_test,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Brier score
    # --------------------------------------------------------

    brier_score = (
        calculate_brier_score(
            y_test,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Weighted calibration error
    # --------------------------------------------------------

    calibration_error = (
        calculate_calibration_error(
            calibration_df,
        )
    )

    # --------------------------------------------------------
    # Calibration curve
    # --------------------------------------------------------

    create_calibration_curve(
        y_test,
        probabilities,
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print_interpretation(
        brier_score,
        calibration_error,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_calibration_results(
        calibration_df,
        brier_score,
        calibration_error,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("= - calibration_analysis.py:754" * 70)
    print("CALIBRATION ANALYSIS COMPLETE - calibration_analysis.py:755")
    print("= - calibration_analysis.py:756" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()