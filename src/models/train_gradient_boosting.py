"""
RazorShield AI - Gradient Boosting Fraud Model

Trains a non-linear HistGradientBoostingClassifier
using the temporal-safe feature set.

This model is compared against the temporal-safe
Logistic Regression baseline.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
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
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - train_gradient_boosting.py:73" * 70)
    print("RAZORSHIELD AI  GRADIENT BOOSTING - train_gradient_boosting.py:74")
    print("= - train_gradient_boosting.py:75" * 70)
    print()

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_PATH}"
        )

    if not TARGET_PATH.exists():

        raise FileNotFoundError(
            f"Target file not found:\n{TARGET_PATH}"
        )

    X = pd.read_csv(
        FEATURE_PATH
    )

    y = pd.read_csv(
        TARGET_PATH
    )["is_fraud"]

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

    return X, y


# ============================================================
# VALIDATION
# ============================================================

def validate_data(
    X,
    y,
):

    print("= - train_gradient_boosting.py:126" * 70)
    print("1. DATA VALIDATION - train_gradient_boosting.py:127")
    print("= - train_gradient_boosting.py:128" * 70)
    print()

    if len(X) != len(y):

        raise ValueError(
            "Feature and target row counts do not match."
        )

    if X.isnull().any().any():

        raise ValueError(
            "Missing values detected."
        )

    if np.isinf(
        X.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    if not all(
        np.issubdtype(
            dtype,
            np.number,
        )
        for dtype in X.dtypes
    ):

        raise ValueError(
            "Non-numeric features detected."
        )

    print(
        "[PASS] Feature/target rows aligned."
    )

    print(
        "[PASS] No missing values."
    )

    print(
        "[PASS] No infinite values."
    )

    print(
        "[PASS] All features are numeric."
    )

    print()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    X,
    y,
):

    print("= - train_gradient_boosting.py:191" * 70)
    print("2. TRAIN / TEST SPLIT - train_gradient_boosting.py:192")
    print("= - train_gradient_boosting.py:193" * 70)
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
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    print()

    print(
        f"Training fraud rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Testing fraud rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    print()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    print("= - train_gradient_boosting.py:250" * 70)
    print("3. MODEL TRAINING - train_gradient_boosting.py:251")
    print("= - train_gradient_boosting.py:252" * 70)
    print()

    model = HistGradientBoostingClassifier(

        learning_rate=0.08,

        max_iter=250,

        max_leaf_nodes=31,

        min_samples_leaf=30,

        l2_regularization=1.0,

        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "[PASS] HistGradientBoostingClassifier trained."
    )

    print()

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    print("= - train_gradient_boosting.py:294" * 70)
    print("4. MODEL EVALUATION - train_gradient_boosting.py:295")
    print("= - train_gradient_boosting.py:296" * 70)
    print()

    # --------------------------------------------------------
    # Predicted probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print()

    tn, fp, fn, tp = cm.ravel()

    print(
        f"True Negatives : {tn:,}"
    )

    print(
        f"False Positives: {fp:,}"
    )

    print(
        f"False Negatives: {fn:,}"
    )

    print(
        f"True Positives : {tp:,}"
    )

    print()

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # CALIBRATION SANITY CHECK
    # ========================================================

    print("= - train_gradient_boosting.py:426" * 70)
    print("5. CALIBRATION SANITY CHECK - train_gradient_boosting.py:427")
    print("= - train_gradient_boosting.py:428" * 70)
    print()

    calibration_bins = [
        (0.00, 0.10),
        (0.10, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.00),
    ]

    calibration_rows = []

    y_test_array = np.asarray(
        y_test
    )

    for lower, upper in calibration_bins:

        if upper == 1.00:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        predicted_mean = float(
            probabilities[mask].mean()
        )

        actual_rate = float(
            y_test_array[mask].mean()
        )

        calibration_rows.append(
            {
                "risk_bucket": (
                    f"{lower:.0%}-{upper:.0%}"
                ),
                "sample_count": count,
                "mean_predicted_risk": (
                    predicted_mean
                ),
                "observed_fraud_rate": (
                    actual_rate
                ),
                "absolute_gap": abs(
                    predicted_mean -
                    actual_rate
                ),
            }
        )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    print(
        "Risk Bucket | Samples | "
        "Predicted Risk | Observed Fraud | Gap"
    )

    print("" * 70)

    for row in calibration_rows:

        print(
            f"{row['risk_bucket']:>10} | "
            f"{row['sample_count']:>7,} | "
            f"{row['mean_predicted_risk'] * 100:>13.2f}% | "
            f"{row['observed_fraud_rate'] * 100:>14.2f}% | "
            f"{row['absolute_gap'] * 100:>5.2f}%"
        )

    print()

    # --------------------------------------------------------
    # Save calibration results
    # --------------------------------------------------------

    calibration_path = (
        PROCESSED_DATA_DIR
        / "gradient_boosting_calibration.csv"
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
    )

    print(
        "Calibration results saved to:"
    )

    print(
        calibration_path
    )

    print()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "calibration": calibration_rows,
    }

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    X_test,
    y_test,
):

    print("= - train_gradient_boosting.py:566" * 70)
    print("6. FEATURE IMPORTANCE - train_gradient_boosting.py:567")
    print("= - train_gradient_boosting.py:568" * 70)
    print()

    from sklearn.inspection import permutation_importance

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------
    #
    # HistGradientBoostingClassifier does not expose
    # feature_importances_ directly.
    #
    # Therefore we use permutation importance on the
    # held-out test set.
    #
    # PR-AUC is used as the scoring metric because this is
    # a fraud-detection problem with class imbalance.
    # --------------------------------------------------------

    print(
        "Calculating permutation importance "
        "on the held-out test set..."
    )

    print(
        "Scoring metric: PR-AUC"
    )

    print()

    importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Build importance dataframe
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": (
                importance_result.importances_mean
            ),
            "importance_std": (
                importance_result.importances_std
            ),
        }
    )

    # Sort from most important to least important
    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    importance_df.insert(
        0,
        "rank",
        range(
            1,
            len(importance_df) + 1,
        ),
    )

    # --------------------------------------------------------
    # Save complete feature importance results
    # --------------------------------------------------------

    importance_path = (
        PROCESSED_DATA_DIR
        / "gradient_boosting_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------
    # Display Top 10
    # --------------------------------------------------------

    print(
        "Top 10 features by permutation importance:"
    )

    print()

    print(
        f"{'Rank':<6}"
        f"{'Feature':<42}"
        f"{'Mean Δ PR-AUC':>16}"
        f"{'Std':>12}"
    )

    print("" * 80)

    for _, row in importance_df.head(10).iterrows():

        print(
            f"{int(row['rank']):<6}"
            f"{row['feature']:<42}"
            f"{row['importance_mean']:>16.6f}"
            f"{row['importance_std']:>12.6f}"
        )

    print()

    print(
        "Full feature-importance results saved to:"
    )

    print(
        importance_path
    )

    print()

    return importance_df


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
):

    print("= - train_gradient_boosting.py:712" * 70)
    print("6. SAVING MODEL - train_gradient_boosting.py:713")
    print("= - train_gradient_boosting.py:714" * 70)
    print()

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    X, y = load_data()

    validate_data(
        X,
        y,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    model = train_model(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    feature_importance = show_feature_importance(
         model,
         X_test,
         y_test,
    )

    save_model(
        model,
    )

    print("= - train_gradient_boosting.py:777" * 70)
    print("GRADIENT BOOSTING COMPLETE - train_gradient_boosting.py:778")
    print("= - train_gradient_boosting.py:779" * 70)
    print()

    print(
        "Baseline comparison target:"
    )

    print(
        "  Temporal Logistic Regression PR-AUC: 0.8384"
    )

    print()

    print(
        f"  Gradient Boosting PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print()

    if metrics["pr_auc"] > 0.8384:

        print(
            "[WIN] Gradient Boosting "
            "beats the temporal baseline."
        )

    else:

        print(
            "[INFO] Gradient Boosting did not "
            "beat the temporal baseline yet."
        )

    print()


if __name__ == "__main__":

    main()