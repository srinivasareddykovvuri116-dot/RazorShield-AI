"""
RazorShield AI - Temporal-Safe Baseline Model

Uses only temporal-safe features generated from information
available before each transaction.

Model:
    Logistic Regression

Evaluation:
    70% chronological training
    15% chronological validation
    15% out-of-time test

Metrics:
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC
    False Positive Rate
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


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
    / "temporal_baseline_logistic_regression.joblib"
)

SCALER_PATH = (
    MODEL_DIR
    / "temporal_baseline_scaler.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - train_temporal_baseline.py:89" * 70)
    print("RAZORSHIELD AI  TEMPORAL BASELINE TRAINING - train_temporal_baseline.py:90")
    print("= - train_temporal_baseline.py:91" * 70)
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

    target_df = pd.read_csv(
        TARGET_PATH
    )

    if "is_fraud" not in target_df.columns:

        raise ValueError(
            "Target column 'is_fraud' was not found "
            f"in {TARGET_PATH}"
        )

    y = target_df[
        "is_fraud"
    ]

    print(
        f"Features: {X.shape[0]:,} rows × "
        f"{X.shape[1]:,} columns"
    )

    print(
        f"Target:   {len(y):,} rows"
    )

    print()

    return X, y


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(X, y):

    print("= - train_temporal_baseline.py:145" * 70)
    print("1. DATA VALIDATION - train_temporal_baseline.py:146")
    print("= - train_temporal_baseline.py:147" * 70)
    print()

    if len(X) != len(y):

        raise ValueError(
            "Feature and target row counts do not match."
        )

    if X.isnull().any().any():

        raise ValueError(
            "Missing values detected in feature matrix."
        )

    numeric_X = X.select_dtypes(
        include=np.number
    )

    if len(numeric_X.columns) != len(X.columns):

        raise ValueError(
            "Non-numeric features detected."
        )

    if np.isinf(
        X.to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected."
        )

    print(
        "[PASS] Feature/target rows aligned."
    )

    print(
        "[PASS] No missing values."
    )

    print(
        "[PASS] All features are numeric."
    )

    print(
        "[PASS] No infinite values."
    )

    print()


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("= - train_temporal_baseline.py:205" * 70)
    print("2. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT - train_temporal_baseline.py:206")
    print("= - train_temporal_baseline.py:207" * 70)
    print()

    total_rows = len(X)

    train_end = int(
        total_rows * 0.70
    )

    validation_end = int(
        total_rows * 0.85
    )

    # --------------------------------------------------------
    # Preserve original transaction order.
    # No shuffling.
    # --------------------------------------------------------

    X_train = X.iloc[
        :train_end
    ].copy()

    X_validation = X.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[
        validation_end:
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    y_validation = y.iloc[
        train_end:validation_end
    ].copy()

    y_test = y.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # Display split information
    # --------------------------------------------------------

    print(
        f"Total samples:       {total_rows:,}"
    )

    print(
        f"Training samples:     {len(X_train):,}"
    )

    print(
        f"Validation samples:   {len(X_validation):,}"
    )

    print(
        f"Out-of-time samples:  {len(X_test):,}"
    )

    print()

    print(
        f"Training fraud rate:    "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Validation fraud rate:  "
        f"{y_validation.mean() * 100:.2f}%"
    )

    print(
        f"Out-of-time fraud rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    print()

    print(
        "[PASS] Data split chronologically."
    )

    print(
        "[PASS] No shuffling was used."
    )

    print(
        "[PASS] Final test period was kept "
        "completely out of training."
    )

    print()

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# SCALE
# ============================================================

def scale_data(
    X_train,
    X_validation,
    X_test,
):

    print("= - train_temporal_baseline.py:323" * 70)
    print("3. FEATURE SCALING - train_temporal_baseline.py:324")
    print("= - train_temporal_baseline.py:325" * 70)
    print()

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit the scaler only on the training data.
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # Validation and test data must use
    # the scaler fitted on training data.
    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "[PASS] Scaler fitted only on training data."
    )

    print(
        "[PASS] Validation and test sets transformed "
        "using the training scaler."
    )

    print()

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(
    X_train,
    y_train,
):

    print("= - train_temporal_baseline.py:374" * 70)
    print("4. MODEL TRAINING - train_temporal_baseline.py:375")
    print("= - train_temporal_baseline.py:376" * 70)
    print()

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "[PASS] Temporal-safe Logistic Regression trained."
    )

    print()

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    X_test,
    y_test,
):

    print("= - train_temporal_baseline.py:412" * 70)
    print("5. TEMPORAL MODEL EVALUATION - train_temporal_baseline.py:413")
    print("= - train_temporal_baseline.py:414" * 70)
    print()

    threshold = 0.50

    # ========================================================
    # VALIDATION PERIOD
    # ========================================================

    print("VALIDATION PERIOD - train_temporal_baseline.py:423")
    print("" * 70)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0,
    )

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    validation_cm = confusion_matrix(
        y_validation,
        validation_predictions,
    )

    if validation_cm.shape != (2, 2):

        raise ValueError(
            "Validation confusion matrix does not contain "
            "both binary classes."
        )

    (
        validation_tn,
        validation_fp,
        validation_fn,
        validation_tp,
    ) = validation_cm.ravel()

    validation_denominator = (
        validation_fp +
        validation_tn
    )

    validation_fpr = (
        validation_fp /
        validation_denominator
        if validation_denominator > 0
        else 0.0
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Precision : {validation_precision:.4f}"
    )

    print(
        f"Recall    : {validation_recall:.4f}"
    )

    print(
        f"F1 Score  : {validation_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {validation_roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {validation_pr_auc:.4f}"
    )

    print(
        f"FPR       : {validation_fpr:.4f}"
    )

    print()

    print("Confusion Matrix: - train_temporal_baseline.py:523")
    print(validation_cm)

    print()

    # ========================================================
    # OUT-OF-TIME TEST PERIOD
    # ========================================================

    print("OUTOFTIME TEST PERIOD - train_temporal_baseline.py:532")
    print("" * 70)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    if test_cm.shape != (2, 2):

        raise ValueError(
            "Out-of-time confusion matrix does not contain "
            "both binary classes."
        )

    (
        test_tn,
        test_fp,
        test_fn,
        test_tp,
    ) = test_cm.ravel()

    test_denominator = (
        test_fp +
        test_tn
    )

    test_fpr = (
        test_fp /
        test_denominator
        if test_denominator > 0
        else 0.0
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Precision : {test_precision:.4f}"
    )

    print(
        f"Recall    : {test_recall:.4f}"
    )

    print(
        f"F1 Score  : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {test_roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {test_pr_auc:.4f}"
    )

    print(
        f"FPR       : {test_fpr:.4f}"
    )

    print()

    print("Confusion Matrix: - train_temporal_baseline.py:632")
    print(test_cm)

    print()

    # ========================================================
    # TEMPORAL PERFORMANCE COMPARISON
    # ========================================================

    print("= - train_temporal_baseline.py:641" * 70)
    print("6. TEMPORAL PERFORMANCE COMPARISON - train_temporal_baseline.py:642")
    print("= - train_temporal_baseline.py:643" * 70)
    print()

    print(
        f"{'Metric':<18}"
        f"{'Validation':>16}"
        f"{'Out-of-Time':>16}"
        f"{'Change':>16}"
    )

    print("" * 70)

    metrics = [
        (
            "Precision",
            validation_precision,
            test_precision,
        ),
        (
            "Recall",
            validation_recall,
            test_recall,
        ),
        (
            "F1",
            validation_f1,
            test_f1,
        ),
        (
            "ROC-AUC",
            validation_roc_auc,
            test_roc_auc,
        ),
        (
            "PR-AUC",
            validation_pr_auc,
            test_pr_auc,
        ),
        (
            "FPR",
            validation_fpr,
            test_fpr,
        ),
    ]

    comparison_rows = []

    for (
        metric_name,
        validation_value,
        test_value,
    ) in metrics:

        change = (
            test_value -
            validation_value
        )

        comparison_rows.append(
            {
                "metric":
                    metric_name,

                "validation":
                    validation_value,

                "out_of_time":
                    test_value,

                "change":
                    change,
            }
        )

        print(
            f"{metric_name:<18}"
            f"{validation_value * 100:>15.2f}%"
            f"{test_value * 100:>15.2f}%"
            f"{change * 100:>15.2f} pp"
        )

    print()

    # ========================================================
    # SAVE TEMPORAL COMPARISON
    # ========================================================

    comparison_path = (
        PROCESSED_DATA_DIR
        / "temporal_validation_comparison.csv"
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        comparison_path,
        index=False,
    )

    print(
        "Temporal comparison saved to:"
    )

    print(
        comparison_path
    )

    print()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("= - train_temporal_baseline.py:756" * 70)
    print("OUTOFTIME CLASSIFICATION REPORT - train_temporal_baseline.py:757")
    print("= - train_temporal_baseline.py:758" * 70)
    print()

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    return {
        "validation": {
            "precision": validation_precision,
            "recall": validation_recall,
            "f1": validation_f1,
            "roc_auc": validation_roc_auc,
            "pr_auc": validation_pr_auc,
            "fpr": validation_fpr,
            "tn": int(validation_tn),
            "fp": int(validation_fp),
            "fn": int(validation_fn),
            "tp": int(validation_tp),
        },

        "out_of_time": {
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "fpr": test_fpr,
            "tn": int(test_tn),
            "fp": int(test_fp),
            "fn": int(test_fn),
            "tp": int(test_tp),
        },

        "comparison": comparison_rows,

        "threshold": threshold,
    }


# ============================================================
# TOP TEMPORAL FEATURES
# ============================================================

def show_feature_importance(
    model,
    feature_names,
):

    print("= - train_temporal_baseline.py:815" * 70)
    print("7. TOP TEMPORAL FEATURES - train_temporal_baseline.py:816")
    print("= - train_temporal_baseline.py:817" * 70)
    print()

    coefficients = model.coef_[0]

    importance = pd.DataFrame({

        "feature":
            feature_names,

        "coefficient":
            coefficients,

        "absolute_importance":
            np.abs(coefficients),

    })

    importance = importance.sort_values(
        "absolute_importance",
        ascending=False,
    )

    print(
        "Top 20 features:"
    )

    print()

    for _, row in importance.head(20).iterrows():

        if row["coefficient"] > 0:

            direction = "↑ fraud risk"

        else:

            direction = "↓ fraud risk"

        print(
            f"{row['feature']:<45} "
            f"{row['coefficient']:>10.4f}  "
            f"{direction}"
        )

    print()

    return importance


# ============================================================
# SAVE
# ============================================================

def save_model(
    model,
    scaler,
):

    print("= - train_temporal_baseline.py:876" * 70)
    print("8. SAVING MODEL - train_temporal_baseline.py:877")
    print("= - train_temporal_baseline.py:878" * 70)
    print()

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Scaler saved to:"
    )

    print(
        SCALER_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(
        X,
        y,
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(
        X,
        y,
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    ) = scale_data(
        X_train,
        X_validation,
        X_test,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train_scaled,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate validation + out-of-time test
    # --------------------------------------------------------

    evaluation_results = evaluate_model(
        model,
        X_validation_scaled,
        y_validation,
        X_test_scaled,
        y_test,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    show_feature_importance(
        model,
        X.columns,
    )

    # --------------------------------------------------------
    # Save model and scaler
    # --------------------------------------------------------

    save_model(
        model,
        scaler,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("= - train_temporal_baseline.py:1007" * 70)
    print("TEMPORAL BASELINE TRAINING COMPLETE - train_temporal_baseline.py:1008")
    print("= - train_temporal_baseline.py:1009" * 70)
    print()

    print(
        "Evaluation threshold: 50%"
    )

    print()

    print(
        "Out-of-Time Results:"
    )

    print(
        f"Precision : "
        f"{evaluation_results['out_of_time']['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{evaluation_results['out_of_time']['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{evaluation_results['out_of_time']['f1'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC   : "
        f"{evaluation_results['out_of_time']['roc_auc'] * 100:.2f}%"
    )

    print(
        f"PR-AUC    : "
        f"{evaluation_results['out_of_time']['pr_auc'] * 100:.2f}%"
    )

    print(
        f"FPR       : "
        f"{evaluation_results['out_of_time']['fpr'] * 100:.2f}%"
    )

    print()

    print(
        "Out-of-Time Confusion Matrix:"
    )

    print(
        f"TN: {evaluation_results['out_of_time']['tn']:,}"
    )

    print(
        f"FP: {evaluation_results['out_of_time']['fp']:,}"
    )

    print(
        f"FN: {evaluation_results['out_of_time']['fn']:,}"
    )

    print(
        f"TP: {evaluation_results['out_of_time']['tp']:,}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

