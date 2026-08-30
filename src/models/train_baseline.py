"""
RazorShield AI - Baseline Fraud Detection Model

Trains a baseline Logistic Regression model and evaluates it
using metrics appropriate for an imbalanced fraud dataset.
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
from sklearn.model_selection import train_test_split
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
    exist_ok=True
)

FEATURE_PATH = (
    PROCESSED_DATA_DIR / "features.csv"
)

TARGET_PATH = (
    PROCESSED_DATA_DIR / "target.csv"
)

MODEL_PATH = (
    MODEL_DIR / "baseline_logistic_regression.joblib"
)

SCALER_PATH = (
    MODEL_DIR / "baseline_scaler.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - train_baseline.py:70" * 70)
    print("RAZORSHIELD AI  BASELINE MODEL TRAINING - train_baseline.py:71")
    print("= - train_baseline.py:72" * 70)
    print()

    X = pd.read_csv(
        FEATURE_PATH
    )

    target_df = pd.read_csv(
        TARGET_PATH
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
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("= - train_baseline.py:107" * 70)
    print("1. TRAIN / TEST SPLIT - train_baseline.py:108")
    print("= - train_baseline.py:109" * 70)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
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
        f"Testing samples:  "
        f"{len(X_test):,}"
    )

    print()

    print(
        f"Training fraud rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Testing fraud rate:  "
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
# SCALE FEATURES
# ============================================================

def scale_features(
    X_train,
    X_test,
):

    print("= - train_baseline.py:161" * 70)
    print("2. FEATURE SCALING - train_baseline.py:162")
    print("= - train_baseline.py:163" * 70)
    print()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "[PASS] StandardScaler fitted on training data."
    )

    print(
        "[PASS] Test data transformed using the same scaler."
    )

    print()

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    print("= - train_baseline.py:202" * 70)
    print("3. MODEL TRAINING - train_baseline.py:203")
    print("= - train_baseline.py:204" * 70)
    print()

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "[PASS] Logistic Regression trained."
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

    print("= - train_baseline.py:238" * 70)
    print("4. MODEL EVALUATION - train_baseline.py:239")
    print("= - train_baseline.py:240" * 70)
    print()

    # Probability of fraud
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Default classification threshold
    y_prediction = (
        y_probability >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability,
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
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_prediction,
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
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_names,
):

    print("= - train_baseline.py:383" * 70)
    print("5. TOP MODEL FEATURES - train_baseline.py:384")
    print("= - train_baseline.py:385" * 70)
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
        "Top 15 features:"
    )

    print()

    for _, row in importance.head(15).iterrows():

        direction = (
            "↑ fraud risk"
            if row["coefficient"] > 0
            else "↓ fraud risk"
        )

        print(
            f"{row['feature']:<40} "
            f"{row['coefficient']:>10.4f}  "
            f"{direction}"
        )

    print()

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    scaler,
):

    print("= - train_baseline.py:441" * 70)
    print("6. SAVING MODEL - train_baseline.py:442")
    print("= - train_baseline.py:443" * 70)
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
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        f"Scaler saved to:"
    )

    print(
        SCALER_PATH
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # Load
    X, y = load_data()

    # Split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    # Scale
    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_test,
    )

    # Train
    model = train_model(
        X_train_scaled,
        y_train,
    )

    # Evaluate
    evaluate_model(
        model,
        X_test_scaled,
        y_test,
    )

    # Feature importance
    show_feature_importance(
        model,
        X.columns,
    )

    # Save
    save_model(
        model,
        scaler,
    )

    print("= - train_baseline.py:532" * 70)
    print("BASELINE TRAINING COMPLETE - train_baseline.py:533")
    print("= - train_baseline.py:534" * 70)
    print()


if __name__ == "__main__":
    main()