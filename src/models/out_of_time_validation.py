"""
RazorShield AI - Out-of-Time Validation

Purpose:
    Evaluate the fraud model using a chronological split so that
    training data comes strictly before evaluation data.

This is a separate robustness experiment.

It does NOT replace the existing champion model or its current
held-out evaluation metrics.

Important:
    Temporal-safe feature engineering is performed before this
    evaluation. The raw transaction timestamps are used only to
    establish chronological train/test boundaries.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transactions.csv"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features.csv"
)

TARGET_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_target.csv"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "out_of_time_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

BLOCK_THRESHOLD = 0.25


# ============================================================
# LOAD RAW TRANSACTION TIMESTAMPS
# ============================================================

def load_raw_transactions():

    print("= - out_of_time_validation.py:89" * 70)
    print("RAZORSHIELD AI  OUTOFTIME VALIDATION - out_of_time_validation.py:90")
    print("= - out_of_time_validation.py:91" * 70)
    print()

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Raw transaction file not found:\n"
            f"{RAW_DATA_PATH}"
        )

    raw = pd.read_csv(
        RAW_DATA_PATH
    )

    if "timestamp" not in raw.columns:

        raise ValueError(
            "Raw transaction dataset does not contain "
            "a timestamp column."
        )

    if "transaction_id" not in raw.columns:

        raise ValueError(
            "Raw transaction dataset does not contain "
            "transaction_id."
        )

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"]
    )

    raw = raw.sort_values(
        [
            "timestamp",
            "transaction_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Raw transactions: "
        f"{len(raw):,}"
    )

    print(
        f"Earliest timestamp: "
        f"{raw['timestamp'].min()}"
    )

    print(
        f"Latest timestamp: "
        f"{raw['timestamp'].max()}"
    )

    print()

    return raw


# ============================================================
# LOAD PROCESSED FEATURES
# ============================================================

def load_processed_data():

    if not FEATURE_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURE_DATA_PATH}"
        )

    if not TARGET_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Target dataset not found:\n"
            f"{TARGET_DATA_PATH}"
        )

    X = pd.read_csv(
        FEATURE_DATA_PATH
    )

    y = pd.read_csv(
        TARGET_DATA_PATH
    )["is_fraud"]

    if len(X) != len(y):

        raise ValueError(
            "Feature and target row counts do not match."
        )

    print(
        f"Processed features: "
        f"{len(X):,}"
    )

    print(
        f"Feature count: "
        f"{X.shape[1]}"
    )

    print()

    return X, y


# ============================================================
# VERIFY ORDER ALIGNMENT
# ============================================================

def verify_alignment(
    raw,
    X,
    y,
):

    print("= - out_of_time_validation.py:211" * 70)
    print("1. VERIFYING CHRONOLOGICAL ALIGNMENT - out_of_time_validation.py:212")
    print("= - out_of_time_validation.py:213" * 70)
    print()

    if len(raw) != len(X):

        raise ValueError(
            "Raw transaction count and processed feature "
            "count do not match.\n"
            f"Raw: {len(raw):,}\n"
            f"Features: {len(X):,}"
        )

    # The temporal feature builder sorts the raw data before
    # constructing features. Verify that the processed target
    # has the same fraud count as the chronologically sorted raw
    # dataset.

    raw_fraud_count = int(
        raw["is_fraud"].sum()
    )

    processed_fraud_count = int(
        y.sum()
    )

    print(
        f"Raw fraud records: "
        f"{raw_fraud_count:,}"
    )

    print(
        f"Processed fraud records: "
        f"{processed_fraud_count:,}"
    )

    if raw_fraud_count != processed_fraud_count:

        raise ValueError(
            "Fraud counts do not match between raw and "
            "processed datasets."
        )

    print(
        "[PASS] Dataset counts are aligned."
    )

    print()

    return True


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    raw,
    X,
    y,
):

    print("= - out_of_time_validation.py:274" * 70)
    print("2. CREATING CHRONOLOGICAL SPLIT - out_of_time_validation.py:275")
    print("= - out_of_time_validation.py:276" * 70)
    print()

    total = len(raw)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = int(
        total
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    # --------------------------------------------------------
    # Train = earliest 70%
    # Validation = next 15%
    # Test = latest 15%
    # --------------------------------------------------------

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    X_validation = X.iloc[
        train_end:validation_end
    ].copy()

    y_validation = y.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[
        validation_end:
    ].copy()

    y_test = y.iloc[
        validation_end:
    ].copy()

    train_times = raw.iloc[
        :train_end
    ]["timestamp"]

    validation_times = raw.iloc[
        train_end:validation_end
    ]["timestamp"]

    test_times = raw.iloc[
        validation_end:
    ]["timestamp"]

    print(
        "TRAIN"
    )

    print(
        f"Rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Time range: "
        f"{train_times.min()} → "
        f"{train_times.max()}"
    )

    print(
        f"Fraud: "
        f"{int(y_train.sum()):,}"
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        f"Rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"Time range: "
        f"{validation_times.min()} → "
        f"{validation_times.max()}"
    )

    print(
        f"Fraud: "
        f"{int(y_validation.sum()):,}"
    )

    print()

    print(
        "FUTURE TEST"
    )

    print(
        f"Rows: "
        f"{len(X_test):,}"
    )

    print(
        f"Time range: "
        f"{test_times.min()} → "
        f"{test_times.max()}"
    )

    print(
        f"Fraud: "
        f"{int(y_test.sum()):,}"
    )

    print()

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        train_times,
        validation_times,
        test_times,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print("= - out_of_time_validation.py:420" * 70)
    print("3. CREATING OUTOFTIME MODEL - out_of_time_validation.py:421")
    print("= - out_of_time_validation.py:422" * 70)
    print()

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print(
        "Model: HistGradientBoostingClassifier"
    )

    print(
        "This model is trained only on the "
        "chronologically earlier training period."
    )

    print()

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold=BLOCK_THRESHOLD,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_legitimate = (
        tn + fp
    )

    false_positive_rate = (
        fp / total_legitimate
        if total_legitimate > 0
        else 0.0
    )

    return {
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "transactions": len(y),
        "fraud_transactions": int(y.sum()),
        "threshold": threshold,
    }


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_evaluation(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):

    model = create_model()

    print("= - out_of_time_validation.py:538" * 70)
    print("4. TRAINING ON EARLIER TRANSACTIONS - out_of_time_validation.py:539")
    print("= - out_of_time_validation.py:540" * 70)
    print()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "[PASS] Model trained."
    )

    print()

    # --------------------------------------------------------
    # Validation period
    # --------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # Future test period
    # --------------------------------------------------------

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    return (
        model,
        validation_metrics,
        test_metrics,
    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics,
):

    print("= - out_of_time_validation.py:590" * 70)
    print(name)
    print("= - out_of_time_validation.py:592" * 70)
    print()

    print(
        f"Transactions:       "
        f"{metrics['transactions']:,}"
    )

    print(
        f"Fraud transactions: "
        f"{metrics['fraud_transactions']:,}"
    )

    print(
        f"PR-AUC:             "
        f"{metrics['pr_auc'] * 100:.2f}%"
    )

    print(
        f"Precision:           "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:              "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:            "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        f"  True Negatives:  {metrics['true_negatives']:,}"
    )

    print(
        f"  False Positives: {metrics['false_positives']:,}"
    )

    print(
        f"  False Negatives: {metrics['false_negatives']:,}"
    )

    print(
        f"  True Positives:  {metrics['true_positives']:,}"
    )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    validation_metrics,
    test_metrics,
    train_times,
    validation_times,
    test_times,
):

    rows = [
        {
            "evaluation_period": "validation",
            "start_timestamp":
                str(validation_times.min()),
            "end_timestamp":
                str(validation_times.max()),
            **validation_metrics,
        },
        {
            "evaluation_period": "future_test",
            "start_timestamp":
                str(test_times.min()),
            "end_timestamp":
                str(test_times.max()),
            **test_metrics,
        },
    ]

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    print("= - out_of_time_validation.py:695" * 70)
    print("5. SAVING OUTOFTIME RESULTS - out_of_time_validation.py:696")
    print("= - out_of_time_validation.py:697" * 70)
    print()

    print(
        f"Saved to:\n"
        f"{OUTPUT_CSV_PATH}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load raw timestamps
    # --------------------------------------------------------

    raw = load_raw_transactions()

    # --------------------------------------------------------
    # Load temporal features and target
    # --------------------------------------------------------

    X, y = load_processed_data()

    # --------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------

    verify_alignment(
        raw,
        X,
        y,
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        train_times,
        validation_times,
        test_times,
    ) = chronological_split(
        raw,
        X,
        y,
    )

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    (
        model,
        validation_metrics,
        test_metrics,
    ) = run_evaluation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_metrics(
        "CHRONOLOGICAL VALIDATION PERIOD",
        validation_metrics,
    )

    print_metrics(
        "FUTURE OUT-OF-TIME TEST PERIOD",
        test_metrics,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        validation_metrics,
        test_metrics,
        train_times,
        validation_times,
        test_times,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("= - out_of_time_validation.py:803" * 70)
    print("6. TEMPORAL ROBUSTNESS CHECK - out_of_time_validation.py:804")
    print("= - out_of_time_validation.py:805" * 70)
    print()

    print(
        "The future test period was not used during "
        "training."
    )

    print()

    print(
        f"Future-test PR-AUC: "
        f"{test_metrics['pr_auc'] * 100:.2f}%"
    )

    print(
        f"Future-test Precision: "
        f"{test_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Future-test Recall: "
        f"{test_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Future-test F1: "
        f"{test_metrics['f1'] * 100:.2f}%"
    )

    print()

    print(
        "This experiment is a separate temporal "
        "robustness check and does not replace the "
        "existing champion-model evaluation."
    )

    print()

    print("= - out_of_time_validation.py:845" * 70)
    print("OUTOFTIME VALIDATION COMPLETE - out_of_time_validation.py:846")
    print("= - out_of_time_validation.py:847" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()