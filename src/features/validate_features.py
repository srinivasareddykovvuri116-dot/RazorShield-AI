"""
RazorShield AI - Feature Validation

Validates the ML-ready feature matrix before model training.

Checks:
    1. Row alignment
    2. Missing values
    3. Infinite values
    4. Duplicate rows
    5. Target leakage
    6. Ground-truth metadata leakage
    7. Constant features
    8. Feature ranges
    9. Class distribution
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

FEATURE_PATH = (
    PROCESSED_DATA_DIR / "features.csv"
)

TARGET_PATH = (
    PROCESSED_DATA_DIR / "target.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - validate_features.py:49" * 70)
    print("RAZORSHIELD AI  FEATURE VALIDATION - validate_features.py:50")
    print("= - validate_features.py:51" * 70)
    print()

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found:\n{FEATURE_PATH}"
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
    )

    print(
        f"Features loaded: {X.shape[0]:,} rows × "
        f"{X.shape[1]:,} columns"
    )

    print(
        f"Target loaded:   {y.shape[0]:,} rows × "
        f"{y.shape[1]:,} columns"
    )

    print()

    return X, y


# ============================================================
# ROW ALIGNMENT
# ============================================================

def check_row_alignment(X, y):

    print("= - validate_features.py:93" * 70)
    print("1. ROW ALIGNMENT - validate_features.py:94")
    print("= - validate_features.py:95" * 70)

    print()

    passed = len(X) == len(y)

    print(
        f"Feature rows: {len(X):,}"
    )

    print(
        f"Target rows:  {len(y):,}"
    )

    print(
        f"Status: {'PASS' if passed else 'FAIL'}"
    )

    print()

    return passed


# ============================================================
# TARGET VALIDATION
# ============================================================

def check_target(y):

    print("= - validate_features.py:124" * 70)
    print("2. TARGET VALIDATION - validate_features.py:125")
    print("= - validate_features.py:126" * 70)

    print()

    if "is_fraud" not in y.columns:

        print(
            "[FAIL] Target column 'is_fraud' is missing."
        )

        print()

        return False

    target = y["is_fraud"]

    unique_values = sorted(
        target.dropna().unique().tolist()
    )

    print(
        f"Target values: {unique_values}"
    )

    valid_values = set(
        unique_values
    ).issubset({0, 1})

    print(
        f"Binary target: "
        f"{'PASS' if valid_values else 'FAIL'}"
    )

    print()

    counts = target.value_counts()

    for value, count in counts.items():

        percentage = (
            count / len(target)
        ) * 100

        label = (
            "Fraud"
            if value == 1
            else "Legitimate"
        )

        print(
            f"{label:<12}: "
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )

    print()

    return valid_values


# ============================================================
# MISSING VALUE CHECK
# ============================================================

def check_missing_values(X):

    print("= - validate_features.py:192" * 70)
    print("3. MISSING VALUE CHECK - validate_features.py:193")
    print("= - validate_features.py:194" * 70)

    print()

    missing = X.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print(
            "[PASS] No missing values found."
        )

        print()

        return True

    print(
        "[FAIL] Missing values detected:"
    )

    for column, count in missing.items():

        percentage = (
            count / len(X)
        ) * 100

        print(
            f"  {column}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()

    return False


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

def check_infinite_values(X):

    print("= - validate_features.py:241" * 70)
    print("4. INFINITE VALUE CHECK - validate_features.py:242")
    print("= - validate_features.py:243" * 70)

    print()

    numeric_X = X.select_dtypes(
        include=np.number
    )

    infinite_mask = np.isinf(
        numeric_X
    )

    infinite_count = int(
        infinite_mask.sum().sum()
    )

    if infinite_count == 0:

        print(
            "[PASS] No infinite values found."
        )

        print()

        return True

    print(
        f"[FAIL] Infinite values found: "
        f"{infinite_count:,}"
    )

    print()

    return False


# ============================================================
# DUPLICATE ROW CHECK
# ============================================================

def check_duplicate_rows(X):

    print("= - validate_features.py:285" * 70)
    print("5. DUPLICATE FEATURE ROW CHECK - validate_features.py:286")
    print("= - validate_features.py:287" * 70)

    print()

    duplicate_count = int(
        X.duplicated().sum()
    )

    print(
        f"Duplicate feature rows: "
        f"{duplicate_count:,}"
    )

    # Duplicate feature rows are not automatically
    # a fatal problem. Multiple transactions can
    # legitimately have identical feature values.
    #
    # Therefore this is informational.

    print(
        "[INFO] Duplicate rows are not automatically a failure."
    )

    print()

    return True


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_target_leakage(X):

    print("= - validate_features.py:321" * 70)
    print("6. TARGET LEAKAGE CHECK - validate_features.py:322")
    print("= - validate_features.py:323" * 70)

    print()

    suspicious_names = [
        "is_fraud",
        "fraud",
        "label",
        "target",
    ]

    suspicious_columns = []

    for column in X.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in suspicious_names
        ):

            suspicious_columns.append(
                column
            )

    if suspicious_columns:

        print(
            "[FAIL] Potential target-leakage columns:"
        )

        for column in suspicious_columns:
            print(
                f"  - {column}"
            )

        print()

        return False

    print(
        "[PASS] No obvious target-leakage columns."
    )

    print()

    return True


# ============================================================
# GROUND-TRUTH METADATA CHECK
# ============================================================

def check_ground_truth_leakage(X):

    print("= - validate_features.py:379" * 70)
    print("7. GROUNDTRUTH METADATA CHECK - validate_features.py:380")
    print("= - validate_features.py:381" * 70)

    print()

    forbidden_columns = [
        "fraud_type",
        "ring_id",
    ]

    leaked_columns = [
        column
        for column in forbidden_columns
        if column in X.columns
    ]

    if leaked_columns:

        print(
            "[FAIL] Ground-truth metadata found:"
        )

        for column in leaked_columns:
            print(
                f"  - {column}"
            )

        print()

        return False

    print(
        "[PASS] fraud_type and ring_id are excluded."
    )

    print()

    return True


# ============================================================
# NON-NUMERIC CHECK
# ============================================================

def check_data_types(X):

    print("= - validate_features.py:426" * 70)
    print("8. FEATURE DATA TYPE CHECK - validate_features.py:427")
    print("= - validate_features.py:428" * 70)

    print()

    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if not non_numeric:

        print(
            "[PASS] All features are numeric."
        )

        print()

        return True

    print(
        "[FAIL] Non-numeric features found:"
    )

    for column in non_numeric:

        print(
            f"  - {column}"
        )

    print()

    return False


# ============================================================
# CONSTANT FEATURE CHECK
# ============================================================

def check_constant_features(X):

    print("= - validate_features.py:467" * 70)
    print("9. CONSTANT FEATURE CHECK - validate_features.py:468")
    print("= - validate_features.py:469" * 70)

    print()

    unique_counts = X.nunique(
        dropna=False
    )

    constant_columns = unique_counts[
        unique_counts <= 1
    ].index.tolist()

    if not constant_columns:

        print(
            "[PASS] No constant features."
        )

        print()

        return True

    print(
        "[WARNING] Constant features:"
    )

    for column in constant_columns:

        print(
            f"  - {column}"
        )

    print()

    return True


# ============================================================
# FEATURE RANGE CHECK
# ============================================================

def check_feature_ranges(X):

    print("= - validate_features.py:512" * 70)
    print("10. FEATURE RANGE CHECK - validate_features.py:513")
    print("= - validate_features.py:514" * 70)

    print()

    numeric_X = X.select_dtypes(
        include=np.number
    )

    problematic = []

    for column in numeric_X.columns:

        minimum = numeric_X[
            column
        ].min()

        maximum = numeric_X[
            column
        ].max()

        if not np.isfinite(
            minimum
        ) or not np.isfinite(
            maximum
        ):

            problematic.append(
                column
            )

    if not problematic:

        print(
            "[PASS] Numeric feature ranges are finite."
        )

    else:

        print(
            "[FAIL] Problematic numeric ranges:"
        )

        for column in problematic:

            print(
                f"  - {column}"
            )

    print()

    return not problematic


# ============================================================
# FEATURE SUMMARY
# ============================================================

def feature_summary(X):

    print("= - validate_features.py:573" * 70)
    print("11. FEATURE SUMMARY - validate_features.py:574")
    print("= - validate_features.py:575" * 70)

    print()

    summary = pd.DataFrame({

        "feature":
            X.columns,

        "dtype":
            X.dtypes.astype(str).values,

        "unique_values":
            [
                X[column].nunique()
                for column in X.columns
            ],

        "missing":
            [
                X[column].isnull().sum()
                for column in X.columns
            ],

        "mean":
            [
                X[column].mean()
                for column in X.columns
            ],

        "std":
            [
                X[column].std()
                for column in X.columns
            ],
    })

    print(
        summary.to_string(
            index=False
        )
    )

    print()


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

def target_correlation(X, y):

    print("= - validate_features.py:627" * 70)
    print("12. FEATURE / TARGET CORRELATION - validate_features.py:628")
    print("= - validate_features.py:629" * 70)

    print()

    target = y[
        "is_fraud"
    ]

    numeric_X = X.select_dtypes(
        include=np.number
    )

    correlations = (
        numeric_X
        .corrwith(target)
        .sort_values(
            ascending=False
        )
    )

    print(
        "Top positive correlations:"
    )

    print(
        correlations
        .tail(10)
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print()

    print(
        "Top negative correlations:"
    )

    print(
        correlations
        .head(10)
        .sort_values()
        .to_string()
    )

    print()

    print(
        "[INFO] High correlation is not automatically leakage."
    )

    print()


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(results):

    print("= - validate_features.py:690" * 70)
    print("13. FINAL VALIDATION RESULT - validate_features.py:691")
    print("= - validate_features.py:692" * 70)

    print()

    passed = 0
    total = len(results)

    for name, result in results.items():

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"[{status}] {name}"
        )

        if result:
            passed += 1

    print()

    print(
        f"Validation checks passed: "
        f"{passed}/{total}"
    )

    print()

    if passed == total:

        print(
            "RESULT: READY FOR MODEL TRAINING"
        )

    else:

        print(
            "RESULT: FIX FAILURES BEFORE MODEL TRAINING"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    X, y = load_data()

    results = {}

    results[
        "Row alignment"
    ] = check_row_alignment(
        X,
        y,
    )

    results[
        "Target validation"
    ] = check_target(
        y
    )

    results[
        "Missing values"
    ] = check_missing_values(
        X
    )

    results[
        "Infinite values"
    ] = check_infinite_values(
        X
    )

    results[
        "Duplicate rows"
    ] = check_duplicate_rows(
        X
    )

    results[
        "Target leakage"
    ] = check_target_leakage(
        X
    )

    results[
        "Ground-truth leakage"
    ] = check_ground_truth_leakage(
        X
    )

    results[
        "Numeric data types"
    ] = check_data_types(
        X
    )

    results[
        "Constant features"
    ] = check_constant_features(
        X
    )

    results[
        "Feature ranges"
    ] = check_feature_ranges(
        X
    )

    feature_summary(
        X
    )

    target_correlation(
        X,
        y,
    )

    final_validation(
        results
    )


if __name__ == "__main__":
    main()