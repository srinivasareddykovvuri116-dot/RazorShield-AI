"""
RazorShield AI - Find Real Decision Examples

Finds real validation transactions for:

    ALLOW
    REVIEW
    BLOCK

This is used to verify that our decision policy
has realistic examples for the final demo.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_target.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_fraud_model.joblib"
)


# ============================================================
# THRESHOLDS
# ============================================================

REVIEW_THRESHOLD = 0.125
BLOCK_THRESHOLD = 0.250


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - find_decision_examples.py:73" * 70)
    print("RAZORSHIELD AI  DECISION EXAMPLES - find_decision_examples.py:74")
    print("= - find_decision_examples.py:75" * 70)
    print()

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_PATH}"
        )

    if not TARGET_PATH.exists():

        raise FileNotFoundError(
            f"Target dataset not found:\n{TARGET_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
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
        f"Loaded {len(X):,} transactions."
    )

    print()

    return X, y, model


# ============================================================
# PREDICT
# ============================================================

def predict(
    X,
    model,
):

    print("= - find_decision_examples.py:126" * 70)
    print("1. GENERATING RISK SCORES - find_decision_examples.py:127")
    print("= - find_decision_examples.py:128" * 70)
    print()

    probabilities = model.predict_proba(
        X
    )[:, 1]

    print(
        "[PASS] Risk scores generated."
    )

    print()

    return probabilities


# ============================================================
# CLASSIFY
# ============================================================

def classify(
    probability,
):

    if probability < REVIEW_THRESHOLD:

        return "ALLOW"

    if probability < BLOCK_THRESHOLD:

        return "REVIEW"

    return "BLOCK"


# ============================================================
# FIND EXAMPLES
# ============================================================

def find_examples(
    X,
    y,
    probabilities,
):

    print("= - find_decision_examples.py:173" * 70)
    print("2. SEARCHING FOR REAL EXAMPLES - find_decision_examples.py:174")
    print("= - find_decision_examples.py:175" * 70)
    print()

    decisions = [
        classify(
            probability
        )
        for probability in probabilities
    ]

    decisions = pd.Series(
        decisions
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "Decision distribution:"
    )

    print(
        decisions.value_counts()
        .to_string()
    )

    print()

    # --------------------------------------------------------
    # Search ranges
    # --------------------------------------------------------

    allow_indices = (
        probabilities < REVIEW_THRESHOLD
    )

    review_indices = (
        (probabilities >= REVIEW_THRESHOLD)
        &
        (probabilities < BLOCK_THRESHOLD)
    )

    block_indices = (
        probabilities >= BLOCK_THRESHOLD
    )

    print(
        f"ALLOW examples : "
        f"{allow_indices.sum():,}"
    )

    print(
        f"REVIEW examples: "
        f"{review_indices.sum():,}"
    )

    print(
        f"BLOCK examples : "
        f"{block_indices.sum():,}"
    )

    print()

    return (
        allow_indices,
        review_indices,
        block_indices,
    )


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

def display_examples(
    X,
    y,
    probabilities,
    mask,
    decision,
    count=5,
):

    indices = (
        X.index[mask]
        .tolist()
    )

    if not indices:

        print(
            f"[WARNING] No {decision} examples found."
        )

        return

    # --------------------------------------------------------
    # For REVIEW we prefer scores close to the middle
    # of the range.
    #
    # For ALLOW and BLOCK we select examples closer
    # to their respective extremes.
    # --------------------------------------------------------

    selected = []

    if decision == "REVIEW":

        middle = (
            REVIEW_THRESHOLD
            + BLOCK_THRESHOLD
        ) / 2

        indices = sorted(
            indices,
            key=lambda index:
                abs(
                    probabilities[index]
                    - middle
                ),
        )

    elif decision == "ALLOW":

        indices = sorted(
            indices,
            key=lambda index:
                probabilities[index],
        )

    else:

        indices = sorted(
            indices,
            key=lambda index:
                probabilities[index],
            reverse=True,
        )

    selected = indices[:count]

    print("= - find_decision_examples.py:317" * 70)
    print(
        f"REAL {decision} EXAMPLES"
    )
    print("= - find_decision_examples.py:321" * 70)
    print()

    for number, index in enumerate(
        selected,
        start=1,
    ):

        probability = (
            probabilities[index]
        )

        actual = int(
            y.iloc[index]
        )

        print(
            f"{number}. Dataset index: {index}"
        )

        print(
            f"   Risk score : "
            f"{probability:.6f}"
        )

        print(
            f"   Risk       : "
            f"{probability * 100:.2f}%"
        )

        print(
            f"   Decision   : "
            f"{decision}"
        )

        print(
            f"   Actual label: "
            f"{'FRAUD' if actual == 1 else 'LEGITIMATE'}"
        )

        # Display important signals when available.

        important_features = [
            "amount",
            "amount_vs_customer_history",
            "is_new_location",
            "location_changed_from_previous",
            "ip_customer_count_before",
            "device_customer_count_before",
            "network_connections_before",
            "customer_txn_count_1h_before",
            "customer_txn_count_24h_before",
            "is_late_night",
            "is_high_value",
        ]

        print(
            "   Signals:"
        )

        for feature in important_features:

            if feature in X.columns:

                value = X.iloc[
                    index
                ][feature]

                print(
                    f"      {feature}: {value}"
                )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    X, y, model = load_data()

    probabilities = predict(
        X,
        model,
    )

    (
        allow_mask,
        review_mask,
        block_mask,
    ) = find_examples(
        X,
        y,
        probabilities,
    )

    display_examples(
        X,
        y,
        probabilities,
        allow_mask,
        "ALLOW",
        count=3,
    )

    display_examples(
        X,
        y,
        probabilities,
        review_mask,
        "REVIEW",
        count=5,
    )

    display_examples(
        X,
        y,
        probabilities,
        block_mask,
        "BLOCK",
        count=3,
    )

    print("= - find_decision_examples.py:446" * 70)
    print(
        "       DECISION EXAMPLE SEARCH COMPLETE"
    )
    print("= - find_decision_examples.py:450" * 70)
    print()


if __name__ == "__main__":

    main()