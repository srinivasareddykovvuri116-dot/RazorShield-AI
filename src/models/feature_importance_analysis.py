"""
RazorShield AI - Feature Importance Analysis

Uses permutation importance on the existing Gradient Boosting
champion model.

This script:
    - Does NOT retrain the model
    - Recreates the same held-out 20% test set
    - Calculates permutation importance
    - Produces a ranked feature-importance CSV
    - Produces a top-10 feature-importance chart

Important:
Permutation importance measures how much model performance
changes when a feature's values are randomly permuted. It is
an importance measure, not a causal explanation.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_importance.csv"
)

OUTPUT_PNG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_importance_top10.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Number of random permutations per feature.
N_REPEATS = 10

# We calculate importance on the held-out test set.
SCORING = "average_precision"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - feature_importance_analysis.py:91" * 70)
    print("RAZORSHIELD AI  FEATURE IMPORTANCE ANALYSIS - feature_importance_analysis.py:92")
    print("= - feature_importance_analysis.py:93" * 70)
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
            f"Model file not found:\n{MODEL_PATH}"
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
        f"Feature dataset: "
        f"{X.shape[0]:,} rows × "
        f"{X.shape[1]:,} columns"
    )

    print(
        f"Model type: "
        f"{type(model).__name__}"
    )

    print(
        f"Target rows: "
        f"{len(y):,}"
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

    print("= - feature_importance_analysis.py:156" * 70)
    print("1. RECREATING HELDOUT TEST SET - feature_importance_analysis.py:157")
    print("= - feature_importance_analysis.py:158" * 70)
    print()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Held-out test samples: "
        f"{len(X_test):,}"
    )

    print(
        f"Test fraud cases: "
        f"{int(y_test.sum()):,}"
    )

    print(
        f"Test legitimate cases: "
        f"{int((y_test == 0).sum()):,}"
    )

    print()

    return X_test, y_test


# ============================================================
# CALCULATE PERMUTATION IMPORTANCE
# ============================================================

def calculate_feature_importance(
    model,
    X_test,
    y_test,
):

    print("= - feature_importance_analysis.py:209" * 70)
    print("2. CALCULATING PERMUTATION IMPORTANCE - feature_importance_analysis.py:210")
    print("= - feature_importance_analysis.py:211" * 70)
    print()

    print(
        "Scoring metric: "
        f"{SCORING}"
    )

    print(
        f"Permutation repeats per feature: "
        f"{N_REPEATS}"
    )

    print(
        "This may take some time because the "
        "analysis uses the held-out test set."
    )

    print()

    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring=SCORING,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,

            "importance_mean":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    # --------------------------------------------------------
    # Convert to percentages for easier interpretation.
    #
    # Permutation importance is a performance change.
    # Negative values are possible and are preserved.
    # --------------------------------------------------------

    importance_df[
        "importance_percent"
    ] = (
        importance_df[
            "importance_mean"
        ]
        * 100
    )

    importance_df[
        "importance_std_percent"
    ] = (
        importance_df[
            "importance_std"
        ]
        * 100
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "[PASS] Permutation importance calculated."
    )

    print()

    return importance_df


# ============================================================
# PRINT TOP FEATURES
# ============================================================

def print_top_features(
    importance_df,
):

    print("= - feature_importance_analysis.py:304" * 70)
    print("3. TOP 10 FEATURES - feature_importance_analysis.py:305")
    print("= - feature_importance_analysis.py:306" * 70)
    print()

    top10 = (
        importance_df
        .head(10)
    )

    print(
        "Rank | Feature                                      | "
        "Importance | Std"
    )

    print("" * 75)

    for index, row in top10.iterrows():

        print(
            f"{index + 1:4d} | "
            f"{row['feature']:<44} | "
            f"{row['importance_percent']:>9.4f}% | "
            f"{row['importance_std_percent']:>6.4f}%"
        )

    print()


# ============================================================
# SAVE CSV
# ============================================================

def save_feature_importance(
    importance_df,
):

    print("= - feature_importance_analysis.py:341" * 70)
    print("4. SAVING FEATURE IMPORTANCE DATA - feature_importance_analysis.py:342")
    print("= - feature_importance_analysis.py:343" * 70)
    print()

    importance_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"Saved to:\n{OUTPUT_CSV}"
    )

    print()


# ============================================================
# CREATE CHART
# ============================================================

def create_chart(
    importance_df,
):

    print("= - feature_importance_analysis.py:366" * 70)
    print("5. CREATING TOP10 FEATURE CHART - feature_importance_analysis.py:367")
    print("= - feature_importance_analysis.py:368" * 70)
    print()

    top10 = (
        importance_df
        .head(10)
        .sort_values(
            "importance_mean",
            ascending=True,
        )
    )

    figure = plt.figure(
        figsize=(11, 7)
    )

    axis = figure.add_subplot(
        111
    )

    axis.barh(
        top10["feature"],
        top10["importance_percent"],
    )

    axis.set_xlabel(
        "Permutation Importance (% change in average precision)"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "RazorShield AI - Top 10 Feature Importance"
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PNG,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Chart saved to:\n{OUTPUT_PNG}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    importance_df,
):

    print("= - feature_importance_analysis.py:437" * 70)
    print("6. FEATURE IMPORTANCE SUMMARY - feature_importance_analysis.py:438")
    print("= - feature_importance_analysis.py:439" * 70)
    print()

    top10 = (
        importance_df
        .head(10)
    )

    positive_importance = (
        importance_df[
            importance_df[
                "importance_mean"
            ] > 0
        ]
    )

    print(
        f"Total features analyzed: "
        f"{len(importance_df)}"
    )

    print(
        f"Features with positive mean importance: "
        f"{len(positive_importance)}"
    )

    if not top10.empty:

        print(
            f"Most important feature: "
            f"{top10.iloc[0]['feature']}"
        )

        print(
            f"Mean permutation importance: "
            f"{top10.iloc[0]['importance_percent']:.4f}%"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "Permutation importance measures the reduction "
        "in held-out average precision when a feature "
        "is randomly permuted."
    )

    print()

    print(
        "Important: this is a model-importance measure, "
        "not a causal explanation."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load existing model/data
    # --------------------------------------------------------

    X, y, model = load_data()

    # --------------------------------------------------------
    # Recreate exact held-out test split
    # --------------------------------------------------------

    X_test, y_test = create_test_set(
        X,
        y,
    )

    # --------------------------------------------------------
    # Calculate permutation importance
    # --------------------------------------------------------

    importance_df = (
        calculate_feature_importance(
            model,
            X_test,
            y_test,
        )
    )

    # --------------------------------------------------------
    # Print ranking
    # --------------------------------------------------------

    print_top_features(
        importance_df,
    )

    # --------------------------------------------------------
    # Save complete ranking
    # --------------------------------------------------------

    save_feature_importance(
        importance_df,
    )

    # --------------------------------------------------------
    # Create top-10 chart
    # --------------------------------------------------------

    create_chart(
        importance_df,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        importance_df,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("= - feature_importance_analysis.py:568" * 70)
    print(
        "FEATURE IMPORTANCE ANALYSIS COMPLETE"
    )
    print("= - feature_importance_analysis.py:572" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()