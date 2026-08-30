"""
RazorShield AI - Champion Risk Scoring Engine

Production-style risk engine using the trained
HistGradientBoosting fraud model.

Decision policy:

    score < 0.125          -> ALLOW
    0.125 <= score < 0.250 -> REVIEW
    score >= 0.250         -> BLOCK
"""


from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_fraud_model.joblib"
)


FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features.csv"
)


POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "risk_policy.json"
)


# ============================================================
# RISK SCORER
# ============================================================

class RiskScorer:

    def __init__(
        self,
        model_path=MODEL_PATH,
        policy_path=POLICY_PATH,
    ):

        self.model_path = Path(
            model_path
        )

        self.policy_path = Path(
            policy_path
        )

        self.model = None
        self.policy = None
        self.feature_names = None

        self.load_policy()
        self.load_model()
        self.load_feature_schema()


    # ========================================================
    # LOAD POLICY
    # ========================================================

    def load_policy(self):

        if not self.policy_path.exists():

            raise FileNotFoundError(
                f"Risk policy not found:\n"
                f"{self.policy_path}"
            )

        with open(
            self.policy_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.policy = json.load(
                file
            )

        thresholds = self.policy[
            "risk_thresholds"
        ]

        self.review_threshold = float(
            thresholds["allow_max"]
        )

        self.block_threshold = float(
            thresholds["review_max"]
        )


    # ========================================================
    # LOAD CHAMPION MODEL
    # ========================================================

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Gradient Boosting model not found:\n"
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )


    # ========================================================
    # LOAD FEATURE SCHEMA
    # ========================================================

    def load_feature_schema(self):

        if not FEATURE_PATH.exists():

            raise FileNotFoundError(
                f"Feature dataset not found:\n"
                f"{FEATURE_PATH}"
            )

        features = pd.read_csv(
            FEATURE_PATH,
            nrows=1,
        )

        self.feature_names = (
            features.columns.tolist()
        )


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def prepare_features(
        self,
        transaction,
    ):

        if isinstance(
            transaction,
            dict,
        ):

            data = pd.DataFrame(
                [transaction]
            )

        elif isinstance(
            transaction,
            pd.DataFrame,
        ):

            data = transaction.copy()

        else:

            raise TypeError(
                "Transaction must be a dictionary "
                "or pandas DataFrame."
            )


        # ----------------------------------------------------
        # Add missing features
        # ----------------------------------------------------

        for feature in self.feature_names:

            if feature not in data.columns:

                data[feature] = 0


        # ----------------------------------------------------
        # Keep exact training order
        # ----------------------------------------------------

        data = data[
            self.feature_names
        ]


        # ----------------------------------------------------
        # Convert everything to numeric
        # ----------------------------------------------------

        data = data.apply(
            pd.to_numeric,
            errors="coerce",
        )


        # ----------------------------------------------------
        # Remove invalid values
        # ----------------------------------------------------

        data = data.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )


        data = data.fillna(
            0
        )

        return data


    # ========================================================
    # PREDICT FRAUD PROBABILITY
    # ========================================================

    def predict_probability(
        self,
        transaction,
    ):

        X = self.prepare_features(
            transaction
        )

        probability = (
            self.model
            .predict_proba(X)
            [:, 1][0]
        )

        return float(
            probability
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    def get_risk_level(
        self,
        probability,
    ):

        levels = self.policy[
            "risk_levels"
        ]


        if probability >= self.block_threshold:

            return levels[
                "critical"
            ]


        if probability >= self.review_threshold:

            return levels[
                "high"
            ]


        return levels[
            "low"
        ]


    # ========================================================
    # DECISION
    # ========================================================

    def get_decision(
        self,
        probability,
    ):

        decisions = self.policy[
            "decisions"
        ]


        if probability >= self.block_threshold:

            return decisions[
                "block"
            ]


        if probability >= self.review_threshold:

            return decisions[
                "review"
            ]


        return decisions[
            "allow"
        ]


    # ========================================================
    # EXPLANATION
    # ========================================================

    def explain_transaction(
        self,
        transaction,
        probability=None,
    ):
        """
        Generate deterministic, rule-based explanations
        for the model-generated risk decision.

        LOW / ALLOW transactions intentionally suppress
        weak individual signals so that harmless background
        network relationships do not appear as fraud warnings.
        """

        if isinstance(
            transaction,
            dict,
        ):

            data = transaction

        else:

            data = (
                transaction
                .iloc[0]
                .to_dict()
            )


        # ----------------------------------------------------
        # LOW-RISK TRANSACTIONS
        # ----------------------------------------------------

        # Do not display weak individual signals for
        # transactions whose model probability is below
        # the review threshold.

        if (
            probability is not None
            and probability < self.review_threshold
        ):

            return [
                "No major behavioral risk signal detected."
            ]


        reasons = []


        # ----------------------------------------------------
        # Amount anomaly
        # ----------------------------------------------------

        try:

            ratio = float(
                data.get(
                    "amount_vs_customer_history",
                    0,
                )
            )


            if ratio >= 5:

                reasons.append(
                    "Transaction amount is more than "
                    "5x the customer's historical average."
                )


            elif ratio >= 3:

                reasons.append(
                    "Transaction amount is significantly "
                    "above the customer's historical behavior."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # New location
        # ----------------------------------------------------

        if data.get(
            "is_new_location",
            0,
        ) == 1:

            reasons.append(
                "Transaction originates from "
                "a new customer location."
            )


        # ----------------------------------------------------
        # Location change
        # ----------------------------------------------------

        if data.get(
            "location_changed_from_previous",
            0,
        ) == 1:

            reasons.append(
                "Customer transaction location "
                "changed from the previous transaction."
            )


        # ----------------------------------------------------
        # Shared IP
        # ----------------------------------------------------

        try:

            ip_customers = float(
                data.get(
                    "ip_customer_count_before",
                    0,
                )
            )


            if ip_customers >= 3:

                reasons.append(
                    "IP address is associated with "
                    "multiple customers."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # Shared device
        # ----------------------------------------------------

        try:

            device_customers = float(
                data.get(
                    "device_customer_count_before",
                    0,
                )
            )


            if device_customers >= 3:

                reasons.append(
                    "Device is associated with "
                    "multiple customers."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # Network connections
        # ----------------------------------------------------

        try:

            connections = float(
                data.get(
                    "network_connections_before",
                    0,
                )
            )


            if connections >= 5:

                reasons.append(
                    "Transaction has multiple "
                    "previous network connections."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        try:

            velocity = float(
                data.get(
                    "customer_txn_count_1h_before",
                    0,
                )
            )


            if velocity >= 5:

                reasons.append(
                    "Unusually high transaction "
                    "velocity detected within one hour."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # High-value transaction
        # ----------------------------------------------------

        try:

            amount = float(
                data.get(
                    "amount",
                    0,
                )
            )


            if amount >= 25000:

                reasons.append(
                    "High-value transaction detected."
                )

        except (
            TypeError,
            ValueError,
        ):

            pass


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not reasons:

            reasons.append(
                "No major behavioral risk signal detected."
            )


        return reasons


    # ========================================================
    # COMPLETE ASSESSMENT
    # ========================================================

    def assess(
        self,
        transaction,
    ):

        # ----------------------------------------------------
        # Fraud probability
        # ----------------------------------------------------

        probability = (
            self.predict_probability(
                transaction
            )
        )


        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        risk_level = (
            self.get_risk_level(
                probability
            )
        )


        # ----------------------------------------------------
        # Operational decision
        # ----------------------------------------------------

        decision = (
            self.get_decision(
                probability
            )
        )


        # ----------------------------------------------------
        # Explain decision
        # ----------------------------------------------------

        reasons = (
            self.explain_transaction(
                transaction,
                probability,
            )
        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "risk_score":
                round(
                    probability,
                    4,
                ),

            "risk_percentage":
                round(
                    probability * 100,
                    2,
                ),

            "risk_level":
                risk_level,

            "decision":
                decision,

            "reasons":
                reasons,
        }


# ============================================================
# DEMO TRANSACTION
# ============================================================

def demo():

    print(
        "=" * 70
    )

    print(
        "RAZORSHIELD AI  CHAMPION RISK ENGINE"
    )

    print(
        "=" * 70
    )

    print()


    scorer = RiskScorer()


    print(
        "[PASS] Gradient Boosting model loaded."
    )

    print(
        "[PASS] Risk policy loaded."
    )

    print()


    print(
        "Configured policy:"
    )


    print(
        f"  ALLOW  < "
        f"{scorer.review_threshold:.3f}"
    )


    print(
        f"  REVIEW "
        f"{scorer.review_threshold:.3f}"
        f" - "
        f"{scorer.block_threshold:.3f}"
    )


    print(
        f"  BLOCK  >= "
        f"{scorer.block_threshold:.3f}"
    )

    print()


    # ========================================================
    # SUSPICIOUS / HIGH-RISK DEMO TRANSACTION
    # ========================================================

    suspicious_transaction = {

        "amount": 45000,

        "amount_log":
            np.log1p(45000),

        "amount_vs_customer_history":
            7.5,

        "is_new_location":
            1,

        "location_changed_from_previous":
            1,

        "ip_customer_count_before":
            5,

        "device_customer_count_before":
            4,

        "device_transactions_before":
            12,

        "ip_transactions_before":
            15,

        "network_connections_before":
            9,

        "customer_transactions_before":
            8,

        "customer_txn_count_1h_before":
            7,

        "customer_txn_count_24h_before":
            12,

        "merchant_transactions_before":
            650,

        "customer_amount_sum_before":
            60000,

        "customer_avg_amount_before":
            6000,

        "hour_of_day":
            3,

        "day_of_week":
            2,

        "is_late_night":
            1,

        "is_weekend":
            0,

        "is_high_value":
            1,

        "is_very_high_value":
            1,

        "device_other_customers_before":
            3,

        "ip_other_customers_before":
            4,

        "device_ip_transactions_before":
            6,
    }


    result = scorer.assess(
        suspicious_transaction
    )


    print(
        "=" * 70
    )

    print(
        "RISK ASSESSMENT"
    )

    print(
        "=" * 70
    )

    print()


    print(
        f"Risk Score : "
        f"{result['risk_score']}"
    )


    print(
        f"Risk       : "
        f"{result['risk_percentage']}%"
    )


    print(
        f"Risk Level : "
        f"{result['risk_level']}"
    )


    print(
        f"Decision   : "
        f"{result['decision']}"
    )

    print()


    print(
        "Risk Reasons:"
    )


    for index, reason in enumerate(
        result["reasons"],
        start=1,
    ):

        print(
            f"  {index}. {reason}"
        )


    print()


    print(
        "=" * 70
    )

    print(
        "CHAMPION RISK ENGINE TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    demo()