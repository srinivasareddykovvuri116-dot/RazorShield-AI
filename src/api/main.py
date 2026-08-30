"""
RazorShield AI - FastAPI Backend

Provides:

    GET  /
    GET  /health
    GET  /api/v1/risk/policy
    GET  /api/v1/risk/history
    DELETE /api/v1/risk/history
    GET  /api/v1/demo/history
    GET  /api/v1/demo/transactions
    POST /api/v1/risk/assess
"""


from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# RISK ENGINE
# ============================================================

from src.risk_engine.risk_scorer import RiskScorer


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RazorShield AI",
    description=(
        "AI-powered payment risk assessment "
        "and fraud decisioning platform."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RISK ENGINE INITIALIZATION
# ============================================================

try:

    risk_scorer = RiskScorer()

    ENGINE_READY = True
    ENGINE_ERROR = None

except Exception as error:

    risk_scorer = None

    ENGINE_READY = False
    ENGINE_ERROR = str(error)


# ============================================================
# DECISION HISTORY
# ============================================================

decision_history = []


# ============================================================
# REQUEST MODEL
# ============================================================

class TransactionRequest(BaseModel):

    # Allow complete temporal feature vectors
    # from the demo dataset.
    model_config = {
        "extra": "allow"
    }

    # ========================================================
    # CORE TRANSACTION FEATURES
    # ========================================================

    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount.",
    )

    account_age_days: float = 0

    avg_transaction_amount: float = 0

    hour_of_day: int = Field(
        default=12,
        ge=0,
        le=23,
    )

    day_of_week: int = Field(
        default=0,
        ge=0,
        le=6,
    )

    day_of_month: int = 1

    is_weekend: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_late_night: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_business_hours: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    amount_log: float | None = None

    # ========================================================
    # AMOUNT FEATURES
    # ========================================================

    amount_vs_customer_avg: float = 0

    amount_difference_from_avg: float = 0

    amount_vs_historical_avg: float = 0

    amount_vs_customer_history: float = 0

    is_high_value: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_very_high_value: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    # ========================================================
    # CUSTOMER TEMPORAL FEATURES
    # ========================================================

    seconds_since_previous_customer_txn: float = 0

    seconds_since_customer_transaction: float = 0

    customer_transaction_count: float = 0

    customer_historical_avg_amount: float = 0

    customer_txn_count_1h: float = 0

    customer_txn_count_24h: float = 0

    customer_amount_1h: float = 0

    customer_amount_24h: float = 0

    # ========================================================
    # DEVICE FEATURES
    # ========================================================

    device_shared_accounts: float = 0

    device_transaction_count: float = 0

    device_reuse_score: float = 0

    is_shared_device: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_seen_device_before: int = 0

    device_customer_count: float = 0

    customer_device_count: float = 0

    device_connected_customers: float = 0

    device_ip_shared_accounts: float = 0

    # ========================================================
    # IP FEATURES
    # ========================================================

    ip_shared_accounts: float = 0

    ip_transaction_count: float = 0

    ip_reuse_score: float = 0

    is_shared_ip: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_seen_ip_before: int = 0

    ip_customer_count: float = 0

    customer_ip_count: float = 0

    ip_connected_customers: float = 0

    # ========================================================
    # LOCATION FEATURES
    # ========================================================

    is_new_location: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    customer_location_count: float = 0

    location_changed_from_previous: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    # ========================================================
    # NETWORK FEATURES
    # ========================================================

    network_connection_score: float = 0

    # ========================================================
    # MERCHANT FEATURES
    # ========================================================

    merchant_transaction_count: float = 0

    merchant_customer_count: float = 0

    merchant_customers_before: float = 0

    customer_payment_method_usage: float = 0

    # ========================================================
    # PAYMENT METHOD
    # ========================================================

    payment_method_card: int = 0

    payment_method_netbanking: int = 0

    payment_method_upi: int = 0

    payment_method_wallet: int = 0

    # ========================================================
    # DEVICE TYPE
    # ========================================================

    device_type_android: int = 0

    device_type_ios: int = 0

    device_type_web: int = 0

    # ========================================================
    # MERCHANT CATEGORY
    # ========================================================

    merchant_category_education: int = 0

    merchant_category_electronics: int = 0

    merchant_category_entertainment: int = 0

    merchant_category_fashion: int = 0

    merchant_category_food: int = 0

    merchant_category_grocery: int = 0

    merchant_category_healthcare: int = 0

    merchant_category_services: int = 0

    merchant_category_travel: int = 0

    merchant_category_utilities: int = 0

    # ========================================================
    # LOCATION ONE-HOT FEATURES
    # ========================================================

    location_Ahmedabad: int = 0

    location_Bengaluru: int = 0

    location_Chennai: int = 0

    location_Delhi: int = 0

    location_Hyderabad: int = 0

    location_Kolkata: int = 0

    location_Mumbai: int = 0

    location_Pune: int = 0

    location_Vijayawada: int = 0

    location_Visakhapatnam: int = 0

    # ========================================================
    # BACKWARD-COMPATIBILITY FIELDS
    # ========================================================

    device_transactions_before: float = 0

    ip_transactions_before: float = 0

    network_connections_before: float = 0

    customer_transactions_before: float = 0

    customer_txn_count_1h_before: float = 0

    customer_txn_count_24h_before: float = 0

    merchant_transactions_before: float = 0

    customer_amount_sum_before: float = 0

    customer_avg_amount_before: float = 0

    device_other_customers_before: float = 0

    ip_other_customers_before: float = 0

    device_ip_transactions_before: float = 0


# ============================================================
# RESPONSE MODEL
# ============================================================

class RiskResponse(BaseModel):

    risk_score: float

    risk_percentage: float

    risk_level: str

    decision: str

    reasons: list[str]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "RazorShield AI",
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if not ENGINE_READY:

        return {
            "status": "error",
            "risk_engine": "unavailable",
            "error": ENGINE_ERROR,
        }

    return {
        "status": "healthy",
        "risk_engine": "ready",
        "model": "gradient_boosting_fraud_model",
    }


# ============================================================
# RISK ASSESSMENT
# ============================================================

@app.post(
    "/api/v1/risk/assess",
    response_model=RiskResponse,
)
def assess_transaction(
    transaction: TransactionRequest,
):

    if not ENGINE_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Risk engine is unavailable: "
                f"{ENGINE_ERROR}"
            ),
        )

    try:

        transaction_data = transaction.model_dump(
            exclude_none=True
        )

        # ----------------------------------------------------
        # Generate amount_log from amount.
        #
        # This guarantees the value is available for
        # manual transactions as well.
        # ----------------------------------------------------

        transaction_data["amount_log"] = np.log1p(
            transaction_data["amount"]
        )

        # ----------------------------------------------------
        # Run AI risk engine
        # ----------------------------------------------------

        result = risk_scorer.assess(
            transaction_data
        )

        # ----------------------------------------------------
        # Store decision in history
        # ----------------------------------------------------

        history_entry = {

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "amount":
                float(
                    transaction_data["amount"]
                ),

            "risk_score":
                float(
                    result["risk_score"]
                ),

            "risk_percentage":
                float(
                    result["risk_percentage"]
                ),

            "risk_level":
                result["risk_level"],

            "decision":
                result["decision"],

            "reasons":
                result["reasons"],
        }

        # Newest first.
        decision_history.insert(
            0,
            history_entry
        )

        # Keep only the latest 100.
        del decision_history[100:]

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk assessment failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# RISK POLICY
# ============================================================

@app.get(
    "/api/v1/risk/policy"
)
def get_policy():

    if not ENGINE_READY:

        raise HTTPException(
            status_code=503,
            detail="Risk engine is unavailable.",
        )

    return {

        "model":
            "HistGradientBoostingClassifier",

        "thresholds": {

            "allow_max":
                risk_scorer.review_threshold,

            "review_max":
                risk_scorer.block_threshold,
        },

        "policy": {

            "allow":
                (
                    f"< "
                    f"{risk_scorer.review_threshold:.3f}"
                ),

            "review":
                (
                    f"{risk_scorer.review_threshold:.3f}"
                    f" - "
                    f"< "
                    f"{risk_scorer.block_threshold:.3f}"
                ),

            "block":
                (
                    f">= "
                    f"{risk_scorer.block_threshold:.3f}"
                ),
        },
    }


# ============================================================
# DECISION HISTORY
# ============================================================

@app.get(
    "/api/v1/risk/history"
)
def get_decision_history():

    return {

        "count":
            len(
                decision_history
            ),

        "decisions":
            decision_history,
    }


# ============================================================
# DEMO DECISION HISTORY
# ============================================================

@app.get(
    "/api/v1/demo/history"
)
def get_demo_history():

    return {
        "count": 8,
        "demo": True,
        "description": (
            "Representative synthetic scenarios for "
            "demonstrating different risk outcomes. "
            "These are separate from live decision history "
            "and held-out model evaluation."
        ),
        "decisions": [

            # ------------------------------------------------
            # 1. SAFE PAYMENT
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 11:03:24 pm",
                "amount": 526.60,
                "risk_score": 0.0001,
                "risk_percentage": 0.01,
                "risk_level": "LOW",
                "decision": "ALLOW",
                "signals": 1,
                "reasons": [
                    "No major behavioral risk signal detected."
                ],
            },

            # ------------------------------------------------
            # 2. NORMAL PAYMENT
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 11:02:51 pm",
                "amount": 1840.25,
                "risk_score": 0.031,
                "risk_percentage": 3.10,
                "risk_level": "LOW",
                "decision": "ALLOW",
                "signals": 1,
                "reasons": [
                    "Transaction behavior is consistent "
                    "with customer history."
                ],
            },

            # ------------------------------------------------
            # 3. REVIEW CASE
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 11:01:43 pm",
                "amount": 3610.98,
                "risk_score": 0.188,
                "risk_percentage": 18.80,
                "risk_level": "HIGH",
                "decision": "REVIEW",
                "signals": 3,
                "reasons": [
                    "IP address is associated with "
                    "multiple customers.",
                    "Device is associated with "
                    "multiple customers.",
                    "Transaction has multiple previous "
                    "network connections.",
                ],
            },

            # ------------------------------------------------
            # 4. HIGH VALUE
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 10:59:18 pm",
                "amount": 12850.00,
                "risk_score": 0.214,
                "risk_percentage": 21.40,
                "risk_level": "HIGH",
                "decision": "REVIEW",
                "signals": 2,
                "reasons": [
                    "Transaction amount is significantly "
                    "above the customer's historical average.",
                    "High-value transaction detected.",
                ],
            },

            # ------------------------------------------------
            # 5. NEW LOCATION
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 10:57:36 pm",
                "amount": 7425.50,
                "risk_score": 0.167,
                "risk_percentage": 16.70,
                "risk_level": "HIGH",
                "decision": "REVIEW",
                "signals": 2,
                "reasons": [
                    "Transaction originates from a "
                    "new customer location.",
                    "Transaction amount differs from "
                    "customer history.",
                ],
            },

            # ------------------------------------------------
            # 6. LOCATION + VELOCITY
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 10:55:12 pm",
                "amount": 27400.00,
                "risk_score": 0.641,
                "risk_percentage": 64.10,
                "risk_level": "CRITICAL",
                "decision": "BLOCK",
                "signals": 5,
                "reasons": [
                    "Transaction originates from a "
                    "new customer location.",
                    "Customer transaction velocity "
                    "is unusually high.",
                    "Transaction amount is significantly "
                    "above historical behavior.",
                    "Shared network activity detected.",
                    "High-value transaction detected.",
                ],
            },

            # ------------------------------------------------
            # 7. NETWORK RISK
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 10:52:47 pm",
                "amount": 18650.75,
                "risk_score": 0.731,
                "risk_percentage": 73.10,
                "risk_level": "CRITICAL",
                "decision": "BLOCK",
                "signals": 6,
                "reasons": [
                    "IP address is associated with "
                    "multiple customers.",
                    "Device is associated with "
                    "multiple customers.",
                    "Multiple previous network connections "
                    "were detected.",
                    "Transaction amount is significantly "
                    "above customer history.",
                    "High transaction velocity detected.",
                    "High-value transaction detected.",
                ],
            },

            # ------------------------------------------------
            # 8. FRAUD ATTACK
            # ------------------------------------------------

            {
                "timestamp": "29 Aug 2026, 10:50:03 pm",
                "amount": 45000.00,
                "risk_score": 0.9926,
                "risk_percentage": 99.26,
                "risk_level": "CRITICAL",
                "decision": "BLOCK",
                "signals": 8,
                "reasons": [
                    "Transaction amount is more than "
                    "5x the customer's historical average.",
                    "Transaction originates from a "
                    "new customer location.",
                    "Customer transaction location changed "
                    "from the previous transaction.",
                    "IP address is associated with "
                    "multiple customers.",
                    "Device is associated with "
                    "multiple customers.",
                    "Transaction has multiple previous "
                    "network connections.",
                    "Unusually high transaction velocity "
                    "detected within one hour.",
                    "High-value transaction detected.",
                ],
            },
        ],
    }


# ============================================================
# CLEAR DECISION HISTORY
# ============================================================

@app.delete(
    "/api/v1/risk/history"
)
def clear_decision_history():

    decision_history.clear()

    return {

        "status":
            "cleared",

        "count":
            0,
    }


# ============================================================
# DEMO DATASET
# ============================================================

DEMO_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features.csv"
)


def load_demo_transactions():

    if not DEMO_FEATURE_PATH.exists():

        raise FileNotFoundError(
            "Demo feature dataset not found:\n"
            f"{DEMO_FEATURE_PATH}"
        )

    df = pd.read_csv(
        DEMO_FEATURE_PATH
    )

    # --------------------------------------------------------
    # Existing demo examples
    # --------------------------------------------------------

    safe_index = 64030

    review_index = 85563

    # Existing high-risk handcrafted example
    # is preserved exactly as before.
    # --------------------------------------------------------

    fraud_transaction = {

        "amount": 45000,

        "amount_vs_customer_history": 7.5,

        "is_new_location": 1,

        "location_changed_from_previous": 1,

        "ip_customer_count_before": 5,

        "device_customer_count_before": 4,

        "device_transactions_before": 12,

        "ip_transactions_before": 15,

        "network_connections_before": 9,

        "customer_transactions_before": 8,

        "customer_txn_count_1h_before": 7,

        "customer_txn_count_24h_before": 12,

        "merchant_transactions_before": 650,

        "customer_amount_sum_before": 60000,

        "customer_avg_amount_before": 6000,

        "hour_of_day": 3,

        "day_of_week": 2,

        "is_late_night": 1,

        "is_weekend": 0,

        "is_high_value": 1,

        "is_very_high_value": 1,

        "device_other_customers_before": 3,

        "ip_other_customers_before": 4,

        "device_ip_transactions_before": 6,
    }

    # --------------------------------------------------------
    # Additional real dataset examples
    #
    # These are selected from actual rows in
    # temporal_features.csv and represent different
    # behavioral / temporal / network patterns.
    # --------------------------------------------------------

    additional_demo_indices = {

        # High-value transaction:
        # amount ≈ ₹24,512 and ≈ 6x customer history.
        "HIGH_VALUE": 48,

        # New customer location:
        # new location + high-value behavior.
        "NEW_LOCATION": 161,

        # Location change:
        # customer location changed from previous transaction.
        "LOCATION_CHANGE": 1505,

        # High velocity:
        # multiple transactions within one hour.
        "HIGH_VELOCITY": 1989,

        # Strong network relationship:
        # high associated customer count + network connections.
        "NETWORK_RISK": 873,
    }

    # --------------------------------------------------------
    # Convert existing dataset rows
    # --------------------------------------------------------

    safe_row = (
        df.iloc[safe_index]
        .replace({np.nan: 0})
        .to_dict()
    )

    review_row = (
        df.iloc[review_index]
        .replace({np.nan: 0})
        .to_dict()
    )

    # --------------------------------------------------------
    # Convert additional real dataset rows
    # --------------------------------------------------------

    high_value_row = (
        df.iloc[
            additional_demo_indices["HIGH_VALUE"]
        ]
        .replace({np.nan: 0})
        .to_dict()
    )

    new_location_row = (
        df.iloc[
            additional_demo_indices["NEW_LOCATION"]
        ]
        .replace({np.nan: 0})
        .to_dict()
    )

    location_change_row = (
        df.iloc[
            additional_demo_indices["LOCATION_CHANGE"]
        ]
        .replace({np.nan: 0})
        .to_dict()
    )

    high_velocity_row = (
        df.iloc[
            additional_demo_indices["HIGH_VELOCITY"]
        ]
        .replace({np.nan: 0})
        .to_dict()
    )

    network_risk_row = (
        df.iloc[
            additional_demo_indices["NETWORK_RISK"]
        ]
        .replace({np.nan: 0})
        .to_dict()
    )

    # --------------------------------------------------------
    # Return demo scenarios
    # --------------------------------------------------------

    return [

        # ====================================================
        # 1. SAFE
        # ====================================================

        {
            "id":
                "DEMO-SAFE-001",

            "scenario":
                "SAFE",

            "description":
                (
                    "Normal transaction from the "
                    "processed temporal feature dataset."
                ),

            "transaction":
                safe_row,
        },

        # ====================================================
        # 2. REVIEW
        # ====================================================

        {
            "id":
                "DEMO-REVIEW-001",

            "scenario":
                "REVIEW",

            "description":
                (
                    "Genuine borderline fraud example "
                    "from the validation dataset."
                ),

            "transaction":
                review_row,
        },

        # ====================================================
        # 3. HIGH VALUE
        # ====================================================

        {
            "id":
                "DEMO-HIGH-VALUE-001",

            "scenario":
                "HIGH_VALUE",

            "description":
                (
                    "Real dataset transaction with "
                    "significant amount deviation and "
                    "high-value behavior."
                ),

            "transaction":
                high_value_row,
        },

        # ====================================================
        # 4. NEW LOCATION
        # ====================================================

        {
            "id":
                "DEMO-NEW-LOCATION-001",

            "scenario":
                "NEW_LOCATION",

            "description":
                (
                    "Real dataset transaction originating "
                    "from a previously unseen customer location."
                ),

            "transaction":
                new_location_row,
        },

        # ====================================================
        # 5. LOCATION CHANGE
        # ====================================================

        {
            "id":
                "DEMO-LOCATION-CHANGE-001",

            "scenario":
                "LOCATION_CHANGE",

            "description":
                (
                    "Real dataset transaction where the "
                    "customer location changed from the "
                    "previous transaction."
                ),

            "transaction":
                location_change_row,
        },

        # ====================================================
        # 6. HIGH VELOCITY
        # ====================================================

        {
            "id":
                "DEMO-HIGH-VELOCITY-001",

            "scenario":
                "HIGH_VELOCITY",

            "description":
                (
                    "Real dataset transaction showing "
                    "elevated transaction velocity within "
                    "one hour."
                ),

            "transaction":
                high_velocity_row,
        },

        # ====================================================
        # 7. NETWORK RISK
        # ====================================================

        {
            "id":
                "DEMO-NETWORK-RISK-001",

            "scenario":
                "NETWORK_RISK",

            "description":
                (
                    "Real dataset transaction with strong "
                    "network and shared-entity relationships."
                ),

            "transaction":
                network_risk_row,
        },

        # ====================================================
        # 8. FRAUD ATTACK
        # ====================================================

        {
            "id":
                "DEMO-BLOCK-001",

            "scenario":
                "HIGH_RISK",

            "description":
                (
                    "High-risk transaction with multiple "
                    "strong fraud signals."
                ),

            "transaction":
                fraud_transaction,
        },
    ]


# ============================================================
# DEMO TRANSACTIONS ENDPOINT
# ============================================================

@app.get(
    "/api/v1/demo/transactions"
)
def get_demo_transactions():

    try:

        transactions = load_demo_transactions()

        return {

            "count":
                len(
                    transactions
                ),

            "transactions":
                transactions,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load demo transactions: "
                f"{str(error)}"
            ),
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )