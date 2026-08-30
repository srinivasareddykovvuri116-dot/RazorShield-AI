import numpy as np
from pathlib import Path
import sys
import pytest


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.risk_engine.risk_scorer import RiskScorer


# ============================================================
# TEST TRANSACTIONS
# ============================================================

SAFE_TRANSACTION = {
    "amount": 750,
    "amount_log": np.log1p(750),
    "amount_vs_customer_history": 0.9,
    "is_new_location": 0,
    "location_changed_from_previous": 0,
    "ip_customer_count_before": 1,
    "device_customer_count_before": 1,
    "device_transactions_before": 4,
    "ip_transactions_before": 4,
    "network_connections_before": 1,
    "customer_transactions_before": 15,
    "customer_txn_count_1h_before": 1,
    "customer_txn_count_24h_before": 4,
    "merchant_transactions_before": 120,
    "customer_amount_sum_before": 12000,
    "customer_avg_amount_before": 833,
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_late_night": 0,
    "is_weekend": 0,
    "is_high_value": 0,
    "is_very_high_value": 0,
    "device_other_customers_before": 0,
    "ip_other_customers_before": 0,
    "device_ip_transactions_before": 1,
}


SUSPICIOUS_TRANSACTION = {
    "amount": 8000,
    "amount_log": np.log1p(8000),
    "amount_vs_customer_history": 3.5,
    "is_new_location": 1,
    "location_changed_from_previous": 1,
    "ip_customer_count_before": 3,
    "device_customer_count_before": 2,
    "device_transactions_before": 7,
    "ip_transactions_before": 8,
    "network_connections_before": 4,
    "customer_transactions_before": 8,
    "customer_txn_count_1h_before": 4,
    "customer_txn_count_24h_before": 9,
    "merchant_transactions_before": 300,
    "customer_amount_sum_before": 30000,
    "customer_avg_amount_before": 2300,
    "hour_of_day": 22,
    "day_of_week": 5,
    "is_late_night": 0,
    "is_weekend": 1,
    "is_high_value": 0,
    "is_very_high_value": 0,
    "device_other_customers_before": 1,
    "ip_other_customers_before": 2,
    "device_ip_transactions_before": 3,
}


HIGH_RISK_TRANSACTION = {
    "amount": 45000,
    "amount_log": np.log1p(45000),
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


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def scorer():
    return RiskScorer()


# ============================================================
# TEST 1 — ENGINE INITIALIZATION
# ============================================================

def test_risk_scorer_initializes():
    engine = RiskScorer()

    assert engine is not None


# ============================================================
# TEST 2 — SAFE TRANSACTION
# ============================================================

def test_safe_transaction(scorer):
    result = scorer.assess(SAFE_TRANSACTION)

    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] == "LOW"
    assert result["decision"] == "ALLOW"
    assert len(result["reasons"]) > 0


# ============================================================
# TEST 3 — SUSPICIOUS TRANSACTION
# ============================================================

def test_suspicious_transaction(scorer):
    result = scorer.assess(SUSPICIOUS_TRANSACTION)

    assert 0 <= result["risk_score"] <= 1
    assert result["decision"] in {
        "ALLOW",
        "REVIEW",
        "BLOCK",
    }
    assert len(result["reasons"]) > 0


# ============================================================
# TEST 4 — HIGH RISK TRANSACTION
# ============================================================

def test_high_risk_transaction(scorer):
    result = scorer.assess(HIGH_RISK_TRANSACTION)

    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] == "CRITICAL"
    assert result["decision"] == "BLOCK"
    assert len(result["reasons"]) >= 3


# ============================================================
# TEST 5 — RISK ORDERING
# ============================================================

def test_high_risk_has_higher_score_than_safe(scorer):
    safe_result = scorer.assess(
        SAFE_TRANSACTION
    )

    high_risk_result = scorer.assess(
        HIGH_RISK_TRANSACTION
    )

    assert (
        high_risk_result["risk_score"]
        >
        safe_result["risk_score"]
    )


# ============================================================
# TEST 6 — ALL PROBABILITIES ARE VALID
# ============================================================

def test_all_risk_scores_are_valid(scorer):

    transactions = [
        SAFE_TRANSACTION,
        SUSPICIOUS_TRANSACTION,
        HIGH_RISK_TRANSACTION,
    ]

    for transaction in transactions:

        result = scorer.assess(
            transaction
        )

        assert (
            0
            <= result["risk_score"]
            <= 1
        )

        assert (
            0
            <= result["risk_percentage"]
            <= 100
        )