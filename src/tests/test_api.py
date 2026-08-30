"""
RazorShield AI - API Tests

Tests the FastAPI risk assessment endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# TEST TRANSACTION
# ============================================================

SAFE_TRANSACTION = {
    "amount": 750,
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


HIGH_RISK_TRANSACTION = {
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


# ============================================================
# TEST 1 — HEALTH
# ============================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["risk_engine"] == "ready"


# ============================================================
# TEST 2 — SAFE TRANSACTION
# ============================================================

def test_assess_safe_transaction():

    response = client.post(
        "/api/v1/risk/assess",
        json=SAFE_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert "risk_score" in data
    assert "risk_percentage" in data
    assert "risk_level" in data
    assert "decision" in data
    assert "reasons" in data

    assert 0 <= data["risk_score"] <= 1
    assert 0 <= data["risk_percentage"] <= 100

    assert data["risk_level"] == "LOW"
    assert data["decision"] == "ALLOW"

    assert isinstance(data["reasons"], list)


# ============================================================
# TEST 3 — HIGH-RISK TRANSACTION
# ============================================================

def test_assess_high_risk_transaction():

    response = client.post(
        "/api/v1/risk/assess",
        json=HIGH_RISK_TRANSACTION,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["risk_score"] <= 1
    assert 0 <= data["risk_percentage"] <= 100

    assert data["risk_level"] == "CRITICAL"
    assert data["decision"] == "BLOCK"

    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) > 0


# ============================================================
# TEST 4 — INVALID REQUEST
# ============================================================

def test_assess_invalid_request():

    response = client.post(
        "/api/v1/risk/assess",
        json={},
    )

    assert response.status_code == 422


# ============================================================
# TEST 5 — HISTORY ENDPOINT
# ============================================================

def test_history_endpoint():

    response = client.get(
        "/api/v1/risk/history"
    )

    assert response.status_code == 200

    data = response.json()

    # API returns an object containing count and decisions.
    assert isinstance(data, dict)

    assert "count" in data
    assert "decisions" in data

    assert isinstance(
        data["count"],
        int
    )

    assert isinstance(
        data["decisions"],
        list
    )

    assert data["count"] == len(
        data["decisions"]
    )


# ============================================================
# TEST 6 — POLICY ENDPOINT
# ============================================================

def test_policy_endpoint():

    response = client.get(
        "/api/v1/risk/policy"
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)