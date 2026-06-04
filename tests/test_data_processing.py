import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from src.data_processing import (
    extract_datetime_features,
    create_aggregate_features,
    compute_rfm,
    assign_risk_labels
)


def test_extract_datetime_features():
    df = pd.DataFrame({
        "TransactionStartTime": ["2023-01-15 10:30:00", "2023-02-20 14:45:00"],
        "Amount": [100, 200]
    })
    result = extract_datetime_features(df)
    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns


def test_create_aggregate_features():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2, 2],
        "Amount": [100, 200, 150, 250],
        "TransactionId": [101, 102, 103, 104],
        "transaction_hour": [10, 11, 12, 13],
        "transaction_month": [1, 1, 2, 2]
    })
    result = create_aggregate_features(df)
    assert "total_transaction_amount" in result.columns
    assert "transaction_count" in result.columns


def test_compute_rfm():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionStartTime": ["2023-01-01", "2023-01-05", "2023-01-10"],
        "Amount": [100, 200, 150],
        "TransactionId": [1, 2, 3]
    })
    result = compute_rfm(df)
    assert "Recency" in result.columns
    assert "Frequency" in result.columns


def test_assign_risk_labels():
    rfm = pd.DataFrame({
        "CustomerId": [1, 2, 3, 4],
        "Recency": [10, 20, 60, 90],
        "Frequency": [8, 4, 2, 1],
        "Monetary": [4000, 1500, 600, 100]
    })
    result = assign_risk_labels(rfm)
    assert "is_high_risk" in result.columns


if __name__ == "__main__":
    pytest.main(["-v"])