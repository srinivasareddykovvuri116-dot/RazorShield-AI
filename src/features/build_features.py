"""
RazorShield AI - Feature Engineering

Converts raw payment transactions into ML-ready features.

IMPORTANT:
    fraud_type and ring_id are ground-truth metadata.
    They are NEVER used as model features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - build_features.py:38" * 70)
    print("RAZORSHIELD AI  FEATURE ENGINEERING - build_features.py:39")
    print("= - build_features.py:40" * 70)
    print()

    transactions = pd.read_csv(
        RAW_DATA_DIR / "transactions.csv"
    )

    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv"
    )

    devices = pd.read_csv(
        RAW_DATA_DIR / "devices.csv"
    )

    ips = pd.read_csv(
        RAW_DATA_DIR / "ips.csv"
    )

    merchants = pd.read_csv(
        RAW_DATA_DIR / "merchants.csv"
    )

    print(
        f"Transactions loaded: {len(transactions):,}"
    )

    print(
        f"Customers loaded:    {len(customers):,}"
    )

    print(
        f"Devices loaded:       {len(devices):,}"
    )

    print(
        f"IPs loaded:           {len(ips):,}"
    )

    print(
        f"Merchants loaded:     {len(merchants):,}"
    )

    print()

    return (
        transactions,
        customers,
        devices,
        ips,
        merchants,
    )


# ============================================================
# PREPARE TRANSACTIONS
# ============================================================

def prepare_transactions(
    transactions,
    customers,
    devices,
    merchants,
):

    print("Preparing transaction data... - build_features.py:105")

    df = transactions.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Merge customer information
    # --------------------------------------------------------

    customer_columns = [
        "customer_id",
        "account_age_days",
        "avg_transaction_amount",
        "usual_location",
        "transaction_frequency",
    ]

    df = df.merge(
        customers[customer_columns],
        on="customer_id",
        how="left",
    )

    # --------------------------------------------------------
    # Merge device information
    # --------------------------------------------------------

    df = df.merge(
        devices[
            [
                "device_id",
                "device_type",
            ]
        ],
        on="device_id",
        how="left",
    )

    # --------------------------------------------------------
    # Merge merchant information
    # --------------------------------------------------------

    df = df.merge(
        merchants[
            [
                "merchant_id",
                "merchant_category",
            ]
        ],
        on="merchant_id",
        how="left",
    )

    print(
        f"Prepared rows: {len(df):,}"
    )

    print()

    return df


# ============================================================
# TIME FEATURES
# ============================================================

def create_time_features(df):

    print("Creating time features... - build_features.py:186")

    df["hour_of_day"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Late-night activity
    df["is_late_night"] = (
        df["hour_of_day"].between(
            0,
            5,
        )
    ).astype(int)

    # Business-hour activity
    df["is_business_hours"] = (
        df["hour_of_day"].between(
            9,
            18,
        )
    ).astype(int)

    return df


# ============================================================
# AMOUNT FEATURES
# ============================================================

def create_amount_features(df):

    print("Creating amount features... - build_features.py:229")

    # --------------------------------------------------------
    # Log amount
    # --------------------------------------------------------

    df["amount_log"] = np.log1p(
        df["amount"]
    )

    # --------------------------------------------------------
    # Amount compared with customer's normal amount
    # --------------------------------------------------------

    df["amount_vs_customer_avg"] = (
        df["amount"]
        /
        df["avg_transaction_amount"].clip(
            lower=1
        )
    )

    # --------------------------------------------------------
    # Difference from customer average
    # --------------------------------------------------------

    df["amount_difference_from_avg"] = (
        df["amount"]
        -
        df["avg_transaction_amount"]
    )

    # --------------------------------------------------------
    # High-value transaction flags
    # --------------------------------------------------------

    df["is_high_value"] = (
        df["amount"] > 10_000
    ).astype(int)

    df["is_very_high_value"] = (
        df["amount"] > 25_000
    ).astype(int)

    return df


# ============================================================
# CUSTOMER HISTORY FEATURES
# ============================================================

def create_customer_history_features(df):

    print("Creating customer history features... - build_features.py:282")

    # IMPORTANT:
    # shift(1) means the current transaction is NOT included
    # in the historical calculation.
    #
    # This prevents target leakage.

    grouped = df.groupby(
        "customer_id",
        sort=False,
    )

    # --------------------------------------------------------
    # Previous transaction time
    # --------------------------------------------------------

    previous_time = grouped[
        "timestamp"
    ].shift(1)

    df["seconds_since_previous_customer_txn"] = (
        df["timestamp"] - previous_time
    ).dt.total_seconds()

    df[
        "seconds_since_previous_customer_txn"
    ] = df[
        "seconds_since_previous_customer_txn"
    ].fillna(999999)

    # --------------------------------------------------------
    # Historical transaction count
    # --------------------------------------------------------

    df["customer_transaction_count"] = (
        grouped.cumcount()
    )

    # --------------------------------------------------------
    # Historical amount statistics
    # --------------------------------------------------------

    previous_amount = grouped[
        "amount"
    ].shift(1)

    historical_sum = (
        previous_amount
        .groupby(
            df["customer_id"]
        )
        .cumsum()
    )

    historical_count = (
        df["customer_transaction_count"]
    )

    df["customer_historical_avg_amount"] = (
        historical_sum
        /
        historical_count.replace(
            0,
            np.nan,
        )
    )

    df[
        "customer_historical_avg_amount"
    ] = df[
        "customer_historical_avg_amount"
    ].fillna(
        df["avg_transaction_amount"]
    )

    # --------------------------------------------------------
    # Historical amount ratio
    # --------------------------------------------------------

    df[
        "amount_vs_historical_avg"
    ] = (
        df["amount"]
        /
        df[
            "customer_historical_avg_amount"
        ].clip(
            lower=1
        )
    )

    return df


# ============================================================
# VELOCITY FEATURES
# ============================================================

def create_velocity_features(df):

    print("Creating velocity features... - build_features.py:383")

    # Data must be sorted chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    indexed = df.set_index(
        "timestamp"
    )

    # --------------------------------------------------------
    # Customer velocity
    # --------------------------------------------------------

    df["customer_txn_count_1h"] = (
        indexed
        .groupby("customer_id")[
            "transaction_id"
        ]
        .rolling("1h")
        .count()
        .reset_index(
            level=0,
            drop=True,
        )
        .to_numpy()
    )

    df["customer_txn_count_24h"] = (
        indexed
        .groupby("customer_id")[
            "transaction_id"
        ]
        .rolling("24h")
        .count()
        .reset_index(
            level=0,
            drop=True,
        )
        .to_numpy()
    )

    # --------------------------------------------------------
    # Customer amount velocity
    # --------------------------------------------------------

    df["customer_amount_1h"] = (
        indexed
        .groupby("customer_id")[
            "amount"
        ]
        .rolling("1h")
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
        .to_numpy()
    )

    df["customer_amount_24h"] = (
        indexed
        .groupby("customer_id")[
            "amount"
        ]
        .rolling("24h")
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
        .to_numpy()
    )

    return df


# ============================================================
# DEVICE FEATURES
# ============================================================

def create_device_features(df):

    print("Creating device network features... - build_features.py:469")

    # Number of distinct customers using each device
    device_customer_count = (
        df.groupby("device_id")[
            "customer_id"
        ]
        .transform("nunique")
    )

    df["device_shared_accounts"] = (
        device_customer_count
    )

    # Number of transactions from device
    df["device_transaction_count"] = (
        df.groupby("device_id")[
            "transaction_id"
        ]
        .transform("count")
    )

    # Customer count per device normalized
    df["device_reuse_score"] = (
        np.log1p(
            df["device_shared_accounts"]
        )
    )

    # Is device shared?
    df["is_shared_device"] = (
        df["device_shared_accounts"] > 1
    ).astype(int)

    return df


# ============================================================
# IP FEATURES
# ============================================================

def create_ip_features(df):

    print("Creating IP network features... - build_features.py:512")

    # Number of distinct customers using IP
    ip_customer_count = (
        df.groupby("ip_id")[
            "customer_id"
        ]
        .transform("nunique")
    )

    df["ip_shared_accounts"] = (
        ip_customer_count
    )

    # Number of transactions from IP
    df["ip_transaction_count"] = (
        df.groupby("ip_id")[
            "transaction_id"
        ]
        .transform("count")
    )

    df["ip_reuse_score"] = (
        np.log1p(
            df["ip_shared_accounts"]
        )
    )

    df["is_shared_ip"] = (
        df["ip_shared_accounts"] > 1
    ).astype(int)

    return df


# ============================================================
# LOCATION FEATURES
# ============================================================

def create_location_features(df):

    print("Creating location features... - build_features.py:553")

    df["is_new_location"] = (
        df["location"]
        !=
        df["usual_location"]
    ).astype(int)

    # Number of different locations seen
    # for each customer across the full dataset.
    #
    # This is descriptive rather than historical.
    # We will create a strictly historical version later
    # if needed for the final model.

    df["customer_location_count"] = (
        df.groupby("customer_id")[
            "location"
        ]
        .transform("nunique")
    )

    return df


# ============================================================
# DEVICE / CUSTOMER RELATIONSHIP
# ============================================================

def create_device_customer_features(df):

    print(
        "Creating device-customer relationship features..."
    )

    # How many customers use the same device?
    device_customer_counts = (
        df.groupby("device_id")[
            "customer_id"
        ]
        .nunique()
    )

    df[
        "device_customer_count"
    ] = df[
        "device_id"
    ].map(
        device_customer_counts
    )

    # How many devices does this customer use?
    customer_device_counts = (
        df.groupby("customer_id")[
            "device_id"
        ]
        .nunique()
    )

    df[
        "customer_device_count"
    ] = df[
        "customer_id"
    ].map(
        customer_device_counts
    )

    return df


# ============================================================
# IP / CUSTOMER RELATIONSHIP
# ============================================================

def create_ip_customer_features(df):

    print(
        "Creating IP-customer relationship features..."
    )

    # How many customers use this IP?
    ip_customer_counts = (
        df.groupby("ip_id")[
            "customer_id"
        ]
        .nunique()
    )

    df[
        "ip_customer_count"
    ] = df[
        "ip_id"
    ].map(
        ip_customer_counts
    )

    # How many IPs does this customer use?
    customer_ip_counts = (
        df.groupby("customer_id")[
            "ip_id"
        ]
        .nunique()
    )

    df[
        "customer_ip_count"
    ] = df[
        "customer_id"
    ].map(
        customer_ip_counts
    )

    return df


# ============================================================
# MERCHANT FEATURES
# ============================================================

def create_merchant_features(df):

    print("Creating merchant features... - build_features.py:674")

    df["merchant_transaction_count"] = (
        df.groupby("merchant_id")[
            "transaction_id"
        ]
        .transform("count")
    )

    df["merchant_customer_count"] = (
        df.groupby("merchant_id")[
            "customer_id"
        ]
        .transform("nunique")
    )

    return df


# ============================================================
# PAYMENT METHOD FEATURES
# ============================================================

def create_payment_features(df):

    print("Creating payment method features... - build_features.py:699")

    # Frequency of customer's payment method.
    customer_method_count = (
        df.groupby(
            [
                "customer_id",
                "payment_method",
            ]
        )["transaction_id"]
        .transform("count")
    )

    df[
        "customer_payment_method_usage"
    ] = customer_method_count

    return df


# ============================================================
# COORDINATION FEATURES
# ============================================================

def create_coordination_features(df):

    print(
        "Creating coordination/network features..."
    )

    # --------------------------------------------------------
    # Number of distinct customers sharing BOTH
    # the same device and IP.
    #
    # This is a stronger signal than device/IP reuse
    # individually.
    # --------------------------------------------------------

    device_ip_customer_count = (
        df.groupby(
            [
                "device_id",
                "ip_id",
            ]
        )["customer_id"]
        .transform("nunique")
    )

    df[
        "device_ip_shared_accounts"
    ] = device_ip_customer_count

    # --------------------------------------------------------
    # Number of customers connected through device
    # --------------------------------------------------------

    df[
        "device_connected_customers"
    ] = (
        df.groupby("device_id")[
            "customer_id"
        ]
        .transform("nunique")
        - 1
    )

    # --------------------------------------------------------
    # Number of customers connected through IP
    # --------------------------------------------------------

    df[
        "ip_connected_customers"
    ] = (
        df.groupby("ip_id")[
            "customer_id"
        ]
        .transform("nunique")
        - 1
    )

    # --------------------------------------------------------
    # Combined network exposure
    # --------------------------------------------------------

    df[
        "network_connection_score"
    ] = (
        df["device_connected_customers"]
        +
        df["ip_connected_customers"]
    )

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categorical_features(df):

    print("Encoding categorical features... - build_features.py:800")

    # One-hot encode low-cardinality categories.
    categorical_columns = [
        "payment_method",
        "device_type",
        "merchant_category",
        "location",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int,
    )

    return df


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

def clean_numeric_values(df):

    print("Cleaning numeric values... - build_features.py:826")

    numeric_columns = df.select_dtypes(
        include=[
            "number",
        ]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # Fill numeric missing values.
    for column in numeric_columns:

        if column == "is_fraud":
            continue

        median = df[column].median()

        df[column] = (
            df[column]
            .fillna(median)
        )

    return df


# ============================================================
# BUILD ML DATASET
# ============================================================

def build_ml_dataset(df):

    print("Building MLready dataset... - build_features.py:867")

    # --------------------------------------------------------
    # Ground-truth / metadata columns
    # --------------------------------------------------------

    metadata_columns = [
        "transaction_id",
        "timestamp",
        "fraud_type",
        "ring_id",
    ]

    # --------------------------------------------------------
    # Identifier columns that should not be directly
    # fed into a traditional ML model.
    # --------------------------------------------------------

    identifier_columns = [
        "customer_id",
        "merchant_id",
        "device_id",
        "ip_id",
    ]

    # Keep target separately.
    target = df[
        "is_fraud"
    ].copy()

    # Remove target + metadata + raw identifiers.
    feature_columns = [
        column
        for column in df.columns
        if column not in (
            metadata_columns
            + identifier_columns
            + ["is_fraud"]
        )
    ]

    X = df[
        feature_columns
    ].copy()

    y = target.rename(
        "is_fraud"
    )

    # --------------------------------------------------------
    # Ensure everything is numeric
    # --------------------------------------------------------

    non_numeric_columns = X.select_dtypes(
        exclude=[
            "number",
        ]
    ).columns.tolist()

    if non_numeric_columns:

        print(
            "WARNING - non-numeric columns found:"
        )

        for column in non_numeric_columns:
            print(
                f"  {column}"
            )

        X = X.drop(
            columns=non_numeric_columns
        )

    # --------------------------------------------------------
    # Save features + target
    # --------------------------------------------------------

    feature_path = (
        PROCESSED_DATA_DIR
        / "features.csv"
    )

    target_path = (
        PROCESSED_DATA_DIR
        / "target.csv"
    )

    X.to_csv(
        feature_path,
        index=False,
    )

    y.to_csv(
        target_path,
        index=False,
    )

    print()
    print(
        f"Features saved to:\n"
        f"{feature_path}"
    )

    print(
        f"Target saved to:\n"
        f"{target_path}"
    )

    print()

    return X, y


# ============================================================
# FEATURE SUMMARY
# ============================================================

def print_feature_summary(
    X,
    y,
):

    print("= - build_features.py:990" * 70)
    print("FEATURE SUMMARY - build_features.py:991")
    print("= - build_features.py:992" * 70)

    print()

    print(
        f"Samples:  {len(X):,}"
    )

    print(
        f"Features: {len(X.columns):,}"
    )

    print(
        f"Fraud:    {int(y.sum()):,}"
    )

    print(
        f"Legit:    {int((y == 0).sum()):,}"
    )

    print()

    print("Feature columns: - build_features.py:1014")

    for i, column in enumerate(
        X.columns,
        start=1,
    ):

        print(
            f"{i:>3}. {column}"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    (
        transactions,
        customers,
        devices,
        ips,
        merchants,
    ) = load_data()

    df = prepare_transactions(
        transactions,
        customers,
        devices,
        merchants,
    )

    df = create_time_features(
        df
    )

    df = create_amount_features(
        df
    )

    df = create_customer_history_features(
        df
    )

    df = create_velocity_features(
        df
    )

    df = create_device_features(
        df
    )

    df = create_ip_features(
        df
    )

    df = create_location_features(
        df
    )

    df = create_device_customer_features(
        df
    )

    df = create_ip_customer_features(
        df
    )

    df = create_merchant_features(
        df
    )

    df = create_payment_features(
        df
    )

    df = create_coordination_features(
        df
    )

    df = encode_categorical_features(
        df
    )

    df = clean_numeric_values(
        df
    )

    X, y = build_ml_dataset(
        df
    )

    print_feature_summary(
        X,
        y,
    )

    print("= - build_features.py:1114" * 70)
    print("FEATURE ENGINEERING COMPLETE - build_features.py:1115")
    print("= - build_features.py:1116" * 70)


if __name__ == "__main__":
    main()