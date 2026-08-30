"""
RazorShield AI - Temporal-Safe Feature Engineering

Every behavioral feature is calculated using information
available BEFORE the current transaction.

This simulates a real-time payment-risk system.
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


TRANSACTIONS_PATH = (
    RAW_DATA_DIR / "transactions.csv"
)

CUSTOMERS_PATH = (
    RAW_DATA_DIR / "customers.csv"
)

DEVICES_PATH = (
    RAW_DATA_DIR / "devices.csv"
)

MERCHANTS_PATH = (
    RAW_DATA_DIR / "merchants.csv"
)


FEATURES_PATH = (
    PROCESSED_DATA_DIR
    / "temporal_features.csv"
)

TARGET_PATH = (
    PROCESSED_DATA_DIR
    / "temporal_target.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - build_temporal_features.py:65" * 70)
    print("RAZORSHIELD AI  TEMPORAL FEATURE ENGINEERING - build_temporal_features.py:66")
    print("= - build_temporal_features.py:67" * 70)
    print()

    transactions = pd.read_csv(
        TRANSACTIONS_PATH
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH
    )

    devices = pd.read_csv(
        DEVICES_PATH
    )

    merchants = pd.read_csv(
        MERCHANTS_PATH
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    print(
        f"Transactions: {len(transactions):,}"
    )

    print(
        f"Customers:    {len(customers):,}"
    )

    print(
        f"Devices:      {len(devices):,}"
    )

    print(
        f"Merchants:    {len(merchants):,}"
    )

    print()

    return (
        transactions,
        customers,
        devices,
        merchants,
    )


# ============================================================
# SORT TRANSACTIONS
# ============================================================

def sort_transactions(df):

    print("Sorting transactions chronologically... - build_temporal_features.py:122")

    df = df.sort_values(
        [
            "timestamp",
            "transaction_id",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Time range: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    print()

    return df


# ============================================================
# CUSTOMER TEMPORAL FEATURES
# ============================================================

def customer_features(df):

    print("Creating customer historical features... - build_temporal_features.py:150")

    # --------------------------------------------------------
    # Previous transaction timestamp
    # --------------------------------------------------------

    df["previous_customer_timestamp"] = (
        df.groupby(
            "customer_id"
        )["timestamp"]
        .shift(1)
    )

    df[
        "seconds_since_customer_transaction"
    ] = (
        df["timestamp"]
        -
        df["previous_customer_timestamp"]
    ).dt.total_seconds()

    df[
        "seconds_since_customer_transaction"
    ] = (
        df[
            "seconds_since_customer_transaction"
        ]
        .fillna(999999)
    )

    # --------------------------------------------------------
    # Historical transaction count
    # --------------------------------------------------------

    df[
        "customer_transactions_before"
    ] = (
        df.groupby(
            "customer_id"
        ).cumcount()
    )

    # --------------------------------------------------------
    # Historical amount sum
    # --------------------------------------------------------

    previous_amount = (
        df.groupby(
            "customer_id"
        )["amount"]
        .shift(1)
    )

    df[
        "customer_amount_sum_before"
    ] = (
        previous_amount
        .fillna(0)
        .groupby(
            df["customer_id"]
        )
        .cumsum()
    )

    # --------------------------------------------------------
    # Historical average amount
    # --------------------------------------------------------

    count_before = (
        df[
            "customer_transactions_before"
        ]
    )

    df[
        "customer_avg_amount_before"
    ] = (
        df[
            "customer_amount_sum_before"
        ]
        /
        count_before.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Use customer profile average only when
    # there is no transaction history.
    # --------------------------------------------------------

    customer_avg_map = (
        pd.read_csv(
            CUSTOMERS_PATH
        )
        .set_index(
            "customer_id"
        )[
            "avg_transaction_amount"
        ]
    )

    df[
        "customer_avg_amount_before"
    ] = (
        df[
            "customer_avg_amount_before"
        ]
        .fillna(
            df["customer_id"].map(
                customer_avg_map
            )
        )
    )

    # --------------------------------------------------------
    # Amount relative to historical behavior
    # --------------------------------------------------------

    df[
        "amount_vs_customer_history"
    ] = (
        df["amount"]
        /
        df[
            "customer_avg_amount_before"
        ].clip(
            lower=1
        )
    )

    # --------------------------------------------------------
    # Historical amount deviation
    # --------------------------------------------------------

    df[
        "amount_difference_from_history"
    ] = (
        df["amount"]
        -
        df[
            "customer_avg_amount_before"
        ]
    )

    return df


# ============================================================
# CUSTOMER VELOCITY
# ============================================================

def customer_velocity_features(df):

    print("Creating customer velocity features... - build_temporal_features.py:305")

    # --------------------------------------------------------
    # For each transaction, count previous transactions
    # in the preceding 1 hour and 24 hours.
    #
    # IMPORTANT:
    # The current transaction itself is excluded.
    # --------------------------------------------------------

    timestamps = (
        df["timestamp"]
        .astype("int64")
        // 10**9
    )

    customer_groups = df.groupby(
        "customer_id",
        sort=False,
    )

    previous_1h = []
    previous_24h = []

    for customer_id, group in customer_groups:

        group_indices = group.index

        group_times = (
            timestamps.loc[
                group_indices
            ]
            .to_numpy()
        )

        values_1h = []
        values_24h = []

        left_1h = 0
        left_24h = 0

        for i, current_time in enumerate(
            group_times
        ):

            while (
                left_1h < i
                and
                current_time
                -
                group_times[left_1h]
                >= 3600
            ):
                left_1h += 1

            while (
                left_24h < i
                and
                current_time
                -
                group_times[left_24h]
                >= 86400
            ):
                left_24h += 1

            values_1h.append(
                i - left_1h
            )

            values_24h.append(
                i - left_24h
            )

        previous_1h.extend(
            zip(
                group_indices,
                values_1h,
            )
        )

        previous_24h.extend(
            zip(
                group_indices,
                values_24h,
            )
        )

    count_1h_map = dict(
        previous_1h
    )

    count_24h_map = dict(
        previous_24h
    )

    df[
        "customer_txn_count_1h_before"
    ] = [
        count_1h_map[index]
        for index in df.index
    ]

    df[
        "customer_txn_count_24h_before"
    ] = [
        count_24h_map[index]
        for index in df.index
    ]

    return df


# ============================================================
# DEVICE TEMPORAL FEATURES
# ============================================================

def device_features(df):

    print("Creating device historical features... - build_temporal_features.py:423")

    # --------------------------------------------------------
    # Number of previous transactions from device
    # --------------------------------------------------------

    df[
        "device_transactions_before"
    ] = (
        df.groupby(
            "device_id"
        ).cumcount()
    )

    # --------------------------------------------------------
    # Number of distinct customers previously seen
    # --------------------------------------------------------

    df[
        "device_customer_count_before"
    ] = (
        df.groupby(
            "device_id"
        )["customer_id"]
        .transform(
            lambda s:
            s.shift(1)
            .groupby(
                df.loc[
                    s.index,
                    "device_id"
                ]
            )
            .transform("nunique")
        )
    )

    # The groupby transform above can produce NaN
    # for the first device transaction.
    df[
        "device_customer_count_before"
    ] = (
        df[
            "device_customer_count_before"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Whether device has been seen before
    # --------------------------------------------------------

    df[
        "is_seen_device_before"
    ] = (
        df[
            "device_transactions_before"
        ] > 0
    ).astype(int)

    return df


# ============================================================
# IP TEMPORAL FEATURES
# ============================================================

def ip_features(df):

    print("Creating IP historical features... - build_temporal_features.py:492")

    # --------------------------------------------------------
    # Previous transactions from IP
    # --------------------------------------------------------

    df[
        "ip_transactions_before"
    ] = (
        df.groupby(
            "ip_id"
        ).cumcount()
    )

    # --------------------------------------------------------
    # Previous distinct customers from IP
    # --------------------------------------------------------

    df[
        "ip_customer_count_before"
    ] = (
        df.groupby(
            "ip_id"
        )["customer_id"]
        .transform(
            lambda s:
            s.shift(1)
            .groupby(
                df.loc[
                    s.index,
                    "ip_id"
                ]
            )
            .transform("nunique")
        )
    )

    df[
        "ip_customer_count_before"
    ] = (
        df[
            "ip_customer_count_before"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Previously seen IP
    # --------------------------------------------------------

    df[
        "is_seen_ip_before"
    ] = (
        df[
            "ip_transactions_before"
        ] > 0
    ).astype(int)

    return df


# ============================================================
# MERCHANT TEMPORAL FEATURES
# ============================================================

def merchant_features(df):

    print("Creating merchant historical features... - build_temporal_features.py:559")

    # --------------------------------------------------------
    # Previous transactions at merchant
    # --------------------------------------------------------

    df[
        "merchant_transactions_before"
    ] = (
        df.groupby(
            "merchant_id"
        ).cumcount()
    )

    # --------------------------------------------------------
    # Previous distinct customers
    # --------------------------------------------------------

    df[
        "merchant_customers_before"
    ] = (
        df.groupby(
            "merchant_id"
        )["customer_id"]
        .transform(
            lambda s:
            s.shift(1)
            .groupby(
                df.loc[
                    s.index,
                    "merchant_id"
                ]
            )
            .transform("nunique")
        )
    )

    df[
        "merchant_customers_before"
    ] = (
        df[
            "merchant_customers_before"
        ]
        .fillna(0)
    )

    return df


# ============================================================
# LOCATION FEATURES
# ============================================================

def location_features(
    df,
    customers,
):

    print("Creating location features... - build_temporal_features.py:617")

    usual_location_map = (
        customers
        .set_index(
            "customer_id"
        )[
            "usual_location"
        ]
    )

    df[
        "usual_location"
    ] = df[
        "customer_id"
    ].map(
        usual_location_map
    )

    # --------------------------------------------------------
    # New location relative to customer profile
    # --------------------------------------------------------

    df[
        "is_new_location"
    ] = (
        df["location"]
        !=
        df["usual_location"]
    ).astype(int)

    # --------------------------------------------------------
    # Previous location
    # --------------------------------------------------------

    df[
        "previous_customer_location"
    ] = (
        df.groupby(
            "customer_id"
        )["location"]
        .shift(1)
    )

    df[
        "location_changed_from_previous"
    ] = (
        (
            df["previous_customer_location"]
            .notna()
        )
        &
        (
            df["location"]
            !=
            df[
                "previous_customer_location"
            ]
        )
    ).astype(int)

    return df


# ============================================================
# TIME FEATURES
# ============================================================

def time_features(df):

    print("Creating time features... - build_temporal_features.py:687")

    df[
        "hour_of_day"
    ] = df[
        "timestamp"
    ].dt.hour

    df[
        "day_of_week"
    ] = df[
        "timestamp"
    ].dt.dayofweek

    df[
        "is_weekend"
    ] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df[
        "is_late_night"
    ] = (
        df["hour_of_day"].between(
            0,
            5,
        )
    ).astype(int)

    return df


# ============================================================
# AMOUNT FEATURES
# ============================================================

def amount_features(df):

    print("Creating amount features... - build_temporal_features.py:725")

    df[
        "amount_log"
    ] = np.log1p(
        df["amount"]
    )

    df[
        "is_high_value"
    ] = (
        df["amount"] > 10_000
    ).astype(int)

    df[
        "is_very_high_value"
    ] = (
        df["amount"] > 25_000
    ).astype(int)

    return df


# ============================================================
# NETWORK FEATURES
# ============================================================

def network_features(df):

    print("Creating historical network features... - build_temporal_features.py:754")

    # --------------------------------------------------------
    # Number of previous transactions for device/IP pair
    # --------------------------------------------------------

    df[
        "device_ip_transactions_before"
    ] = (
        df.groupby(
            [
                "device_id",
                "ip_id",
            ]
        ).cumcount()
    )

    # --------------------------------------------------------
    # Current device previously used by other customers
    #
    # We calculate cumulative unique customer counts
    # WITHOUT including the current transaction.
    # --------------------------------------------------------

    device_customer_sets = {}

    device_customer_count = []

    for row in df.itertuples():

        device = row.device_id
        customer = row.customer_id

        previous_customers = (
            device_customer_sets
            .get(
                device,
                set(),
            )
        )

        device_customer_count.append(
            len(
                previous_customers
                -
                {customer}
            )
        )

        if device not in device_customer_sets:

            device_customer_sets[
                device
            ] = set()

        device_customer_sets[
            device
        ].add(
            customer
        )

    df[
        "device_other_customers_before"
    ] = device_customer_count

    # --------------------------------------------------------
    # Same for IP
    # --------------------------------------------------------

    ip_customer_sets = {}

    ip_customer_count = []

    for row in df.itertuples():

        ip = row.ip_id
        customer = row.customer_id

        previous_customers = (
            ip_customer_sets
            .get(
                ip,
                set(),
            )
        )

        ip_customer_count.append(
            len(
                previous_customers
                -
                {customer}
            )
        )

        if ip not in ip_customer_sets:

            ip_customer_sets[
                ip
            ] = set()

        ip_customer_sets[
            ip
        ].add(
            customer
        )

    df[
        "ip_other_customers_before"
    ] = ip_customer_count

    # --------------------------------------------------------
    # Combined network score
    # --------------------------------------------------------

    df[
        "network_connections_before"
    ] = (
        df[
            "device_other_customers_before"
        ]
        +
        df[
            "ip_other_customers_before"
        ]
    )

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categories(df):

    print("Encoding categorical features... - build_temporal_features.py:889")

    columns = [
        "payment_method",
        "device_type",
        "merchant_category",
        "location",
    ]

    existing_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=existing_columns,
        prefix=existing_columns,
        dtype=int,
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("Cleaning feature data... - build_temporal_features.py:920")

    # --------------------------------------------------------
    # Remove metadata and identifiers
    # --------------------------------------------------------

    forbidden_columns = [
        "transaction_id",
        "timestamp",
        "customer_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "fraud_type",
        "ring_id",
        "previous_customer_timestamp",
        "previous_customer_location",
        "usual_location",
    ]

    target = df[
        "is_fraud"
    ].copy()

    X = df.drop(
        columns=[
            column
            for column in forbidden_columns
            if column in df.columns
        ]
        + [
            "is_fraud",
        ]
    )

    # --------------------------------------------------------
    # Keep numeric features only
    # --------------------------------------------------------

    non_numeric = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:

        print(
            "Dropping non-numeric columns:"
        )

        for column in non_numeric:

            print(
                f"  - {column}"
            )

        X = X.drop(
            columns=non_numeric
        )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].isna().any():

            median = X[column].median()

            X[column] = X[
                column
            ].fillna(
                median
            )

    return X, target


# ============================================================
# SAVE
# ============================================================

def save_data(
    X,
    y,
):

    print("Saving temporalsafe dataset... - build_temporal_features.py:1019")

    X.to_csv(
        FEATURES_PATH,
        index=False,
    )

    y.to_csv(
        TARGET_PATH,
        index=False,
    )

    print()

    print(
        f"Features saved to:\n"
        f"{FEATURES_PATH}"
    )

    print(
        f"Target saved to:\n"
        f"{TARGET_PATH}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    X,
    y,
):

    print("= - build_temporal_features.py:1055" * 70)
    print("TEMPORAL FEATURE SUMMARY - build_temporal_features.py:1056")
    print("= - build_temporal_features.py:1057" * 70)
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
        f"Legitimate: {int((y == 0).sum()):,}"
    )

    print()

    print(
        "Feature groups include:"
    )

    groups = [
        "Customer history",
        "Customer velocity",
        "Device history",
        "IP history",
        "Merchant history",
        "Location behavior",
        "Network connections",
        "Amount behavior",
        "Time behavior",
    ]

    for group in groups:

        print(
            f"  ✓ {group}"
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
        merchants,
    ) = load_data()

    df = sort_transactions(
        transactions
    )

    df = customer_features(
        df
    )

    df = customer_velocity_features(
        df
    )

    df = device_features(
        df
    )

    # IP features require the raw IP column,
    # which is already present in transactions.
    df = ip_features(
        df
    )

    df = merchant_features(
        df
    )

    df = location_features(
        df,
        customers,
    )

    df = time_features(
        df
    )

    df = amount_features(
        df
    )

    df = network_features(
        df
    )

    df = encode_categories(
        df
    )

    X, y = clean_data(
        df
    )

    save_data(
        X,
        y,
    )

    print_summary(
        X,
        y,
    )

    print("= - build_temporal_features.py:1177" * 70)
    print("TEMPORAL FEATURE ENGINEERING COMPLETE - build_temporal_features.py:1178")
    print("= - build_temporal_features.py:1179" * 70)
    print()


if __name__ == "__main__":
    main()