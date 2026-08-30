"""
Dataset Quality Inspection
RazorShield AI

Checks the synthetic payment dataset before ML training.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("= - inspect_dataset.py:28" * 70)
    print("RAZORSHIELD AI  DATASET INSPECTION - inspect_dataset.py:29")
    print("= - inspect_dataset.py:30" * 70)
    print()

    files = {
        "customers": "customers.csv",
        "merchants": "merchants.csv",
        "devices": "devices.csv",
        "ips": "ips.csv",
        "abuse_rings": "abuse_rings.csv",
        "transactions": "transactions.csv",
    }

    data = {}

    for name, filename in files.items():

        path = RAW_DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"Loaded {filename:<20} "
            f"{len(data[name]):>8,} rows"
        )

    print()

    return data


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def inspect_basic_info(data):

    print("= - inspect_dataset.py:71" * 70)
    print("1. BASIC DATASET INFORMATION - inspect_dataset.py:72")
    print("= - inspect_dataset.py:73" * 70)

    for name, df in data.items():

        print()
        print(f"{name.upper()} - inspect_dataset.py:78")

        print(f"Rows:    {len(df):,} - inspect_dataset.py:80")
        print(f"Columns: {len(df.columns)} - inspect_dataset.py:81")

        print("Columns: - inspect_dataset.py:83")

        for column in df.columns:
            print(f"{column} - inspect_dataset.py:86")

    print()


# ============================================================
# DUPLICATE CHECK
# ============================================================

def inspect_duplicates(data):

    print("= - inspect_dataset.py:97" * 70)
    print("2. DUPLICATE CHECK - inspect_dataset.py:98")
    print("= - inspect_dataset.py:99" * 70)

    transactions = data["transactions"]

    duplicate_transactions = transactions[
        transactions["transaction_id"].duplicated()
    ]

    print(
        f"Duplicate transaction IDs: "
        f"{len(duplicate_transactions):,}"
    )

    customers = data["customers"]

    duplicate_customers = customers[
        customers["customer_id"].duplicated()
    ]

    print(
        f"Duplicate customer IDs: "
        f"{len(duplicate_customers):,}"
    )

    print()


# ============================================================
# MISSING VALUE CHECK
# ============================================================

def inspect_missing_values(data):

    print("= - inspect_dataset.py:132" * 70)
    print("3. MISSING VALUE CHECK - inspect_dataset.py:133")
    print("= - inspect_dataset.py:134" * 70)

    for name, df in data.items():

        missing = df.isnull().sum()

        missing = missing[missing > 0]

        print()

        if missing.empty:

            print(
                f"{name}: No missing values"
            )

        else:

            print(f"{name}: - inspect_dataset.py:152")

            for column, count in missing.items():

                percentage = (
                    count / len(df)
                ) * 100

                print(
                    f"  {column}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

    print()


# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

def inspect_fraud_distribution(data):

    print("= - inspect_dataset.py:175" * 70)
    print("4. FRAUD DISTRIBUTION - inspect_dataset.py:176")
    print("= - inspect_dataset.py:177" * 70)

    transactions = data["transactions"]

    counts = transactions["is_fraud"].value_counts()

    legitimate = counts.get(0, 0)

    fraud = counts.get(1, 0)

    total = len(transactions)

    print()

    print(f"Total transactions: {total:,} - inspect_dataset.py:191")
    print(f"Legitimate:         {legitimate:,} - inspect_dataset.py:192")
    print(f"Fraud:              {fraud:,} - inspect_dataset.py:193")

    print()

    print(
        f"Legitimate rate: "
        f"{legitimate / total * 100:.2f}%"
    )

    print(
        f"Fraud rate:       "
        f"{fraud / total * 100:.2f}%"
    )

    print()

    print("Fraud type distribution: - inspect_dataset.py:209")

    fraud_types = (
        transactions[
            transactions["is_fraud"] == 1
        ]["fraud_type"]
        .value_counts()
    )

    for fraud_type, count in fraud_types.items():

        percentage = (
            count / fraud
        ) * 100

        print(
            f"  {fraud_type:<20}"
            f"{count:>6,} "
            f"({percentage:>5.2f}%)"
        )

    print()


# ============================================================
# DEVICE REUSE ANALYSIS
# ============================================================

def inspect_device_reuse(data):

    print("= - inspect_dataset.py:239" * 70)
    print("5. DEVICE REUSE ANALYSIS - inspect_dataset.py:240")
    print("= - inspect_dataset.py:241" * 70)

    transactions = data["transactions"]

    device_accounts = (
        transactions
        .groupby("device_id")["customer_id"]
        .nunique()
    )

    print()

    print(
        f"Unique devices used: "
        f"{device_accounts.shape[0]:,}"
    )

    print(
        f"Average accounts/device: "
        f"{device_accounts.mean():.2f}"
    )

    print(
        f"Maximum accounts/device: "
        f"{device_accounts.max():,}"
    )

    shared_devices = (
        device_accounts[
            device_accounts > 1
        ]
    )

    print(
        f"Devices shared by >1 account: "
        f"{len(shared_devices):,}"
    )

    print()

    print("Top 10 most shared devices: - inspect_dataset.py:281")

    print(
        device_accounts
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    print()


# ============================================================
# IP REUSE ANALYSIS
# ============================================================

def inspect_ip_reuse(data):

    print("= - inspect_dataset.py:299" * 70)
    print("6. IP REUSE ANALYSIS - inspect_dataset.py:300")
    print("= - inspect_dataset.py:301" * 70)

    transactions = data["transactions"]

    ip_accounts = (
        transactions
        .groupby("ip_id")["customer_id"]
        .nunique()
    )

    print()

    print(
        f"Unique IPs used: "
        f"{ip_accounts.shape[0]:,}"
    )

    print(
        f"Average accounts/IP: "
        f"{ip_accounts.mean():.2f}"
    )

    print(
        f"Maximum accounts/IP: "
        f"{ip_accounts.max():,}"
    )

    shared_ips = (
        ip_accounts[
            ip_accounts > 1
        ]
    )

    print(
        f"IPs shared by >1 account: "
        f"{len(shared_ips):,}"
    )

    print()

    print("Top 10 most shared IPs: - inspect_dataset.py:341")

    print(
        ip_accounts
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    print()


# ============================================================
# ABUSE RING ANALYSIS
# ============================================================

def inspect_abuse_rings(data):

    print("= - inspect_dataset.py:359" * 70)
    print("7. ABUSE RING ANALYSIS - inspect_dataset.py:360")
    print("= - inspect_dataset.py:361" * 70)

    rings = data["abuse_rings"]

    transactions = data["transactions"]

    print()

    print(
        f"Abuse-ring records: "
        f"{len(rings):,}"
    )

    if rings.empty:

        print("No abuse rings found. - inspect_dataset.py:376")

        return

    ring_sizes = (
        rings
        .groupby("ring_id")["customer_id"]
        .nunique()
    )

    print(
        f"Unique abuse rings: "
        f"{ring_sizes.shape[0]:,}"
    )

    print(
        f"Average ring size: "
        f"{ring_sizes.mean():.2f}"
    )

    print(
        f"Largest ring: "
        f"{ring_sizes.max():,} accounts"
    )

    print()

    ring_transactions = transactions[
        transactions["ring_id"].notnull()
    ]

    print(
        f"Transactions associated with rings: "
        f"{len(ring_transactions):,}"
    )

    fraud_ring_transactions = (
        ring_transactions[
            ring_transactions["is_fraud"] == 1
        ]
    )

    print(
        f"Fraud transactions in rings: "
        f"{len(fraud_ring_transactions):,}"
    )

    print()


# ============================================================
# AMOUNT ANALYSIS
# ============================================================

def inspect_transaction_amounts(data):

    print("= - inspect_dataset.py:432" * 70)
    print("8. TRANSACTION AMOUNT ANALYSIS - inspect_dataset.py:433")
    print("= - inspect_dataset.py:434" * 70)

    transactions = data["transactions"]

    legitimate = transactions[
        transactions["is_fraud"] == 0
    ]["amount"]

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]["amount"]

    print()

    print("LEGITIMATE - inspect_dataset.py:448")

    print(
        f"Average: ₹{legitimate.mean():,.2f}"
    )

    print(
        f"Median:  ₹{legitimate.median():,.2f}"
    )

    print(
        f"Maximum: ₹{legitimate.max():,.2f}"
    )

    print()

    print("FRAUD - inspect_dataset.py:464")

    print(
        f"Average: ₹{fraud.mean():,.2f}"
    )

    print(
        f"Median:  ₹{fraud.median():,.2f}"
    )

    print(
        f"Maximum: ₹{fraud.max():,.2f}"
    )

    print()


# ============================================================
# FRAUD TYPE AMOUNT ANALYSIS
# ============================================================

def inspect_fraud_amount_by_type(data):

    print("= - inspect_dataset.py:487" * 70)
    print("9. FRAUD TYPE ANALYSIS - inspect_dataset.py:488")
    print("= - inspect_dataset.py:489" * 70)

    transactions = data["transactions"]

    fraud_transactions = transactions[
        transactions["is_fraud"] == 1
    ]

    summary = (
        fraud_transactions
        .groupby("fraud_type")["amount"]
        .agg([
            "count",
            "mean",
            "median",
            "max",
        ])
        .sort_values("count", ascending=False)
    )

    print()

    print(summary.to_string())

    print()


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

def inspect_possible_leakage(data):

    print("= - inspect_dataset.py:522" * 70)
    print("10. POTENTIAL DATA LEAKAGE CHECK - inspect_dataset.py:523")
    print("= - inspect_dataset.py:524" * 70)

    transactions = data["transactions"]

    print()

    suspicious_columns = [
        "fraud_type",
        "ring_id",
    ]

    print(
        "The following columns contain information "
        "that should NOT be given directly to the ML model:"
    )

    for column in suspicious_columns:

        if column in transactions.columns:

            print(
                f"  ⚠ {column}"
            )

    print()

    print(
        "These columns are ground-truth / investigation "
        "metadata and must be excluded from model features."
    )

    print()


# ============================================================
# FINAL QUALITY SUMMARY
# ============================================================

def quality_summary(data):

    print("= - inspect_dataset.py:564" * 70)
    print("11. FINAL QUALITY SUMMARY - inspect_dataset.py:565")
    print("= - inspect_dataset.py:566" * 70)

    transactions = data["transactions"]

    checks = []

    # Transaction count
    checks.append((
        "100,000 transactions",
        len(transactions) == 100_000
    ))

    # Unique IDs
    checks.append((
        "Unique transaction IDs",
        transactions["transaction_id"].is_unique
    ))

    # Fraud rate
    fraud_rate = transactions["is_fraud"].mean()

    checks.append((
        "Fraud rate between 2% and 6%",
        0.02 <= fraud_rate <= 0.06
    ))

    # Missing values
    checks.append((
        "No missing transaction values",
        not transactions.isnull().any().any()
    ))

    # Fraud labels
    checks.append((
        "Both legitimate and fraud records exist",
        transactions["is_fraud"].nunique() == 2
    ))

    print()

    for description, passed in checks:

        status = "PASS" if passed else "FAIL"

        print(
            f"[{status}] {description}"
        )

    print()

    passed_count = sum(
        passed for _, passed in checks
    )

    print(
        f"Quality checks passed: "
        f"{passed_count}/{len(checks)}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    data = load_data()

    inspect_basic_info(data)

    inspect_duplicates(data)

    inspect_missing_values(data)

    inspect_fraud_distribution(data)

    inspect_device_reuse(data)

    inspect_ip_reuse(data)

    inspect_abuse_rings(data)

    inspect_transaction_amounts(data)

    inspect_fraud_amount_by_type(data)

    inspect_possible_leakage(data)

    quality_summary(data)

    print("= - inspect_dataset.py:658" * 70)
    print("INSPECTION COMPLETE - inspect_dataset.py:659")
    print("= - inspect_dataset.py:660" * 70)


if __name__ == "__main__":
    main()