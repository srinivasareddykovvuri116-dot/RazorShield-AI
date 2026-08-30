"""
RazorShield AI - Synthetic Payment Risk Dataset Generator V2

Purpose:
    Generate a realistic synthetic payment ecosystem for training
    and evaluating an AI-powered payment risk detection system.

Important:
    - fraud_type and ring_id are ground-truth metadata
    - these fields must NOT be used as ML input features
    - legitimate users can share devices/IPs
    - abuse-ring members can also make legitimate transactions
    - fraud is not determined only by transaction amount
"""

from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_CUSTOMERS = 10_000
NUM_MERCHANTS = 500
NUM_DEVICES = 5_000
NUM_IPS = 8_000
NUM_TRANSACTIONS = 100_000

NUM_ABUSE_RINGS = 100

# Approximate overall fraud rate
TARGET_FRAUD_RATE = 0.04

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 8, 29)


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

LOCATIONS = [
    "Hyderabad",
    "Bengaluru",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Vijayawada",
    "Visakhapatnam",
]

MERCHANT_CATEGORIES = [
    "electronics",
    "fashion",
    "grocery",
    "food",
    "travel",
    "education",
    "healthcare",
    "entertainment",
    "utilities",
    "services",
]

PAYMENT_METHODS = [
    "card",
    "upi",
    "netbanking",
    "wallet",
]

DEVICE_TYPES = [
    "android",
    "ios",
    "web",
]

FRAUD_TYPES = [
    "amount_anomaly",
    "velocity_attack",
    "new_device",
    "new_location",
    "device_reuse",
    "ip_reuse",
    "abuse_ring",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_timestamp():
    """Return a random timestamp within the dataset period."""

    total_seconds = int(
        (END_DATE - START_DATE).total_seconds()
    )

    seconds = random.randint(
        0,
        total_seconds
    )

    return START_DATE + timedelta(
        seconds=seconds
    )


def random_amount(base_amount):
    """
    Generate a normal transaction amount around
    the customer's typical spending level.
    """

    amount = np.random.lognormal(
        mean=np.log(max(base_amount, 100)),
        sigma=0.45,
    )

    return round(
        float(np.clip(amount, 50, 100_000)),
        2,
    )


def generate_ip_address():
    """Generate a synthetic IPv4 address."""

    return (
        f"{random.randint(10, 223)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(1, 254)}"
    )


# ============================================================
# CUSTOMERS
# ============================================================

def generate_customers():

    print("Generating customers... - generate_dataset.py:175")

    customers = []

    for i in range(NUM_CUSTOMERS):

        customer_id = f"C{i + 1:05d}"

        account_age_days = random.randint(
            30,
            1500,
        )

        avg_transaction_amount = round(
            float(
                np.random.lognormal(
                    np.log(1800),
                    0.65,
                )
            ),
            2,
        )

        usual_location = random.choice(
            LOCATIONS
        )

        # Most customers have relatively normal
        # spending behavior.
        transaction_frequency = random.choice([
            "low",
            "medium",
            "high",
        ])

        customers.append({
            "customer_id": customer_id,
            "account_age_days": account_age_days,
            "avg_transaction_amount": avg_transaction_amount,
            "usual_location": usual_location,
            "transaction_frequency": transaction_frequency,
        })

    return pd.DataFrame(customers)


# ============================================================
# MERCHANTS
# ============================================================

def generate_merchants():

    print("Generating merchants... - generate_dataset.py:227")

    merchants = []

    for i in range(NUM_MERCHANTS):

        merchants.append({
            "merchant_id": f"M{i + 1:04d}",
            "merchant_category": random.choice(
                MERCHANT_CATEGORIES
            ),
        })

    return pd.DataFrame(merchants)


# ============================================================
# DEVICES
# ============================================================

def generate_devices():

    print("Generating devices... - generate_dataset.py:249")

    devices = []

    for i in range(NUM_DEVICES):

        devices.append({
            "device_id": f"D{i + 1:05d}",
            "device_type": random.choice(
                DEVICE_TYPES
            ),
        })

    return pd.DataFrame(devices)


# ============================================================
# IPS
# ============================================================

def generate_ips():

    print("Generating IP addresses... - generate_dataset.py:271")

    ips = []

    for i in range(NUM_IPS):

        ips.append({
            "ip_id": f"IP{i + 1:05d}",
            "ip_address": generate_ip_address(),
        })

    return pd.DataFrame(ips)


# ============================================================
# CUSTOMER DEVICE ASSIGNMENT
# ============================================================

def assign_customer_devices(
    customers,
    devices,
):
    """
    Assign normal devices to customers.

    Some devices are naturally shared by multiple
    customers. This is intentional because shared
    infrastructure is not automatically fraudulent.
    """

    print("Assigning customer devices... - generate_dataset.py:301")

    device_ids = devices[
        "device_id"
    ].tolist()

    assignments = {}

    for customer_id in customers[
        "customer_id"
    ]:

        number_of_devices = random.choices(
            [1, 2, 3],
            weights=[0.60, 0.32, 0.08],
            k=1,
        )[0]

        assignments[customer_id] = random.sample(
            device_ids,
            number_of_devices,
        )

    return assignments


# ============================================================
# CUSTOMER IP ASSIGNMENT
# ============================================================

def assign_customer_ips(
    customers,
    ips,
):
    """
    Assign normal IP addresses.

    Some IPs naturally serve multiple customers.
    """

    print("Assigning customer IP addresses... - generate_dataset.py:341")

    ip_ids = ips[
        "ip_id"
    ].tolist()

    assignments = {}

    for customer_id in customers[
        "customer_id"
    ]:

        number_of_ips = random.choices(
            [1, 2, 3, 4],
            weights=[0.50, 0.32, 0.13, 0.05],
            k=1,
        )[0]

        assignments[customer_id] = random.sample(
            ip_ids,
            number_of_ips,
        )

    return assignments


# ============================================================
# ABUSE RINGS
# ============================================================

def generate_abuse_rings(
    customers,
    devices,
    ips,
):
    """
    Create coordinated abuse groups.

    Each ring:
        - contains multiple customers
        - shares one device
        - shares one IP
        - will later generate coordinated transactions
    """

    print("Creating coordinated abuse rings... - generate_dataset.py:386")

    customer_ids = customers[
        "customer_id"
    ].tolist()

    device_ids = devices[
        "device_id"
    ].tolist()

    ip_ids = ips[
        "ip_id"
    ].tolist()

    available_customers = customer_ids.copy()

    random.shuffle(
        available_customers
    )

    rings = []

    cursor = 0

    for ring_number in range(
        1,
        NUM_ABUSE_RINGS + 1,
    ):

        ring_size = random.randint(
            5,
            10,
        )

        if cursor + ring_size > len(
            available_customers
        ):
            break

        ring_customers = available_customers[
            cursor:
            cursor + ring_size
        ]

        cursor += ring_size

        shared_device = random.choice(
            device_ids
        )

        shared_ip = random.choice(
            ip_ids
        )

        ring_id = f"R{ring_number:04d}"

        for customer_id in ring_customers:

            rings.append({
                "ring_id": ring_id,
                "customer_id": customer_id,
                "shared_device_id": shared_device,
                "shared_ip_id": shared_ip,
            })

    return pd.DataFrame(rings)


# ============================================================
# TRANSACTION GENERATOR
# ============================================================

def generate_transactions(
    customers,
    merchants,
    devices,
    ips,
    customer_devices,
    customer_ips,
    abuse_rings,
):
    """
    Generate transactions in two stages:

    Stage 1:
        Normal transactions

    Stage 2:
        Inject structured fraud patterns

    This prevents the dataset from being entirely
    determined by random fraud labels.
    """

    print("Generating transactions... - generate_dataset.py:480")

    customer_records = customers.to_dict(
        "records"
    )

    merchant_ids = merchants[
        "merchant_id"
    ].tolist()

    device_ids = devices[
        "device_id"
    ].tolist()

    ip_ids = ips[
        "ip_id"
    ].tolist()

    transactions = []

    # --------------------------------------------------------
    # Build ring lookup
    # --------------------------------------------------------

    ring_lookup = {}

    for record in abuse_rings.to_dict(
        "records"
    ):

        ring_lookup[
            record["customer_id"]
        ] = record

    # --------------------------------------------------------
    # Transaction count per customer
    # --------------------------------------------------------

    customer_weights = []

    for customer in customer_records:

        frequency = customer[
            "transaction_frequency"
        ]

        if frequency == "low":
            weight = 0.6

        elif frequency == "medium":
            weight = 1.0

        else:
            weight = 1.6

        customer_weights.append(
            weight
        )

    customer_probability = (
        np.array(customer_weights)
        / sum(customer_weights)
    )

    # --------------------------------------------------------
    # Generate base transactions
    # --------------------------------------------------------

    for i in range(
        NUM_TRANSACTIONS
    ):

        customer_index = rng.choice(
            len(customer_records),
            p=customer_probability,
        )

        customer = customer_records[
            customer_index
        ]

        customer_id = customer[
            "customer_id"
        ]

        amount = random_amount(
            customer[
                "avg_transaction_amount"
            ]
        )

        device_id = random.choice(
            customer_devices[
                customer_id
            ]
        )

        ip_id = random.choice(
            customer_ips[
                customer_id
            ]
        )

        location = customer[
            "usual_location"
        ]

        transactions.append({

            "transaction_id":
                f"TX{i + 1:07d}",

            "customer_id":
                customer_id,

            "merchant_id":
                random.choice(
                    merchant_ids
                ),

            "amount":
                amount,

            "timestamp":
                random_timestamp(),

            "payment_method":
                random.choice(
                    PAYMENT_METHODS
                ),

            "device_id":
                device_id,

            "ip_id":
                ip_id,

            "location":
                location,

            "fraud_type":
                "none",

            "ring_id":
                None,

            "is_fraud":
                0,
        })

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    transactions = pd.DataFrame(
        transactions
    )

    # --------------------------------------------------------
    # Fraud target
    # --------------------------------------------------------

    target_fraud_count = int(
        NUM_TRANSACTIONS
        * TARGET_FRAUD_RATE
    )

    fraud_indices = set(
        random.sample(
            range(NUM_TRANSACTIONS),
            target_fraud_count,
        )
    )

    # --------------------------------------------------------
    # Select fraud types
    # --------------------------------------------------------

    fraud_type_distribution = {

        "amount_anomaly": 0.16,

        "velocity_attack": 0.16,

        "new_device": 0.14,

        "new_location": 0.14,

        "device_reuse": 0.12,

        "ip_reuse": 0.12,

        "abuse_ring": 0.16,
    }

    fraud_types = list(
        fraud_type_distribution.keys()
    )

    fraud_probabilities = list(
        fraud_type_distribution.values()
    )

    # --------------------------------------------------------
    # Candidate ring customers
    # --------------------------------------------------------

    ring_customer_ids = set(
        ring_lookup.keys()
    )

    # --------------------------------------------------------
    # Inject fraud
    # --------------------------------------------------------

    for index in fraud_indices:

        row = transactions.loc[
            index
        ]

        customer_id = row[
            "customer_id"
        ]

        customer = customers[
            customers["customer_id"]
            == customer_id
        ].iloc[0]

        fraud_type = random.choices(
            fraud_types,
            weights=fraud_probabilities,
            k=1,
        )[0]

        ring_id = None

        # ====================================================
        # AMOUNT ANOMALY
        # ====================================================

        if fraud_type == "amount_anomaly":

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            # Some legitimate transactions can also
            # be large. Fraud is not always extreme.
            multiplier = random.uniform(
                3.0,
                8.0,
            )

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount * multiplier,
                2,
            )

        # ====================================================
        # VELOCITY ATTACK
        # ====================================================

        elif fraud_type == "velocity_attack":

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount
                * random.uniform(
                    1.2,
                    4.0,
                ),
                2,
            )

        # ====================================================
        # NEW DEVICE
        # ====================================================

        elif fraud_type == "new_device":

            current_device = row[
                "device_id"
            ]

            candidate_devices = [
                device
                for device in device_ids
                if device != current_device
            ]

            transactions.at[
                index,
                "device_id"
            ] = random.choice(
                candidate_devices
            )

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount
                * random.uniform(
                    2.0,
                    5.0,
                ),
                2,
            )

        # ====================================================
        # NEW LOCATION
        # ====================================================

        elif fraud_type == "new_location":

            normal_location = customer[
                "usual_location"
            ]

            unusual_locations = [
                location
                for location in LOCATIONS
                if location != normal_location
            ]

            transactions.at[
                index,
                "location"
            ] = random.choice(
                unusual_locations
            )

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount
                * random.uniform(
                    1.5,
                    5.0,
                ),
                2,
            )

        # ====================================================
        # DEVICE REUSE
        # ====================================================

        elif fraud_type == "device_reuse":

            if not abuse_rings.empty:

                ring_record = random.choice(
                    abuse_rings.to_dict(
                        "records"
                    )
                )

                transactions.at[
                    index,
                    "device_id"
                ] = ring_record[
                    "shared_device_id"
                ]

                ring_id = ring_record[
                    "ring_id"
                ]

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount
                * random.uniform(
                    1.5,
                    5.0,
                ),
                2,
            )

        # ====================================================
        # IP REUSE
        # ====================================================

        elif fraud_type == "ip_reuse":

            if not abuse_rings.empty:

                ring_record = random.choice(
                    abuse_rings.to_dict(
                        "records"
                    )
                )

                transactions.at[
                    index,
                    "ip_id"
                ] = ring_record[
                    "shared_ip_id"
                ]

                ring_id = ring_record[
                    "ring_id"
                ]

            normal_amount = customer[
                "avg_transaction_amount"
            ]

            transactions.at[
                index,
                "amount"
            ] = round(
                normal_amount
                * random.uniform(
                    1.5,
                    5.0,
                ),
                2,
            )

        # ====================================================
        # ABUSE RING
        # ====================================================

        elif fraud_type == "abuse_ring":

            if ring_customer_ids:

                ring_record = random.choice(
                    abuse_rings.to_dict(
                        "records"
                    )
                )

                transactions.at[
                    index,
                    "customer_id"
                ] = ring_record[
                    "customer_id"
                ]

                transactions.at[
                    index,
                    "device_id"
                ] = ring_record[
                    "shared_device_id"
                ]

                transactions.at[
                    index,
                    "ip_id"
                ] = ring_record[
                    "shared_ip_id"
                ]

                ring_id = ring_record[
                    "ring_id"
                ]

                # Ring transactions are not
                # necessarily enormous.
                ring_customer = customers[
                    customers["customer_id"]
                    == ring_record[
                        "customer_id"
                    ]
                ].iloc[0]

                normal_amount = ring_customer[
                    "avg_transaction_amount"
                ]

                transactions.at[
                    index,
                    "amount"
                ] = round(
                    normal_amount
                    * random.uniform(
                        1.5,
                        6.0,
                    ),
                    2,
                )

        # ----------------------------------------------------
        # Mark fraud
        # ----------------------------------------------------

        transactions.at[
            index,
            "fraud_type"
        ] = fraud_type

        transactions.at[
            index,
            "ring_id"
        ] = ring_id

        transactions.at[
            index,
            "is_fraud"
        ] = 1

    # --------------------------------------------------------
    # Add coordinated ring activity
    # --------------------------------------------------------

    transactions = inject_ring_activity(
        transactions,
        customers,
        abuse_rings,
        target_additional_transactions=2500,
    )

    return transactions


# ============================================================
# RING ACTIVITY INJECTION
# ============================================================

def inject_ring_activity(
    transactions,
    customers,
    abuse_rings,
    target_additional_transactions=2500,
):
    """
    Add coordinated transactions for abuse rings.

    Important:
        Some ring-member transactions are legitimate.

    This prevents:
        ring membership == fraud

    from becoming a trivial shortcut.
    """

    print("Injecting coordinated ring activity... - generate_dataset.py:1044")

    if abuse_rings.empty:
        return transactions

    ring_records = abuse_rings.to_dict(
        "records"
    )

    new_transactions = []

    existing_transaction_count = len(
        transactions
    )

    for i in range(
        target_additional_transactions
    ):

        ring = random.choice(
            ring_records
        )

        customer_id = ring[
            "customer_id"
        ]

        customer = customers[
            customers["customer_id"]
            == customer_id
        ].iloc[0]

        normal_amount = customer[
            "avg_transaction_amount"
        ]

        # About 25% of ring activity is
        # legitimate-looking behavior.
        is_ring_fraud = (
            random.random() < 0.75
        )

        if is_ring_fraud:

            amount = normal_amount * random.uniform(
                1.5,
                6.0,
            )

            fraud_type = "abuse_ring"

            is_fraud = 1

        else:

            amount = random_amount(
                normal_amount
            )

            fraud_type = "none"

            is_fraud = 0

        # Cluster timestamps around random
        # points to create temporal relationships.
        base_time = random_timestamp()

        timestamp = base_time + timedelta(
            minutes=random.randint(
                -10,
                10,
            )
        )

        new_transactions.append({

            "transaction_id":
                f"RTX{i + 1:07d}",

            "customer_id":
                customer_id,

            "merchant_id":
                f"M{random.randint(1, NUM_MERCHANTS):04d}",

            "amount":
                round(
                    float(
                        np.clip(
                            amount,
                            50,
                            100_000,
                        )
                    ),
                    2,
                ),

            "timestamp":
                timestamp,

            "payment_method":
                random.choice(
                    PAYMENT_METHODS
                ),

            "device_id":
                ring[
                    "shared_device_id"
                ],

            "ip_id":
                ring[
                    "shared_ip_id"
                ],

            "location":
                customer[
                    "usual_location"
                ],

            "fraud_type":
                fraud_type,

            "ring_id":
                ring[
                    "ring_id"
                ]
                if is_ring_fraud
                else None,

            "is_fraud":
                is_fraud,
        })

    additional = pd.DataFrame(
        new_transactions
    )

    combined = pd.concat(
        [
            transactions,
            additional,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Keep dataset approximately around 100k records
    # --------------------------------------------------------

    # We intentionally generated additional ring activity.
    # Now sample back to exactly 100,000 records while
    # preserving all fraud records and a substantial amount
    # of ring activity.
    fraud_rows = combined[
        combined["is_fraud"] == 1
    ]

    legitimate_rows = combined[
        combined["is_fraud"] == 0
    ]

    target_total = NUM_TRANSACTIONS

    target_fraud = int(
        target_total
        * TARGET_FRAUD_RATE
    )

    # If ring injection created too many fraud examples,
    # sample fraud records down.
    if len(fraud_rows) > target_fraud:

        fraud_rows = fraud_rows.sample(
            n=target_fraud,
            random_state=RANDOM_SEED,
        )

    remaining_count = (
        target_total
        - len(fraud_rows)
    )

    legitimate_rows = legitimate_rows.sample(
        n=remaining_count,
        random_state=RANDOM_SEED,
    )

    final = pd.concat(
        [
            fraud_rows,
            legitimate_rows,
        ],
        ignore_index=True,
    )

    final = final.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )

    # Reassign unique transaction IDs.
    final[
        "transaction_id"
    ] = [
        f"TX{i + 1:07d}"
        for i in range(len(final))
    ]

    return final


# ============================================================
# SAVE DATA
# ============================================================

def save_datasets(
    customers,
    merchants,
    devices,
    ips,
    abuse_rings,
    transactions,
):

    print("Saving datasets... - generate_dataset.py:1271")

    customers.to_csv(
        RAW_DATA_DIR / "customers.csv",
        index=False,
    )

    merchants.to_csv(
        RAW_DATA_DIR / "merchants.csv",
        index=False,
    )

    devices.to_csv(
        RAW_DATA_DIR / "devices.csv",
        index=False,
    )

    ips.to_csv(
        RAW_DATA_DIR / "ips.csv",
        index=False,
    )

    abuse_rings.to_csv(
        RAW_DATA_DIR / "abuse_rings.csv",
        index=False,
    )

    transactions.to_csv(
        RAW_DATA_DIR / "transactions.csv",
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    customers,
    merchants,
    devices,
    ips,
    abuse_rings,
    transactions,
):

    print()
    print("= - generate_dataset.py:1318" * 70)
    print("DATASET GENERATION COMPLETE - generate_dataset.py:1319")
    print("= - generate_dataset.py:1320" * 70)
    print()

    print(
        f"Customers:       {len(customers):,}"
    )

    print(
        f"Merchants:       {len(merchants):,}"
    )

    print(
        f"Devices:         {len(devices):,}"
    )

    print(
        f"IP addresses:    {len(ips):,}"
    )

    print(
        f"Abuse ring rows: {len(abuse_rings):,}"
    )

    print(
        f"Transactions:    {len(transactions):,}"
    )

    fraud_count = int(
        transactions["is_fraud"].sum()
    )

    fraud_rate = (
        fraud_count
        / len(transactions)
        * 100
    )

    print(
        f"Fraud records:   {fraud_count:,}"
    )

    print(
        f"Fraud rate:      {fraud_rate:.2f}%"
    )

    print()
    print("Fraud type distribution: - generate_dataset.py:1366")

    fraud_distribution = (
        transactions[
            transactions["is_fraud"] == 1
        ]["fraud_type"]
        .value_counts()
    )

    print(
        fraud_distribution.to_string()
    )

    print()
    print(
        f"Files saved to:\n"
        f"{RAW_DATA_DIR}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("= - generate_dataset.py:1395" * 70)
    print("RAZORSHIELD AI  DATA GENERATOR V2 - generate_dataset.py:1396")
    print("= - generate_dataset.py:1397" * 70)
    print()

    customers = generate_customers()

    merchants = generate_merchants()

    devices = generate_devices()

    ips = generate_ips()

    customer_devices = assign_customer_devices(
        customers,
        devices,
    )

    customer_ips = assign_customer_ips(
        customers,
        ips,
    )

    abuse_rings = generate_abuse_rings(
        customers,
        devices,
        ips,
    )

    transactions = generate_transactions(
        customers,
        merchants,
        devices,
        ips,
        customer_devices,
        customer_ips,
        abuse_rings,
    )

    save_datasets(
        customers,
        merchants,
        devices,
        ips,
        abuse_rings,
        transactions,
    )

    print_summary(
        customers,
        merchants,
        devices,
        ips,
        abuse_rings,
        transactions,
    )


if __name__ == "__main__":
    main()