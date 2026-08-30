"""
RazorShield AI - Historical Graph Cluster Analysis

Creates a genuine relational signal based on historical connections
between customers through shared devices and IP addresses.

Important:
    - Transactions are processed chronologically.
    - The current transaction is NOT used to create its own cluster.
    - Only previously observed customer-device and customer-IP
      relationships are considered.
    - This is an analysis feature and does NOT modify the existing
      champion model.
    - The existing 44-feature model remains unchanged.

Output:
    data/processed/graph_cluster_features.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transactions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_cluster_features.csv"
)


# ============================================================
# UNION-FIND / DISJOINT SET
# ============================================================

class UnionFind:
    """
    Lightweight disjoint-set structure used to maintain
    connected customer components as historical relationships
    are observed.
    """

    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, node):
        """
        Add a customer if it does not already exist.
        """

        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node):
        """
        Find the root of a customer component.
        """

        if self.parent[node] != node:

            self.parent[node] = self.find(
                self.parent[node]
            )

        return self.parent[node]

    def union(self, first, second):
        """
        Connect two customers.

        Uses union-by-size to keep the tree shallow.
        """

        self.add(first)
        self.add(second)

        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        # Attach smaller component to larger component.
        if (
            self.size[root_first]
            < self.size[root_second]
        ):
            root_first, root_second = (
                root_second,
                root_first,
            )

        self.parent[root_second] = root_first

        self.size[root_first] += (
            self.size[root_second]
        )

    def component_size(self, node):
        """
        Return the size of the customer's historical
        connected component.
        """

        if node not in self.parent:
            return 1

        root = self.find(node)

        return self.size[root]


# ============================================================
# LOAD DATA
# ============================================================

def load_transactions():

    print("= - build_graph_clusters.py:135" * 70)
    print("RAZORSHIELD AI  HISTORICAL GRAPH CLUSTER ANALYSIS - build_graph_clusters.py:136")
    print("= - build_graph_clusters.py:137" * 70)
    print()

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Transaction dataset not found:\n"
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    required_columns = [
        "transaction_id",
        "timestamp",
        "customer_id",
        "device_id",
        "ip_id",
        "is_fraud",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # The source dataset may be shuffled.
    #
    # Sort by real transaction time before constructing
    # historical relationships.
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Transactions: "
        f"{len(df):,}"
    )

    print(
        f"Time range: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    print()

    return df


# ============================================================
# BUILD HISTORICAL GRAPH FEATURES
# ============================================================

def build_graph_features(df):

    print("= - build_graph_clusters.py:221" * 70)
    print("1. BUILDING HISTORICAL CUSTOMER GRAPH - build_graph_clusters.py:222")
    print("= - build_graph_clusters.py:223" * 70)
    print()

    union_find = UnionFind()

    # --------------------------------------------------------
    # Historical mappings
    #
    # device -> customers previously seen
    # ip     -> customers previously seen
    #
    # These mappings are updated ONLY after the current
    # transaction's feature values have been calculated.
    # --------------------------------------------------------

    device_customers = {}
    ip_customers = {}

    cluster_sizes = []
    cluster_connections = []
    device_cluster_sizes = []
    ip_cluster_sizes = []

    # Track whether a customer has appeared previously.
    seen_customers = set()

    total = len(df)

    for position, row in enumerate(
        df.itertuples(
            index=False
        ),
        start=1,
    ):

        customer = row.customer_id
        device = row.device_id
        ip = row.ip_id

        # ----------------------------------------------------
        # Make sure customer exists in the graph.
        # ----------------------------------------------------

        union_find.add(
            customer
        )

        # ----------------------------------------------------
        # BEFORE updating the graph:
        #
        # Calculate the historical component size.
        # ----------------------------------------------------

        current_cluster_size = (
            union_find.component_size(
                customer
            )
        )

        cluster_sizes.append(
            current_cluster_size
        )

        # ----------------------------------------------------
        # Historical device-connected customers
        # ----------------------------------------------------

        previous_device_customers = (
            device_customers.get(
                device,
                set(),
            )
        )

        # Exclude current customer if they already appear
        # in the historical set.
        historical_device_customers = (
            previous_device_customers
            - {customer}
        )

        device_cluster_size = 1

        if historical_device_customers:

            device_roots = {
                union_find.find(
                    other_customer
                )
                for other_customer
                in historical_device_customers
            }

            device_cluster_size = max(
                [
                    union_find.size[root]
                    for root in device_roots
                ],
                default=1,
            )

        device_cluster_sizes.append(
            device_cluster_size
        )

        # ----------------------------------------------------
        # Historical IP-connected customers
        # ----------------------------------------------------

        previous_ip_customers = (
            ip_customers.get(
                ip,
                set(),
            )
        )

        historical_ip_customers = (
            previous_ip_customers
            - {customer}
        )

        ip_cluster_size = 1

        if historical_ip_customers:

            ip_roots = {
                union_find.find(
                    other_customer
                )
                for other_customer
                in historical_ip_customers
            }

            ip_cluster_size = max(
                [
                    union_find.size[root]
                    for root in ip_roots
                ],
                default=1,
            )

        ip_cluster_sizes.append(
            ip_cluster_size
        )

        # ----------------------------------------------------
        # Number of historical customers directly connected
        # through this transaction's device or IP.
        # ----------------------------------------------------

        directly_connected = (
            historical_device_customers
            | historical_ip_customers
        )

        cluster_connections.append(
            len(directly_connected)
        )

        # ----------------------------------------------------
        # NOW update the graph.
        #
        # This happens AFTER feature calculation, ensuring
        # that the current transaction cannot influence its
        # own historical features.
        # ----------------------------------------------------

        for other_customer in (
            historical_device_customers
        ):

            union_find.union(
                customer,
                other_customer,
            )

        for other_customer in (
            historical_ip_customers
        ):

            union_find.union(
                customer,
                other_customer,
            )

        # ----------------------------------------------------
        # Register current customer against current device/IP.
        # ----------------------------------------------------

        if device not in device_customers:

            device_customers[
                device
            ] = set()

        device_customers[
            device
        ].add(
            customer
        )

        if ip not in ip_customers:

            ip_customers[
                ip
            ] = set()

        ip_customers[
            ip
        ].add(
            customer
        )

        seen_customers.add(
            customer
        )

        # ----------------------------------------------------
        # Progress indicator
        # ----------------------------------------------------

        if (
            position % 10000 == 0
            or position == total
        ):

            print(
                f"Processed "
                f"{position:,}/{total:,} "
                f"transactions"
            )

    result = pd.DataFrame(
        {
            "transaction_id":
                df["transaction_id"].values,

            "timestamp":
                df["timestamp"].values,

            "customer_id":
                df["customer_id"].values,

            "is_fraud":
                df["is_fraud"].values,

            "graph_customer_cluster_size_before":
                cluster_sizes,

            "graph_device_cluster_size_before":
                device_cluster_sizes,

            "graph_ip_cluster_size_before":
                ip_cluster_sizes,

            "graph_direct_connections_before":
                cluster_connections,
        }
    )

    print()

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    result,
):

    print("= - build_graph_clusters.py:496" * 70)
    print("2. VALIDATING GRAPH FEATURES - build_graph_clusters.py:497")
    print("= - build_graph_clusters.py:498" * 70)
    print()

    feature_columns = [
        "graph_customer_cluster_size_before",
        "graph_device_cluster_size_before",
        "graph_ip_cluster_size_before",
        "graph_direct_connections_before",
    ]

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        result[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_values}"
    )

    if missing_values != 0:

        raise ValueError(
            "Graph features contain missing values."
        )

    # --------------------------------------------------------
    # Non-negative values
    # --------------------------------------------------------

    negative_values = (
        result[
            feature_columns
        ]
        < 0
    ).sum().sum()

    print(
        f"Negative feature values: "
        f"{negative_values}"
    )

    if negative_values != 0:

        raise ValueError(
            "Graph features contain negative values."
        )

    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    chronological = (
        result["timestamp"]
        .is_monotonic_increasing
    )

    print(
        f"Chronological order: "
        f"{chronological}"
    )

    if not chronological:

        raise ValueError(
            "Graph output is not chronologically ordered."
        )

    # --------------------------------------------------------
    # Current transaction leakage sanity check
    # --------------------------------------------------------

    first_transaction_cluster = (
        result.iloc[0][
            "graph_customer_cluster_size_before"
        ]
    )

    print(
        f"First transaction cluster size: "
        f"{first_transaction_cluster}"
    )

    if first_transaction_cluster != 1:

        raise ValueError(
            "First transaction should have a "
            "historical cluster size of 1."
        )

    print()

    print(
        "[PASS] Graph features passed validation."
    )

    print()


# ============================================================
# SAVE
# ============================================================

def save_results(
    result,
):

    print("= - build_graph_clusters.py:613" * 70)
    print("3. SAVING GRAPH CLUSTER FEATURES - build_graph_clusters.py:614")
    print("= - build_graph_clusters.py:615" * 70)
    print()

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    result,
):

    print("= - build_graph_clusters.py:639" * 70)
    print("4. GRAPH CLUSTER SUMMARY - build_graph_clusters.py:640")
    print("= - build_graph_clusters.py:641" * 70)
    print()

    columns = [
        "graph_customer_cluster_size_before",
        "graph_device_cluster_size_before",
        "graph_ip_cluster_size_before",
        "graph_direct_connections_before",
    ]

    print(
        result[
            columns
        ]
        .describe()
        .round(3)
        .to_string()
    )

    print()

    # --------------------------------------------------------
    # Transactions with historical relational connections
    # --------------------------------------------------------

    connected = (
        result[
            "graph_customer_cluster_size_before"
        ] > 1
    )

    print(
        f"Transactions with historical "
        f"customer-cluster connection: "
        f"{connected.sum():,}"
    )

    print(
        f"Percentage of transactions with "
        f"historical cluster connection: "
        f"{connected.mean() * 100:.2f}%"
    )

    print()

    # --------------------------------------------------------
    # Largest observed cluster
    # --------------------------------------------------------

    max_cluster = (
        result[
            "graph_customer_cluster_size_before"
        ].max()
    )

    print(
        f"Largest historical customer cluster: "
        f"{int(max_cluster)}"
    )

    print()

    # --------------------------------------------------------
    # Fraud comparison
    # --------------------------------------------------------

    fraud_mask = (
        result["is_fraud"] == 1
    )

    legitimate_mask = (
        result["is_fraud"] == 0
    )

    fraud_cluster_mean = (
        result.loc[
            fraud_mask,
            "graph_customer_cluster_size_before",
        ].mean()
    )

    legitimate_cluster_mean = (
        result.loc[
            legitimate_mask,
            "graph_customer_cluster_size_before",
        ].mean()
    )

    print(
        f"Average cluster size before "
        f"fraud transactions: "
        f"{fraud_cluster_mean:.3f}"
    )

    print(
        f"Average cluster size before "
        f"legitimate transactions: "
        f"{legitimate_cluster_mean:.3f}"
    )

    print()

    print(
        "These comparisons are descriptive only."
    )

    print(
        "They do not establish that cluster size "
        "causes fraud."
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load chronological transactions
    # --------------------------------------------------------

    transactions = (
        load_transactions()
    )

    # --------------------------------------------------------
    # Build graph-derived historical features
    # --------------------------------------------------------

    graph_features = (
        build_graph_features(
            transactions
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        graph_features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        graph_features
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        graph_features
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("= - build_graph_clusters.py:807" * 70)
    print(
        "HISTORICAL GRAPH CLUSTER ANALYSIS COMPLETE"
    )
    print("= - build_graph_clusters.py:811" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()