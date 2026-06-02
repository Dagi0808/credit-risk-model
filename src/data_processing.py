"""
data_processing.py
------------------
Feature engineering pipeline for credit risk modeling.
Covers Task 3 (feature engineering) and Task 4 (proxy target variable).

Run directly:
    python src/data_processing.py
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# File paths
RAW_DATA_PATH = os.path.join("data", "raw", "data.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_data.csv")


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse TransactionStartTime and extract hour, day, month, year.

    Why: The raw timestamp string is useless to a model.
    Extracted components let the model learn time-based patterns
    (e.g., late-night transactions may signal higher risk).
    """
    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True
    )

    df["transaction_hour"]  = df["TransactionStartTime"].dt.hour
    df["transaction_day"]   = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"]  = df["TransactionStartTime"].dt.year

    logger.info("Datetime features extracted: hour, day, month, year")
    return df


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse transaction-level rows into one row per customer.

    The model scores customers, not transactions.
    These aggregates summarize each customer's overall behavior.
    """
    logger.info("Creating aggregate features per customer...")

    agg = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min"),
            avg_transaction_hour=("transaction_hour", "mean"),
            unique_months_active=("transaction_month", "nunique"),
        )
        .reset_index()
    )

    # A customer with only 1 transaction has no std -> fill with 0
    agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)

    logger.info(f"Aggregate features created. Shape: {agg.shape}")
    return agg



def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proxy target variable for credit risk modeling.

    Since we do not have real default labels, we define risk
    using behavioral rules (common in credit scoring bootstrapping).
    """
    df = df.copy()

    logger.info("Creating proxy target variable (is_high_risk)...")

    # Compute thresholds using quantiles (data-driven rules)
    low_activity_threshold = df["transaction_count"].quantile(0.25)
    low_amount_threshold = df["avg_transaction_amount"].quantile(0.25)

    # Risk logic
    df["is_high_risk"] = (
        (df["transaction_count"] <= low_activity_threshold) |
        (df["avg_transaction_amount"] <= low_amount_threshold)
    ).astype(int)

    logger.info("Proxy target created successfully")
    logger.info(f"High risk ratio: {df['is_high_risk'].mean():.2f}")

    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns to numbers using Label Encoding.

    We use Label Encoding here for the transaction-level data before
    aggregation. One-Hot Encoding is handled inside the sklearn Pipeline
    for features that feed directly into the model.
    """
    df = df.copy()

    label_encode_cols = ["ProductCategory", "ChannelId", "PricingStrategy"]

    le = LabelEncoder()
    for col in label_encode_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Label encoded: {col}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.

    - Numerical columns   -> fill with median (robust to outliers)
    - Categorical columns -> fill with mode (most frequent value)

    From EDA we know missing values are minimal, but we handle
    them explicitly for production robustness.
    """
    df = df.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {col} with median: {median_val:.2f}")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {col} with mode: {mode_val}")

    logger.info("Missing value handling complete")
    return df


def build_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list
) -> ColumnTransformer:
    """
    Build a ColumnTransformer Pipeline:
      - Numerical  -> median imputation + StandardScaler
      - Categorical -> mode imputation  + OneHotEncoder

    This object is fitted ONCE on training data and applied
    identically to new data -- this prevents data leakage.

    Parameters
    ----------
    numerical_cols   : list of numerical column names
    categorical_cols : list of categorical column names

    Returns
    -------
    ColumnTransformer (unfitted)
    """

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop"
    )

    logger.info("Preprocessing pipeline built (unfitted)")
    return preprocessor


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 5
) -> pd.DataFrame:
    """
    Calculate WoE and IV for one numerical feature vs a binary target.

    WoE formula:
        WoE_i = ln( (Events_i / Total_Events) / (Non-Events_i / Total_Non-Events) )

    IV formula:
        IV = sum( (Events% - Non-Events%) * WoE )

    Parameters
    ----------
    df      : DataFrame with feature and target columns
    feature : numerical column name
    target  : binary target column name (0 = low risk, 1 = high risk)
    bins    : number of quantile bins

    Returns
    -------
    DataFrame with WoE and IV per bin
    """
    df = df[[feature, target]].copy().dropna()

    df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")

    total_events = df[target].sum()
    total_non_events = len(df) - total_events

    woe_table = []

    for bin_label, group in df.groupby("bin", observed=True):
        events = group[target].sum()
        non_events = len(group) - events

        # Add 0.5 smoothing to avoid log(0)
        dist_events     = (events + 0.5)     / (total_events + 0.5)
        dist_non_events = (non_events + 0.5) / (total_non_events + 0.5)

        woe    = np.log(dist_events / dist_non_events)
        iv_bin = (dist_events - dist_non_events) * woe

        woe_table.append({
            "feature":    feature,
            "bin":        str(bin_label),
            "events":     int(events),
            "non_events": int(non_events),
            "woe":        round(woe, 4),
            "iv_bin":     round(iv_bin, 4),
        })

    woe_df = pd.DataFrame(woe_table)
    woe_df["iv_total"] = woe_df["iv_bin"].sum()
    return woe_df


def calculate_all_iv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calculate IV for ALL numerical features and rank by predictive power.

    IV Interpretation:
        < 0.02   -> Useless
        0.02-0.1 -> Weak
        0.1-0.3  -> Medium
        0.3-0.5  -> Strong
        > 0.5    -> Suspicious (possible data leakage)

    Parameters
    ----------
    df     : DataFrame with features and target
    target : binary target column name

    Returns
    -------
    DataFrame sorted by IV descending
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != target]

    iv_summary = []

    for col in numerical_cols:
        try:
            woe_df = calculate_woe_iv(df, col, target)
            iv_val = woe_df["iv_total"].iloc[0]

            if iv_val < 0.02:
                power = "Useless"
            elif iv_val < 0.1:
                power = "Weak"
            elif iv_val < 0.3:
                power = "Medium"
            elif iv_val < 0.5:
                power = "Strong"
            else:
                power = "Suspicious"

            iv_summary.append({
                "feature":          col,
                "iv":               round(iv_val, 4),
                "predictive_power": power,
            })
        except Exception as e:
            logger.warning(f"Could not compute IV for {col}: {e}")

    iv_df = pd.DataFrame(iv_summary).sort_values("iv", ascending=False)
    return iv_df



def compute_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary values per customer.

    Recency   = days since last transaction (HIGH = bad, disengaged)
    Frequency = total number of transactions (LOW = bad)
    Monetary  = total amount spent (LOW = bad)

    Parameters
    ----------
    df            : raw transaction DataFrame (before aggregation)
    snapshot_date : reference date for recency calculation.
                    Defaults to 1 day after the latest transaction.

    Returns
    -------
    DataFrame with columns: CustomerId, Recency, Frequency, Monetary
    """
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True
    )

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    logger.info(f"Snapshot date for RFM: {snapshot_date}")

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    logger.info(f"RFM computed. Shape: {rfm.shape}")
    logger.info(f"Recency  — min: {rfm['Recency'].min()},  max: {rfm['Recency'].max()}")
    logger.info(f"Frequency — min: {rfm['Frequency'].min()}, max: {rfm['Frequency'].max()}")
    logger.info(f"Monetary  — min: {rfm['Monetary'].min():.0f}, max: {rfm['Monetary'].max():.0f}")

    return rfm


def assign_risk_labels(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cluster customers into n_clusters groups using K-Means on RFM features.
    Then label the most disengaged cluster as is_high_risk = 1.

    The high-risk cluster is identified as the one with:
        - Highest Recency   (hasn't transacted recently)
        - Lowest Frequency  (rarely transacts)
        - Lowest Monetary   (spends the least)

    We score each cluster with a composite rank and pick the worst.

    Parameters
    ----------
    rfm          : DataFrame with Recency, Frequency, Monetary columns
    n_clusters   : number of K-Means clusters (default 3)
    random_state : for reproducibility (required by Basel II)

    Returns
    -------
    DataFrame with columns: CustomerId, Recency, Frequency, Monetary,
                             cluster, is_high_risk
    """
    rfm = rfm.copy()

    # Scale RFM before clustering so no single feature dominates
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # K-Means with fixed random_state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Analyze each cluster's average RFM values
    cluster_profile = rfm.groupby("cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    logger.info(f"\nCluster profiles:\n{cluster_profile.round(2)}")

    # Rank clusters to find the most disengaged (highest risk)
    cluster_profile["risk_score"] = (
        cluster_profile["Recency"].rank(ascending=False)   # high recency = bad
        + cluster_profile["Frequency"].rank(ascending=True) # low frequency = bad
        + cluster_profile["Monetary"].rank(ascending=True)  # low monetary = bad
    )

    high_risk_cluster = int(cluster_profile["risk_score"].idxmax())

    logger.info(f"\nCluster risk scores:\n{cluster_profile['risk_score']}")
    logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")

    # Assign binary label: 1 = high risk, 0 = low risk
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    high_risk_count = rfm["is_high_risk"].sum()
    high_risk_pct   = rfm["is_high_risk"].mean() * 100
    logger.info(f"High-risk customers: {high_risk_count} ({high_risk_pct:.1f}%)")

    return rfm


def run_full_pipeline(
    raw_path: str = RAW_DATA_PATH,
    output_path: str = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """
    Complete pipeline including RFM target variable (Tasks 3 + 4).

    Steps:
    1. Load raw data
    2. Extract datetime features          (Task 3)
    3. Encode categoricals                (Task 3)
    4. Build per-customer aggregate features (Task 3)
    5. Handle missing values              (Task 3)
    6. Compute RFM metrics                (Task 4)
    7. Cluster with K-Means               (Task 4)
    8. Assign is_high_risk label          (Task 4)
    9. Merge everything and save          (Task 4)

    Returns
    -------
    Final DataFrame with is_high_risk column ready for model training
    """
    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE (Tasks 3 + 4)")
    logger.info("=" * 60)

    # Load raw data once
    logger.info(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    logger.info(f"Raw data shape: {raw_df.shape}")

    # ── Task 3: Feature Engineering ───────────────────────────
    logger.info("\n--- Task 3: Feature Engineering ---")
    df = extract_datetime_features(raw_df)
    df = encode_categoricals(df)
    customer_df = create_aggregate_features(df)
    customer_df = handle_missing_values(customer_df)

    # ── Task 4: RFM + Risk Labels ──────────────────────────────
    # Uses raw_df because compute_rfm needs original timestamps
    logger.info("\n--- Task 4: RFM + Proxy Target Variable ---")
    rfm = compute_rfm(raw_df)
    rfm_labeled = assign_risk_labels(rfm)

    # Only keep what we need from RFM result
    rfm_to_merge = rfm_labeled[[
        "CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"
    ]]

    # Merge RFM features + label into the customer feature table
    final_df = customer_df.merge(rfm_to_merge, on="CustomerId", how="left")

    # Save final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Final shape:     {final_df.shape}")
    logger.info(f"Columns:         {list(final_df.columns)}")
    logger.info(f"High-risk rate:  {final_df['is_high_risk'].mean():.2%}")
    logger.info(f"Saved to:        {output_path}")
    logger.info("=" * 60)

    return final_df


if __name__ == "__main__":
    final = run_full_pipeline()

    print("\n--- Sample of final data ---")
    print(final.head())
    print(f"\nShape: {final.shape}")
    print(f"\nColumns:\n{list(final.columns)}")
    print(f"\nTarget distribution:")
    print(final["is_high_risk"].value_counts())
    print(f"\nHigh-risk rate: {final['is_high_risk'].mean():.2%}")