"""
data_processing.py
------------------
Combined Tasks 3 + 4: Feature Engineering + Proxy Target Creation
Run with: python src/data_processing.py
"""

import logging
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw", "data.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_data_with_target.csv")


class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"], utc=True)
        X["transaction_hour"] = X["TransactionStartTime"].dt.hour
        X["transaction_day"] = X["TransactionStartTime"].dt.day
        X["transaction_month"] = X["TransactionStartTime"].dt.month
        X["transaction_year"] = X["TransactionStartTime"].dt.year
        return X.drop(columns=["TransactionStartTime"])


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        agg = X.groupby("CustomerId").agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min"),
            avg_transaction_hour=("transaction_hour", "mean"),
            unique_months_active=("transaction_month", "nunique"),
        ).reset_index()
        
        agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)
        return agg


class ProxyTargetCreator(BaseEstimator, TransformerMixin):
    """Task 4: Create is_high_risk using RFM + KMeans"""
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        X = X.copy()
        # Use key behavioral features
        features = ['transaction_count', 'total_transaction_amount', 'avg_transaction_amount']
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        X['cluster'] = kmeans.fit_predict(scaled)
        
        # Identify high-risk cluster (lowest engagement)
        cluster_profile = X.groupby('cluster')[features].mean()
        high_risk_cluster = cluster_profile.sort_values(by=['transaction_count', 'total_transaction_amount']).index[0]
        
        X['is_high_risk'] = (X['cluster'] == high_risk_cluster).astype(int)
        X = X.drop(columns=['cluster'])
        
        logger.info(f"High-risk customers: {X['is_high_risk'].sum()} ({X['is_high_risk'].mean():.2%})")
        return X


def run_pipeline():
    logger.info("=== Running Full Pipeline (Task 3 + Task 4) ===")
    
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Raw data shape: {df.shape}")
    
    pipeline = Pipeline([
        ('date_features', DateFeatures()),
        ('aggregate_features', AggregateFeatures()),
        ('proxy_target', ProxyTargetCreator())
    ])
    
    processed_df = pipeline.fit_transform(df)
    
    # Save
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    logger.info(f"✅ Saved processed data with target: {PROCESSED_DATA_PATH}")
    logger.info(f"Final shape: {processed_df.shape}")
    logger.info(f"Columns: {list(processed_df.columns)}")
    
    return processed_df


if __name__ == "__main__":
    run_pipeline()