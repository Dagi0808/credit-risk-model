"""
src/train.py
Task 5: Model Training and Tracking with MLflow
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

# ========================= CONFIG =========================
RANDOM_STATE = 42

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_Risk_Task5")

# Updated to match your actual processed file from data_processing.py
DATA_PATH = "data/processed/processed_data.csv"   # ← Changed to match your file
TARGET_COL = "is_high_risk"


def load_data():
    """Load the processed dataset from Task 4"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
        if TARGET_COL in df.columns:
            print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")
        else:
            print(f"⚠️  Warning: Target column '{TARGET_COL}' not found!")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed file not found at {DATA_PATH}\n"
            "Please run: python src/data_processing.py first!"
        )


def split_data(df):
    """Split data into train/test sets"""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset!")
    
    # Drop non-feature columns
    drop_cols = [TARGET_COL]
    if 'CustomerId' in df.columns:
        drop_cols.append('CustomerId')
    
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    
    return train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y
    )


def get_models():
    """Define models for training"""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=2000, 
            random_state=RANDOM_STATE, 
            class_weight='balanced'
        ),
        "Decision_Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, 
            class_weight='balanced'
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, 
            random_state=RANDOM_STATE, 
            class_weight='balanced'
        )
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average='weighted', zero_division=0),
        "recall": recall_score(y_test, preds, average='weighted', zero_division=0),
        "f1": f1_score(y_test, preds, average='weighted', zero_division=0),
    }
    if probs is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
    
    return metrics


def log_to_mlflow(model, name, metrics, params):
    """Log model and metrics to MLflow"""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        print(f"📊 Logged → {name} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")


def main():
    # Load data
    df = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Get models
    models = get_models()
    best_model = None
    best_f1 = 0
    best_name = ""

    print("\n🚀 Starting model training...\n")

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        log_to_mlflow(model, name, metrics, model.get_params())

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = name

    print(f"\n🏆 Best performing model: {best_name} (F1 = {best_f1:.4f})")

    # Register the best model
    with mlflow.start_run(run_name="Best_Model_Registration"):
        mlflow.sklearn.log_model(
            best_model, 
            "best_model", 
            registered_model_name="Credit_Risk_Best_Model"
        )
        print("✅ Best model successfully registered in MLflow Model Registry!")

    print("\n🎉 Task 5 Completed Successfully!")


if __name__ == "__main__":
    main()