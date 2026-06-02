import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42

mlflow.set_experiment("task_5_model_training")


# =========================
# STEP 2 — DATA PREPARATION
# =========================

def load_data(path: str):
    return pd.read_csv(path)


def split_data(df, target_column):
    if target_column not in df.columns:
        raise ValueError("Target column not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = y if y.nunique() <= 20 else None

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify
    )


# =========================
# STEP 3 — MODELS
# =========================

def get_models():
    return {
        "log_reg": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE)
    }


# =========================
# STEP 4 — TUNING
# =========================

def tune_model(model, params, X_train, y_train, method="grid"):
    if method == "grid":
        search = GridSearchCV(model, params, cv=3, scoring="f1", n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, params, cv=3, scoring="f1", n_jobs=-1)

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# =========================
# STEP 5 — EVALUATION
# =========================

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
        "f1": f1_score(y_test, preds, average="weighted"),
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
        except:
            pass

    return metrics


# =========================
# STEP 6 — MLflow LOGGING
# =========================

def log_model(model, name, metrics, params):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")


# =========================
# STEP 7 — MAIN PIPELINE
# =========================

def main():

    df = load_data("data.csv")

    X_train, X_test, y_train, y_test = split_data(df, "target")

    models = get_models()

    best_model = None
    best_score = 0
    best_name = ""

    # Train baseline models
    for name, model in models.items():

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        log_model(model, name, metrics, model.get_params())

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model
            best_name = name

    print(f"Best model: {best_name}")

    # Hyperparameter tuning ONLY for Random Forest
    if best_name == "random_forest":

        tuned_model, best_params = tune_model(
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10]
            },
            X_train,
            y_train,
            method="grid"
        )

        tuned_metrics = evaluate(tuned_model, X_test, y_test)

        log_model(tuned_model, "tuned_random_forest", tuned_metrics, best_params)

        best_model = tuned_model

    # Register best model
    mlflow.sklearn.log_model(
        best_model,
        "best_model",
        registered_model_name="BestModelTask5"
    )

    print("Training complete. Model registered in MLflow.")


if __name__ == "__main__":
    main()