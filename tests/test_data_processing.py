import pandas as pd
from src.train import split_data


def test_split_data_shapes():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [6, 5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_data(df, "target")

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_target_removed():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "target": [0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_data(df, "target")

    assert "target" not in X_train.columns
    assert "target" not in X_test.columns