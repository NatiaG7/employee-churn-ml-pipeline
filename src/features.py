"""Feature engineering for HR churn dataset."""

import pandas as pd


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load HR dataset and strip column names."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode categoricals and split features / target.

    Target: left (1 = employee left, 0 = stayed)
    """
    X = df.drop("left", axis=1)
    y = df["left"]

    cat_cols = ["sales", "salary"]
    num_cols = X.columns.difference(cat_cols)

    X_cat = pd.get_dummies(X[cat_cols], drop_first=True)
    X_num = X[num_cols]
    X_final = pd.concat([X_num, X_cat], axis=1)

    return X_final, y
