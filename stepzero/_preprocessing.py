from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, cat_cols


def build_tabular_pipeline(estimator, *, scale: bool = False, X=None) -> Pipeline:
    """Wrap an estimator in a preprocessing + estimator pipeline.

    If X is a DataFrame with mixed types, uses ColumnTransformer.
    If X is a plain numpy array, uses a simple numeric pipeline.
    scale=True adds StandardScaler for linear models.
    """
    if X is not None and isinstance(X, pd.DataFrame):
        numeric_cols, cat_cols = _detect_column_types(X)

        numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(numeric_steps)

        # Use OrdinalEncoder for tree models, OneHotEncoder for linear models
        if scale:
            cat_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        else:
            cat_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if cat_cols:
            transformers.append(("cat", cat_transformer, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    else:
        # Plain numpy array — assume all numeric
        steps: list = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)


def to_numpy(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, pd.Series):
        return X.to_numpy()
    return np.asarray(X)


def to_1d(y) -> np.ndarray:
    arr = to_numpy(y)
    return arr.ravel()
