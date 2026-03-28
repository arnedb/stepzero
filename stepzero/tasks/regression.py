from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from .._headroom import compute_headroom
from .._preprocessing import build_tabular_pipeline, to_1d
from .._runner import fit_best, run_cv
from .._types import RegressionResult


def regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    *,
    cv: int = 5,
    random_state: int = 42,
) -> RegressionResult:
    """Run Ridge regression and Decision Tree regressor.

    Returns the best model with RMSE scores, feature importance,
    and a headroom signal comparing against a naive mean baseline.

    Args:
        X: Feature matrix (numpy array or pandas DataFrame).
        y: Target values.
        cv: Number of cross-validation folds (default 5).
        random_state: Seed for reproducibility.

    Returns:
        RegressionResult with best_model, scores, feature_importance, and headroom.

    Example:
        >>> from sklearn.datasets import load_diabetes
        >>> X, y = load_diabetes(return_X_y=True)
        >>> result = regression(X, y)
        >>> print(result)
        RegressionResult(best='ridge', root mean squared error=53.476, headroom='Medium')
        >>> result.feature_importance.head(5)
    """
    y_arr = to_1d(y)
    baseline_rmse = float(np.std(y_arr))  # predict-mean baseline

    pipelines = {
        "ridge": build_tabular_pipeline(
            Ridge(alpha=1.0),
            scale=True,
            X=X,
        ),
        "tree": build_tabular_pipeline(
            DecisionTreeRegressor(max_depth=5, random_state=random_state),
            scale=False,
            X=X,
        ),
    }

    scores, cv_scores_map = run_cv(
        pipelines,
        X,
        y_arr,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        stratified=False,
        higher_is_better=False,
        random_state=random_state,
    )

    best_name = scores[0].name
    best_pipeline = pipelines[best_name]
    fit_best(best_pipeline, X, y_arr)

    # Feature importance
    model = best_pipeline.named_steps["model"]
    if isinstance(X, pd.DataFrame):
        try:
            feature_names = _get_feature_names(best_pipeline, X)
        except Exception:
            feature_names = None
    else:
        feature_names = None

    n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        fi = np.abs(coef.ravel())
        total = fi.sum()
        fi = fi / (total + 1e-12)  # normalize to sum=1
    else:
        fi = np.zeros(n_features)

    feature_importance = pd.Series(
        fi, index=feature_names or range(len(fi))
    ).sort_values(ascending=False)

    best_cv_scores = cv_scores_map[best_name]
    headroom = compute_headroom(
        cv_scores=best_cv_scores,
        best_score=scores[0].score,
        task_type="regression",
        n_samples=len(y_arr),
        n_features=n_features,
        baseline_score=baseline_rmse,
        higher_is_better=False,
    )

    return RegressionResult(
        best_model=best_pipeline,
        best_model_name=best_name,
        scores=scores,
        headroom=headroom,
        feature_importance=feature_importance,
    )


def _get_feature_names(pipeline, X: pd.DataFrame) -> list[str] | None:
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return list(X.columns)
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None
