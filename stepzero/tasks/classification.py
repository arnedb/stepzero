from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from .._headroom import compute_headroom
from .._preprocessing import build_tabular_pipeline, to_1d
from .._runner import fit_best, run_cv
from .._types import ClassifyResult, ModelScore


def classification(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, list],
    *,
    cv: int = 5,
    random_state: int = 42,
) -> ClassifyResult:
    """Run Logistic Regression, Decision Tree, and Naive Bayes.

    Automatically handles scaling, missing values, and categorical encoding.
    Returns the best model with CV scores and a headroom signal.

    Args:
        X: Feature matrix (numpy array or pandas DataFrame).
        y: Target labels.
        cv: Number of cross-validation folds (default 5).
        random_state: Seed for reproducibility.

    Returns:
        ClassifyResult with best_model, scores, headroom, and feature_importance.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> result = classification(X, y)
        >>> print(result)
        ClassifyResult(best='logistic', accuracy=0.960, headroom='Low')
        >>> result.best_model.predict(X[:5])
    """
    y_arr = to_1d(y)

    pipelines = {
        "logistic": build_tabular_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state),
            scale=True,
            X=X,
        ),
        "tree": build_tabular_pipeline(
            DecisionTreeClassifier(max_depth=5, random_state=random_state),
            scale=False,
            X=X,
        ),
        "naive_bayes": build_tabular_pipeline(
            GaussianNB(),
            scale=False,
            X=X,
        ),
    }

    scores, cv_scores_map = run_cv(
        pipelines,
        X,
        y_arr,
        scoring="accuracy",
        cv=cv,
        stratified=True,
        higher_is_better=True,
        random_state=random_state,
    )

    best_name = scores[0].name
    best_pipeline = pipelines[best_name]
    fit_best(best_pipeline, X, y_arr)

    # Extract feature importance from best model
    feature_importance: pd.Series | None = None
    model = best_pipeline.named_steps["model"]
    if isinstance(X, pd.DataFrame):
        try:
            feature_names = _get_feature_names(best_pipeline, X)
        except Exception:
            feature_names = None
    else:
        feature_names = None

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feature_importance = pd.Series(fi, index=feature_names or range(len(fi))).sort_values(
            ascending=False
        )
    elif hasattr(model, "coef_"):
        coef = model.coef_
        fi = np.abs(coef).max(axis=0) if coef.ndim > 1 else np.abs(coef)
        feature_importance = pd.Series(fi, index=feature_names or range(len(fi))).sort_values(
            ascending=False
        )

    best_cv_scores = cv_scores_map[best_name]
    headroom = compute_headroom(
        cv_scores=best_cv_scores,
        best_score=scores[0].score,
        task_type="classification",
        n_samples=len(y_arr),
        n_features=X.shape[1] if hasattr(X, "shape") else len(X[0]),
        higher_is_better=True,
    )

    return ClassifyResult(
        best_model=best_pipeline,
        best_model_name=best_name,
        scores=scores,
        headroom=headroom,
        feature_importance=feature_importance,
    )


def _get_feature_names(pipeline, X: pd.DataFrame) -> list[str] | None:
    """Try to recover feature names after preprocessing."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return list(X.columns)
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None
