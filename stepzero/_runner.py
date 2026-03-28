from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ._types import ModelScore


def run_cv(
    pipelines: dict[str, Pipeline],
    X,
    y,
    *,
    scoring: str,
    cv: int = 5,
    stratified: bool = True,
    higher_is_better: bool = True,
    random_state: int = 42,
) -> tuple[list[ModelScore], dict[str, np.ndarray]]:
    """Run cross-validation for each pipeline.

    Returns:
        scores: list[ModelScore] sorted best first
        cv_scores_per_model: dict mapping model name -> array of fold scores
    """
    if stratified:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    model_scores: list[ModelScore] = []
    cv_scores_per_model: dict[str, np.ndarray] = {}

    metric_name = scoring.lstrip("neg_").replace("_", " ")
    if scoring.startswith("neg_"):
        metric_name = scoring[4:].replace("_", " ")

    for name, pipeline in pipelines.items():
        raw = cross_val_score(pipeline, X, y, scoring=scoring, cv=splitter)
        # sklearn negates loss metrics — un-negate for display
        if scoring.startswith("neg_"):
            display_scores = -raw
        else:
            display_scores = raw
        mean_score = float(np.mean(display_scores))
        cv_scores_per_model[name] = display_scores
        model_scores.append(ModelScore(name=name, score=mean_score, metric=metric_name))

    model_scores.sort(key=lambda s: s.score, reverse=higher_is_better)
    return model_scores, cv_scores_per_model


def fit_best(pipeline: Pipeline, X, y) -> Pipeline:
    """Refit a pipeline on the full dataset."""
    pipeline.fit(X, y)
    return pipeline
