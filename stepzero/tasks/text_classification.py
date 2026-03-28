from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .._headroom import compute_headroom
from .._preprocessing import to_1d
from .._runner import fit_best, run_cv
from .._types import ModelScore, TextClassifyResult


def text_classification(
    texts: Union[list[str], pd.Series],
    labels: Union[np.ndarray, pd.Series, list],
    *,
    cv: int = 5,
    random_state: int = 42,
) -> TextClassifyResult:
    """Run TF-IDF + Logistic Regression and TF-IDF + Naive Bayes.

    Automatically builds a TF-IDF vectorizer with unigrams and bigrams.
    Returns the best model with top discriminative features per class.

    Args:
        texts: List or Series of raw text strings.
        labels: Target class labels.
        cv: Number of cross-validation folds (default 5).
        random_state: Seed for reproducibility.

    Returns:
        TextClassifyResult with best_model, scores, headroom, and top_features_per_class.

    Example:
        >>> texts = ["I love this", "This is great", "Terrible", "Awful product"]
        >>> labels = [1, 1, 0, 0]
        >>> result = text_classification(texts, labels)
        >>> result.top_features_per_class
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [str(t) for t in texts]
    y_arr = to_1d(labels)

    def _make_tfidf():
        return TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )

    pipelines = {
        "tfidf_logistic": Pipeline([
            ("tfidf", _make_tfidf()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "tfidf_naive_bayes": Pipeline([
            ("tfidf", _make_tfidf()),
            ("model", MultinomialNB()),
        ]),
    }

    scores, cv_scores_map = run_cv(
        pipelines,
        texts,
        y_arr,
        scoring="accuracy",
        cv=cv,
        stratified=True,
        higher_is_better=True,
        random_state=random_state,
    )

    best_name = scores[0].name
    best_pipeline = pipelines[best_name]
    fit_best(best_pipeline, texts, y_arr)

    top_features = _extract_top_features(best_pipeline, y_arr)

    best_cv_scores = cv_scores_map[best_name]
    headroom = compute_headroom(
        cv_scores=best_cv_scores,
        best_score=scores[0].score,
        task_type="text_classification",
        n_samples=len(y_arr),
        n_features=10_000,
        higher_is_better=True,
    )

    return TextClassifyResult(
        best_model=best_pipeline,
        best_model_name=best_name,
        scores=scores,
        headroom=headroom,
        top_features_per_class=top_features,
    )


def _extract_top_features(
    pipeline: Pipeline, y: np.ndarray, top_n: int = 10
) -> dict[str, list[str]] | None:
    try:
        tfidf = pipeline.named_steps["tfidf"]
        model = pipeline.named_steps["model"]
        feature_names = tfidf.get_feature_names_out()
        classes = np.unique(y)

        top_features: dict[str, list[str]] = {}

        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.shape[0] == 1:
                # Binary classifier returns shape (1, n_features)
                top_features[str(classes[1])] = [
                    feature_names[i] for i in np.argsort(coef[0])[-top_n:][::-1]
                ]
                top_features[str(classes[0])] = [
                    feature_names[i] for i in np.argsort(coef[0])[:top_n]
                ]
            else:
                for i, cls in enumerate(classes):
                    top_features[str(cls)] = [
                        feature_names[j] for j in np.argsort(coef[i])[-top_n:][::-1]
                    ]
        elif hasattr(model, "feature_log_prob_"):
            for i, cls in enumerate(classes):
                top_features[str(cls)] = [
                    feature_names[j]
                    for j in np.argsort(model.feature_log_prob_[i])[-top_n:][::-1]
                ]

        return top_features if top_features else None
    except Exception:
        return None
