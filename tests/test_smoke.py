"""Smoke tests — verify each task function runs end-to-end without errors."""

import numpy as np
import pandas as pd
import pytest

import stepzero as sz


def test_classification():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True, as_frame=True)
    result = sz.classification(X, y)
    assert result.best_model_name in {"logistic", "tree", "naive_bayes"}
    assert len(result.scores) == 3
    assert result.headroom.level in {"Low", "Medium", "High"}
    assert result.feature_importance is not None
    preds = result.best_model.predict(X)
    assert len(preds) == len(y)


def test_classification_numpy():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = sz.classification(X, y)
    assert result.best_model is not None


def test_regression():
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    result = sz.regression(X, y)
    assert result.best_model_name in {"ridge", "tree"}
    assert len(result.scores) == 2
    assert result.headroom.level in {"Low", "Medium", "High"}
    assert len(result.feature_importance) > 0
    preds = result.best_model.predict(X)
    assert len(preds) == len(y)


def test_forecasting_numpy():
    rng = np.random.default_rng(42)
    values = np.sin(np.linspace(0, 4 * np.pi, 50)) + rng.normal(0, 0.1, 50)
    result = sz.forecasting(values, horizon=10)
    assert result.best_model_name in {"seasonal_naive", "linear_trend"}
    assert len(result.forecast) == 10
    assert result.headroom.level in {"Low", "Medium", "High"}


def test_forecasting_datetime_index():
    idx = pd.date_range("2020-01", periods=36, freq="ME")
    ts = pd.Series(np.arange(36, dtype=float) + np.random.default_rng(0).normal(0, 1, 36), index=idx)
    result = sz.forecasting(ts, horizon=6)
    assert len(result.forecast) == 6
    assert isinstance(result.forecast.index, pd.DatetimeIndex)


def test_anomaly_detection_auto():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 200)
    data[[10, 50, 100, 150]] = 10.0
    result = sz.anomaly_detection(data)
    assert result.method in {"zscore", "iqr"}
    assert len(result.anomalies) == 200
    assert result.anomalies.dtype == bool
    assert result.anomalies[[10, 50, 100, 150]].all()


def test_anomaly_detection_explicit_methods():
    data = np.array([1.0, 2.0, 1.5, 100.0, 1.8, 2.1])
    for m in ("zscore", "iqr"):
        result = sz.anomaly_detection(data, method=m)
        assert result.method == m
        assert result.anomalies[3]  # index 3 is the spike


def test_text_classification():
    texts = [
        "I love this product",
        "Absolutely fantastic",
        "Great quality",
        "Really good value",
        "Terrible experience",
        "Very disappointed",
        "Worst purchase ever",
        "Complete waste of money",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    result = sz.text_classification(texts, labels, cv=2)
    assert result.best_model_name in {"tfidf_logistic", "tfidf_naive_bayes"}
    assert result.headroom.level in {"Low", "Medium", "High"}
    preds = result.best_model.predict(texts)
    assert len(preds) == 8


def test_clustering():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=0)
    result = sz.clustering(X, k_range=(2, 8))
    assert 2 <= result.best_k <= 8
    assert len(result.labels) == 200
    assert result.centers.shape == (result.best_k, X.shape[1])
    assert result.headroom.level in {"Low", "Medium", "High"}


def test_result_repr():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    result = sz.classification(X, y)
    r = repr(result)
    assert "ClassificationResult" in r
    assert "headroom=" in r


def test_headroom_signal_str():
    from stepzero import HeadroomSignal
    h = HeadroomSignal(level="Low", reason="test reason")
    assert "[Low]" in str(h)
