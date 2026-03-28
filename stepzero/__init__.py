"""stepzero — Task-first ML baselines.

Run the simplest model that could work for your task, compare it against
a few alternatives, and get a signal on whether it's worth going further.

Usage:
    import stepzero as sz

    result = sz.classify(X, y)
    result = sz.regress(X, y)
    result = sz.forecast(series, horizon=12)
    result = sz.detect_anomalies(series)
    result = sz.text_classify(texts, labels)
    result = sz.cluster(X)

Every result includes:
    .best_model       — fitted sklearn-compatible model (call .predict() directly)
    .best_model_name  — which model won
    .scores           — all models compared, with CV scores
    .headroom         — HeadroomSignal: Low/Medium/High + actionable next step
"""

from .tasks.classify import classify
from .tasks.cluster import cluster
from .tasks.detect_anomalies import detect_anomalies
from .tasks.forecast import forecast
from .tasks.regress import regress
from .tasks.text_classify import text_classify
from ._types import (
    AnomalyResult,
    ClassifyResult,
    ClusterResult,
    ForecastResult,
    HeadroomSignal,
    ModelScore,
    RegressResult,
    TextClassifyResult,
)

__version__ = "0.1.0"

__all__ = [
    "classify",
    "regress",
    "forecast",
    "detect_anomalies",
    "text_classify",
    "cluster",
    "ClassifyResult",
    "RegressResult",
    "ForecastResult",
    "AnomalyResult",
    "TextClassifyResult",
    "ClusterResult",
    "HeadroomSignal",
    "ModelScore",
]
