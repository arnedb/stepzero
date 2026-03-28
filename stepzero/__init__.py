"""stepzero — Task-first ML baselines.

Run the simplest model that could work for your task, compare it against
a few alternatives, and get a signal on whether it's worth going further.

Usage:
    import stepzero as sz

    result = sz.classification(X, y)
    result = sz.regression(X, y)
    result = sz.forecasting(series, horizon=12)
    result = sz.anomaly_detection(series)
    result = sz.text_classification(texts, labels)
    result = sz.clustering(X)

Every result includes:
    .best_model       — fitted sklearn-compatible model (call .predict() directly)
    .best_model_name  — which model won
    .scores           — all models compared, with CV scores
    .headroom         — HeadroomSignal: Low/Medium/High + actionable next step
"""

from .tasks.classification import classification
from .tasks.clustering import clustering
from .tasks.anomaly_detection import anomaly_detection
from .tasks.forecasting import forecasting
from .tasks.regression import regression
from .tasks.text_classification import text_classification
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
    "classification",
    "regression",
    "forecasting",
    "anomaly_detection",
    "text_classification",
    "clustering",
    "ClassifyResult",
    "RegressResult",
    "ForecastResult",
    "AnomalyResult",
    "TextClassifyResult",
    "ClusterResult",
    "HeadroomSignal",
    "ModelScore",
]
