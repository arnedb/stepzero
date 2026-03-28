from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelScore:
    name: str
    score: float
    metric: str


@dataclass
class HeadroomSignal:
    """Estimate of how much improvement a more complex model could offer."""

    level: str  # "low" | "medium" | "high"
    reason: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.reason}"


@dataclass
class ClassificationResult:
    best_model: Any
    best_model_name: str
    scores: list[ModelScore]
    headroom: HeadroomSignal
    feature_importance: Optional[pd.Series] = None

    def __repr__(self) -> str:
        score = next(s for s in self.scores if s.name == self.best_model_name)
        return (
            f"ClassificationResult(best={self.best_model_name!r}, "
            f"{score.metric}={score.score:.3f}, "
            f"headroom={self.headroom.level!r})"
        )


@dataclass
class RegressionResult:
    best_model: Any
    best_model_name: str
    scores: list[ModelScore]
    headroom: HeadroomSignal
    feature_importance: pd.Series = field(default_factory=pd.Series)

    def __repr__(self) -> str:
        score = next(s for s in self.scores if s.name == self.best_model_name)
        return (
            f"RegressionResult(best={self.best_model_name!r}, "
            f"{score.metric}={score.score:.3f}, "
            f"headroom={self.headroom.level!r})"
        )


@dataclass
class ForecastingResult:
    best_model: Any
    best_model_name: str
    scores: list[ModelScore]
    forecast: pd.Series
    headroom: HeadroomSignal

    def __repr__(self) -> str:
        score = next(s for s in self.scores if s.name == self.best_model_name)
        return (
            f"ForecastingResult(best={self.best_model_name!r}, "
            f"{score.metric}={score.score:.3f}, "
            f"horizon={len(self.forecast)}, "
            f"headroom={self.headroom.level!r})"
        )


@dataclass
class AnomalyDetectionResult:
    method: str
    threshold: float
    anomalies: pd.Series
    scores: pd.Series
    headroom: HeadroomSignal

    def __repr__(self) -> str:
        n = self.anomalies.sum()
        pct = 100 * n / len(self.anomalies)
        return (
            f"AnomalyDetectionResult(method={self.method!r}, "
            f"anomalies={n} ({pct:.1f}%), "
            f"headroom={self.headroom.level!r})"
        )


@dataclass
class TextClassificationResult:
    best_model: Any
    best_model_name: str
    scores: list[ModelScore]
    headroom: HeadroomSignal
    top_features_per_class: Optional[dict[str, list[str]]] = None

    def __repr__(self) -> str:
        score = next(s for s in self.scores if s.name == self.best_model_name)
        return (
            f"TextClassificationResult(best={self.best_model_name!r}, "
            f"{score.metric}={score.score:.3f}, "
            f"headroom={self.headroom.level!r})"
        )


@dataclass
class ClusteringResult:
    best_k: int
    labels: np.ndarray
    scores: list[ModelScore]
    centers: np.ndarray
    headroom: HeadroomSignal

    def __repr__(self) -> str:
        sil = next((s for s in self.scores if s.metric == "silhouette"), None)
        sil_str = f", silhouette={sil.score:.3f}" if sil else ""
        return (
            f"ClusteringResult(best_k={self.best_k}"
            f"{sil_str}, "
            f"headroom={self.headroom.level!r})"
        )
