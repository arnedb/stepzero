from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .._headroom import headroom_from_agreement
from .._types import AnomalyDetectionResult


def _zscore_scores(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        return np.zeros_like(values)
    return np.abs((values - mean) / std)


def _iqr_scores(values: np.ndarray) -> np.ndarray:
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        median = np.nanmedian(values)
        return np.abs(values - median)
    median = np.nanmedian(values)
    return np.abs(values - median) / iqr


def _auto_threshold(scores: np.ndarray, contamination: float = 0.05) -> float:
    """Set threshold so at most `contamination` fraction of points are flagged."""
    return float(np.nanpercentile(scores, 100 * (1 - contamination)))


def anomaly_detection(
    series: Union[pd.Series, np.ndarray, list],
    *,
    threshold: Optional[float] = None,
    method: str = "auto",
) -> AnomalyDetectionResult:
    """Detect anomalies using Z-score and/or IQR methods.

    Args:
        series: 1-D numeric time series or array.
        threshold: Detection threshold. If None, auto-set to flag ~5% of points.
        method: "zscore", "iqr", or "auto" (default). "auto" runs both and
                returns the result from the method with higher internal consistency.

    Returns:
        AnomalyDetectionResult with anomaly mask, scores, and headroom signal.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> data = rng.normal(0, 1, 100)
        >>> data[[10, 50, 90]] = 10  # inject anomalies
        >>> result = anomaly_detection(data)
        >>> result.anomalies[result.anomalies].index.tolist()
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(np.asarray(series, dtype=float))

    values = series.to_numpy(dtype=float)

    zscore_s = _zscore_scores(values)
    iqr_s = _iqr_scores(values)

    if method == "zscore":
        chosen_method = "zscore"
        chosen_scores = zscore_s
        chosen_threshold = threshold if threshold is not None else _auto_threshold(zscore_s)
    elif method == "iqr":
        chosen_method = "iqr"
        chosen_scores = iqr_s
        chosen_threshold = threshold if threshold is not None else _auto_threshold(iqr_s)
    else:
        # "auto": pick method, auto-set thresholds, measure agreement
        t_z = threshold if threshold is not None else _auto_threshold(zscore_s)
        t_i = threshold if threshold is not None else _auto_threshold(iqr_s)

        flags_z = zscore_s > t_z
        flags_i = iqr_s > t_i

        intersection = int((flags_z & flags_i).sum())
        union = int((flags_z | flags_i).sum())
        agreement_ratio = intersection / (union + 1e-9)

        # Choose the method with the cleaner threshold (less extreme)
        if np.nanmean(zscore_s[flags_z]) >= np.nanmean(iqr_s[flags_i]):
            chosen_method = "zscore"
            chosen_scores = zscore_s
            chosen_threshold = t_z
        else:
            chosen_method = "iqr"
            chosen_scores = iqr_s
            chosen_threshold = t_i

        headroom = headroom_from_agreement(agreement_ratio, "anomaly")

        anomaly_mask = pd.Series(chosen_scores > chosen_threshold, index=series.index)
        score_series = pd.Series(chosen_scores, index=series.index)
        return AnomalyDetectionResult(
            method=chosen_method,
            threshold=float(chosen_threshold),
            anomalies=anomaly_mask,
            scores=score_series,
            headroom=headroom,
        )

    anomaly_mask = pd.Series(chosen_scores > chosen_threshold, index=series.index)
    score_series = pd.Series(chosen_scores, index=series.index)

    # For single-method runs, estimate headroom from score distribution spread
    flagged_frac = float(anomaly_mask.mean())
    # If very few or many points flagged, the threshold may not be well-calibrated
    if 0.01 <= flagged_frac <= 0.15:
        agreement_ratio = 0.8  # plausible contamination range -> Low headroom
    else:
        agreement_ratio = 0.3  # suspicious -> High headroom

    headroom = headroom_from_agreement(agreement_ratio, "anomaly")

    return AnomalyDetectionResult(
        method=chosen_method,
        threshold=float(chosen_threshold),
        anomalies=anomaly_mask,
        scores=score_series,
        headroom=headroom,
    )
