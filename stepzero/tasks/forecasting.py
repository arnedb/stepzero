from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .._headroom import compute_headroom
from .._types import ForecastResult, ModelScore

_FREQ_TO_SEASON = {
    "H": 24,
    "h": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "A": 1,
    "AS": 1,
    "Y": 1,
    "YS": 1,
    "YE": 1,
    "T": 60,
    "min": 60,
}


def _infer_season_length(series: pd.Series, freq: Optional[str]) -> int:
    if freq is not None:
        base = freq.rstrip("0123456789-")
        return _FREQ_TO_SEASON.get(base, 1)
    if isinstance(series.index, pd.DatetimeIndex):
        inferred = pd.infer_freq(series.index)
        if inferred:
            base = inferred.rstrip("0123456789-")
            return _FREQ_TO_SEASON.get(base, 1)
    return 1


def _seasonal_naive_predict(
    train: np.ndarray, horizon: int, season_length: int
) -> np.ndarray:
    """Repeat the last full seasonal cycle."""
    if season_length <= 1 or len(train) < season_length:
        return np.full(horizon, train[-1])
    last_cycle = train[-season_length:]
    repeats = (horizon // season_length) + 1
    tiled = np.tile(last_cycle, repeats)
    return tiled[:horizon]


def _linear_trend_predict(
    train: np.ndarray, horizon: int, season_length: int
) -> np.ndarray:
    """Linear trend + seasonal mean residuals."""
    n = len(train)
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, train, 1)
    slope, intercept = coeffs

    # Detrend
    trend = slope * t + intercept
    residuals = train - trend

    # Seasonal component: mean of residuals at each season position
    if season_length > 1 and n >= season_length:
        seasonal = np.zeros(season_length)
        for s in range(season_length):
            indices = np.arange(s, n, season_length)
            seasonal[s] = residuals[indices].mean()
    else:
        seasonal = np.zeros(1)

    # Forecast
    t_future = np.arange(n, n + horizon, dtype=float)
    trend_forecast = slope * t_future + intercept
    if season_length > 1 and n >= season_length:
        seasonal_forecast = np.array(
            [seasonal[(n + i) % season_length] for i in range(horizon)]
        )
    else:
        seasonal_forecast = np.zeros(horizon)

    return trend_forecast + seasonal_forecast


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _build_future_index(series: pd.Series, horizon: int) -> pd.Index:
    if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 1:
        freq = pd.infer_freq(series.index)
        if freq:
            last = series.index[-1]
            return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]
    return pd.RangeIndex(start=len(series), stop=len(series) + horizon)


def forecasting(
    series: Union[pd.Series, np.ndarray, list],
    horizon: int,
    *,
    freq: Optional[str] = None,
    cv_splits: int = 3,
) -> ForecastResult:
    """Run Seasonal Naive and Linear Trend models on a time series.

    Selects the best model via time-series cross-validation (MAE),
    then forecasts the next `horizon` steps.

    Args:
        series: 1-D time series. Use a pandas Series with DatetimeIndex
                for automatic frequency detection and a date-indexed forecast.
        horizon: Number of future steps to forecast.
        freq: Pandas offset string for seasonality inference (e.g., "M", "D").
              Auto-inferred from a DatetimeIndex if not provided.
        cv_splits: Number of time-series CV splits (default 3).

    Returns:
        ForecastResult with best_model_name, scores, forecast Series, and headroom.

    Example:
        >>> import pandas as pd
        >>> ts = pd.Series(range(48), index=pd.date_range("2020-01", periods=48, freq="M"))
        >>> result = forecasting(ts, horizon=12)
        >>> result.forecast
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(np.asarray(series, dtype=float))

    values = series.to_numpy(dtype=float)
    season_length = _infer_season_length(series, freq)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    mae_scores: dict[str, list[float]] = {"seasonal_naive": [], "linear_trend": []}

    for train_idx, test_idx in tscv.split(values):
        train, test = values[train_idx], values[test_idx]
        h = len(test)

        sn_pred = _seasonal_naive_predict(train, h, season_length)
        mae_scores["seasonal_naive"].append(_mae(test, sn_pred))

        lt_pred = _linear_trend_predict(train, h, season_length)
        mae_scores["linear_trend"].append(_mae(test, lt_pred))

    mean_maes = {k: float(np.mean(v)) for k, v in mae_scores.items()}
    cv_array = {k: np.array(v) for k, v in mae_scores.items()}

    scores = [
        ModelScore(name=k, score=v, metric="mae") for k, v in mean_maes.items()
    ]
    scores.sort(key=lambda s: s.score)  # lower MAE is better

    best_name = scores[0].name
    best_cv = cv_array[best_name]

    # Produce the actual forecast using the full series
    if best_name == "seasonal_naive":
        pred_values = _seasonal_naive_predict(values, horizon, season_length)
    else:
        pred_values = _linear_trend_predict(values, horizon, season_length)

    future_index = _build_future_index(series, horizon)
    forecast_series = pd.Series(pred_values, index=future_index, name="forecast")

    # Headroom: compare whether linear trend is meaningfully better than seasonal naive
    sn_mae = mean_maes["seasonal_naive"]
    lt_mae = mean_maes["linear_trend"]
    relative_improvement = abs(sn_mae - lt_mae) / (sn_mae + 1e-9)

    headroom = compute_headroom(
        cv_scores=best_cv,
        best_score=scores[0].score,
        task_type="forecast",
        n_samples=len(values),
        n_features=1,
        baseline_score=sn_mae,
        higher_is_better=False,
    )

    # If neither model clearly beats the other, bump headroom up
    if relative_improvement < 0.05 and headroom.level == "Low":
        from .._types import HeadroomSignal
        headroom = HeadroomSignal(
            level="Medium",
            reason=(
                f"Seasonal naive and linear trend perform similarly (MAE gap < 5%). "
                f"Both simple models may be missing non-linear structure. "
                f"Try ARIMA, Prophet, or a neural forecasting model."
            ),
        )

    return ForecastResult(
        best_model=None,  # stateless — reconstruct from best_model_name + series
        best_model_name=best_name,
        scores=scores,
        forecast=forecast_series,
        headroom=headroom,
    )
