from __future__ import annotations

import numpy as np

from ._types import HeadroomSignal

_NEXT_STEP = {
    "classification": "a gradient boosted tree (e.g., XGBoost or LightGBM)",
    "regression": "a gradient boosted tree (e.g., XGBoost or LightGBM)",
    "forecast": "ARIMA, Prophet, or a neural forecasting model",
    "anomaly": "Isolation Forest or a learned autoencoder",
    "text_classification": "a fine-tuned transformer (e.g., DistilBERT)",
    "cluster": "HDBSCAN or Gaussian Mixture Models",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_headroom(
    *,
    cv_scores: np.ndarray,
    best_score: float,
    task_type: str,
    n_samples: int,
    n_features: int,
    baseline_score: float | None = None,
    higher_is_better: bool = True,
) -> HeadroomSignal:
    """Estimate how much headroom remains for a more complex model.

    Returns a HeadroomSignal with level ("Low"/"Medium"/"High") and an
    actionable reason string naming the next model to try.
    """
    next_step = _NEXT_STEP.get(task_type, "a more complex model")

    # --- Gap to ceiling ---
    if higher_is_better:
        gap = 1.0 - best_score  # fraction of ceiling not reached
    else:
        # For RMSE-style metrics, compare against a naive baseline
        if baseline_score and baseline_score > 0:
            gap = _clamp(1.0 - best_score / baseline_score, 0.0, 1.0)
        else:
            gap = 0.3  # assume medium gap if we have no baseline

    # --- CV instability ---
    cv_std = float(np.std(cv_scores)) if len(cv_scores) > 1 else 0.0
    denom = abs(best_score) + 1e-9
    variance_signal = _clamp(cv_std / denom, 0.0, 1.0)

    # --- Data richness (more data = more potential for complex models) ---
    richness = _clamp(np.log1p(n_samples) / np.log1p(10_000), 0.0, 1.0)

    headroom_score = 0.5 * gap + 0.3 * variance_signal + 0.2 * richness
    headroom_score = _clamp(headroom_score, 0.0, 1.0)

    if headroom_score < 0.3:
        level = "Low"
        if higher_is_better:
            reason = (
                f"Score of {best_score:.2f} with low variance (±{cv_std:.3f}). "
                f"The simple baseline is already performing well. "
                f"Trying {next_step} is unlikely to offer a meaningful improvement."
            )
        else:
            reason = (
                f"Score of {best_score:.4f} with low variance (±{cv_std:.4f}). "
                f"The simple baseline is already performing well. "
                f"Trying {next_step} is unlikely to offer a meaningful improvement."
            )
    elif headroom_score < 0.6:
        level = "Medium"
        if higher_is_better:
            gap_pct = gap * 100
            reason = (
                f"Score of {best_score:.2f} (±{cv_std:.3f}). "
                f"A {gap_pct:.0f}% gap to ceiling remains. "
                f"Trying {next_step} may yield a moderate improvement."
            )
        else:
            reason = (
                f"Score of {best_score:.4f} (±{cv_std:.4f}). "
                f"Moderate headroom remains. "
                f"Trying {next_step} may yield a moderate improvement."
            )
    else:
        level = "High"
        if higher_is_better:
            reason = (
                f"Score of {best_score:.2f} with high variance (±{cv_std:.3f}). "
                f"The simple baseline is underperforming. "
                f"Try {next_step} — there is likely significant improvement available."
            )
        else:
            reason = (
                f"Score of {best_score:.4f} with high variance (±{cv_std:.4f}). "
                f"The simple baseline is underperforming. "
                f"Try {next_step} — there is likely significant improvement available."
            )

    return HeadroomSignal(level=level, reason=reason)


def headroom_from_agreement(agreement_ratio: float, task_type: str) -> HeadroomSignal:
    """Headroom based on agreement between two unsupervised detectors."""
    next_step = _NEXT_STEP.get(task_type, "a more complex model")
    if agreement_ratio > 0.7:
        return HeadroomSignal(
            level="Low",
            reason=(
                f"Both detection methods agree on {agreement_ratio:.0%} of flagged points. "
                f"The anomaly pattern is clear. {next_step} is unlikely to improve substantially."
            ),
        )
    elif agreement_ratio > 0.4:
        return HeadroomSignal(
            level="Medium",
            reason=(
                f"Methods agree on {agreement_ratio:.0%} of flagged points. "
                f"Some ambiguity remains. Consider {next_step} for a more robust detection."
            ),
        )
    else:
        return HeadroomSignal(
            level="High",
            reason=(
                f"Methods agree on only {agreement_ratio:.0%} of flagged points. "
                f"The anomaly structure may be complex. Try {next_step}."
            ),
        )


def headroom_from_silhouette(silhouette: float, task_type: str = "cluster") -> HeadroomSignal:
    """Headroom based on clustering silhouette score."""
    next_step = _NEXT_STEP.get(task_type, "a more complex model")
    if silhouette > 0.6:
        return HeadroomSignal(
            level="Low",
            reason=(
                f"Silhouette score of {silhouette:.3f} indicates well-separated clusters. "
                f"K-means is working well here."
            ),
        )
    elif silhouette > 0.3:
        return HeadroomSignal(
            level="Medium",
            reason=(
                f"Silhouette score of {silhouette:.3f} indicates moderate cluster separation. "
                f"Try {next_step} for potentially better cluster structure."
            ),
        )
    else:
        return HeadroomSignal(
            level="High",
            reason=(
                f"Silhouette score of {silhouette:.3f} indicates poor cluster separation. "
                f"K-means may not be the right fit. Try {next_step}."
            ),
        )
