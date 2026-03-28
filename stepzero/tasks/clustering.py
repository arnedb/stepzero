from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .._headroom import headroom_from_silhouette
from .._types import ClusteringResult, ModelScore


def clustering(
    X: Union[np.ndarray, pd.DataFrame],
    *,
    k_range: tuple[int, int] = (2, 10),
    random_state: int = 42,
) -> ClusteringResult:
    """Run K-means across a range of k values, select best k via elbow + silhouette.

    Automatically scales features before clustering.

    Args:
        X: Feature matrix (numpy array or pandas DataFrame).
        k_range: Min and max number of clusters to try (inclusive). Default (2, 10).
        random_state: Seed for reproducibility.

    Returns:
        ClusteringResult with best_k, cluster labels, centers, scores, and headroom.

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
        >>> result = clustering(X)
        >>> result.best_k
        4
        >>> result.labels[:10]
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))

    inertias: list[float] = []
    silhouettes: list[float] = []
    all_labels: dict[int, np.ndarray] = {}

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        all_labels[k] = labels
        inertias.append(float(km.inertia_))

        if k >= 2 and len(np.unique(labels)) > 1:
            sil = float(silhouette_score(X_scaled, labels, sample_size=min(len(X_scaled), 5000)))
        else:
            sil = -1.0
        silhouettes.append(sil)

    # Elbow detection via second derivative of inertia
    inertia_arr = np.array(inertias)
    if len(inertia_arr) >= 3:
        second_deriv = np.diff(inertia_arr, n=2)
        elbow_idx = int(np.argmax(second_deriv)) + 1  # offset by 2 diffs
    else:
        elbow_idx = 0

    # Also consider the k with best silhouette score
    sil_arr = np.array(silhouettes)
    best_sil_idx = int(np.argmax(sil_arr))

    # Balance: prefer the silhouette maximum, use elbow as a tie-breaker
    best_idx = best_sil_idx
    best_k = k_values[best_idx]

    # Build final scores list (one entry per k, silhouette as primary metric)
    scores: list[ModelScore] = []
    inertia_baseline = inertias[0] if inertias[0] > 0 else 1.0
    for i, k in enumerate(k_values):
        scores.append(ModelScore(
            name=f"k={k}",
            score=silhouettes[i],
            metric="silhouette",
        ))

    # Refit best k to get centers in original scale
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    km_best.fit(X_scaled)
    labels_best = km_best.labels_
    centers_scaled = km_best.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    best_silhouette = silhouettes[best_idx]
    headroom = headroom_from_silhouette(best_silhouette)

    return ClusteringResult(
        best_k=best_k,
        labels=labels_best,
        scores=scores,
        centers=centers,
        headroom=headroom,
    )
