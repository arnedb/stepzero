"""Script to generate the demo notebooks programmatically."""

import json
import uuid
from pathlib import Path


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def _uid():
    return uuid.uuid4().hex[:8]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": _uid()}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        "id": _uid(),
    }


# ── Notebook 1: Quickstart ────────────────────────────────────────────────────

nb1 = nb([
    md("# stepzero — Quickstart\n\nA 5-minute tour of all six tasks."),
    code("import stepzero as sz\nprint(sz.__version__)"),

    md("## 1. Classification — Iris"),
    code("""\
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True, as_frame=True)

result = sz.classify(X, y)
print(result)
print()
print(result.headroom)
"""),
    code("""\
# All models compared
for s in result.scores:
    print(f"  {s.name:<15} {s.metric} = {s.score:.3f}")
"""),
    code("""\
# Feature importance from the winning model
result.feature_importance.plot(kind="barh", title="Feature importance")
"""),

    md("## 2. Regression — Diabetes"),
    code("""\
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

result = sz.regress(X, y)
print(result)
print()
print(result.headroom)
"""),
    code("""\
result.feature_importance.plot(kind="barh", title="Feature importance (normalized)")
"""),

    md("## 3. Forecasting — Synthetic trend + seasonality"),
    code("""\
import numpy as np
import pandas as pd

# 4 years of monthly data: upward trend + yearly seasonality
idx = pd.date_range("2020-01", periods=48, freq="ME")
trend = np.linspace(100, 160, 48)
seasonal = 10 * np.sin(2 * np.pi * np.arange(48) / 12)
noise = np.random.default_rng(0).normal(0, 3, 48)
ts = pd.Series(trend + seasonal + noise, index=idx, name="value")

ts.plot(title="Training series", figsize=(10, 3))
"""),
    code("""\
result = sz.forecast(ts, horizon=12)
print(result)
print()
print(result.headroom)
"""),
    code("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ts.plot(ax=ax, label="observed")
result.forecast.plot(ax=ax, label=f"forecast ({result.best_model_name})", linestyle="--", marker="o")
ax.legend()
ax.set_title("stepzero forecast — 12-month horizon")
"""),

    md("## 4. Anomaly detection — Synthetic sensor data"),
    code("""\
rng = np.random.default_rng(42)
sensor = pd.Series(rng.normal(0, 1, 200), name="sensor")
sensor.iloc[[20, 80, 140, 180]] = [6, -7, 8, -6]  # inject 4 spikes

result = sz.detect_anomalies(sensor)
print(result)
print()
print(result.headroom)
"""),
    code("""\
fig, ax = plt.subplots(figsize=(10, 3))
sensor.plot(ax=ax, label="signal", alpha=0.7)
sensor[result.anomalies].plot(ax=ax, style="rv", label="anomaly", markersize=10)
ax.legend()
ax.set_title(f"Anomaly detection ({result.method}, threshold={result.threshold:.2f})")
"""),

    md("## 5. Text classification — 20 Newsgroups (2 categories)"),
    code("""\
from sklearn.datasets import fetch_20newsgroups

cats = ["sci.space", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=cats)

result = sz.text_classify(train.data, train.target, cv=3)
print(result)
print()
print(result.headroom)
"""),
    code("""\
# Top discriminative terms per class
for cls, terms in result.top_features_per_class.items():
    label = train.target_names[int(cls)]
    print(f"  {label}: {', '.join(terms[:8])}")
"""),

    md("## 6. Clustering — make_blobs"),
    code("""\
from sklearn.datasets import make_blobs

X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

result = sz.cluster(X_blobs, k_range=(2, 8))
print(result)
print()
print(result.headroom)
"""),
    code("""\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: cluster assignments
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=result.labels, cmap="tab10", s=20)
axes[0].scatter(result.centers[:, 0], result.centers[:, 1], marker="X", s=200, c="black", label="centers")
axes[0].set_title(f"K-means, k={result.best_k}")
axes[0].legend()

# Right: silhouette score vs k
ks = [int(s.name.split("=")[1]) for s in result.scores]
sils = [s.score for s in result.scores]
axes[1].plot(ks, sils, marker="o")
axes[1].axvline(result.best_k, color="red", linestyle="--", label=f"best k={result.best_k}")
axes[1].set_xlabel("k")
axes[1].set_ylabel("silhouette score")
axes[1].set_title("Silhouette vs k")
axes[1].legend()
"""),
])


# ── Notebook 2: Tabular difficulty ───────────────────────────────────────────

nb2 = nb([
    md("""\
# When does simple win? Three tabular datasets at increasing difficulty

This notebook runs `stepzero` on three well-known classification datasets and lets the
headroom signal tell you what to do next.

| Dataset | Expected difficulty | Why |
|---------|-------------------|-----|
| Wine | Easy | Linearly separable classes |
| Titanic | Medium | Missing values, non-linear patterns |
| California Housing | Hard | Complex non-linear relationships |
"""),
    code("import stepzero as sz\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt"),

    md("## Dataset 1 — Wine (easy)\n\nThree wine cultivars separated by chemical measurements. Logistic regression should nail this."),
    code("""\
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True, as_frame=True)
print(f"{len(X)} samples, {X.shape[1]} features, {len(set(y))} classes")
X.head(3)
"""),
    code("""\
result = sz.classify(X, y)
print(result)
print()
print(result.headroom)
"""),
    code("""\
result.feature_importance.head(8).plot(kind="barh", title="Wine: top features")
"""),

    md("## Dataset 2 — Titanic (medium)\n\nThe Kaggle classic. Simple models get ~80%; feature engineering and ensembles push toward 83–85%."),
    code("""\
import seaborn as sns
titanic = sns.load_dataset("titanic")

# Basic feature prep — keep the columns a simple model can use
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

df = titanic[features + [target]].dropna(subset=[target])
X = df[features]
y = df[target]
print(f"{len(X)} samples after dropping missing target")
X.head(3)
"""),
    code("""\
result = sz.classify(X, y)
print(result)
print()
print(result.headroom)
"""),
    code("""\
if result.feature_importance is not None:
    result.feature_importance.head(10).plot(kind="barh", title="Titanic: top features")
"""),

    md("""\
### What stepzero found
The headroom signal should be **Medium** — simple models plateau around 80% accuracy on Titanic.
Features like `title` (extracted from name), `family_size`, and `deck` are the usual gains.
"""),

    md("## Dataset 3 — California Housing (hard)\n\nPredict house prices from 8 features. Non-linear geography effects (latitude/longitude) mean\nlinear models leave significant RMSE on the table."),
    code("""\
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
print(f"{len(X)} samples, {X.shape[1]} features")
print(f"Target range: ${y.min()*100_000:.0f} – ${y.max()*100_000:.0f}")
X.head(3)
"""),
    code("""\
result = sz.regress(X, y)
print(result)
print()
print(result.headroom)
"""),
    code("""\
result.feature_importance.plot(kind="barh", title="California Housing: feature importance")
"""),
    code("""\
# Visualise residuals of the best model
y_pred = result.best_model.predict(X)
residuals = y - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred, residuals, alpha=0.1, s=5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("predicted")
axes[0].set_ylabel("residual")
axes[0].set_title(f"Residuals — {result.best_model_name}")

axes[1].hist(residuals, bins=60, edgecolor="none")
axes[1].set_title("Residual distribution")
"""),

    md("""\
### Summary

| Dataset | Best model | Score | Headroom |
|---------|-----------|-------|----------|
| Wine | — | — | — |
| Titanic | — | — | — |
| California Housing | — | — | — |

Fill in from the outputs above. The progression from Low → Medium → High headroom shows
exactly when it's worth reaching for gradient boosting.
"""),
])


# ── Notebook 3: Time series ───────────────────────────────────────────────────

nb3 = nb([
    md("""\
# Time series: forecasting and anomaly detection

Three scenarios at increasing difficulty:

1. **Strong trend + seasonality** — linear trend model should win comfortably
2. **Noisy seasonal data** — neither simple model fits well → High headroom
3. **IoT sensor with anomalies** — both detection methods should agree
"""),
    code("""\
import stepzero as sz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""),

    md("## Scenario 1 — Strong trend + seasonality (easy)\n\nClear upward trend and yearly cycle. `linear_trend` should beat `seasonal_naive` clearly."),
    code("""\
rng = np.random.default_rng(0)
idx = pd.date_range("2018-01", periods=60, freq="ME")
trend = np.linspace(100, 200, 60)
seasonal = 15 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = rng.normal(0, 2, 60)
ts_easy = pd.Series(trend + seasonal + noise, index=idx, name="sales")

ts_easy.plot(title="Scenario 1: trend + seasonality", figsize=(10, 3))
"""),
    code("""\
result = sz.forecast(ts_easy, horizon=12)
print(result)
print()
print(result.headroom)
print()
for s in result.scores:
    print(f"  {s.name:<18} MAE = {s.score:.2f}")
"""),
    code("""\
fig, ax = plt.subplots(figsize=(10, 4))
ts_easy.plot(ax=ax, label="observed")
result.forecast.plot(ax=ax, linestyle="--", marker="o", label=f"{result.best_model_name} forecast")
ax.legend()
ax.set_title("Scenario 1 — 12-month forecast")
"""),

    md("## Scenario 2 — Noisy seasonal data (hard)\n\nHigh noise swamps the seasonal signal. Neither simple model fits well."),
    code("""\
rng2 = np.random.default_rng(42)
idx2 = pd.date_range("2020-01", periods=36, freq="ME")
seasonal2 = 5 * np.sin(2 * np.pi * np.arange(36) / 12)
noise2 = rng2.normal(0, 20, 36)  # noise >> signal
ts_hard = pd.Series(50 + seasonal2 + noise2, index=idx2, name="demand")

ts_hard.plot(title="Scenario 2: high-noise seasonal", figsize=(10, 3))
"""),
    code("""\
result2 = sz.forecast(ts_hard, horizon=6)
print(result2)
print()
print(result2.headroom)
"""),
    code("""\
fig, ax = plt.subplots(figsize=(10, 4))
ts_hard.plot(ax=ax, label="observed")
result2.forecast.plot(ax=ax, linestyle="--", marker="o", label=f"{result2.best_model_name} forecast")
ax.legend()
ax.set_title("Scenario 2 — high headroom: neither simple model is reliable")
"""),

    md("## Scenario 3 — IoT sensor with anomalies\n\nA temperature sensor running normally with occasional spikes and a slow drift."),
    code("""\
rng3 = np.random.default_rng(7)
t = np.arange(500)
normal_signal = 20 + 0.01 * t + rng3.normal(0, 0.5, 500)

# Inject anomalies: point spikes and a step change
anomaly_idx = [50, 150, 250, 350, 450]
normal_signal[anomaly_idx] += rng3.choice([-8, 8, 10, -10, 9])
# Brief step change
normal_signal[200:215] += 5

sensor = pd.Series(normal_signal, name="temperature_C")

fig, ax = plt.subplots(figsize=(12, 3))
sensor.plot(ax=ax, alpha=0.8)
ax.axvspan(200, 215, alpha=0.1, color="orange", label="step anomaly")
ax.set_title("IoT sensor signal")
ax.legend()
"""),
    code("""\
result3 = sz.detect_anomalies(sensor)
print(result3)
print()
print(result3.headroom)
"""),
    code("""\
# Compare z-score vs IQR side by side
result_z = sz.detect_anomalies(sensor, method="zscore")
result_iqr = sz.detect_anomalies(sensor, method="iqr")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, res, title in zip(
    axes,
    [result3, result_z, result_iqr],
    [f"auto → {result3.method}", "z-score", "iqr"],
):
    sensor.plot(ax=ax, alpha=0.6, label="signal")
    sensor[res.anomalies].plot(ax=ax, style="rv", markersize=8, label="flagged")
    ax.set_title(f"{title}  (threshold={res.threshold:.2f}, flagged={res.anomalies.sum()})")
    ax.legend(loc="upper right")

plt.tight_layout()
"""),

    md("""\
### Takeaways

- **Scenario 1**: `linear_trend` beats `seasonal_naive` → headroom Low/Medium, simple model is production-ready
- **Scenario 2**: Both models struggle with high noise → headroom High, reach for ARIMA or Prophet
- **Scenario 3**: Z-score and IQR agree on clear point anomalies; the step change at t=200–215 may need a more sophisticated detector (e.g., Isolation Forest, CUSUM)
"""),
])


# ── Write notebooks ───────────────────────────────────────────────────────────

out = Path(__file__).parent

for filename, notebook in [
    ("01_quickstart.ipynb", nb1),
    ("02_tabular_difficulty.ipynb", nb2),
    ("03_time_series.ipynb", nb3),
]:
    path = out / filename
    path.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {path}")
