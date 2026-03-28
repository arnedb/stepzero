# stepzero

[![Tests](https://github.com/arnedb/stepzero/actions/workflows/tests.yml/badge.svg)](https://github.com/arnedb/stepzero/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/stepzero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Task-first ML baselines. Run the simplest thing that could work.**

Before reaching for XGBoost or a neural net, run `stepzero`. It fits the simplest sensible model for your task, compares a few alternatives, and tells you whether your baseline is good enough or what to try next.

```python
import stepzero as sz

result = sz.classification(X, y)
print(result)
# ClassificationResult(best='logistic', accuracy=0.960, headroom='low')

print(result.headroom)
# [low] Score of 0.96 with low variance (±0.012). The simple baseline is already
# performing well. Trying a gradient boosted tree is unlikely to offer a meaningful improvement.
```

---

## Install

```bash
pip install stepzero
```

**Requirements**: Python 3.10+, numpy, pandas, scikit-learn, scipy.

---

## Tasks

### Classification

```python
result = sz.classification(X, y)

result.best_model          # fitted sklearn Pipeline — call .predict(X_new) directly
result.best_model_name     # "logistic" | "tree" | "naive_bayes"
result.scores              # [ModelScore(name, score, metric), ...]
result.feature_importance  # pd.Series sorted by importance
result.headroom            # HeadroomSignal(level, reason)
```

- **Methods:** logistic regression, decision tree, naive bayes
- **Metric:** accuracy (5-fold stratified CV)

### Regression

```python
result = sz.regression(X, y)

result.best_model_name     # "ridge" | "tree"
result.feature_importance  # normalized importances as pd.Series
result.headroom
```

- **Methods:** ridge, decision tree
- **Metric:** RMSE (5-fold CV)

### Forecasting

```python
result = sz.forecasting(series, horizon=12)

result.forecast        # pd.Series with future timestamps as index
result.best_model_name # "seasonal_naive" | "linear_trend"
result.scores          # MAE per model
result.headroom
```

- **Methods:** seasonal naive, linear trend
- **Parameters:** `horizon`, `freq` (optional — inferred from DatetimeIndex), `cv_splits`
- **Metric:** MAE (time-series CV)

### Anomaly Detection

```python
result = sz.anomaly_detection(series)

result.anomalies   # pd.Series[bool], same index as input
result.scores      # raw anomaly scores
result.method      # "zscore" | "iqr"
result.threshold   # auto-determined threshold
result.headroom
```

- **Methods:** z-score, IQR
- **Parameters:** `threshold` (optional — auto-set to flag ~5% of points), `method`
- **Metric:** inter-method agreement

### Text Classification

```python
result = sz.text_classification(texts, labels)

result.best_model_name        # "tfidf_logistic" | "tfidf_naive_bayes"
result.top_features_per_class # {"class_0": ["word1", ...], ...}
result.headroom
```

- **Methods:** TF-IDF + logistic regression, TF-IDF + naive bayes
- **Metric:** accuracy (5-fold stratified CV)

### Clustering

```python
result = sz.clustering(X, k_range=(2, 10))

result.best_k    # selected number of clusters
result.labels    # cluster assignment per sample (np.ndarray)
result.centers   # cluster centroids in original feature space
result.scores    # silhouette score per k tried
result.headroom
```

- **Methods:** k-means
- **Parameters:** `k_range`
- **Metric:** silhouette score

---

## The headroom signal

Every result has a `.headroom` attribute:

```python
result.headroom.level   # "low" | "medium" | "high"
result.headroom.reason  # actionable explanation + what to try next
print(result.headroom)
# [medium] CV accuracy of 0.81 ± 0.04. A 19% gap to ceiling remains.
# A gradient boosted tree (e.g., XGBoost or LightGBM) is a natural next step.
```

- **low** means that the simple model is already doing well; complexity buys little
- **medium** means that meaningful headroom remains; a tuned model may help
- **high** means that the baseline is underperforming; a more complex model is likely worth it

---

## Design philosophy

- **Task-first, not model-first.** You describe the problem; stepzero picks the approach.
- **Opinionated defaults.** Auto-scaling for linear models, missing value imputation, sensible eval.
- **No false modesty.** The models are genuinely simple — logistic regression, decision trees, seasonal naive. No AutoML hidden underneath.
- **Ready to deploy.** `result.best_model` is a fitted sklearn `Pipeline`. Call `.predict()` on new data immediately.
- **Minimal footprint.** Only numpy, pandas, scikit-learn, and scipy. No optional heavy dependencies required for core functionality.

---

## When to use stepzero

- ✅ Starting a new ML project and want a defensible baseline in 5 minutes
- ✅ Proving (or disproving) that a simple model is good enough
- ✅ Teaching or demonstrating ML without the XGBoost-first bias
- ✅ Kaggle competitions — establish your baseline before tuning

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

In short: branch from `develop`, open a PR targeting `develop`. All PRs run the test suite automatically across Python 3.10–3.12.

## Reporting issues

Open an issue on GitHub. Include your Python version, stepzero version, and a minimal reproducible example.

---

## License

MIT
