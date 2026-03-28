# stepzero

**Task-first ML baselines. Run the simplest thing that could work.**

Before reaching for XGBoost or a neural net, run `stepzero`. It fits the simplest sensible model for your task, compares a few alternatives, and tells you whether your baseline is good enough or what to try next.

```python
import stepzero as sz

result = sz.classification(X, y)
print(result)
# ClassificationResult(best='logistic', accuracy=0.960, headroom='Low')

print(result.headroom)
# [Low] Score of 0.96 with low variance (±0.012). The simple baseline is already
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

Models compared: Logistic Regression, Decision Tree (depth 5), Gaussian Naive Bayes.
Metric: accuracy (5-fold stratified CV).

### Regression

```python
result = sz.regression(X, y)

result.best_model_name     # "ridge" | "tree"
result.feature_importance  # normalized importances as pd.Series
result.headroom            # compared against predict-mean baseline
```

Models compared: Ridge (α=1), Decision Tree (depth 5).
Metric: RMSE.

### Forecasting

```python
import pandas as pd

ts = pd.Series(..., index=pd.date_range("2020-01", periods=48, freq="ME"))
result = sz.forecasting(ts, horizon=12)

result.forecast            # pd.Series with future index
result.best_model_name     # "seasonal_naive" | "linear_trend"
result.scores              # MAE for each model
result.headroom
```

Models compared: Seasonal Naive, Linear Trend + Seasonal Decomposition.
Metric: MAE (time-series CV).

### Anomaly Detection

```python
result = sz.anomaly_detection(series)

result.anomalies           # pd.Series[bool], same index as input
result.scores              # raw anomaly scores
result.method              # "zscore" | "iqr"
result.threshold           # auto-determined threshold
result.headroom
```

Methods: Z-score, IQR. `method="auto"` (default) runs both and picks the most consistent one.

### Text Classification

```python
result = sz.text_classification(texts, labels)

result.best_model_name          # "tfidf_logistic" | "tfidf_naive_bayes"
result.top_features_per_class   # {"class_0": ["word1", ...], ...}
result.headroom
```

Models compared: TF-IDF + Logistic Regression, TF-IDF + Multinomial Naive Bayes.

### Clustering

```python
result = sz.clustering(X, k_range=(2, 10))

result.best_k     # selected number of clusters
result.labels     # cluster assignment per sample (np.ndarray)
result.centers    # cluster centroids in original feature space
result.scores     # silhouette score for each k tried
result.headroom
```

Method: K-means with auto-k via elbow + silhouette maximization.

---

## The headroom signal

Every result has a `.headroom` attribute:

```python
result.headroom.level   # "Low" | "Medium" | "High"
result.headroom.reason  # actionable explanation + what to try next
print(result.headroom)
# [Medium] CV accuracy of 0.81 ± 0.04. A 19% gap to ceiling remains.
# A gradient boosted tree (e.g., XGBoost or LightGBM) is a natural next step.
```

- **Low** — the simple model is already doing well; complexity buys little
- **Medium** — meaningful headroom remains; a tuned model may help
- **High** — the baseline is underperforming; a more complex model is likely worth it

---

## Design philosophy

- **Task-first, not model-first.** You describe the problem; stepzero picks the approach.
- **Opinionated defaults.** Auto-scaling for linear models, missing value imputation, sensible eval.
- **No false modesty.** The models are genuinely simple — logistic regression, decision trees, seasonal naive. No AutoML hidden underneath.
- **Ready to deploy.** `result.best_model` is a fitted sklearn `Pipeline`. Call `.predict()` on new data immediately.
- **Minimal footprint.** Only numpy, pandas, scikit-learn, and scipy. No optional heavy dependencies required for core functionality.

---

## When to use stepzero

✅ Starting a new ML project and want a defensible baseline in 5 minutes
✅ Proving (or disproving) that a simple model is good enough
✅ Teaching or demonstrating ML without the XGBoost-first bias
✅ Kaggle competitions — establish your baseline before tuning

❌ You already know a complex model is needed
❌ You need production-grade model selection with hyperparameter tuning (use AutoML)

---

## License

MIT
