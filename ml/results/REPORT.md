# GLOBAL STATUS MONITORING SYSTEM — Machine Learning Laboratory Report

```
24B11CS489: Viswajeet
24B11CS357: Priyamwda Rai
24B11CS461: Satya Sri Hema
24B11CS256: Uma Ram Charan
```

Generated 2026-10-04T13:42:43.690296+00:00 from **2919 raw articles** (2409 after cleaning, 510 near-duplicates removed, 0 numeric values imputed), **1875 USGS earthquakes**, and **2919 discretized concept rows**. Chronological split: 1686 train / 361 validation / 362 test (oldest→newest; the test set is strictly future relative to training). Seeds pinned at 42.

## 1. Data provenance

Real, keyless, public feeds — the same ones the WorldMonitor dashboard ingests: GDELT DOC API v2 artlist (21 topical and source-country queries, trailing window) and the USGS GeoJSON all-week earthquake feed. The concept table discretizes the same articles for symbolic learning.

## 2. Supervised classification (target: derived risk band)

Labels are the keyword-classifier risk bands (low/medium/high/critical) computed from the headline text by transparent rules — **derived heuristics, not ground truth**.

| model | accuracy | precision(macro) | recall(macro) | f1(macro) | roc-auc | 5-fold CV accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| GaussianNB | 0.1851 | 0.3345 | 0.3948 | 0.1060 | n/a | 0.2991 ± 0.1350 |
| kNN(k=5,distance) | 0.9834 | 0.4972 | 0.4417 | 0.4639 | n/a | 0.9541 ± 0.0029 |
| LogisticRegression | 0.9779 | 0.3296 | 0.3296 | 0.3296 | n/a | 0.9565 ± 0.0024 |
| LinearSVC | 0.9862 | 0.3296 | 0.3324 | 0.3310 | n/a | 0.9580 ± 0.0010 |
| SVC(rbf) | 0.9890 | 0.3297 | 0.3333 | 0.3315 | n/a | 0.9585 ± 0.0000 |

Holdout leader: **SVC(rbf)** (accuracy 0.9890).

## 3. Linear regression (target: derived risk_score 0-100)

| metric | value |
| --- | --- |
| MAE | 5.080 |
| MSE | 67.170 |
| RMSE | 8.196 |
| R² (test) | -0.7642 |
| R² (train) | 0.1329 |
| mean actual | 4.68 |

Standardized coefficients (largest magnitude first):

| feature | coefficient |
| --- | --- |
| sentiment_score | -3.0022 |
| word_count | +1.6420 |
| corroboration_count | +1.2311 |
| text_len | -0.8446 |
| upper_ratio | -0.6593 |
| is_breaking | +0.4571 |
| velocity_per_hour | +0.3237 |

## 4. Clustering

Outlier screen: 12 of 2409 rows excluded from K-Means/Ward/GMM (single extreme artifacts; DBSCAN sees the full matrix and labels them noise).

- **K-Means**: best k=3, silhouette=0.2808, sizes={'0': 1692, '1': 696, '2': 9}.
- **Agglomerative (Ward)**: best k=2, silhouette=0.827, sizes={'0': 2388, '1': 9}.
- **DBSCAN**: eps=3.2903 (k-distance knee), min_samples=5 → 4 clusters, 9 noise points (0.4%), silhouette=0.6895.
- **GMM (EM)**: components=3, silhouette=0.4586, sizes={'0': 18, '1': 2248, '2': 131}, converged=True.
- **semantic event clustering** (K-Means over 32-dim LSA embeddings of all headlines): best k=2, silhouette=0.754, sizes={'0': 2396, '1': 13}.
- PCA(2) of the numeric features explains 79.7% of variance (69.3% + 10.4%).

## 5. Ensemble methods (target: derived risk band)

| ensemble | accuracy | f1(macro) | #importance-entries |
| --- | --- | --- | --- |
| RandomForest | 0.9917 | 0.4986 | 7 |
| AdaBoost | 0.9862 | 0.4981 | 7 |
| Bagging(stumps) | 0.9890 | 0.3315 | n/a |

Random Forest feature importances (top 5):

| feature | importance |
| --- | --- |
| sentiment_score | 0.3362 |
| upper_ratio | 0.1914 |
| velocity_per_hour | 0.1635 |
| text_len | 0.1552 |
| word_count | 0.1130 |

## 6. Evaluation toolbox

- **Bootstrap CI** (accuracy of LogisticRegression on the test split, n_boot=200): point 0.9779, 95% CI [0.9641, 0.9917].
- **Bias/variance (tree depth sweep)**:

| depth | train error | test error |
| --- | --- | --- |
| 1 | 0.0415 | 0.0110 |
| 2 | 0.0415 | 0.0110 |
| 3 | 0.0386 | 0.0138 |
| 4 | 0.0352 | 0.0193 |
| 5 | 0.0308 | 0.0166 |
| 6 | 0.0249 | 0.0193 |
| 7 | 0.0225 | 0.0331 |
| 8 | 0.0191 | 0.0552 |
| 9 | 0.0166 | 0.0552 |
| 10 | 0.0132 | 0.0773 |
| 11 | 0.0098 | 0.0414 |
| 12 | 0.0059 | 0.0746 |

- **Regularization (LogisticRegression C sweep)**:

| C | train accuracy | test accuracy | gap |
| --- | --- | --- | --- |
| 0.001 | 0.9585 | 0.9862 | -0.0277 |
| 0.01 | 0.9580 | 0.9834 | -0.0254 |
| 0.1 | 0.9585 | 0.9862 | -0.0277 |
| 1.0 | 0.9580 | 0.9862 | -0.0282 |
| 10.0 | 0.9580 | 0.9862 | -0.0282 |
| 100.0 | 0.9580 | 0.9862 | -0.0282 |

## 7. Concept learning (Find-S and Candidate Elimination)

Over 2919 discretized event rows (28 positive for `high_risk`), attributes: `category`, `region`, `alert_level`, `sentiment_class`, `corroboration_class`, `breaking`.

**Find-S** hypothesis (most specific, '?' = any value):

```json
{
  "category": "?",
  "region": "?",
  "alert_level": "high",
  "sentiment_class": "?",
  "corroboration_class": "?",
  "breaking": "0"
}
```

**Candidate Elimination**: |S|=1, |G|=1, version-space boundary pairs=1, 0 noisy examples skipped (0.0% of the table). First step of the S boundary: 

```json
{
  "category": "?",
  "region": "?",
  "alert_level": "high",
  "sentiment_class": "?",
  "corroboration_class": "?",
  "breaking": "0"
}
```

## 8. Global Status engine (academic derived score)

Per-country 0-100 status from real signals: weighted sum of severity(30) + event_volume(20) + negativity(15) + corroboration(10) + seismic(15) + news_velocity(10). Window: 68.25h. **This is an academic derived score for this report — not the dashboard's production CII.**

| country | score | band | events | worst level | avg sentiment | quakes | max mag | velocity/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Global | 85 | Critical | 1924 | critical | -0.003 | 766 | 6.6 | 28.19 |
| US | 85 | Critical | 271 | critical | -0.018 | 1017 | 4.2 | 3.97 |
| RU | 78 | Critical | 37 | critical | -0.308 | 20 | 4.6 | 0.54 |
| IQ | 74 | High | 76 | critical | -0.149 | 1 | 5.3 | 1.11 |
| IL | 71 | High | 66 | critical | -0.381 | 0 | 0.0 | 0.97 |
| IR | 70 | High | 141 | critical | +0.008 | 0 | 0.0 | 2.07 |
| IN | 63 | High | 96 | high | +0.086 | 1 | 4.5 | 1.41 |
| UA | 60 | High | 23 | critical | -0.320 | 0 | 0.0 | 0.34 |
| CA | 58 | High | 14 | high | +0.054 | 14 | 5.2 | 0.21 |
| JP | 58 | High | 26 | high | -0.089 | 7 | 4.9 | 0.38 |
| MX | 55 | High | 10 | high | -0.201 | 203 | 4.5 | 0.15 |
| NG | 54 | High | 20 | critical | -0.033 | 0 | 0.0 | 0.29 |

Component breakdown for the top country (Global, score 85):

| component | value (0-1) | weight |
| --- | --- | --- |
| severity | 1.000 | 30 |
| event_volume | 1.000 | 20 |
| negativity | 0.003 | 15 |
| corroboration | 1.000 | 10 |
| seismic | 1.000 | 15 |
| news_velocity | 1.000 | 10 |

## 9. Semantic retrieval demo (LSA nearest neighbours)

Query headline: *Atiku asks FG to publish Nigeria - US health agreement after Ghana rejection - Latest News In Nigeria , Nigeria News Today , Your Online Nigerian Newspaper*

| rank | neighbour headline | cosine |
| --- | --- | --- |
| 1 | Atiku asks FG to publish Nigeria - US health agreement after Ghana rejection - Latest News In Nigeria , Nigeria News Today , Your Online Nigerian Newspaper | 1.0000 |
| 2 | Submit applications online for temporary crackers shop | 0.8095 |
| 3 | Israel - Bound Flight Diverted After Fight Breaks Out Between Pilots - Latest News In Nigeria , Nigeria News Today , Your Online Nigerian Newspaper | 0.7614 |
| 4 | Nigeria cuts Ways and Means debt by ₦613bn in Q2  26 | 0.7154 |
| 5 | Genocide Is Genocide : Why Palestine Matters to Nigeria | 0.7055 |

## Limitations (honest)

1. **All labels are derived heuristics.** risk_score, risk_band and the concept table's high_risk flag come from the transparent keyword classifier; classifiers trained on them measure agreement with that labeling function, not with reality. High accuracy here is expected and is NOT evidence of real-world predictive power.
2. **Short window, topical queries.** The corpus covers a few days fetched through topical and source-country queries; it is not a representative sample of world news, and GDELT coverage varies by country and language (English only).
3. **Feature leakage risk is structural, not accidental.** risk_score is a deterministic function of features that include the classifier's level weights; regression R² and band accuracy partly rediscover that formula. This is intentional for the lab (it makes the pipeline auditable) but should not be read as discovery.
4. **Country attribution is coarse.** Gazetteer NER misses countries not named in the headline; the quake bounding boxes are generous rectangles with known overlaps; unmatched events/quakes pool into 'Global'.
5. **Simple models, small feature set.** Seven numeric features + 400 TF-IDF terms; no deep learning, no multilingual text, no image data. Class imbalance (rare 'critical' band) is handled with macro averages and zero_division guards, but rare-class metrics remain noisy — hence the bootstrap interval rather than a single number.
6. **DBSCAN eps is a heuristic** (k-distance knee), silhouette has known biases on uneven cluster sizes, and the LSA subspace uses a fixed 20 iterations rather than full convergence diagnostics.

---

```
24B11CS489: Viswajeet
24B11CS357: Priyamwda Rai
24B11CS461: Satya Sri Hema
24B11CS256: Uma Ram Charan
```
