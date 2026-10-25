# Global Status Monitoring System — Machine Learning Project Documentation

## Project Identity

**Project Name:** Global Status Monitoring System  
**Course:** Machine Learning (241AI005) — B.Tech CSE 5th Semester  
**Institution:** Common to CSE, IT, AIML & CSE(DS)  
**Academic Year:** 2024-2025

### Student Team

| Roll Number | Name |
|-------------|------|
| 24B11CS489 | Viswajeet |
| 24B11CS357 | Priyamwda Rai |
| 24B11CS461 | Satya Sri Hema |
| 24B11CS256 | Uma Ram Charan |

---

## Problem Statement

The Global Status Monitoring System provides real-time situational awareness by combining heterogeneous global news and event information into a unified dashboard. The system performs data cleaning, deduplication, preprocessing, NLP, entity extraction, sentiment analysis, TF-IDF/embedding-based representation, semantic similarity and event clustering.

Machine Learning techniques from the course syllabus are applied to the resulting data through classification, regression, instance-based learning, SVM, Bayesian learning, clustering and ensemble learning. Random Forest and AdaBoost are used for ensemble-based event/status analysis, while k-NN supports similarity-based historical/event retrieval. K-Means, hierarchical clustering and DBSCAN are used for unsupervised pattern discovery. Model performance is evaluated using accuracy, precision, recall, F1-score, ROC/AUC, cross-validation and bootstrap-based analysis.

The system presents these outputs through an interactive global dashboard, making the project both an operational monitoring interface and a practical demonstration of the Machine Learning syllabus.

---

## Objectives

1. **Data Pipeline**: Build a clean ML-ready event/article representation from real-time global news feeds (GDELT, RSS) and event sources (USGS, UCDP, ACLED)
2. **NLP Layer**: Implement TF-IDF, sentiment analysis, Named Entity Recognition, text classification, sentence embeddings, and semantic clustering
3. **Supervised Learning**: Implement Logistic Regression, SVM (Linear/RBF), k-NN, Gaussian Naive Bayes for event risk classification
4. **Regression**: Implement Linear Regression for continuous risk score estimation
5. **Concept Learning**: Implement Find-S and Candidate Elimination algorithms on structured event data
6. **Unsupervised Learning**: Implement K-Means, Hierarchical Clustering, DBSCAN, and EM/Gaussian Mixture for pattern discovery
7. **Ensemble Learning**: Implement Random Forest, AdaBoost, and Bagging with feature importance
8. **Model Evaluation**: Comprehensive evaluation with cross-validation, bootstrap, bias/variance analysis
9. **Global Status Engine**: Transparent per-country risk scoring (0-100) with explainable drivers
10. **Interactive Dashboard**: Web-based ML Lab panel demonstrating the full pipeline on live data

---

## Dataset

### Primary Sources

| Source | Type | Access | Refresh Rate |
|--------|------|--------|--------------|
| GDELT DOC API v2 | News articles | Keyless (public) | 15 min |
| USGS Earthquake Feed | Seismic events | Public GeoJSON | 5 min |
| UCDP Conflict Events | Armed conflict | Public CSV | Weekly |
| ACLED | Political violence | Public API | Daily |
| RSS Feeds (500+) | Curated news | Public | 15-60 min |

### Derived Dataset (Committed Snapshot)

The project ships with a committed offline snapshot for reproducible demos:

- **ml/data/gsms_events.csv** — ~5-8k news articles with derived features
- **ml/data/gsms_quakes.csv** — ~1k seismic events
- **ml/data/gsms_concept.csv** — Synthetic concept learning dataset

### ML-Ready Event Representation

Each event/article contains:

```python
{
    'headline': str,           # Article title
    'source': str,             # News source
    'pub_date': int,           # Epoch milliseconds
    'country': str,            # ISO country code (where available)
    'region': str,             # Geographic region
    'category': str,           # Event category (conflict, disaster, economic, etc.)
    'lat': float,              # Latitude
    'lon': float,              # Longitude
    'sentiment': float,        # AFINN-based sentiment [-1, 1]
    'entities': List[str],     # Gazetteer-matched entities
    'threat_level': str,       # info/low/medium/high/critical
    'risk_band': str,          # Derived: low/medium/high/critical
    'high_risk': bool,         # Binary: risk_band in {high, critical}
    'corroboration': int,      # Multi-source count
    'credibility': float,      # Source credibility score [0,1]
    'text_features': dict,     # Length, word count, upper-case ratio, hour
}
```

### Label Derivation (Heuristic — Not Ground Truth)

**Risk Bands (4-class):**
- `critical`: threat_level=critical OR (threat_level=high AND corroboration≥3)
- `high`: threat_level=high OR (threat_level=medium AND corroboration≥2)
- `medium`: threat_level=medium OR corroboration≥1
- `low`: threat_level=low/info with no corroboration

**High-Risk (Binary):** `risk_band ∈ {high, critical}`

**Regression Target:** Continuous risk score = Σ(level_weight × corroboration) + sentiment_penalty

> **Important**: All labels are derived heuristics from the project's own threat classifier and corroboration counts. They are NOT human-annotated ground truth. This is clearly documented throughout the codebase and UI.

---

## Preprocessing Pipeline

Implemented in `ml/gsms_ml/preprocess.py`:

1. **Duplicate Removal**: Jaccard similarity on content tokens (threshold 0.85)
2. **Missing Value Handling**: Median imputation (numeric), mode (categorical)
3. **Normalization/Standardization**: Z-score fitted on training split only
4. **Categorical Encoding**: Ordinal for risk bands, one-hot for categories
5. **Text Preprocessing**: Lowercase, stopword removal, tokenization
6. **Chronological Split**: 80/20 train/test by publication date (prevents future-data leakage)

---

## NLP Layer

Implemented across `ml/gsms_ml/text.py` and browser worker `src/workers/ml-lab.worker.ts`:

| Component | Implementation | Location |
|-----------|---------------|----------|
| TF-IDF | scikit-learn TfidfVectorizer (max_features=5000) + TruncatedSVD (n=100) | Python: `text.py` |
| Sentiment | AFINN-165 lexicon (committed) | Python: `lexicon.py`, TS: `ml-lab.worker.ts` |
| NER | Gazetteer-based (countries, cities, orgs, persons) | Python: `gazetteer.py` |
| Text Classification | Keyword-based threat classifier (shared) | Shared: `threat-keyword-classifier.ts` |
| Embeddings | TF-IDF + SVD → cosine similarity | Python: `text.py` |
| Semantic Clustering | Agglomerative on embeddings | Python: `clustering.py` |

---

## Supervised Learning

Implemented in `ml/gsms_ml/supervised.py` (Python) and `src/workers/ml-lab.worker.ts` (TypeScript):

### Classification Models

| Model | Python (sklearn) | TypeScript (from-scratch) | Target |
|-------|------------------|---------------------------|--------|
| Logistic Regression | `LogisticRegression(max_iter=1000)` | SGD (mini-batch) | risk_band (4-class), high_risk (binary) |
| Linear SVM | `LinearSVC(dual='auto')` | Pegasos primal SGD | risk_band, high_risk |
| RBF SVM | `SVC(kernel='rbf')` | Kernelized (subsampled) | risk_band, high_risk |
| Gaussian Naive Bayes | `GaussianNB()` | Analytical Gaussian | risk_band, high_risk |
| k-NN | `KNeighborsClassifier(n_neighbors=5)` | Brute-force cosine/euclidean | risk_band, high_risk |

### Metrics Computed (Per Model)

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion Matrix
- ROC-AUC (OvR for multi-class)

### Evaluation Protocol

- **Chronological 5-fold CV**: Time-series aware splits
- **Bootstrap CI**: 1000 resamples, 95% percentile intervals
- **Bias-Variance**: Tree depth sweep (1-10)
- **Regularization**: Logistic regression λ sweep (1e-4 to 1e2)

---

## Regression

Implemented in `ml/gsms_ml/regression.py` and browser worker:

**Target:** Continuous risk score (0-100) derived from threat levels, corroboration, sentiment

**Model:** Linear Regression via Normal Equations (Python: `LinearRegression`, TS: analytical solution)

**Metrics:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Predicted vs Actual scatter (sample)

> **Disclaimer**: This is statistical estimation on derived features, NOT a geopolitical forecast.

---

## Concept Learning

Implemented in `ml/gsms_ml/concept_learning.py` and browser worker:

### Find-S Algorithm
- Reads structured CSV (`gsms_concept.csv`)
- Demonstrates most-specific hypothesis generalization from positive examples
- Outputs step-by-step hypothesis evolution

### Candidate Elimination
- Tracks Specific (S) and General (G) boundaries
- Shows version space evolution across examples
- Handles inconsistent data with notes

---

## Unsupervised Learning

Implemented in `ml/gsms_ml/clustering.py` and browser worker:

| Algorithm | Python (sklearn) | TypeScript | Notes |
|-----------|------------------|------------|-------|
| K-Means | `KMeans(n_clusters=k, init='k-means++')` | k-means++ with PRNG | Multiple k tested |
| Hierarchical | `AgglomerativeClustering` | Bottom-up (subsampled) | Ward linkage |
| DBSCAN | `DBSCAN(eps, min_samples)` | Exact (O(n²) optimized) | Noise points identified |
| EM/GMM | `GaussianMixture` | Not in browser (heavy) | Python only |

**Visualization:** PCA via Jacobi eigen decomposition (2D projection)

**Metrics:** Silhouette score, inertia, cluster sizes, noise count

---

## Ensemble Learning

Implemented in `ml/gsms_ml/ensemble.py` and browser worker:

| Ensemble | Python | TypeScript | Feature Importance |
|----------|--------|------------|-------------------|
| Random Forest | `RandomForestClassifier(n_estimators=200)` | Decision tree ensemble (subsampled) | Gini importance |
| AdaBoost | `AdaBoostClassifier(n_estimators=100)` | Staged decision stumps | N/A |
| Bagging | `BaggingClassifier(base_estimator=DecisionTree)` | Bootstrap aggregating (stumps) | N/A |

---

## Model Evaluation

Implemented in `ml/gsms_ml/evaluation.py`:

- **Train/Test Split**: Chronological 80/20
- **5-Fold Cross-Validation**: Time-series aware (no shuffle)
- **Bootstrap**: 1000 resamples, percentile CI
- **Bias-Variance Decomposition**: Decision tree depth sweep
- **Regularization Path**: Logistic regression λ sweep
- **Overfitting Demonstration**: Train vs test error across complexity

---

## Global Status Engine

Implemented in `ml/gsms_ml/status_engine.py`:

**Per-Country Risk Score (0-100):**

```
score = w1×event_frequency + w2×news_velocity + w3×sentiment
      + w4×conflict_indicator + w5×military_indicator
      + w6×protest_indicator + w7×disaster_indicator
      + w8×economic_indicator + w9×cyber_indicator
      + w10×infrastructure_indicator
```

**Status Bands:**
- 0-25: Stable
- 26-50: Elevated
- 51-75: High Risk
- 76-100: Critical

**Outputs:** Country code, score, band, driver breakdown, event count

---

## Architecture

### Python ML Lab (`ml/`)

```
ml/
├── requirements.txt          # Pinned dependencies
├── fetch_dataset.py          # Live data fetcher (GDELT + USGS)
├── run_all.py                # Orchestrates all experiments
├── data/                     # Committed snapshots (CSV)
│   ├── gsms_events.csv
│   ├── gsms_quakes.csv
│   └── gsms_concept.csv
├── gsms_ml/                  # Core ML package
│   ├── __init__.py
│   ├── gazetteer.py          # NER gazetteers
│   ├── lexicon.py            # AFINN sentiment
│   ├── text.py               # TF-IDF, embeddings, similarity
│   ├── preprocess.py         # Cleaning, encoding, splitting
│   ├── concept_learning.py   # Find-S, Candidate Elimination
│   ├── supervised.py         # Classification models
│   ├── regression.py         # Linear regression
│   ├── clustering.py         # K-Means, Hierarchical, DBSCAN, GMM
│   ├── ensemble.py           # RF, AdaBoost, Bagging
│   ├── evaluation.py         # Metrics, CV, bootstrap, bias/var
│   └── status_engine.py      # Global Status scoring
├── results/                  # JSON outputs + REPORT.md
│   ├── supervised.json
│   ├── regression.json
│   ├── clustering.json
│   ├── ensembles.json
│   ├── evaluation.json
│   ├── concept_learning.json
│   ├── status_engine.json
│   └── REPORT.md
└── tests/                    # pytest suites
    ├── test_preprocess.py
    ├── test_concept_learning.py
    ├── test_evaluation.py
    └── test_text.py
```

### TypeScript Browser ML Lab (`src/`)

```
src/
├── workers/
│   └── ml-lab.worker.ts      # Full ML suite in Web Worker (no deps)
├── services/
│   └── ml-lab-worker.ts      # Worker manager (singleton, typed)
├── components/
│   └── MLLabPanel.ts         # Dashboard panel with 8 tabs
├── data/
│   └── ml-sample-events.generated.ts  # Embedded sample (offline)
└── config/
    └── panels.ts             # 'ml-lab' panel registration
```

---

## How to Run

### Python ML Lab (Offline, Reproducible)

```bash
cd worldmonitor/ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run_all.py
# Results in ml/results/*.json and ml/results/REPORT.md
```

### Python Tests

```bash
cd worldmonitor/ml
python3 -m pytest tests/ -v
```

### Browser Dashboard (Development)

```bash
cd worldmonitor
npm run dev
# Opens at http://localhost:5173/dashboard
# Navigate to "ML Lab" panel in Intelligence category
```

### Browser Build & Typecheck

```bash
cd worldmonitor
npm run typecheck
npm run lint
npm run build
```

---

## Limitations

1. **Heuristic Labels**: All classification targets are derived from the project's own keyword classifier and corroboration counts — not human-annotated ground truth.
2. **No External LLM Dependency**: Core ML runs without external APIs (OpenAI, Anthropic, etc.). LLM features (summarization, analyst chat) are supplementary.
3. **Browser Worker Constraints**: TypeScript implementations are from-scratch (no sklearn) for zero-dependency operation. RBF SVM and GMM are subsampled/omitted for performance.
4. **Data Freshness**: Committed snapshot is static. Live mode fetches from GDELT/USGS but requires internet.
5. **Chronological Split Only**: No spatial or source-based stratification in CV.
6. **Binary/4-Class Only**: No ordinal regression or multi-label classification.
7. **No Hyperparameter Tuning**: Fixed hyperparameters for reproducibility.

---

## Syllabus Coverage Summary

| Syllabus Unit | Topics | Implemented |
|---------------|--------|-------------|
| Unit I | ML concepts, Statistical Learning, Preprocessing | ✅ Preprocessing pipeline, data loading |
| Unit II | Concept Learning (Find-S, Candidate Elimination), Bayesian Learning, Instance-based (k-NN) | ✅ Find-S, Candidate Elimination, Gaussian NB, k-NN |
| Unit III | Linear Models, Logistic Regression, SVM (Linear, RBF) | ✅ Linear/Logistic Regression, Linear SVM, RBF SVM |
| Unit IV | Evaluation (Accuracy, Precision, Recall, F1, ROC, CV, Bootstrapping), Clustering (K-Means, Hierarchical, DBSCAN) | ✅ All metrics, 5-fold CV, bootstrap, K-Means, Hierarchical, DBSCAN |
| Unit V | Ensemble Learning (Bagging, Boosting, Random Forest, AdaBoost) | ✅ Random Forest, AdaBoost, Bagging |

---

## File Index (Key Implementation Files)

| Component | File |
|-----------|------|
| Data Fetching | `ml/fetch_dataset.py` |
| Preprocessing | `ml/gsms_ml/preprocess.py` |
| NLP (TF-IDF, Sentiment, NER, Embeddings) | `ml/gsms_ml/text.py`, `ml/gsms_ml/lexicon.py`, `ml/gsms_ml/gazetteer.py` |
| Concept Learning | `ml/gsms_ml/concept_learning.py` |
| Classification | `ml/gsms_ml/supervised.py` |
| Regression | `ml/gsms_ml/regression.py` |
| Clustering | `ml/gsms_ml/clustering.py` |
| Ensembles | `ml/gsms_ml/ensemble.py` |
| Evaluation | `ml/gsms_ml/evaluation.py` |
| Global Status Engine | `ml/gsms_ml/status_engine.py` |
| Orchestration | `ml/run_all.py` |
| Browser Worker | `src/workers/ml-lab.worker.ts` |
| Worker Manager | `src/services/ml-lab-worker.ts` |
| Dashboard Panel | `src/components/MLLabPanel.ts` |
| Sample Data | `src/data/ml-sample-events.generated.ts` |
| Panel Registration | `src/config/panels.ts` |

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `cd ml && python3 run_all.py` | Run full Python ML experiment suite |
| `cd ml && python3 -m pytest tests/ -v` | Run Python tests |
| `npm run dev` | Start development dashboard |
| `npm run typecheck` | TypeScript type checking |
| `npm run lint` | Lint (Biome + boundary checks) |
| `npm run build` | Production build |
| `npm run test:data` | Run data tests (Node) |
| `npm run test:dom` | Run DOM tests (Vitest) |
| `npm run test:sidecar` | Run sidecar tests |

---

## Results Location

Python experiment results are written to `ml/results/`:

- `supervised.json` — Classification metrics for all models
- `regression.json` — Regression metrics
- `clustering.json` — Clustering results (K-Means, Hierarchical, DBSCAN, GMM)
- `ensembles.json` — Ensemble metrics + feature importance
- `evaluation.json` — CV, bootstrap, bias/variance, regularization
- `concept_learning.json` — Find-S and Candidate Elimination outputs
- `status_engine.json` — Per-country Global Status scores
- `REPORT.md` — Human-readable summary report

---

## References

- Course Syllabus: 241AI005 Machine Learning (B.Tech CSE 2024 Curriculum)
- Textbook: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron)
- GDELT Project: https://www.gdeltproject.org/
- USGS Earthquake Hazards: https://earthquake.usgs.gov/
- UCDP: https://ucdp.uu.se/
- AFINN Sentiment Lexicon: https://github.com/fnielsen/afinn