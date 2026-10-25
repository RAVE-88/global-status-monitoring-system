# Global Status Monitoring System (GSMS)

**An ML-powered global intelligence platform** — monitors world events, news, geopolitical conditions, disasters, and infrastructure signals in real time, and applies a full machine-learning stack to classify events, predict impact, and support global risk assessment.

[![Live Demo](https://img.shields.io/badge/%F0%9F%8C%90_Live_Demo-rave--88.github.io-2ea44f)](https://rave-88.github.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](ml/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)](ml/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

> **▶ Live interface:** <https://rave-88.github.io/> — open the ☰ menu → **Intelligence → ML Lab** to explore all ML models running on live world-event data in your browser.

---

## Abstract

> The Global Status Monitoring System (GSMS) is an ML-powered intelligence platform designed to monitor and analyse global events, news, geopolitical conditions, disasters, and infrastructure-related signals. The Machine Learning component of the system applies multiple machine learning techniques to event and news data to identify patterns, classify events, predict impact, and support global risk assessment. The system uses data collected from sources such as news feeds and global event datasets, followed by preprocessing operations including deduplication, missing-value handling, standardization, text processing, TF-IDF feature extraction, dimensionality reduction using SVD, and feature engineering. Various supervised learning algorithms, including Logistic Regression, Linear and RBF Support Vector Machines, Gaussian Naive Bayes, and k-Nearest Neighbors, are implemented for classification tasks. Linear Regression is used for numerical prediction, while K-Means++, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models are used to discover patterns in event data. Ensemble techniques such as Random Forest, AdaBoost, and Bagging are also implemented to improve prediction robustness. The system evaluates models using metrics such as Accuracy, Precision, Recall, F1-Score, MAE, MSE, RMSE, and R², along with chronological cross-validation and bootstrap confidence intervals. The project demonstrates how machine learning can transform large-scale global event data into meaningful patterns, predictions, and risk indicators, providing a foundation for explainable and data-driven situational awareness.
>
> **Keywords:** Machine Learning, Global Event Monitoring, Geopolitical Risk, Classification, Regression, Clustering, Ensemble Learning, Natural Language Processing, Model Evaluation, Predictive Analysis

## Team

| Roll No. | Name |
|----------|------|
| 24B11CS489 | Viswajeet |
| 24B11CS357 | Priyamwda Rai |
| 24B11CS461 | Satya Sri Hema |
| 24B11CS256 | Uma Ram Charan |

**Course:** 241AI005 — Machine Learning (B.Tech CSE, 5th Semester)

---

## The Machine Learning System

Two complementary implementations of the same pipeline:

1. **Python ML Lab** ([`ml/`](ml/)) — reproducible offline analysis with scikit-learn. Ingests real data (GDELT news articles + USGS earthquakes), runs the full preprocessing → modeling → evaluation pipeline, and emits `ml/results/REPORT.md` plus nine JSON result files. Seeded (42), chronological splits, fully deterministic.
2. **Browser ML Lab** ([`src/components/MLLabPanel.ts`](src/components/MLLabPanel.ts)) — the same algorithms re-implemented as zero-dependency TypeScript running in Web Workers, so every model executes **live in the browser** on the real-time event stream, with no backend and no API keys.

### Pipeline

```
GDELT news (artlist API) ─┐
                          ├─► deduplication ─► missing-value handling ─► standardization
USGS earthquakes (4.5wk) ─┘        │
                                   ├─► text processing: TF-IDF ─► SVD (dim. reduction)
                                   ├─► feature engineering: sentiment, velocity, corroboration…
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Classification   Logistic Regression · Linear SVM ·     │
        │                   RBF SVM · Gaussian NB · k-NN           │
        │  Regression       Linear Regression (Normal Equations)   │
        │  Clustering       K-Means++ · Ward · DBSCAN · GMM (EM)   │
        │  Ensembles        Random Forest · AdaBoost · Bagging     │
        │  Concept Learning Find-S · Candidate Elimination         │
        └──────────────────────────────────────────────────────────┘
                                   ▼
   Evaluation: Accuracy · Precision · Recall · F1 · ROC-AUC · MAE/MSE/RMSE/R²
   5-fold chronological CV · 1000-sample bootstrap CIs · bias–variance sweep
                                   ▼
        Global Status Engine — explainable 0–100 risk score per country
```

### Results (from the latest full run — [`ml/results/REPORT.md`](ml/results/REPORT.md))

Built from **2,919 raw articles** (2,409 after cleaning; 510 near-duplicates removed) and **1,875 earthquakes**, with a strictly future test split:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | 5-fold CV accuracy |
|---|---|---|---|---|---|
| **RBF SVM** (holdout leader) | **0.9890** | 0.3297 | 0.3333 | 0.3315 | 0.9585 ± 0.0000 |
| Linear SVM | 0.9862 | 0.3324 | 0.3324 | 0.3310 | 0.9580 ± 0.0010 |
| k-NN (k=5, distance) | 0.9834 | 0.4972 | 0.4417 | 0.4639 | 0.9541 ± 0.0029 |
| Logistic Regression | 0.9779 | 0.3296 | 0.3296 | 0.3296 | 0.9565 ± 0.0024 |
| Gaussian Naive Bayes | 0.1851 | 0.3345 | 0.3948 | 0.1060 | 0.2991 ± 0.1350 |

Clustering: K-Means++ (best k=3, silhouette 0.281) · Ward (silhouette 0.827) · DBSCAN (eps via k-distance knee, 4 clusters, 0.4% noise, silhouette 0.690) · GMM (3 components, silhouette 0.459). Linear regression predicts the derived risk score (MAE 5.08, RMSE 8.20) with interpretable standardized coefficients (sentiment −3.00, word count +1.64, corroboration +1.23).

### Syllabus coverage

Every unit of the 241AI005 syllabus is implemented and traceable — see the [syllabus mapping guide](docs/ml-syllabus-mapping.md) and the [full technical documentation](docs/ml-project.md).

| Unit | Topics | Implementation |
|---|---|---|
| I — Introduction | ML concepts, preprocessing | `ml/gsms_ml/preprocess.py`, `ml/gsms_ml/text.py` |
| II — Supervised Learning | Find-S, Candidate Elimination, Naive Bayes, k-NN | `ml/gsms_ml/supervised.py`, `ml/gsms_ml/concept_learning.py` |
| III — Linear Models | Linear/Logistic Regression, SVM (linear + RBF) | `ml/gsms_ml/regression.py`, `ml/gsms_ml/supervised.py` |
| IV — Evaluation & Unsupervised | Metrics, CV, bootstrapping, clustering | `ml/gsms_ml/evaluation.py`, `ml/gsms_ml/clustering.py` |
| V — Ensemble Learning | Bagging, Boosting, Random Forest | `ml/gsms_ml/ensemble.py` |

---

## The Interface

A real-time global intelligence dashboard: 3D globe + WebGL map, curated news feeds, cross-stream correlation (military, economic, disaster, escalation), a Country Instability Index for 31 countries, a finance radar, and the interactive **ML Lab** (8 tabs — Overview, Dataset, Classification, Regression, Clustering, Ensembles, Evaluation, Concept Learning) running all models on live data.

Tech: vanilla TypeScript + Vite, globe.gl/Three.js, deck.gl/MapLibre GL, Transformers.js (MiniLM embeddings, DistilBERT sentiment, BERT NER) in workers, Tauri 2 desktop shell.

---

## Running It

### Interface (browser dashboard + live ML Lab)

```bash
npm install
npm run dev
```

Open http://localhost:3006/ (binds 0.0.0.0, auto-selects a free port), then ☰ menu → **Intelligence → ML Lab**. Runs with no environment variables; keyless data sources (GDELT, USGS) work out of the box.

### Python ML Lab (reproducible pipeline)

```bash
cd ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_all.py          # full pipeline → ml/results/REPORT.md + 9 JSONs
python -m pytest tests/ -v # test suite
```

## Repository Structure

```
├── ml/                     Python ML Lab (pipeline, tests, data, results)
│   ├── gsms_ml/            preprocessing · supervised · regression · clustering
│   │                       ensemble · concept_learning · evaluation · status_engine
│   ├── data/               GDELT + USGS datasets (CSV)
│   └── results/            REPORT.md + JSON metrics from the latest run
├── src/                    Dashboard app (panels, map, ML Lab UI, workers)
├── shared/                 Isomorphic ML cores (clustering, threat classifier)
├── docs/                   ml-project.md · ml-syllabus-mapping.md · APP-README.md
├── api/                    Serverless endpoints (self-hosted deployments)
└── dist/                   Production build (not required for dev)
```

## Data Sources

Keyless public feeds, ingested live: **GDELT DOC API v2** (global news), **USGS** (earthquake GeoJSON), plus optional keyed sources (see `.env.example`). Full catalog in [docs/data-sources](docs/).

## License & Attribution

Source licensed **AGPL-3.0-only** (see [LICENSE](LICENSE)). The dashboard platform is built on the open-source [WorldMonitor](https://github.com/koala73/worldmonitor) codebase by Elie Habib; the ML system (Python lab, browser ML Lab, status engine) was developed by this team on top of it. Copyright (C) 2024-2026 Elie Habib for the underlying platform code.
