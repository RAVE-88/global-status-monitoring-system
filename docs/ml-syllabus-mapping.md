# Machine Learning Syllabus Mapping — Global Status Monitoring System

**Course:** 241AI005 Machine Learning (B.Tech CSE 5th Semester, 2024 Curriculum)  
**Project:** Global Status Monitoring System  
**Team:** 24B11CS489 (Viswajeet), 24B11CS357 (Priyamwda Rai), 24B11CS461 (Satya Sri Hema), 24B11CS256 (Uma Ram Charan)

---

## Mapping Table

| Syllabus Topic | Course Outcome | Implemented Feature | Actual File/Module | UI / Demo |
|----------------|----------------|---------------------|-------------------|-----------|
| **UNIT I: Introduction** | | | | |
| Machine Learning, Types of ML Systems, Main Challenges | CO1 | Project overview, problem framing | `docs/ml-project.md` | ML Lab → Pipeline Overview |
| Statistical Learning: Supervised/Unsupervised, Learning Problem, Feasibility | CO1 | Preprocessing pipeline, train/test methodology | `ml/gsms_ml/preprocess.py` | ML Lab → Dataset tab |
| Practice: Jupyter/Spider notebook, basic commands | CO1 | Python ML lab with venv, requirements.txt | `ml/requirements.txt`, `ml/run_all.py` | Run `python3 run_all.py` |
| Practice: Loading dataset, preprocessing, analysis | CO1 | GDELT/USGS fetch, CSV snapshot, dedupe, impute, standardize | `ml/fetch_dataset.py`, `ml/gsms_ml/preprocess.py` | ML Lab → Dataset tab |
| **UNIT II: Supervised Learning — Basic Methods** | | | | |
| Concept Learning: General-to-Specific, Find-S, Candidate Elimination, Version Space, Inductive Bias | CO3 | Find-S algorithm, Candidate Elimination on structured event data | `ml/gsms_ml/concept_learning.py` | ML Lab → Concept Learning tab |
| Bayesian Learning: Probability, MLE/MAP, Gaussian Naive Bayes, Bayesian Networks | CO3 | Gaussian Naive Bayes classifier | `ml/gsms_ml/supervised.py` (Python), `src/workers/ml-lab.worker.ts` (TS) | ML Lab → Classification tab |
| Instance-based Learning: k-NN, Voronoi, Distance-weighted k-NN, Distance Metrics, Curse of Dimensionality | CO3 | k-NN classifier (k=5, Euclidean/cosine) | `ml/gsms_ml/supervised.py`, `src/workers/ml-lab.worker.ts` | ML Lab → Classification tab |
| Practice: Implement Find-S from CSV | CO3 | `concept_learning.py::FindS` reads `gsms_concept.csv` | `ml/gsms_ml/concept_learning.py` | ML Lab → Concept Learning tab |
| Practice: Candidate Elimination from CSV | CO3 | `concept_learning.py::CandidateElimination` reads `gsms_concept.csv` | `ml/gsms_ml/concept_learning.py` | ML Lab → Concept Learning tab |
| Practice: Binary Classifier real-world problems | CO3 | High-risk vs non-high-risk binary classification | `ml/gsms_ml/supervised.py` | ML Lab → Classification tab |
| Practice: k-NN on Iris dataset | CO3 | k-NN on project's event dataset (not Iris — uses real data) | `ml/gsms_ml/supervised.py` | ML Lab → Classification tab |
| **UNIT III: Linear Models and Regression / SVM** | | | | |
| Linear Classification, Linear Regression, Non-linear Transformation, Logistic Regression | CO2 | Logistic Regression (SGD), Linear Regression (Normal Equations) | `ml/gsms_ml/supervised.py`, `ml/gsms_ml/regression.py` | ML Lab → Classification / Regression tabs |
| SVM: Decision Boundary, Support Vectors, Primal-Dual, Soft Margin, Non-linear Boundary, Kernel Functions, RBF | CO2 | Linear SVM (Pegasos), RBF SVM (kernelized, subsampled) | `ml/gsms_ml/supervised.py`, `src/workers/ml-lab.worker.ts` | ML Lab → Classification tab |
| Practice: Linear Regression real-world | CO2 | Risk score regression on derived continuous target | `ml/gsms_ml/regression.py` | ML Lab → Regression tab |
| Practice: Logistic Regression real-world | CO2 | Risk band (4-class) and high-risk (binary) classification | `ml/gsms_ml/supervised.py` | ML Lab → Classification tab |
| Practice: SVM on Patient/Heart dataset | CO2 | SVM on project's global event dataset (risk classification) | `ml/gsms_ml/supervised.py` | ML Lab → Classification tab |
| **UNIT IV: Evaluation / Unsupervised Learning** | | | | |
| Classifier Evaluation: Accuracy, Precision, Recall, F-Measure, ROC, Sampling, Bootstrapping, Cross-validation | CO4 | All metrics + 5-fold chronological CV + 1000-sample bootstrap CI | `ml/gsms_ml/evaluation.py` | ML Lab → Evaluation tab |
| Clustering: Partitional, Hierarchical, Cluster Types, Attributes, Hierarchical/Density-based, Inter/Intra Similarity, Cohesion/Separation, MST, DBSCAN | CO4 | K-Means (k-means++), Agglomerative (Ward), DBSCAN, GMM/EM | `ml/gsms_ml/clustering.py` | ML Lab → Clustering tab |
| Practice: Bias, Variance, Remove duplicates, Cross Validation | CO4 | Bias-variance sweep (tree depth), dedupe in preprocess, 5-fold CV | `ml/gsms_ml/evaluation.py`, `ml/gsms_ml/preprocess.py` | ML Lab → Evaluation tab |
| Practice: One-hot Encoding | CO4 | Categorical encoding in preprocessing pipeline | `ml/gsms_ml/preprocess.py` | ML Lab → Dataset tab |
| Practice: Categorical Encoding | CO4 | Ordinal (risk bands) + one-hot (categories) | `ml/gsms_ml/preprocess.py` | ML Lab → Dataset tab |
| **UNIT V: Ensemble Learning** | | | | |
| Ensemble Learning: Bagging, Boosting, AdaBoost, Random Forest | CO5 | Random Forest (200 trees), AdaBoost (100 stumps), Bagging (stumps) | `ml/gsms_ml/ensemble.py` | ML Lab → Ensembles tab |
| Computational Learning Theory: Error/Noise, Train vs Test, Generalization, PAC, VC Dimension, Overfitting, Regularization, Validation | CO5 | Bias-variance decomposition, regularization path (λ sweep), overfitting demo | `ml/gsms_ml/evaluation.py` | ML Lab → Evaluation tab |
| Practice: Random Forest classifier demo | CO5 | RF on event classification with feature importance | `ml/gsms_ml/ensemble.py` | ML Lab → Ensembles tab |
| Practice: Sentiment analysis with Bagging/Boosting on Twitter data | CO5 | AdaBoost + Bagging on project's event data (not Twitter — uses real global events) | `ml/gsms_ml/ensemble.py` | ML Lab → Ensembles tab |
| Additional: Naive Bayesian Classifier on documents | CO5 | Gaussian NB on event features (text-derived) | `ml/gsms_ml/supervised.py` | ML Lab → Classification tab |
| Additional: EM Algorithm on Heart Disease vs k-Means comparison | CO5 | GMM/EM vs K-Means on event embeddings (not Heart Disease — uses project data) | `ml/gsms_ml/clustering.py` | ML Lab → Clustering tab |

---

## Detailed File-to-Syllabus Traceability

### Python Implementation (`ml/gsms_ml/`)

| File | Syllabus Units | Key Functions/Classes |
|------|----------------|----------------------|
| `preprocess.py` | Unit I, II, IV | `deduplicate_events`, `impute_missing`, `standardize_fit_transform`, `chronological_split`, `encode_categorical` |
| `text.py` | Unit I, II | `TfidfVectorizer`, `TruncatedSVD`, `cosine_similarity`, `sentiment_score`, `extract_entities` |
| `lexicon.py` | Unit I, II | `AFINN_LEXICON` (committed), `sentiment_score` |
| `gazetteer.py` | Unit I, II | `COUNTRY_GAZETTEER`, `CITY_GAZETTEER`, `ORG_GAZETTEER`, `PERSON_GAZETTEER` |
| `concept_learning.py` | Unit II | `FindS`, `CandidateElimination` (both read `gsms_concept.csv`) |
| `supervised.py` | Unit II, III | `train_gaussian_nb`, `train_knn`, `train_logistic_regression`, `train_linear_svc`, `train_rbf_svc` |
| `regression.py` | Unit III | `train_linear_regression` (Normal Equations), `evaluate_regression` |
| `clustering.py` | Unit IV | `train_kmeans`, `train_hierarchical`, `train_dbscan`, `train_gmm`, `pca_jacobi` |
| `ensemble.py` | Unit V | `train_random_forest`, `train_adaboost`, `train_bagging`, `feature_importance` |
| `evaluation.py` | Unit IV, V | `compute_metrics`, `cross_val_chronological`, `bootstrap_ci`, `bias_variance_sweep`, `regularization_path` |
| `status_engine.py` | Course Project | `compute_global_status` (per-country 0-100 score) |

### TypeScript Browser Implementation (`src/`)

| File | Syllabus Units | Purpose |
|------|----------------|---------|
| `workers/ml-lab.worker.ts` | All Units | Full ML suite in Web Worker (zero deps, deterministic PRNG) |
| `services/ml-lab-worker.ts` | All Units | Typed worker manager (singleton, ready handshake, timeouts) |
| `components/MLLabPanel.ts` | All Units | Dashboard panel with 8 tabs: Overview, Dataset, Classification, Regression, Clustering, Ensembles, Evaluation, Concept Learning |
| `data/ml-sample-events.generated.ts` | Unit I | Embedded sample events for offline/first-paint demo |
| `config/panels.ts` | — | Panel registration (`ml-lab` in FULL_PANELS, intelligence category) |

---

## UI Demonstration Points

| Demo Scenario | Panel Tab | Syllabus Connection |
|---------------|-----------|---------------------|
| Live news → preprocessing → features → train/test | Dataset | Unit I: Preprocessing, Unit II: Train/test split |
| Headline → sentiment → entities → category → risk level → cluster | Pipeline Overview | Unit I/II: NLP pipeline, classification, clustering |
| Model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC) | Classification | Unit IV: Evaluation metrics |
| Confusion matrices per model | Classification | Unit IV: Confusion matrix |
| 5-fold CV mean/std per model | Classification | Unit IV: Cross-validation |
| Bootstrap 95% CI per model | Classification | Unit IV: Bootstrapping |
| Predicted vs Actual scatter (Regression) | Regression | Unit III: Linear Regression evaluation |
| PCA 2D projection with cluster assignments | Clustering | Unit IV: K-Means, Hierarchical, DBSCAN, visualization |
| Silhouette scores, inertia, noise points | Clustering | Unit IV: Cluster quality metrics |
| Random Forest feature importance | Ensembles | Unit V: Ensemble feature importance |
| AdaBoost/Bagging metrics vs base learners | Ensembles | Unit V: Boosting, Bagging |
| Bias-variance tradeoff curve (depth sweep) | Evaluation | Unit IV/V: Bias-variance, overfitting |
| Regularization path (λ vs train/test accuracy) | Evaluation | Unit V: Regularization, validation |
| Find-S hypothesis generalization steps | Concept Learning | Unit II: Find-S algorithm |
| Candidate Elimination S/G boundaries + history | Concept Learning | Unit II: Candidate Elimination |

---

## Running the Demos

### Python (Offline, Reproducible)
```bash
cd worldmonitor/ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 run_all.py          # Generates all results + REPORT.md
python3 -m pytest tests/ -v  # Runs test suite
```

### Browser Dashboard (Live + Offline Fallback)
```bash
cd worldmonitor
npm run dev                  # http://localhost:5173/dashboard
# Open "ML Lab" panel in Intelligence category
# Toggle "Use sample data" for offline demo
# Click "Run Experiments" for live analysis
```

### Verification Commands
```bash
cd worldmonitor
npm run typecheck            # TypeScript compilation check
npm run lint                 # Biome + boundary checks
npm run build                # Production build
npm run test:data            # Node.js data tests
npm run test:dom             # DOM tests (Vitest)
npm run test:sidecar         # Sidecar + API tests
```

---

## Notes on Adaptations

1. **Iris Dataset → Project Event Data**: The syllabus suggests k-NN on Iris. We use the project's own global event dataset with derived risk bands for authentic real-world demonstration.

2. **Heart Disease / Patient Dataset → Project Event Data**: SVM and EM/GMM practices reference medical datasets. We apply these algorithms to the project's geopolitical event embeddings for domain-relevant demonstration.

3. **Twitter Sentiment → GDELT News Sentiment**: The syllabus suggests Twitter sentiment with bagging/boosting. We use GDELT global news with AFINN sentiment — same technique, domain-appropriate data.

4. **Ground Truth Labels**: All classification targets are heuristic derivations (threat keyword classifier + corroboration). This is explicitly documented and surfaced in the UI as "Derived heuristic labels — not ground truth."

5. **RBF SVM / GMM in Browser**: Full RBF SVM (O(n²) kernel matrix) and EM/GMM are computationally heavy for browser. Python uses sklearn; TypeScript worker uses subsampled RBF kernel and omits GMM (documented in code).

6. **Chronological Splitting**: All train/test splits and CV folds are chronological (by publication date) to prevent future-data leakage — critical for time-series event data.

---

## Student Details (for Project Report)

```
24B11CS489: Viswajeet
24B11CS357: Priyamwda Rai
24B11CS461: Satya Sri Hema
24B11CS256: Uma Ram Charan

Project Title: GLOBAL STATUS MONITORING SYSTEM
Course: 241AI005 Machine Learning
Semester: 5th Semester B.Tech CSE
```