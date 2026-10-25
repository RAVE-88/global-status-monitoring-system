# GSMS Machine Learning Laboratory

The Python ML laboratory for the college project **GLOBAL STATUS MONITORING
SYSTEM** (WorldMonitor). It runs the full classical-ML syllabus end to end —
preprocessing, TF-IDF/LSA text representation, supervised classification,
regression, clustering, ensembles, concept learning (Find-S / Candidate
Elimination), evaluation methodology (cross-validation, bootstrap confidence
intervals, bias/variance and regularization demonstrations) — over **real
data** collected from the same keyless public feeds the dashboard itself
ingests, and derives a transparent per-country **Global Status score** from
real signals.

**Everything here is coursework.** All labels (risk_score, risk_band,
high_risk) are derived heuristics from a transparent keyword classifier —
never ground truth — and the Global Status score is an academic derived
score, separate from the dashboard's production CII.

## Run it

With a virtual environment (recommended):

```bash
cd ml
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python fetch_dataset.py   # ~3-6 min, GDELT + USGS, no API keys
.venv/bin/python run_all.py         # waits for data if a fetch is running
```

No-venv shortcut on a machine that already has the pinned versions for
`python3.10`:

```bash
cd ml
python3.10 fetch_dataset.py
python3.10 run_all.py
```

Both commands print progress and exit 0 on success. Results land in
`ml/results/`: one JSON file per experiment plus `REPORT.md`, a
human-readable summary with every metric table.

Tests (stdlib `unittest`, no pytest needed):

```bash
cd <repo root>
python3.10 -m unittest discover -s ml/tests -t . -v
```

## Dataset provenance

| File | Source | Notes |
|---|---|---|
| `data/gsms_events.csv` | GDELT DOC API v2 artlist | 21 topical + source-country queries, English, trailing window; one row per unique article URL; labels computed from headline text |
| `data/gsms_quakes.csv` | USGS GeoJSON `all_week` feed | global seismic events for the trailing week |
| `data/gsms_concept.csv` | derived from `gsms_events.csv` | tidy categorical (discretized) table for Find-S / Candidate Elimination |

Both remote feeds are public, keyless, and identical to what the dashboard's
edge APIs fetch, so the lab operates on the same real world the app shows.
Re-running `fetch_dataset.py` refreshes the snapshot; `run_all.py` polls for
the files for up to 10 minutes so the two can be launched back to back.

## File map

```
ml/
├── fetch_dataset.py          builds the three CSVs in data/ (GDELT + USGS)
├── run_all.py                orchestrates every experiment, writes results/
├── requirements.txt          pinned tested versions (Python 3.10+)
├── README.md                 this file
├── data/                     collected datasets (built by fetch_dataset.py)
├── results/                  per-experiment JSONs + REPORT.md (built by run_all.py)
├── gsms_ml/
│   ├── __init__.py           package overview
│   ├── lexicon.py            committed AFINN-style sentiment lexicon
│   ├── gazetteer.py          country-name NER + region mapping
│   ├── classifier.py         Python port of the dashboard's threat classifier
│   ├── preprocess.py         dedupe, imputation, one-hot/numeric assembly,
│   │                         z-score Standardizer (fit-on-train), chronological split
│   ├── text.py               tokenizer, TF-IDF, cosine, LSA embeddings
│   │                         (subspace iteration, deterministic), semantic neighbours
│   ├── concept_learning.py   Find-S (+ trace) and Candidate Elimination
│   │                         with noise-tolerant rollback
│   ├── supervised.py         GaussianNB, k-NN, LogisticRegression, linear
│   │                         and RBF SVM + evaluate_model metric suite
│   ├── regression.py         LinearRegression on risk_score: MAE/MSE/RMSE/R²
│   ├── clustering.py         K-Means (k sweep), Ward, DBSCAN (knee eps),
│   │                         GMM/EM, PCA-2D, semantic (LSA) clustering
│   ├── ensemble.py           RandomForest, AdaBoost (SAMME), Bagging +
│   │                         feature importances
│   ├── evaluation.py         stratified CV, bootstrap CIs, bias/variance
│   │                         depth sweep, regularization C sweep
│   └── status_engine.py      per-country Global Status 0-100 from real
│                             signals (transparent weights; academic score)
└── tests/                    stdlib unittest suite
    ├── test_preprocess.py
    ├── test_concept_learning.py
    ├── test_evaluation.py
    └── test_text.py
```

## What the experiments show

- **Supervised**: five classifiers predict the derived risk band on a
  chronological test set (strictly future relative to training), with
  macro-averaged metrics, confusion matrices, ROC-AUC (binary cases) and
  5-fold stratified CV.
- **Regression**: ordinary least squares predicts the 0-100 risk_score;
  MAE/MSE/RMSE/R² plus a 50-point predicted-vs-actual sample for plotting.
- **Clustering**: K-Means/Ward/DBSCAN/GMM on standardized numeric features
  with silhouette-based k selection, a deterministic k-distance-knee eps for
  DBSCAN, a shared 2-component PCA projection — and a separate "semantic
  event clustering" run of K-Means over LSA headline embeddings.
- **Ensembles**: Random Forest (with feature importances mapped to feature
  names), AdaBoost (SAMME, per scikit-learn 1.5 deprecation), bagged stumps.
- **Concept learning**: Find-S converges on the discretized event table;
  Candidate Elimination maintains S/G boundaries with noisy rows rolled back
  and recorded instead of crashing.
- **Status engine**: per-country 0-100 Global Status with a full component
  breakdown (severity, event volume, negativity, corroboration, seismic,
  news velocity), banded Low/Medium/High/Critical.

## Limitations (honest)

1. **Labels are derived heuristics**, not ground truth. Model accuracy
   measures agreement with the keyword classifier's own labeling function.
2. **Small, topical, English-only window** (a few days of GDELT artlists);
   coverage varies by country and source.
3. **Structural leakage**: risk_score is a deterministic function of the
   features, so regression/band accuracy partly rediscovers that formula.
   Intentional for auditability; not discovery.
4. **Coarse geography**: gazetteer NER only sees countries named in
   headlines; quake attribution uses generous bounding boxes with known
   overlaps; unmatched items pool into "Global".
5. **Simple models and heuristics**: DBSCAN eps is a knee heuristic,
   silhouette has known biases, LSA uses a fixed 20 subspace iterations, and
   rare classes (critical band) make some metrics noisy — hence bootstrap
   intervals in the report.

---

```
24B11CS489: Viswajeet
24B11CS357: Priyamwda Rai
24B11CS461: Satya Sri Hema
24B11CS256: Uma Ram Charan
```

Project title: **GLOBAL STATUS MONITORING SYSTEM**.
