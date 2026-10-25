"""gsms_ml — Machine Learning laboratory for the Global Status Monitoring System.

Classical-ML experiments (scikit-learn) over real global news/event data collected
from the same keyless sources the WorldMonitor dashboard ingests (GDELT DOC API,
USGS earthquake feeds). All labels are derived heuristics, never ground truth;
see docs/ml-project.md for the full methodology.

Package layout
--------------
lexicon            sentiment word scores + keyword → event-category rules
gazetteer          country-name → ISO-2 recognition for headlines
classifier         Python port of shared/threat-keyword-classifier.ts (same labels
                   the dashboard shows)
text               text preprocessing, TF-IDF, LSA embeddings, cosine similarity
preprocess         dedupe, missing-value handling, standardization, one-hot,
                   chronological splitting
concept_learning   Find-S and Candidate Elimination (syllabus UNIT-II demos)
supervised         Gaussian NB, k-NN, Logistic Regression, linear + RBF SVM
regression         Linear Regression with MAE/MSE/RMSE/R²
clustering         K-Means, hierarchical, DBSCAN, EM/GMM + PCA projection
ensemble           Random Forest, AdaBoost, Bagging + feature importance
evaluation         metric suite, 5-fold CV, bootstrap CIs, bias/variance and
                   overfitting/regularization demonstrations
status_engine      Global Status score (0–100) per country from real signals
"""

__version__ = "1.0.0"
