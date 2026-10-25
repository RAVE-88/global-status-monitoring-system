#!/usr/bin/env python3
"""GSMS ML laboratory orchestrator: every experiment, one deterministic run.

Loads the collected datasets from ``ml/data/`` (poll-waiting up to 10 minutes
for the fetch process to finish), then runs every experiment of the lab and
writes one JSON result per experiment to ``ml/results/`` plus a human-readable
``ml/results/REPORT.md``.

Pipeline
--------
1. load raw event rows -> ``preprocess.clean_rows`` (dedupe, impute, encode)
2. chronological train/validation/test split (no future leakage)
3. supervised classifiers (risk-band prediction): GaussianNB, k-NN,
   LogisticRegression, LinearSVC, RBF-SVC — holdout metrics + stratified CV
4. linear regression on the derived risk_score (MAE/MSE/RMSE/R²)
5. clustering: K-Means / Ward / DBSCAN / GMM on standardized numeric features,
   plus K-Means over LSA text embeddings (semantic event clustering)
6. ensembles: Random Forest, AdaBoost (SAMME), Bagging + feature importances
7. evaluation toolbox: bootstrap CI, bias/variance depth sweep, regularization
   C sweep
8. concept learning: Find-S and Candidate Elimination on the discretized
   concept table
9. status engine: per-country Global Status scores from real signals
10. semantic retrieval demo: LSA nearest neighbours for a sample headline

Determinism: python, numpy and model seeds are pinned (42); every transformer
here is fit-on-train. All labels in the data are DERIVED HEURISTICS from the
transparent keyword classifier — never ground truth; the report says so.

Usage:  python3.10 run_all.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from gsms_ml import clustering, concept_learning, ensemble, evaluation  # noqa: E402
from gsms_ml import regression, status_engine, text  # noqa: E402
from gsms_ml.preprocess import (  # noqa: E402
    Standardizer,
    clean_rows,
    chronological_split,
    load_raw_events,
)
from gsms_ml.supervised import MODEL_FACTORIES, evaluate_model  # noqa: E402
from gsms_ml.text import semantic_neighbours  # noqa: E402

DATA_DIR = ML_DIR / "data"
RESULTS_DIR = ML_DIR / "results"
REQUIRED_FILES = ("gsms_events.csv", "gsms_quakes.csv", "gsms_concept.csv")
WAIT_TIMEOUT_S = 600  # 10 minutes
POLL_S = 10

STUDENT_BLOCK = [
    "24B11CS489: Viswajeet",
    "24B11CS357: Priyamwda Rai",
    "24B11CS461: Satya Sri Hema",
    "24B11CS256: Uma Ram Charan",
]
PROJECT_TITLE = "GLOBAL STATUS MONITORING SYSTEM"


def log(message: str) -> None:
    """Progress line with an elapsed timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def wait_for_data() -> None:
    """Poll ml/data/ until all three CSVs exist (up to 10 minutes)."""
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if not missing:
        return
    log(f"waiting for dataset files (missing: {', '.join(missing)})...")
    while missing:
        if time.monotonic() > deadline:
            raise SystemExit(
                f"timed out after {WAIT_TIMEOUT_S}s waiting for {missing} in {DATA_DIR}; "
                "run `python3.10 fetch_dataset.py` first"
            )
        time.sleep(POLL_S)
        missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    log("dataset files found")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_write(name: str, payload: dict) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def run_supervised(splits, X_std) -> dict:
    """Holdout metrics on the chronological test split for every classifier."""
    train, validation, test = splits
    results = {}
    for name, factory in MODEL_FACTORIES.items():
        log(f"  supervised: {name}")
        result = evaluate_model(
            factory,
            X_std["train"], train.y_band,
            X_std["test"], test.y_band,
        )
        # 5-fold stratified CV on the development set (train + validation),
        # so the test split stays untouched by model selection
        dev_X = X_std["train"] + X_std["validation"]
        dev_y = train.y_band + validation.y_band
        result["cv"] = evaluation.cross_validate_accuracy(factory, dev_X, dev_y, k=5)
        results[name] = result
    return results


def run_ensembles(splits, X_std, feature_names) -> dict:
    train, validation, test = splits
    results = {}
    for name, factory in ensemble.ENSEMBLE_FACTORIES.items():
        log(f"  ensemble: {name}")
        results[name] = ensemble.evaluate_ensemble(
            factory,
            X_std["train"], train.y_band,
            X_std["test"], test.y_band,
            feature_names=feature_names,
        )
    return results


def run_concept_learning() -> dict:
    """Find-S + Candidate Elimination on the discretized concept table."""
    rows = read_csv_rows(DATA_DIR / "gsms_concept.csv")
    # source_domain is a high-cardinality identifier, not a concept attribute
    for row in rows:
        row.pop("source_domain", None)
    positives = sum(1 for row in rows if row.get("high_risk") == "1")
    log(f"  find-s over {len(rows)} concept rows ({positives} positive)")
    hypothesis = concept_learning.find_s(rows, "high_risk", "1")
    trace = concept_learning.find_s_trace(rows[:12], "high_risk", "1")
    log("  candidate elimination (full table)")
    elimination = concept_learning.candidate_elimination(rows, "high_risk", "1")
    # keep the JSON light: history of the first 25 steps only
    elimination["history"] = elimination["history"][:25]
    return {
        "n_examples": len(rows),
        "n_positive": positives,
        "label": "high_risk (risk_score >= 50)",
        "attributes": concept_learning.attributes_of(rows, "high_risk"),
        "find_s": {key: value for key, value in hypothesis.items()},
        "find_s_trace_first_12": trace,
        "candidate_elimination": elimination,
    }


def build_report(payload: dict) -> str:
    """Assemble REPORT.md from every computed experiment payload."""
    meta = payload["meta"]
    supervised_results = payload["supervised"]
    regression_result = payload["regression"]
    clustering_result = payload["clustering"]
    ensemble_results = payload["ensembles"]
    eval_results = payload["evaluation"]
    concept_result = payload["concept_learning"]
    status_result = payload["status_engine"]
    retrieval = payload["semantic_retrieval"]

    lines: list[str] = []
    lines.append(f"# {PROJECT_TITLE} — Machine Learning Laboratory Report")
    lines.append("")
    lines.append("```")
    lines.extend(STUDENT_BLOCK)
    lines.append("```")
    lines.append("")
    lines.append(
        f"Generated {meta['generated_utc']} from **{meta['n_raw_events']} raw "
        f"articles** ({meta['n_events']} after cleaning, {meta['removed_duplicates']} "
        f"near-duplicates removed, {meta['imputed_values']} numeric values imputed), "
        f"**{meta['n_quakes']} USGS earthquakes**, and **{meta['n_concept_rows']} "
        f"discretized concept rows**. Chronological split: "
        f"{meta['n_train']} train / {meta['n_validation']} validation / "
        f"{meta['n_test']} test (oldest→newest; the test set is strictly future "
        f"relative to training). Seeds pinned at 42."
    )
    lines.append("")

    lines.append("## 1. Data provenance")
    lines.append("")
    lines.append(
        "Real, keyless, public feeds — the same ones the WorldMonitor dashboard "
        "ingests: GDELT DOC API v2 artlist (21 topical and source-country queries, "
        "trailing window) and the USGS GeoJSON all-week earthquake feed. "
        "The concept table discretizes the same articles for symbolic learning."
    )
    lines.append("")

    lines.append("## 2. Supervised classification (target: derived risk band)")
    lines.append("")
    lines.append(
        "Labels are the keyword-classifier risk bands (low/medium/high/critical) "
        "computed from the headline text by transparent rules — **derived "
        "heuristics, not ground truth**."
    )
    lines.append("")
    rows = []
    for name, result in supervised_results.items():
        roc = f"{result['roc_auc']:.4f}" if result.get("roc_auc") is not None else "n/a"
        cv = result["cv"]
        cv_text = (
            f"{cv['mean']:.4f} ± {cv['std']:.4f}" if cv["mean"] is not None else "n/a"
        )
        rows.append([
            name,
            f"{result['accuracy']:.4f}",
            f"{result['precision_macro']:.4f}",
            f"{result['recall_macro']:.4f}",
            f"{result['f1_macro']:.4f}",
            roc,
            cv_text,
        ])
    lines.append(md_table(
        ["model", "accuracy", "precision(macro)", "recall(macro)", "f1(macro)",
         "roc-auc", "5-fold CV accuracy"],
        rows,
    ))
    lines.append("")
    best = max(supervised_results, key=lambda name: supervised_results[name]["accuracy"])
    lines.append(
        f"Holdout leader: **{best}** "
        f"(accuracy {supervised_results[best]['accuracy']:.4f})."
    )
    lines.append("")

    lines.append("## 3. Linear regression (target: derived risk_score 0-100)")
    lines.append("")
    lines.append(md_table(
        ["metric", "value"],
        [
            ["MAE", f"{regression_result['mae']:.3f}"],
            ["MSE", f"{regression_result['mse']:.3f}"],
            ["RMSE", f"{regression_result['rmse']:.3f}"],
            ["R² (test)", f"{regression_result['r2']:.4f}"],
            ["R² (train)", f"{regression_result['train_r2']:.4f}"],
            ["mean actual", f"{regression_result['mean_actual']:.2f}"],
        ],
    ))
    lines.append("")
    coef_rows = [
        [name, f"{value:+.4f}"]
        for name, value in sorted(
            regression_result["coefficients"].items(), key=lambda kv: -abs(kv[1])
        )
    ]
    lines.append("Standardized coefficients (largest magnitude first):")
    lines.append("")
    lines.append(md_table(["feature", "coefficient"], coef_rows))
    lines.append("")

    lines.append("## 4. Clustering")
    lines.append("")
    screen = clustering_result.get("outlier_screen", {})
    if screen:
        lines.append(
            f"Outlier screen: {screen.get('excluded_count', 0)} of "
            f"{clustering_result['n_samples']} rows excluded from "
            f"K-Means/Ward/GMM (single extreme artifacts; DBSCAN sees the "
            f"full matrix and labels them noise)."
        )
        lines.append("")
    kmeans_result = clustering_result["kmeans"]
    lines.append(
        f"- **K-Means**: best k={kmeans_result['best_k']}, "
        f"silhouette={kmeans_result['silhouette']}, sizes={kmeans_result['sizes']}."
    )
    agglomerative_result = clustering_result["agglomerative"]
    lines.append(
        f"- **Agglomerative (Ward)**: best k={agglomerative_result['best_k']}, "
        f"silhouette={agglomerative_result['silhouette']}, "
        f"sizes={agglomerative_result['sizes']}."
    )
    dbscan_result = clustering_result["dbscan"]
    lines.append(
        f"- **DBSCAN**: eps={dbscan_result['eps']} (k-distance knee), "
        f"min_samples={dbscan_result['min_samples']} → "
        f"{dbscan_result['n_clusters']} clusters, "
        f"{dbscan_result['noise_count']} noise points "
        f"({100.0 * dbscan_result['noise_count'] / max(clustering_result['n_samples'], 1):.1f}%), "
        f"silhouette={dbscan_result['silhouette']}."
    )
    gmm_result = clustering_result["gmm"]
    lines.append(
        f"- **GMM (EM)**: components={gmm_result['k']}, "
        f"silhouette={gmm_result['silhouette']}, sizes={gmm_result['sizes']}, "
        f"converged={gmm_result['converged']}."
    )
    sem = clustering_result.get("semantic_kmeans", {})
    if sem:
        lines.append(
            f"- **semantic event clustering** (K-Means over {sem['embedding_dim']}-dim "
            f"LSA embeddings of all headlines): best k={sem['kmeans']['best_k']}, "
            f"silhouette={sem['kmeans']['silhouette']}, sizes={sem['kmeans']['sizes']}."
        )
    evr = clustering_result["pca_2d"]["explained_variance_ratio"]
    lines.append(
        f"- PCA(2) of the numeric features explains "
        f"{100 * (evr[0] + evr[1]):.1f}% of variance "
        f"({100 * evr[0]:.1f}% + {100 * evr[1]:.1f}%)."
    )
    lines.append("")

    lines.append("## 5. Ensemble methods (target: derived risk band)")
    lines.append("")
    rows = []
    for name, result in ensemble_results.items():
        rows.append([
            name,
            f"{result['accuracy']:.4f}",
            f"{result['f1_macro']:.4f}",
            str(len(result.get("feature_importances", [])) or "n/a"),
        ])
    lines.append(md_table(
        ["ensemble", "accuracy", "f1(macro)", "#importance-entries"], rows))
    lines.append("")
    forest = ensemble_results.get("RandomForest", {})
    if forest.get("feature_importances"):
        lines.append("Random Forest feature importances (top 5):")
        lines.append("")
        lines.append(md_table(
            ["feature", "importance"],
            [
                [entry["feature"], f"{entry['importance']:.4f}"]
                for entry in forest["feature_importances"][:5]
            ],
        ))
        lines.append("")

    lines.append("## 6. Evaluation toolbox")
    lines.append("")
    ci = eval_results["bootstrap_accuracy"]
    lines.append(
        f"- **Bootstrap CI** (accuracy of {ci['model']} on the test split, "
        f"n_boot={ci['n_boot']}): point {ci['point']:.4f}, "
        f"95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]."
    )
    bv = eval_results["bias_variance"]
    lines.append("- **Bias/variance (tree depth sweep)**:")
    lines.append("")
    lines.append(md_table(
        ["depth", "train error", "test error"],
        [[str(row["depth"]), f"{row['train_error']:.4f}", f"{row['test_error']:.4f}"]
         for row in bv["table"]],
    ))
    lines.append("")
    reg = eval_results["regularization"]
    lines.append("- **Regularization (LogisticRegression C sweep)**:")
    lines.append("")
    lines.append(md_table(
        ["C", "train accuracy", "test accuracy", "gap"],
        [[str(row["C"]), f"{row['train_accuracy']:.4f}",
          f"{row['test_accuracy']:.4f}", f"{row['gap']:+.4f}"]
         for row in reg["table"]],
    ))
    lines.append("")

    lines.append("## 7. Concept learning (Find-S and Candidate Elimination)")
    lines.append("")
    lines.append(
        f"Over {concept_result['n_examples']} discretized event rows "
        f"({concept_result['n_positive']} positive for `high_risk`), attributes: "
        f"{', '.join(f'`{a}`' for a in concept_result['attributes'])}."
    )
    lines.append("")
    lines.append("**Find-S** hypothesis (most specific, '?' = any value):")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(concept_result["find_s"], indent=2))
    lines.append("```")
    lines.append("")
    ce = concept_result["candidate_elimination"]
    lines.append(
        f"**Candidate Elimination**: |S|={len(ce['specific'])}, |G|={len(ce['general'])}, "
        f"version-space boundary pairs={ce['version_space_size']}, "
        f"{ce['skipped_examples']} noisy examples skipped "
        f"({100.0 * ce['skipped_examples'] / max(concept_result['n_examples'], 1):.1f}% "
        f"of the table). First step of the S boundary: "
    )
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(ce["specific"][0] if ce["specific"] else {}, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 8. Global Status engine (academic derived score)")
    lines.append("")
    lines.append(
        "Per-country 0-100 status from real signals: weighted sum of severity(30) "
        "+ event_volume(20) + negativity(15) + corroboration(10) + seismic(15) + "
        f"news_velocity(10). Window: {status_result['window_hours']}h. "
        "**This is an academic derived score for this report — not the "
        "dashboard's production CII.**"
    )
    lines.append("")
    top = status_result["countries"][:12]
    lines.append(md_table(
        ["country", "score", "band", "events", "worst level", "avg sentiment",
         "quakes", "max mag", "velocity/h"],
        [
            [
                row["country"], str(row["score"]), row["band"],
                str(row["event_count"]), row["worst_level"],
                f"{row['avg_sentiment']:+.3f}", str(row["quake_count"]),
                f"{row['max_magnitude']:.1f}", f"{row['news_velocity_per_hour']:.2f}",
            ]
            for row in top
        ],
    ))
    lines.append("")
    if top:
        breakdown = top[0]
        comp_rows = [
            [name, f"{value:.3f}", f"{status_engine.COMPONENT_WEIGHTS[name]:.0f}"]
            for name, value in breakdown["components"].items()
        ]
        lines.append(
            f"Component breakdown for the top country ({breakdown['country']}, "
            f"score {breakdown['score']}):"
        )
        lines.append("")
        lines.append(md_table(["component", "value (0-1)", "weight"], comp_rows))
        lines.append("")

    lines.append("## 9. Semantic retrieval demo (LSA nearest neighbours)")
    lines.append("")
    lines.append(f"Query headline: *{retrieval['query']}*")
    lines.append("")
    lines.append(md_table(
        ["rank", "neighbour headline", "cosine"],
        [
            [str(rank + 1), neighbour["headline"], f"{neighbour['cosine']:.4f}"]
            for rank, neighbour in enumerate(retrieval["neighbours"])
        ],
    ))
    lines.append("")

    lines.append("## Limitations (honest)")
    lines.append("")
    lines.append(
        "1. **All labels are derived heuristics.** risk_score, risk_band and the "
        "concept table's high_risk flag come from the transparent keyword "
        "classifier; classifiers trained on them measure agreement with that "
        "labeling function, not with reality. High accuracy here is expected and "
        "is NOT evidence of real-world predictive power."
    )
    lines.append(
        "2. **Short window, topical queries.** The corpus covers a few days "
        "fetched through topical and source-country queries; it is not a "
        "representative sample of world news, and GDELT coverage varies by "
        "country and language (English only)."
    )
    lines.append(
        "3. **Feature leakage risk is structural, not accidental.** risk_score is "
        "a deterministic function of features that include the classifier's level "
        "weights; regression R² and band accuracy partly rediscover that formula. "
        "This is intentional for the lab (it makes the pipeline auditable) but "
        "should not be read as discovery."
    )
    lines.append(
        "4. **Country attribution is coarse.** Gazetteer NER misses countries not "
        "named in the headline; the quake bounding boxes are generous rectangles "
        "with known overlaps; unmatched events/quakes pool into 'Global'."
    )
    lines.append(
        "5. **Simple models, small feature set.** Seven numeric features + 400 TF-IDF "
        "terms; no deep learning, no multilingual text, no image data. Class "
        "imbalance (rare 'critical' band) is handled with macro averages and "
        "zero_division guards, but rare-class metrics remain noisy — hence the "
        "bootstrap interval rather than a single number."
    )
    lines.append(
        "6. **DBSCAN eps is a heuristic** (k-distance knee), silhouette has known "
        "biases on uneven cluster sizes, and the LSA subspace uses a fixed 20 "
        "iterations rather than full convergence diagnostics."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("```")
    lines.extend(STUDENT_BLOCK)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("GSMS ML laboratory run starting")
    wait_for_data()

    log("loading and cleaning data")
    raw_events = load_raw_events(DATA_DIR / "gsms_events.csv")
    quakes = status_engine.load_quakes(DATA_DIR / "gsms_quakes.csv")
    concept_rows_count = len(read_csv_rows(DATA_DIR / "gsms_concept.csv"))
    dataset = clean_rows(raw_events)
    splits = chronological_split(dataset)
    train, validation, test = splits
    log(
        f"  events: {len(raw_events)} raw -> {len(dataset.X)} clean "
        f"({dataset.removed_duplicate_count} near-duplicates removed, "
        f"{dataset.imputed_value_count} values imputed); "
        f"splits {len(train.X)}/{len(validation.X)}/{len(test.X)}; "
        f"{len(quakes)} quakes"
    )

    standardizer = Standardizer().fit(train.X)
    X_std = {
        "train": standardizer.transform(train.X),
        "validation": standardizer.transform(validation.X),
        "test": standardizer.transform(test.X),
    }

    log("supervised classification experiments")
    supervised_results = run_supervised(splits, X_std)

    log("linear regression experiment")
    y_score_train = [float(row["risk_score"]) for row in train.raw_rows]
    y_score_test = [float(row["risk_score"]) for row in test.raw_rows]
    regression_result = regression.run_regression(
        X_std["train"], y_score_train, X_std["test"], y_score_test,
        feature_names=dataset.feature_names,
    )

    log("clustering experiments (numeric + semantic)")
    all_X_std = standardizer.transform(dataset.X)
    clustering_result = clustering.cluster_numeric(all_X_std, dataset.feature_names)
    semantic_result = clustering.cluster_semantic(dataset.headlines)
    clustering_result["semantic_kmeans"] = {
        "embedding_dim": semantic_result["embedding_dim"],
        "kmeans": semantic_result["kmeans"],
    }

    log("ensemble experiments")
    ensemble_results = run_ensembles(splits, X_std, dataset.feature_names)

    log("evaluation toolbox (bootstrap, bias/variance, regularization)")
    dev_X = X_std["train"] + X_std["validation"]
    dev_y = train.y_band + validation.y_band
    bootstrap_model_name = "LogisticRegression"
    bootstrap_split_predictions = MODEL_FACTORIES[bootstrap_model_name]().fit(
        X_std["train"], train.y_band
    ).predict(X_std["test"])
    bootstrap_result = evaluation.bootstrap_metric(
        test.y_band, list(bootstrap_split_predictions),
        accuracy_score,
        n_boot=200,
    )
    bootstrap_result["model"] = bootstrap_model_name
    eval_results = {
        "bootstrap_accuracy": bootstrap_result,
        "bias_variance": evaluation.bias_variance_demo(
            dev_X, dev_y, X_std["test"], test.y_band
        ),
        "regularization": evaluation.regularization_demo(
            dev_X, dev_y, X_std["test"], test.y_band
        ),
        "cross_validation_all_models": {
            name: supervised_results[name]["cv"] for name in supervised_results
        },
    }

    log("concept learning (Find-S + Candidate Elimination)")
    concept_result = run_concept_learning()

    log("status engine (per-country Global Status)")
    status_result = status_engine.global_status(raw_events, quakes)

    log("semantic retrieval demo")
    # pick the test headline richest in fitted vocabulary, so the demo shows
    # meaningful similarities rather than an out-of-vocabulary query
    from gsms_ml.text import preprocess_text

    if test.headlines:
        embedder = text.LsaEmbedder().fit(dataset.headlines)
        vocab = embedder.vectorizer.vocabulary

        def vocab_hits(headline: str) -> int:
            return sum(1 for token in preprocess_text(headline) if token in vocab)

        query = max(test.headlines, key=vocab_hits)
    else:
        embedder = text.LsaEmbedder().fit(dataset.headlines)
        query = ""
    neighbours_raw = semantic_neighbours(query, dataset.headlines, embedder, k=5)
    retrieval_result = {
        "query": query,
        "embedding_dim": embedder.embedding_dim,
        "neighbours": [
            {"index": index, "cosine": round(score, 4),
             "headline": dataset.headlines[index]}
            for index, score in neighbours_raw
        ],
    }

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_title": PROJECT_TITLE,
        "students": STUDENT_BLOCK,
        "n_raw_events": len(raw_events),
        "n_events": len(dataset.X),
        "removed_duplicates": dataset.removed_duplicate_count,
        "imputed_values": dataset.imputed_value_count,
        "n_quakes": len(quakes),
        "n_concept_rows": concept_rows_count,
        "n_train": len(train.X),
        "n_validation": len(validation.X),
        "n_test": len(test.X),
        "feature_names": dataset.feature_names,
    }

    payload = {
        "meta": meta,
        "supervised": supervised_results,
        "regression": regression_result,
        "clustering": clustering_result,
        "ensembles": ensemble_results,
        "evaluation": eval_results,
        "concept_learning": concept_result,
        "status_engine": status_result,
        "semantic_retrieval": retrieval_result,
    }

    log("writing results")
    json_write("meta.json", meta)
    json_write("supervised.json", supervised_results)
    json_write("regression.json", regression_result)
    json_write("clustering.json", clustering_result)
    json_write("ensembles.json", ensemble_results)
    json_write("evaluation.json", eval_results)
    json_write("concept_learning.json", concept_result)
    json_write("status_engine.json", status_result)
    json_write("semantic_retrieval.json", retrieval_result)

    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text(build_report(payload), encoding="utf-8")
    log(f"wrote {report_path} and 9 JSON files to {RESULTS_DIR}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
