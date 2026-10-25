# Global Status Monitoring System

[简体中文](README.zh-CN.md) | [日本語](README.ja-JP.md) | [Русский](README.ru.md)

**Real-time global intelligence dashboard** — AI-powered news aggregation, geopolitical monitoring, and infrastructure tracking in a unified situational awareness interface.

[![GitHub stars](https://img.shields.io/github/stars/koala73/worldmonitor?style=social)](https://github.com/koala73/worldmonitor/stargazers)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Last commit](https://img.shields.io/github/last-commit/koala73/worldmonitor)](https://github.com/koala73/worldmonitor/commits/main)
[![Latest release](https://img.shields.io/github/v/release/koala73/worldmonitor?style=flat)](https://github.com/koala73/worldmonitor/releases/latest)
[![npm: global-status-monitoring-system](https://img.shields.io/npm/v/global-status-monitoring-system?logo=npm&label=npm)](https://www.npmjs.com/package/global-status-monitoring-system)
[![skills.sh](https://skills.sh/b/koala73/worldmonitor)](https://skills.sh/koala73/worldmonitor)

<p align="center">
  <a href="https://www.worldmonitor.app"><img src="https://img.shields.io/badge/Web_App-worldmonitor.app-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Web App"></a>&nbsp;
  <a href="https://tech.worldmonitor.app"><img src="https://img.shields.io/badge/Tech_Variant-tech.worldmonitor.app-0891b2?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Tech Variant"></a>&nbsp;
  <a href="https://finance.worldmonitor.app"><img src="https://img.shields.io/badge/Finance_Variant-finance.worldmonitor.app-059669?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Finance Variant"></a>&nbsp;
  <a href="https://commodity.worldmonitor.app"><img src="https://img.shields.io/badge/Commodity_Variant-commodity.worldmonitor.app-b45309?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Commodity Variant"></a>&nbsp;
  <a href="https://happy.worldmonitor.app"><img src="https://img.shields.io/badge/Happy_Variant-happy.worldmonitor.app-f59e0b?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Happy Variant"></a>&nbsp;
  <a href="https://energy.worldmonitor.app"><img src="https://img.shields.io/badge/Energy_Variant-energy.worldmonitor.app-eab308?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Energy Variant"></a>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/global-status-monitoring-system"><img src="https://img.shields.io/npm/v/global-status-monitoring-system?style=for-the-badge&logo=npm&logoColor=white&label=npm%20i%20global-status-monitoring-system&color=CB3837" alt="npm i global-status-monitoring-system"></a>&nbsp;
  <a href="https://www.npmjs.com/package/global-status-monitoring-system"><img src="https://img.shields.io/badge/CLI-npx%20global-status-monitoring-system-CB3837?style=for-the-badge&logo=npm&logoColor=white" alt="npx global-status-monitoring-system"></a>
</p>

<p align="center">
  <a href="https://www.worldmonitor.app/api/download?platform=windows-exe"><img src="https://img.shields.io/badge/Download-Windows_(.exe)-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download Windows"></a>&nbsp;
  <a href="https://www.worldmonitor.app/api/download?platform=macos-arm64"><img src="https://img.shields.io/badge/Download-macOS_Apple_Silicon-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Download macOS ARM"></a>&nbsp;
  <a href="https://www.worldmonitor.app/api/download?platform=macos-x64"><img src="https://img.shields.io/badge/Download-macOS_Intel-555555?style=for-the-badge&logo=apple&logoColor=white" alt="Download macOS Intel"></a>&nbsp;
  <a href="https://www.worldmonitor.app/api/download?platform=linux-appimage"><img src="https://img.shields.io/badge/Download-Linux_(.AppImage)-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Download Linux"></a>
</p>

<p align="center">
  <a href="https://www.worldmonitor.app/docs/documentation"><strong>Documentation</strong></a> &nbsp;·&nbsp;
  <a href="https://github.com/koala73/worldmonitor/releases/latest"><strong>Releases</strong></a> &nbsp;·&nbsp;
  <a href="https://www.worldmonitor.app/docs/contributing"><strong>Contributing</strong></a>
</p>

![Global Status Monitoring System Dashboard](docs/images/worldmonitor-7-mar-2026.jpg)

---

## What It Does

- **Curated news feeds** across global and regional categories, AI-synthesized into briefs
- **Dual map engine** — 3D globe (globe.gl) and WebGL flat map (deck.gl) with a shared map-layer catalog
- **Panel inventory** — concrete panel implementations across specialized variants
- **Cross-stream correlation** — military, economic, disaster, and escalation signal convergence
- **[Country Instability Index (CII)](https://www.worldmonitor.app/country-instability-index/)** — live CII v8 scores, bands, and approximate 24-hour movement for 31 Tier-1 countries
- **Finance radar** — stock exchanges, commodities, crypto, and a market composite
- **Local AI** — run everything with Ollama, no API keys required
- **Site variants** from a single codebase (world, tech, finance, commodity, happy, energy)
- **Native desktop app** (Tauri 2) for macOS, Windows, and Linux
- **Multilingual UI** with native-language feeds and RTL support
- **Machine Learning Lab** — browser-side ML experiments on live news (classification, regression, clustering, ensembles, concept learning)

For the full feature list, architecture, data sources, and algorithms, see the **[documentation](https://www.worldmonitor.app/docs/documentation)**.

---

## Machine Learning Project (B.Tech CSE 5th Semester) - 241AI005

This repository contains a complete **Machine Learning implementation** of the 241AI005 syllabus, integrated into the Global Status Monitoring System. The project demonstrates end-to-end ML pipelines for real-time geopolitical risk monitoring using live world event data.

### 🎓 Syllabus Coverage

All units of the 241AI005 Machine Learning syllabus are implemented and traceable:

| Syllabus Unit | Topics Covered | Implementation |
|---------------|----------------|----------------|
| **Unit I: Introduction** | ML concepts, Types of ML systems, Statistical Learning, Supervised/Unsupervised, Data preprocessing | `ml/gsms_ml/preprocessing.py`, `ml/gsms_ml/nlp.py` |
| **Unit II: Supervised Learning** | Concept Learning (Find-S, Candidate Elimination), Bayesian Learning (Naive Bayes), Instance-based (k-NN), Linear Models (Logistic Regression, SVM), Decision Boundary | `ml/gsms_ml/supervised.py`, `ml/gsms_ml/concept_learning.py` |
| **Unit III: Linear Models & Regression** | Linear Regression, Logistic Regression, SVM (Linear/RBF), Non-linear transformation | `ml/gsms_ml/supervised.py`, `ml/gsms_ml/regression.py` |
| **Unit IV: Evaluation & Unsupervised** | Accuracy, Precision, Recall, F1, ROC, Cross-validation, Bootstrapping, Clustering (K-Means, Hierarchical, DBSCAN) | `ml/gsms_ml/evaluation.py`, `ml/gsms_ml/clustering.py` |
| **Unit V: Ensemble Learning** | Bagging, Boosting (AdaBoost), Random Forest, Computational Learning Theory | `ml/gsms_ml/ensembles.py` |

### 🔧 Algorithms Implemented

**Preprocessing & NLP**
- Deduplication, imputation, standardization, chronological train/test splits
- TF-IDF + SVD embeddings, AFINN sentiment, Gazetteer NER, cosine similarity
- Text preprocessing pipeline with TF-IDF + SVD embeddings

**Concept Learning**
- Find-S algorithm (most specific hypothesis)
- Candidate Elimination (version space tracking with S/G boundaries)

**Classification**
- Logistic Regression (gradient descent)
- Linear SVM (Pegasos primal solver)
- RBF SVM (subsampled kernel for browser performance)
- Gaussian Naive Bayes
- k-Nearest Neighbors (k=5, cosine distance)

**Regression**
- Linear Regression (Normal Equations)
- MAE, MSE, RMSE, R² metrics

**Clustering**
- K-Means++ (with smart initialization)
- Hierarchical (Ward/Agglomerative linkage)
- DBSCAN (density-based)
- Gaussian Mixture Model (EM algorithm)

**Ensembles**
- Random Forest (feature importance via permutation)
- AdaBoost (decision stumps)
- Bagging (decision stumps with feature importance)

**Evaluation**
- 5-fold chronological cross-validation (prevents future-data leakage)
- 1000-sample bootstrap confidence intervals (95%)
- Bias-variance sweep (tree depth)
- Regularization path (λ sweep)

**Application Layer**
- **Global Status Engine**: Weighted risk scoring (0-100) per country with explainable drivers
- Browser-side ML via zero-dependency Web Workers
- Python ML Lab for reproducible offline experiments

### 📚 Documentation

- [ML Project Technical Documentation](docs/ml-project.md) — Full implementation details
- [ML Syllabus Mapping Guide](docs/ml-syllabus-mapping.md) — Direct traceability to 241AI005 course outcomes

### ▶️ How to Run the ML Components

#### 🐍 **Python ML Lab (Offline, Reproducible Analysis)**
> *For detailed reports, parameter tuning, and syllabus-compliant verification*

```bash
# Navigate to ML lab directory
cd ml

# Setup virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Execute complete ML pipeline - generates REPORT.md + 9 JSON files in ml/results/
python3 run_all.py

# Run test suite (optional)
python3 -m pytest tests/ -v
```

**Outputs**: `REPORT.md` (full analysis), `supervised.json`, `regression.json`, `clustering.json`, `ensembles.json`, `evaluation.json`, `concept_learning.json`, `status_engine.json`, `semantic_retrieval.json`, `meta.json`

#### 🌐 **Browser ML Lab (Live Interactive Dashboard)**
> *For real-time exploration of ML on live world events - no setup beyond dev server*

```bash
# From project root (global-status-monitoring-system/):
npm install          # First time only
npm run dev          # Starts development server
```

**Access Instructions:**
1. Open browser to **http://localhost:3006/** (or check terminal for actual port if 3006 is busy)
2. Click the menu button (☰) in top-left corner
3. Navigate to: **Intelligence → ML Lab**
4. Explore all 8 tabs with live geopolitical data:
   - Overview: Global risk scores (0-100) per country
   - Dataset: Live event stream from GDELT/USGS/NOAA/ACLED
   - Classification: Real-time event labeling (Logistic Regression, SVM, etc.)
   - Regression: Impact prediction (fatalities, economic damage)
   - Clustering: Pattern discovery (K-Means++, DBSCAN, Hierarchical)
   - Ensembles: Robust predictions (Random Forest, AdaBoost, Bagging)
   - Evaluation: Model validation (temporal CV, bootstrap CI, bias-variance)
   - Concept Learning: Interpretable rules (Find-S, Candidate Elimination version space)

**Key Features:**
- Server binds to `0.0.0.0` (accessible from any device on your network)
- Auto-selects available port (defaults to 3006 if 3000-3005 are in use)
- Zero-dependency Web Workers - all ML runs in-browser, no backend API calls
- Updates live as new world events arrive (every 15 minutes for news feed)

---

## Quick Start

```bash
git clone https://github.com/koala73/worldmonitor.git
cd worldmonitor
npm install
npm run dev
```

Open [localhost:3000](http://localhost:3000) (override the port with `DEV_PORT` in `.env.local`). The app runs with no environment variables.

Feature-specific data sources may require credentials. See `.env.example` for the full list.

For variant-specific development:

```bash
npm run dev:tech       # tech.worldmonitor.app
npm run dev:finance    # finance.worldmonitor.app
npm run dev:commodity  # commodity.worldmonitor.app
npm run dev:happy      # happy.worldmonitor.app
npm run dev:energy     # energy.worldmonitor.app
```

See the **[self-hosting guide](https://www.worldmonitor.app/docs/getting-started)** for deployment options (Vercel, Docker, static).

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Vanilla TypeScript, Vite, globe.gl + Three.js, deck.gl + MapLibre GL |
| **Desktop** | Tauri 2 (Rust) with Node.js sidecar |
| **AI/ML** | Ollama / Groq / OpenRouter, Transformers.js (browser-side), scikit-learn (Python ML lab) |
| **API Contracts** | Protocol Buffers and sebuf HTTP annotations |
| **Deployment** | Vercel Edge Functions, Railway relay, Tauri, PWA |
| **Caching** | Redis (Upstash), 3-tier cache, CDN, service worker |

Full stack details in the **[architecture docs](https://www.worldmonitor.app/docs/architecture)**.

---

## Programmatic Access

The application is built for agents and scripts as well as browsers:

- **MCP server** — `https://worldmonitor.app/mcp` (Streamable HTTP). Public `tools/list`; `tools/call` authenticates with a `X-WorldMonitor-Key` header or OAuth.
  The server also publishes its Agent Skills through the draft `io.modelcontextprotocol/skills` extension (`skills/list`, `skills/get`, and `skill://…` resource reads).
- **REST API** — base `https://api.worldmonitor.app`, described by the [OpenAPI spec](https://worldmonitor.app/openapi.yaml).
- **CLI** — the official npm package (source in [`cli/`](cli/)):
  ```sh
  npx global-status-monitoring-system tools          # run ad-hoc — list every MCP tool (no key needed)
  npm install -g global-status-monitoring-system     # or install the command
  global-status-monitoring-system risk IR --api-key wm_xxx
  ```
- **SDKs** — official zero-dependency client libraries: Python, Ruby, Go (source in [`sdk/`](sdk/)).

Agent discovery files: [`llms.txt`](https://worldmonitor.app/llms.txt) · [agent-skills manifest](https://worldmonitor.app/.well-known/agent-skills/index.json) · [api-catalog](https://worldmonitor.app/.well-known/api-catalog).

---

## Flight Data

Flight data provided graciously by [Wingbits](https://wingbits.com?utm_source=worldmonitor&utm_medium=referral&utm_campaign=worldmonitor), the most advanced ADS-B flight data solution.

---

## Data Sources

Aggregates attributed upstream sources across geopolitics, finance, energy, climate, aviation, cyber, military, infrastructure, and news intelligence. Curated feeds and freshness-tracked source groups are published in the full [data sources catalog](https://www.worldmonitor.app/docs/data-sources), with provider, feed-tier, license-posture, and collection-method details.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

```bash
npm run typecheck        # Type checking
npm run build:full       # Production build
```

---

## License

**AGPL-3.0-only** for the source code. Commercial use is permitted under the AGPL when you comply with its copyleft and source-availability terms.

| Use Case | Allowed? |
|----------|----------|
| Personal / research / educational | Yes, under AGPL-3.0-only |
| Self-hosted instance | Yes, under AGPL-3.0-only |
| Fork and modify | Yes, share source under AGPL-3.0-only when required |
| Commercial use / SaaS | Yes, under AGPL-3.0-only when you comply with AGPL obligations |
| Private-source proprietary use or official branding rights | Separate commercial or trademark permission needed |

See [LICENSE](LICENSE) for the full code license and [docs/license.mdx](docs/license.mdx) for a plain-language summary. Commercial licensing is available as an alternative option for teams that need non-AGPL terms.

Copyright (C) 2024-2026 Elie Habib. All rights reserved.

---

## Author

**Elie Habib** — [GitHub](https://github.com/koala73)

## Contributors

<a href="https://github.com/koala73/worldmonitor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=koala73/worldmonitor" />
</a>

## Security Acknowledgments

We thank the following researchers for responsibly disclosing security issues:

- **Cody Richard** — Disclosed three security findings covering IPC command exposure, renderer-to-sidecar trust boundary analysis, and fetch patch credential injection architecture (2026)

See our [Security Policy](./SECURITY.md) for responsible disclosure guidelines.

---

<p align="center">
  <a href="https://www.worldmonitor.app">worldmonitor.app</a> &nbsp;·&nbsp;
  <a href="https://www.worldmonitor.app/docs/documentation">docs.worldmonitor.app</a> &nbsp;·&nbsp;
  <a href="https://finance.worldmonitor.app">finance.worldmonitor.app</a> &nbsp;·&nbsp;
  <a href="https://commodity.worldmonitor.app">commodity.worldmonitor.app</a>
</p>

## Star History

<a href="https://star-history.dera.page/#koala73/worldmonitor&type=Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://star-history.dera.page/svg?repos=koala73/worldmonitor&type=Date&theme=dark" />
   <img alt="Star History Chart" src="https://star-history.dera.page/svg?repos=koala73/worldmonitor&type=Date" />
 </picture>
</a>

---

## Academic Project Details

**Course:** 241AI005 Machine Learning (B.Tech CSE 5th Semester, 2024 Curriculum)  
**Project Title:** GLOBAL STATUS MONITORING SYSTEM  
**Team:**  
- 24B11CS489: Viswajeet  
- 24B11CS357: Priyamwda Rai  
- 24B11CS461: Satya Sri Hema  
- 24B11CS256: Uma Ram Charan