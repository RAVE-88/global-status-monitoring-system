/**
 * ML Lab Panel — the browser-side Machine Learning laboratory for the
 * Global Status Monitoring System.
 *
 * Runs the full ML experiment suite over live news headlines via a dedicated
 * Web Worker (ml-lab.worker.ts). Displays results across internal tabs:
 * Pipeline Overview, Dataset, Classification, Regression, Clustering,
 * Ensembles, Evaluation, Concept Learning.
 */
import { Panel } from '@/components/Panel';
import { mlLabWorker, type MLLabInputItem, type MLLabResults, MLLabUnavailableError } from '@/services/ml-lab-worker';
import { t } from '@/services/i18n';
import { showToast } from '@/utils';
import { setTrustedHtml, trustedHtml } from '@/utils/dom-utils';
import { debounce } from '@/utils';

// Embedded real sample data (generated from the committed Python snapshot)
// Falls back to this when live news is not available or offline.
import ML_SAMPLE_EVENTS from '@/data/ml-sample-events.generated';

interface TabState {
  id: string;
  label: string;
  render: (results: MLLabResults) => string;
}

const TAB_ORDER: TabState[] = [
  { id: 'overview', label: 'Pipeline Overview', render: renderOverviewTab },
  { id: 'dataset', label: 'Dataset', render: renderDatasetTab },
  { id: 'classification', label: 'Classification', render: renderClassificationTab },
  { id: 'regression', label: 'Regression', render: renderRegressionTab },
  { id: 'clustering', label: 'Clustering', render: renderClusteringTab },
  { id: 'ensembles', label: 'Ensembles', render: renderEnsemblesTab },
  { id: 'evaluation', label: 'Evaluation', render: renderEvaluationTab },
  { id: 'concept', label: 'Concept Learning', render: renderConceptTab },
];

const REFRESH_DEBOUNCE_MS = 30000; // 30 seconds between auto-refreshes
const MAX_LIVE_ITEMS = 400; // Cap live items sent to worker

export class MLLabPanel extends Panel {
  private activeTab = 'overview';
  private currentResults: MLLabResults | null = null;
  private isRunning = false;
  private lastRunAt = 0;
  private tabButtons: HTMLButtonElement[] = [];
  private contentEl: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;
  private runButton: HTMLButtonElement | null = null;
  private useSampleCheckbox: HTMLInputElement | null = null;
  private refreshTimer: ReturnType<typeof setTimeout> | null = null;
  private debouncedRefresh = debounce(() => this.runExperiments(), REFRESH_DEBOUNCE_MS);

  constructor() {
    super({
      id: 'ml-lab',
      title: t('panels.mlLab', { defaultValue: 'ML Lab' }),
      defaultRowSpan: 2,
      className: 'panel-wide',
    });
  }

  protected async onMount(): Promise<void> {
    this.renderShell();
    this.bindEvents();
    // Initial run with sample data (fast, shows something immediately)
    await this.runExperiments(true);
    // Then try live data if available
    this.scheduleLiveRefresh();
  }

  protected onUnmount(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  private renderShell(): void {
    const tabsHtml = TAB_ORDER.map((tab, i) => `
      <button class="ml-lab-tab${i === 0 ? ' active' : ''}" data-tab="${tab.id}" role="tab" aria-selected="${i === 0}">
        ${tab.label}
      </button>
    `).join('');

    setTrustedHtml(this.content, trustedHtml(`
      <div class="ml-lab-panel">
        <div class="ml-lab-header">
          <div class="ml-lab-title-row">
            <h3 class="ml-lab-title">${t('panels.mlLab', { defaultValue: 'ML Lab' })}</h3>
            <span class="ml-lab-badge">${t('common.live', { defaultValue: 'LIVE' })}</span>
          </div>
          <div class="ml-lab-controls">
            <label class="ml-lab-sample-toggle">
              <input type="checkbox" id="mlLabUseSample" ${!navigator.onLine ? 'checked' : ''}>
              <span>${t('mlLab.useSampleData', { defaultValue: 'Use sample data (offline)' })}</span>
            </label>
            <button class="ml-lab-run-btn" id="mlLabRunBtn" disabled>
              <span class="ml-lab-btn-text">${t('mlLab.runExperiments', { defaultValue: 'Run Experiments' })}</span>
              <span class="ml-lab-spinner" aria-hidden="true"></span>
            </button>
          </div>
        </div>
        <div class="ml-lab-status" id="mlLabStatus" aria-live="polite"></div>
        <nav class="ml-lab-tabs" role="tablist" aria-label="ML Lab sections">
          ${tabsHtml}
        </nav>
        <div class="ml-lab-content" id="mlLabContent" role="tabpanel">
          <div class="ml-lab-placeholder">${t('mlLab.selectTab', { defaultValue: 'Select a tab to view results' })}</div>
        </div>
      </div>
    `, 'MLLabPanel shell'));

    this.tabButtons = Array.from(this.content.querySelectorAll('.ml-lab-tab'));
    this.contentEl = this.content.querySelector('#mlLabContent') as HTMLElement;
    this.statusEl = this.content.querySelector('#mlLabStatus') as HTMLElement;
    this.runButton = this.content.querySelector('#mlLabRunBtn') as HTMLButtonElement;
    this.useSampleCheckbox = this.content.querySelector('#mlLabUseSample') as HTMLInputElement;
  }

  private bindEvents(): void {
    // Tab switching
    this.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => this.switchTab(btn.dataset.tab!));
    });

    // Run button
    this.runButton?.addEventListener('click', () => this.runExperiments());

    // Sample data checkbox
    this.useSampleCheckbox?.addEventListener('change', () => {
      this.runExperiments(true);
    });
  }

  private switchTab(tabId: string): void {
    this.activeTab = tabId;
    this.tabButtons.forEach(btn => {
      const isActive = btn.dataset.tab === tabId;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive.toString());
    });
    this.renderCurrentTab();
  }

  private renderCurrentTab(): void {
    if (!this.contentEl || !this.currentResults) return;
    const tab = TAB_ORDER.find(t => t.id === this.activeTab);
    if (!tab) return;
    setTrustedHtml(this.contentEl, trustedHtml(tab.render(this.currentResults), `MLLabPanel:${tab.id}`));
  }

  private async runExperiments(forceSample = false): Promise<void> {
    if (this.isRunning) return;

    const useSample = forceSample || this.useSampleCheckbox?.checked || !navigator.onLine;
    this.isRunning = true;
    this.updateRunButton(true);
    this.updateStatus(useSample
      ? t('mlLab.runningSample', { defaultValue: 'Running experiments on sample data…' })
      : t('mlLab.runningLive', { defaultValue: 'Running experiments on live news…' })
    );

    try {
      let items: MLLabInputItem[];

      if (useSample) {
        items = ML_SAMPLE_EVENTS.slice(0, MAX_LIVE_ITEMS);
      } else {
        // Pull live news from the app context
        const ctx = (window as any).__WORLDMONITOR_CTX__;
        const allNews = ctx?.allNews ?? [];
        items = allNews
          .filter((n: any) => n.title && n.pubDate)
          .slice(0, MAX_LIVE_ITEMS)
          .map((n: any) => ({
            title: n.title,
            source: n.source || 'unknown',
            pubDate: n.pubDate,
            isAlert: n.isAlert || false,
            tier: n.tier,
            corroborationCount: n.corroborationCount,
            credibilityScore: n.credibilityScore,
            snippet: n.snippet || n.summary || '',
          }));
      }

      if (items.length === 0) {
        throw new Error(t('mlLab.noData', { defaultValue: 'No news items available for analysis' }));
      }

      const results = await mlLabWorker.runExperiments(items);
      this.currentResults = results;
      this.lastRunAt = Date.now();
      this.updateStatus(t('mlLab.completed', {
        defaultValue: 'Experiments completed at {{time}}',
        time: new Date(this.lastRunAt).toLocaleTimeString(),
      }));
      this.renderCurrentTab();
      this.scheduleLiveRefresh();
    } catch (error) {
      const msg = error instanceof MLLabUnavailableError
        ? t('mlLab.workerUnavailable', { defaultValue: 'ML Lab worker unavailable — browser may not support module workers' })
        : error instanceof Error
          ? error.message
          : t('mlLab.unknownError', { defaultValue: 'Unknown error' });
      this.updateStatus(msg, 'error');
      showToast(msg);
    } finally {
      this.isRunning = false;
      this.updateRunButton(false);
    }
  }

  private scheduleLiveRefresh(): void {
    if (this.refreshTimer) clearTimeout(this.refreshTimer);
    // Only auto-refresh live data when online and not using sample
    if (navigator.onLine && !this.useSampleCheckbox?.checked) {
      this.refreshTimer = setTimeout(() => {
        this.debouncedRefresh();
      }, REFRESH_DEBOUNCE_MS);
    }
  }

  private updateRunButton(running: boolean): void {
    if (!this.runButton) return;
    this.runButton.disabled = running;
    const textEl = this.runButton.querySelector('.ml-lab-btn-text');
    const spinnerEl = this.runButton.querySelector('.ml-lab-spinner');
    if (textEl) textEl.textContent = running
      ? t('mlLab.running', { defaultValue: 'Running…' })
      : t('mlLab.runExperiments', { defaultValue: 'Run Experiments' });
    if (spinnerEl) (spinnerEl as HTMLElement).style.display = running ? 'inline-block' : 'none';
  }

  private updateStatus(message: string, type: 'info' | 'error' = 'info'): void {
    if (!this.statusEl) return;
    this.statusEl.className = `ml-lab-status ${type}`;
    this.statusEl.textContent = message;
  }
}

// ============================================
// Tab renderers
// ============================================

function renderOverviewTab(r: MLLabResults): string {
  const { preprocess, pipelinePreview, statusEngine } = r;
  const topCountries = statusEngine.countries.slice(0, 10);

  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.dataSummary', { defaultValue: 'Data Summary' })}</h4>
        <div class="ml-lab-grid ml-lab-grid-4">
          <div class="ml-lab-stat">
            <span class="ml-lab-stat-value">${preprocess.input}</span>
            <span class="ml-lab-stat-label">${t('mlLab.rawArticles', { defaultValue: 'Raw Articles' })}</span>
          </div>
          <div class="ml-lab-stat">
            <span class="ml-lab-stat-value">${preprocess.afterDedupe}</span>
            <span class="ml-lab-stat-label">${t('mlLab.afterDedupe', { defaultValue: 'After Dedupe' })}</span>
          </div>
          <div class="ml-lab-stat">
            <span class="ml-lab-stat-value">${preprocess.trainSize}</span>
            <span class="ml-lab-stat-label">${t('mlLab.trainSamples', { defaultValue: 'Train Samples' })}</span>
          </div>
          <div class="ml-lab-stat">
            <span class="ml-lab-stat-value">${preprocess.testSize}</span>
            <span class="ml-lab-stat-label">${t('mlLab.testSamples', { defaultValue: 'Test Samples' })}</span>
          </div>
        </div>
        <details class="ml-lab-details">
          <summary>${t('mlLab.classDistribution', { defaultValue: 'Class Distribution' })}</summary>
          <pre class="ml-lab-pre">${JSON.stringify(preprocess.classDistribution, null, 2)}</pre>
        </details>
      </section>

      <section class="ml-lab-section">
        <h4>${t('mlLab.pipelinePreview', { defaultValue: 'Pipeline Preview (Latest Headlines)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead>
              <tr>
                <th>${t('mlLab.headline', { defaultValue: 'Headline' })}</th>
                <th>${t('mlLab.sentiment', { defaultValue: 'Sentiment' })}</th>
                <th>${t('mlLab.entities', { defaultValue: 'Entities' })}</th>
                <th>${t('mlLab.category', { defaultValue: 'Category' })}</th>
                <th>${t('mlLab.riskLevel', { defaultValue: 'Risk Level' })}</th>
                <th>${t('mlLab.cluster', { defaultValue: 'Cluster' })}</th>
              </tr>
            </thead>
            <tbody>
              ${pipelinePreview.slice(0, 15).map(p => `
                <tr>
                  <td class="ml-lab-headline">${escapeHtml(p.headline)}</td>
                  <td><span class="ml-lab-sentiment ml-lab-sentiment-${p.sentiment >= 0 ? 'pos' : 'neg'}">${p.sentiment.toFixed(2)}</span></td>
                  <td>${p.entities.slice(0, 3).map(escapeHtml).join(', ') || '—'}</td>
                  <td><span class="ml-lab-badge ml-lab-badge-${p.category}">${escapeHtml(p.category)}</span></td>
                  <td><span class="ml-lab-badge ml-lab-level-${p.level}">${escapeHtml(p.level)}</span></td>
                  <td>${p.clusterId !== null ? p.clusterId : '—'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>

      <section class="ml-lab-section">
        <h4>${t('mlLab.globalStatusTop', { defaultValue: 'Global Status — Top 10 Countries' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead>
              <tr>
                <th>${t('mlLab.country', { defaultValue: 'Country' })}</th>
                <th>${t('mlLab.score', { defaultValue: 'Score' })}</th>
                <th>${t('mlLab.band', { defaultValue: 'Band' })}</th>
                <th>${t('mlLab.events', { defaultValue: 'Events' })}</th>
                <th>${t('mlLab.drivers', { defaultValue: 'Top Drivers' })}</th>
              </tr>
            </thead>
            <tbody>
              ${topCountries.map(c => `
                <tr>
                  <td><strong>${escapeHtml(c.code)}</strong></td>
                  <td>${c.score.toFixed(1)}</td>
                  <td><span class="ml-lab-badge ml-lab-band-${c.band}">${escapeHtml(c.band)}</span></td>
                  <td>${c.events}</td>
                  <td>${c.drivers.slice(0, 3).map(d => `${escapeHtml(d.label)}: ${d.value.toFixed(1)}`).join('; ')}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <p class="ml-lab-methodology">${t('mlLab.methodology', { defaultValue: 'Methodology' })}: ${statusEngine.methodology}</p>
      </section>
    </div>
  `;
}

function renderDatasetTab(r: MLLabResults): string {
  const { preprocess } = r;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.features', { defaultValue: 'Feature Vector (d=14)' })}</h4>
        <ul class="ml-lab-feature-list">
          ${preprocess.features.map(f => `<li><code>${escapeHtml(f)}</code></li>`).join('')}
        </ul>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.preprocessingSteps', { defaultValue: 'Preprocessing Steps' })}</h4>
        <ol class="ml-lab-steps">
          <li>${t('mlLab.stepDedupe', { defaultValue: 'Near-duplicate removal via Jaccard similarity on content tokens (threshold 0.85)' })}</li>
          <li>${t('mlLab.stepImpute', { defaultValue: 'Missing value imputation: median for numeric, mode for categorical' })}</li>
          <li>${t('mlLab.stepStandardize', { defaultValue: 'Standardization (z-score) fitted on training split only' })}</li>
          <li>${t('mlLab.stepChronological', { defaultValue: 'Chronological train/test split (80/20) to avoid future-data leakage' })}</li>
          <li>${t('mlLab.stepLabels', { defaultValue: 'Derived heuristic labels: risk_band (low/medium/high/critical) from threat classifier level + corroboration' })}</li>
        </ol>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.classBalance', { defaultValue: 'Class Balance' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>${t('mlLab.class', { defaultValue: 'Class' })}</th><th>${t('mlLab.trainCount', { defaultValue: 'Train' })}</th><th>${t('mlLab.testCount', { defaultValue: 'Test' })}</th></tr></thead>
            <tbody>
              ${Object.entries(preprocess.classDistribution).map(([cls, count]) => `
                <tr><td>${escapeHtml(cls)}</td><td>${count}</td><td>${Math.round(count * 0.2)}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

function renderClassificationTab(r: MLLabResults): string {
  const reports = r.classification;
  if (!reports.length) return '<p class="ml-lab-empty">No classification results</p>';

  return `
    <div class="ml-lab-tab-content">
      ${reports.map(rep => `
        <section class="ml-lab-section">
          <h4>${t('mlLab.task', { defaultValue: 'Task' })}: ${escapeHtml(rep.task)} (train=${rep.trainSize}, test=${rep.testSize})</h4>
          <div class="ml-lab-table-wrap">
            <table class="ml-lab-table">
              <thead>
                <tr>
                  <th>${t('mlLab.model', { defaultValue: 'Model' })}</th>
                  <th>${t('mlLab.accuracy', { defaultValue: 'Accuracy' })}</th>
                  <th>${t('mlLab.precision', { defaultValue: 'Precision' })}</th>
                  <th>${t('mlLab.recall', { defaultValue: 'Recall' })}</th>
                  <th>${t('mlLab.f1', { defaultValue: 'F1' })}</th>
                  <th>${t('mlLab.rocAuc', { defaultValue: 'ROC-AUC' })}</th>
                </tr>
              </thead>
              <tbody>
                ${rep.rows.map(row => `
                  <tr>
                    <td><strong>${escapeHtml(row.model)}</strong></td>
                    <td>${row.accuracy.toFixed(3)}</td>
                    <td>${row.precision.toFixed(3)}</td>
                    <td>${row.recall.toFixed(3)}</td>
                    <td>${row.f1.toFixed(3)}</td>
                    <td>${row.rocAuc !== null ? row.rocAuc.toFixed(3) : 'N/A'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
          <details class="ml-lab-details">
            <summary>${t('mlLab.confusionMatrices', { defaultValue: 'Confusion Matrices' })}</summary>
            ${Object.entries(rep.confusion).map(([model, cm]) => `
              <h5>${escapeHtml(model)}</h5>
              <div class="ml-lab-confusion">
                <table class="ml-lab-table">
                  <thead><tr><th></th>${cm.labels.map(l => `<th>${escapeHtml(l)}</th>`).join('')}</tr></thead>
                  <tbody>
                    ${cm.matrix.map((row, i) => `
                      <tr><th>${escapeHtml(cm.labels[i] ?? '')}</th>${row.map(v => `<td>${v}</td>`).join('')}</tr>
                    `).join('')}
                  </tbody>
                </table>
              </div>
            `).join('')}
          </details>
          <details class="ml-lab-details">
            <summary>${t('mlLab.crossValidation', { defaultValue: '5-Fold Cross-Validation' })}</summary>
            <div class="ml-lab-table-wrap">
              <table class="ml-lab-table">
                <thead><tr><th>${t('mlLab.model', { defaultValue: 'Model' })}</th><th>${t('mlLab.mean', { defaultValue: 'Mean' })}</th><th>${t('mlLab.std', { defaultValue: 'Std' })}</th><th>${t('mlLab.folds', { defaultValue: 'Folds' })}</th></tr></thead>
                <tbody>
                  ${Object.entries(rep.crossVal).map(([model, cv]) => `
                    <tr><td>${escapeHtml(model)}</td><td>${cv.mean.toFixed(3)}</td><td>${cv.std.toFixed(3)}</td><td>${cv.folds}</td></tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </details>
          <details class="ml-lab-details">
            <summary>${t('mlLab.bootstrapCI', { defaultValue: 'Bootstrap Confidence Intervals (95%)' })}</summary>
            <div class="ml-lab-table-wrap">
              <table class="ml-lab-table">
                <thead><tr><th>${t('mlLab.model', { defaultValue: 'Model' })}</th><th>${t('mlLab.mean', { defaultValue: 'Mean' })}</th><th>${t('mlLab.ci', { defaultValue: '95% CI' })}</th></tr></thead>
                <tbody>
                  ${Object.entries(rep.bootstrap).map(([model, bs]) => `
                    <tr><td>${escapeHtml(model)}</td><td>${bs.mean.toFixed(3)}</td><td>[${bs.lo.toFixed(3)}, ${bs.hi.toFixed(3)}]</td></tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      `).join('')}
    </div>
  `;
}

function renderRegressionTab(r: MLLabResults): string {
  const { regression } = r;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.regressionTarget', { defaultValue: 'Target' })}: ${escapeHtml(regression.target)}</h4>
        <div class="ml-lab-grid ml-lab-grid-4">
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${regression.mae.toFixed(3)}</span><span class="ml-lab-stat-label">${t('mlLab.mae', { defaultValue: 'MAE' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${regression.mse.toFixed(3)}</span><span class="ml-lab-stat-label">${t('mlLab.mse', { defaultValue: 'MSE' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${regression.rmse.toFixed(3)}</span><span class="ml-lab-stat-label">${t('mlLab.rmse', { defaultValue: 'RMSE' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${regression.r2.toFixed(3)}</span><span class="ml-lab-stat-label">${t('mlLab.r2', { defaultValue: 'R²' })}</span></div>
        </div>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.predictedVsActual', { defaultValue: 'Predicted vs Actual (Sample)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>${t('mlLab.actual', { defaultValue: 'Actual' })}</th><th>${t('mlLab.predicted', { defaultValue: 'Predicted' })}</th><th>${t('mlLab.residual', { defaultValue: 'Residual' })}</th></tr></thead>
            <tbody>
              ${regression.samplePoints.slice(0, 20).map(p => `
                <tr><td>${p.actual.toFixed(3)}</td><td>${p.predicted.toFixed(3)}</td><td>${(p.predicted - p.actual).toFixed(3)}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <p class="ml-lab-note">${t('mlLab.regressionNote', { defaultValue: 'Statistical estimation on derived risk score — not a geopolitical forecast.' })}</p>
      </section>
    </div>
  `;
}

function renderClusteringTab(r: MLLabResults): string {
  const { clustering } = r;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.pcaProjection', { defaultValue: 'PCA 2D Projection' })}</h4>
        <div class="ml-lab-grid ml-lab-grid-3">
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.kmeans.k}</span><span class="ml-lab-stat-label">${t('mlLab.kmeansK', { defaultValue: 'K-Means K' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.kmeans.inertia.toFixed(2)}</span><span class="ml-lab-stat-label">${t('mlLab.inertia', { defaultValue: 'Inertia' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.kmeans.silhouette !== null ? clustering.kmeans.silhouette.toFixed(3) : 'N/A'}</span><span class="ml-lab-stat-label">${t('mlLab.silhouette', { defaultValue: 'Silhouette' })}</span></div>
        </div>
        <p class="ml-lab-note">${t('mlLab.clusterSizes', { defaultValue: 'Cluster sizes' })}: ${clustering.kmeans.sizes.join(', ')}</p>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.dbscan', { defaultValue: 'DBSCAN' })}</h4>
        <div class="ml-lab-grid ml-lab-grid-3">
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.dbscan.eps}</span><span class="ml-lab-stat-label">${t('mlLab.eps', { defaultValue: 'Eps' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.dbscan.minPts}</span><span class="ml-lab-stat-label">${t('mlLab.minPts', { defaultValue: 'MinPts' })}</span></div>
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.dbscan.noiseCount}</span><span class="ml-lab-stat-label">${t('mlLab.noisePoints', { defaultValue: 'Noise Points' })}</span></div>
        </div>
        <p>${t('mlLab.clusterSizes', { defaultValue: 'Cluster sizes' })}: ${clustering.dbscan.sizes.join(', ')}</p>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.hierarchical', { defaultValue: 'Hierarchical (Agglomerative)' })}</h4>
        <div class="ml-lab-grid ml-lab-grid-2">
          <div class="ml-lab-stat"><span class="ml-lab-stat-value">${clustering.hierarchical.silhouette !== null ? clustering.hierarchical.silhouette.toFixed(3) : 'N/A'}</span><span class="ml-lab-stat-label">${t('mlLab.silhouette', { defaultValue: 'Silhouette' })}</span></div>
        </div>
        <p>${t('mlLab.clusterSizes', { defaultValue: 'Cluster sizes' })}: ${clustering.hierarchical.sizes.join(', ')}</p>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.pcaPoints', { defaultValue: 'PCA Points (first 50)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>X</th><th>Y</th><th>${t('mlLab.cluster', { defaultValue: 'Cluster' })}</th><th>${t('mlLab.noise', { defaultValue: 'Noise' })}</th></tr></thead>
            <tbody>
              ${clustering.pca.points.slice(0, 50).map(p => `
                <tr><td>${p.x.toFixed(3)}</td><td>${p.y.toFixed(3)}</td><td>${p.cluster}</td><td>${p.noise ? 'Yes' : 'No'}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

function renderEnsemblesTab(r: MLLabResults): string {
  const { ensembles } = r;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.ensembleMetrics', { defaultValue: 'Ensemble Metrics' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead>
              <tr>
                <th>${t('mlLab.model', { defaultValue: 'Model' })}</th>
                <th>${t('mlLab.accuracy', { defaultValue: 'Accuracy' })}</th>
                <th>${t('mlLab.precision', { defaultValue: 'Precision' })}</th>
                <th>${t('mlLab.recall', { defaultValue: 'Recall' })}</th>
                <th>${t('mlLab.f1', { defaultValue: 'F1' })}</th>
                <th>${t('mlLab.rocAuc', { defaultValue: 'ROC-AUC' })}</th>
              </tr>
            </thead>
            <tbody>
              ${ensembles.rows.map(row => `
                <tr>
                  <td><strong>${escapeHtml(row.model)}</strong></td>
                  <td>${row.accuracy.toFixed(3)}</td>
                  <td>${row.precision.toFixed(3)}</td>
                  <td>${row.recall.toFixed(3)}</td>
                  <td>${row.f1.toFixed(3)}</td>
                  <td>${row.rocAuc !== null ? row.rocAuc.toFixed(3) : 'N/A'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.featureImportance', { defaultValue: 'Feature Importance (Random Forest)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>${t('mlLab.feature', { defaultValue: 'Feature' })}</th><th>${t('mlLab.importance', { defaultValue: 'Importance' })}</th></tr></thead>
            <tbody>
              ${ensembles.featureImportance.map(f => `
                <tr><td><code>${escapeHtml(f.feature)}</code></td><td>${f.importance.toFixed(4)}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

function renderEvaluationTab(r: MLLabResults): string {
  const { biasVariance, regularization } = r;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.biasVariance', { defaultValue: 'Bias-Variance Tradeoff (Tree Depth Sweep)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>${t('mlLab.depth', { defaultValue: 'Depth' })}</th><th>${t('mlLab.trainError', { defaultValue: 'Train Error' })}</th><th>${t('mlLab.testError', { defaultValue: 'Test Error' })}</th></tr></thead>
            <tbody>
              ${biasVariance.sweep.values.map(v => `
                <tr><td>${v.depth}</td><td>${v.trainError.toFixed(3)}</td><td>${v.testError.toFixed(3)}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.regularization', { defaultValue: 'Regularization Sweep (Logistic Regression)' })}</h4>
        <div class="ml-lab-table-wrap">
          <table class="ml-lab-table">
            <thead><tr><th>λ (Lambda)</th><th>${t('mlLab.trainAcc', { defaultValue: 'Train Acc' })}</th><th>${t('mlLab.testAcc', { defaultValue: 'Test Acc' })}</th></tr></thead>
            <tbody>
              ${regularization.lambdas.map((lambda, i) => `
                <tr><td>${lambda}</td><td>${regularization.trainAccuracy[i]?.toFixed(3) ?? ''}</td><td>${regularization.testAccuracy[i]?.toFixed(3) ?? ''}</td></tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

function renderConceptTab(r: MLLabResults): string {
  const { conceptLearning } = r;
  const { finds, ce } = conceptLearning;
  return `
    <div class="ml-lab-tab-content">
      <section class="ml-lab-section">
        <h4>${t('mlLab.findS', { defaultValue: 'Find-S Algorithm' })}</h4>
        <p>${t('mlLab.finalHypothesis', { defaultValue: 'Final Hypothesis' })}:</p>
        <pre class="ml-lab-pre">${JSON.stringify(finds.hypothesis, null, 2)}</pre>
        <details class="ml-lab-details">
          <summary>${t('mlLab.steps', { defaultValue: 'Generalization Steps' })}</summary>
          <div class="ml-lab-table-wrap">
            <table class="ml-lab-table">
              <thead><tr><th>Example #</th><th>Hypothesis</th></tr></thead>
              <tbody>
                ${finds.steps.map(s => `
                  <tr><td>${s.exampleIndex}</td><td><pre>${JSON.stringify(s.hypothesis, null, 2)}</pre></td></tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </details>
      </section>
      <section class="ml-lab-section">
        <h4>${t('mlLab.candidateElimination', { defaultValue: 'Candidate Elimination' })}</h4>
        <div class="ml-lab-grid ml-lab-grid-2">
          <div>
            <h5>${t('mlLab.specificBoundary', { defaultValue: 'Specific Boundary (S)' })}</h5>
            <pre class="ml-lab-pre">${JSON.stringify(ce.specific, null, 2)}</pre>
          </div>
          <div>
            <h5>${t('mlLab.generalBoundary', { defaultValue: 'General Boundary (G)' })}</h5>
            <pre class="ml-lab-pre">${JSON.stringify(ce.general, null, 2)}</pre>
          </div>
        </div>
        <details class="ml-lab-details">
          <summary>${t('mlLab.versionSpaceHistory', { defaultValue: 'Version Space History' })}</summary>
          ${ce.history.map(h => `
            <div class="ml-lab-step">
              <strong>Step ${h.step}</strong>
              <div class="ml-lab-grid ml-lab-grid-2">
                <div><strong>S:</strong> <pre>${JSON.stringify(h.specific, null, 2)}</pre></div>
                <div><strong>G:</strong> <pre>${JSON.stringify(h.general, null, 2)}</pre></div>
              </div>
            </div>
          `).join('')}
        </details>
        ${ce.notes.length ? `
          <details class="ml-lab-details">
            <summary>${t('mlLab.notes', { defaultValue: 'Notes' })}</summary>
            <ul>${ce.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>
          </details>
        ` : ''}
      </section>
    </div>
  `;
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}