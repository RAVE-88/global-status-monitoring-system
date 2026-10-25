import { type AuthSession, getAuthState, subscribeAuthState } from '@/services/auth-state';
import { getResilienceScore, type ResilienceScoreResponse } from '@/services/resilience';
import { h, replaceChildren } from '@/utils/dom-utils';
import {
  RESILIENCE_VISUAL_LEVEL_COLORS,
  formatBaselineStress,
  formatResilienceMethodologyHelpTitle,
  formatResilienceScoreInterval,
  getResilienceOverallDisplay,
  getResilienceDomainLabel,
  getResilienceTrendArrow,
  getResilienceVisualLevel,
  shouldRenderResilienceBaselineStress,
} from './resilience-widget-utils';

function normalizeCountryCode(countryCode: string | null | undefined): string | null {
  const normalized = String(countryCode || '').trim().toUpperCase();
  return /^[A-Z]{2}$/.test(normalized) ? normalized : null;
}

function clampScore(score: number): number {
  if (!Number.isFinite(score)) return 0;
  return Math.min(100, Math.max(0, score));
}

const METHODOLOGY_HELP_TITLE = formatResilienceMethodologyHelpTitle();

export class ResilienceWidget {
  private readonly element: HTMLElement;
  private authState: AuthSession = getAuthState();
  private unsubscribeAuth: (() => void) | null = null;
  private currentCountryCode: string | null = null;
  private currentData: ResilienceScoreResponse | null = null;
  private loading = false;
  private errorMessage: string | null = null;
  private requestVersion = 0;

  constructor(countryCode?: string | null) {
    this.element = document.createElement('section');
    this.element.className = 'cdp-card resilience-widget';

    this.unsubscribeAuth = subscribeAuthState((state) => {
      this.authState = state;
      this.render();
    });

    this.setCountryCode(countryCode ?? null);
  }

  public getElement(): HTMLElement {
    return this.element;
  }

  public setCountryCode(countryCode: string | null): void {
    const normalized = normalizeCountryCode(countryCode);
    if (normalized === this.currentCountryCode) return;

    this.currentCountryCode = normalized;
    this.currentData = null;
    this.errorMessage = null;
    this.loading = false;
    this.requestVersion += 1;

    if (!normalized) {
      this.render();
      return;
    }

    void this.refresh();
  }

  public async refresh(): Promise<void> {
    if (!this.currentCountryCode) {
      this.render();
      return;
    }

    if (this.authState.isPending) {
      this.render();
      return;
    }

    const requestVersion = ++this.requestVersion;
    this.loading = true;
    this.errorMessage = null;
    this.render();

    try {
      const response = await getResilienceScore(this.currentCountryCode);
      if (requestVersion !== this.requestVersion) return;
      this.currentData = response;
      this.loading = false;
      this.errorMessage = null;
      this.render();
    } catch (error) {
      if (requestVersion !== this.requestVersion) return;
      this.loading = false;
      this.currentData = null;
      this.errorMessage = error instanceof Error ? error.message : 'Unable to load resilience score.';
      this.render();
    }
  }

  public destroy(): void {
    this.requestVersion += 1;
    this.unsubscribeAuth?.();
    this.unsubscribeAuth = null;
  }

  private render(): void {
    const body = this.renderBody();

    replaceChildren(
      this.element,
      h(
        'div',
        { className: 'resilience-widget__header' },
        h('h3', { className: 'cdp-card-title resilience-widget__title' }, 'Resilience Score'),
        h(
          'span',
          {
            className: 'resilience-widget__help',
            title: METHODOLOGY_HELP_TITLE,
            'aria-label': 'Resilience score methodology',
          },
          '?',
        ),
      ),
      body,
    );
  }

  private renderBody(): HTMLElement {
    if (!this.currentCountryCode) {
      return h('div', { className: 'cdp-card-body' }, this.makeEmpty('Resilience data loads when a country is selected.'));
    }

    if (this.loading) {
      return h('div', { className: 'cdp-card-body' }, this.makeLoading('Loading resilience score…'));
    }

    if (this.errorMessage) {
      return this.renderError(this.errorMessage);
    }

    if (!this.currentData) {
      return h('div', { className: 'cdp-card-body' }, this.makeEmpty('Resilience score unavailable.'));
    }

    return this.renderScoreCard(this.currentData);
  }

  private renderError(message: string): HTMLElement {
    return h(
      'div',
      { className: 'cdp-card-body resilience-widget__error' },
      h('div', { className: 'cdp-empty' }, message),
      h(
        'button',
        {
          type: 'button',
          className: 'cdp-action-btn resilience-widget__retry',
          onclick: () => void this.refresh(),
        },
        'Retry',
      ),
    );
  }

  private renderScoreCard(data: ResilienceScoreResponse): HTMLElement {
    const overallDisplay = getResilienceOverallDisplay(data);
    const levelColor = RESILIENCE_VISUAL_LEVEL_COLORS[overallDisplay.visualLevel];
    const scoreInterval = overallDisplay.hasScore ? formatResilienceScoreInterval(data.scoreInterval) : null;

    return h(
      'div',
      { className: 'cdp-card-body resilience-widget__body' },
      h(
        'div',
        { className: 'resilience-widget__overall' },
        this.renderBarBlock(
          overallDisplay.scoreForBar,
          levelColor,
          h(
            'div',
            { className: 'resilience-widget__overall-meta' },
            h('span', { className: 'resilience-widget__overall-score' }, overallDisplay.scoreLabel),
            ...(scoreInterval
              ? [h('span', {
                  className: 'resilience-widget__overall-interval',
                  title: scoreInterval.title,
                }, scoreInterval.label)]
              : []),
            h(
              'span',
              {
                className: 'resilience-widget__overall-level',
                style: { color: levelColor },
                title: overallDisplay.serverLevelLabel,
              },
              overallDisplay.visualLevelLabel,
            ),
            ...(overallDisplay.hasScore
              ? [h('span', { className: 'resilience-widget__overall-trend' }, `${getResilienceTrendArrow(data.trend)} ${data.trend}`)]
              : []),
          ),
        ),
      ),
      ...(shouldRenderResilienceBaselineStress(data, overallDisplay)
        ? [h(
            'div',
            { className: 'resilience-widget__baseline-stress' },
            h('span', { className: 'resilience-widget__baseline-stress-text' },
              formatBaselineStress(data.baselineScore, data.stressScore)),
          )]
        : []),
      h(
        'div',
        { className: 'resilience-widget__domains' },
        ...data.domains.map((domain) => this.renderDomainRow(domain)),
      ),
    );
  }

  private renderBarBlock(
    score: number,
    color: string,
    meta: HTMLElement,
  ): HTMLElement {
    const clamped = clampScore(score);
    return h(
      'div',
      { className: 'resilience-widget__bar-block' },
      h(
        'div',
        { className: 'resilience-widget__bar-wrap' },
        h('div', {
          className: 'resilience-widget__bar',
          style: { width: `${clamped}%`, backgroundColor: color },
        }),
      ),
      meta,
    );
  }

  private renderDomainRow(domain: { id: string; score?: number }): HTMLElement {
    const score = domain.score ?? 0;
    const visualLevel = getResilienceVisualLevel(score);
    const color = RESILIENCE_VISUAL_LEVEL_COLORS[visualLevel];

    const labelEl = document.createElement('div');
    labelEl.className = 'resilience-widget__domain-label';

    const nameEl = document.createElement('span');
    nameEl.className = 'resilience-widget__domain-name';
    nameEl.textContent = getResilienceDomainLabel(domain.id);
    labelEl.appendChild(nameEl);

    const scoreWrapEl = document.createElement('div');
    scoreWrapEl.className = 'resilience-widget__domain-score-wrap';

    if (Number.isFinite(domain.score ?? 0) && domain.score !== undefined) {
      const scoreDiv = document.createElement('div');
      scoreDiv.className = 'resilience-widget__domain-score';

      const barDiv = document.createElement('div');
      barDiv.className = 'resilience-widget__domain-bar';
      barDiv.style.width = `${clampScore(domain.score!)}%`;
      barDiv.style.backgroundColor = color;
      scoreWrapEl.appendChild(barDiv);

      const valueEl = document.createElement('span');
      valueEl.className = 'resilience-widget__domain-value';
      valueEl.textContent = String(Math.round(domain.score!));
      scoreWrapEl.appendChild(valueEl);
    } else {
      const naEl = document.createElement('span');
      naEl.className = 'resilience-widget__domain-na';
      naEl.textContent = '—';
      scoreWrapEl.appendChild(naEl);
    }

    const cellsEl = document.createElement('div');
    cellsEl.className = 'resilience-widget__domain-cells';
    cellsEl.appendChild(labelEl);
    cellsEl.appendChild(scoreWrapEl);

    const rowEl = document.createElement('div');
    rowEl.className = 'resilience-widget__domain-row';
    rowEl.appendChild(cellsEl);

    return rowEl;
  }

  private makeLoading(text: string): HTMLElement {
    return h('div', { className: 'cdp-loading-inline' }, text);
  }

  private makeEmpty(text: string): HTMLElement {
    return h('div', { className: 'cdp-empty' }, text);
  }
}