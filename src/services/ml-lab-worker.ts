/**
 * Worker Manager for the ML Lab experiments worker (`ml-lab.worker.ts`).
 * Provides a typed async interface to run the browser-side ML engine off the
 * main thread.
 *
 * Copied from the `analysis-worker.ts` manager pattern: lazy singleton init,
 * a `{type:'ready'}` handshake with a 10s timeout, request ids + per-request
 * timeouts, a `workerUnavailable` latch for graceful degradation, and
 * cleanup-on-error so the next call re-initializes a fresh worker instead of
 * posting into a dead one.
 */

import type { MLLabInputItem, MLLabResults } from '@/workers/ml-lab.worker';

// Import worker using Vite's worker syntax
import MLWorker from '@/workers/ml-lab.worker?worker';

export type {
  BiasVarianceReport,
  CandidateEliminationResult,
  ClassificationReport,
  ClusteringReport,
  ConfusionMatrix,
  EnsembleReport,
  FindSResult,
  MetricRow,
  MLLabInputItem,
  MLLabResults,
  PipelinePreview,
  RegressionReport,
  StatusEntry,
} from '@/workers/ml-lab.worker';

/**
 * Thrown by {@link mlLabWorker.runExperiments} when the ML lab worker cannot be
 * created (legacy browsers without module-worker support) or never signals
 * ready. The ML lab panel can catch this and degrade to a hidden/disabled lab.
 */
export class MLLabUnavailableError extends Error {
  constructor(message = 'ML lab worker unavailable') {
    super(message);
    this.name = 'MLLabUnavailableError';
    // Keep `instanceof` working after the error crosses a downleveled class
    // boundary (target ES2020 compiles classes to functions).
    Object.setPrototypeOf(this, MLLabUnavailableError.prototype);
  }
}

interface PendingRequest {
  resolve: (value: MLLabResults) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}

type WorkerResult =
  | { type: 'ready' }
  | { type: 'result'; id: string; results: MLLabResults }
  | { type: 'error'; id: string; error: string };

class MLLabWorkerManager {
  private worker: Worker | null = null;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private requestIdCounter = 0;
  private isReady = false;
  // Set true when worker construction throws synchronously (e.g. legacy
  // browsers without module-worker support). Latches so we don't re-attempt
  // construction on every call, and gates callers into graceful degradation.
  private workerUnavailable = false;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private readyTimeout: ReturnType<typeof setTimeout> | null = null;

  private static readonly READY_TIMEOUT_MS = 10000; // 10 seconds to become ready
  private static readonly REQUEST_TIMEOUT_MS = 60000; // 60 seconds per experiment run

  /**
   * Initialize the worker. Called lazily on first use.
   */
  private initWorker(): void {
    if (this.worker || this.workerUnavailable) return;

    this.readyPromise = new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    // Set ready timeout - reject if worker doesn't become ready in time
    this.readyTimeout = setTimeout(() => {
      if (!this.isReady) {
        const error = new Error('ML lab worker failed to become ready within timeout');
        console.error('[MLLabWorker]', error.message);
        this.readyReject?.(error);
        this.cleanup();
      }
    }, MLLabWorkerManager.READY_TIMEOUT_MS);

    try {
      this.worker = new MLWorker();
    } catch (error) {
      // Legacy browsers without module-worker support throw synchronously here.
      // Do NOT reject readyPromise: cleanup() nulls readyPromise in this same
      // synchronous tick, so waitForReady's `await this.readyPromise` would
      // await `null` and the rejection would be orphaned into an unhandled
      // promise rejection (same pitfall as AnalysisWorker WORLDMONITOR-V2/V6).
      // Instead latch `workerUnavailable`, resolve the ready gate, and let
      // runExperiments throw the typed MLLabUnavailableError.
      console.error('[MLLabWorker] Failed to create worker:', error);
      this.workerUnavailable = true;
      this.readyResolve?.();
      this.cleanup();
      return;
    }

    this.worker.onmessage = (event: MessageEvent<WorkerResult>) => {
      const data = event.data;

      if (data.type === 'ready') {
        this.isReady = true;
        if (this.readyTimeout) {
          clearTimeout(this.readyTimeout);
          this.readyTimeout = null;
        }
        this.readyResolve?.();
        return;
      }

      if (data.type === 'result') {
        const pending = this.pendingRequests.get(data.id);
        if (pending) {
          clearTimeout(pending.timeout);
          this.pendingRequests.delete(data.id);
          pending.resolve(data.results);
        }
        return;
      }

      if (data.type === 'error') {
        const pending = this.pendingRequests.get(data.id);
        if (pending) {
          clearTimeout(pending.timeout);
          this.pendingRequests.delete(data.id);
          pending.reject(new Error(data.error));
        }
      }
    };

    this.worker.onerror = (error) => {
      console.error('[MLLabWorker] Error:', error);

      // If not ready yet, reject the ready promise
      if (!this.isReady) {
        this.readyReject?.(new Error(`ML lab worker failed to initialize: ${error.message}`));
        this.cleanup();
        return;
      }

      // Reject all pending requests
      for (const [id, pending] of this.pendingRequests) {
        clearTimeout(pending.timeout);
        pending.reject(new Error(`ML lab worker error: ${error.message}`));
        this.pendingRequests.delete(id);
      }

      // The worker errored after becoming ready. Terminate it and reset state
      // so the next runExperiments call re-initializes a fresh worker instead
      // of posting into a dead one and hanging until the request timeout.
      this.cleanup();
    };
  }

  /**
   * Cleanup worker state (for re-initialization)
   */
  private cleanup(): void {
    if (this.readyTimeout) {
      clearTimeout(this.readyTimeout);
      this.readyTimeout = null;
    }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.isReady = false;
    this.readyPromise = null;
    this.readyResolve = null;
    this.readyReject = null;
  }

  /**
   * Wait for worker to be ready
   */
  private async waitForReady(): Promise<void> {
    this.initWorker();
    if (this.isReady) return;
    try {
      await this.readyPromise;
    } catch {
      // Ready timeout or worker init failure -> typed error for the panel.
      throw new MLLabUnavailableError('ML lab worker failed to initialize');
    }
  }

  /**
   * Generate unique request ID
   */
  private generateId(): string {
    return `ml-lab-${++this.requestIdCounter}-${Date.now()}`;
  }

  /**
   * Run the full ML experiment suite over a batch of news items.
   * Throws {@link MLLabUnavailableError} when workers are unsupported; other
   * rejections are request timeouts or worker runtime errors.
   */
  async runExperiments(items: MLLabInputItem[]): Promise<MLLabResults> {
    await this.waitForReady();
    if (!this.isReady || this.workerUnavailable) {
      throw new MLLabUnavailableError();
    }
    const worker = this.worker;
    if (!worker) throw new MLLabUnavailableError();

    return new Promise<MLLabResults>((resolve, reject) => {
      const id = this.generateId();
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        // A 60s run means the worker is wedged: tear it down so the next call
        // re-initializes a fresh worker instead of queueing behind this one.
        this.cleanup();
        reject(new Error('ML lab experiments timed out after 60s'));
      }, MLLabWorkerManager.REQUEST_TIMEOUT_MS);

      this.pendingRequests.set(id, { resolve, reject, timeout });
      worker.postMessage({ type: 'run', id, items });
    });
  }

  /**
   * Terminate worker (cleanup)
   */
  terminate(): void {
    // Reject all pending requests
    for (const [id, pending] of this.pendingRequests) {
      clearTimeout(pending.timeout);
      pending.reject(new MLLabUnavailableError('ML lab worker terminated'));
      this.pendingRequests.delete(id);
    }
    this.cleanup();
  }

  /**
   * Check if worker is available and ready
   */
  get ready(): boolean {
    return this.isReady;
  }
}

// Singleton instance
export const mlLabWorker = new MLLabWorkerManager();
