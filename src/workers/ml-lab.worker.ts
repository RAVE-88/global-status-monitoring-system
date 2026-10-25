/**
 * ML Lab Web Worker — the browser-side machine-learning engine for the
 * Global Status Monitoring System.
 *
 * Runs, for real and from scratch (no ML dependencies), on batches of news
 * headlines: preprocessing with near-duplicate removal, a committed AFINN-style
 * sentiment lexicon, the shared keyword threat classifier, derived transparent
 * labels, a zoo of classifiers (logistic regression, linear/RBF SVM, naive
 * Bayes, kNN, decision tree, random forest, AdaBoost, bagging), linear
 * regression via normal equations, clustering (k-means++, DBSCAN, agglomerative,
 * PCA via Jacobi eigen decomposition), honest evaluation (chronological 5-fold
 * CV, bootstrap CIs, bias/variance and regularization sweeps), Mitchell-style
 * concept learning (Find-S + Candidate Elimination), a transparent per-country
 * status engine, and a pipeline preview for the most recent headlines.
 *
 * Determinism contract: no `Math.random`, no `Date.now` — every stochastic
 * step (bootstrap, Pegasos sampling, bagging, k-means++ seeding) draws from a
 * committed-seed mulberry32 PRNG, so identical input produces identical output.
 * All message handling and heavy compute stays inside this worker.
 */

import { classifyByKeyword } from '../../shared/threat-keyword-classifier';

// ---------------------------------------------------------------------------
// Result types — single source of truth, re-exported by the worker manager.
// ---------------------------------------------------------------------------

export interface MetricRow {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number | null;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface ClassificationReport {
  task: 'risk_band' | 'high_risk';
  trainSize: number;
  testSize: number;
  rows: MetricRow[];
  confusion: Record<string, ConfusionMatrix>;
  crossVal: Record<string, { mean: number; std: number; folds: number }>;
  bootstrap: Record<string, { mean: number; lo: number; hi: number }>;
}

export interface RegressionReport {
  target: string;
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  samplePoints: { actual: number; predicted: number }[];
}

export interface ClusteringReport {
  pca: { points: { x: number; y: number; cluster: number; noise: boolean }[] };
  kmeans: { k: number; sizes: number[]; inertia: number; silhouette: number | null };
  dbscan: { eps: number; minPts: number; sizes: number[]; noiseCount: number };
  hierarchical: { sizes: number[]; silhouette: number | null };
}

export interface EnsembleReport {
  rows: MetricRow[];
  featureImportance: { feature: string; importance: number }[];
}

export interface BiasVarianceReport {
  sweep: { values: { depth: number; trainError: number; testError: number }[] };
}

export interface FindSResult {
  hypothesis: Record<string, string>;
  steps: { exampleIndex: number; hypothesis: Record<string, string> }[];
}

export interface CandidateEliminationResult {
  specific: Record<string, string>;
  general: Record<string, string>[];
  history: { step: number; specific: Record<string, string>; general: Record<string, string>[] }[];
  notes: string[];
}

export interface PipelinePreview {
  headline: string;
  sentiment: number;
  entities: string[];
  category: string;
  level: string;
  clusterId: number | null;
}

export interface StatusEntry {
  code: string;
  score: number;
  band: string;
  drivers: { label: string; value: number }[];
  events: number;
}

export interface MLLabResults {
  builtAtEpochMs: number;
  sourceCount: number;
  droppedCount: number;
  preprocess: {
    input: number;
    afterDedupe: number;
    imputed: number;
    features: string[];
    trainSize: number;
    testSize: number;
    classDistribution: Record<string, number>;
  };
  classification: ClassificationReport[];
  regression: RegressionReport;
  clustering: ClusteringReport;
  ensembles: EnsembleReport;
  biasVariance: BiasVarianceReport;
  regularization: { lambdas: number[]; trainAccuracy: number[]; testAccuracy: number[] };
  conceptLearning: { finds: FindSResult; ce: CandidateEliminationResult };
  pipelinePreview: PipelinePreview[];
  statusEngine: { countries: StatusEntry[]; methodology: string };
}

export interface MLLabInputItem {
  title: string;
  source: string;
  pubDate: number;
  isAlert: boolean;
  tier?: number;
  corroborationCount?: number;
  credibilityScore?: number;
  snippet?: string;
}

// ---------------------------------------------------------------------------
// Worker protocol.
// ---------------------------------------------------------------------------

interface RunMessage {
  type: 'run';
  id: string;
  items: MLLabInputItem[];
}

type WorkerOut =
  | { type: 'ready' }
  | { type: 'result'; id: string; results: MLLabResults }
  | { type: 'error'; id: string; error: string };

// ---------------------------------------------------------------------------
// Committed constants.
// ---------------------------------------------------------------------------

/** Hard cap on items per run; the most recent items are kept. */
const MAX_ITEMS = 600;

/** Ordered risk bands (canonical, ascending severity). */
const BANDS = ['low', 'medium', 'high', 'critical'] as const;

/** Ordinal encoding of the shared classifier's threat levels. */
const LEVEL_ORDINAL: Record<string, number> = { info: 0, low: 1, medium: 2, high: 3, critical: 4 };

/** Transparent level weights for the derived risk score. */
const LEVEL_WEIGHT: Record<string, number> = { critical: 60, high: 40, medium: 20, low: 8, info: 2 };

/** Feature order for every model; d = 14. */
const FEATURE_NAMES = [
  'sentiment',
  'conflictFlag',
  'militaryFlag',
  'disasterFlag',
  'protestFlag',
  'economicFlag',
  'cyberFlag',
  'levelOrdinal',
  'textLen',
  'wordCount',
  'upperRatio',
  'corroboration',
  'isBreaking',
  'hourOfDay',
];

const FEATURE_COUNT = FEATURE_NAMES.length;

/** Stopwords for the content token set (Jaccard dedupe only). */
const STOPWORDS = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was',
  'one', 'our', 'out', 'get', 'has', 'his', 'how', 'its', 'new', 'now', 'say',
  'she', 'too', 'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from',
  'they', 'been', 'more', 'than', 'them', 'were', 'what', 'into', 'over',
  'such', 'also', 'who', 'when', 'where', 'while', 'about', 'after', 'just',
]);

/** Negators flip the sign of a nearby sentiment-bearing token. */
const NEGATORS = new Set([
  'no', 'not', 'never', 'none', 'nothing', 'neither', 'nor', 'without',
  'hardly', 'barely', 'deny', 'denies', 'denied', 'lacks', 'lacking',
]);

/**
 * Committed compact AFINN-style lexicon (~120 stems, score -4..4). Matching is
 * prefix-based for stems of length >= 4 (so "attack" hits attacks/attacked/
 * attacking) and exact otherwise (so "war" does not hit "warrant").
 */
const LEXICON_RAW = [
  // positive
  'excellent:4', 'best:3', 'good:3', 'great:3', 'love:3', 'happy:3', 'hope:2',
  'gain:2', 'growth:3', 'profit:3', 'success:3', 'win:3', 'wins:3', 'won:3',
  'victor:3', 'triumph:4', 'breakthrough:3', 'strong:2', 'boost:2',
  'improve:2', 'recover:2', 'rebound:2', 'peace:3', 'agree:2', 'deal:2',
  'support:2', 'backed:2', 'praise:3', 'celebrat:3', 'rally:2', 'upbeat:2',
  'positive:2', 'progress:3', 'advance:2', 'safe:2', 'rescue:3', 'relief:2',
  'aid:1', 'reopen:2', 'stabiliz:2', 'upgrad:2', 'surge:2', 'thrive:3',
  // negative
  'bad:-3', 'worst:-3', 'worse:-3', 'poor:-2', 'weak:-2', 'decline:-2',
  'drop:-2', 'fall:-2', 'fell:-2', 'crash:-4', 'plunge:-4', 'slump:-3',
  'crisis:-3', 'fear:-2', 'threat:-2', 'threaten:-3', 'risk:-2', 'danger:-3',
  'warn:-2', 'concern:-2', 'worry:-2', 'fail:-3', 'collapse:-3', 'chaos:-3',
  'corrupt:-3', 'scandal:-3', 'protest:-2', 'riot:-3', 'unrest:-2',
  'violenc:-3', 'war:-3', 'conflict:-2', 'fight:-2', 'clash:-2', 'attack:-3',
  'assault:-3', 'invas:-4', 'bomb:-3', 'explosion:-3', 'blast:-3',
  'missile:-2', 'kill:-3', 'dead:-3', 'death:-3', 'casualt:-3', 'injur:-2',
  'wound:-2', 'hostage:-3', 'terror:-3', 'sanction:-2', 'embargo:-2',
  'recession:-3', 'inflation:-2', 'unemploy:-2', 'layoff:-2', 'bankrupt:-3',
  'outbreak:-3', 'epidemic:-3', 'pandemic:-3', 'virus:-2', 'infect:-2',
  'famine:-4', 'drought:-2', 'flood:-2', 'earthquake:-3', 'tsunami:-4',
  'hurricane:-3', 'typhoon:-3', 'wildfire:-2', 'landslide:-2', 'blackout:-2',
  'outage:-2', 'breach:-2', 'hack:-2', 'ransomware:-3', 'cyberattack:-3',
  'leak:-2', 'turmoil:-3', 'panic:-3', 'emergenc:-2', 'evacuat:-2',
  'disaster:-3', 'catastroph:-4',
].join(' ');

const LEXICON: Array<[string, number]> = LEXICON_RAW.split(/\s+/).map(pair => {
  const sep = pair.indexOf(':');
  return [pair.slice(0, sep), Number(pair.slice(sep + 1))];
});

/**
 * Committed gazetteer: major countries + capitals (alias -> ISO2). Aliases may
 * contain spaces (matched as consecutive word tokens); the first alias listed
 * for a code doubles as its display name via ISO2_PRIMARY below.
 */
const GAZETTEER_RAW = [
  'united states|US', 'america|US', 'usa|US', 'us|US', 'washington dc|US',
  'united kingdom|GB', 'britain|GB', 'uk|GB', 'england|GB', 'scotland|GB',
  'london|GB', 'canada|CA', 'ottawa|CA', 'mexico|MX', 'mexico city|MX',
  'france|FR', 'paris|FR', 'germany|DE', 'berlin|DE', 'italy|IT', 'rome|IT',
  'spain|ES', 'madrid|ES', 'portugal|PT', 'lisbon|PT', 'ireland|IE',
  'dublin|IE', 'netherlands|NL', 'amsterdam|NL', 'belgium|BE', 'brussels|BE',
  'switzerland|CH', 'geneva|CH', 'austria|AT', 'vienna|AT', 'sweden|SE',
  'stockholm|SE', 'norway|NO', 'oslo|NO', 'denmark|DK', 'copenhagen|DK',
  'finland|FI', 'helsinki|FI', 'poland|PL', 'warsaw|PL', 'czechia|CZ',
  'prague|CZ', 'hungary|HU', 'budapest|HU', 'romania|RO', 'bucharest|RO',
  'bulgaria|BG', 'sofia|BG', 'greece|GR', 'athens|GR', 'turkey|TR',
  'ankara|TR', 'istanbul|TR', 'russia|RU', 'moscow|RU', 'st petersburg|RU',
  'ukraine|UA', 'kyiv|UA', 'belarus|BY', 'minsk|BY', 'moldova|MD',
  'serbia|RS', 'belgrade|RS', 'croatia|HR', 'zagreb|HR', 'bosnia|BA',
  'sarajevo|BA', 'albania|AL', 'tirana|AL', 'lithuania|LT', 'latvia|LV',
  'estonia|EE', 'china|CN', 'beijing|CN', 'shanghai|CN', 'taiwan|TW',
  'taipei|TW', 'japan|JP', 'tokyo|JP', 'south korea|KR', 'korea|KR',
  'seoul|KR', 'north korea|KP', 'pyongyang|KP', 'india|IN', 'new delhi|IN',
  'delhi|IN', 'pakistan|PK', 'islamabad|PK', 'karachi|PK', 'bangladesh|BD',
  'dhaka|BD', 'sri lanka|LK', 'colombo|LK', 'nepal|NP', 'kathmandu|NP',
  'afghanistan|AF', 'kabul|AF', 'kazakhstan|KZ', 'astana|KZ',
  'uzbekistan|UZ', 'tashkent|UZ', 'iran|IR', 'tehran|IR', 'iraq|IQ',
  'baghdad|IQ', 'syria|SY', 'damascus|SY', 'lebanon|LB', 'beirut|LB',
  'israel|IL', 'jerusalem|IL', 'tel aviv|IL', 'gaza|PS', 'west bank|PS',
  'palestine|PS', 'jordan|JO', 'amman|JO', 'saudi arabia|SA', 'riyadh|SA',
  'uae|AE', 'emirates|AE', 'dubai|AE', 'abu dhabi|AE', 'qatar|QA', 'doha|QA',
  'kuwait|KW', 'oman|OM', 'muscat|OM', 'yemen|YE', 'sanaa|YE', 'egypt|EG',
  'cairo|EG', 'libya|LY', 'tripoli|LY', 'tunisia|TN', 'tunis|TN',
  'algeria|DZ', 'algiers|DZ', 'morocco|MA', 'rabat|MA', 'casablanca|MA',
  'sudan|SD', 'khartoum|SD', 'south sudan|SS', 'juba|SS', 'ethiopia|ET',
  'addis ababa|ET', 'eritrea|ER', 'somalia|SO', 'mogadishu|SO', 'kenya|KE',
  'nairobi|KE', 'tanzania|TZ', 'uganda|UG', 'kampala|UG', 'rwanda|RW',
  'kigali|RW', 'dr congo|CD', 'kinshasa|CD', 'bangui|CF', 'cameroon|CM',
  'nigeria|NG', 'abuja|NG', 'lagos|NG', 'ghana|GH', 'accra|GH',
  'ivory coast|CI', 'abidjan|CI', 'senegal|SN', 'dakar|SN', 'mali|ML',
  'bamako|ML', 'burkina faso|BF', 'ouagadougou|BF', 'niger|NE', 'niamey|NE',
  'chad|TD', 'ndjamena|TD', 'south africa|ZA', 'pretoria|ZA',
  'johannesburg|ZA', 'zimbabwe|ZW', 'harare|ZW', 'mozambique|MZ',
  'maputo|MZ', 'angola|AO', 'luanda|AO', 'brazil|BR', 'brasilia|BR',
  'buenos aires|AR', 'argentina|AR', 'chile|CL', 'santiago|CL',
  'colombia|CO', 'bogota|CO', 'venezuela|VE', 'caracas|VE', 'peru|PE',
  'lima|PE', 'bolivia|BO', 'la paz|BO', 'ecuador|EC', 'quito|EC', 'cuba|CU',
  'havana|CU', 'haiti|HT', 'australia|AU', 'canberra|AU', 'sydney|AU',
  'new zealand|NZ', 'wellington|NZ', 'indonesia|ID', 'jakarta|ID',
  'malaysia|MY', 'kuala lumpur|MY', 'singapore|SG', 'thailand|TH',
  'bangkok|TH', 'vietnam|VN', 'hanoi|VN', 'philippines|PH', 'manila|PH',
  'myanmar|MM', 'yangon|MM', 'burma|MM', 'cambodia|KH', 'armenia|AM',
  'yerevan|AM', 'azerbaijan|AZ', 'baku|AZ',
].join(',');

const GAZETTEER = new Map<string, string>();
for (const pair of GAZETTEER_RAW.split(',')) {
  const bar = pair.indexOf('|');
  if (bar > 0) GAZETTEER.set(pair.slice(0, bar), pair.slice(bar + 1));
}

/** ISO2 -> display name for the pipeline preview's entity list. */
const ISO2_PRIMARY_RAW = [
  'US:United States', 'GB:United Kingdom', 'CA:Canada', 'MX:Mexico',
  'FR:France', 'DE:Germany', 'IT:Italy', 'ES:Spain', 'PT:Portugal',
  'IE:Ireland', 'NL:Netherlands', 'BE:Belgium', 'CH:Switzerland',
  'AT:Austria', 'SE:Sweden', 'NO:Norway', 'DK:Denmark', 'FI:Finland',
  'PL:Poland', 'CZ:Czechia', 'HU:Hungary', 'RO:Romania', 'BG:Bulgaria',
  'GR:Greece', 'TR:Turkey', 'RU:Russia', 'UA:Ukraine', 'BY:Belarus',
  'MD:Moldova', 'RS:Serbia', 'HR:Croatia', 'BA:Bosnia', 'AL:Albania',
  'LT:Lithuania', 'LV:Latvia', 'EE:Estonia', 'CN:China', 'TW:Taiwan',
  'JP:Japan', 'KR:South Korea', 'KP:North Korea', 'IN:India', 'PK:Pakistan',
  'BD:Bangladesh', 'LK:Sri Lanka', 'NP:Nepal', 'AF:Afghanistan',
  'KZ:Kazakhstan', 'UZ:Uzbekistan', 'IR:Iran', 'IQ:Iraq', 'SY:Syria',
  'LB:Lebanon', 'IL:Israel', 'PS:Palestine', 'JO:Jordan', 'SA:Saudi Arabia',
  'AE:UAE', 'QA:Qatar', 'KW:Kuwait', 'OM:Oman', 'YE:Yemen', 'EG:Egypt',
  'LY:Libya', 'TN:Tunisia', 'DZ:Algeria', 'MA:Morocco', 'SD:Sudan',
  'SS:South Sudan', 'ET:Ethiopia', 'ER:Eritrea', 'SO:Somalia', 'KE:Kenya',
  'TZ:Tanzania', 'UG:Uganda', 'RW:Rwanda', 'CD:DR Congo',
  'CF:Central African Rep.', 'CM:Cameroon', 'NG:Nigeria', 'GH:Ghana',
  'CI:Ivory Coast', 'SN:Senegal', 'ML:Mali', 'BF:Burkina Faso', 'NE:Niger',
  'TD:Chad', 'ZA:South Africa', 'ZW:Zimbabwe', 'MZ:Mozambique', 'AO:Angola',
  'BR:Brazil', 'AR:Argentina', 'CL:Chile', 'CO:Colombia', 'VE:Venezuela',
  'PE:Peru', 'BO:Bolivia', 'EC:Ecuador', 'CU:Cuba', 'HT:Haiti',
  'AU:Australia', 'NZ:New Zealand', 'ID:Indonesia', 'MY:Malaysia',
  'SG:Singapore', 'TH:Thailand', 'VN:Vietnam', 'PH:Philippines',
  'MM:Myanmar', 'KH:Cambodia', 'AM:Armenia', 'AZ:Azerbaijan',
].join(',');

const ISO2_PRIMARY = new Map<string, string>();
for (const pair of ISO2_PRIMARY_RAW.split(',')) {
  const sep = pair.indexOf(':');
  if (sep > 0) ISO2_PRIMARY.set(pair.slice(0, sep), pair.slice(sep + 1));
}

/**
 * Committed editorial snapshot of regions with active armed conflict; drives
 * the small regionConflict bonus in the transparent risk formula.
 */
const CONFLICT_REGION_ISO2 = new Set([
  'UA', 'RU', 'IL', 'PS', 'SY', 'IQ', 'IR', 'YE', 'SD', 'SS', 'AF', 'PK',
  'SO', 'LY', 'ET', 'CD', 'CF', 'ML', 'HT', 'MM', 'KP', 'VE', 'NG', 'CM',
  'BF', 'MZ', 'CO', 'MX', 'LB',
]);

/** Committed hyperparameters. */
const LOGISTIC_LAMBDA = 0.01;
const LOGISTIC_LR = 0.1;
const LOGISTIC_ITERS = 300;
const PEGASOS_LAMBDA = 0.01;
const PEGASOS_STEPS = 2000;
const RBF_GAMMA = 0.5;
const RBF_STEPS = 1000;
const RBF_MAX_TRAIN = 200;
const KNN_K = 5;
const RF_TREES = 25;
const RF_MAX_DEPTH = 8;
const BOOST_ROUNDS = 25;
const KMEANS_ITERS = 20;
const DBSCAN_MIN_PTS = 5;
const HIER_SAMPLE = 150;
const BOOTSTRAP_ROUNDS = 200;
const CV_FOLDS = 5;
const REG_LAMBDAS = [0.001, 0.01, 0.1, 1, 10, 100];

const STATUS_METHODOLOGY =
  'Per-country Global Status (0-100) is a transparent weighted sum: ' +
  '0.30 x event volume (share of the busiest country in the window) + ' +
  '0.25 x conflict/military event share + 0.15 x mean |headline sentiment| + ' +
  '0.15 x peak corroboration (capped at 10 sources) + 0.15 x alert share. ' +
  'Countries are detected in titles and snippets with a committed gazetteer of ' +
  'major countries and capitals matched on word boundaries. ' +
  'Bands: low < 25, medium < 50, high < 75, critical >= 75.';

// ---------------------------------------------------------------------------
// Small math helpers (all deterministic).
// ---------------------------------------------------------------------------

/** mulberry32 — tiny seeded PRNG; stands in for Math.random everywhere. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function roundTo(x: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(x * factor) / factor;
}

function meanOf(values: number[]): number {
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function stdOf(values: number[]): number {
  if (values.length < 2) return 0;
  const m = meanOf(values);
  return Math.sqrt(values.reduce((acc, v) => acc + (v - m) * (v - m), 0) / (values.length - 1));
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-clamp(z, -35, 35)));
}

/** Squared Euclidean distance; callers sqrt() only when they need true distance. */
function dist2(a: number[], b: number[]): number {
  let sum = 0;
  for (let j = 0; j < a.length; j++) {
    const diff = a[j]! - b[j]!;
    sum += diff * diff;
  }
  return sum;
}

function argmaxIndex(values: number[]): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }
  return best;
}

/** Deterministic "evenly spaced" index sample covering [0, n). */
function evenIndices(n: number, m: number): number[] {
  if (n === 0) return [];
  if (m >= n) return Array.from({ length: n }, (_, i) => i);
  if (m <= 1) return [Math.floor((n - 1) / 2)];
  return Array.from({ length: m }, (_, i) => Math.floor((i * (n - 1)) / (m - 1)));
}

function randomSubset(n: number, m: number, rng: () => number): number[] {
  const pool = Array.from({ length: n }, (_, i) => i);
  const out: number[] = [];
  const take = Math.max(0, Math.min(m, n));
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(rng() * (n - i));
    const tmp = pool[i]!;
    pool[i] = pool[j]!;
    pool[j] = tmp;
    out.push(pool[i]!);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Preprocessing.
// ---------------------------------------------------------------------------

/** Raw lowercase word tokens (kept unfiltered for sentiment + negation). */
function rawTokens(text: string): string[] {
  return text.toLowerCase().match(/[a-z]+/g) ?? [];
}

/** Content token set: lowercase words, length >= 3, stopwords dropped. */
function contentTokenSet(text: string): Set<string> {
  const set = new Set<string>();
  for (const token of rawTokens(text)) {
    if (token.length >= 3 && !STOPWORDS.has(token)) set.add(token);
  }
  return set;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let intersection = 0;
  for (const token of a) if (b.has(token)) intersection++;
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/** Longest-prefix lexicon lookup (prefix rule only for stems of length >= 4). */
function lexiconScore(token: string): number {
  let score = 0;
  let bestLen = 0;
  for (const [stem, value] of LEXICON) {
    if (token === stem || (stem.length >= 4 && token.startsWith(stem))) {
      if (stem.length > bestLen) {
        score = value;
        bestLen = stem.length;
      }
    }
  }
  return score;
}

/** AFINN-style score for a raw token stream: negation flip + tanh to [-1, 1]. */
function scoreSentiment(tokens: string[]): number {
  let raw = 0;
  for (let i = 0; i < tokens.length; i++) {
    const value = lexiconScore(tokens[i]!);
    if (value === 0) continue;
    let sign = 1;
    for (let back = 1; back <= 3 && i - back >= 0; back++) {
      if (NEGATORS.has(tokens[i - back]!)) {
        sign = -1;
        break;
      }
    }
    raw += sign * value;
  }
  return Math.tanh(raw / 4);
}

/** Word-boundary country detection over unigram/bigram gazetteer aliases. */
function detectCountries(text: string): string[] {
  const found: string[] = [];
  const tokens = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const pushUnique = (iso: string): void => {
    if (!found.includes(iso)) found.push(iso);
  };
  for (let i = 0; i < tokens.length; i++) {
    const current = tokens[i]!;
    const next = tokens[i + 1];
    if (next !== undefined) {
      const iso = GAZETTEER.get(current + ' ' + next);
      if (iso) {
        pushUnique(iso);
        i++;
        continue;
      }
    }
    const single = GAZETTEER.get(current);
    if (single) pushUnique(single);
  }
  return found;
}

function utcDayKey(epochMs: number): number {
  return Math.floor(safeMs(epochMs) / 86400000);
}

function safeMs(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

interface ProcessedItem {
  index: number;
  title: string;
  epochMs: number;
  isAlert: boolean;
  tier: number | null;
  corroboration: number;
  sentiment: number;
  category: string;
  level: string;
  flags: {
    conflict: boolean;
    military: boolean;
    disaster: boolean;
    protest: boolean;
    economic: boolean;
    cyber: boolean;
  };
  isBreaking: boolean;
  countries: string[];
  textLen: number;
  wordCount: number;
  upperRatio: number;
  hourOfDay: number;
  tokenSet: Set<string>;
  riskScore: number;
  band: string;
  highRisk: boolean;
}

function computeRisk(
  level: string,
  category: string,
  sentiment: number,
  corroboration: number,
  isBreaking: boolean,
  regionConflict: boolean
): { riskScore: number; band: string; highRisk: boolean } {
  let score = LEVEL_WEIGHT[level] ?? 2;
  score += corroboration >= 3 ? 8 : corroboration === 2 ? 4 : 0;
  score += isBreaking ? 6 : 0;
  score += 4 * Math.abs(sentiment);
  const conflictFamily =
    category === 'conflict' || category === 'military' || category === 'terrorism';
  if (regionConflict && conflictFamily) score += 5;
  const riskScore = clamp(score, 0, 100);
  const band =
    riskScore >= 75 ? 'critical' : riskScore >= 50 ? 'high' : riskScore >= 25 ? 'medium' : 'low';
  return { riskScore, band, highRisk: band === 'high' || band === 'critical' };
}

interface IndexedItem {
  it: MLLabInputItem;
  i: number;
}

/**
 * Preprocess a raw batch: sort chronologically, assemble features, then remove
 * near-duplicate titles (Jaccard >= 0.9 within the same UTC day, keeping the
 * earlier item). Returns the processed items plus the number of duplicates
 * dropped.
 */
function preprocessBatch(
  items: IndexedItem[]
): { processed: ProcessedItem[]; dedupeDropped: number; imputed: number } {
  const ascending = items
    .slice()
    .sort((a, b) => safeMs(a.it.pubDate) - safeMs(b.it.pubDate) || a.i - b.i);

  const corroborationValues = ascending
    .map(entry => entry.it.corroborationCount)
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  const corroborationMedian =
    corroborationValues.length > 0 ? medianOf(corroborationValues) : 1;
  let imputed = 0;

  const keptTokens: Array<{ dayKey: number; tokens: Set<string> }> = [];
  const processed: ProcessedItem[] = [];
  let dedupeDropped = 0;

  for (const entry of ascending) {
    const item = entry.it;
    const titleTokens = rawTokens(item.title);
    const tokenSet = contentTokenSet(item.title);
    const dayKey = utcDayKey(item.pubDate);

    let duplicate = false;
    for (const kept of keptTokens) {
      if (kept.dayKey === dayKey && jaccard(kept.tokens, tokenSet) >= 0.9) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      dedupeDropped++;
      continue;
    }
    keptTokens.push({ dayKey, tokens: tokenSet });

    const classification = classifyByKeyword(item.title);
    const category = classification.category;
    const letters = item.title.match(/[A-Za-z]/g) ?? [];
    let upperCount = 0;
    for (const ch of letters) if (ch >= 'A' && ch <= 'Z') upperCount++;
    const lowerTitle = item.title.toLowerCase();
    const hasCorroboration =
      typeof item.corroborationCount === 'number' && Number.isFinite(item.corroborationCount);
    const corroboration =
      typeof item.corroborationCount === 'number' && Number.isFinite(item.corroborationCount)
        ? item.corroborationCount
        : corroborationMedian;
    if (!hasCorroboration) imputed++;
    const countries = detectCountries(item.title + ' ' + (item.snippet ?? ''));
    const regionConflict = countries.some(code => CONFLICT_REGION_ISO2.has(code));
    const isBreaking = /breaking|just in|urgent/.test(lowerTitle);
    const sentiment = scoreSentiment(titleTokens);
    const risk = computeRisk(
      classification.level,
      category,
      sentiment,
      corroboration,
      isBreaking,
      regionConflict
    );

    processed.push({
      index: processed.length,
      title: item.title,
      epochMs: safeMs(item.pubDate),
      isAlert: item.isAlert,
      tier: typeof item.tier === 'number' && Number.isFinite(item.tier) ? item.tier : null,
      corroboration,
      sentiment,
      category,
      level: classification.level,
      flags: {
        conflict: category === 'conflict' || category === 'terrorism',
        military: category === 'military',
        disaster: category === 'disaster',
        protest: category === 'protest',
        economic: category === 'economic',
        cyber: category === 'cyber',
      },
      isBreaking,
      countries,
      textLen: item.title.length,
      wordCount: titleTokens.length,
      upperRatio: letters.length > 0 ? upperCount / letters.length : 0,
      hourOfDay: new Date(safeMs(item.pubDate)).getUTCHours(),
      tokenSet,
      riskScore: risk.riskScore,
      band: risk.band,
      highRisk: risk.highRisk,
    });
  }

  return { processed, dedupeDropped, imputed };
}

function featureVector(item: ProcessedItem): number[] {
  return [
    item.sentiment,
    item.flags.conflict ? 1 : 0,
    item.flags.military ? 1 : 0,
    item.flags.disaster ? 1 : 0,
    item.flags.protest ? 1 : 0,
    item.flags.economic ? 1 : 0,
    item.flags.cyber ? 1 : 0,
    LEVEL_ORDINAL[item.level] ?? 0,
    item.textLen,
    item.wordCount,
    item.upperRatio,
    item.corroboration,
    item.isBreaking ? 1 : 0,
    item.hourOfDay,
  ];
}

// ---------------------------------------------------------------------------
// Standardization (fit on train only).
// ---------------------------------------------------------------------------

interface Standardizer {
  mean: number[];
  std: number[];
}

function fitStandardizer(rows: number[][]): Standardizer {
  const mean = new Array<number>(FEATURE_COUNT).fill(0);
  const std = new Array<number>(FEATURE_COUNT).fill(0);
  const n = rows.length;
  if (n === 0) {
    for (let j = 0; j < FEATURE_COUNT; j++) std[j] = 1;
    return { mean, std };
  }
  for (const row of rows) {
    for (let j = 0; j < FEATURE_COUNT; j++) mean[j]! += row[j] ?? 0;
  }
  for (let j = 0; j < FEATURE_COUNT; j++) mean[j] = mean[j]! / n;
  for (const row of rows) {
    for (let j = 0; j < FEATURE_COUNT; j++) {
      const diff = (row[j] ?? 0) - mean[j]!;
      std[j]! += diff * diff;
    }
  }
  for (let j = 0; j < FEATURE_COUNT; j++) {
    const value = n > 1 ? Math.sqrt(std[j]! / (n - 1)) : 0;
    std[j] = value < 1e-9 ? 1 : value;
  }
  return { mean, std };
}

function applyStandardizer(rows: number[][], z: Standardizer): number[][] {
  return rows.map(row =>
    FEATURE_NAMES.map((_, j) => ((row[j] ?? 0) - z.mean[j]!) / z.std[j]!)
  );
}

// ---------------------------------------------------------------------------
// Classification models.
// ---------------------------------------------------------------------------

interface ClassModel {
  fit(X: number[][], y: number[], k: number): void;
  /** Per-class scores; predict = argmax (first index wins ties). */
  scores(x: number[]): number[];
}

function predictClass(model: ClassModel, x: number[]): number {
  return argmaxIndex(model.scores(x));
}

/** Batch-gradient-descent logistic regression with L2, one-vs-rest. */
class LogisticOvR implements ClassModel {
  private learners: Array<{ w: number[]; b: number }> = [];
  private k = 0;

  constructor(
    private readonly lambda: number,
    private readonly lr: number = LOGISTIC_LR,
    private readonly iters: number = LOGISTIC_ITERS
  ) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.learners = [];
    const n = X.length;
    const denom = Math.max(n, 1);
    for (let c = 0; c < k; c++) {
      const w = new Array<number>(FEATURE_COUNT).fill(0);
      let b = 0;
      for (let iter = 0; iter < this.iters; iter++) {
        const gradW = new Array<number>(FEATURE_COUNT).fill(0);
        let gradB = 0;
        for (let i = 0; i < n; i++) {
          const xi = X[i]!;
          let z = b;
          for (let j = 0; j < FEATURE_COUNT; j++) z += w[j]! * xi[j]!;
          const err = sigmoid(z) - (y[i]! === c ? 1 : 0);
          for (let j = 0; j < FEATURE_COUNT; j++) gradW[j]! += err * xi[j]!;
          gradB += err;
        }
        for (let j = 0; j < FEATURE_COUNT; j++) {
          w[j]! -= this.lr * (gradW[j]! / denom + this.lambda * w[j]!);
        }
        b -= this.lr * (gradB / denom);
      }
      this.learners.push({ w, b });
    }
  }

  scores(x: number[]): number[] {
    return this.learners.map(l => {
      let z = l.b;
      for (let j = 0; j < x.length; j++) z += l.w[j]! * x[j]!;
      return z;
    });
  }

  get classCount(): number {
    return this.k;
  }
}

/** Pegasos hinge-loss SGD linear SVM, one-vs-rest. */
class LinearSVM implements ClassModel {
  private learners: Array<{ w: number[]; b: number }> = [];
  private k = 0;

  constructor(
    private readonly seed: number,
    private readonly lambda: number = PEGASOS_LAMBDA,
    private readonly steps: number = PEGASOS_STEPS
  ) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.learners = [];
    const n = X.length;
    for (let c = 0; c < k; c++) {
      const rng = mulberry32(this.seed + c * 7919 + 1);
      const w = new Array<number>(FEATURE_COUNT).fill(0);
      let b = 0;
      for (let t = 1; t <= this.steps; t++) {
        const i = Math.min(Math.max(n, 1) - 1, Math.floor(rng() * Math.max(n, 1)));
        const xi = X[i]!;
        const yi = y[i]! === c ? 1 : -1;
        let margin = b;
        for (let j = 0; j < FEATURE_COUNT; j++) margin += w[j]! * xi[j]!;
        const eta = 1 / (this.lambda * t);
        const scale = 1 - eta * this.lambda;
        if (yi * margin < 1) {
          for (let j = 0; j < FEATURE_COUNT; j++) w[j] = w[j]! * scale + eta * yi * xi[j]!;
          b = b * scale + eta * yi;
        } else {
          for (let j = 0; j < FEATURE_COUNT; j++) w[j] = w[j]! * scale;
          b *= scale;
        }
      }
      this.learners.push({ w, b });
    }
  }

  scores(x: number[]): number[] {
    return this.learners.map(l => {
      let z = l.b;
      for (let j = 0; j < x.length; j++) z += l.w[j]! * x[j]!;
      return z;
    });
  }

  get classCount(): number {
    return this.k;
  }
}

/** Kernelized (RBF) Pegasos on a deterministic 200-point subsample, OvR. */
class RBFSVM implements ClassModel {
  private learners: Array<{ svX: number[][]; svY: number[]; alpha: number[] }> = [];
  private k = 0;

  constructor(
    private readonly seed: number,
    private readonly gamma: number = RBF_GAMMA,
    private readonly lambda: number = PEGASOS_LAMBDA,
    private readonly steps: number = RBF_STEPS,
    private readonly maxTrain: number = RBF_MAX_TRAIN
  ) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.learners = [];
    const n = X.length;
    const subIdx = evenIndices(n, Math.min(this.maxTrain, n));
    const sub = subIdx.map(i => X[i]!);
    const m = sub.length;
    for (let c = 0; c < k; c++) {
      const rng = mulberry32(this.seed + c * 104729 + 3);
      const svX: number[][] = [];
      const svY: number[] = [];
      const alpha: number[] = [];
      for (let t = 1; t <= this.steps && m > 0; t++) {
        const i = Math.min(m - 1, Math.floor(rng() * m));
        const xi = sub[i]!;
        const yi = y[subIdx[i]!]! === c ? 1 : -1;
        let f = 0;
        for (let s = 0; s < svX.length; s++) {
          f += alpha[s]! * svY[s]! * Math.exp(-this.gamma * dist2(svX[s]!, xi));
        }
        if (yi * f < 1) {
          svX.push(xi);
          svY.push(yi);
          alpha.push(1 / (this.lambda * t));
        }
      }
      this.learners.push({ svX, svY, alpha });
    }
  }

  scores(x: number[]): number[] {
    return this.learners.map(l => {
      let f = 0;
      for (let s = 0; s < l.svX.length; s++) {
        f += l.alpha[s]! * l.svY[s]! * Math.exp(-this.gamma * dist2(l.svX[s]!, x));
      }
      return f;
    });
  }

  get classCount(): number {
    return this.k;
  }
}

/** Gaussian naive Bayes: per-class mean/variance + log posterior. */
class GaussianNBModel implements ClassModel {
  private means: number[][] = [];
  private variances: number[][] = [];
  private priors: number[] = [];
  private k = 0;

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    const n = X.length;
    const count = new Array<number>(k).fill(0);
    const sums: number[][] = Array.from({ length: k }, () => new Array<number>(FEATURE_COUNT).fill(0));
    for (let i = 0; i < n; i++) {
      const c = y[i]!;
      count[c]!++;
      const row = X[i]!;
      const sum = sums[c]!;
      for (let j = 0; j < FEATURE_COUNT; j++) sum[j]! += row[j]!;
    }
    this.means = [];
    this.priors = [];
    for (let c = 0; c < k; c++) {
      const mean = new Array<number>(FEATURE_COUNT).fill(0);
      const denom = Math.max(count[c]!, 1);
      for (let j = 0; j < FEATURE_COUNT; j++) mean[j] = sums[c]![j]! / denom;
      this.means.push(mean);
      this.priors.push(count[c]! / Math.max(n, 1));
    }
    const squared: number[][] = Array.from({ length: k }, () => new Array<number>(FEATURE_COUNT).fill(0));
    for (let i = 0; i < n; i++) {
      const c = y[i]!;
      const row = X[i]!;
      const mean = this.means[c]!;
      const acc = squared[c]!;
      for (let j = 0; j < FEATURE_COUNT; j++) {
        const diff = row[j]! - mean[j]!;
        acc[j]! += diff * diff;
      }
    }
    this.variances = [];
    for (let c = 0; c < k; c++) {
      const variance = new Array<number>(FEATURE_COUNT).fill(0);
      const denom = Math.max(count[c]! - 1, 1);
      for (let j = 0; j < FEATURE_COUNT; j++) {
        variance[j] = Math.max(squared[c]![j]! / denom, 1e-9);
      }
      this.variances.push(variance);
    }
  }

  scores(x: number[]): number[] {
    const out: number[] = [];
    for (let c = 0; c < this.k; c++) {
      const mean = this.means[c];
      const variance = this.variances[c];
      if (!mean || !variance) {
        out.push(-Infinity);
        continue;
      }
      let score = Math.log(Math.max(this.priors[c] ?? 1e-9, 1e-9));
      for (let j = 0; j < x.length; j++) {
        const diff = x[j]! - mean[j]!;
        score -= 0.5 * (Math.log(2 * Math.PI * variance[j]!) + (diff * diff) / variance[j]!);
      }
      out.push(score);
    }
    return out;
  }
}

/** k-nearest neighbours (k=5), Euclidean, distance-weighted votes. */
class KNNModel implements ClassModel {
  private X: number[][] = [];
  private y: number[] = [];
  private k = 0;

  constructor(private readonly neighbors: number = KNN_K) {}

  fit(X: number[][], y: number[], k: number): void {
    this.X = X;
    this.y = y;
    this.k = k;
  }

  scores(x: number[]): number[] {
    const out = new Array<number>(this.k).fill(0);
    const n = this.X.length;
    if (n === 0) return out;
    const dists = this.X.map(row => dist2(row, x));
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => dists[a]! - dists[b]!);
    const take = Math.min(this.neighbors, n);
    for (let t = 0; t < take; t++) {
      const i = order[t]!;
      const weight = 1 / (Math.sqrt(dists[i]!) + 1e-9);
      out[this.y[i]!]! += weight;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Decision trees (CART, gini) + ensembles.
// ---------------------------------------------------------------------------

type TreeNode =
  | { kind: 'leaf'; counts: number[]; total: number }
  | { kind: 'split'; feature: number; threshold: number; left: TreeNode; right: TreeNode };

type LeafNode = Extract<TreeNode, { kind: 'leaf' }>;

interface TreeParams {
  maxDepth: number;
  minLeaf: number;
  /** Random feature subspace per node (0 = all features). */
  subspace: number;
}

function treeCounts(k: number, idx: number[], y: number[], w: number[]): { counts: number[]; total: number } {
  const counts = new Array<number>(k).fill(0);
  let total = 0;
  for (const i of idx) {
    counts[y[i]!]! += w[i]!;
    total += w[i]!;
  }
  return { counts, total };
}

function giniFrom(counts: number[], total: number): number {
  if (total <= 1e-12) return 0;
  let sum = 0;
  for (const c of counts) {
    const p = c / total;
    sum += p * p;
  }
  return 1 - sum;
}

function buildTree(
  X: number[][],
  y: number[],
  w: number[],
  idx: number[],
  depth: number,
  params: TreeParams,
  rng: () => number,
  k: number,
  importance: number[] | null
): TreeNode {
  const { counts, total } = treeCounts(k, idx, y, w);
  const nodeGini = giniFrom(counts, total);
  if (depth >= params.maxDepth || idx.length < 2 * params.minLeaf || nodeGini <= 1e-12) {
    return { kind: 'leaf', counts, total };
  }

  const feats =
    params.subspace > 0
      ? randomSubset(FEATURE_COUNT, params.subspace, rng)
      : Array.from({ length: FEATURE_COUNT }, (_, j) => j);

  let bestFeature = -1;
  let bestThreshold = 0;
  let bestQuality = Infinity;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const f of feats) {
    const sorted = idx.slice().sort((a, b) => X[a]![f]! - X[b]![f]! || a - b);
    const leftCounts = new Array<number>(k).fill(0);
    const rightCounts = counts.slice();
    let leftW = 0;
    let leftN = 0;
    for (let s = 0; s < sorted.length - 1; s++) {
      const i = sorted[s]!;
      const label = y[i]!;
      leftCounts[label]! += w[i]!;
      rightCounts[label]! -= w[i]!;
      leftW += w[i]!;
      leftN++;
      const value = X[i]![f]!;
      const nextValue = X[sorted[s + 1]!]![f]!;
      if (nextValue - value <= 1e-12) continue;
      const rightN = sorted.length - leftN;
      if (leftN < params.minLeaf || rightN < params.minLeaf) continue;
      const rightW = total - leftW;
      const quality =
        (leftW / total) * giniFrom(leftCounts, leftW) +
        (rightW / total) * giniFrom(rightCounts, rightW);
      if (quality < bestQuality - 1e-12) {
        bestQuality = quality;
        bestFeature = f;
        bestThreshold = (value + nextValue) / 2;
        bestLeft = [];
        bestRight = [];
        for (const j of idx) {
          (X[j]![f]! <= bestThreshold ? bestLeft : bestRight).push(j);
        }
      }
    }
  }

  if (bestFeature < 0 || bestLeft.length === 0 || bestRight.length === 0) {
    return { kind: 'leaf', counts, total };
  }
  if (importance) importance[bestFeature]! += (nodeGini - bestQuality) * total;

  return {
    kind: 'split',
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildTree(X, y, w, bestLeft, depth + 1, params, rng, k, importance),
    right: buildTree(X, y, w, bestRight, depth + 1, params, rng, k, importance),
  };
}

function treeLeaf(node: TreeNode, x: number[]): LeafNode {
  let current: TreeNode = node;
  while (current.kind === 'split') {
    current = x[current.feature]! <= current.threshold ? current.left : current.right;
  }
  return current;
}

function treePredict(node: TreeNode, x: number[]): number {
  const leaf = treeLeaf(node, x);
  let best = 0;
  let bestCount = -1;
  leaf.counts.forEach((count, j) => {
    if (count > bestCount) {
      bestCount = count;
      best = j;
    }
  });
  return best;
}

function treeLeafRates(node: TreeNode, x: number[]): number[] {
  const leaf = treeLeaf(node, x);
  const total = Math.max(leaf.total, 1e-12);
  return leaf.counts.map(c => c / total);
}

function bootstrapIndices(n: number, rng: () => number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(Math.min(Math.max(n, 1) - 1, Math.floor(rng() * Math.max(n, 1))));
  return out;
}

/** Single CART tree (used standalone and inside ensembles). */
class DecisionTreeModel implements ClassModel {
  private root: TreeNode | null = null;
  private k = 0;

  constructor(private readonly maxDepth: number, private readonly seed: number) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    const n = X.length;
    const weights = new Array<number>(n).fill(1);
    const idx = Array.from({ length: n }, (_, i) => i);
    const params: TreeParams = { maxDepth: this.maxDepth, minLeaf: 2, subspace: 0 };
    this.root = buildTree(X, y, weights, idx, 0, params, mulberry32(this.seed), k, null);
  }

  scores(x: number[]): number[] {
    if (!this.root) return new Array<number>(this.k).fill(0);
    return treeLeafRates(this.root, x);
  }
}

/** Random forest: bagged CART trees with sqrt(d) feature subspace per node. */
class RandomForestModel implements ClassModel {
  private trees: TreeNode[] = [];
  private importance = new Array<number>(FEATURE_COUNT).fill(0);
  private k = 0;

  constructor(
    private readonly seed: number,
    private readonly nTrees: number = RF_TREES,
    private readonly maxDepth: number = RF_MAX_DEPTH
  ) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.trees = [];
    this.importance = new Array<number>(FEATURE_COUNT).fill(0);
    const rng = mulberry32(this.seed);
    const n = X.length;
    const params: TreeParams = {
      maxDepth: this.maxDepth,
      minLeaf: 2,
      subspace: Math.max(1, Math.floor(Math.sqrt(FEATURE_COUNT))),
    };
    for (let t = 0; t < this.nTrees; t++) {
      const boot = bootstrapIndices(n, rng);
      const Xb = boot.map(i => X[i]!);
      const yb = boot.map(i => y[i]!);
      const weights = new Array<number>(boot.length).fill(1);
      const idx = Array.from({ length: boot.length }, (_, i) => i);
      this.trees.push(buildTree(Xb, yb, weights, idx, 0, params, rng, k, this.importance));
    }
  }

  scores(x: number[]): number[] {
    const out = new Array<number>(this.k).fill(0);
    if (this.trees.length === 0) return out;
    for (const tree of this.trees) {
      const rates = treeLeafRates(tree, x);
      for (let c = 0; c < this.k; c++) out[c]! += rates[c] ?? 0;
    }
    return out.map(v => v / this.trees.length);
  }

  normalizedImportance(): number[] {
    const sum = this.importance.reduce((a, b) => a + b, 0);
    if (sum <= 1e-12) return this.importance.slice();
    return this.importance.map(v => v / sum);
  }
}

/** AdaBoost with depth-1 stumps (binary). Weight-vector variant: the stump
 * trainer consumes example weights directly, which is the deterministic
 * equivalent of weighted resampling. */
class AdaBoostModel implements ClassModel {
  private stumps: Array<{ node: TreeNode; alpha: number }> = [];
  private k = 0;

  constructor(private readonly seed: number, private readonly rounds: number = BOOST_ROUNDS) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.stumps = [];
    const n = X.length;
    if (n === 0) return;
    const weights = new Array<number>(n).fill(1 / n);
    const idx = Array.from({ length: n }, (_, i) => i);
    const params: TreeParams = { maxDepth: 1, minLeaf: 2, subspace: 0 };
    for (let round = 0; round < this.rounds; round++) {
      const rng = mulberry32(this.seed + round * 1013 + 7);
      const node = buildTree(X, y, weights, idx, 0, params, rng, k, null);
      let err = 0;
      for (let i = 0; i < n; i++) {
        if (treePredict(node, X[i]!) !== y[i]!) err += weights[i]!;
      }
      if (err >= 0.5 - 1e-12) break;
      if (err <= 1e-9) {
        this.stumps.push({ node, alpha: 1 });
        break;
      }
      const alpha = 0.5 * Math.log((1 - err) / err);
      this.stumps.push({ node, alpha });
      let z = 0;
      for (let i = 0; i < n; i++) {
        const h = treePredict(node, X[i]!) === 1 ? 1 : -1;
        const yi = y[i]! === 1 ? 1 : -1;
        weights[i]! *= Math.exp(-alpha * yi * h);
        z += weights[i]!;
      }
      for (let i = 0; i < n; i++) weights[i] = weights[i]! / z;
    }
  }

  scores(x: number[]): number[] {
    const out = new Array<number>(this.k).fill(0);
    if (this.k !== 2) return out;
    let margin = 0;
    for (const stump of this.stumps) {
      margin += stump.alpha * (treePredict(stump.node, x) === 1 ? 1 : -1);
    }
    out[0] = -margin;
    out[1] = margin;
    return out;
  }
}

/** Bagging: bootstrap-resampled depth-1 stumps (binary). */
class BaggingModel implements ClassModel {
  private stumps: TreeNode[] = [];
  private k = 0;

  constructor(private readonly seed: number, private readonly rounds: number = BOOST_ROUNDS) {}

  fit(X: number[][], y: number[], k: number): void {
    this.k = k;
    this.stumps = [];
    const rng = mulberry32(this.seed);
    const n = X.length;
    const params: TreeParams = { maxDepth: 1, minLeaf: 2, subspace: 0 };
    for (let round = 0; round < this.rounds; round++) {
      const boot = bootstrapIndices(n, rng);
      const Xb = boot.map(i => X[i]!);
      const yb = boot.map(i => y[i]!);
      const weights = new Array<number>(boot.length).fill(1);
      const idx = Array.from({ length: boot.length }, (_, i) => i);
      this.stumps.push(buildTree(Xb, yb, weights, idx, 0, params, rng, k, null));
    }
  }

  scores(x: number[]): number[] {
    const out = new Array<number>(this.k).fill(0);
    if (this.k !== 2 || this.stumps.length === 0) return out;
    let positiveRate = 0;
    for (const stump of this.stumps) positiveRate += treeLeafRates(stump, x)[1] ?? 0;
    positiveRate /= this.stumps.length;
    out[0] = 1 - positiveRate;
    out[1] = positiveRate;
    return out;
  }
}

// ---------------------------------------------------------------------------
// Linear regression via normal equations + Gaussian elimination.
// ---------------------------------------------------------------------------

function linearRegressionFit(X: number[][], y: number[]): number[] | null {
  const n = X.length;
  const p = FEATURE_COUNT + 1;
  if (n === 0) return null;
  const A: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    const xi = X[i]!;
    const row = [1, ...xi];
    for (let r = 0; r < p; r++) {
      const rv = row[r] ?? 0;
      b[r]! += rv * y[i]!;
      for (let c = 0; c < p; c++) A[r]![c]! += rv * (row[c] ?? 0);
    }
  }
  // Tiny ridge term keeps the normal equations solvable for collinear features.
  for (let j = 0; j < p; j++) A[j]![j]! += 1e-8;
  return gaussSolve(A, b);
}

/** Gaussian elimination with partial pivoting; null when singular. */
function gaussSolve(Ain: number[][], bin: number[]): number[] | null {
  const p = bin.length;
  const A = Ain.map(row => row.slice());
  const b = bin.slice();
  for (let col = 0; col < p; col++) {
    let pivotRow = col;
    let best = Math.abs(A[col]![col]!);
    for (let r = col + 1; r < p; r++) {
      const value = Math.abs(A[r]![col]!);
      if (value > best) {
        best = value;
        pivotRow = r;
      }
    }
    if (best < 1e-10) return null;
    if (pivotRow !== col) {
      const tmpA = A[col]!;
      A[col] = A[pivotRow]!;
      A[pivotRow] = tmpA;
      const tmpB = b[col]!;
      b[col] = b[pivotRow]!;
      b[pivotRow] = tmpB;
    }
    const pivot = A[col]![col]!;
    for (let r = col + 1; r < p; r++) {
      const factor = A[r]![col]! / pivot;
      if (factor === 0) continue;
      for (let c = col; c < p; c++) A[r]![c]! -= factor * A[col]![c]!;
      b[r]! -= factor * b[col]!;
    }
  }
  const x = new Array<number>(p).fill(0);
  for (let r = p - 1; r >= 0; r--) {
    let sum = b[r]!;
    for (let c = r + 1; c < p; c++) sum -= A[r]![c]! * x[c]!;
    x[r] = sum / A[r]![r]!;
  }
  return x;
}

function linearPredict(beta: number[], x: number[]): number {
  let sum = beta[0] ?? 0;
  for (let j = 0; j < x.length; j++) sum += (beta[j + 1] ?? 0) * x[j]!;
  return sum;
}

// ---------------------------------------------------------------------------
// Clustering: k-means++, DBSCAN, agglomerative (single linkage), PCA.
// ---------------------------------------------------------------------------

function chooseClusterCount(n: number): number {
  if (n >= 50) return 5;
  if (n >= 20) return 4;
  if (n >= 8) return 3;
  if (n >= 2) return 2;
  return 1;
}

function kmeansCluster(
  X: number[][],
  k: number,
  iters: number,
  seed: number
): { labels: number[]; centroids: number[][]; inertia: number } {
  const n = X.length;
  const labels = new Array<number>(n).fill(0);
  if (n === 0) return { labels, centroids: [], inertia: 0 };
  const kk = Math.max(1, Math.min(k, n));
  const rng = mulberry32(seed);

  const centroids: number[][] = [];
  centroids.push(X[Math.min(n - 1, Math.floor(rng() * n))]!.slice());
  const minD2 = new Array<number>(n).fill(Infinity);
  while (centroids.length < kk) {
    const last = centroids[centroids.length - 1]!;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const value = dist2(X[i]!, last);
      if (value < minD2[i]!) minD2[i] = value;
      sum += minD2[i]!;
    }
    if (sum <= 1e-12) {
      centroids.push(X[Math.min(n - 1, Math.floor(rng() * n))]!.slice());
      continue;
    }
    let r = rng() * sum;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) {
      r -= minD2[i]!;
      if (r <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push(X[chosen]!.slice());
  }

  for (let iter = 0; iter < iters; iter++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = dist2(X[i]!, centroids[0]!);
      for (let c = 1; c < centroids.length; c++) {
        const value = dist2(X[i]!, centroids[c]!);
        if (value < bestD) {
          bestD = value;
          best = c;
        }
      }
      if (labels[i] !== best) {
        labels[i] = best;
        moved = true;
      }
    }
    const sums = centroids.map(() => new Array<number>(X[0]!.length).fill(0));
    const counts = new Array<number>(centroids.length).fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i]!;
      counts[c]!++;
      const sum = sums[c]!;
      for (let j = 0; j < X[0]!.length; j++) sum[j]! += X[i]![j]!;
    }
    for (let c = 0; c < centroids.length; c++) {
      if (counts[c] === 0) {
        // Empty cluster: re-seed with the point farthest from its assignment.
        let far = 0;
        let farD = -1;
        for (let i = 0; i < n; i++) {
          const value = dist2(X[i]!, centroids[labels[i]!]!);
          if (value > farD) {
            farD = value;
            far = i;
          }
        }
        centroids[c] = X[far]!.slice();
        labels[far] = c;
        continue;
      }
      const sum = sums[c]!;
      for (let j = 0; j < X[0]!.length; j++) centroids[c]![j] = sum[j]! / counts[c]!;
    }
    if (!moved) break;
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += dist2(X[i]!, centroids[labels[i]!]!);
  return { labels, centroids, inertia };
}

function dbscanCluster(X: number[][], minPts: number): { labels: number[]; eps: number } {
  const n = X.length;
  const labels = new Array<number>(n).fill(-2); // -2 unvisited, -1 noise, >=0 cluster
  if (n === 0) return { labels, eps: 0 };

  // eps heuristic: 2 x median of each point's 5-NN distance.
  const kn = Math.min(5, n - 1);
  const kDists: number[] = [];
  if (kn >= 1) {
    for (let i = 0; i < n; i++) {
      const ds: number[] = [];
      for (let j = 0; j < n; j++) if (j !== i) ds.push(Math.sqrt(dist2(X[i]!, X[j]!)));
      ds.sort((a, b) => a - b);
      kDists.push(ds[kn - 1]!);
    }
  }
  const eps = kDists.length > 0 ? 2 * medianOf(kDists) : 1;

  const regionQuery = (i: number): number[] => {
    const out: number[] = [];
    for (let j = 0; j < n; j++) {
      if (Math.sqrt(dist2(X[i]!, X[j]!)) <= eps) out.push(j);
    }
    return out;
  };

  let clusterId = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) {
      labels[i] = -1;
      continue;
    }
    const cluster = clusterId++;
    labels[i] = cluster;
    const queue = neighbors.filter(j => j !== i);
    for (let qi = 0; qi < queue.length; qi++) {
      const q = queue[qi]!;
      if (labels[q] === -1) {
        labels[q] = cluster; // border point
        continue;
      }
      if (labels[q] !== -2) continue;
      labels[q] = cluster;
      const expanded = regionQuery(q);
      if (expanded.length >= minPts) {
        for (const nb of expanded) {
          if (labels[nb] === -2 || labels[nb] === -1) queue.push(nb);
        }
      }
    }
  }
  return { labels, eps };
}

function dbscanSizes(labels: number[]): number[] {
  const counts = new Map<number, number>();
  for (const label of labels) {
    if (label < 0) continue;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return [...counts.keys()].sort((a, b) => a - b).map(id => counts.get(id) ?? 0);
}

/** Agglomerative single-linkage on an evenly spaced sample, cut to k clusters. */
function agglomerativeSingle(
  X: number[][],
  k: number,
  sampleSize: number
): { labels: number[]; sizes: number[] } {
  const n = X.length;
  const m = Math.max(1, Math.min(sampleSize, n));
  const sampleIdx = evenIndices(n, m);
  const points = sampleIdx.map(i => X[i]!);
  const dist: number[][] = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (let a = 0; a < m; a++) {
    for (let b = a + 1; b < m; b++) {
      const value = Math.sqrt(dist2(points[a]!, points[b]!));
      dist[a]![b] = value;
      dist[b]![a] = value;
    }
  }

  const members: number[][] = points.map((_, i) => [i]);
  const active: boolean[] = points.map(() => true);
  const D = dist.map(row => row.slice());
  let activeCount = m;
  const target = Math.max(1, Math.min(k, m));

  while (activeCount > target) {
    let bestA = -1;
    let bestB = -1;
    let bestD = Infinity;
    for (let a = 0; a < m; a++) {
      if (!active[a]) continue;
      for (let b = a + 1; b < m; b++) {
        if (!active[b]) continue;
        if (D[a]![b]! < bestD) {
          bestD = D[a]![b]!;
          bestA = a;
          bestB = b;
        }
      }
    }
    if (bestA < 0 || bestB < 0) break;
    for (let c = 0; c < m; c++) {
      if (!active[c] || c === bestA || c === bestB) continue;
      const value = Math.min(D[bestA]![c]!, D[bestB]![c]!);
      D[bestA]![c] = value;
      D[c]![bestA] = value;
    }
    members[bestA] = members[bestA]!.concat(members[bestB]!);
    members[bestB] = [];
    active[bestB] = false;
    activeCount--;
  }

  const labels = new Array<number>(m).fill(-1);
  const sizes: number[] = [];
  let clusterId = 0;
  for (let a = 0; a < m; a++) {
    if (!active[a] || members[a]!.length === 0) continue;
    sizes.push(members[a]!.length);
    for (const p of members[a]!) labels[p] = clusterId;
    clusterId++;
  }
  return { labels, sizes: sizes.sort((a, b) => b - a) };
}

/** Mean silhouette width; null unless there are >= 2 nonempty clusters. */
function silhouetteScore(X: number[][], labels: number[]): number | null {
  const n = X.length;
  if (n === 0) return null;
  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const label = labels[i]!;
    const group = groups.get(label);
    if (group) group.push(i);
    else groups.set(label, [i]);
  }
  if (groups.size < 2) return null;
  const clusterIds = [...groups.keys()];
  let total = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const own = labels[i]!;
    const sums = new Map<number, number>();
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const label = labels[j]!;
      sums.set(label, (sums.get(label) ?? 0) + Math.sqrt(dist2(X[i]!, X[j]!)));
    }
    const ownGroup = groups.get(own)!;
    const a = ownGroup.length > 1 ? (sums.get(own) ?? 0) / (ownGroup.length - 1) : 0;
    let b = Infinity;
    for (const id of clusterIds) {
      if (id === own) continue;
      const meanDist = (sums.get(id) ?? 0) / groups.get(id)!.length;
      if (meanDist < b) b = meanDist;
    }
    if (!Number.isFinite(b)) continue;
    const denom = Math.max(a, b);
    total += denom > 1e-12 ? (b - a) / denom : 0;
    counted++;
  }
  return counted > 0 ? roundTo(total / counted, 4) : null;
}

/** Symmetric Jacobi eigen decomposition (cyclic rotations). */
function jacobiEigen(input: number[][]): { values: number[]; vectors: number[][] } {
  const d = input.length;
  const a = input.map(row => row.slice());
  const v: number[][] = Array.from({ length: d }, (_, r) => {
    const row = new Array<number>(d).fill(0);
    row[r] = 1;
    return row;
  });
  for (let sweep = 0; sweep < 100 && d > 1; sweep++) {
    let off = 0;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) off += a[p]![q]! * a[p]![q]!;
    }
    if (off < 1e-14) break;
    for (let p = 0; p < d - 1; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-12) continue;
        const app = a[p]![p]!;
        const aqq = a[q]![q]!;
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        a[p]![p] = app - t * apq;
        a[q]![q] = aqq + t * apq;
        a[p]![q] = 0;
        a[q]![p] = 0;
        for (let r = 0; r < d; r++) {
          if (r === p || r === q) continue;
          const arp = a[r]![p]!;
          const arq = a[r]![q]!;
          a[r]![p] = c * arp - s * arq;
          a[p]![r] = arp * c - arq * s;
          a[r]![q] = s * arp + c * arq;
          a[q]![r] = arp * s + arq * c;
        }
        for (let r = 0; r < d; r++) {
          const vrp = v[r]![p]!;
          const vrq = v[r]![q]!;
          v[r]![p] = c * vrp - s * vrq;
          v[r]![q] = s * vrp + c * vrq;
        }
      }
    }
  }
  // Eigenvector j is column j of V.
  const vectors: number[][] = Array.from({ length: d }, (_, j) =>
    Array.from({ length: d }, (_, r) => v[r]![j]!)
  );
  return { values: a.map((row, r) => row[r]!), vectors };
}

/** Project (already standardized) rows onto the top-2 principal components. */
function pcaProject2(X: number[][]): { xs: number[]; ys: number[] } {
  const xs: number[] = [];
  const ys: number[] = [];
  const n = X.length;
  if (n === 0) return { xs, ys };
  const d = X[0]!.length;
  const mu = new Array<number>(d).fill(0);
  for (const row of X) {
    for (let j = 0; j < d; j++) mu[j]! += row[j] ?? 0;
  }
  for (let j = 0; j < d; j++) mu[j] = mu[j]! / n;

  const cov: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const row of X) {
    const centered = row.map((value, j) => (value ?? 0) - mu[j]!);
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) cov[a]![b]! += centered[a]! * centered[b]!;
    }
  }
  const denom = Math.max(1, n - 1);
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      cov[a]![b] = cov[a]![b]! / denom;
      cov[b]![a] = cov[a]![b]!;
    }
  }

  const eig = jacobiEigen(cov);
  const order = Array.from({ length: d }, (_, i) => i).sort(
    (a, b) => eig.values[b]! - eig.values[a]!
  );
  const v0 = eig.vectors[order[0] ?? 0] ?? new Array<number>(d).fill(0);
  const v1 = order.length > 1 ? eig.vectors[order[1]!] ?? new Array<number>(d).fill(0) : new Array<number>(d).fill(0);

  for (const row of X) {
    let px = 0;
    let py = 0;
    for (let j = 0; j < d; j++) {
      const centered = (row[j] ?? 0) - mu[j]!;
      px += centered * v0[j]!;
      py += centered * v1[j]!;
    }
    xs.push(px);
    ys.push(py);
  }
  return { xs, ys };
}

// ---------------------------------------------------------------------------
// Evaluation.
// ---------------------------------------------------------------------------

function accuracyOf(yTrue: number[], yPred: number[]): number {
  if (yTrue.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++;
  }
  return roundTo(correct / yTrue.length, 4);
}

function buildConfusion(k: number, yTrue: number[], yPred: number[]): number[][] {
  const matrix: number[][] = [];
  for (let r = 0; r < k; r++) matrix.push(new Array<number>(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    matrix[yTrue[i]!]![yPred[i]!]! += 1;
  }
  return matrix;
}

function macroPRF(
  k: number,
  yTrue: number[],
  yPred: number[]
): { precision: number; recall: number; f1: number } {
  let precisionSum = 0;
  let recallSum = 0;
  let f1Sum = 0;
  for (let c = 0; c < k; c++) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const isTrue = yTrue[i]! === c;
      const isPred = yPred[i]! === c;
      if (isPred && isTrue) tp++;
      else if (isPred && !isTrue) fp++;
      else if (!isPred && isTrue) fn++;
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    precisionSum += precision;
    recallSum += recall;
    f1Sum += f1;
  }
  return {
    precision: roundTo(precisionSum / k, 4),
    recall: roundTo(recallSum / k, 4),
    f1: roundTo(f1Sum / k, 4),
  };
}

/** ROC-AUC by rank statistics (handles score ties with average ranks). */
function rocAucFromScores(scores: number[], yTrue: number[]): number | null {
  const n = scores.length;
  let positives = 0;
  let negatives = 0;
  for (const v of yTrue) {
    if (v === 1) positives++;
    else negatives++;
  }
  if (positives === 0 || negatives === 0) return null;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a]! - scores[b]!);
  const rank = new Array<number>(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[order[j + 1]!]! === scores[order[i]!]!) j++;
    const averageRank = (i + j + 2) / 2;
    for (let t = i; t <= j; t++) rank[order[t]!] = averageRank;
    i = j + 1;
  }
  let positiveRankSum = 0;
  for (let t = 0; t < n; t++) {
    if (yTrue[t] === 1) positiveRankSum += rank[t]!;
  }
  return roundTo((positiveRankSum - (positives * (positives + 1)) / 2) / (positives * negatives), 4);
}

interface EvaluatedModels {
  rows: MetricRow[];
  confusion: Record<string, ConfusionMatrix>;
  testPreds: Map<string, number[]>;
  fitted: Array<{ name: string; model: ClassModel }>;
}

function evaluateModels(
  models: Array<{ name: string; model: ClassModel }>,
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  labels: string[],
  binary: boolean
): EvaluatedModels {
  const rows: MetricRow[] = [];
  const confusion: Record<string, ConfusionMatrix> = {};
  const testPreds = new Map<string, number[]>();
  for (const { name, model } of models) {
    model.fit(XTrain, yTrain, labels.length);
    const preds = XTest.map(x => predictClass(model, x));
    const prf = macroPRF(labels.length, yTest, preds);
    const rocAuc = binary
      ? rocAucFromScores(XTest.map(x => model.scores(x)[1] ?? 0), yTest)
      : null;
    rows.push({
      model: name,
      accuracy: accuracyOf(yTest, preds),
      precision: prf.precision,
      recall: prf.recall,
      f1: prf.f1,
      rocAuc,
    });
    confusion[name] = { labels: labels.slice(), matrix: buildConfusion(labels.length, yTest, preds) };
    testPreds.set(name, preds);
  }
  return { rows, confusion, testPreds, fitted: models };
}

/** Chronological 5-fold CV over contiguous blocks (no shuffling). RBF-SVM is
 * excluded from CV to keep the fold loop bounded; it stays in the main rows. */
const CV_BUILDERS: Array<{ name: string; make: () => ClassModel }> = [
  { name: 'LogisticRegression', make: () => new LogisticOvR(LOGISTIC_LAMBDA) },
  { name: 'LinearSVM', make: () => new LinearSVM(12) },
  { name: 'GaussianNB', make: () => new GaussianNBModel() },
  { name: 'KNN', make: () => new KNNModel() },
];

function crossValidate(
  rawX: number[][],
  y: number[],
  k: number
): Record<string, { mean: number; std: number; folds: number }> {
  const out: Record<string, { mean: number; std: number; folds: number }> = {};
  for (const builder of CV_BUILDERS) {
    out[builder.name] = { mean: 0, std: 0, folds: 0 };
  }
  const n = rawX.length;
  if (n < 2 * CV_FOLDS) return out;
  const base = Math.floor(n / CV_FOLDS);
  const accs: Record<string, number[]> = {};
  for (const builder of CV_BUILDERS) accs[builder.name] = [];
  for (let fold = 0; fold < CV_FOLDS; fold++) {
    const testStart = fold * base;
    const testEnd = fold === CV_FOLDS - 1 ? n : testStart + base;
    const trainIdx: number[] = [];
    const testIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i >= testStart && i < testEnd) testIdx.push(i);
      else trainIdx.push(i);
    }
    if (trainIdx.length === 0 || testIdx.length === 0) continue;
    const trainRaw = trainIdx.map(i => rawX[i]!);
    const z = fitStandardizer(trainRaw);
    const trainZ = applyStandardizer(trainRaw, z);
    const testZ = applyStandardizer(testIdx.map(i => rawX[i]!), z);
    for (const builder of CV_BUILDERS) {
      const model = builder.make();
      model.fit(trainZ, trainIdx.map(i => y[i]!), k);
      let correct = 0;
      for (let i = 0; i < testZ.length; i++) {
        if (predictClass(model, testZ[i]!) === y[testIdx[i]!]) correct++;
      }
      accs[builder.name]!.push(correct / testZ.length);
    }
  }
  for (const builder of CV_BUILDERS) {
    const values = accs[builder.name]!;
    out[builder.name] = {
      mean: roundTo(meanOf(values), 4),
      std: roundTo(stdOf(values), 4),
      folds: values.length,
    };
  }
  return out;
}

/** Seeded bootstrap (200 resamples) accuracy CIs over shared resample indices. */
function bootstrapCI(
  yTest: number[],
  testPreds: Map<string, number[]>,
  seed: number
): Record<string, { mean: number; lo: number; hi: number }> {
  const out: Record<string, { mean: number; lo: number; hi: number }> = {};
  const n = yTest.length;
  if (n === 0) return out;
  const rng = mulberry32(seed);
  const names = [...testPreds.keys()];
  const accs: Record<string, number[]> = {};
  for (const name of names) accs[name] = [];
  for (let round = 0; round < BOOTSTRAP_ROUNDS; round++) {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) idx.push(Math.floor(rng() * n));
    for (const name of names) {
      const preds = testPreds.get(name)!;
      let correct = 0;
      for (let i = 0; i < n; i++) {
        if (preds[idx[i]!] === yTest[idx[i]!]) correct++;
      }
      accs[name]!.push(correct / n);
    }
  }
  for (const name of names) {
    const sorted = accs[name]!.sort((a, b) => a - b);
    out[name] = {
      mean: roundTo(meanOf(sorted), 4),
      lo: roundTo(sorted[Math.floor(0.025 * (BOOTSTRAP_ROUNDS - 1))]!, 4),
      hi: roundTo(sorted[Math.ceil(0.975 * (BOOTSTRAP_ROUNDS - 1))]!, 4),
    };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Concept learning (Mitchell-style, discrete attributes).
// ---------------------------------------------------------------------------

const CONCEPT_ATTRS = [
  'category',
  'threatLevel',
  'tierClass',
  'sentimentClass',
  'corroborationClass',
  'breaking',
];

const EMPTY = '∅'; // ∅ — accepts nothing (most specific)
const ANY = '?'; // ? — accepts anything (most general)

interface ConceptExample {
  attrs: string[];
  label: boolean;
}

function conceptExamples(items: ProcessedItem[]): ConceptExample[] {
  return items.map(item => ({
    attrs: [
      item.category,
      item.level,
      item.tier === null ? 't3' : item.tier <= 1 ? 't1' : item.tier <= 2 ? 't2' : 't3',
      item.sentiment < -0.2 ? 'neg' : item.sentiment > 0.2 ? 'pos' : 'neu',
      item.corroboration < 2 ? 'low' : item.corroboration < 5 ? 'med' : 'high',
      item.isBreaking ? 'yes' : 'no',
    ],
    label: item.highRisk,
  }));
}

function vecToRecord(names: string[], vec: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let j = 0; j < names.length; j++) out[names[j]!] = vec[j] ?? EMPTY;
  return out;
}

/** Find-S: most specific generalization over the positive examples. */
function findS(examples: ConceptExample[], attrNames: string[]): FindSResult {
  const hypothesis = attrNames.map(() => EMPTY);
  const steps: { exampleIndex: number; hypothesis: Record<string, string> }[] = [];
  for (let i = 0; i < examples.length; i++) {
    const example = examples[i]!;
    if (!example.label) continue; // Find-S ignores negatives
    let changed = false;
    for (let j = 0; j < attrNames.length; j++) {
      const current = hypothesis[j]!;
      const value = example.attrs[j]!;
      if (current === EMPTY) {
        hypothesis[j] = value;
        changed = true;
      } else if (current !== value) {
        hypothesis[j] = ANY;
        changed = true;
      }
    }
    if (changed) steps.push({ exampleIndex: i, hypothesis: vecToRecord(attrNames, hypothesis) });
  }
  return { hypothesis: vecToRecord(attrNames, hypothesis), steps };
}

/** h ⊒ x in the Mitchell generalization lattice: an '∅' position in x is the
 * bottom element (covered by every h), and h must accept every concrete value
 * x accepts (h[j] === ANY or h[j] === x[j]). */
function coversVec(hypothesis: string[], x: string[]): boolean {
  for (let j = 0; j < hypothesis.length; j++) {
    const xv = x[j];
    if (xv === undefined || xv === EMPTY) continue;
    const hv = hypothesis[j]!;
    if (hv !== ANY && hv !== xv) return false;
  }
  return true;
}

function sameVec(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((v, j) => v === b[j]);
}

/** Unique minimal generalization of s needed to cover the positive x. */
function minimalGeneralization(s: string[], x: string[]): string[] {
  return s.map((value, j) => {
    if (value === x[j] || value === ANY) return value;
    return value === EMPTY ? x[j]! : ANY;
  });
}

/** Candidate Elimination with version-space boundaries S and G. Conflicting
 * examples (same attribute tuple, different label) are skipped with a note. */
function candidateElimination(examples: ConceptExample[], attrNames: string[]): CandidateEliminationResult {
  const notes: string[] = [];
  const attrCount = attrNames.length;
  const domains: string[][] = Array.from({ length: attrCount }, () => []);
  for (const example of examples) {
    for (let j = 0; j < attrCount; j++) {
      const value = example.attrs[j]!;
      if (!domains[j]!.includes(value)) domains[j]!.push(value);
    }
  }

  let S: string[][] = [attrNames.map(() => EMPTY)];
  let G: string[][] = [attrNames.map(() => ANY)];
  const positives: string[][] = [];
  const negatives: string[][] = [];
  const seenLabels = new Map<string, boolean>();
  const history: CandidateEliminationResult['history'] = [];
  let step = 0;

  const consistentWithHistory = (h: string[]): boolean => {
    for (const p of positives) if (!coversVec(h, p)) return false;
    for (const n of negatives) if (coversVec(h, n)) return false;
    return true;
  };

  for (let i = 0; i < examples.length; i++) {
    const example = examples[i]!;
    const tupleKey = example.attrs.join('\u0001');
    const seenLabel = seenLabels.get(tupleKey);
    if (seenLabel !== undefined) {
      if (seenLabel !== example.label) {
        notes.push(
          'Skipped example ' + i + ': attribute tuple seen earlier with the opposite label ' +
          '(conflicting evidence); the version space cannot absorb it.'
        );
      }
      continue;
    }
    seenLabels.set(tupleKey, example.label);
    step++;

    if (example.label) {
      positives.push(example.attrs);
      G = G.filter(g => coversVec(g, example.attrs));
      const candidates: string[][] = [];
      for (const s of S) {
        if (coversVec(s, example.attrs)) {
          if (!candidates.some(c => sameVec(c, s))) candidates.push(s);
          continue;
        }
        const generalized = minimalGeneralization(s, example.attrs);
        if (
          consistentWithHistory(generalized) &&
          G.some(g => coversVec(g, generalized)) &&
          !candidates.some(c => sameVec(c, generalized))
        ) {
          candidates.push(generalized);
        }
      }
      // Keep only the minimal (most specific) generalizations.
      const minimal: string[][] = [];
      for (const candidate of candidates) {
        const dominated = candidates.some(
          other => !sameVec(other, candidate) && coversVec(candidate, other) && !coversVec(other, candidate)
        );
        if (!dominated && !minimal.some(m => sameVec(m, candidate))) minimal.push(candidate);
      }
      S = minimal;
      if (S.length > 64) {
        S = S.slice(0, 64);
        notes.push('S boundary truncated to 64 hypotheses.');
      }
      if (G.length === 0) {
        history.push({ step, specific: vecToRecord(attrNames, attrNames.map(() => EMPTY)), general: [] });
        notes.push('G boundary emptied after example ' + i + ': no consistent general hypothesis remains. Stopping.');
        break;
      }
    } else {
      negatives.push(example.attrs);
      S = S.filter(s => !coversVec(s, example.attrs));
      const candidates: string[][] = [];
      for (const g of G) {
        if (!coversVec(g, example.attrs)) {
          if (!candidates.some(c => sameVec(c, g))) candidates.push(g);
          continue;
        }
        for (let j = 0; j < attrCount; j++) {
          if (g[j] !== ANY) continue;
          for (const value of domains[j]!) {
            if (value === example.attrs[j]) continue;
            const specialized = g.slice();
            specialized[j] = value;
            if (coversVec(specialized, example.attrs)) continue;
            if (!consistentWithHistory(specialized)) continue;
            if (!S.some(s => coversVec(specialized, s))) continue;
            if (!candidates.some(c => sameVec(c, specialized))) candidates.push(specialized);
          }
        }
      }
      G = candidates;
      if (G.length > 200) {
        G = G.slice(0, 200);
        notes.push('G boundary truncated to 200 hypotheses.');
      }
      if (S.length === 0) {
        history.push({ step, specific: vecToRecord(attrNames, attrNames.map(() => EMPTY)), general: G.map(h => vecToRecord(attrNames, h)) });
        notes.push('Version space collapsed after example ' + i + ': S boundary is empty (data not separable by these attributes). Stopping.');
        break;
      }
    }

    history.push({
      step,
      specific: vecToRecord(attrNames, S[0]!),
      general: G.slice(0, 32).map(h => vecToRecord(attrNames, h)),
    });
    if (history.length >= 120) {
      notes.push('History truncated to the first 120 updates.');
      break;
    }
  }

  if (positives.length === 0) notes.push('No positive examples: S boundary remains fully specific.');

  return {
    specific: vecToRecord(attrNames, S[0] ?? attrNames.map(() => EMPTY)),
    general: G.map(h => vecToRecord(attrNames, h)),
    history,
    notes,
  };
}

// ---------------------------------------------------------------------------
// Status engine (transparent per-country aggregation).
// ---------------------------------------------------------------------------

function statusBandOf(score: number): string {
  return score >= 75 ? 'critical' : score >= 50 ? 'high' : score >= 25 ? 'medium' : 'low';
}

function computeStatusEngine(items: ProcessedItem[]): { countries: StatusEntry[]; methodology: string } {
  interface CountryAcc {
    events: number;
    conflict: number;
    absSentiment: number;
    maxCorroboration: number;
    alerts: number;
  }
  const byCountry = new Map<string, CountryAcc>();
  for (const item of items) {
    for (const code of item.countries) {
      let acc = byCountry.get(code);
      if (!acc) {
        acc = { events: 0, conflict: 0, absSentiment: 0, maxCorroboration: 0, alerts: 0 };
        byCountry.set(code, acc);
      }
      acc.events++;
      if (item.flags.conflict || item.flags.military) acc.conflict++;
      acc.absSentiment += Math.abs(item.sentiment);
      acc.maxCorroboration = Math.max(acc.maxCorroboration, item.corroboration);
      if (item.isAlert) acc.alerts++;
    }
  }

  let maxEvents = 1;
  for (const acc of byCountry.values()) maxEvents = Math.max(maxEvents, acc.events);

  const entries: StatusEntry[] = [];
  for (const [code, acc] of byCountry) {
    const volume = acc.events / maxEvents;
    const conflictShare = acc.conflict / acc.events;
    const absSentiment = acc.absSentiment / acc.events;
    const corroboration = Math.min(acc.maxCorroboration / 10, 1);
    const alertShare = acc.alerts / acc.events;
    const score = roundTo(
      100 * (0.3 * volume + 0.25 * conflictShare + 0.15 * absSentiment + 0.15 * corroboration + 0.15 * alertShare),
      1
    );
    entries.push({
      code,
      score,
      band: statusBandOf(score),
      drivers: [
        { label: 'Event volume', value: roundTo(30 * volume, 2) },
        { label: 'Conflict / military share', value: roundTo(25 * conflictShare, 2) },
        { label: 'Mean |sentiment|', value: roundTo(15 * absSentiment, 2) },
        { label: 'Peak corroboration', value: roundTo(15 * corroboration, 2) },
        { label: 'Alert share', value: roundTo(15 * alertShare, 2) },
      ],
      events: acc.events,
    });
  }

  entries.sort(
    (a, b) => b.score - a.score || b.events - a.events || (a.code < b.code ? -1 : a.code > b.code ? 1 : 0)
  );

  return { countries: entries.slice(0, 12), methodology: STATUS_METHODOLOGY };
}

// ---------------------------------------------------------------------------
// Pipeline preview.
// ---------------------------------------------------------------------------

function buildPipelinePreview(items: ProcessedItem[], kmeansLabels: number[]): PipelinePreview[] {
  return items
    .slice(-8)
    .reverse()
    .map(item => ({
      headline: item.title,
      sentiment: roundTo(item.sentiment, 3),
      entities: item.countries.map(code => ISO2_PRIMARY.get(code) ?? code),
      category: item.category,
      level: item.level,
      clusterId: kmeansLabels[item.index] ?? null,
    }));
}

// ---------------------------------------------------------------------------
// Orchestration.
// ---------------------------------------------------------------------------

function makeBaseModels(): Array<{ name: string; model: ClassModel }> {
  return [
    { name: 'LogisticRegression', model: new LogisticOvR(LOGISTIC_LAMBDA) },
    { name: 'LinearSVM', model: new LinearSVM(12) },
    { name: 'RBFSVM', model: new RBFSVM(13) },
    { name: 'GaussianNB', model: new GaussianNBModel() },
    { name: 'KNN', model: new KNNModel() },
  ];
}

function makeEnsembleModels(): Array<{ name: string; model: ClassModel }> {
  return [
    { name: 'DecisionTree', model: new DecisionTreeModel(6, 21) },
    { name: 'RandomForest', model: new RandomForestModel(22) },
    { name: 'AdaBoost', model: new AdaBoostModel(23) },
    { name: 'Bagging', model: new BaggingModel(24) },
  ];
}

function splitIndex(n: number): number {
  if (n <= 1) return n;
  const cut = clamp(Math.floor(n * 0.8), 1, n - 1);
  return cut;
}

function emptyRegression(): RegressionReport {
  return { target: 'riskScore', mae: 0, mse: 0, rmse: 0, r2: 0, samplePoints: [] };
}

function emptyClustering(): ClusteringReport {
  return {
    pca: { points: [] },
    kmeans: { k: 0, sizes: [], inertia: 0, silhouette: null },
    dbscan: { eps: 0, minPts: DBSCAN_MIN_PTS, sizes: [], noiseCount: 0 },
    hierarchical: { sizes: [], silhouette: null },
  };
}

function runAll(rawItems: MLLabInputItem[]): MLLabResults {
  const withIndex = rawItems.map((it, i): IndexedItem => ({ it, i }));
  const capped = withIndex
    .slice()
    .sort((a, b) => safeMs(b.it.pubDate) - safeMs(a.it.pubDate) || a.i - b.i)
    .slice(0, MAX_ITEMS);
  const droppedCount = Math.max(0, rawItems.length - capped.length);
  const sourceCount = new Set(capped.map(entry => entry.it.source)).size;

  const { processed, imputed } = preprocessBatch(capped);

  const classDistribution: Record<string, number> = {};
  for (const item of processed) {
    classDistribution[item.band] = (classDistribution[item.band] ?? 0) + 1;
  }

  const preprocessSummary = {
    input: capped.length,
    afterDedupe: processed.length,
    imputed,
    features: FEATURE_NAMES.slice(),
    trainSize: 0,
    testSize: 0,
    classDistribution,
  };

  const emptyConceptLearning = {
    finds: findS([], CONCEPT_ATTRS),
    ce: candidateElimination([], CONCEPT_ATTRS),
  };

  if (processed.length === 0) {
    return {
      builtAtEpochMs: 0, // stamped by the manager (workers avoid Date.now)
      sourceCount,
      droppedCount,
      preprocess: preprocessSummary,
      classification: [],
      regression: emptyRegression(),
      clustering: emptyClustering(),
      ensembles: { rows: [], featureImportance: [] },
      biasVariance: { sweep: { values: [] } },
      regularization: { lambdas: REG_LAMBDAS.slice(), trainAccuracy: [], testAccuracy: [] },
      conceptLearning: emptyConceptLearning,
      pipelinePreview: [],
      statusEngine: { countries: [], methodology: STATUS_METHODOLOGY },
    };
  }

  const rawX = processed.map(item => featureVector(item));
  const trainCut = splitIndex(processed.length);
  const standardizer = fitStandardizer(rawX.slice(0, trainCut));
  const zX = applyStandardizer(rawX, standardizer);
  const zTrain = zX.slice(0, trainCut);
  const zTest = zX.slice(trainCut);
  preprocessSummary.trainSize = trainCut;
  preprocessSummary.testSize = processed.length - trainCut;

  // --- Task 1: 4-class risk band ---
  const bandLabels: string[] = BANDS.filter(band => (classDistribution[band] ?? 0) > 0);
  const yBand = processed.map(item => Math.max(0, bandLabels.indexOf(item.band)));
  const bandEval = evaluateModels(
    makeBaseModels(),
    zTrain,
    yBand.slice(0, trainCut),
    zTest,
    yBand.slice(trainCut),
    bandLabels,
    false
  );
  const bandReport: ClassificationReport = {
    task: 'risk_band',
    trainSize: trainCut,
    testSize: processed.length - trainCut,
    rows: bandEval.rows,
    confusion: bandEval.confusion,
    crossVal: crossValidate(rawX, yBand, bandLabels.length),
    bootstrap: bootstrapCI(yBand.slice(trainCut), bandEval.testPreds, 51),
  };

  // --- Task 2: binary high risk + ensembles ---
  const yBin = processed.map(item => (item.highRisk ? 1 : 0));
  const binEval = evaluateModels(
    makeBaseModels(),
    zTrain,
    yBin.slice(0, trainCut),
    zTest,
    yBin.slice(trainCut),
    ['no', 'yes'],
    true
  );
  const ensEval = evaluateModels(
    makeEnsembleModels(),
    zTrain,
    yBin.slice(0, trainCut),
    zTest,
    yBin.slice(trainCut),
    ['no', 'yes'],
    true
  );
  const mergedPreds = new Map([...binEval.testPreds, ...ensEval.testPreds]);
  const binReport: ClassificationReport = {
    task: 'high_risk',
    trainSize: trainCut,
    testSize: processed.length - trainCut,
    rows: binEval.rows,
    confusion: { ...binEval.confusion, ...ensEval.confusion },
    crossVal: crossValidate(rawX, yBin, 2),
    bootstrap: bootstrapCI(yBin.slice(trainCut), mergedPreds, 52),
  };

  // --- Feature importance from the fitted random forest ---
  const rfEntry = ensEval.fitted.find(entry => entry.name === 'RandomForest');
  const featureImportance: { feature: string; importance: number }[] =
    rfEntry && rfEntry.model instanceof RandomForestModel
      ? rfEntry.model
          .normalizedImportance()
          .map((importance, j) => ({
            feature: FEATURE_NAMES[j] ?? String(j),
            importance: roundTo(importance, 4),
          }))
          .sort((a, b) => b.importance - a.importance)
      : [];

  // --- Linear regression on the transparent risk score ---
  const riskValues = processed.map(item => item.riskScore);
  let regression = emptyRegression();
  const beta = linearRegressionFit(zTrain, riskValues.slice(0, trainCut));
  if (beta) {
    const actual = riskValues.slice(trainCut);
    const predicted = zTest.map(x => linearPredict(beta, x));
    let sse = 0;
    let sst = 0;
    const testMean = meanOf(actual);
    for (let i = 0; i < actual.length; i++) {
      const err = actual[i]! - (predicted[i] ?? 0);
      sse += err * err;
      const centered = actual[i]! - testMean;
      sst += centered * centered;
    }
    const mse = actual.length > 0 ? sse / actual.length : 0;
    regression = {
      target: 'riskScore',
      mae: roundTo(
        actual.length > 0
          ? actual.reduce((acc, value, i) => acc + Math.abs(value - (predicted[i] ?? 0)), 0) / actual.length
          : 0,
        4
      ),
      mse: roundTo(mse, 4),
      rmse: roundTo(Math.sqrt(mse), 4),
      r2: roundTo(sst > 1e-12 ? 1 - sse / sst : 0, 4),
      samplePoints: actual
        .slice(0, 60)
        .map((value, i) => ({ actual: roundTo(value, 2), predicted: roundTo(predicted[i] ?? 0, 2) })),
    };
  }

  // --- Bias/variance (tree depth) and regularization (L2 lambda) sweeps ---
  const biasVarianceSweep: { depth: number; trainError: number; testError: number }[] = [];
  for (let depth = 1; depth <= 10; depth++) {
    const tree = new DecisionTreeModel(depth, 41);
    tree.fit(zTrain, yBin.slice(0, trainCut), 2);
    const trainAcc = accuracyOf(yBin.slice(0, trainCut), zTrain.map(x => predictClass(tree, x)));
    const testAcc = processed.length - trainCut > 0
      ? accuracyOf(yBin.slice(trainCut), zTest.map(x => predictClass(tree, x)))
      : 0;
    biasVarianceSweep.push({
      depth,
      trainError: roundTo(1 - trainAcc, 4),
      testError: roundTo(1 - testAcc, 4),
    });
  }

  const regularizationTrain: number[] = [];
  const regularizationTest: number[] = [];
  for (const lambda of REG_LAMBDAS) {
    const model = new LogisticOvR(lambda);
    model.fit(zTrain, yBin.slice(0, trainCut), 2);
    regularizationTrain.push(accuracyOf(yBin.slice(0, trainCut), zTrain.map(x => predictClass(model, x))));
    regularizationTest.push(
      processed.length - trainCut > 0
        ? accuracyOf(yBin.slice(trainCut), zTest.map(x => predictClass(model, x)))
        : 0
    );
  }

  // --- Clustering ---
  const k = chooseClusterCount(processed.length);
  const km = kmeansCluster(zX, k, KMEANS_ITERS, 17);
  const db = dbscanCluster(zX, DBSCAN_MIN_PTS);
  const hier = agglomerativeSingle(zX, km.centroids.length, HIER_SAMPLE);
  const hierSample = evenIndices(zX.length, Math.max(1, Math.min(HIER_SAMPLE, zX.length)));
  const pca = pcaProject2(zX);
  const clustering: ClusteringReport = {
    pca: {
      points: zX.map((_, i) => ({
        x: roundTo(pca.xs[i] ?? 0, 3),
        y: roundTo(pca.ys[i] ?? 0, 3),
        cluster: km.labels[i] ?? 0,
        noise: (db.labels[i] ?? -1) === -1,
      })),
    },
    kmeans: {
      k: km.centroids.length,
      sizes: km.centroids.map((_, c) => km.labels.filter(label => label === c).length),
      inertia: roundTo(km.inertia, 3),
      silhouette: silhouetteScore(zX, km.labels),
    },
    dbscan: {
      eps: roundTo(db.eps, 4),
      minPts: DBSCAN_MIN_PTS,
      sizes: dbscanSizes(db.labels),
      noiseCount: db.labels.filter(label => label === -1).length,
    },
    hierarchical: {
      sizes: hier.sizes,
      silhouette: silhouetteScore(hierSample.map(i => zX[i]!), hier.labels),
    },
  };

  // --- Concept learning + status engine + pipeline preview ---
  const examples = conceptExamples(processed);
  const conceptLearning = {
    finds: findS(examples, CONCEPT_ATTRS),
    ce: candidateElimination(examples, CONCEPT_ATTRS),
  };
  const statusEngine = computeStatusEngine(processed);
  const pipelinePreview = buildPipelinePreview(processed, km.labels);

  return {
    builtAtEpochMs: 0, // stamped by the manager (workers avoid Date.now)
    sourceCount,
    droppedCount,
    preprocess: preprocessSummary,
    classification: [bandReport, binReport],
    regression,
    clustering,
    ensembles: { rows: ensEval.rows, featureImportance },
    biasVariance: { sweep: { values: biasVarianceSweep } },
    regularization: {
      lambdas: REG_LAMBDAS.slice(),
      trainAccuracy: regularizationTrain,
      testAccuracy: regularizationTest,
    },
    conceptLearning,
    pipelinePreview,
    statusEngine,
  };
}

// ---------------------------------------------------------------------------
// Worker message loop.
// ---------------------------------------------------------------------------

self.onmessage = (event: MessageEvent<RunMessage>) => {
  const message = event.data;
  if (message.type !== 'run') return;
  try {
    const results = runAll(Array.isArray(message.items) ? message.items : []);
    const out: WorkerOut = { type: 'result', id: message.id, results };
    self.postMessage(out);
  } catch (error) {
    const out: WorkerOut = {
      type: 'error',
      id: message.id,
      error: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(out);
  }
};

// Signal readiness to the manager.
self.postMessage({ type: 'ready' } satisfies WorkerOut);
