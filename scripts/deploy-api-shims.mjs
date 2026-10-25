// Deploy-time static shims for GitHub Pages (no backend there).
// Writes ONLY into the build output (dist/). Run after `npx vite build`:
//   node scripts/deploy-api-shims.mjs [distDir]
// The real /api functions still win on any serverless deployment — these
// files are shadowed there, so this never affects the source repo.
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const dist = process.argv[2] || 'dist';
if (!existsSync(dist)) {
  console.error(`dist dir not found: ${dist}`);
  process.exit(1);
}

const severity = (level) =>
  ({ critical: 'SEVERITY_LEVEL_HIGH', high: 'SEVERITY_LEVEL_HIGH', elevated: 'SEVERITY_LEVEL_MEDIUM' })[String(level || '').toLowerCase()] || 'SEVERITY_LEVEL_LOW';

// 1. get-risk-scores — merge the published CII snapshot (Tier-1, has 24h
// movement) with the per-country cii.json snapshots (wider coverage).
const ranking = JSON.parse(readFileSync('public/country-instability-index/cii-ranking.json', 'utf8'));
const scores = new Map();
for (const c of ranking.countries || []) {
  scores.set(c.code, {
    region: c.code,
    staticBaseline: c.score,
    dynamicScore: c.approximate24HourMovement ?? 0,
    combinedScore: c.score,
    trend: (c.approximate24HourMovement ?? 0) > 1 ? 'TREND_DIRECTION_RISING' : (c.approximate24HourMovement ?? 0) < -1 ? 'TREND_DIRECTION_FALLING' : 'TREND_DIRECTION_STABLE',
    computedAt: Date.parse(c.observedAt || ranking.capturedAt) || Date.now(),
    methodologyVersion: ranking.methodologyVersion || 'v8',
    eventMultiplier: 1,
    advisoryLevel: severity(c.instabilityLevel),
    advisoryProvenance: 'published-cii-snapshot',
  });
}
const countriesDir = 'public/countries';
if (existsSync(countriesDir)) {
  for (const slug of readdirSync(countriesDir)) {
    const p = join(countriesDir, slug, 'cii.json');
    if (!existsSync(p)) continue;
    try {
      const c = JSON.parse(readFileSync(p, 'utf8'));
      if (!c.countryCode || scores.has(c.countryCode)) continue;
      scores.set(c.countryCode, {
        region: c.countryCode,
        staticBaseline: c.score,
        dynamicScore: c.approximate24HourMovement ?? 0,
        combinedScore: c.score,
        trend: 'TREND_DIRECTION_STABLE',
        computedAt: Date.parse(c.observedAt || c.capturedAt) || Date.now(),
        methodologyVersion: c.methodologyVersion || 'v8',
        eventMultiplier: 1,
        advisoryLevel: severity(c.instabilityLevel),
        advisoryProvenance: 'published-cii-snapshot',
      });
    } catch { /* skip malformed snapshots */ }
  }
}
const ciiScores = [...scores.values()].sort((a, b) => b.combinedScore - a.combinedScore);

const riskScores = {
  ciiScores,
  strategicRisks: [],
  degraded: false,
  stale: true, // honest: this is a published snapshot, not a live computation
};

// 2. bootstrap — valid empty payload: keys hydrate as cleanly unavailable.
const bootstrap = { data: {}, partial: false, generatedAt: null };

// 3. rss-proxy — valid empty feed so country news parses instead of 404ing.
const rss = '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel><title>Global Status Monitoring System</title><link>https://rave-88.github.io/</link><description>Static deployment feed</description></channel></rss>';

// 4. Country-facts interceptor — deploy-only. The per-country facts RPC
// cannot be a static file (query varies per country), so on the static
// deployment we compose the same response shape live from CORS-open public
// sources (restcountries.com + Wikipedia REST). Falls through to the real
// network call on any failure, which the UI already renders as unavailable.
const isoNames = {};
const wikiOverrides = { GE: 'Georgia (country)' };
const addName = (code, name) => { if (code && name && !isoNames[code]) isoNames[code] = name; };
for (const c of ranking.countries || []) addName(c.code, c.name);
if (existsSync(countriesDir)) {
  for (const slug of readdirSync(countriesDir)) {
    const p2 = join(countriesDir, slug, 'cii.json');
    if (!existsSync(p2)) continue;
    try { const c = JSON.parse(readFileSync(p2, 'utf8')); addName(c.countryCode, c.countryName); } catch {}
  }
}

const FACTS_SHIM = `<script>
(function () {
  if (window.__gsmsFactsShim) return;
  window.__gsmsFactsShim = true;
  var NAMES = ${JSON.stringify(isoNames)};
  var OVERRIDES = ${JSON.stringify(wikiOverrides)};
  var orig = window.fetch ? window.fetch.bind(window) : null;
  if (!orig) return;
  window.fetch = async function (input, init) {
    try {
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      if (url.indexOf('/api/intelligence/v1/get-country-facts?') !== -1 && (!init || !init.method || init.method === 'GET')) {
        var iso = ((new URL(url, location.origin)).searchParams.get('country_code') || '').toUpperCase();
        var name = OVERRIDES[iso] || NAMES[iso];
        if (name) {
          var summary = '', thumb = '';
          var wiki = await orig('https://en.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(name));
          if (wiki.ok) {
            var w = await wiki.json();
            summary = w.extract || '';
            thumb = (w.thumbnail && w.thumbnail.source) || '';
          }
          var body = {
            headOfState: '',
            headOfStateTitle: '',
            wikipediaSummary: summary,
            wikipediaThumbnailUrl: thumb,
            population: 0,
            capital: '',
            languages: [],
            currencies: [],
            areaSqKm: 0,
            countryName: name
          };
          return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
        }
      }
    } catch (e) { /* fall through to the real network call */ }
    return orig(input, init);
  };
})();
</script>`;

function injectFactsShim(htmlPath) {
  const target = join(dist, htmlPath);
  if (!existsSync(target)) return;
  let html = readFileSync(target, 'utf8');
  // Replace any previously injected shim block, then inject before the first script.
  html = html.replace(/<script>\s*\(function \(\) \{\s*if \(window\.__gsmsFactsShim\)[\s\S]*?<\/script>\n?/, '');
  const at = html.indexOf('<script');
  if (at === -1) return;
  html = html.slice(0, at) + FACTS_SHIM + '\n' + html.slice(at);
  writeFileSync(target, html);
  console.log(`injected facts shim into ${htmlPath}`);
}

for (const entry of ['index.html', 'dashboard.html']) injectFactsShim(entry);

const files = [
  ['api/intelligence/v1/get-risk-scores', JSON.stringify(riskScores)],
  ['api/bootstrap', JSON.stringify(bootstrap)],
  ['api/rss-proxy', rss],
];
for (const [rel, body] of files) {
  const target = join(dist, rel);
  mkdirSync(join(target, '..'), { recursive: true });
  writeFileSync(target, body);
  console.log(`wrote ${target} (${body.length} bytes)`);
}
console.log(`ciiScores: ${ciiScores.length} countries`);
