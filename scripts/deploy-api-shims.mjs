// Deploy-time static shims for GitHub Pages (no backend there).
// Writes ONLY into the build output (dist/). Run after `npx vite build`:
//   node scripts/deploy-api-shims.mjs [distDir]
// The real /api functions still win on any serverless deployment — these
// files are shadowed there, so this never affects the source repo.
//
// Country facts come from scripts/country-facts-cache.json — build/refresh it
// with `node scripts/fetch-country-facts.mjs` (Wikidata + Wikipedia snapshot,
// mirroring server/worldmonitor/intelligence/v1/get-country-facts.ts).
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
    computedAt: Date.parse(c.observedAt || ranking.capturedAt) || 0,
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
        computedAt: Date.parse(c.observedAt || c.capturedAt) || 0,
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

// 4. Country facts — one static snapshot file (built by
// scripts/fetch-country-facts.mjs from the same Wikidata + Wikipedia sources
// the real backend uses) + a tiny fetch interceptor in the entry pages that
// answers the per-country RPC from it. Falls through to the real network
// call when the snapshot lacks the code, which the UI renders as unavailable.
const factsCachePath = 'scripts/country-facts-cache.json';
const factsCache = existsSync(factsCachePath) ? JSON.parse(readFileSync(factsCachePath, 'utf8')) : null;

const FACTS_SHIM = `<script>
(function () {
  if (window.__gsmsFactsShim) return;
  window.__gsmsFactsShim = true;
  var orig = window.fetch ? window.fetch.bind(window) : null;
  if (!orig) return;
  var facts = null;
  window.fetch = async function (input, init) {
    try {
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      if (url.indexOf('/api/intelligence/v1/get-country-facts?') !== -1 && (!init || !init.method || init.method === 'GET')) {
        var iso = ((new URL(url, location.origin)).searchParams.get('country_code') || '').toUpperCase();
        if (iso) {
          if (!facts) {
            var fr = await orig('/api-shim/country-facts.json');
            facts = fr.ok ? await fr.json() : {};
          }
          var f = facts[iso];
          if (f) return new Response(JSON.stringify(f), { status: 200, headers: { 'Content-Type': 'application/json' } });
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

// 5. Higher-precision country boundaries for hover/click hit-testing. The
// repo ships a coarse countries.geojson; Natural Earth 50m keeps the click
// resolution consistent with the drawn 50m borders. Also upgrade the mobile
// base topology from 110m to 50m so drawn borders match everywhere.
const boundariesSrc = 'scripts/ne-50m-country-boundaries.geojson';
if (existsSync(boundariesSrc)) {
  files.push(['data/countries.geojson', readFileSync(boundariesSrc)]);
  console.log('country boundaries: Natural Earth 50m');
}
if (factsCache) {
  files.push(['api-shim/country-facts.json', JSON.stringify(factsCache)]);
  console.log(`country facts snapshot: ${Object.keys(factsCache).length} countries`);
} else {
  console.log('no country-facts snapshot found (run scripts/fetch-country-facts.mjs)');
}
for (const [rel, body] of files) {
  const target = join(dist, rel);
  mkdirSync(join(target, '..'), { recursive: true });
  writeFileSync(target, body);
  console.log(`wrote ${target} (${body.length} bytes)`);
}

// Mobile topology: 110m -> 50m in the built bundle so drawn borders are as
// precise as the hover/click geometry everywhere (deploy-only string patch).
const assetsDir = join(dist, 'assets');
if (existsSync(assetsDir)) {
  let patched = 0;
  for (const f of readdirSync(assetsDir)) {
    if (!f.endsWith('.js')) continue;
    const fp = join(assetsDir, f);
    const content = readFileSync(fp, 'utf8');
    if (content.includes('/data/countries-110m.json')) {
      writeFileSync(fp, content.replaceAll('/data/countries-110m.json', '/data/countries-50m.json'));
      patched += 1;
    }
  }
  if (patched) console.log(`patched mobile topology to 50m in ${patched} bundle(s)`);
}
console.log(`ciiScores: ${ciiScores.length} countries`);
