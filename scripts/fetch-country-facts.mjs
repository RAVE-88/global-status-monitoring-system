// One-shot build-time snapshot of country facts, mirroring the app's own
// server logic in server/worldmonitor/intelligence/v1/get-country-facts.ts
// (Wikidata SPARQL + Wikipedia REST). Writes scripts/country-facts-cache.json
// which deploy-api-shims.mjs bakes into the static deployment.
//   node scripts/fetch-country-facts.mjs
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const UA = 'WorldMonitorCountryFacts-deploysnapshot/1.0 (static deployment snapshot)';
const OUT = 'scripts/country-facts-cache.json';
const unToIso2 = JSON.parse(readFileSync('shared/un-to-iso2.json', 'utf8'));
const ISO2_TO_M49 = Object.fromEntries(Object.entries(unToIso2).map(([m49, iso2]) => [iso2, m49]));

// Country codes covered by the deployment snapshots.
const codes = new Set();
for (const slug of existsSync('public/countries') ? readdirSync('public/countries') : []) {
  try {
    const c = JSON.parse(readFileSync(join('public/countries', slug, 'cii.json'), 'utf8'));
    if (c.countryCode) codes.add(c.countryCode);
  } catch {}
}
import { readdirSync } from 'node:fs';

async function sparql(query) {
  const url = `https://query.wikidata.org/sparql?format=json&query=${encodeURIComponent(query)}`;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const resp = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/sparql-results+json' }, signal: AbortSignal.timeout(30000) });
      if (resp.status === 429 || resp.status >= 500) throw new Error(`status ${resp.status}`);
      if (!resp.ok) throw new Error(`status ${resp.status}`);
      return (await resp.json()).results?.bindings ?? [];
    } catch (e) {
      if (attempt === 2) throw e;
      await new Promise(r => setTimeout(r, 3000 * (attempt + 1)));
    }
  }
}

const clean = v => { const s = (v ?? '').trim(); return /^Q\d+$/.test(s) ? '' : s; };
const unique = values => {
  const seen = new Set(), out = [];
  for (const v of values) {
    const label = clean(v);
    const key = label.toLocaleLowerCase('en');
    if (!label || seen.has(key)) continue;
    seen.add(key); out.push(label);
  }
  return out;
};

// 1. Bulk entity resolution: every country with an ISO-3166-1 alpha-2 code.
const entityBindings = await sparql('SELECT ?country ?countryLabel ?iso2 WHERE { ?country wdt:P297 ?iso2. FILTER NOT EXISTS { ?country wdt:P576 ?dissolved } SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul" } }');
const byCode = new Map();
for (const b of entityBindings) {
  const iso2 = (b.iso2?.value || '').toUpperCase();
  const id = b.country?.value?.match(/\/entity\/(Q\d+)$/)?.[1];
  const label = clean(b.countryLabel?.value);
  if (!iso2 || !id || !ISO2_TO_M49[iso2]) continue;
  if (!byCode.has(iso2)) byCode.set(iso2, { id, label });
}
console.log(`resolved ${byCode.size} entities`);

// 2. Bulk facts query for all resolved entities, grouped per country.
const entities = [...byCode.values()];
const values = entities.map(e => `wd:${e.id}`).join(' ');
const factsBindings = await sparql(`SELECT ?country ?headLabel ?officeLabel ?population ?area ?capitalLabel ?languageLabel ?currencyLabel WHERE { VALUES ?country { ${values} } OPTIONAL { ?country p:P35 ?headStatement. ?headStatement ps:P35 ?head. FILTER NOT EXISTS { ?headStatement pq:P582 ?headEnd } OPTIONAL { ?headStatement pq:P39 ?office } } OPTIONAL { ?country wdt:P1082 ?population } OPTIONAL { ?country wdt:P2046 ?area } OPTIONAL { ?country wdt:P36 ?capital } OPTIONAL { ?country p:P37 ?languageStatement. ?languageStatement ps:P37 ?language. FILTER NOT EXISTS { ?languageStatement pq:P518 ?languageRegion } FILTER NOT EXISTS { ?languageStatement pq:P582 ?languageEnd } } OPTIONAL { ?country p:P38 ?currencyStatement. ?currencyStatement ps:P38 ?currency. FILTER NOT EXISTS { ?currencyStatement pq:P582 ?currencyEnd } } SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul" } }`);
console.log(`facts bindings: ${factsBindings.length}`);

const groups = new Map();
for (const b of factsBindings) {
  const id = b.country?.value?.match(/\/entity\/(Q\d+)$/)?.[1];
  if (!id) continue;
  if (!groups.has(id)) groups.set(id, []);
  groups.get(id).push(b);
}

function firstLabel(bindings, field) {
  for (const b of bindings) { const l = clean(b[field]?.value); if (l) return l; }
  return '';
}
function firstNumber(bindings, field) {
  for (const b of bindings) { const v = Number(b[field]?.value); if (Number.isFinite(v) && v > 0) return v; }
  return 0;
}

const displayName = code => { try { return new Intl.DisplayNames(['en'], { type: 'region' }).of(code) ?? ''; } catch { return ''; } };

const cache = {};
for (const [code, info] of byCode) {
  const bindings = groups.get(info.id) ?? [];
  cache[code] = {
    headOfState: firstLabel(bindings, 'headLabel'),
    headOfStateTitle: firstLabel(bindings, 'officeLabel'),
    population: Math.trunc(firstNumber(bindings, 'population')),
    areaSqKm: firstNumber(bindings, 'area'),
    capital: firstLabel(bindings, 'capitalLabel'),
    languages: unique(bindings.map(b => b.languageLabel?.value)),
    currencies: unique(bindings.map(b => b.currencyLabel?.value)),
    countryName: displayName(code) || info.label || firstLabel(bindings, 'countryLabel'),
    wikipediaSummary: '',
    wikipediaThumbnailUrl: '',
  };
}
console.log(`parsed facts for ${Object.keys(cache).length} countries`);

// 3. Wikipedia summaries (same source the server uses).
let done = 0;
for (const [code, f] of Object.entries(cache)) {
  try {
    const resp = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(f.countryName)}`, { headers: { 'User-Agent': UA }, signal: AbortSignal.timeout(15000) });
    if (resp.ok) {
      const data = await resp.json();
      f.wikipediaSummary = data.extract ?? '';
      f.wikipediaThumbnailUrl = data.thumbnail?.source ?? '';
    }
  } catch {}
  done += 1;
  if (done % 40 === 0) console.log(`summaries ${done}/${Object.keys(cache).length}`);
  await new Promise(r => setTimeout(r, 120));
}
writeFileSync(OUT, JSON.stringify(cache));
console.log(`wrote ${OUT} (${Object.keys(cache).length} countries)`);
