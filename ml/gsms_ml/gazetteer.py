"""Headline country recognition (gazetteer NER).

A gazetteer matcher is the interpretable, dependency-free form of NER that the
dashboard itself relies on for country attribution (see
``src/services/country-geometry.ts`` / curated keyword tables). This module
maps headline mentions of countries, capitals and famous places to ISO-2
codes, restricted to a committed table of entries that appear in global news
with meaningful frequency. It is a derived heuristic: multi-name regions
(e.g. "Gaza") map to their governing entity, and any headline naming no
tabled entry gets an empty country set rather than a guess.
"""

from __future__ import annotations

import re

# name → ISO-2. Keys are lowercase multi-word phrases matched on word
# boundaries; longest match wins.
_GAZETTEER: dict[str, str] = {
    "united states": "US", "u s": "US", "us": "US", "america": "US",
    "american": "US", "washington": "US", "pentagon": "US", "white house": "US",
    "washington dc": "US",
    "russia": "RU", "russian": "RU", "moscow": "RU", "kremlin": "RU",
    "putin": "RU",
    "ukraine": "UA", "ukrainian": "UA", "kyiv": "UA", "kiev": "UA",
    "zelensky": "UA", "zelenskyy": "UA", "donbas": "UA", "crimea": "UA",
    "china": "CN", "chinese": "CN", "beijing": "CN", "xi jinping": "CN",
    "peking": "CN",
    "taiwan": "TW", "taipei": "TW",
    "india": "IN", "indian": "IN", "delhi": "IN", "new delhi": "IN",
    "mumbai": "IN", "modi": "IN",
    "pakistan": "PK", "pakistani": "PK", "islamabad": "PK", "karachi": "PK",
    "iran": "IR", "iranian": "IR", "tehran": "IR", "isfahan": "IR",
    "tabriz": "IR", "khamenei": "IR", "irgc": "IR",
    "israel": "IL", "israeli": "IL", "jerusalem": "IL", "tel aviv": "IL",
    "netanyahu": "IL",
    "gaza": "PS", "palestin": "PS", "west bank": "PS", "ramallah": "PS",
    "west bank and gaza": "PS",
    "lebanon": "LB", "beirut": "LB", "hezbollah": "LB",
    "syria": "SY", "syrian": "SY", "damascus": "SY", "aleppo": "SY",
    "iraq": "IQ", "iraqi": "IQ", "baghdad": "IQ",
    "yemen": "YE", "sanaa": "YE", "houthi": "YE", "houthis": "YE",
    "saudi": "SA", "saudi arabia": "SA", "riyadh": "SA",
    "uae": "AE", "united arab emirates": "AE", "dubai": "AE", "abu dhabi": "AE",
    "qatar": "QA", "doha": "QA",
    "egypt": "EG", "cairo": "EG", "sisi": "EG",
    "turkey": "TR", "turkiye": "TR", "turkish": "TR", "ankara": "TR",
    "istanbul": "TR", "erdogan": "TR",
    "iran": "IR",
    "jordan": "JO", "amman": "JO",
    "syria": "SY",
    "afghanistan": "AF", "kabul": "AF", "taliban": "AF",
    "pakistan": "PK",
    "north korea": "KP", "pyongyang": "KP", "dprk": "KP",
    "south korea": "KR", "seoul": "KR",
    "japan": "JP", "japanese": "JP", "tokyo": "JP",
    "china": "CN",
    "taiwan": "TW",
    "vietnam": "VN", "hanoi": "VN",
    "thailand": "TH", "bangkok": "TH",
    "indonesia": "ID", "jakarta": "ID",
    "malaysia": "MY", "kuala lumpur": "MY",
    "philippines": "PH", "manila": "PH",
    "singapore": "SG",
    "myanmar": "MM", "burma": "MM", "yangon": "MM",
    "bangladesh": "BD", "dhaka": "BD",
    "sri lanka": "LK", "colombo": "LK",
    "nepal": "NP", "kathmandu": "NP",
    "pakistan": "PK",
    "australia": "AU", "australian": "AU", "canberra": "AU", "sydney": "AU",
    "new zealand": "NZ", "auckland": "NZ", "wellington": "NZ",
    "uk": "GB", "u k": "GB", "britain": "GB", "british": "GB",
    "united kingdom": "GB", "england": "GB", "london": "GB",
    "downing street": "GB", "scotland": "GB", "wales": "GB",
    "france": "FR", "french": "FR", "paris": "FR", "macron": "FR",
    "germany": "DE", "german": "DE", "berlin": "DE", "munich": "DE",
    "scholz": "DE", "merz": "DE",
    "italy": "IT", "italian": "IT", "rome": "IT",
    "spain": "ES", "spanish": "ES", "madrid": "ES", "barcelona": "ES",
    "portugal": "PT", "lisbon": "PT",
    "netherlands": "NL", "dutch": "NL", "amsterdam": "NL", "the hague": "NL",
    "belgium": "BE", "brussels": "BE",
    "switzerland": "CH", "geneva": "CH", "davos": "CH", "zurich": "CH",
    "austria": "AT", "vienna": "AT",
    "sweden": "SE", "stockholm": "SE",
    "norway": "NO", "oslo": "NO",
    "denmark": "DK", "copenhagen": "DK",
    "finland": "FI", "helsinki": "FI",
    "poland": "PL", "polish": "PL", "warsaw": "PL",
    "czech republic": "CZ", "prague": "CZ",
    "hungary": "HU", "budapest": "HU", "orban": "HU",
    "romania": "RO", "bucharest": "RO",
    "bulgaria": "BG", "sofia": "BG",
    "greece": "GR", "athens": "GR",
    "turkey": "TR",
    "ukraine": "UA",
    "ireland": "IE", "dublin": "IE",
    "portugal": "PT",
    "iceland": "IS", "reykjavik": "IS",
    "ukraine": "UA",
    "canada": "CA", "canadian": "CA", "ottawa": "CA", "toronto": "CA",
    "ottawa": "CA",
    "mexico": "MX", "mexican": "MX", "mexico city": "MX",
    "brazil": "BR", "brazilian": "BR", "brasilia": "BR", "sao paulo": "BR",
    "lula": "BR",
    "argentina": "AR", "buenos aires": "AR", "milei": "AR",
    "chile": "CL", "santiago": "CL",
    "colombia": "CO", "bogota": "CO",
    "peru": "PE", "lima": "PE",
    "venezuela": "VE", "caracas": "VE", "maduro": "VE",
    "ecuador": "EC", "quito": "EC",
    "bolivia": "BO", "la paz": "BO",
    "uruguay": "UY", "montevideo": "UY",
    "paraguay": "PY", "asuncion": "PY",
    "south africa": "ZA", "pretoria": "ZA", "johannesburg": "ZA",
    "nigeria": "NG", "abuja": "NG", "lagos": "NG",
    "kenya": "KE", "nairobi": "KE",
    "ethiopia": "ET", "addis ababa": "ET",
    "egypt": "EG",
    "sudan": "SD", "khartoum": "SD",
    "libya": "LY", "tripoli": "LY",
    "algeria": "DZ", "algiers": "DZ",
    "morocco": "MA", "rabat": "MA", "marrakech": "MA",
    "tunisia": "TN", "tunis": "TN",
    "ghana": "GH", "accra": "GH",
    "somalia": "SO", "mogadishu": "SO",
    "ethiopia": "ET",
    "mali": "ML", "bamako": "ML",
    "niger": "NE", "niamey": "NE",
    "chad": "TD", "ndjamena": "TD",
    "cameroon": "CM", "yaounde": "CM",
    "congo": "CD", "kinshasa": "CD", "drc": "CD",
    "zimbabwe": "ZW", "harare": "ZW",
    "zambia": "ZM", "lusaka": "ZM",
    "mozambique": "MZ", "maputo": "MZ",
    "angola": "AO", "luanda": "AO",
    "uganda": "UG", "kampala": "UG",
    "tanzania": "TZ", "dodoma": "TZ",
    "ethiopia": "ET",
    "ukraine": "UA",
    "taiwan strait": "TW", "south china sea": "CN", "east china sea": "CN",
    "taipei": "TW",
    "kosovo": "XK", "pristina": "XK",
    "serbia": "RS", "belgrade": "RS",
    "croatia": "HR", "zagreb": "HR",
    "slovakia": "SK", "bratislava": "SK",
    "slovenia": "SI", "ljubljana": "SI",
    "bosnia": "BA", "sarajevo": "BA",
    "albania": "AL", "tirana": "AL",
    "north macedonia": "MK", "skopje": "MK",
    "moldova": "MD", "chisinau": "MD",
    "belarus": "BY", "minsk": "BY", "lukashenko": "BY",
    "ukraine": "UA",
    "georgia": "GE", "tbilisi": "GE",
    "armenia": "AM", "yerevan": "AM",
    "azerbaijan": "AZ", "baku": "AZ",
    "kazakhstan": "KZ", "astana": "KZ",
    "uzbekistan": "UZ", "tashkent": "UZ",
    "turkmenistan": "TM", "ashgabat": "TM",
    "kyrgyzstan": "KG", "bishkek": "KG",
    "tajikistan": "TJ", "dushanbe": "TJ",
    "mongolia": "MN", "ulaanbaatar": "MN",
    "taiwan": "TW",
    "vietnam": "VN",
    "cambodia": "KH", "phnom penh": "KH",
    "laos": "LA", "vientiane": "LA",
    "north korea": "KP",
    "taiwan": "TW",
    "papua new guinea": "PG", "port moresby": "PG",
    "fiji": "FJ", "suva": "FJ",
    "taiwan": "TW",
    "venezuela": "VE",
    "guyana": "GY", "georgetown": "GY",
    "suriname": "SR", "paramaribo": "SR",
    "panama": "PA", "panama city": "PA",
    "costa rica": "CR", "san jose": "CR",
    "nicaragua": "NI", "managua": "NI",
    "honduras": "HN", "tegucigalpa": "HN",
    "el salvador": "SV", "san salvador": "SV",
    "guatemala": "GT", "guatemala city": "GT",
    "cuba": "CU", "havana": "CU",
    "haiti": "HT", "port au prince": "HT",
    "dominican republic": "DO", "santo domingo": "DO",
    "jamaica": "JM", "kingston": "JM",
    "puerto rico": "PR", "san juan": "PR",
    "greenland": "GL", "nuuk": "GL",
    "taiwan": "TW",
}

_ABBREVIATION_FIXES = re.compile(r"\bu\.s\.|\bu\.k\.|\bd\.c\.")


def detect_countries(text: str) -> list[str]:
    """Return deduplicated ISO-2 codes for countries named in `text`.

    Longest phrase wins; matches respect word boundaries so "us" inside
    "russo" never matches. Ordering follows first appearance.
    """
    lowered = _ABBREVIATION_FIXES.sub(
        lambda m: {"u.s.": "us", "u.k.": "uk", "d.c.": "dc"}[m.group(0)],
        text.lower(),
    )
    found: list[str] = []
    for name in sorted(_GAZETTEER, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(name) + r"(?![a-z])"
        if re.search(pattern, lowered):
            code = _GAZETTEER[name]
            if code not in found:
                found.append(code)
    return found


# GDELT source-country and gazetteer codes → macro region.
REGION_BY_CODE: dict[str, str] = {
    "US": "North America", "CA": "North America", "MX": "North America",
    "GL": "North America", "PR": "North America",
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "PE": "South America", "VE": "South America",
    "EC": "South America", "BO": "South America", "UY": "South America",
    "PY": "South America", "GY": "South America", "SR": "South America",
    "GB": "Europe", "IE": "Europe", "FR": "Europe", "DE": "Europe",
    "IT": "Europe", "ES": "Europe", "PT": "Europe", "NL": "Europe",
    "BE": "Europe", "CH": "Europe", "AT": "Europe", "SE": "Europe",
    "NO": "Europe", "DK": "Europe", "FI": "Europe", "PL": "Europe",
    "CZ": "Europe", "HU": "Europe", "RO": "Europe", "BG": "Europe",
    "GR": "Europe", "IS": "Europe", "UA": "Europe", "BY": "Europe",
    "MD": "Europe", "RS": "Europe", "HR": "Europe", "SK": "Europe",
    "SI": "Europe", "BA": "Europe", "AL": "Europe", "MK": "Europe",
    "XK": "Europe", "LT": "Europe", "LV": "Europe", "EE": "Europe",
    "RU": "Europe",
    "TR": "Middle East", "SY": "Middle East", "LB": "Middle East",
    "IL": "Middle East", "PS": "Middle East", "JO": "Middle East",
    "IQ": "Middle East", "IR": "Middle East", "SA": "Middle East",
    "AE": "Middle East", "QA": "Middle East", "KW": "Middle East",
    "BH": "Middle East", "OM": "Middle East", "YE": "Middle East",
    "CY": "Middle East",
    "EG": "Africa", "LY": "Africa", "TN": "Africa", "DZ": "Africa",
    "MA": "Africa", "SD": "Africa", "ET": "Africa", "SO": "Africa",
    "KE": "Africa", "TZ": "Africa", "UG": "Africa", "RW": "Africa",
    "NG": "Africa", "GH": "Africa", "ML": "Africa", "NE": "Africa",
    "TD": "Africa", "CM": "Africa", "CD": "Africa", "ZA": "Africa",
    "ZW": "Africa", "ZM": "Africa", "MZ": "Africa", "AO": "Africa",
    "SN": "Africa", "CI": "Africa", "BF": "Africa", "GN": "Africa",
    "MW": "Africa", "BW": "Africa", "NA": "Africa", "MG": "Africa",
    "IN": "Asia", "PK": "Asia", "BD": "Asia", "LK": "Asia", "NP": "Asia",
    "BT": "Asia", "MM": "Asia", "TH": "Asia", "VN": "Asia", "KH": "Asia",
    "LA": "Asia", "MY": "Asia", "SG": "Asia", "ID": "Asia", "PH": "Asia",
    "BN": "Asia", "MN": "Asia",
    "CN": "Asia", "JP": "Asia", "KR": "Asia", "KP": "Asia", "TW": "Asia",
    "HK": "Asia", "MO": "Asia", "MN": "Asia",
    "KZ": "Asia", "UZ": "Asia", "TM": "Asia", "KG": "Asia", "TJ": "Asia",
    "AF": "Asia", "GE": "Asia", "AM": "Asia", "AZ": "Asia",
    "AU": "Oceania", "NZ": "Oceania", "PG": "Oceania", "FJ": "Oceania",
}


def region_for_codes(codes: list[str]) -> str:
    """Map the first recognized country code to a macro region, else 'Global'."""
    for code in codes:
        region = REGION_BY_CODE.get(code)
        if region:
            return region
    return "Global"


def region_for_source_country(source_country: str | None) -> str:
    """Map a GDELT source-country code to a macro region, else 'Global'."""
    if not source_country:
        return "Global"
    code = source_country.strip().upper()
    return REGION_BY_CODE.get(code, "Global")
