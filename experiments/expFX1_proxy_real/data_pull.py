"""
data_pull.py — Public API fetchers for FX1-PROXY-REAL.

Sources:
  - CISA KEV:       https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
  - NVD CVE 2.0:    https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=200
  - MITRE ATT&CK:   https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json

Caches responses to data/raw/. Re-uses cache if <24 hours old.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _DIR / "data" / "raw"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TIMEOUT_S  = 30
_CACHE_AGE_S = 24 * 3600   # 24 hours

CISA_KEV_URL   = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
NVD_CVE_URL    = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=200"
MITRE_ATT_URL  = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_AGE_S


def _fetch_url(url: str, name: str) -> dict | list | None:
    """Fetch URL with caching. Returns parsed JSON or None on error."""
    cache = _cache_path(name)
    if _cache_valid(cache):
        print(f"  [CACHE] {name} — using cached copy ({cache.name})")
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)

    print(f"  [FETCH] {name} — pulling from {url[:70]}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "cross-graph-research/1.0 (academic)"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read()
        data = json.loads(raw)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        print(f"  [FETCH] {name} — saved to cache ({len(raw)//1024} KB)")
        return data
    except urllib.error.URLError as exc:
        print(f"  [ERROR] {name} fetch failed: {exc}")
        return None
    except Exception as exc:
        print(f"  [ERROR] {name} unexpected error: {exc}")
        return None


def fetch_cisa_kev() -> list[dict]:
    """Pull CISA KEV JSON. Returns list of vulnerability dicts."""
    data = _fetch_url(CISA_KEV_URL, "cisa_kev")
    if data is None:
        return []
    vulns = data.get("vulnerabilities", [])
    print(f"  [DATA] CISA KEV: {len(vulns)} records")
    return vulns


def fetch_nvd_cves(max_results: int = 200) -> list[dict]:
    """Pull single page of NVD CVEs. Returns list of CVE item dicts."""
    url  = f"https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage={max_results}"
    data = _fetch_url(url, "nvd_cves")
    if data is None:
        return []
    items = data.get("vulnerabilities", [])
    print(f"  [DATA] NVD CVEs:  {len(items)} records")
    return items


def fetch_mitre_attack() -> list[dict]:
    """Pull ATT&CK STIX bundle. Returns list of technique objects."""
    data = _fetch_url(MITRE_ATT_URL, "mitre_attack")
    if data is None:
        return []
    objects = data.get("objects", [])
    techniques = [
        o for o in objects
        if o.get("type") == "attack-pattern"
        and not o.get("revoked", False)
        and not o.get("x_mitre_deprecated", False)
    ]
    print(f"  [DATA] MITRE ATT&CK: {len(techniques)} techniques "
          f"(of {len(objects)} total objects)")
    return techniques


if __name__ == "__main__":
    print("Testing data_pull.py ...")
    kev   = fetch_cisa_kev()
    cves  = fetch_nvd_cves()
    techs = fetch_mitre_attack()
    print(f"\nSummary: KEV={len(kev)}, NVD={len(cves)}, ATT&CK={len(techs)}")
