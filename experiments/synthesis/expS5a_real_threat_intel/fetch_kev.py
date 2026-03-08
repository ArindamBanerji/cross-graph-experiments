"""
fetch_kev.py — Fetch and parse CISA Known Exploited Vulnerabilities catalog.
experiments/synthesis/expS5a_real_threat_intel/fetch_kev.py

Fetches from: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
Caches to: kev_raw.json (skip fetch if exists)
Saves parsed claims to: kev_claims.json

Claim fields match the ThreatClaim schema used by run.py.
All KEV entries are actively exploited → tier=1, direction=+1, actions=[escalate_incident, enrich_and_watch].
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, date, timezone
from pathlib import Path

EXP_DIR = Path(__file__).parent
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
RAW_PATH    = EXP_DIR / "kev_raw.json"
CLAIMS_PATH = EXP_DIR / "kev_claims.json"

MAX_AGE_DAYS = 730   # 2-year window

# Canonical SOC taxonomy (must match SOC domain config exactly)
SOC_CATEGORIES: list[str] = [
    "travel_anomaly", "credential_access", "threat_intel_match",
    "insider_behavioral", "cloud_infrastructure",
]
SOC_ACTIONS: list[str] = ["escalate", "investigate", "suppress", "monitor"]


def infer_soc_categories(text: str) -> list[str]:
    """Map KEV description text to canonical SOC categories. Cap at 3."""
    text = text.lower()
    matched: list[str] = []
    if any(k in text for k in [
        "credential", "authentication", "login", "password",
        "oauth", "saml", "ldap", "active directory", "kerberos",
        "brute force", "stuffing"]):
        matched.append("credential_access")
    if any(k in text for k in [
        "phishing", "campaign", "threat actor", "apt", "ioc",
        "indicator", "malware", "ransomware", "backdoor", "c2",
        "command and control"]):
        matched.append("threat_intel_match")
    if any(k in text for k in [
        "vpn", "travel", "geolocation", "impossible travel",
        "anomalous login location", "off-hours"]):
        matched.append("travel_anomaly")
    if any(k in text for k in [
        "exfiltration", "data theft", "insider", "lateral movement",
        "privilege escalation", "data leak", "unauthorized access",
        "policy violation"]):
        matched.append("insider_behavioral")
    if any(k in text for k in [
        "remote code", "rce", "code execution", "command injection",
        "buffer overflow", "deserialization", "cloud", "infrastructure",
        "container", "kubernetes", "aws", "azure", "api", "network"]):
        matched.append("cloud_infrastructure")
    if not matched:
        matched = ["cloud_infrastructure", "credential_access"]
    assert all(c in SOC_CATEGORIES for c in matched), f"Bad category: {matched}"
    return matched[:3]


def infer_promoted_actions(urgency: float) -> list[str]:
    """CVSS/KEV urgency → canonical SOC actions. Never suppress on threat intel."""
    if urgency >= 0.7:
        return ["escalate", "investigate"]
    elif urgency >= 0.4:
        return ["investigate", "monitor"]
    else:
        return ["monitor"]


def fetch_raw() -> dict:
    """Fetch KEV JSON from CISA or load from cache."""
    if RAW_PATH.exists():
        print(f"  [KEV] Loading from cache: {RAW_PATH}")
        with open(RAW_PATH, encoding="utf-8") as f:
            return json.load(f)

    print(f"  [KEV] Fetching from CISA: {KEV_URL}")
    req = urllib.request.Request(
        KEV_URL,
        headers={"User-Agent": "CrossGraphExperiments/1.0 (research)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"[KEV] Fetch failed: {e}") from e

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    print(f"  [KEV] Cached to {RAW_PATH}")
    return raw


def parse_claims(raw: dict) -> list[dict]:
    """Parse KEV catalog into ThreatClaim dicts."""
    today      = date.today()
    vulns      = raw.get("vulnerabilities", [])
    claims: list[dict] = []
    skipped = 0

    for entry in vulns:
        date_added_str = entry.get("dateAdded", "")
        try:
            date_added = datetime.strptime(date_added_str, "%Y-%m-%d").date()
        except ValueError:
            skipped += 1
            continue

        age_days = (today - date_added).days
        if age_days > MAX_AGE_DAYS:
            skipped += 1
            continue

        text    = " ".join([
            entry.get("shortDescription", ""),
            entry.get("vendorProject", ""),
            entry.get("product", ""),
            entry.get("vulnerabilityName", ""),
        ])
        cats    = infer_soc_categories(text)
        actions = infer_promoted_actions(0.8)   # KEV = always high urgency
        assert all(a in SOC_ACTIONS for a in actions), f"Bad action: {actions}"

        claims.append({
            "type":                  "cve_actively_exploited",
            "cve_id":                entry.get("cveID", "UNKNOWN"),
            "source":                "CISA_KEV",
            "tier":                  1,
            "direction":             1,
            "strength":              0.8,
            "confidence":            1.0,
            "extraction_confidence": 1.0,
            "urgency":               0.8,
            "age_days":              age_days,
            "decay_class":           "campaign",
            "categories_affected":   cats,
            "actions_promoted":      actions,
            "description":           entry.get("shortDescription", "")[:200],
            "vendor":                entry.get("vendorProject", ""),
            "product":               entry.get("product", ""),
            "date_added":            date_added_str,
        })

    return claims, skipped


def run() -> list[dict]:
    """Main entry point: fetch, parse, save, print summary."""
    print("=== CISA KEV Fetch ===")
    raw    = fetch_raw()
    vulns  = raw.get("vulnerabilities", [])
    total  = len(vulns)

    claims, skipped = parse_claims(raw)
    filtered = total - skipped

    # Category distribution
    cat_counts: dict[str, int] = defaultdict(int)
    for cl in claims:
        for c in cl["categories_affected"]:
            cat_counts[c] += 1

    print(f"  Total entries in catalog:   {total}")
    print(f"  Entries within 2-year window: {filtered}")
    print(f"  Claims generated:           {len(claims)}")
    print(f"  Category distribution:      {dict(cat_counts)}")

    if claims:
        sorted_by_recency = sorted(claims, key=lambda c: c["age_days"])[:5]
        print(f"\n  Top 5 most recent claims:")
        for cl in sorted_by_recency:
            print(f"    {cl['cve_id']:20s}  vendor={cl['vendor'][:20]:20s}  "
                  f"cats={cl['categories_affected']}  age={cl['age_days']}d")

    with open(CLAIMS_PATH, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)
    print(f"\n  Saved {len(claims)} claims to {CLAIMS_PATH}")

    return claims


if __name__ == "__main__":
    claims = run()
    print(f"\nfetch_kev.py done — {len(claims)} claims")
    sys.exit(0)
