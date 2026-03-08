"""
fetch_nvd.py — Fetch and parse recent CVEs from NIST NVD API 2.0.
experiments/synthesis/expS5a_real_threat_intel/fetch_nvd.py

Fetches last 30 days, max 20 CVEs.
Caches to: nvd_raw.json (skip fetch if exists)
Saves parsed claims to: nvd_claims.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

EXP_DIR  = Path(__file__).parent
NVD_URL  = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RAW_PATH    = EXP_DIR / "nvd_raw.json"
CLAIMS_PATH = EXP_DIR / "nvd_claims.json"

MAX_RESULTS = 20
WINDOW_DAYS = 30

# Canonical SOC taxonomy (must match SOC domain config exactly)
SOC_CATEGORIES: list[str] = [
    "travel_anomaly", "credential_access", "threat_intel_match",
    "insider_behavioral", "cloud_infrastructure",
]
SOC_ACTIONS: list[str] = ["escalate", "investigate", "suppress", "monitor"]


def cvss_to_urgency(score: float) -> tuple[float, int]:
    """Convert CVSS base score to (urgency, tier)."""
    if score >= 9.0:
        return 0.95, 1    # CRITICAL
    elif score >= 7.0:
        return 0.75, 1    # HIGH
    elif score >= 4.0:
        return 0.45, 2    # MEDIUM
    elif score > 0.0:
        return 0.15, 3    # LOW
    else:
        return 0.30, 2    # Unknown


def severity_label(score: float) -> str:
    if score >= 9.0: return "CRITICAL"
    elif score >= 7.0: return "HIGH"
    elif score >= 4.0: return "MEDIUM"
    elif score > 0.0: return "LOW"
    else: return "UNKNOWN"


def extract_cvss(cve: dict) -> tuple[float, str]:
    """Extract CVSS base score and attack vector; try v3.1 → v3.0 → v2."""
    metrics = cve.get("metrics", {})

    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            data = entries[0].get("cvssData", {})
            return float(data.get("baseScore", 0.0)), data.get("attackVector", "")

    entries = metrics.get("cvssMetricV2", [])
    if entries:
        data = entries[0].get("cvssData", {})
        av_v2 = entries[0].get("accessVector", "")
        return float(data.get("baseScore", 0.0)), av_v2

    return 0.0, ""


def infer_soc_categories(text: str) -> list[str]:
    """Map NVD CVE description text to canonical SOC categories. Cap at 3."""
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
    """CVSS-derived canonical SOC action promotion. Never suppress on threat intel."""
    if urgency >= 0.7:
        return ["escalate", "investigate"]
    elif urgency >= 0.4:
        return ["investigate", "monitor"]
    else:
        return ["monitor"]


def fetch_raw() -> dict:
    """Fetch NVD recent CVEs or load from cache."""
    if RAW_PATH.exists():
        print(f"  [NVD] Loading from cache: {RAW_PATH}")
        with open(RAW_PATH, encoding="utf-8") as f:
            return json.load(f)

    today    = date.today()
    start_dt = today - timedelta(days=WINDOW_DAYS)

    params = {
        "resultsPerPage": MAX_RESULTS,
        "pubStartDate":   f"{start_dt.isoformat()}T00:00:00.000",
        "pubEndDate":     f"{today.isoformat()}T23:59:59.999",
    }
    url = NVD_URL + "?" + urllib.parse.urlencode(params)
    print(f"  [NVD] Fetching: {url[:100]}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CrossGraphExperiments/1.0 (research)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"[NVD] HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"[NVD] Fetch failed: {e}") from e

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    print(f"  [NVD] Cached to {RAW_PATH}")
    return raw


def parse_claims(raw: dict) -> list[dict]:
    """Parse NVD CVE list into ThreatClaim dicts."""
    today = date.today()
    vuln_items = raw.get("vulnerabilities", [])
    claims: list[dict] = []
    severity_counts: dict[str, int] = defaultdict(int)

    for item in vuln_items:
        cve = item.get("cve", {})
        cve_id = cve.get("id", "UNKNOWN")

        published_str = cve.get("published", "")
        try:
            published = datetime.fromisoformat(
                published_str.replace("Z", "+00:00")
            ).date()
        except ValueError:
            published = today
        age_days = (today - published).days

        # CVSS score + attack vector
        score, attack_vector = extract_cvss(cve)

        # Filter: no PHYSICAL-only CVEs
        if attack_vector.upper() == "PHYSICAL":
            continue

        urgency, tier = cvss_to_urgency(score)
        sev = severity_label(score)
        severity_counts[sev] += 1

        # Description (English) — needed for category inference
        descs = cve.get("descriptions", [])
        desc  = next((d["value"] for d in descs if d.get("lang") == "en"), "")

        # Categories from description text (canonical SOC taxonomy)
        cats = infer_soc_categories(desc)
        assert all(c in SOC_CATEGORIES for c in cats), f"Bad category: {cats}"

        # Actions from urgency (canonical SOC actions)
        actions = infer_promoted_actions(urgency)
        assert all(a in SOC_ACTIONS for a in actions), f"Bad action: {actions}"

        # Filter: skip if no CVSS and defaulted to generic categories
        if score == 0.0 and cats == ["cloud_infrastructure", "credential_access"]:
            continue

        claims.append({
            "type":                  "cve_published",
            "cve_id":                cve_id,
            "source":                "NVD",
            "tier":                  tier,
            "direction":             1,
            "strength":              urgency,
            "confidence":            0.9,
            "extraction_confidence": 1.0,
            "urgency":               urgency,
            "age_days":              age_days,
            "decay_class":           "standard",
            "categories_affected":   cats,
            "actions_promoted":      actions,
            "description":           desc[:200],
            "vendor":                "",
            "product":               "",
            "cvss_score":            score,
            "severity":              sev,
            "attack_vector":         attack_vector,
            "published":             published_str[:10],
        })

    return claims, dict(severity_counts)


def run() -> list[dict]:
    """Main entry point."""
    print("=== NVD Fetch ===")
    raw    = fetch_raw()
    total  = raw.get("totalResults", 0)
    fetched = len(raw.get("vulnerabilities", []))

    claims, sev_counts = parse_claims(raw)

    cat_counts: dict[str, int] = defaultdict(int)
    for cl in claims:
        for c in cl["categories_affected"]:
            cat_counts[c] += 1

    print(f"  CVEs fetched:               {fetched} (total available: {total})")
    c_cnt = sev_counts.get("CRITICAL", 0)
    h_cnt = sev_counts.get("HIGH", 0)
    m_cnt = sev_counts.get("MEDIUM", 0)
    l_cnt = sev_counts.get("LOW", 0)
    u_cnt = sev_counts.get("UNKNOWN", 0)
    print(f"  Severity breakdown:         "
          f"CRITICAL={c_cnt}, HIGH={h_cnt}, MEDIUM={m_cnt}, "
          f"LOW={l_cnt}, UNKNOWN={u_cnt}")
    print(f"  Claims after filter:        {len(claims)}")
    print(f"  Category distribution:      {dict(cat_counts)}")

    with open(CLAIMS_PATH, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)
    print(f"\n  Saved {len(claims)} claims to {CLAIMS_PATH}")

    return claims


if __name__ == "__main__":
    claims = run()
    print(f"\nfetch_nvd.py done — {len(claims)} claims")
    sys.exit(0)
