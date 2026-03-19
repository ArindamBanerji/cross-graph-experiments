"""
factor_mapper.py — Map CISA KEV + NVD CVE + MITRE ATT&CK records to SOC factor space.

Three factors extracted (all in [0, 1]):
  threat_intel        — CVSS baseScore / 10
  asset_criticality   — CWE-type bucket (auth/priv → 0.8+, exec → 0.7+, etc.)
  pattern_history     — KEV recurrence proxy (single = 0.1; scale with count)

Records where a factor cannot be computed are skipped for that factor only.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# CWE-type → asset_criticality bucket
# ---------------------------------------------------------------------------

# (regex on CWE description or ID range, value)
_CWE_RULES: list[tuple[re.Pattern, float]] = [
    # Authentication / Authorization / Privilege escalation
    (re.compile(r"(auth|privilege|escalat|access.control|bypass)", re.I), 0.85),
    # Memory safety / Code execution / Buffer
    (re.compile(r"(buffer|heap|stack|overflow|use.after.free|code.exec|rce|inject)", re.I), 0.75),
    # Information disclosure / Exposure
    (re.compile(r"(disclos|exposur|information.leak|path.trav)", re.I), 0.50),
    # Cross-site / Web
    (re.compile(r"(xss|cross.site|csrf|open.redirect)", re.I), 0.35),
    # Denial of service
    (re.compile(r"(denial|dos|crash|availability)", re.I), 0.30),
]
_CWE_DEFAULT = 0.30


def _cwe_to_criticality(cwe_desc: str) -> float:
    for pattern, val in _CWE_RULES:
        if pattern.search(cwe_desc):
            return val
    return _CWE_DEFAULT


# ---------------------------------------------------------------------------
# Extract from NVD CVE item
# ---------------------------------------------------------------------------

def _extract_nvd(item: dict) -> dict:
    """
    Returns dict with keys: cvss_score, cwe_desc, cve_id.
    Values may be None if missing.
    """
    cve = item.get("cve", {})
    cve_id = cve.get("id", "")

    # CVSS score — prefer v3.1, fall back to v3.0, then v2
    cvss_score = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            try:
                cvss_score = float(entries[0]["cvssData"]["baseScore"])
            except (KeyError, TypeError, ValueError):
                pass
            if cvss_score is not None:
                break

    # CWE description
    cwe_desc = ""
    weaknesses = cve.get("weaknesses", [])
    for w in weaknesses:
        for desc in w.get("description", []):
            if desc.get("lang", "") in ("en", ""):
                cwe_desc += " " + desc.get("value", "")

    return {"cve_id": cve_id, "cvss_score": cvss_score, "cwe_desc": cwe_desc.strip()}


def _extract_kev(item: dict) -> dict:
    """
    CISA KEV record: extract CVSS-like proxy (ransomware → 0.9, other → 0.6)
    and vulnerability description for CWE proxy.
    """
    ransomware = item.get("knownRansomwareCampaignUse", "Unknown").lower()
    # KEV doesn't carry CVSS — proxy: ransomware use → high threat
    cvss_proxy = 0.85 if ransomware == "known" else 0.60
    desc = item.get("shortDescription", "") + " " + item.get("vulnerabilityName", "")
    return {
        "cve_id":     item.get("cveID", ""),
        "cvss_score": cvss_proxy,
        "cwe_desc":   desc.strip(),
    }


def _extract_technique(technique: dict) -> dict | None:
    """
    ATT&CK technique: map to threat_intel using detection coverage,
    asset_criticality via tactic, pattern_history via procedure count.
    Returns None if insufficient data.
    """
    # Threat intel proxy: if technique has CVSS equivalent, skip — use detection score
    # Use x_mitre_detection length as engagement signal, normalized
    detection_text = technique.get("x_mitre_detection", "")
    det_score = min(len(detection_text) / 2000.0, 1.0)   # longer = better-characterized

    # Asset criticality from tactic
    kill_chain = technique.get("kill_chain_phases", [])
    tactics = [p.get("phase_name", "") for p in kill_chain]
    crit = 0.30
    for t in tactics:
        if any(x in t for x in ("privilege", "credential", "execution", "impact")):
            crit = 0.85
            break
        elif any(x in t for x in ("persistence", "lateral", "exfiltration")):
            crit = 0.70
            break
        elif any(x in t for x in ("discovery", "collection")):
            crit = 0.50
            break
        elif any(x in t for x in ("initial", "defense")):
            crit = 0.40
            break

    # Pattern history: technique platforms (more platforms = more widespread)
    platforms = technique.get("x_mitre_platforms", [])
    pattern = min(len(platforms) / 8.0, 1.0)

    return {
        "threat_intel":       det_score,
        "asset_criticality":  crit,
        "pattern_history":    pattern,
        "_source":            "mitre",
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def map_to_factors(
    kev_records:   list[dict],
    nvd_records:   list[dict],
    mitre_records: list[dict],
) -> dict[str, np.ndarray]:
    """
    Map all records to 3-factor arrays in [0, 1].

    Returns:
      {
        'threat_intel':      np.ndarray,
        'asset_criticality': np.ndarray,
        'pattern_history':   np.ndarray,
      }

    All three arrays have the same length (one row per successfully mapped record).
    Records where any factor is missing are skipped.
    """
    rows: list[dict] = []
    n_skip = 0

    # ---- NVD CVEs ----
    for item in nvd_records:
        ext = _extract_nvd(item)
        if ext["cvss_score"] is None:
            n_skip += 1
            continue
        ti   = float(ext["cvss_score"]) / 10.0
        ac   = _cwe_to_criticality(ext["cwe_desc"])
        ph   = 0.10   # NVD single record — no recurrence info
        rows.append({"threat_intel": ti, "asset_criticality": ac,
                     "pattern_history": ph, "_source": "nvd"})

    # ---- CISA KEV (richer — ransomware use ↑ pattern_history) ----
    # Count CVE-ID recurrence within KEV list (multi-advisory same CVE)
    kev_id_counts = Counter(r.get("cveID", "") for r in kev_records)
    for item in kev_records:
        ext = _extract_kev(item)
        ti  = float(ext["cvss_score"])
        ac  = _cwe_to_criticality(ext["cwe_desc"])
        ph  = min(kev_id_counts[ext["cve_id"]] / 10.0, 1.0)
        ph  = max(ph, 0.10)   # floor at 0.10 for single appearance
        rows.append({"threat_intel": ti, "asset_criticality": ac,
                     "pattern_history": ph, "_source": "kev"})

    # ---- MITRE ATT&CK ----
    for tech in mitre_records:
        mapped = _extract_technique(tech)
        if mapped is None:
            n_skip += 1
            continue
        rows.append(mapped)

    if n_skip > 0:
        print(f"  [MAPPER] Skipped {n_skip} records (missing CVSS or insufficient data)")

    if not rows:
        print("  [MAPPER] WARNING: no records mapped — returning empty arrays")
        empty = np.array([], dtype=float)
        return {"threat_intel": empty, "asset_criticality": empty,
                "pattern_history": empty}

    ti_arr = np.clip([r["threat_intel"]      for r in rows], 0.0, 1.0)
    ac_arr = np.clip([r["asset_criticality"] for r in rows], 0.0, 1.0)
    ph_arr = np.clip([r["pattern_history"]   for r in rows], 0.0, 1.0)

    sources = Counter(r["_source"] for r in rows)
    print(f"  [MAPPER] Mapped {len(rows)} records: "
          f"NVD={sources['nvd']}, KEV={sources['kev']}, ATT&CK={sources['mitre']}")

    return {
        "threat_intel":      np.array(ti_arr, dtype=float),
        "asset_criticality": np.array(ac_arr, dtype=float),
        "pattern_history":   np.array(ph_arr, dtype=float),
    }
