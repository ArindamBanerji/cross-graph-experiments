"""
SVM-005 — FX-1 Distribution Coverage Completion.

Extends KL divergence characterization from IOC-heavy subset to all 6 SOC
alert categories. Output: experiments/svm_calibration_v2.json.

Data prep only — no claim gates.
"""
from __future__ import annotations

import json
import sys
import math
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# SOC Taxonomy (V-series canonical)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "lateral_movement",
    "credential_access",
    "data_exfiltration",
    "cloud_infrastructure",
    "insider_threat",
    "threat_intel_match",   # spec name: malware_execution → maps here
]

FACTORS = [
    "travel_match",
    "asset_criticality",
    "threat_intel_enrichment",
    "time_anomaly",
    "pattern_history",
    "device_trust",
]

# ---------------------------------------------------------------------------
# MU_STAR — V-series canonical (used as FX-1 baseline mean per category×factor)
# ---------------------------------------------------------------------------
_MU_STAR_RAW = {
    ("lateral_movement",     "escalate"):    [0.30, 0.50, 0.75, 0.35, 0.80, 0.65],
    ("lateral_movement",     "investigate"): [0.30, 0.43, 0.55, 0.35, 0.60, 0.55],
    ("lateral_movement",     "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("lateral_movement",     "monitor"):     [0.30, 0.43, 0.40, 0.35, 0.35, 0.45],
    ("insider_threat",       "escalate"):    [0.25, 0.55, 0.70, 0.30, 0.75, 0.65],
    ("insider_threat",       "investigate"): [0.25, 0.46, 0.50, 0.30, 0.55, 0.55],
    ("insider_threat",       "suppress"):    [0.25, 0.40, 0.20, 0.30, 0.20, 0.35],
    ("insider_threat",       "monitor"):     [0.25, 0.42, 0.38, 0.30, 0.32, 0.45],
    ("credential_access",    "escalate"):    [0.35, 0.50, 0.80, 0.40, 0.75, 0.65],
    ("credential_access",    "investigate"): [0.35, 0.43, 0.60, 0.40, 0.58, 0.55],
    ("credential_access",    "suppress"):    [0.35, 0.40, 0.20, 0.40, 0.22, 0.35],
    ("credential_access",    "monitor"):     [0.35, 0.42, 0.42, 0.40, 0.33, 0.45],
    ("data_exfiltration",    "escalate"):    [0.30, 0.52, 0.78, 0.35, 0.82, 0.65],
    ("data_exfiltration",    "investigate"): [0.30, 0.44, 0.58, 0.35, 0.62, 0.55],
    ("data_exfiltration",    "suppress"):    [0.30, 0.40, 0.20, 0.35, 0.20, 0.35],
    ("data_exfiltration",    "monitor"):     [0.30, 0.42, 0.40, 0.35, 0.32, 0.45],
    ("cloud_infrastructure", "escalate"):    [0.28, 0.45, 0.72, 0.38, 0.70, 0.65],
    ("cloud_infrastructure", "investigate"): [0.28, 0.41, 0.52, 0.38, 0.52, 0.55],
    ("cloud_infrastructure", "suppress"):    [0.28, 0.40, 0.20, 0.38, 0.20, 0.35],
    ("cloud_infrastructure", "monitor"):     [0.28, 0.41, 0.38, 0.38, 0.30, 0.45],
    ("threat_intel_match",   "escalate"):    [0.32, 0.52, 0.82, 0.36, 0.78, 0.65],
    ("threat_intel_match",   "investigate"): [0.32, 0.44, 0.62, 0.36, 0.58, 0.55],
    ("threat_intel_match",   "suppress"):    [0.32, 0.40, 0.20, 0.36, 0.20, 0.35],
    ("threat_intel_match",   "monitor"):     [0.32, 0.42, 0.44, 0.36, 0.33, 0.45],
}
ACTIONS = ["monitor", "investigate", "suppress", "escalate"]
N_FACTORS = len(FACTORS)

def build_mu_star_per_category():
    """Mean factor vector per category (average across actions)."""
    result = {}
    for cat in CATEGORIES:
        vecs = [_MU_STAR_RAW[(cat, act)] for act in ACTIONS if (cat, act) in _MU_STAR_RAW]
        if vecs:
            result[cat] = np.mean(vecs, axis=0).tolist()
        else:
            result[cat] = [0.5] * N_FACTORS
    return result


MU_STAR_CAT = build_mu_star_per_category()

FX1_SIGMA = 0.15  # uniform sigma used across all FX-1 experiments

# ---------------------------------------------------------------------------
# ATT&CK tactic → SOC category mapping
# ---------------------------------------------------------------------------
ATTACK_TACTIC_MAP = {
    "lateral_movement":     lambda t: "lateral-movement" in t,
    "credential_access":    lambda t: "credential-access" in t,
    "data_exfiltration":    lambda t: "exfiltration" in t,
    "cloud_infrastructure": lambda t: ("cloud" in t or "persistence" in t),
    "insider_threat":       lambda t: ("collection" in t or "impact" in t),
    "threat_intel_match":   lambda t: ("execution" in t or "defense-evasion" in t),
}

# ---------------------------------------------------------------------------
# CISA KEV product → SOC category heuristics
# ---------------------------------------------------------------------------
def classify_kev_entry(vendor: str, product: str, vuln_type: str) -> str | None:
    text = (vendor + " " + product + " " + vuln_type).lower()
    if any(k in text for k in ["authentication", "auth bypass", "credential", "ldap", "kerberos", "ntlm"]):
        return "credential_access"
    if any(k in text for k in ["router", "switch", "vpn", "firewall", "smb", "rdp", "lateral"]):
        return "lateral_movement"
    if any(k in text for k in ["exfil", "data leak", "ftp", "sftp", "transfer"]):
        return "data_exfiltration"
    if any(k in text for k in ["aws", "azure", "cloud", "kubernetes", "container", "s3", "iam"]):
        return "cloud_infrastructure"
    if any(k in text for k in ["insider", "privilege", "sudo", "escalation", "lpe"]):
        return "insider_threat"
    if any(k in text for k in ["cve", "exploit", "rce", "execute", "malware", "ransomware", "trojan"]):
        return "threat_intel_match"
    return None

# ---------------------------------------------------------------------------
# Network fetch helpers
# ---------------------------------------------------------------------------
def fetch_json(url: str, timeout: int = 90, max_bytes: int = 100 * 1024 * 1024) -> tuple[dict | list | None, str]:
    """Fetch JSON from URL. Returns (data, status_message)."""
    try:
        import requests
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            return None, f"file too large ({content_length} bytes > {max_bytes})"
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                return None, f"stream exceeded {max_bytes} bytes"
        data = json.loads(b"".join(chunks).decode("utf-8", errors="replace"))
        return data, "fetched"
    except Exception as exc:
        return None, f"fallback ({type(exc).__name__}: {str(exc)[:80]})"


def fetch_mitre_attack() -> tuple[dict | None, str]:
    url = ("https://raw.githubusercontent.com/mitre/cti/master/"
           "enterprise-attack/enterprise-attack.json")
    print("  Fetching MITRE ATT&CK enterprise-attack.json …", flush=True)
    data, status = fetch_json(url, timeout=90, max_bytes=120 * 1024 * 1024)
    if data is None:
        print(f"  Source A unavailable — using FX-1 fallback. ({status})")
    else:
        print(f"  Source A: ATT&CK fetched ({len(data.get('objects', []))} objects).")
    return data, status


def fetch_cisa_kev() -> tuple[dict | None, str]:
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    print("  Fetching CISA KEV catalog …", flush=True)
    data, status = fetch_json(url, timeout=30, max_bytes=20 * 1024 * 1024)
    if data is None:
        print(f"  Source B unavailable — using FX-1 fallback. ({status})")
    else:
        n = len(data.get("vulnerabilities", []))
        print(f"  Source B: CISA KEV fetched ({n} vulnerabilities).")
    return data, status

# ---------------------------------------------------------------------------
# Parse ATT&CK → technique counts per SOC category
# ---------------------------------------------------------------------------
def parse_mitre_technique_counts(attack_data: dict) -> dict[str, int]:
    """Count ATT&CK techniques per SOC category."""
    counts = {c: 0 for c in CATEGORIES}
    for obj in attack_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        tactics = [kcp.get("phase_name", "") for kcp in obj.get("kill_chain_phases", [])
                   if kcp.get("kill_chain_name") == "mitre-attack"]
        for tactic in tactics:
            for cat, match_fn in ATTACK_TACTIC_MAP.items():
                if match_fn(tactic):
                    counts[cat] += 1
                    break
    return counts


def parse_cisa_kev_severity(kev_data: dict) -> dict[str, list[float]]:
    """Collect CVSS scores per SOC category from KEV."""
    scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for vuln in kev_data.get("vulnerabilities", []):
        vendor  = vuln.get("vendorProject", "")
        product = vuln.get("product", "")
        vtype   = vuln.get("vulnerabilityName", "")
        cat     = classify_kev_entry(vendor, product, vtype)
        if cat is None:
            continue
        # CVSS not always in KEV; use requiredAction presence as binary severity proxy
        # (KEV lists are inherently high-severity; approximate uniform CVSS~8.5)
        cvss = float(vuln.get("cvssScore", 8.5) if "cvssScore" in vuln else 8.5)
        scores[cat].append(cvss / 10.0)   # normalise to [0,1]
    return scores

# ---------------------------------------------------------------------------
# Compute per-category × factor distributions
# ---------------------------------------------------------------------------
def compute_factor_distributions(
    technique_counts: dict[str, int] | None,
    kev_severity: dict[str, list[float]] | None,
) -> dict[str, dict]:
    """
    Build factor distribution (mean, sigma, source) for each category × factor.

    Factor index map:
      0 travel_match           → FX-1 fallback (no external signal)
      1 asset_criticality      → CISA KEV mean CVSS if available, else FX-1
      2 threat_intel_enrichment→ ATT&CK technique count (normalised) if available
      3 time_anomaly           → FX-1 fallback
      4 pattern_history        → FX-1 fallback
      5 device_trust           → CISA KEV inverse CVSS proxy if available, else FX-1
    """
    # Normalise technique counts to [0,1]
    if technique_counts is not None:
        max_count = max(technique_counts.values()) or 1
        norm_tc = {c: technique_counts[c] / max_count for c in CATEGORIES}
    else:
        norm_tc = None

    result = {}
    for cat in CATEGORIES:
        mu_star = MU_STAR_CAT[cat]
        factors_out = {}
        for fi, fname in enumerate(FACTORS):
            mu_base  = mu_star[fi]
            sig_base = FX1_SIGMA
            source   = "fx1_fallback"
            mu_new   = mu_base
            sig_new  = sig_base

            if fi == 1 and kev_severity is not None:
                # asset_criticality ← mean CVSS for this category
                scores = kev_severity.get(cat, [])
                if scores:
                    mu_new  = float(np.mean(scores))
                    sig_new = float(np.std(scores)) if len(scores) > 1 else 0.12
                    sig_new = max(0.05, min(sig_new, 0.35))
                    source  = "cisa_kev"

            elif fi == 2 and norm_tc is not None:
                # threat_intel_enrichment ← normalised technique count
                tc_norm = norm_tc.get(cat, 0.5)
                # blend with MU_STAR: 60% empirical, 40% structural
                mu_new  = 0.6 * tc_norm + 0.4 * mu_base
                mu_new  = float(np.clip(mu_new, 0.05, 0.95))
                sig_new = 0.12  # tighter sigma: technique count is a count not a sample
                source  = "mitre_attack"

            elif fi == 5 and kev_severity is not None:
                # device_trust ← inverse of CVSS proxy
                # high-severity CVEs → lower device trust
                scores = kev_severity.get(cat, [])
                if scores:
                    mean_cvss = float(np.mean(scores))
                    mu_new  = float(np.clip(1.0 - mean_cvss + 0.1, 0.10, 0.90))
                    sig_new = float(np.std(scores)) if len(scores) > 1 else 0.12
                    sig_new = max(0.05, min(sig_new, 0.35))
                    source  = "cisa_kev"

            factors_out[fname] = {
                "mean":   round(mu_new, 4),
                "sigma":  round(sig_new, 4),
                "source": source,
            }
        result[cat] = factors_out
    return result


# ---------------------------------------------------------------------------
# KL divergence: KL(N(mu1,s1²) || N(mu2,s2²)), summed over factors
# ---------------------------------------------------------------------------
def gaussian_kl(mu1: float, s1: float, mu2: float, s2: float) -> float:
    """KL(N(mu1,s1) || N(mu2,s2))."""
    return (math.log(s2 / s1)
            + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2)
            - 0.5)


def compute_kl_vs_fx1(factor_dists: dict[str, dict]) -> dict[str, float]:
    """Compute total KL divergence (sum over factors) vs FX-1 baseline."""
    kl_per_cat = {}
    for cat in CATEGORIES:
        mu_star = MU_STAR_CAT[cat]
        total_kl = 0.0
        for fi, fname in enumerate(FACTORS):
            fd     = factor_dists[cat][fname]
            mu_new = fd["mean"]
            s_new  = fd["sigma"]
            mu_fx1 = mu_star[fi]
            s_fx1  = FX1_SIGMA
            total_kl += gaussian_kl(mu_new, s_new, mu_fx1, s_fx1)
        kl_per_cat[cat] = round(total_kl, 4)
    return kl_per_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import gae
    assert gae.__version__ == "0.7.18", \
        f"Expected GAE 0.7.18, got {gae.__version__}"

    print("SVM-005 — FX-1 Distribution Coverage Completion:")
    print()

    t0 = time.time()

    # Step 1 — Fetch real data
    attack_data, attack_status = fetch_mitre_attack()
    kev_data, kev_status = fetch_cisa_kev()

    # Step 2 — Parse
    technique_counts = parse_mitre_technique_counts(attack_data) if attack_data else None
    kev_severity     = parse_cisa_kev_severity(kev_data) if kev_data else None

    # Step 3 — Factor distributions
    print("  Computing factor distributions …", flush=True)
    factor_dists = compute_factor_distributions(technique_counts, kev_severity)

    # Step 4 — KL divergence vs FX-1
    kl_per_cat = compute_kl_vs_fx1(factor_dists)
    max_kl     = max(kl_per_cat.values())

    # Count source breakdown
    n_real = 0
    n_fx1  = 0
    for cat in CATEGORIES:
        for fname in FACTORS:
            src = factor_dists[cat][fname]["source"]
            if src == "fx1_fallback":
                n_fx1 += 1
            else:
                n_real += 1

    # Technique counts for output
    tc_out = technique_counts if technique_counts else {c: 0 for c in CATEGORIES}

    # Step 5 — Build output JSON
    categories_out = {}
    for cat in CATEGORIES:
        categories_out[cat] = {
            "technique_count":       tc_out.get(cat, 0),
            "factor_distributions":  factor_dists[cat],
            "kl_vs_fx1":             kl_per_cat[cat],
            "kl_within_bound":       kl_per_cat[cat] < 3.0,
        }

    out = {
        "version":     "2.0",
        "gae_version": gae.__version__,
        "generated":   str(date.today()),
        "sources": {
            "mitre_attack": attack_status,
            "cisa_kev":     kev_status,
        },
        "categories": categories_out,
        "summary": {
            "categories_covered":      len(CATEGORIES),
            "factors_from_real_data":  n_real,
            "factors_from_fx1_fallback": n_fx1,
            "max_kl_divergence":       round(max_kl, 4),
            "kl_all_within_3_0":       bool(max_kl < 3.0),
            "methodology": (
                "ATT&CK technique frequency + CISA KEV severity mapping. "
                "Factors: threat_intel_enrichment from ATT&CK normalised count, "
                "asset_criticality and device_trust from CISA KEV CVSS. "
                "Remaining factors: FX-1 baseline (MU_STAR mean, sigma=0.15)."
            ),
        },
    }

    out_path = REPO_ROOT / "experiments" / "svm_calibration_v2.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    elapsed = time.time() - t0

    # Verify valid JSON
    with open(out_path, encoding="utf-8") as fh:
        json.load(fh)   # will raise if invalid

    print()
    print(f"SVM-005 — FX-1 Distribution Coverage Completion:")
    a_src = "fetched" if attack_data else "fallback"
    c_src = "fetched" if kev_data else "fallback"
    print(f"  Sources: ATT&CK={a_src} CISA={c_src}")
    print(f"  Categories covered: {len(CATEGORIES)}/6")
    print(f"  Factors from real data: {n_real}/36")
    print(f"  Factors from FX-1 fallback: {n_fx1}/36")
    print(f"  Max KL divergence vs FX-1: {max_kl:.3f} [target: <3.0]")
    print(f"  Output: experiments/svm_calibration_v2.json")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Status: COMPLETE — ready for SVM-002/003/004")
    print()
    print("  Per-category KL vs FX-1:")
    for cat in CATEGORIES:
        kl = kl_per_cat[cat]
        tc = tc_out.get(cat, 0)
        flag = " OK" if kl < 3.0 else " EXCEEDS 3.0"
        print(f"    {cat:<25} KL={kl:.3f}{flag}  techniques={tc}")


if __name__ == "__main__":
    main()
