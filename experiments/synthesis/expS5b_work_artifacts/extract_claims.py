"""
extract_claims.py — LLM + template claim extraction for EXP-S5b.
experiments/synthesis/expS5b_work_artifacts/extract_claims.py

Two extraction paths:
  1. LLM (Claude claude-sonnet-4-6): structured JSON from /v1/messages
  2. Template fallback: keyword-based rule extraction (used when no API key)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
load_dotenv()

# Canonical SOC taxonomy
SOC_CATEGORIES: List[str] = [
    "travel_anomaly", "credential_access", "threat_intel_match",
    "insider_behavioral", "cloud_infrastructure",
]
SOC_ACTIONS: List[str] = ["escalate", "investigate", "suppress", "monitor"]
CLAIM_TYPES: List[str] = [
    "active_campaign", "ciso_risk_directive", "cve_actively_exploited",
    "vulnerability_patched", "known_change", "known_fp_pattern",
]

LLM_EXTRACTION_PROMPT = """\
You are a security operations claim extractor. Given a work artifact, extract
structured claims that should influence SOC alert triage decisions.

Return ONLY a JSON array. No preamble, no markdown fences. If no actionable
claims exist, return [].

Each claim must have exactly these fields:
{{
  "type": one of {claim_types},
  "categories_affected": subset of {soc_categories},
  "direction": +1 (promote escalation/investigation) or -1 (promote suppression/monitoring),
  "actions_promoted": subset of {soc_actions},
  "urgency": float 0.0 to 1.0,
  "confidence": float 0.0 to 1.0 (your extraction confidence),
  "summary": one sentence describing this claim
}}

Rules:
- direction=+1 means this claim suggests MORE scrutiny (escalate or investigate)
- direction=-1 means this claim suggests LESS scrutiny (suppress or monitor)
- actions_promoted must be consistent with direction:
  direction=+1 -> ["escalate","investigate"], direction=-1 -> ["suppress","monitor"]
- urgency reflects how time-sensitive this claim is
- confidence reflects how clearly the artifact supports this claim
- Never invent categories not listed above

ARTIFACT TYPE: {artifact_type}
SOURCE: {artifact_source} (authority: {artifact_authority})

ARTIFACT TEXT:
{artifact_text}"""


def infer_soc_categories(text: str) -> List[str]:
    """Map text to canonical SOC categories. Cap at 3. (mirrors fetch_kev.py)"""
    text = text.lower()
    matched: List[str] = []
    if any(k in text for k in [
        "credential", "authentication", "login", "password",
        "oauth", "saml", "ldap", "active directory", "kerberos",
        "brute force", "stuffing", "auth"]):
        matched.append("credential_access")
    if any(k in text for k in [
        "phishing", "campaign", "threat actor", "apt", "ioc",
        "indicator", "malware", "ransomware", "backdoor", "c2",
        "command and control", "fin12", "spearphish"]):
        matched.append("threat_intel_match")
    if any(k in text for k in [
        "vpn", "travel", "geolocation", "impossible travel",
        "anomalous login", "off-hours", "travel_anomaly", "geolocat"]):
        matched.append("travel_anomaly")
    if any(k in text for k in [
        "exfiltration", "data theft", "insider", "lateral movement",
        "privilege escalation", "data leak", "unauthorized access",
        "policy violation", "departing", "resignation", "employee"]):
        matched.append("insider_behavioral")
    if any(k in text for k in [
        "remote code", "rce", "code execution", "command injection",
        "buffer overflow", "deserialization", "cloud", "infrastructure",
        "container", "kubernetes", "aws", "azure", "api", "network",
        "cve", "patch", "vulnerability", "pan-os", "globalprotect"]):
        matched.append("cloud_infrastructure")
    if not matched:
        matched = ["cloud_infrastructure", "credential_access"]
    return matched[:3]


def call_anthropic_api(prompt: str) -> str:
    """POST to /v1/messages and return the assistant's text content."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    payload = json.dumps({
        "model":       "claude-sonnet-4-6",
        "max_tokens":  2048,
        "temperature": 0.0,
        "messages":    [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


def validate_and_normalize_claim(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate one LLM-extracted claim dict. Returns None if invalid type.
    Filters categories and actions to canonical sets; fills safe defaults.
    """
    try:
        claim_type = raw.get("type", "")
        if claim_type not in CLAIM_TYPES:
            return None

        cats = [c for c in raw.get("categories_affected", []) if c in SOC_CATEGORIES]
        if not cats:
            cats = ["cloud_infrastructure"]

        direction = int(raw.get("direction", 1))
        if direction not in (1, -1):
            direction = 1

        acts = [a for a in raw.get("actions_promoted", []) if a in SOC_ACTIONS]
        if not acts:
            acts = ["escalate", "investigate"] if direction == 1 else ["suppress", "monitor"]

        urgency    = float(raw.get("urgency",    0.5))
        confidence = float(raw.get("confidence", 0.7))
        urgency    = max(0.0, min(1.0, urgency))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "type":                  claim_type,
            "categories_affected":   cats,
            "direction":             direction,
            "actions_promoted":      acts,
            "urgency":               urgency,
            "confidence":            confidence,
            "extraction_confidence": confidence,
            "summary":               str(raw.get("summary", "")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def parse_llm_json(raw: str) -> list:
    """Parse LLM response that may contain markdown fences or preamble.

    Tries in order:
    1. Direct parse (clean JSON)
    2. Strip ```json ... ``` fences and retry
    3. Strip ``` ... ``` fences (no language tag) and retry
    4. Find first '[' and last ']' and extract substring
    5. Raise ValueError if all attempts fail
    """
    text = raw.strip()
    # Normalize: LLM sometimes emits +1 for direction values; +N is invalid JSON
    text = re.sub(r':\s*\+(\d)', r': \1', text)

    # Attempt 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip ```json fences
    if "```json" in text:
        try:
            inner = text.split("```json")[1].split("```")[0].strip()
            return json.loads(inner)
        except (json.JSONDecodeError, IndexError):
            pass

    # Attempt 3: strip ``` fences (no language tag)
    if "```" in text:
        try:
            inner = text.split("```")[1].strip()
            return json.loads(inner)
        except (json.JSONDecodeError, IndexError):
            pass

    # Attempt 4: extract first [...] substring
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM response as JSON: {text[:200]}")


def extract_claims_llm(artifact: Dict[str, Any]) -> Tuple[List[Dict], str]:
    """
    Extract claims using Claude claude-sonnet-4-6 via /v1/messages.
    Returns (claims, method_used) where method_used is "llm" or "template_fallback".
    Falls back to template if no ANTHROPIC_API_KEY or API error.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"  [!] No ANTHROPIC_API_KEY — template fallback for {artifact['id']}")
        return extract_claims_template(artifact), "template_fallback"

    prompt = LLM_EXTRACTION_PROMPT.format(
        claim_types     = json.dumps(CLAIM_TYPES),
        soc_categories  = json.dumps(SOC_CATEGORIES),
        soc_actions     = json.dumps(SOC_ACTIONS),
        artifact_type   = artifact["type"],
        artifact_source = artifact["source"],
        artifact_authority = artifact["authority"],
        artifact_text   = artifact["text"],
    )

    try:
        raw_text = call_anthropic_api(prompt)
        raw = parse_llm_json(raw_text)
        claims = [validate_and_normalize_claim(c) for c in raw]
        return [c for c in claims if c is not None], "llm"
    except ValueError as e:
        print(f"  [!] LLM parse failed for {artifact['id']}: {e} — template fallback")
        return extract_claims_template(artifact), "template_fallback"
    except Exception as e:
        print(f"  [!] LLM error for {artifact['id']}: {type(e).__name__}: {e} — template fallback")
        return extract_claims_template(artifact), "template_fallback"


def extract_claims_template(artifact: Dict[str, Any]) -> List[Dict]:
    """
    Rule-based template extraction. Keyword matching on artifact text.
    Used as fallback when no API key or LLM failure.
    """
    text = artifact["text"]
    text_lower = text.lower()
    claims: List[Dict] = []
    authority = artifact.get("authority", "medium")
    conf = 0.9 if authority == "high" else 0.6 if authority == "medium" else 0.4

    # --- Escalation signals ---
    if any(k in text_lower for k in [
        "escalate", "high priority", "immediate", "critical",
        "exploitation", "campaign", "ransomware", "threat actor",
        "fin12", "ciso", "effective immediately", "directive",
    ]):
        cats = infer_soc_categories(text)
        claim_type = "ciso_risk_directive" if artifact["source"] == "CISO" else "active_campaign"
        claims.append({
            "type":                  claim_type,
            "categories_affected":   cats,
            "direction":             +1,
            "actions_promoted":      ["escalate", "investigate"],
            "urgency":               0.8,
            "confidence":            conf,
            "extraction_confidence": conf,
            "summary":               f"Escalation signal detected in {artifact['type']} from {artifact['source']}",
        })

    # --- CVE / vulnerability signals ---
    if any(k in text_lower for k in ["cve-", "cvss", "authentication bypass", "exploitation observed"]):
        cats = infer_soc_categories(text)
        cve_type = "cve_actively_exploited" if "exploited" in text_lower else "vulnerability_patched"
        claims.append({
            "type":                  cve_type,
            "categories_affected":   cats,
            "direction":             +1 if "exploited" in text_lower else -1,
            "actions_promoted":      ["escalate", "investigate"] if "exploited" in text_lower else ["suppress", "monitor"],
            "urgency":               0.9 if "cvss 9" in text_lower else 0.6,
            "confidence":            conf,
            "extraction_confidence": conf,
            "summary":               f"CVE signal in {artifact['type']}",
        })

    # --- Suppression signals ---
    if any(k in text_lower for k in [
        "suppress", "false positive", " fp ", "known pattern",
        "maintenance", "planned", "onboarding", "accept risk",
        "no patient data", "resolved", "closed", "return to normal",
        "expected", "new hire", "not escalate",
    ]):
        cats = infer_soc_categories(text)
        supp_type = "known_fp_pattern" if "false positive" in text_lower or " fp " in text_lower else "known_change"
        claims.append({
            "type":                  supp_type,
            "categories_affected":   cats,
            "direction":             -1,
            "actions_promoted":      ["suppress", "monitor"],
            "urgency":               0.2,
            "confidence":            conf,
            "extraction_confidence": conf,
            "summary":               f"Suppression signal detected in {artifact['type']} from {artifact['source']}",
        })

    return claims


def compute_extraction_f1(
    extracted: List[Dict],
    expected:  List[Dict],
) -> float:
    """
    F1 score against expected claims.
    Match criterion: (type match) AND (categories overlap >= 1) AND (direction match).
    Precision = matched / extracted. Recall = matched / expected.
    """
    if not expected:
        return 1.0 if not extracted else 0.0
    if not extracted:
        return 0.0

    matched_exp: set = set()
    matched_ext: set = set()

    for i, ext in enumerate(extracted):
        for j, exp in enumerate(expected):
            if j in matched_exp:
                continue
            type_match  = ext.get("type") == exp.get("type")
            ext_cats    = set(ext.get("categories_affected", []))
            exp_cats    = set(exp.get("categories", []))
            cat_overlap = bool(ext_cats & exp_cats)
            dir_match   = int(ext.get("direction", 1)) == int(exp.get("direction", 1))
            if type_match and cat_overlap and dir_match:
                matched_exp.add(j)
                matched_ext.add(i)
                break

    precision = len(matched_ext) / len(extracted)
    recall    = len(matched_exp) / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    from sample_artifacts import SAMPLE_ARTIFACTS
    art06 = next(a for a in SAMPLE_ARTIFACTS if a["id"] == "art_06")
    claims, method = extract_claims_llm(art06)
    print(f"art_06 smoke test: method={method}, claims={len(claims)}")
    for c in claims:
        print(f"  [{c['type']}] dir={c['direction']} {c['summary'][:60]}")
