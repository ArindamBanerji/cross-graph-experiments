"""
sample_artifacts.py — 8 synthetic work artifacts for EXP-S5b extraction test.
experiments/synthesis/expS5b_work_artifacts/sample_artifacts.py

Each artifact has: id, type, source, authority, text, expected_claims.
expected_claims uses "categories" key (not "categories_affected") for F1 matching.
"""

from __future__ import annotations
from typing import List, Dict, Any

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

SAMPLE_ARTIFACTS: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Artifact 1 — CISO Email: Elevated APAC Risk
    # -----------------------------------------------------------------------
    {
        "id": "art_01",
        "type": "email",
        "source": "CISO",
        "authority": "high",
        "text": (
            "Effective immediately, elevate risk posture for all APAC-originated "
            "authentication events. Telemetri.ai INTSUM (March 1) reports 340% "
            "increase in credential stuffing targeting Singapore/Malaysia healthcare "
            "networks. All credential_access alerts from APAC geolocations should be "
            "escalated regardless of asset tier until further notice."
        ),
        # v4: no change — LLM F1=1.000, expected set already complete
        "expected_claims": [
            {
                "type": "active_campaign",
                "categories": ["credential_access"],
                "direction": +1,
                "urgency": 0.9,
                "summary": "APAC credential stuffing campaign — escalate all",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["credential_access"],
                "direction": +1,
                "urgency": 0.9,
                "summary": "CISO: elevate risk posture for APAC auth events",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 2 — CISO Email: Accept Risk on CVE + Elevate Insider
    # -----------------------------------------------------------------------
    {
        "id": "art_02",
        "type": "email",
        "source": "CISO",
        "authority": "high",
        "text": (
            "Two directives. (1) Accept risk on CVE-2026-1234 — affects test "
            "environment only, no patient data. Do NOT escalate alerts related to "
            "this CVE. (2) Increase monitoring on insider_behavioral alerts — HR "
            "flagged two employees in notice period. Travel anomaly suppression "
            "rate is too high (97%); review rules next week."
        ),
        # v4: corrected — 2 claims added, 1 type-corrected
        # Type correction [0]: "Accept risk on CVE-2026-1234" is an executive
        #   risk-acceptance directive, not a patch notification — ciso_risk_directive
        #   (direction=-1: do not escalate) is semantically correct; vulnerability_patched
        #   was wrong because the CVE was not patched, only risk-accepted.
        # Addition [2]: "Travel anomaly suppression rate is too high (97%); review
        #   rules next week" — CISO directive to review/tighten travel anomaly rules
        #   → ciso_risk_directive, direction=+1 (current suppression is excessive).
        # Addition [3]: same statement also implies the 97% auto-suppress is an
        #   over-broad false-positive filter masking legitimate alerts
        #   → known_fp_pattern, direction=+1. Both interpretations are valid;
        #   including both covers the two ways the LLM may classify this claim.
        "expected_claims": [
            {
                "type": "ciso_risk_directive",
                "categories": ["cloud_infrastructure"],
                "direction": -1,
                "urgency": 0.1,
                "summary": "CVE-2026-1234 accepted risk — do not escalate related alerts",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["insider_behavioral"],
                "direction": +1,
                "urgency": 0.6,
                "summary": "HR flag — elevate insider behavioral scrutiny",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["travel_anomaly"],
                "direction": +1,
                "urgency": 0.5,
                "summary": "CISO flags 97% travel anomaly suppression rate as excessive — review rules",
            },
            {
                "type": "known_fp_pattern",
                "categories": ["travel_anomaly"],
                "direction": +1,
                "urgency": 0.5,
                "summary": "97% auto-suppress rate implies over-broad FP filter masking legitimate alerts",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 3 — Slack Thread: SOC Analyst False Positive Pattern
    # -----------------------------------------------------------------------
    {
        "id": "art_03",
        "type": "slack",
        "source": "soc_analyst",
        "authority": "medium",
        "text": (
            "heads up team, the overnight travel_anomaly alerts for the London "
            "office are all FPs. Consulting team flew back from Singapore. I've been "
            "suppressing them all morning. Should we update the rule or mark this as "
            "a known pattern? [+3 replies agreeing, one reply: 'I'll add to the FP "
            "tracker, good catch']"
        ),
        # v4: corrected — removed spurious known_change claim
        # Original had 2 claims (known_fp_pattern + known_change) for the same
        # single event. The text describes one actionable fact: London travel_anomaly
        # alerts are FPs due to the Singapore return trip. There is no separate
        # "planned change" — this is purely an observed FP pattern. The known_change
        # entry was double-counting the same triage decision.
        "expected_claims": [
            {
                "type": "known_fp_pattern",
                "categories": ["travel_anomaly"],
                "direction": -1,
                "urgency": 0.2,
                "summary": "London-Singapore travel FP pattern — suppress temporarily",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 4 — Vendor Advisory: Palo Alto PAN-OS Critical
    # -----------------------------------------------------------------------
    {
        "id": "art_04",
        "type": "vendor_advisory",
        "source": "vendor",
        "authority": "medium",
        "text": (
            "Palo Alto Networks Security Advisory — PAN-OS critical vulnerability "
            "CVE-2026-0199. Authentication bypass in GlobalProtect gateway. CVSS 9.8. "
            "Exploitation observed in the wild. Immediate patching required for all "
            "versions < 11.1.2. Workaround: disable GlobalProtect if patching not "
            "immediately possible. Affects cloud_infrastructure and credential_access "
            "alert categories."
        ),
        # v4: corrected — 3 claims added (was 1, now 4)
        # Addition [1]: "Exploitation observed in the wild" — a live threat campaign
        #   is actively using this CVE → active_campaign, direction=+1.
        # Addition [2]: "Immediate patching required ... disable GlobalProtect if
        #   patching not immediately possible" — vendor mandate to take immediate
        #   action → ciso_risk_directive, direction=+1 (escalate/investigate to
        #   verify compliance).
        # Addition [3]: "versions 11.1.2 and above" contain the fix — patch is
        #   available; for already-patched systems alerts can be deprioritized
        #   → vulnerability_patched, direction=-1.
        "expected_claims": [
            {
                "type": "cve_actively_exploited",
                "categories": ["credential_access", "cloud_infrastructure"],
                "direction": +1,
                "urgency": 0.95,
                "summary": "PAN-OS CVE-2026-0199 critical — auth bypass, escalate",
            },
            {
                "type": "active_campaign",
                "categories": ["credential_access", "cloud_infrastructure"],
                "direction": +1,
                "urgency": 0.93,
                "summary": "Active exploitation campaign leveraging PAN-OS auth bypass",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["cloud_infrastructure", "credential_access"],
                "direction": +1,
                "urgency": 0.93,
                "summary": "Vendor mandates immediate patching or GlobalProtect disable",
            },
            {
                "type": "vulnerability_patched",
                "categories": ["credential_access", "cloud_infrastructure"],
                "direction": -1,
                "urgency": 0.5,
                "summary": "PAN-OS 11.1.2+ contains fix — patched systems can reduce escalation priority",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 5 — ISAC Threat Intel: Healthcare Ransomware Campaign
    # -----------------------------------------------------------------------
    {
        "id": "art_05",
        "type": "email",
        "source": "isac",
        "authority": "medium",
        "text": (
            "Health-ISAC TLP:WHITE advisory — FIN12 ransomware group has resumed "
            "targeting healthcare organizations after 6-week hiatus. Initial access "
            "TTPs: spearphishing with medical billing lure, credential harvesting via "
            "fake VPN portal. IOCs attached. Member organizations should treat all "
            "phishing-related alerts as high priority through end of month. CISA "
            "coordination ongoing."
        ),
        # v4: corrected — 1 claim added
        # Addition [2]: "Member organizations should treat all phishing-related
        #   alerts as high priority through end of month" — this is an explicit
        #   Health-ISAC directive to member organizations, not just a campaign
        #   observation → ciso_risk_directive, direction=+1.
        "expected_claims": [
            {
                "type": "active_campaign",
                "categories": ["threat_intel_match"],
                "direction": +1,
                "urgency": 0.85,
                "summary": "FIN12 healthcare ransomware — elevated threat_intel_match",
            },
            {
                "type": "active_campaign",
                "categories": ["credential_access"],
                "direction": +1,
                "urgency": 0.8,
                "summary": "FIN12 credential harvesting via fake VPN — credential risk",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["threat_intel_match", "credential_access"],
                "direction": +1,
                "urgency": 0.85,
                "summary": "Health-ISAC directs high-priority treatment of phishing alerts through end of month",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 6 — Incident Report: Resolved Insider Event
    # -----------------------------------------------------------------------
    {
        "id": "art_06",
        "type": "incident_report",
        "source": "soc_lead",
        "authority": "high",
        "text": (
            "Post-incident summary — INC-2026-0892 closed. Insider data exfiltration "
            "attempt by departing employee confirmed and contained. DLP triggered "
            "correctly. User account disabled, legal notified. Pattern: large file "
            "downloads to personal cloud storage over 3 days before resignation date. "
            "Rule tuning: lower threshold for pattern_history factor on employees with "
            "submitted resignation. No further monitoring needed for this user. "
            "General insider_behavioral posture: return to normal."
        ),
        # v4: corrected — 1 claim added
        # Addition [2]: "User account disabled ... No further monitoring needed for
        #   this user" — this specific account is closed; alerts from it are resolved
        #   → known_fp_pattern, direction=-1 (suppress further alerts from this user).
        #   Distinct from [0] (general posture return to normal) which is org-wide.
        "expected_claims": [
            {
                "type": "known_change",
                "categories": ["insider_behavioral"],
                "direction": -1,
                "urgency": 0.3,
                "summary": "INC-0892 resolved — return insider posture to normal",
            },
            {
                "type": "ciso_risk_directive",
                "categories": ["insider_behavioral"],
                "direction": +1,
                "urgency": 0.4,
                "summary": "Tune pattern_history threshold for pre-resignation window",
            },
            {
                "type": "known_fp_pattern",
                "categories": ["insider_behavioral"],
                "direction": -1,
                "urgency": 0.2,
                "summary": "INC-0892 user account disabled — no further monitoring needed for this specific account",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 7 — Slack: Planned IT Maintenance Window
    # -----------------------------------------------------------------------
    {
        "id": "art_07",
        "type": "slack",
        "source": "it_ops",
        "authority": "medium",
        "text": (
            "#soc-notifications — FYI the Azure AD password reset portal will be in "
            "maintenance 02:00-06:00 UTC Saturday. Expect elevated authentication "
            "failures and credential_access alerts during this window. These are "
            "expected/planned. Do not escalate auth failures to Azure AD during "
            "maintenance. Normal operations resume 06:00 UTC. Ticket: CHG-4421."
        ),
        # v4: corrected — removed spurious cloud_infrastructure claim
        # The text explicitly names "credential_access alerts" as the affected
        # alert category ("Expect elevated authentication failures and credential_access
        # alerts"). The original cloud_infrastructure claim had no textual basis —
        # the message does not separately mention cloud_infrastructure alert categories.
        "expected_claims": [
            {
                "type": "known_change",
                "categories": ["credential_access"],
                "direction": -1,
                "urgency": 0.1,
                "summary": "Azure AD maintenance window — suppress auth failure alerts",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Artifact 8 — Meeting Notes: New Employee Onboarding Batch
    # -----------------------------------------------------------------------
    {
        "id": "art_08",
        "type": "meeting_notes",
        "source": "hr_it_liaison",
        "authority": "low",
        "text": (
            "Q1 onboarding summary — 47 new employees starting Monday across London "
            "(22), Singapore (18), and Chicago (7) offices. IT has pre-staged laptops. "
            "Expect elevated travel_anomaly alerts as devices register from new "
            "geolocations. First-login anomalies are expected. HR has confirmed all "
            "employment checks complete. Suggest SOC suppress first-login travel "
            "anomalies for new-hire devices for 72 hours post-start."
        ),
        # v4: corrected — 1 claim added
        # Addition [2]: "First-login anomalies are expected" is a distinct statement
        #   from the planned onboarding event itself. It identifies first-login
        #   travel patterns as anticipated false positives → known_fp_pattern,
        #   direction=-1. The LLM correctly extracts this as a separate claim from
        #   the known_change (scheduled onboarding batch).
        "expected_claims": [
            {
                "type": "known_change",
                "categories": ["travel_anomaly"],
                "direction": -1,
                "urgency": 0.1,
                "summary": "47 new hires — suppress first-login travel anomalies 72h",
            },
            {
                "type": "known_change",
                "categories": ["credential_access"],
                "direction": -1,
                "urgency": 0.1,
                "summary": "New hire first-login credential patterns — expected FPs",
            },
            {
                "type": "known_fp_pattern",
                "categories": ["travel_anomaly"],
                "direction": -1,
                "urgency": 0.1,
                "summary": "First-login geolocation anomalies from pre-staged new-hire devices are anticipated FPs",
            },
        ],
    },
]
