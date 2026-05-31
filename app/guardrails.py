"""Safety guardrails.

This runs BEFORE retrieval and generation. If a query describes an urgent
or high-risk situation, we do not attempt to "answer" it from documents at
all — we return an escalation message that points the person to emergency
care. For a patient-facing health tool, a false positive (escalating a
non-emergency) is far cheaper than a false negative, so the matching is
intentionally broad and we only suppress on explicit negation.

Detection is rule-based (regex over word boundaries) on purpose: it is
transparent, deterministic, testable, and easy to audit/extend — qualities
that matter more than cleverness for a safety gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Each category maps to (compiled patterns, escalation message).
_EMERGENCY_PATTERNS: Dict[str, List[str]] = {
    "cardiac": [
        r"chest pain",
        r"chest pressure",
        r"chest tightness",
        r"crushing (?:feeling|pain) in (?:my )?chest",
        r"pain (?:spreading|radiating) (?:to|down) (?:my )?(?:arm|jaw|shoulder|back)",
        r"heart attack",
    ],
    "respiratory": [
        r"can(?:'?t| ?not) breathe",
        r"struggling to breathe",
        r"gasping (?:for (?:air|breath))?",
        r"severe(?:ly)? short(?:ness)? of breath",
        r"sudden short(?:ness)? of breath",
        r"choking",
    ],
    "neurological": [
        r"fainted|fainting|passed out|black(?:ed)? out|syncope",
        r"severe(?:ly)? dizz(?:y|iness)",
        r"slurred speech",
        r"face (?:is )?droop",
        r"sudden (?:weakness|numbness)",
        r"can(?:'?t| ?not) move (?:my )?(?:arm|leg|side)",
        r"sudden (?:confusion|severe headache)",
        r"\bstroke\b",
        r"seizure",
    ],
    "general_emergency": [
        r"coughing up blood",
        r"vomiting blood",
        r"unconscious|unresponsive",
        r"call 9-?1-?1",
        r"life[- ]threatening",
    ],
}

# Self-harm / crisis is handled as its own category with supportive routing.
_CRISIS_PATTERNS: List[str] = [
    r"suicid",
    r"kill (?:myself|himself|herself)",
    r"end (?:my|his|her) life",
    r"don'?t want to (?:live|be alive)",
    r"harm (?:myself|himself|herself)",
    r"hurt (?:myself|himself|herself)",
]

# Light negation guard: skip a match if it is directly negated, e.g.
# "I have no chest pain" or "without chest pain". Kept narrow on purpose.
_NEGATION_WINDOW = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\bdon'?t have\b|\bdenies?\b)\s+(?:\w+\s+){0,2}$"
)

_EMERGENCY_MESSAGE = (
    "This looks like it could be a medical emergency. I can't give medical advice "
    "for urgent symptoms. If you are experiencing symptoms like chest pain or "
    "pressure, severe or sudden shortness of breath, fainting, severe dizziness, "
    "sudden weakness or trouble speaking, or any symptom that feels life-threatening, "
    "please call your local emergency number (911 in Canada/US) or go to the nearest "
    "emergency department right now. If you are with someone showing these symptoms, "
    "stay with them and call for help immediately."
)

_CRISIS_MESSAGE = (
    "I'm really sorry you're going through this, and I'm not able to help with this "
    "safely as an information tool. If you are thinking about harming yourself or are "
    "in crisis, please reach out right now to someone who can help. In Canada and the "
    "US you can call or text 988 to reach a suicide and crisis line, available 24/7. "
    "If you are in immediate danger, please call 911 or go to your nearest emergency "
    "department. You deserve support from a real person who can talk this through with you."
)


@dataclass
class GuardrailResult:
    triggered: bool
    category: Optional[str] = None  # cardiac | respiratory | neurological | general_emergency | crisis
    message: Optional[str] = None
    matched_terms: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.matched_terms is None:
            self.matched_terms = []


def _is_negated(text: str, match_start: int) -> bool:
    """True if the few words immediately before the match are a negation."""
    preceding = text[:match_start]
    return bool(_NEGATION_WINDOW.search(preceding))


def _scan(text: str, patterns: List[str]) -> List[str]:
    hits: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if _is_negated(text, m.start()):
                continue
            hits.append(m.group(0))
    return hits


def check_guardrails(question: str) -> GuardrailResult:
    text = question.strip().lower()

    # Crisis routing takes precedence over everything else.
    crisis_hits = _scan(text, _CRISIS_PATTERNS)
    if crisis_hits:
        return GuardrailResult(True, "crisis", _CRISIS_MESSAGE, crisis_hits)

    for category, patterns in _EMERGENCY_PATTERNS.items():
        hits = _scan(text, patterns)
        if hits:
            return GuardrailResult(True, category, _EMERGENCY_MESSAGE, hits)

    return GuardrailResult(False)
