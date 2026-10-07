"""Red-Flag Rule Layer for Symptom Triage.

Overrides ML predictions to force HIGH urgency whenever life-threatening
emergency triggers are present, with explicit negation awareness to avoid
false alarms (e.g., 'no chest pain').
"""

import re
from typing import List, Optional, Tuple

# Core emergency patterns that demand immediate medical attention
RED_FLAG_PATTERNS = [
    # Cardiac / Acute Chest
    r"\b(?:crushing|severe|radiating|intense|squeezing)\s+(?:chest\s+pain|chest\s+pressure|heart\s+ache)\b",
    r"\bchest\s+(?:pain|tightness|pressure|ache|discomfort)\b",
    r"\bheart\s+attack\b",
    r"\bpain\s+(?:radiating\s+to\s+(?:left\s+arm|jaw|back|neck))\b",

    # Respiratory
    r"\b(?:can\'?t|cannot|unable\s+to)\s+breathe\b",
    r"\b(?:gasping\s+for\s+air|struggling\s+to\s+breathe|severe\s+shortness\s+of\s+breath|suffocating)\b",
    r"\b(?:choking|respiratory\s+failure|blue\s+lips|cyanosis)\b",

    # Neurological / Stroke
    r"\b(?:stroke|facial\s+droop|face\s+drooping|slurred\s+speech)\b",
    r"\b(?:sudden\s+paralysis|numbness\s+on\s+one\s+side|loss\s+of\s+speech|cannot\s+speak)\b",
    r"\b(?:sudden\s+blindness|sudden\s+loss\s+of\s+vision)\b",
    r"\b(?:seizure|convulsion|grand\s+mal|status\s+epilepticus)\b",

    # Consciousness / Trauma
    r"\b(?:unconscious|passed\s+out|collapsed|unresponsive|fainted|blacked\s+out)\b",
    r"\b(?:loss\s+of\s+consciousness|syncope)\b",

    # Severe Hemorrhage
    r"\b(?:coughing\s+up\s+blood|vomiting\s+blood|throwing\s+up\s+blood|hematemesis|hemoptysis)\b",
    r"\b(?:severe\s+bleeding|uncontrolled\s+bleeding|gushing\s+blood|hemorrhag\w*)\b",

    # Anaphylaxis / Allergic
    r"\b(?:anaphylaxis|throat\s+closing|throat\s+swelling|tongue\s+swelling|cannot\s+swallow\s+air)\b",
    r"\bsevere\s+allergic\s+reaction\b",

    # Psychiatric / Harm
    r"\b(?:want\s+to\s+die|kill\s+myself|suicid\w*|end\s+my\s+life|self\s*harm)\b",

    # Acute Abdominal / Sepsis
    r"\b(?:rigid\s+abdomen|unbearable\s+abdominal\s+pain|ruptured\s+appendix)\b",
    r"\b(?:sepsis|septic\s+shock|high\s+fever\s+and\s+confused)\b",
]

# Negation terms that precede or follow symptoms
NEGATION_PREFIXES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bnegative\s+for\b",
    r"\bfree\s+of\b",
    r"\bruled\s+out\b",
    r"\bnever\s+had\b",
    r"\bdoes\s*n\'?t\s+have\b",
    r"\bdid\s*n\'?t\s+have\b",
]


def check_red_flags(text: str) -> Tuple[bool, List[str]]:
    """Scan raw text for emergency red flags, taking into account negation.

    Args:
        text: User input symptom text.

    Returns:
        Tuple of (is_flagged: bool, list_of_triggered_reasons: List[str]).
    """
    if not text or not text.strip():
        return False, []

    lower_text = text.lower()
    # Split into clinical clauses by punctuation to localize negation
    clauses = re.split(r"[,.;!?]|\b(?:but|however|except|although)\b", lower_text)
    
    triggered_reasons = []

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        for pattern in RED_FLAG_PATTERNS:
            match = re.search(pattern, clause, re.IGNORECASE)
            if match:
                matched_span = match.span()
                matched_text = match.group(0)
                
                # Check if there is an active negation before the matched symptom in this clause
                prefix_subclause = clause[: matched_span[0]]
                is_negated = False
                for neg in NEGATION_PREFIXES:
                    # Look for negation within 4 words preceding the trigger
                    neg_match = re.search(neg + r"(?:\s+\w+){0,3}\s*$", prefix_subclause)
                    if neg_match:
                        is_negated = True
                        break

                if not is_negated:
                    reason = f"Emergency Red-Flag detected: '{matched_text}'"
                    if reason not in triggered_reasons:
                        triggered_reasons.append(reason)

    return len(triggered_reasons) > 0, triggered_reasons
