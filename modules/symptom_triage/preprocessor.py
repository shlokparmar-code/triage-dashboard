"""Text Preprocessing, Negation Handling, and Input Validation for Symptom Triage."""

import re
from typing import Dict, Tuple

# Common medical typos & colloquial spellings mapping
TYPO_CORRECTIONS = {
    "chesst": "chest",
    "cheest": "chest",
    "hart": "heart",
    "breathin": "breathing",
    "brethen": "breathing",
    "dizzines": "dizziness",
    "dizzyness": "dizziness",
    "hedache": "headache",
    "headche": "headache",
    "hedache": "headache",
    "migrane": "migraine",
    "stomac": "stomach",
    "stomache": "stomach",
    "fevr": "fever",
    "feverr": "fever",
    "nausia": "nausea",
    "nauseous": "nausea",
    "nauseas": "nausea",
    "vomitting": "vomiting",
    "vomitted": "vomited",
    "diarea": "diarrhea",
    "diarhea": "diarrhea",
    "coughh": "cough",
    "coughed": "cough",
    "faintt": "faint",
    "swolen": "swollen",
    "sweling": "swelling",
    "bleading": "bleeding",
    "alergy": "allergy",
    "alergic": "allergic",
    "shuger": "sugar",
}

CONTRACTIONS = {
    r"can\'t": "cannot",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}

# Negation trigger keywords
NEGATION_WORDS = {"no", "not", "denies", "without", "never", "hardly", "free"}

# Common medical vocabulary words for validity check
COMMON_MEDICAL_TERMS = {
    "pain", "ache", "hurts", "fever", "cough", "cold", "flu", "sore", "throat",
    "headache", "chest", "breath", "breathing", "stomach", "nausea", "vomit",
    "dizzy", "dizziness", "tired", "fatigue", "weak", "weakness", "swelling",
    "rash", "burn", "blood", "bleeding", "cut", "wound", "cramp", "pressure",
    "heart", "palpitation", "diarrhea", "constipation", "vision", "blur", "eye",
    "ear", "back", "neck", "joint", "muscle", "knee", "leg", "arm", "numb",
    "tingling", "infection", "chills", "sweat", "appetite", "weight", "sleep",
    "insomnia", "itch", "itching", "bruise", "urination", "urine", "discharge",
    "seizure", "faint", "allergy", "allergic", "sneezing", "congestion", "sinus",
    "stroke", "droop", "drooping", "speech", "slurred", "paralysis", "numbness",
    "unconscious", "passed", "collapse", "choke", "choking", "suicide", "gasp", "gasping"
}


def clean_and_normalize_text(text: str) -> str:
    """Clean punctuation, expand contractions, and correct common medical misspellings."""
    if not text:
        return ""

    text = text.lower().strip()

    # Expand contractions
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # Replace punctuation except sentence boundaries
    text = re.sub(r"[^\w\s.,;:!?]", " ", text)

    # Word-level typo correction
    words = text.split()
    corrected_words = [TYPO_CORRECTIONS.get(w, w) for w in words]
    return " ".join(corrected_words)


def apply_negation_tagging(text: str) -> str:
    """Transform negated tokens (e.g., 'no fever' -> 'not_fever') to prevent false positive triggers.

    Transforms words within a 3-word window following a negation word in the same clause.
    """
    clauses = re.split(r"[,.;:!?]|\b(?:but|however|except|although)\b", text)
    processed_clauses = []

    for clause in clauses:
        words = clause.strip().split()
        if not words:
            continue

        negation_active = 0
        new_words = []
        for word in words:
            if word in NEGATION_WORDS or word.endswith("n't"):
                negation_active = 3  # Mark next 3 words as negated
                new_words.append(word)
            elif negation_active > 0:
                new_words.append(f"not_{word}")
                negation_active -= 1
            else:
                new_words.append(word)
        processed_clauses.append(" ".join(new_words))

    return " ".join(processed_clauses)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: normalization + negation tagging."""
    cleaned = clean_and_normalize_text(text)
    negated = apply_negation_tagging(cleaned)
    return negated


def validate_input(text: str) -> Tuple[bool, str, str]:
    """Validate user input against edge cases: empty, gibberish, too short/long, or non-medical.

    Returns:
        Tuple of (is_valid: bool, status_code: str, user_message: str)
    """
    if not text or not text.strip():
        return False, "EMPTY", "No symptoms entered. Please describe what you are experiencing."

    cleaned = text.strip()
    if len(cleaned) < 4:
        return False, "TOO_SHORT", "Description is too short. Please provide more details."

    if len(cleaned) > 2000:
        # Cap length to prevent DOS / resource exhaustion
        cleaned = cleaned[:2000]

    # Check for gibberish: high character repetition or random consonant clustering
    if re.search(r"(.)\1{4,}", cleaned):  # e.g., 'aaaaaa' or 'zzzzzz'
        return False, "GIBBERISH", "Input appears to contain repeated random characters. Please describe your symptoms."

    words = re.findall(r"\b[a-zA-Z]{2,}\b", cleaned.lower())
    if not words:
        return False, "GIBBERISH", "No recognizable words found. Please describe symptoms clearly."

    # Check vowel ratio in words
    vowel_count = sum(1 for c in cleaned.lower() if c in "aeiou")
    letter_count = sum(1 for c in cleaned if c.isalpha())
    if letter_count > 10 and (vowel_count / letter_count < 0.15 or vowel_count / letter_count > 0.85):
        return False, "GIBBERISH", "Input text does not resemble natural language. Please enter standard symptoms."

    # Check medical relevance: test if any medical root or keyword appears
    # Also permit general complaints like 'hurts', 'sick', 'unwell', 'bad'
    general_health_cues = {"sick", "ill", "unwell", "bad", "problem", "suffering", "feeling", "hurt", "sore"}
    has_medical_cues = any(w in COMMON_MEDICAL_TERMS or w in general_health_cues for w in words)
    
    # Check if input is purely a greeting or conversational noise
    common_greetings = {"hello", "hi", "hey", "good morning", "good evening", "how are you", "who are you", "test", "testing"}
    is_pure_greeting = cleaned.lower().strip() in common_greetings

    if is_pure_greeting:
        return False, "NON_MEDICAL", "Unable to assess — please describe symptoms (e.g., headache, fever, cough)."

    if len(words) >= 4 and not has_medical_cues:
        # If text is substantial but has zero medical or symptom terms, flag as non-medical
        return False, "NON_MEDICAL", "Unable to assess — please describe symptoms (e.g., pain, dizziness, nausea)."

    return True, "VALID", "Input is valid for clinical NLP assessment."
