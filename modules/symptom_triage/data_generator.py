"""Dataset generator and synthesizer for clinical symptom triage NLP.

Produces a diverse, balanced dataset of >= 1800 labeled sentences across
LOW, MEDIUM, and HIGH urgency tiers with varied sentence lengths, colloquialisms,
medical typos, and negation patterns.

Label Assignment Clinical Protocol:
- HIGH (Emergency / Immediate care): Conditions with acute threat to life, organ,
  or limb (e.g., acute coronary syndrome, acute dyspnea, acute focal neurological
  deficits, severe uncontrolled hemorrhage, anaphylaxis, severe psychiatric crisis).
- MEDIUM (Urgent / 24-48h physician consultation): Progressive, infectious, or
  moderate systemic conditions (e.g., persistent high fever, moderate localized
  abdominal pain, productive lower respiratory infections, suspected fractures,
  urinary tract infections with systemic signs).
- LOW (Non-urgent / Primary self-care): Self-limiting, mild, or chronic non-acute
  complaints (e.g., upper respiratory coryza, mild tension headache, post-exercise
  myalgia, minor dermatological blemishes, indigestion).
"""

import random
import pandas as pd
from pathlib import Path
import sys

# Ensure root is importable
sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

RANDOM_SEED = config.RANDOM_SEED
random.seed(RANDOM_SEED)

# HIGH TIER TEMPLATES & ATOMS (Emergencies)
HIGH_COMPONENTS = {
    "subjects": ["I", "My father", "My mother", "Patient", "My husband", "My wife", "I am", "Suddenly I"],
    "cardiac": [
        "crushing chest pain radiating down my left arm and jaw",
        "intense squeezing pressure in my chest with cold sweats",
        "sudden heavy pressure on chest and cannot catch my breath",
        "severe chest pain like an elephant sitting on my chest",
        "sharp chest pain with numbness in left fingers and sweating",
        "heart is racing at 160 bpm and I feel like passing out",
        "sudden central chest tightness accompanied by severe dizziness",
    ],
    "respiratory": [
        "cannot breathe and lips are turning blue",
        "gasping for air and inhaler is not working at all",
        "severe shortness of breath, cannot speak full sentences",
        "choking sensation and throat feels like it is closing completely",
        "sudden severe asthma attack with extreme wheezing and suffocation",
        "struggling to inhale and feeling dizzy from lack of oxygen",
    ],
    "neurological": [
        "face is drooping on the right side and speech is completely slurred",
        "sudden paralysis in right arm and unable to move my leg",
        "sudden worst headache of my life like a thunderclap",
        "having continuous seizures and shaking uncontrollably for 5 minutes",
        "sudden loss of vision in both eyes and extreme confusion",
        "collapsed to the floor and passed out cold for 2 minutes",
    ],
    "hemorrhage_trauma": [
        "coughing up large amounts of bright red blood",
        "vomiting dark blood and severe black tarry stools",
        "spurting arterial bleeding from deep leg laceration that will not stop",
        "profuse bleeding after head injury and losing consciousness",
        "uncontrolled bleeding from open wound after severe trauma",
    ],
    "anaphylaxis": [
        "severe allergic reaction, throat is swelling shut after peanut exposure",
        "anaphylactic shock symptoms, tongue swollen and cannot swallow",
        "allergic reaction to medication with full body hives and wheezing",
    ],
    "psychiatric": [
        "feeling completely overwhelmed and having active thoughts of suicide",
        "cannot go on, want to end my life right now",
    ]
}

# MEDIUM TIER TEMPLATES & ATOMS (Urgent / 24-48h care)
MEDIUM_COMPONENTS = {
    "infections": [
        "high fever of 103F for three days with persistent shivering and body aches",
        "fever of 39C that does not break with paracetamol, feeling very weak",
        "deep chesty productive cough with thick yellow and green mucus for 4 days",
        "shivering chills with persistent temperature and severe sweating at night",
    ],
    "abdominal": [
        "sharp pain in lower right side of abdomen that hurts more when pressed",
        "persistent stomach cramping with diarrhea and nausea for over 36 hours",
        "moderate abdominal pain after eating fatty foods with nausea",
        "severe burning sensation in stomach accompanied by repeated vomiting",
    ],
    "urinary": [
        "burning during urination with fever and moderate lower back ache",
        "frequent painful urination and urine appears cloudy with pink tint",
        "intense flank pain radiating to groin when passing urine",
    ],
    "musculoskeletal": [
        "twisted ankle very badly while running, swollen and cannot put any weight on it",
        "fell down stairs and wrist is swollen, deformed and very painful to touch",
        "intense knee pain with swelling and locking after a sports injury",
    ],
    "neurological_sensory": [
        "throbbing migraine headache for 48 hours with severe sensitivity to light and vomiting",
        "earache with thick yellow fluid draining from the ear canal",
        "intense vertigo and dizziness where room is spinning constantly when I turn head",
    ],
    "metabolic_dermatological": [
        "blood sugar monitor reading 320 mg/dL with extreme thirst and frequent urination",
        "red swollen tender rash spreading across lower leg that feels warm to the touch",
        "deep cut on index finger from kitchen knife that needs stitches to close",
    ]
}

# LOW TIER TEMPLATES & ATOMS (Self-care / Non-urgent)
LOW_COMPONENTS = {
    "respiratory_cold": [
        "mild runny nose and sneezing for 2 days with no fever",
        "slight dry scratchy throat and clear nasal discharge",
        "mild congestion in nose and occasional dry cough",
        "common cold symptoms, stuffy nose and mild sneezing",
        "mild tickle in throat, feeling a little run down but eating normally",
    ],
    "headache_fatigue": [
        "mild tension headache after staring at computer screen all day",
        "feeling tired and mild headache from lack of sleep last night",
        "slight forehead ache after stressful workday, relieved by rest",
        "mild fatigue after working long hours this week",
    ],
    "musculoskeletal_minor": [
        "sore quadriceps and calf muscles after gym workout yesterday",
        "mild neck stiffness from sleeping in an awkward position",
        "slight lower back soreness after doing yard work",
        "minor ache in shoulder after carrying heavy groceries",
    ],
    "dermatological_minor": [
        "small paper cut on thumb that stopped bleeding immediately",
        "minor bruise on shin from bumping into the coffee table",
        "dry flaky itchy skin patches on elbows during dry winter weather",
        "mild sunburn on shoulders from being at the beach yesterday",
        "small pimple on chin that is slightly tender",
    ],
    "gastrointestinal_minor": [
        "mild bloating and gas after eating heavy meal with beans",
        "slight indigestion and mild heartburn after drinking acidic coffee",
        "mild constipation for one day, no severe pain or vomiting",
        "slight stomach rumbling and mild loss of appetite today",
    ],
    "negation_cases": [
        "mild headache but no chest pain and no dizziness",
        "slight cough for two days, no shortness of breath, no fever",
        "minor knee soreness, no swelling, no numbness, can walk easily",
        "slight stomach upset without any vomiting and without high fever",
        "felt a bit tired today, denies any breathing problems or heart palpitations",
    ]
}

TYPO_PROBABILITY = 0.20
TYPO_REPLACEMENTS = {
    "chest": "chesst",
    "breathing": "breathin",
    "dizziness": "dizzines",
    "headache": "hedache",
    "stomach": "stomac",
    "fever": "fevr",
    "nausea": "nausia",
    "vomiting": "vomitting",
    "cough": "coughh",
    "swelling": "sweling",
    "allergic": "alergic",
}


def inject_typos_and_slang(text: str) -> str:
    """Randomly inject realistic typos or colloquial slang."""
    words = text.split()
    new_words = []
    for w in words:
        clean_w = w.lower().strip(".,;:!?")
        if random.random() < TYPO_PROBABILITY and clean_w in TYPO_REPLACEMENTS:
            replacement = TYPO_REPLACEMENTS[clean_w]
            # preserve capitalization
            if w.istitle():
                replacement = replacement.capitalize()
            new_words.append(replacement)
        else:
            new_words.append(w)
    return " ".join(new_words)


def generate_symptom_dataset(samples_per_tier: int = 650) -> pd.DataFrame:
    """Generate a balanced, clinically structured symptom triage dataset.

    Total samples = 3 * samples_per_tier (default >= 1950 samples).
    """
    records = []

    # HIGH TIER
    high_all = (
        HIGH_COMPONENTS["cardiac"]
        + HIGH_COMPONENTS["respiratory"]
        + HIGH_COMPONENTS["neurological"]
        + HIGH_COMPONENTS["hemorrhage_trauma"]
        + HIGH_COMPONENTS["anaphylaxis"]
        + HIGH_COMPONENTS["psychiatric"]
    )
    time_phrases_high = ["started 20 minutes ago", "happened suddenly", "getting progressively worse", "right now", "for the last hour"]

    for _ in range(samples_per_tier):
        base = random.choice(high_all)
        # Random variations
        r = random.random()
        if r < 0.35:
            subj = random.choice(HIGH_COMPONENTS["subjects"])
            sent = f"{subj} {base}."
        elif r < 0.65:
            t = random.choice(time_phrases_high)
            sent = f"{base}, {t}."
        else:
            sent = base
        
        sent = inject_typos_and_slang(sent)
        records.append({"text": sent, "urgency": config.TIER_HIGH, "category": "Emergency/Critical"})

    # MEDIUM TIER
    med_all = (
        MEDIUM_COMPONENTS["infections"]
        + MEDIUM_COMPONENTS["abdominal"]
        + MEDIUM_COMPONENTS["urinary"]
        + MEDIUM_COMPONENTS["musculoskeletal"]
        + MEDIUM_COMPONENTS["neurological_sensory"]
        + MEDIUM_COMPONENTS["metabolic_dermatological"]
    )
    time_phrases_med = ["for the past 2 days", "since yesterday morning", "for about 3 days now", "getting somewhat uncomfortable"]

    for _ in range(samples_per_tier):
        base = random.choice(med_all)
        r = random.random()
        if r < 0.35:
            sent = f"Patient reports {base.lower()}."
        elif r < 0.65:
            t = random.choice(time_phrases_med)
            sent = f"{base}, {t}."
        else:
            sent = base
        
        sent = inject_typos_and_slang(sent)
        records.append({"text": sent, "urgency": config.TIER_MEDIUM, "category": "Urgent/Physician_Consult"})

    # LOW TIER
    low_all = (
        LOW_COMPONENTS["respiratory_cold"]
        + LOW_COMPONENTS["headache_fatigue"]
        + LOW_COMPONENTS["musculoskeletal_minor"]
        + LOW_COMPONENTS["dermatological_minor"]
        + LOW_COMPONENTS["gastrointestinal_minor"]
        + LOW_COMPONENTS["negation_cases"]
    )
    prefixes_low = ["I have", "Just experiencing", "Mild complaint:", "Dealing with", "Note:"]

    for _ in range(samples_per_tier):
        base = random.choice(low_all)
        r = random.random()
        if r < 0.30:
            p = random.choice(prefixes_low)
            sent = f"{p} {base.lower()}."
        else:
            sent = base
        
        sent = inject_typos_and_slang(sent)
        records.append({"text": sent, "urgency": config.TIER_LOW, "category": "NonUrgent/SelfCare"})

    df = pd.DataFrame(records).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def save_symptom_dataset(filepath: Path = None) -> Path:
    """Generate and persist the labeled symptoms dataset to CSV."""
    if filepath is None:
        filepath = config.DATA_DIR / "symptoms_labeled.csv"
    
    df = generate_symptom_dataset(samples_per_tier=650)
    df.to_csv(filepath, index=False)
    print(f"[Dataset] Generated {len(df)} samples and saved to {filepath}")
    print(df["urgency"].value_counts())
    return filepath


if __name__ == "__main__":
    save_symptom_dataset()
