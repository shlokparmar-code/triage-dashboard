"""Honest evaluation of Module C (symptom triage) on hand-written test sentences.

Run from the project root:   python evaluate_symptom_triage.py

Why this exists: Module C is trained on synthetic text, so its training accuracy
says little about real wording. This script scores it on a small set of
independent, realistic sentences and reports the numbers that matter for safety:
recall on HIGH (missed emergencies) and under-triage.

IMPORTANT: have a teammate (ideally someone with medical knowledge) review the
labels below and add your own sentences. More cases make the numbers more reliable.
Do not copy sentences from the training data generator.
"""
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

TIERS = ["LOW", "MEDIUM", "HIGH"]
RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# (text, expected_tier, tag)   tag: plain | typo | short | negation
CASES: List[Tuple[str, str, str]] = [
    # ---------------- HIGH ----------------
    ("tightness in my chest and short of breath even while sitting", "HIGH", "plain"),
    ("crushing pain in the middle of my chest spreading to my left arm", "HIGH", "plain"),
    ("my father suddenly cannot speak properly and one side of his face is drooping", "HIGH", "plain"),
    ("she had a fit and is not waking up", "HIGH", "plain"),
    ("i am bleeding heavily and the cloth is soaked in minutes", "HIGH", "plain"),
    ("my baby is blue around the lips and breathing very fast", "HIGH", "plain"),
    ("i feel like ending my life", "HIGH", "plain"),
    ("lips and tongue swelling after eating peanuts, trouble swallowing", "HIGH", "plain"),
    ("fainted and now unable to stand, heart racing", "HIGH", "plain"),
    ("vomiting blood since morning", "HIGH", "plain"),
    ("sudden severe headache, worst of my life, with stiff neck", "HIGH", "plain"),
    ("cant breathe", "HIGH", "short"),
    ("chest pain pls help", "HIGH", "short"),
    ("snake bit my leg an hour ago and the leg is swelling fast", "HIGH", "plain"),
    ("8 months pregnant with severe headache, blurry vision and swollen face", "HIGH", "plain"),
    ("he is unconscious and not responding when i shake him", "HIGH", "plain"),
    ("i swallowed a lot of tablets by mistake and feel drowsy", "HIGH", "plain"),
    ("severe burns over my arm and chest from boiling oil", "HIGH", "plain"),
    # ---------------- MEDIUM ----------------
    ("fever for 3 days with body ache and chills", "MEDIUM", "plain"),
    ("burning when i pass urine and lower belly pain since 2 days", "MEDIUM", "plain"),
    ("persistent cough for two weeks with some yellow phlegm", "MEDIUM", "plain"),
    ("twisted my ankle yesterday, swollen and painful to walk on", "MEDIUM", "plain"),
    ("vomiting and loose motions since last night, feeling weak", "MEDIUM", "plain"),
    ("ear pain with fever in my 4 year old", "MEDIUM", "plain"),
    ("sugar level is high, very thirsty and passing urine often", "MEDIUM", "plain"),
    ("back pain for a week that goes down my leg", "MEDIUM", "plain"),
    ("bad stomach pain after eating, comes and goes", "MEDIUM", "plain"),
    ("deep cut on my hand that may need stitches, bleeding has stopped", "MEDIUM", "plain"),
    ("headache with vomiting since morning, no neck stiffness", "MEDIUM", "negation"),
    ("rash all over body with itching and mild fever", "MEDIUM", "plain"),
    ("fever of 102 for two days, not coming down with paracetamol", "MEDIUM", "plain"),
    ("sore throat and cant swallow food well since yesterday", "MEDIUM", "plain"),
    ("blood pressure reading was 160/100 today with mild headache", "MEDIUM", "plain"),
    ("wound on my foot is red and warm with pus", "MEDIUM", "plain"),
    ("dizzy whenever i stand up, for the last few days", "MEDIUM", "plain"),
    ("feverrr n bodyache since 3 days", "MEDIUM", "typo"),
    # ---------------- LOW ----------------
    ("dry cough off and on since two days, nothing else", "LOW", "plain"),
    ("runny nose and sneezing since morning", "LOW", "plain"),
    ("slight headache after a long day, no other symptoms", "LOW", "negation"),
    ("small scratch on my knee, not bleeding", "LOW", "plain"),
    ("feeling a bit tired after poor sleep", "LOW", "plain"),
    ("mild sore throat, no fever", "LOW", "negation"),
    ("dry skin on my hands in winter", "LOW", "plain"),
    ("occasional heartburn after spicy food", "LOW", "plain"),
    ("minor muscle ache after exercise yesterday", "LOW", "plain"),
    ("blocked nose, no fever, no cough", "LOW", "negation"),
    ("i have no chest pain and no breathing problem, just a mild cold", "LOW", "negation"),
    ("no fever, no vomiting, only slight acidity", "LOW", "negation"),
    ("tiny pimples on my face", "LOW", "plain"),
    ("bit of constipation for a day", "LOW", "plain"),
    ("insect bite, small itchy bump", "LOW", "plain"),
    ("mild cold", "LOW", "short"),
    ("slight headche since mornin", "LOW", "typo"),
    ("sneezing a lot from dust allergy, no trouble breathing", "LOW", "negation"),
]

EDGE_INPUTS = ["", "   ", "asdkjh qwerty zxcv", "the weather is nice today", "ab " * 2000]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(
    predict_fn: Callable[[str], Dict[str, Any]],
    cases: List[Tuple[str, str, str]] = CASES,
) -> Dict[str, Any]:
    """Run predict_fn on every case and compute the safety metrics."""
    rows = []
    for text, expected, tag in cases:
        try:
            pred = str(predict_fn(text).get("urgency", "ERROR")).upper()
        except Exception:
            pred = "ERROR"
        rows.append({"text": text, "expected": expected, "pred": pred, "tag": tag})

    confusion = {e: Counter() for e in TIERS}
    for r in rows:
        confusion[r["expected"]][r["pred"]] += 1

    per_class = {}
    for t in TIERS:
        support = sum(confusion[t].values())
        tp = confusion[t][t]
        predicted = sum(confusion[e][t] for e in TIERS)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[t] = {"support": support, "recall": recall, "precision": precision, "f1": f1}

    correct = sum(1 for r in rows if r["pred"] == r["expected"])
    under = [r for r in rows if r["pred"] in RANK and RANK[r["pred"]] < RANK[r["expected"]]]
    over = [r for r in rows if r["pred"] in RANK and RANK[r["pred"]] > RANK[r["expected"]]]
    high_missed = [r for r in rows if r["expected"] == "HIGH" and r["pred"] != "HIGH"]
    high_n = per_class["HIGH"]["support"]
    lo, hi = wilson_interval(high_n - len(high_missed), high_n)

    by_tag = {}
    for tag in sorted({r["tag"] for r in rows}):
        sub = [r for r in rows if r["tag"] == tag]
        by_tag[tag] = (sum(1 for r in sub if r["pred"] == r["expected"]), len(sub))

    return {
        "rows": rows,
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": sum(per_class[t]["f1"] for t in TIERS) / len(TIERS),
        "per_class": per_class,
        "confusion": confusion,
        "under_triage": under,
        "over_triage": over,
        "high_missed": high_missed,
        "high_recall_ci": (lo, hi),
        "by_tag": by_tag,
    }


def run_edge_cases(predict_fn: Callable[[str], Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Check that bad input never crashes. Returns (input_preview, urgency, status)."""
    out = []
    for text in EDGE_INPUTS:
        preview = (text[:30] + "...") if len(text) > 30 else repr(text)
        try:
            res = predict_fn(text)
            status = str(res.get("details", {}).get("validation_status", "n/a"))
            out.append((preview, str(res.get("urgency")), status))
        except Exception as exc:  # a crash is a finding
            out.append((preview, "CRASH", type(exc).__name__))
    return out


def build_report(result: Dict[str, Any], edge: List[Tuple[str, str, str]]) -> str:
    """Markdown report for docs/ and the write-up."""
    pc = result["per_class"]
    lo, hi = result["high_recall_ci"]
    L = [
        "# Module C: Evaluation on Independent Hand-Written Sentences",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        f"{result['n']} sentences ({pc['HIGH']['support']} HIGH, {pc['MEDIUM']['support']} MEDIUM, {pc['LOW']['support']} LOW).",
        "",
        "> Sentences were written by the project team, not taken from the training data. "
        "The sample is small, so treat these numbers as indicative, not as clinical validation.",
        "",
        "## Headline safety numbers",
        f"- **HIGH recall (emergencies caught): {pc['HIGH']['recall']:.0%}** "
        f"(95% CI {lo:.0%} to {hi:.0%}); missed emergencies: {len(result['high_missed'])}",
        f"- Under-triage (predicted lower than expected): {len(result['under_triage'])} of {result['n']}",
        f"- Over-triage (predicted higher than expected): {len(result['over_triage'])} of {result['n']}",
        f"- Accuracy: {result['accuracy']:.0%}   Macro-F1: {result['macro_f1']:.2f}",
        "",
        "## Per-class results",
        "| Tier | Cases | Recall | Precision | F1 |",
        "|---|---|---|---|---|",
    ]
    for t in TIERS:
        c = pc[t]
        L.append(f"| {t} | {c['support']} | {c['recall']:.0%} | {c['precision']:.0%} | {c['f1']:.2f} |")
    L += ["", "## Confusion matrix (rows = expected, columns = predicted)",
          "| Expected | LOW | MEDIUM | HIGH | Other |", "|---|---|---|---|---|"]
    for e in TIERS:
        other = sum(v for k, v in result["confusion"][e].items() if k not in TIERS)
        L.append(f"| {e} | " + " | ".join(str(result["confusion"][e][t]) for t in TIERS) + f" | {other} |")
    L += ["", "## Accuracy by sentence type", "| Type | Correct | Total |", "|---|---|---|"]
    for tag, (ok, tot) in result["by_tag"].items():
        L.append(f"| {tag} | {ok} | {tot} |")
    L += ["", "## Missed emergencies (most important)"]
    L += [f"- \"{r['text']}\" predicted {r['pred']}" for r in result["high_missed"]] or ["- None on this set."]
    L += ["", "## Other under-triaged cases"]
    others = [r for r in result["under_triage"] if r["expected"] != "HIGH"]
    L += [f"- \"{r['text']}\" expected {r['expected']}, predicted {r['pred']}" for r in others] or ["- None."]
    L += ["", "## Edge-case inputs (must not crash)", "| Input | Urgency | Status |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in edge]
    L += ["", "## Limitations",
          "- Small sample written by the team; labels need review by a clinician.",
          "- English only. Regional languages and code-mixed text are not covered.",
          "- Not a clinical validation. For decision support only."]
    return "\n".join(L) + "\n"


def main() -> None:
    from modules.symptom_triage.model import predict_urgency

    result = evaluate(predict_urgency)
    edge = run_edge_cases(predict_urgency)
    report = build_report(result, edge)
    out_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "symptom_eval_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()