"""Extra analyses for the paper, from saved MIMIC outputs only. Local only; writes aggregates.

uv run python scripts/paper_extras.py   # -> evals/results/paper_extras.json

- patients behind the 200-note and 51-note evaluation sets
- potential false starts: `started` medications whose name (first word of 4+ letters) appears in
  the note's admission-medication section, for the rules, the model-made reference and each model,
  with how many of the rules' flagged note-drug pairs each model repeats
- paired, patient-level bootstrap differences in micro F1 for the comparisons the paper relies on
- reference support (items per field) and macro F1 over fields
- the rules' scores against each reference, with the same bootstrap CI as `mimic-bench`

No note text, label text, prediction text or identifier is written or printed.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ctagent.benchmark import bootstrap
from ctagent.extract_eval import FIELDS, evaluate
from ctagent.extraction import DischargeExtraction
from ctagent.mimic import MimicPaths
from ctagent.mimic_bench import gold_notes
from ctagent.notes import EXTRACTION_SECTIONS

REPO = Path(__file__).resolve().parents[1]
MODELS = [
    "ctagent-extractor_zero",
    "ctagent-extractor-s2500_zero",
    "ctagent-extractor-c1250_zero",
    "ctagent-extractor-c2500_zero",
    "qwen3-8b-local_zero",
    "qwen3-8b_zero",
    "qwen3-8b_few",
    "qwen3-14b_zero",
    "qwen3-14b_few",
]
DX = ("principal_diagnosis", "secondary_diagnoses")
# (reference, fields, model a, model b): micro F1 of a minus b
PAIRED = [
    ("hand", DX, "ctagent-extractor_zero", "qwen3-14b_zero"),
    ("hand", DX, "ctagent-extractor-c1250_zero", "qwen3-8b_few"),
    ("model-made", FIELDS, "qwen3-14b_zero", "ctagent-extractor_zero"),
    ("model-made", FIELDS, "qwen3-14b_zero", "ctagent-extractor-c2500_zero"),
]
HEAD = re.compile(r"^(" + "|".join(s.upper() for s in EXTRACTION_SECTIONS) + r"):\s*$", re.M)

Ext = DischargeExtraction | None


def admission_section(text: str) -> str:
    parts = HEAD.split(text)
    sections = {parts[i].lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return sections.get("meds_admission", "").lower()


def first_word(name: str) -> str:
    words = re.findall(r"[a-z][a-z-]{3,}", name.lower())
    return words[0] if words else ""


def false_starts(
    exts: dict[str, Ext], adm: dict[str, str]
) -> tuple[int, int, set[tuple[str, str]]]:
    """Started rows, flagged rows, and flagged (note, drug word) pairs."""
    started = flagged = 0
    pairs: set[tuple[str, str]] = set()
    for folder, ext in exts.items():
        for m in ext.medication_changes if ext else []:
            if m.change.value != "started":
                continue
            started += 1
            word = first_word(m.name)
            if word and word in adm[folder]:
                flagged += 1
                pairs.add((folder, word))
    return started, flagged, pairs


def note_counts(
    preds: dict[str, Ext], ref: dict[str, DischargeExtraction], fields: tuple[str, ...]
):  # type: ignore[no-untyped-def]
    out = {}
    for folder, label in ref.items():
        rep = evaluate([(preds.get(folder), label)], fields)
        out[folder] = np.array(
            [[rep.relaxed[f].tp, rep.relaxed[f].fp, rep.relaxed[f].fn] for f in fields]
        ).sum(0)
    return out


def micro_f1(c: np.ndarray) -> float:
    tp, fp, fn = (float(x) for x in c)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main(n_boot: int = 2000, seed: int = 17) -> dict[str, Any]:
    load_dotenv(REPO / ".env")
    work = MimicPaths.from_env().work_dir
    notes = {f: (text, lab) for f, text, lab in gold_notes(work, "silver")}
    hand = {f: lab for f, _, lab in gold_notes(work, "gold")}
    made = {f: lab for f, _, lab in gold_notes(work, "claude")}
    test = pd.read_parquet(work / "test.parquet", columns=["note_id", "subject_id"])
    subject = dict(zip(test["note_id"], test["subject_id"], strict=True))
    patient = {f: subject[f.split("_", 1)[1]] for f in notes}

    out: dict[str, Any] = {"patients": {}}
    for name, folders in (("200-note set", list(notes)), ("51 hand-labeled", list(hand))):
        per = Counter(patient[f] for f in folders)
        out["patients"][name] = {"notes": len(folders), "patients": len(per)}

    preds: dict[str, dict[str, Ext]] = {}
    for tag in MODELS:
        lines = (work / "preds" / f"silver_{tag}.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        preds[tag] = {
            r["folder"]: DischargeExtraction.model_validate(r["pred"]) if r["pred"] else None
            for r in rows
        }

    adm = {f: admission_section(text) for f, (text, _) in notes.items()}
    s, fl, rule_pairs = false_starts({f: lab for f, (_, lab) in notes.items()}, adm)
    starts = {"rules": {"started": s, "flagged": fl, "pairs": len(rule_pairs)}}
    for name, exts in [("model-made reference", dict(made))] + [(t, preds[t]) for t in MODELS]:
        s, fl, pairs = false_starts(exts, adm)  # type: ignore[arg-type]
        starts[name] = {
            "started": s,
            "flagged": fl,
            "share": round(fl / s, 3) if s else 0.0,
            "pairs": len(pairs),
            "pairs_shared_with_rules": len(pairs & rule_pairs),
        }
    out["potential_false_starts"] = starts

    refs = {"hand": hand, "model-made": made}
    rng = np.random.default_rng(seed)
    out["paired_differences"] = []
    for ref_name, fields, a, b in PAIRED:
        ref = refs[ref_name]
        ca, cb = note_counts(preds[a], ref, fields), note_counts(preds[b], ref, fields)
        groups: dict[Any, list[str]] = {}
        for f in ref:
            groups.setdefault(patient[f], []).append(f)
        keys = list(groups)
        diffs = []
        for _ in range(n_boot):
            pick = [f for k in rng.choice(len(keys), len(keys)) for f in groups[keys[k]]]
            diffs.append(micro_f1(sum(ca[f] for f in pick)) - micro_f1(sum(cb[f] for f in pick)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["paired_differences"].append(
            {
                "reference": ref_name,
                "fields": list(fields),
                "a": a,
                "b": b,
                "difference": round(micro_f1(sum(ca.values())) - micro_f1(sum(cb.values())), 3),
                "ci95": [round(float(lo), 3), round(float(hi), 3)],
            }
        )

    def support(ref: dict[str, DischargeExtraction], fields: tuple[str, ...]) -> dict[str, int]:
        return {
            f: sum(
                (1 if getattr(lab, f) else 0) if f == DX[0] else len(getattr(lab, f))
                for lab in ref.values()
            )
            for f in fields
        }

    out["support"] = {"hand": support(hand, DX), "model-made": support(made, FIELDS)}
    rules = {f: lab for f, (_, lab) in notes.items()}
    out["rules"] = {}
    for ref_name, fields in (("hand", DX), ("model-made", FIELDS)):
        rule_pairs = [(rules[f], lab) for f, lab in refs[ref_name].items()]
        rep = evaluate(rule_pairs, fields).summary()
        rel = rep["relaxed"]
        out["rules"][ref_name] = {
            "relaxed_f1": {k: round(rel[k]["f1"], 3) for k in (*fields, "micro")},  # type: ignore[index]
            "micro_ci95": [round(x, 3) for x in bootstrap(rule_pairs, fields=fields)["micro"]],
        }
        if "medication_changes" in fields:
            out["rules"][ref_name]["med_change_type_accuracy"] = round(
                rep["med_change_type_accuracy"],  # type: ignore[arg-type]
                2,
            )
    out["macro_f1_model_made"] = {}
    for tag, exts in [("rules", rules)] + [(t, preds[t]) for t in MODELS]:
        rel = evaluate([(exts.get(f), lab) for f, lab in made.items()]).summary()["relaxed"]
        out["macro_f1_model_made"][tag] = round(sum(rel[f]["f1"] for f in FIELDS) / len(FIELDS), 3)  # type: ignore[index]
    out["bootstrap"] = {"resamples": n_boot, "seed": seed, "unit": "patient"}
    (REPO / "evals" / "results" / "paper_extras.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
