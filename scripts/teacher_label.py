"""Teacher labels for follow-up, pending tests and red flags (option C). Runs locally on the GPU.

  .venv-train/Scripts/python scripts/teacher_label.py --n-train 2500 --n-val 300

qwen3:14b with the M6 few-shot prompt (3 train examples, seed 5) labels notes in the training
script's shuffled order (seed 3), so the train notes are a subset of the first fine-tune's. Notes
whose training example would not fit in --max-len tokens are skipped before labeling, so no GPU
time goes to notes the trainer would drop; --margin leaves room for the teacher's longer answers.
The few-shot example notes are skipped too. Rerun the same command to resume after a stop.
Writes MIMIC_WORK_DIR/teacher/{validation,train}.jsonl, outside the repo; prints counts only.
"""

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from train_qlora import fits, pair

from ctagent import teacher
from ctagent.extractor import Extractor
from ctagent.llm import OllamaLLM
from ctagent.mimic_bench import NUM_PREDICT, pick_shots
from ctagent.privacy import assert_outside_repo


def candidates(
    work: Path, split: str, seed: int, tok: Any, max_len: int, skip: set[str]
) -> Iterator[tuple[str, str]]:
    """(note_id, input) in the training script's order, for notes whose silver example fits."""
    import pandas as pd

    df = pd.read_parquet(work / f"{split}.parquet", columns=["note_id", "input", "silver"])
    df = df.sample(frac=1.0, random_state=seed)
    for note_id, text, silver in zip(df["note_id"], df["input"], df["silver"], strict=True):
        if text not in skip and fits(tok, pair(str(text), str(silver)), max_len):
            yield str(note_id), str(text)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="the student's tokenizer")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--margin", type=int, default=192)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from transformers import AutoTokenizer

    work = Path(os.environ["MIMIC_WORK_DIR"])
    out = assert_outside_repo(args.out or work / "teacher")
    tok = AutoTokenizer.from_pretrained(args.base)
    shots = pick_shots(work, 3, 5)
    ext = Extractor(OllamaLLM(args.model, num_ctx=16384, num_predict=NUM_PREDICT), shots)
    skip = {ex.note for ex in shots}
    for split, n in (("validation", args.n_val), ("train", args.n_train)):
        notes = candidates(work, split, args.seed, tok, args.max_len - args.margin, skip)
        res = teacher.label(notes, ext, out / f"{split}.jsonl", n)
        print(json.dumps({"split": split, "teacher": args.model, **res}, indent=2), flush=True)


if __name__ == "__main__":
    main()
