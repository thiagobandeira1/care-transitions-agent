"""QLoRA fine-tuning of the discharge-summary extractor (M7). Runs locally on the GPU.

  (environment: UV_PROJECT_ENVIRONMENT=.venv-train uv sync --group dev --group train)
  .venv-train/Scripts/python scripts/train_qlora.py --source mimic --base Qwen/Qwen3-8B
  .venv-train/Scripts/python scripts/train_qlora.py --source synthetic --base Qwen/Qwen3-0.6B
  # option C: notes labeled by scripts/teacher_label.py; --labels silver is the control
  .venv-train/Scripts/python scripts/train_qlora.py --source mimic --teacher-dir D --labels teacher

Inputs: MIMIC parquet splits from `mimic-prep` (MIMIC_WORK_DIR, outside the repo), or the public
synthetic benchmark (pipeline test only). The adapter is written OUTSIDE the repo
(MIMIC_WORK_DIR/models by default); MIMIC-trained weights are never committed or uploaded.
Training uses prompt-completion pairs, so the loss covers only the JSON answer. The prompt matches
the zero-shot extractor exactly, so the fine-tuned model is served with the same prompt.
Prints aggregates only (loss, counts), never note text.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ctagent import teacher
from ctagent.benchmark import load as load_synthetic
from ctagent.extraction import DischargeExtraction
from ctagent.extractor import SYSTEM, Extractor
from ctagent.llm import FakeLLM
from ctagent.privacy import assert_outside_repo


def pair(note: str, label: str) -> dict[str, Any]:
    """One prompt-completion example; the prompt is the zero-shot extractor's, byte for byte."""
    DischargeExtraction.model_validate_json(label)  # every target must parse
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": Extractor(FakeLLM()).prompt(note)},
        ],
        "completion": [{"role": "assistant", "content": label}],
    }


def examples(
    source: str,
    split: str,
    limit: int,
    seed: int,
    teacher_dir: Path | None = None,
    labels: str = "silver",
) -> list[dict[str, Any]]:
    """With `teacher_dir`, only the notes the teacher labeled, in labeling order, so a smaller
    limit is a nested subset. `labels` 'teacher' merges teacher follow-up, pending tests and red
    flags into silver; 'silver' trains the same notes on silver alone, as a control."""
    if labels == "teacher" and teacher_dir is None:
        raise ValueError("teacher labels need --teacher-dir")
    if source == "synthetic":
        items = load_synthetic("train" if split == "train" else "test")
        pairs = [(x["note"], json.dumps(x["reference"])) for x in items]
    elif teacher_dir is None:
        import pandas as pd

        work = Path(os.environ["MIMIC_WORK_DIR"])
        df = pd.read_parquet(work / f"{split}.parquet").sample(frac=1.0, random_state=seed)
        # MIMIC inputs are already the extraction sections; the prompt builder re-sections them
        pairs = [(str(i), str(s)) for i, s in zip(df["input"], df["silver"], strict=True)]
    else:
        import pandas as pd

        work = Path(os.environ["MIMIC_WORK_DIR"])
        weak = teacher.load(teacher_dir / f"{split}.jsonl")
        df = pd.read_parquet(work / f"{split}.parquet").set_index("note_id").loc[list(weak)]
        pairs = []
        for note_id, text, silver in zip(df.index, df["input"], df["silver"], strict=True):
            if labels == "teacher":
                silver = teacher.merge(
                    DischargeExtraction.model_validate_json(silver), weak[note_id]
                ).model_dump_json()
            pairs.append((str(text), str(silver)))
    return [pair(note, label) for note, label in pairs[:limit]]


def fits(tok: Any, ex: dict[str, Any], max_len: int) -> bool:
    msgs = ex["prompt"] + ex["completion"]
    return (
        len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"]) <= max_len
    )


def fitting(tok: Any, items: list[dict[str, Any]], max_len: int) -> list[dict[str, Any]]:
    """Examples whose prompt and answer fit in max_len tokens. Longer ones are dropped, not
    truncated: truncation cuts the end of the JSON answer and trains on invalid JSON."""
    return [ex for ex in items if fits(tok, ex, max_len)]


def use_efficient_attention() -> None:
    """Repeat the key/value heads instead of passing enable_gqa to SDPA.

    Without flash attention (PyTorch on Windows), SDPA's grouped-query path falls back to the math
    kernel, which builds the full attention matrix: about 5 GB per layer at 3,072 tokens for
    Qwen3-8B, measured. With repeated heads the memory-efficient kernel runs, using about 0.2 GB.
    """
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mimic", "synthetic"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-train", type=int, default=10000)
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    # 16 GB GPU: a synthetic probe peaks at 10.6 GB at 3k tokens and 12.0 GB at 4k. Real runs need
    # ~2 GB more (CUDA context, fragmentation from varying lengths) and the Windows desktop ~1.3 GB,
    # so 4096 would spill into system RAM and crawl. 3072 keeps ~89% of MIMIC examples.
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--teacher-dir", type=Path, default=None, help="train on teacher-labeled notes")
    ap.add_argument("--labels", choices=["silver", "teacher"], default="silver")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    # On Windows a full GPU spills into system RAM instead of raising out-of-memory, and training
    # slows several-fold without an error. A cap makes PyTorch free its cache before reaching it.
    ap.add_argument("--gpu-fraction", type=float, default=0.85, help="cap on GPU memory use")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    use_efficient_attention()
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    work = Path(os.environ.get("MIMIC_WORK_DIR", Path.home() / "ctagent_work"))
    name = f"{args.base.split('/')[-1]}-extractor-{args.source}"
    if args.teacher_dir:
        name += f"-{args.labels}{args.max_train}"
    out = assert_outside_repo(args.out or work / "models" / name)
    tok = AutoTokenizer.from_pretrained(args.base)
    eval_split = "validation" if args.source == "mimic" else "test"
    raw = {
        s: examples(args.source, s, n, args.seed, args.teacher_dir, args.labels)
        for s, n in (("train", args.max_train), (eval_split, args.max_eval))
    }
    kept = {s: fitting(tok, items, args.max_len) for s, items in raw.items()}
    train, evals = Dataset.from_list(kept["train"]), Dataset.from_list(kept[eval_split])
    dropped = {s: len(raw[s]) - len(kept[s]) for s in raw}
    print(f"train examples {len(train)} | eval examples {len(evals)} | base {args.base}")
    print(f"dropped over {args.max_len} tokens: {dropped}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # the default of 8 would not fit at max_len
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_len,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        warmup_steps=0.03,  # transformers 5: a float below 1 is a ratio of total steps
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train,
        eval_dataset=evals,
        processing_class=tok,
        peft_config=lora,
    )
    run = trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(out / "adapter"))
    metrics = {**run.metrics, **trainer.evaluate()}
    metrics["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    metrics["train_examples"], metrics["dropped_over_max_len"] = len(train), dropped
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}))
    print(f"adapter saved outside the repo: {out / 'adapter'}")


if __name__ == "__main__":
    main()
