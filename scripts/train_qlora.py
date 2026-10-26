"""QLoRA fine-tuning of the discharge-summary extractor (M7). Runs locally on the GPU.

  (environment: UV_PROJECT_ENVIRONMENT=.venv-train uv sync --group dev --group train)
  .venv-train/Scripts/python scripts/train_qlora.py --source mimic --base Qwen/Qwen3-8B
  .venv-train/Scripts/python scripts/train_qlora.py --source synthetic --base Qwen/Qwen3-0.6B

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

from dotenv import load_dotenv

from ctagent.benchmark import load as load_synthetic
from ctagent.extraction import DischargeExtraction
from ctagent.extractor import SYSTEM, Extractor
from ctagent.llm import FakeLLM
from ctagent.privacy import assert_outside_repo


def examples(source: str, split: str, limit: int, seed: int) -> list[dict[str, object]]:
    prompter = Extractor(FakeLLM())  # zero-shot prompt builder only; no model is called
    if source == "synthetic":
        items = load_synthetic("train" if split == "train" else "test")
        pairs = [(x["note"], json.dumps(x["reference"])) for x in items]
    else:
        import pandas as pd

        work = Path(os.environ["MIMIC_WORK_DIR"])
        df = pd.read_parquet(work / f"{split}.parquet").sample(frac=1.0, random_state=seed)
        # MIMIC inputs are already the extraction sections; the prompt builder re-sections them
        pairs = [(str(i), str(s)) for i, s in zip(df["input"], df["silver"], strict=True)]
    out = []
    for note, label in pairs[:limit]:
        DischargeExtraction.model_validate_json(label)  # every target must parse
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompter.prompt(note)},
                ],
                "completion": [{"role": "assistant", "content": label}],
            }
        )
    return out


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
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    work = Path(os.environ.get("MIMIC_WORK_DIR", Path.home() / "ctagent_work"))
    name = f"{args.base.split('/')[-1]}-extractor-{args.source}"
    out = assert_outside_repo(args.out or work / "models" / name)
    train = Dataset.from_list(examples(args.source, "train", args.max_train, args.seed))
    eval_split = "validation" if args.source == "mimic" else "test"
    evals = Dataset.from_list(examples(args.source, eval_split, args.max_eval, args.seed))
    print(f"train examples {len(train)} | eval examples {len(evals)} | base {args.base}")

    tok = AutoTokenizer.from_pretrained(args.base)
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
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    metrics = trainer.evaluate()
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}))
    print(f"adapter saved outside the repo: {out / 'adapter'}")


if __name__ == "__main__":
    main()
