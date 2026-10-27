"""Merge a trained LoRA adapter into its base model and import it into local Ollama (M8).

  .venv-train/Scripts/python scripts/export_ollama.py --base Qwen/Qwen3-8B --name ctagent-extractor
      --adapter <MIMIC_WORK_DIR>/models/Qwen3-8B-extractor-mimic/adapter

Writes the merged weights and a Modelfile next to the adapter (OUTSIDE the repo), then runs
`ollama create <name> --quantize q4_K_M`. The result is a local model tag used by the agent's
`extract` node with the same zero-shot prompt it was trained on. MIMIC-trained weights stay on this
machine (never pushed to any registry).

The Modelfile copies the prompt template and stop tokens of the matching library model
(`--format-from qwen3:8b`), so the fine-tuned model sees exactly the prompt format of the zero- and
few-shot baselines (thinking switched off the same way) and the comparison differs only in the
weights. Ollama's own guess from the tokenizer's chat template could differ.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from ctagent.privacy import assert_outside_repo


def modelfile(merged: Path, template: str, parameters: str) -> str:
    """Modelfile for the merged weights, with the reference model's template and stop tokens."""
    stops = [line.split(None, 1)[1] for line in parameters.splitlines() if line.startswith("stop")]
    lines = [f"FROM {merged.as_posix()}", f'TEMPLATE """{template}"""']
    lines += [f"PARAMETER stop {s}" for s in stops]
    lines += ["PARAMETER temperature 0", "PARAMETER num_ctx 16384"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--name", default="ctagent-extractor")
    ap.add_argument("--quantize", default="q4_K_M")
    ap.add_argument("--format-from", default="qwen3:8b", help="local model whose template to copy")
    args = ap.parse_args()

    import ollama
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = assert_outside_repo(args.adapter)
    merged = adapter.parent / "merged"
    ref = ollama.Client(host="http://localhost:11434").show(args.format_from)
    if not ref.template or "<|im_start|>" not in ref.template:
        raise SystemExit(f"{args.format_from} has no Qwen chat template to copy")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(merged))
    path = adapter.parent / "Modelfile"
    path.write_text(modelfile(merged, ref.template, ref.parameters or ""), encoding="utf-8")

    exe = shutil.which("ollama") or str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")
    subprocess.run(  # noqa: S603 - fixed local ollama binary, arguments from this script's CLI
        [exe, "create", args.name, "--quantize", args.quantize, "-f", str(path)], check=True
    )
    print(f"created local Ollama model {args.name!r} from {merged}")


if __name__ == "__main__":
    main()
