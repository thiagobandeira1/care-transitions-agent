"""Merge a trained LoRA adapter into its base model and import it into local Ollama (M8).

  .venv-train/Scripts/python scripts/export_ollama.py --base Qwen/Qwen3-8B --name ctagent-extractor
      --adapter <MIMIC_WORK_DIR>/models/Qwen3-8B-extractor-mimic/adapter

Writes the merged weights and a Modelfile next to the adapter (OUTSIDE the repo), then runs
`ollama create <name> --quantize q4_K_M`. The result is a local model tag used by the agent's
`extract` node with the same zero-shot prompt it was trained on. MIMIC-trained weights stay on this
machine (never pushed to any registry).
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from ctagent.privacy import assert_outside_repo

MODELFILE = """FROM {merged}
PARAMETER temperature 0
PARAMETER num_ctx 16384
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--name", default="ctagent-extractor")
    ap.add_argument("--quantize", default="q4_K_M")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = assert_outside_repo(args.adapter)
    merged = adapter.parent / "merged"
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(merged))
    modelfile = adapter.parent / "Modelfile"
    modelfile.write_text(MODELFILE.format(merged=merged.as_posix()))

    ollama = shutil.which("ollama") or str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")
    subprocess.run(  # noqa: S603 - fixed local ollama binary, arguments from this script's CLI
        [ollama, "create", args.name, "--quantize", args.quantize, "-f", str(modelfile)], check=True
    )
    print(f"created local Ollama model {args.name!r} from {merged}")


if __name__ == "__main__":
    main()
