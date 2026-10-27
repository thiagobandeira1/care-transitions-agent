"""Merge a trained LoRA adapter into its base model, quantize it to GGUF, and import it into local
Ollama (M8).

  .venv-train/Scripts/python scripts/export_ollama.py --base Qwen/Qwen3-8B --name ctagent-extractor
      --adapter <MIMIC_WORK_DIR>/models/Qwen3-8B-extractor-mimic/adapter

Steps, all next to the adapter (OUTSIDE the repo): merge into the bf16 base on the CPU, convert to a
bf16 GGUF with llama.cpp's `convert_hf_to_gguf.py`, quantize to Q4_K_M with `llama-quantize`, then
`ollama create <name>` from the GGUF. Q4_K_M is the quantization of the qwen3:8b library model, so
the zero- and few-shot baselines and the fine-tuned model differ only in the weights. (Ollama 0.40
no longer quantizes safetensors to Q4_K_M or loads LoRA adapters, hence llama.cpp.) MIMIC-trained
weights stay on this machine (never pushed to any registry).

llama.cpp setup (once): `git clone --depth 1 https://github.com/ggml-org/llama.cpp`, a venv in the
checkout with `requirements/requirements-convert_hf_to_gguf.txt`, and the official CPU binaries
from a llama.cpp release for `llama-quantize`.

The Modelfile copies the prompt template and stop tokens of the matching library model
(`--format-from qwen3:8b`), so the fine-tuned model sees exactly the prompt format of the
baselines (thinking switched off the same way).
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from ctagent.privacy import assert_outside_repo

TOOLS = Path.home() / "tools"


def modelfile(model: Path, template: str, parameters: str) -> str:
    """Modelfile for the model file, with the reference model's template and stop tokens."""
    stops = [line.split(None, 1)[1] for line in parameters.splitlines() if line.startswith("stop")]
    lines = [f"FROM {model.as_posix()}", f'TEMPLATE """{template}"""']
    lines += [f"PARAMETER stop {s}" for s in stops]
    lines += ["PARAMETER temperature 0", "PARAMETER num_ctx 16384"]
    return "\n".join(lines) + "\n"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)  # noqa: S603 - local tools, arguments from this script's CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--name", default="ctagent-extractor")
    ap.add_argument("--quantize", default="Q4_K_M")
    ap.add_argument("--format-from", default="qwen3:8b", help="local model whose template to copy")
    ap.add_argument("--llama-cpp", type=Path, default=TOOLS / "llama.cpp", help="checkout")
    ap.add_argument("--convert-python", type=Path, help="python with the converter's requirements")
    ap.add_argument("--quantize-bin", type=Path, help="llama-quantize executable")
    args = ap.parse_args()

    import ollama

    adapter = assert_outside_repo(args.adapter)
    out = adapter.parent
    merged = out / "merged"
    ref = ollama.Client(host="http://localhost:11434").show(args.format_from)
    if not ref.template or "<|im_start|>" not in ref.template:
        raise SystemExit(f"{args.format_from} has no Qwen chat template to copy")
    convert_py = args.convert_python or args.llama_cpp / ".venv-convert" / "Scripts" / "python.exe"
    quantize = args.quantize_bin or next(TOOLS.glob("llama-bin-*/bin/llama-quantize.exe"), None)
    if quantize is None or not Path(quantize).exists() or not convert_py.exists():
        raise SystemExit("llama.cpp tools not found; see the setup notes in this script")

    if not (merged / "model.safetensors").exists() and not any(merged.glob("*.safetensors")):
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        bf = torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=bf, device_map="cpu")
        model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
        model.save_pretrained(str(merged), safe_serialization=True)
        # The fine-tune leaves the tokenizer unchanged. Copy the base model's own files: files
        # re-saved by transformers 5 can't be read by the transformers 4 that the converter pins.
        files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        src = Path(args.base)
        if not src.is_dir():
            src = Path(snapshot_download(args.base, allow_patterns=files))
        for f in files:
            shutil.copy(src / f, merged / f)

    bf16 = out / "extractor-bf16.gguf"
    quantized = out / f"extractor-{args.quantize}.gguf"
    if not bf16.exists():
        convert = str(args.llama_cpp / "convert_hf_to_gguf.py")
        run([str(convert_py), convert, str(merged), "--outtype", "bf16", "--outfile", str(bf16)])
    run([str(quantize), str(bf16), str(quantized), args.quantize])
    path = out / "Modelfile"
    path.write_text(modelfile(quantized, ref.template, ref.parameters or ""), encoding="utf-8")

    exe = shutil.which("ollama") or str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")
    run([exe, "create", args.name, "-f", str(path)])
    print(f"created local Ollama model {args.name!r} from {quantized}")


if __name__ == "__main__":
    main()
