"""The export's Modelfile (no torch or Ollama needed): the base model's prompt format is copied."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_ollama.py"


def load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("export_ollama", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modelfile_copies_template_and_stop_tokens_only() -> None:
    template = "<|im_start|>user\n{{ .Content }}<|im_end|>\n"
    parameters = (
        'repeat_penalty                 1\nstop                           "<|im_start|>"\n'
        'stop                           "<|im_end|>"\ntemperature                    0.6'
    )
    text = load_script().modelfile(Path("C:/w/merged"), template, parameters)
    lines = text.splitlines()
    assert lines[0] == "FROM C:/w/merged"
    assert f'TEMPLATE """{template}"""' in text
    assert 'PARAMETER stop "<|im_start|>"' in lines and 'PARAMETER stop "<|im_end|>"' in lines
    # sampling settings are not copied: the extractor decodes greedily
    assert "PARAMETER temperature 0" in lines and "0.6" not in text
    assert "repeat_penalty" not in text
