"""OllamaLLM and the benchmark, with the chat model stubbed (no Ollama or GPU in CI)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import ctagent.bench as bench
import ctagent.llm as llm_mod
from ctagent.drafts import CallScript, PlanExplanation, ProviderBrief
from ctagent.llm import FakeLLM, LLMOutputError, OllamaLLM
from ctagent.privacy import PrivacyError
from fixtures import SyntheaFixture
from test_graph import good


class StubChat:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.result: Any = PlanExplanation(summary="ok")

    def with_structured_output(self, schema: type[BaseModel], method: str) -> "StubChat":
        assert method == "json_schema"
        return self

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", StubChat)


@pytest.mark.usefixtures("stub")
def test_ollama_llm_is_local_and_thinking_off() -> None:
    m = OllamaLLM("qwen3:8b", cpu_only=True)
    kw = m._chat.kwargs  # type: ignore[attr-defined]
    assert kw["reasoning"] is False and kw["num_gpu"] == 0 and m.name == "qwen3:8b (cpu)"
    assert m.generate(PlanExplanation, "sys", "prompt").summary == "ok"
    with pytest.raises(PrivacyError):
        OllamaLLM("gpt-oss:120b-cloud")


@pytest.mark.usefixtures("stub")
def test_unload_asks_ollama_to_drop_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import ollama

    seen: list[tuple[str, dict[str, Any]]] = []

    class StubClient:
        def __init__(self, host: str) -> None:
            seen.append(("host", {"host": host}))

        def generate(self, **kwargs: Any) -> None:
            seen.append(("generate", kwargs))

    monkeypatch.setattr(ollama, "Client", StubClient)
    OllamaLLM("qwen3:14b", cpu_only=True).unload()
    assert seen == [
        ("host", {"host": "http://localhost:11434"}),
        ("generate", {"model": "qwen3:14b", "keep_alive": 0}),
    ]


@pytest.mark.usefixtures("stub")
def test_ollama_llm_wraps_parse_failures() -> None:
    m = OllamaLLM("qwen3:8b")
    try:
        PlanExplanation.model_validate({})
    except ValidationError as e:
        m._chat.result = e  # type: ignore[attr-defined]
    with pytest.raises(LLMOutputError):
        m.generate(PlanExplanation, "s", "p")
    m._chat.result = CallScript(  # type: ignore[attr-defined]
        opening="", questions=[], medication_review=[], warning_signs=[], scheduling=""
    )
    with pytest.raises(LLMOutputError, match="expected PlanExplanation"):
        m.generate(PlanExplanation, "s", "p")


def test_bench_writes_aggregates_and_synthetic_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = SyntheaFixture().patient("A").patient("B").patient("ZZ")
    fx.encounter("A", "inpatient", 3000, 3004)
    fx.encounter("B", "inpatient", 3000, 3002)
    fx.encounter("ZZ", "ambulatory", 3400, 3400.1)
    responses = good()
    responses["ProviderBrief"] = lambda p: ProviderBrief(
        reason_for_admission="x",
        key_conditions=[],
        medication_changes=[],
        watch_for=[],
        risk_summary="",
    )
    fake = FakeLLM(responses, name="qwen3:8b")
    monkeypatch.setattr(bench, "OllamaLLM", lambda model, cpu_only: fake)
    monkeypatch.setattr(bench, "REPO", tmp_path)
    monkeypatch.setattr(llm_mod, "OllamaLLM", None)
    result = bench.run_bench(fx.write(tmp_path / "csv"), "qwen3:8b", n=2, cpu_only=True, samples=1)
    assert result["n"] == 2 and result["device"] == "cpu"
    assert fake.unloaded  # a CPU-only runner must not linger for the next GPU job to reuse
    assert result["discharges_with_schema_failure"] == 0
    saved = json.loads((tmp_path / "evals" / "results" / "drafting_qwen3-8b_cpu.json").read_text())
    assert saved["seconds_per_discharge"]["mean"] >= 0
    samples = json.loads(
        (tmp_path / "synthetic" / "samples" / "drafts_qwen3-8b_cpu.json").read_text()
    )
    assert len(samples) == 1 and "call_script" in samples[0]
