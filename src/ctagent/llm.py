"""Narrow LLM calls with schema validation: one retry, then the case is flagged (ADR 0002)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .privacy import assert_local_model, assert_no_cloud_tracing

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The model's output could not be parsed into the requested schema."""


class StructuredLLM(Protocol):
    name: str

    def generate(self, schema: type[T], system: str, prompt: str) -> T: ...


@dataclass
class CallResult[R: BaseModel]:
    value: R | None
    attempts: int
    seconds: float
    error: str = ""


def call_with_retry[R: BaseModel](
    llm: StructuredLLM, schema: type[R], system: str, prompt: str, retries: int = 1
) -> CallResult[R]:
    """Validated structured call. On a second failure the caller flags the case for a human."""
    start, last_error = time.perf_counter(), ""
    for attempt in range(1, retries + 2):
        try:
            value = llm.generate(schema, system, prompt)
            return CallResult(
                schema.model_validate(value.model_dump()), attempt, time.perf_counter() - start
            )
        except (LLMOutputError, ValidationError, ValueError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
    return CallResult(None, retries + 1, time.perf_counter() - start, last_error)


class OllamaLLM:
    """Local Ollama model via langchain-ollama, JSON-schema constrained, thinking off."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        cpu_only: bool = False,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        num_predict: int = 1500,
    ) -> None:
        assert_no_cloud_tracing()
        assert_local_model(model, base_url)
        from langchain_ollama import ChatOllama

        self.name = f"{model}{' (cpu)' if cpu_only else ''}"
        self._model, self._base_url = model, base_url
        # num_predict caps runaway generations (a model stuck repeating JSON would otherwise run
        # until the context fills); a capped output fails validation, retries once, then flags.
        self._chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            reasoning=False,
            num_gpu=0 if cpu_only else None,
            keep_alive="10m",
        )

    def unload(self) -> None:
        """Drop the model from Ollama's memory now.

        Ollama reuses a loaded runner for later requests that don't set num_gpu, so a model
        left loaded by a CPU-only run would make the next GPU job run on CPU.
        """
        import ollama

        ollama.Client(host=self._base_url).generate(model=self._model, keep_alive=0)

    def complete(self, system: str, prompt: str) -> str:
        """Free text (used only to write synthetic notes)."""
        return str(self._chat.invoke([("system", system), ("human", prompt)]).content)

    def generate(self, schema: type[T], system: str, prompt: str) -> T:
        runnable = self._chat.with_structured_output(schema, method="json_schema")
        try:
            out = runnable.invoke([("system", system), ("human", prompt)])
        except (ValidationError, ValueError) as e:
            raise LLMOutputError(str(e)) from e
        if not isinstance(out, schema):
            raise LLMOutputError(f"expected {schema.__name__}, got {type(out).__name__}")
        return out


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests and keyless CI.

    `responses` maps a schema name to a factory returning a model or an exception to raise.
    """

    responses: dict[str, Callable[[str], BaseModel | Exception]] = field(default_factory=dict)
    name: str = "fake"
    calls: list[tuple[str, str]] = field(default_factory=list)
    unloaded: bool = False

    def unload(self) -> None:
        self.unloaded = True

    def generate(self, schema: type[T], system: str, prompt: str) -> T:
        self.calls.append((schema.__name__, prompt))
        out: Any = self.responses[schema.__name__](prompt)
        if isinstance(out, Exception):
            raise out
        return schema.model_validate(out.model_dump())
