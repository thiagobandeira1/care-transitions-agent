"""Narrow LLM calls with schema validation: one retry, then the case is flagged (ADR 0002)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from types import NoneType, UnionType
from typing import Annotated, Any, Protocol, TypeVar, Union, get_args, get_origin

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
            last_error = f"{type(e).__name__}: {_shorten(str(e))}"
    return CallResult(None, retries + 1, time.perf_counter() - start, last_error)


def _shorten(msg: str, limit: int = 320) -> str:
    """Keep the start and the end: a parse error ends with the validator's reason."""
    return msg if len(msg) <= limit else f"{msg[:120]} ... {msg[-(limit - 125) :]}"


def field_guide(schema: type[BaseModel]) -> str:
    """The schema's fields and descriptions as prompt text.

    Ollama uses a JSON schema only to constrain decoding; the model never sees the field
    descriptions. Without this, guidance such as "2-3 sentences" never reached the model and
    plan explanations ran on until the token cap (ADR 0007).
    """
    return "Return JSON with these fields:\n" + "\n".join(_field_lines(schema, ""))


def _field_lines(model: type[BaseModel], indent: str) -> list[str]:
    lines = []
    for name, f in model.model_fields.items():
        kind, inner = _kind(f.annotation)
        desc = f": {f.description}" if f.description else ""
        lines.append(f"{indent}- {name} ({kind}){desc}")
        if inner is not None:
            lines += _field_lines(inner, indent + "  ")
    return lines


def _kind(tp: Any) -> tuple[str, type[BaseModel] | None]:
    """A plain-language type name, plus the nested model to describe under it."""
    origin, args = get_origin(tp), get_args(tp)
    if origin is Annotated:
        return _kind(args[0])
    if origin in (Union, UnionType):
        kind, inner = _kind(next(a for a in args if a is not NoneType))
        return (f"{kind} or null" if NoneType in args else kind), inner
    if origin is list:
        kind, inner = _kind(args[0])
        return ("list of objects" if inner else f"list of {kind}"), inner
    if isinstance(tp, type) and issubclass(tp, Enum):
        return "one of: " + ", ".join(str(m.value) for m in tp), None
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return "object", tp
    names = {str: "text", int: "integer", float: "number", bool: "true or false"}
    return names.get(tp, "value"), None


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
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        num_predict: int = 1500,
    ) -> None:
        assert_no_cloud_tracing()
        assert_local_model(model, base_url)
        from langchain_ollama import ChatOllama

        self.name = f"{model}{' (cpu)' if cpu_only else ''}"
        self._model, self._base_url = model, base_url
        # with a seed, call k uses seed + k: a run is repeatable, and a retry samples a new answer
        self._seed, self._calls = seed, 0
        # num_predict caps runaway generations (a model stuck repeating JSON would otherwise run
        # until the context fills); a capped output fails validation, retries once, then flags.
        self._chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            reasoning=False,
            top_p=top_p,
            top_k=top_k,
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

    def _next_chat(self) -> Any:
        if self._seed is None:
            return self._chat
        self._calls += 1
        return self._chat.model_copy(update={"seed": self._seed + self._calls})

    def complete(self, system: str, prompt: str) -> str:
        """Free text (used only to write synthetic notes)."""
        return str(self._next_chat().invoke([("system", system), ("human", prompt)]).content)

    def generate(self, schema: type[T], system: str, prompt: str) -> T:
        runnable = self._next_chat().with_structured_output(schema, method="json_schema")
        try:
            out = runnable.invoke([("system", system), ("human", prompt)])
        except (ValidationError, ValueError) as e:
            raise LLMOutputError(str(e)) from e
        if not isinstance(out, schema):
            raise LLMOutputError(f"expected {schema.__name__}, got {type(out).__name__}")
        return out


def drafting_llm(model: str, *, cpu_only: bool = False, seed: int = 7) -> OllamaLLM:
    """Free-text drafting with Qwen3's non-thinking sampling (ADR 0007).

    The Qwen3 model card warns against greedy decoding ("endless repetitions"); at temperature 0
    most plan explanations looped until the token cap. Extraction stays greedy (short JSON).
    """
    return OllamaLLM(model, cpu_only=cpu_only, temperature=0.7, top_p=0.8, top_k=20, seed=seed)


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
