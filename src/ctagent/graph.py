"""Per-discharge LangGraph workflow with a fixed order of steps (ADR 0002).

ingest -> ehr_context -> score -> extract -> plan -> draft -> human_review -> track

State holds JSON-ready dicts only, so checkpoints never need custom deserialization. The LLM
writes text inside `plan` (explanation) and `draft` (call script, provider brief); everything else
is deterministic. `human_review` pauses with `interrupt` until a coordinator or provider decides.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, GraphOutput, interrupt
from pydantic import BaseModel

from .drafts import (
    SYSTEM,
    CallScript,
    PlanExplanation,
    ProviderBrief,
    grounding_violations,
    record_text,
)
from .llm import CallResult, StructuredLLM, call_with_retry, field_guide
from .models import DischargeEvent, FollowUpPlan, PatientContext, RiskScore
from .rules import DEFAULT_RULES, PlanRules, plan_follow_up
from .scoring import score_discharge
from .tracking import Tracker

Decision = Literal["approve", "edit", "reject"]


class State(TypedDict):
    event: dict[str, Any]
    note_text: NotRequired[str | None]
    context: NotRequired[dict[str, Any]]
    score: NotRequired[dict[str, Any]]
    extraction: NotRequired[dict[str, Any] | None]
    plan: NotRequired[dict[str, Any]]
    explanation: NotRequired[str]
    call_script: NotRequired[dict[str, Any] | None]
    brief: NotRequired[dict[str, Any] | None]
    flags: Annotated[list[str], operator.add]
    llm_seconds: Annotated[float, operator.add]
    review: NotRequired[dict[str, Any]]
    status: NotRequired[str]


class ReviewDecision(BaseModel):
    decision: Decision
    reviewer: str
    call_script: CallScript | None = None  # replacement drafts when decision == "edit"
    brief: ProviderBrief | None = None
    note: str = ""


@dataclass
class Deps:
    llm: StructuredLLM
    contexts: Callable[[DischargeEvent], PatientContext]
    tracker: Tracker
    rules: PlanRules = DEFAULT_RULES
    extractor: Callable[[str], BaseModel] | None = None


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver[Any]) -> CompiledStateGraph[Any]:
    def event_of(s: State) -> DischargeEvent:
        return DischargeEvent.model_validate(s["event"])

    def ingest(s: State) -> dict[str, Any]:
        return {"event": event_of(s).model_dump(mode="json"), "status": "ingested"}

    def ehr_context(s: State) -> dict[str, Any]:
        return {"context": deps.contexts(event_of(s)).model_dump(mode="json")}

    def score(s: State) -> dict[str, Any]:
        ctx = PatientContext.model_validate(s["context"])
        risk = score_discharge(event_of(s), [c.code for c in ctx.conditions])
        return {"score": risk.model_dump(mode="json")}

    def extract(s: State) -> dict[str, Any]:
        note = s.get("note_text")
        if not note or deps.extractor is None:
            return {"extraction": None}
        try:
            return {"extraction": deps.extractor(note).model_dump(mode="json")}
        except Exception as e:  # extraction is optional context; a human sees the flag
            return {"extraction": None, "flags": [f"extraction failed: {type(e).__name__}"]}

    def plan(s: State) -> dict[str, Any]:
        ev, risk = event_of(s), RiskScore.model_validate(s["score"])
        p = plan_follow_up(ev.discharge.date(), risk, ev.disposition, deps.rules)
        prompt = _prompt(
            s, p, "Explain the follow-up tier and next steps for the coordinator.", PlanExplanation
        )
        r = call_with_retry(deps.llm, PlanExplanation, SYSTEM, prompt)
        out: dict[str, Any] = {"plan": p.model_dump(mode="json"), "llm_seconds": r.seconds}
        if r.value is None:
            out |= {
                "explanation": "; ".join(p.reasons),
                "flags": [f"plan explanation failed after {r.attempts} tries: {r.error}"],
            }
        else:
            out["explanation"] = r.value.summary
        return out

    def draft(s: State) -> dict[str, Any]:
        ev, p = event_of(s), FollowUpPlan.model_validate(s["plan"])
        ctx = PatientContext.model_validate(s["context"])
        flags: list[str] = []
        script = call_with_retry(
            deps.llm,
            CallScript,
            SYSTEM,
            _prompt(
                s,
                p,
                "Write the coordinator's phone call script for this patient, plain language.",
                CallScript,
            ),
        )
        brief = call_with_retry(
            deps.llm,
            ProviderBrief,
            SYSTEM,
            _prompt(
                s,
                p,
                "Write a one-page brief for the provider seeing this patient after discharge.",
                ProviderBrief,
            ),
        )
        results: list[tuple[str, CallResult[Any]]] = [
            ("call script", script),
            ("provider brief", brief),
        ]
        for name, r in results:
            if r.value is None:
                flags.append(f"{name} failed after {r.attempts} tries: {r.error}")
        flags += grounding_violations(ctx, ev.reason, script.value, brief.value)
        risk = RiskScore.model_validate(s["score"])
        deps.tracker.record_plan(
            ev.episode_id,
            ev.patient_id,
            ev.discharge.date(),
            s["plan"],
            risk.lace,
            s.get("flags", []) + flags,
        )
        return {
            "call_script": None if script.value is None else script.value.model_dump(mode="json"),
            "brief": None if brief.value is None else brief.value.model_dump(mode="json"),
            "flags": flags,
            "llm_seconds": script.seconds + brief.seconds,
            "status": "pending_review",
        }

    def human_review(s: State) -> dict[str, Any]:
        raw = interrupt(
            {
                "episode_id": s["event"]["episode_id"],
                "tier": s["plan"]["tier"],
                "lace": s["score"]["lace"],
                "explanation": s.get("explanation", ""),
                "flags": s.get("flags", []),
                "call_script": s.get("call_script"),
                "brief": s.get("brief"),
                "plan": s["plan"],
            }
        )
        d = ReviewDecision.model_validate(raw)
        out: dict[str, Any] = {"review": d.model_dump(mode="json"), "status": d.decision}
        if d.decision == "edit":
            if d.call_script is not None:
                out["call_script"] = d.call_script.model_dump(mode="json")
            if d.brief is not None:
                out["brief"] = d.brief.model_dump(mode="json")
        return out

    def track(s: State) -> dict[str, Any]:
        review = s["review"]
        check_ins = [str(x) for x in s["plan"]["check_ins"]]
        deps.tracker.record_review(
            s["event"]["episode_id"], review["decision"], review["reviewer"], check_ins
        )
        return {"status": "approved" if review["decision"] != "reject" else "rejected"}

    g = StateGraph(State)
    steps: list[tuple[str, Callable[[State], dict[str, Any]]]] = [
        ("ingest", ingest),
        ("ehr_context", ehr_context),
        ("score", score),
        ("extract", extract),
        ("plan", plan),
        ("draft", draft),
        ("human_review", human_review),
        ("track", track),
    ]
    for name, fn in steps:
        g.add_node(name, cast(Any, fn))  # LangGraph's node overloads don't accept a plain Callable
    for a, b in pairwise([START, *(n for n, _ in steps), END]):
        g.add_edge(a, b)
    return g.compile(checkpointer=checkpointer)


def _prompt(s: State, p: FollowUpPlan, task: str, schema: type[BaseModel]) -> str:
    ev = DischargeEvent.model_validate(s["event"])
    ctx = PatientContext.model_validate(s["context"])
    risk = RiskScore.model_validate(s["score"])
    extra = ""
    if s.get("extraction"):
        extra = f"\nFrom the discharge summary: {s['extraction']}"
    record = record_text(ev, ctx, risk, p)
    return f"{task}\n\n{field_guide(schema)}\n\nRECORD\n{record}{extra}"


def thread(episode_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": episode_id}}


def start_discharge(
    graph: CompiledStateGraph[Any], event: DischargeEvent, note_text: str | None = None
) -> GraphOutput[Any]:
    """Run a discharge up to human review. Returns the interrupt payload in `.interrupts`."""
    state: State = {
        "event": event.model_dump(mode="json"),
        "note_text": note_text,
        "flags": [],
        "llm_seconds": 0.0,
    }
    return graph.invoke(state, thread(event.episode_id), version="v2")


def resume_review(
    graph: CompiledStateGraph[Any], episode_id: str, decision: ReviewDecision
) -> GraphOutput[Any]:
    """Resume a paused discharge with the reviewer's decision (validated before resuming)."""
    payload = ReviewDecision.model_validate(decision.model_dump()).model_dump(mode="json")
    return graph.invoke(Command(resume=payload), thread(episode_id), version="v2")
