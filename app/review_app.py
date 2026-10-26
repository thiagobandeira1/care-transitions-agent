"""Coordinator review screen (Streamlit, local only). Synthea patients in the public demo.

uv run python -m ctagent demo --n 10                     # morning batch into data/demo
uv run streamlit run app/review_app.py -- --workspace data/demo
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ctagent.drafts import CallScript, ProviderBrief
from ctagent.graph import ReviewDecision
from ctagent.llm import OllamaLLM
from ctagent.models import DischargeEvent, PatientContext
from ctagent.review import Workspace, decide, open_graph, pending, review_payload
from ctagent.tracking import Tracker


def _no_new_discharges(event: DischargeEvent) -> PatientContext:
    raise RuntimeError("the review screen only reviews; run `python -m ctagent demo` to process")


@st.cache_resource
def load(workspace: str, model: str) -> tuple[CompiledStateGraph[Any], Tracker]:
    return open_graph(Workspace(Path(workspace)), OllamaLLM(model), _no_new_discharges)


def show_script(s: dict[str, Any]) -> None:
    st.markdown(f"**Opening.** {s['opening']}")
    st.markdown("**Questions**\n" + "\n".join(f"- {q}" for q in s["questions"]))
    if s["medication_review"]:
        st.markdown(
            "**Medication review**\n"
            + "\n".join(
                f"- {m['name']} ({m['change']}): {m['note']}" for m in s["medication_review"]
            )
        )
    st.markdown("**Warning signs**\n" + "\n".join(f"- {w}" for w in s["warning_signs"]))
    st.markdown(f"**Scheduling.** {s['scheduling']}")


def show_brief(b: dict[str, Any]) -> None:
    st.markdown(f"**Reason for admission.** {b['reason_for_admission']}")
    st.markdown("**Key conditions:** " + "; ".join(b["key_conditions"]))
    if b["medication_changes"]:
        st.markdown(
            "**Medication changes**\n"
            + "\n".join(
                f"- {m['name']} ({m['change']}): {m['note']}" for m in b["medication_changes"]
            )
        )
    st.markdown("**Watch for**\n" + "\n".join(f"- {w}" for w in b["watch_for"]))
    st.markdown(f"**Risk.** {b['risk_summary']}")


def review_tab(graph: CompiledStateGraph[Any], tracker: Tracker, reviewer: str) -> None:
    queue = pending(tracker)
    if not queue:
        st.success("Nothing waiting for review.")
        return
    st.dataframe([{**q, "flags": len(json.loads(q["flags"]))} for q in queue], hide_index=True)
    ep = st.selectbox("Discharge", [q["episode_id"] for q in queue])
    payload = review_payload(graph, ep)
    if payload is None:
        st.warning("This discharge is not waiting for review.")
        return
    plan = payload["plan"]
    tier = "🔴 HIGH" if payload["tier"] == "high" else "🟢 standard"
    st.subheader(f"{tier} · LACE {payload['lace']}")
    st.write(payload["explanation"])
    st.markdown(
        f"TCM contact by **{plan['contact_by']}** · visit by **{plan['visit_by']}** · "
        f"{plan['tcm_code']} face-to-face by **{plan['tcm_visit_by']}** · check-ins: "
        f"{', '.join(map(str, plan['check_ins'])) or 'none'}"
    )
    for f in payload["flags"]:
        st.warning(f"Check: {f}")
    left, right = st.columns(2)
    with left:
        st.markdown("### Call script")
        if payload["call_script"]:
            show_script(payload["call_script"])
    with right:
        st.markdown("### Provider brief")
        if payload["brief"]:
            show_brief(payload["brief"])
    with st.expander("Edit drafts (JSON)"):
        script_txt = st.text_area(
            "Call script", json.dumps(payload["call_script"], indent=2), height=240
        )
        brief_txt = st.text_area(
            "Provider brief", json.dumps(payload["brief"], indent=2), height=240
        )
    note = st.text_input("Note (required to reject)")
    a, b, c = st.columns(3)
    decision: ReviewDecision | None = None
    if a.button("✅ Approve", disabled=not reviewer):
        decision = ReviewDecision(decision="approve", reviewer=reviewer)
    if b.button("✏️ Approve with edits", disabled=not reviewer):
        try:
            decision = ReviewDecision(
                decision="edit",
                reviewer=reviewer,
                call_script=CallScript.model_validate_json(script_txt),
                brief=ProviderBrief.model_validate_json(brief_txt),
            )
        except ValidationError as e:
            st.error(f"Edited JSON doesn't match the schema: {e.error_count()} problem(s)")
    if c.button("❌ Reject", disabled=not (reviewer and note)):
        decision = ReviewDecision(decision="reject", reviewer=reviewer, note=note)
    if decision is not None:
        st.success(f"Saved: {decide(graph, ep, decision)}")
        st.rerun()


def check_in_tab(tracker: Tracker) -> None:
    day = st.date_input("Due on or before", date.today())
    due = tracker.due_check_ins(day)
    if not due:
        st.success("No check-ins due.")
    for ep, due_date in due:
        with st.container(border=True):
            st.write(f"{ep}: due {due_date}")
            outcome = st.text_input("Outcome", key=f"o-{ep}-{due_date}")
            if st.button("Mark done", key=f"b-{ep}-{due_date}", disabled=not outcome):
                tracker.complete_check_in(ep, due_date, outcome)
                st.rerun()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="data/demo")
    p.add_argument("--model", default="qwen3:14b")
    args = p.parse_args(sys.argv[1:])
    st.set_page_config(page_title="Discharge follow-up", layout="wide")
    st.title("Hospital follow-up · review queue")
    st.caption(
        "Drafts are suggestions from a local model. Nothing reaches a patient until you approve."
    )
    graph, tracker = load(args.workspace, args.model)
    reviewer = st.sidebar.text_input("Your name (recorded with each decision)")
    st.sidebar.write(tracker.summary())
    queue_tab, checkin_tab = st.tabs(["Review queue", "Check-ins due"])
    with queue_tab:
        review_tab(graph, tracker, reviewer)
    with checkin_tab:
        check_in_tab(tracker)


main()
