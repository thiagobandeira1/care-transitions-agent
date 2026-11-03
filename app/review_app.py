"""Coordinator review screen (Streamlit, local only). Synthea patients in the public demo.

uv run python -m ctagent demo --n 10                     # morning batch into data/demo
uv run streamlit run app/review_app.py -- --workspace data/demo
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ctagent.drafts import CallScript, ProviderBrief
from ctagent.graph import ReviewDecision
from ctagent.llm import drafting_llm
from ctagent.models import DischargeEvent, PatientContext
from ctagent.review import (
    Workspace,
    decide,
    draft_from_form,
    draft_to_form,
    follow_ups,
    open_graph,
    outcome_fields,
    pending,
    review_payload,
)
from ctagent.tracking import Tracker


def _no_new_discharges(event: DischargeEvent) -> PatientContext:
    raise RuntimeError("the review screen only reviews; run `python -m ctagent demo` to process")


@st.cache_resource
def load(workspace: str, model: str) -> tuple[CompiledStateGraph[Any], Tracker]:
    return open_graph(Workspace(Path(workspace)), drafting_llm(model), _no_new_discharges)


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
    started = st.session_state.setdefault(f"t0-{ep}", time.time())  # review time, for the pilot
    plan = payload["plan"]
    tier = "🔴 HIGH" if payload["tier"] == "high" else "🟢 standard"
    st.subheader(f"{tier} · LACE {payload['lace']}")
    st.write(payload["explanation"])
    st.markdown(
        f"TCM contact by **{plan['contact_by']}** · visit by **{plan['visit_by']}** · "
        f"face-to-face window if billed as {plan['tcm_code']}: **{plan['tcm_visit_by']}** · "
        f"check-ins: {', '.join(map(str, plan['check_ins'])) or 'none'}"
    )
    st.caption(
        f"{plan['tcm_code']} is a candidate only: the code billed depends on the provider's "
        "documented medical decision making and the other TCM requirements."
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
    forms: dict[str, dict[str, str]] = {}
    with st.expander("Edit drafts"):
        st.caption(
            "One item per line. Medications: name | started, stopped, dose_changed or continued "
            "| note. Then click Approve with edits."
        )
        for col, (doc, title) in zip(
            st.columns(2),
            (("call_script", "Call script"), ("brief", "Provider brief")),
            strict=True,
        ):
            col.markdown(f"**{title}**")
            forms[doc] = {
                k: col.text_area(
                    k.replace("_", " ").capitalize(),
                    v,
                    height=68 if "\n" not in v and len(v) < 80 else 140,
                    key=f"{doc}-{ep}-{k}",
                )
                for k, v in draft_to_form(payload[doc]).items()
            }
    real = None
    if payload["flags"]:
        real = st.radio(
            "Were these flags real problems?", ["yes", "some", "no"], index=None, horizontal=True
        )
    note = st.text_input("Note (required to reject)")
    a, b, c = st.columns(3)
    decision: ReviewDecision | None = None
    common: dict[str, Any] = {"reviewer": reviewer, "flags_real": real}
    if a.button("✅ Approve", disabled=not reviewer):
        decision = ReviewDecision(decision="approve", **common)
    if b.button("✏️ Approve with edits", disabled=not reviewer):
        try:
            decision = ReviewDecision(
                decision="edit",
                call_script=draft_from_form(CallScript, forms["call_script"])
                if payload["call_script"]
                else None,
                brief=draft_from_form(ProviderBrief, forms["brief"]) if payload["brief"] else None,
                **common,
            )
        except ValidationError as e:
            st.error(f"Not saved: {e.error_count()} problem(s) in the edits.")
            for err in e.errors():
                st.write(f"- {' → '.join(map(str, err['loc']))}: {err['msg']}")
    if c.button("❌ Reject", disabled=not (reviewer and note)):
        decision = ReviewDecision(decision="reject", note=note, **common)
    if decision is not None:
        decision.seconds = round(time.time() - started, 1)
        st.success(f"Saved: {decide(graph, ep, decision)}")
        st.rerun()


def _day(value: object) -> date | None:
    return date.fromisoformat(str(value)[:10]) if value else None


def follow_up_tab(tracker: Tracker) -> None:
    rows = follow_ups(tracker)
    if not rows:
        st.info("No approved discharges yet.")
        return
    st.dataframe(rows, hide_index=True)
    ep = st.selectbox("Discharge", [r["episode_id"] for r in rows], key="follow-up")
    now = tracker.outcome(ep)
    readmit = {1: "yes", 0: "no"}.get(now.get("readmitted"), "not known")
    with st.form(f"outcome-{ep}"):
        a, b, c = st.columns(3)
        contacted = a.date_input("Interactive contact on", _day(now.get("contacted_at")))
        attempts = b.number_input("Contact attempts", 0, value=now.get("contact_attempts"))
        minutes = c.number_input("Minutes on the phone", 0, value=now.get("contact_minutes"))
        d, e, f = st.columns(3)
        scheduled = d.date_input("Visit scheduled for", _day(now.get("visit_scheduled")))
        attended = e.date_input("Visit attended on", _day(now.get("visit_attended")))
        options = ["not known", "no", "yes"]
        readmitted = f.selectbox("Readmitted within 30 days", options, options.index(readmit))
        if st.form_submit_button("Save"):
            tracker.record_outcome(
                ep,
                **outcome_fields(contacted, attempts, minutes, scheduled, attended, readmitted),
            )
            st.success("Saved.")
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
    queue_tab, followup_tab, checkin_tab = st.tabs(["Review queue", "Follow-up", "Check-ins due"])
    with queue_tab:
        review_tab(graph, tracker, reviewer)
    with followup_tab:
        follow_up_tab(tracker)
    with checkin_tab:
        check_in_tab(tracker)


main()
