"""Gold-set annotation screen (Streamlit, local only). MIMIC notes never leave this PC.

uv run streamlit run app/annotate_app.py                         # the gold set in MIMIC_WORK_DIR
uv run python -m ctagent gold-demo --n 10 --overlap 3            # synthetic practice set
uv run streamlit run app/annotate_app.py -- --gold-dir data/gold-demo
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from ctagent.annotate import (
    SECOND_PASS_DAYS,
    TABLES,
    Note,
    from_form,
    next_todo,
    notes,
    progress,
    queue,
    save,
    second_pass_opens,
    start_labels,
    status,
    to_rows,
)
from ctagent.models import MedChangeType

ICONS = {"done": "✅", "invalid": "⚠️", "todo": "⬜", "waiting": "⏳", "not_started": "·"}
TITLES = {
    "secondary_diagnoses": "Secondary diagnoses",
    "medication_changes": "Medication changes",
    "follow_up": "Follow-up",
    "pending_tests": "Pending tests",
    "red_flags": "Red flags",
}
HELP = {
    "secondary_diagnoses": "Other diagnoses listed under Discharge Diagnosis. Not past history.",
    "medication_changes": "Every drug on either list: started (discharge list only), stopped "
    "(admission list only, or stated stopped), dose_changed, continued. Lowercase generic name.",
    "follow_up": "Each appointment or follow-up instruction: who, when, why. Not general advice.",
    "pending_tests": "Results explicitly pending at discharge. Not tests already resulted.",
    "red_flags": "Symptoms that should prompt a call or ER visit. Not general education.",
}
COLUMNS = {
    "text": "Text",
    "icd10": "ICD-10",
    "name": "Drug",
    "change": "Change",
    "detail": "Dose / detail",
    "with_whom": "With whom",
    "timeframe": "When",
    "reason": "Why",
}


def default_gold_dir() -> Path:
    load_dotenv()
    from ctagent.mimic import MimicPaths

    return MimicPaths.from_env().work_dir / "gold"


def column_config(field: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {c: st.column_config.TextColumn(COLUMNS[c]) for c in TABLES[field]}
    if field == "medication_changes":
        cfg["change"] = st.column_config.SelectboxColumn(
            "Change", options=[c.value for c in MedChangeType], required=True
        )
    return cfg


def show_note(note: Note) -> None:
    text = note.text()
    query = st.text_input("Find in note", key=f"find-{note.name}")
    if query:
        hits = [
            f"{i:>4}  {line}"
            for i, line in enumerate(text.splitlines(), 1)
            if query.casefold() in line.casefold()
        ]
        st.caption(f"{len(hits)} line(s) contain “{query}”")
        if hits:
            st.code("\n".join(hits[:40]), language=None, wrap_lines=True)
    with st.container(height=760, border=True):
        st.code(text, language=None, wrap_lines=True, line_numbers=True)


def form(note: Note, second: bool, all_notes: list[Note], today: date, pick_key: str) -> None:
    ver_key = f"ver-{second}-{note.name}"
    ver = st.session_state.setdefault(ver_key, 0)
    key = f"{int(second)}-{note.name}-{ver}"
    start, prior_minutes, invalid = start_labels(note, second)
    started = st.session_state.setdefault(f"t0-{key}", time.time())
    if invalid:
        st.warning(
            "The saved file for this note is broken or doesn't match the schema, so the form "
            "opened the prefill. Saving replaces it."
        )

    pd_ = start.principal_diagnosis
    left, right = st.columns([3, 1])
    principal = left.text_input(
        "Principal diagnosis",
        pd_.text if pd_ else "",
        key=f"p-{key}",
        help="The main reason for the stay, as Discharge Diagnosis words it. Empty if none.",
    )
    principal_icd = right.text_input("ICD-10", (pd_.icd10 or "") if pd_ else "", key=f"pi-{key}")

    rows: dict[str, list[dict[str, Any]]] = {}
    for field, items in to_rows(start).items():
        st.markdown(f"**{TITLES[field]}**", help=HELP[field])
        edited = st.data_editor(
            pd.DataFrame(items, columns=list(TABLES[field]), dtype=object),
            num_rows="dynamic",
            hide_index=True,
            column_config=column_config(field),
            key=f"t-{field}-{key}",
        )
        rows[field] = edited.to_dict("records")

    minutes = prior_minutes + (time.time() - started) / 60
    st.caption(f"Time on this note: {minutes:.1f} min (saved with the annotation)")
    a, b = st.columns(2)
    save_next = a.button("💾 Save and next", type="primary", width="stretch")
    save_only = b.button("Save", width="stretch")
    if not (save_next or save_only):
        return
    try:
        ext = from_form(principal, principal_icd, rows)
    except ValidationError as e:
        st.error(f"Not saved: {e.error_count()} problem(s).")
        for err in e.errors():
            st.write(f"- {' → '.join(map(str, err['loc']))}: {err['msg']}")
        return
    path = save(note, ext, minutes, second)
    st.session_state[ver_key] = ver + 1
    nxt = next_todo(all_notes, second, today) if save_next else note
    if nxt is not None:
        st.session_state["goto"] = (pick_key, nxt.name)
    st.session_state["saved"] = f"Saved {path.name} for {note.name}."
    st.rerun()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", type=Path, default=None)
    args = p.parse_args(sys.argv[1:])
    st.set_page_config(page_title="Gold-set annotation", layout="wide")
    gold_dir = args.gold_dir or default_gold_dir()
    all_notes = notes(gold_dir)
    today = date.today()
    if not all_notes:
        st.error(f"No notes found in {gold_dir}. Run `python -m ctagent gold-export` first.")
        return

    with st.sidebar:
        st.title("Gold-set annotation")
        st.caption(f"Folder: `{gold_dir}`")
        second = st.radio("Pass", ["First pass", "Second pass"], horizontal=True) == "Second pass"
        pr = progress(all_notes, today)
        if second:
            st.metric("Second pass done", f"{pr['second_pass_done']} / {pr['overlap_notes']}")
            st.caption(
                f"{pr['second_pass_open']} open now. Each overlap note opens {SECOND_PASS_DAYS} "
                "days after its first pass, starting again from the prefill."
            )
        else:
            st.metric("First pass done", f"{pr['first_pass_done']} / {pr['notes']}")
        work = queue(all_notes, second)
        if not work:
            st.info("No overlap notes in this folder.")
            return
        pick_key = f"pick-{int(second)}"
        goto = st.session_state.pop("goto", None)
        if goto and goto[0] == pick_key:
            st.session_state[pick_key] = goto[1]
        if pick_key not in st.session_state:
            first = next_todo(all_notes, second, today) or work[0]
            st.session_state[pick_key] = first.name
        by_name = {n.name: n for n in work}

        def jump() -> None:
            nxt = next_todo(all_notes, second, today)
            if nxt is not None:
                st.session_state[pick_key] = nxt.name

        st.button("Next to do", on_click=jump, width="stretch")
        name = st.selectbox(
            "Note",
            list(by_name),
            key=pick_key,
            format_func=lambda n: f"{ICONS[status(by_name[n], second, today)]} {n}",
        )

    note = by_name[name]
    state = status(note, second, today)
    if msg := st.session_state.pop("saved", None):
        st.toast(msg, icon="✅")
    st.subheader(f"{ICONS[state]} {note.name}")
    if second and state == "not_started":
        st.info("Finish this note's first pass first.")
        return
    if second and state == "waiting":
        st.info(f"The second pass for this note opens on {second_pass_opens(note)}.")
        return
    if second:
        st.caption("Second pass: work from the note alone. Don't open gold.json.")
    left, right = st.columns([5, 6])
    with left:
        show_note(note)
    with right:
        form(note, second, all_notes, today, pick_key)


main()
