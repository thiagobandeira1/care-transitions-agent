"""Sectioning and silver labels on a fictional summary written for this test (no real data)."""

from ctagent.extraction import Diagnosis
from ctagent.models import MedChangeType
from ctagent.notes import extraction_input, med_entries, silver_labels, split_sections

NOTE = """Name: ___  Unit No: ___
Admission Date: ___  Discharge Date: ___
Service: MEDICINE
Chief Complaint:
Shortness of breath
History of Present Illness:
Fictional patient with heart failure admitted with dyspnea.
Note: this line has a colon but is not a section
Brief Hospital Course:
Diuresed with IV furosemide. Blood cultures pending at discharge.
TSH pending.
Medications on Admission:
1. Furosemide 20 mg PO DAILY
2. Metoprolol Succinate XL 25 mg PO DAILY
3. Aspirin 81 mg PO DAILY
Discharge Medications:
1. Furosemide 40 mg PO DAILY
2. Metoprolol Succinate XL 25 mg PO DAILY
3. Spironolactone 25 mg PO DAILY
Discharge Disposition:
Home
Discharge Diagnosis:
Acute on chronic systolic heart failure
Chronic kidney disease
Discharge Instructions:
You were admitted for heart failure.
Please call your doctor or return to the emergency department if you have:
- weight gain of more than 3 lbs in a day
- shortness of breath at rest
- chest pain
Followup Instructions:
1. Primary care within 7 days
2. Cardiology in 2 weeks
"""


def test_split_sections_finds_known_headers_only() -> None:
    sec = split_sections(NOTE)
    assert set(sec) == {
        "chief_complaint",
        "hpi",
        "hospital_course",
        "meds_admission",
        "meds_discharge",
        "disposition",
        "diagnosis",
        "instructions",
        "followup",
    }
    assert "not a section" in sec["hpi"]  # unknown colon lines stay in the previous section
    assert sec["disposition"] == "Home"


MARKDOWN_NOTE = """**Discharge Summary**

**Patient Name:** ___
**Chief Complaint**
Shortness of breath

---

## Brief Hospital Course
Hospital course was uneventful. Diuresed with IV furosemide.

---

**Discharge Diagnosis:**
Primary:
1. Acute on chronic systolic heart failure
Secondary:
2. Chronic kidney disease

**Pending at Discharge**:
- Urine culture

**Followup Instructions**
- Primary care within **3 days**
- Schedule a visit with the cardiologist in 2 weeks.
"""


def test_split_sections_reads_markdown_headers() -> None:
    sec = split_sections(MARKDOWN_NOTE)
    assert set(sec) == {"chief_complaint", "hospital_course", "diagnosis", "pending", "followup"}
    # a sentence starting with a header name is body text, and rules/bold markers are dropped
    assert sec["hospital_course"].startswith("Hospital course was uneventful.")
    assert sec["chief_complaint"] == "Shortness of breath"
    assert sec["followup"].startswith("- Primary care within 3 days\n")
    lab = silver_labels(MARKDOWN_NOTE)
    assert lab.principal_diagnosis is not None
    assert lab.principal_diagnosis.text == "Acute on chronic systolic heart failure"
    assert [d.text for d in lab.secondary_diagnoses] == ["Chronic kidney disease"]
    assert lab.pending_tests == ["Urine culture"]
    assert [f.with_whom for f in lab.follow_up] == ["Primary care within 3 days", "cardiologist"]


def test_extraction_input_is_ordered_and_trimmed() -> None:
    text, trimmed = extraction_input(NOTE)
    assert text.startswith("DIAGNOSIS:") and "MEDS_DISCHARGE:" in text and not trimmed
    short, trimmed = extraction_input(NOTE, max_chars=50)
    assert len(short) == 50 and trimmed
    assert extraction_input("no headers at all")[0] == "no headers at all"


def test_med_entries_normalize_names() -> None:
    meds = med_entries("1. Furosemide 20 mg PO DAILY\n2. Metoprolol Succinate XL 25 mg\n- Insulin")
    assert set(meds) == {"furosemide", "metoprolol succinate", "insulin"}


def test_silver_labels() -> None:
    lab = silver_labels(NOTE)
    changes = {m.name: m.change for m in lab.medication_changes}
    assert changes == {
        "furosemide": MedChangeType.DOSE_CHANGED,
        "metoprolol succinate": MedChangeType.CONTINUED,
        "aspirin": MedChangeType.STOPPED,
        "spironolactone": MedChangeType.STARTED,
    }
    assert lab.principal_diagnosis is not None
    assert lab.principal_diagnosis.text == "Acute on chronic systolic heart failure"
    assert [f.with_whom for f in lab.follow_up] == [
        "Primary care within 7 days",
        "Cardiology in 2 weeks",
    ]
    assert lab.pending_tests == ["Blood cultures pending at discharge", "TSH pending"]
    assert lab.red_flags == [
        "weight gain of more than 3 lbs in a day",
        "shortness of breath at rest",
        "chest pain",
    ]


def test_structured_diagnoses_override_text_and_free_text_followup() -> None:
    note = NOTE.replace("1. Primary care within 7 days\n2. Cardiology in 2 weeks", "See PCP Monday")
    lab = silver_labels(
        note,
        principal=Diagnosis(text="Heart failure", icd10="I5023"),
        secondary=[Diagnosis(text="CKD", icd10="N183")],
    )
    assert lab.principal_diagnosis == Diagnosis(text="Heart failure", icd10="I5023")
    assert [f.with_whom for f in lab.follow_up] == ["See PCP Monday"]
    assert silver_labels("nothing").principal_diagnosis is None


def test_redacted_followup_section_is_no_followup() -> None:
    note = NOTE.replace("1. Primary care within 7 days\n2. Cardiology in 2 weeks", "___")
    assert silver_labels(note).follow_up == []
