"""Charlson comorbidity from SNOMED CT condition codes (Synthea).

Categories and weights follow Charlson (1987) and the Quan (2005) category definitions. The SNOMED
list is curated from the disorder codes that actually occur in this Synthea population, each code
mapped by its clinical meaning to the Quan ICD-10 category it would code to. It is deliberately
conservative: borderline codes (carcinoma in situ, cystic fibrosis) are left out. Hierarchy:
severe liver disease, diabetes with complications and metastatic tumor replace the milder ones.
"""

from collections.abc import Iterable

# category: (weight, {snomed code: description})
CHARLSON_SNOMED: dict[str, tuple[int, dict[str, str]]] = {
    "mi": (
        1,
        {
            "401303003": "Acute ST segment elevation myocardial infarction",
            "401314000": "Acute non-ST segment elevation myocardial infarction",
            "22298006": "Myocardial infarction",
            "399211009": "History of myocardial infarction",
        },
    ),
    "chf": (1, {"88805009": "Chronic congestive heart failure", "84114007": "Heart failure"}),
    "cevd": (1, {"230690007": "Cerebrovascular accident"}),
    "dementia": (
        1,
        {
            "26929004": "Alzheimer's disease",
            "230265002": "Familial Alzheimer's disease of early onset",
        },
    ),
    "cpd": (
        1,
        {
            "185086009": "Chronic obstructive bronchitis",
            "87433001": "Pulmonary emphysema",
            "195967001": "Asthma",
            "233678006": "Childhood asthma",
        },
    ),
    "rheum": (1, {"69896004": "Rheumatoid arthritis", "200936003": "Lupus erythematosus"}),
    "mild_liver": (
        1,
        {"128302006": "Chronic hepatitis C", "61977001": "Chronic type B viral hepatitis"},
    ),
    "diab": (
        1,
        {
            "44054006": "Diabetes mellitus type 2",
            "427089005": "Diabetes mellitus due to cystic fibrosis",
        },
    ),
    "diab_cc": (
        2,
        {
            "127013003": "Disorder of kidney due to diabetes mellitus",
            "90781000119102": "Microalbuminuria due to type 2 diabetes mellitus",
            "157141000119108": "Proteinuria due to type 2 diabetes mellitus",
            "368581000119106": "Neuropathy due to type 2 diabetes mellitus",
            "97331000119101": "Macular edema and retinopathy due to type 2 diabetes mellitus",
            "60951000119105": "Blindness due to type 2 diabetes mellitus",
        },
    ),
    "hemiplegia": (
        2,
        {
            "698754002": "Chronic paralysis due to lesion of spinal cord",
            "128188000": "Cerebral palsy",
        },
    ),
    "renal": (
        2,
        {
            "431855005": "Chronic kidney disease stage 1",
            "431856006": "Chronic kidney disease stage 2",
            "433144002": "Chronic kidney disease stage 3",
            "431857002": "Chronic kidney disease stage 4",
            "46177005": "End-stage renal disease",
            "129721000119106": "Acute renal failure on dialysis",
            "213150003": "Kidney transplant failure and rejection",
        },
    ),
    "cancer": (
        2,
        {
            "254837009": "Malignant neoplasm of breast",
            "363406005": "Malignant neoplasm of colon",
            "93761005": "Primary malignant neoplasm of colon",
            "109838007": "Overlapping malignant neoplasm of colon",
            "424132000": "Non-small cell carcinoma of lung TNM stage 1",
            "254637007": "Non-small cell lung cancer",
            "67811000119102": "Primary small cell malignant neoplasm of lung TNM stage 1",
            "254632001": "Small cell carcinoma of lung",
            "93143009": "Leukemia disease",
            "109989006": "Multiple myeloma",
            "126906006": "Neoplasm of prostate",
        },
    ),
    "metastatic": (
        6,
        {
            "94260004": "Metastatic malignant neoplasm to colon",
            "94503003": "Metastatic malignant neoplasm to prostate",
        },
    ),
    "hiv": (
        6,
        {
            "62479008": "Acquired immune deficiency syndrome",
            "86406008": "Human immunodeficiency virus infection",
        },
    ),
}
HIERARCHY = {"severe_liver": "mild_liver", "diab_cc": "diab", "metastatic": "cancer"}
CODE_TO_CATEGORY = {code: cat for cat, (_, codes) in CHARLSON_SNOMED.items() for code in codes}
WEIGHTS = {cat: w for cat, (w, _) in CHARLSON_SNOMED.items()}


def charlson_categories(codes: Iterable[str]) -> set[str]:
    found = {CODE_TO_CATEGORY[c] for c in codes if c in CODE_TO_CATEGORY}
    for severe, mild in HIERARCHY.items():
        if severe in found:
            found.discard(mild)
    return found


def charlson_score(codes: Iterable[str]) -> tuple[int, list[str]]:
    cats = charlson_categories(codes)
    return sum(WEIGHTS.get(c, 0) for c in cats), sorted(cats)
