"""Charlson comorbidity from SNOMED CT codes (Synthea) or ICD-9-CM / ICD-10 codes (MIMIC-IV).

Categories and weights follow Charlson (1987) and the Quan (2005) category definitions. The SNOMED
list is curated from the disorder codes that actually occur in this Synthea population, each code
mapped by its clinical meaning to the Quan ICD-10 category it would code to. It is deliberately
conservative: borderline codes (carcinoma in situ, cystic fibrosis) are left out. Hierarchy:
severe liver disease, diabetes with complications and metastatic tumor replace the milder ones.

ICD codes use Quan's enhanced ICD-9-CM and ICD-10 algorithms (Med Care 2005;43:1130-9), matched by
prefix on codes without dots, as MIMIC-IV stores them. A code says its version as "ICD9:4280" or
"ICD10:I509", so the two code systems never collide. ICD-10-CM codes match the ICD-10 prefixes.
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
# Charlson (1987) weights for all 17 categories; the SNOMED map uses the ones Synthea produces
WEIGHTS = {
    "mi": 1,
    "chf": 1,
    "pvd": 1,
    "cevd": 1,
    "dementia": 1,
    "cpd": 1,
    "rheum": 1,
    "pud": 1,
    "mild_liver": 1,
    "diab": 1,
    "diab_cc": 2,
    "hemiplegia": 2,
    "renal": 2,
    "cancer": 2,
    "severe_liver": 3,
    "metastatic": 6,
    "hiv": 6,
}


def _codes(prefix: str, items: str) -> tuple[str, ...]:
    """'I42', '05678' -> I420, I425, I426, I427, I428 (one code per trailing character)."""
    return tuple(prefix + c for c in items)


def _span(prefix: str, lo: int, hi: int, width: int = 2) -> tuple[str, ...]:
    """'C', 0, 26 -> C00 ... C26 (an inclusive range of numbered prefixes)."""
    return tuple(f"{prefix}{n:0{width}d}" for n in range(lo, hi + 1))


QUAN_ICD10: dict[str, tuple[str, ...]] = {
    "mi": ("I21", "I22", "I252"),
    "chf": ("I099", "I110", "I130", "I132", "I255", *_codes("I42", "056789"), "I43", "I50", "P290"),
    "pvd": (
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ),
    "cevd": ("G45", "G46", "H340", *_span("I", 60, 69)),
    "dementia": ("F00", "F01", "F02", "F03", "F051", "G30", "G311"),
    "cpd": ("I278", "I279", *_span("J", 40, 47), *_span("J", 60, 67), "J684", "J701", "J703"),
    "rheum": ("M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"),
    "pud": ("K25", "K26", "K27", "K28"),
    "mild_liver": (
        "B18",
        *_codes("K70", "01239"),
        *_codes("K71", "3457"),
        "K73",
        "K74",
        *_codes("K76", "023489"),
        "Z944",
    ),
    "diab": tuple(f"E1{t}{d}" for t in "01234" for d in "01689"),
    "diab_cc": tuple(f"E1{t}{d}" for t in "01234" for d in "23457"),
    "hemiplegia": ("G041", "G114", "G801", "G802", "G81", "G82", *_codes("G83", "012349")),
    "renal": (
        "I120",
        "I131",
        *_codes("N03", "234567"),
        *_codes("N05", "234567"),
        "N18",
        "N19",
        "N250",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ),
    "cancer": (
        *_span("C", 0, 26),
        *_span("C", 30, 34),
        *_span("C", 37, 41),
        "C43",
        *_span("C", 45, 58),
        *_span("C", 60, 76),
        *_span("C", 81, 85),
        "C88",
        *_span("C", 90, 97),
    ),
    "severe_liver": (
        "I850",
        "I859",
        "I864",
        "I982",
        "K704",
        "K711",
        "K721",
        "K729",
        "K765",
        "K766",
        "K767",
    ),
    "metastatic": ("C77", "C78", "C79", "C80"),
    "hiv": ("B20", "B21", "B22", "B24"),
}
QUAN_ICD9: dict[str, tuple[str, ...]] = {
    "mi": ("410", "412"),
    "chf": (
        "39891",
        "40201",
        "40211",
        "40291",
        "40401",
        "40403",
        "40411",
        "40413",
        "40491",
        "40493",
        *_codes("425", "456789"),
        "428",
    ),
    "pvd": (
        "0930",
        "4373",
        "440",
        "441",
        *_codes("443", "123456789"),
        "4471",
        "5571",
        "5579",
        "V434",
    ),
    "cevd": ("36234", *_span("", 430, 438, 3)),
    "dementia": ("290", "2941", "3312"),
    "cpd": ("4168", "4169", *_span("", 490, 505, 3), "5064", "5081", "5088"),
    "rheum": ("4465", *_codes("710", "01234"), *_codes("714", "0128"), "725"),
    "pud": ("531", "532", "533", "534"),
    "mild_liver": (
        "07022",
        "07023",
        "07032",
        "07033",
        "07044",
        "07054",
        "0706",
        "0709",
        "570",
        "571",
        "5733",
        "5734",
        "5738",
        "5739",
        "V427",
    ),
    "diab": _codes("250", "012389"),
    "diab_cc": _codes("250", "4567"),
    "hemiplegia": ("3341", "342", "343", *_codes("344", "01234569")),
    "renal": (
        "40301",
        "40311",
        "40391",
        "40402",
        "40403",
        "40412",
        "40413",
        "40492",
        "40493",
        "582",
        *_codes("583", "01234567"),
        "585",
        "586",
        "5880",
        "V420",
        "V451",
        "V56",
    ),
    "cancer": (
        *_span("", 140, 172, 3),
        *_span("", 174, 194, 3),
        *_codes("195", "012345678"),
        *_span("", 200, 208, 3),
        "2386",
    ),
    "severe_liver": ("4560", "4561", "4562", *_codes("572", "2345678")),
    "metastatic": ("196", "197", "198", "199"),
    "hiv": ("042", "043", "044"),
}
# prefix -> categories, per code system; prefixes are 3 to 5 characters long
_ICD_PREFIXES: dict[str, dict[str, set[str]]] = {}
for _system, _table in (("ICD9", QUAN_ICD9), ("ICD10", QUAN_ICD10)):
    _index: dict[str, set[str]] = {}
    for _cat, _prefixes in _table.items():
        for _p in _prefixes:
            _index.setdefault(_p, set()).add(_cat)
    _ICD_PREFIXES[_system] = _index


def code_categories(code: str) -> set[str]:
    """The Charlson categories of one code: SNOMED, or 'ICD9:...' / 'ICD10:...'."""
    system, sep, icd = code.partition(":")
    if not sep or system not in _ICD_PREFIXES:
        return {CODE_TO_CATEGORY[code]} if code in CODE_TO_CATEGORY else set()
    index, icd = _ICD_PREFIXES[system], icd.replace(".", "").upper()
    return set().union(*(index.get(icd[:n], set()) for n in (3, 4, 5)))


def charlson_categories(codes: Iterable[str]) -> set[str]:
    found: set[str] = set().union(*(code_categories(c) for c in codes))
    for severe, mild in HIERARCHY.items():
        if severe in found:
            found.discard(mild)
    return found


def charlson_score(codes: Iterable[str]) -> tuple[int, list[str]]:
    cats = charlson_categories(codes)
    return sum(WEIGHTS.get(c, 0) for c in cats), sorted(cats)
