"""
services/clinical_rules_config.py – Master Declarative Clinical Rules and Constants.

RESPONSIBILITIES:
  1. Declarative definition of compound/combination clinical rules.
  2. Clinical exclusivity and hard-rejection prefix mappings.
  3. Entity-to-prefix coverage validation maps.
  4. Mandatory clinical group definitions for fallback stability.
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPOUND RULES — Declarative condition-pair → result code
# ─────────────────────────────────────────────────────────────────────────────
COMPOUND_RULES: list[dict] = [
    {
        "id": "DM2_PERIPHERAL_NEUROPATHY",
        "conditions": ["type 2", "diabet", "neuropath"],
        "code": "E11.42",
        "desc": "Type 2 diabetes mellitus with diabetic peripheral neuropathy, unspecified",
        "suppresses": ["G57", "G58", "G59", "G60", "G61", "G62", "G63", "G64", "E11.9", "E11.40", "E11.41", "E11.638", "E11.628", "E11.49", "E13.8"],
        "covers_groups": ["neuropathy", "diabet", "dm2", "type 2 diabetes"]
    },
    {
        "id": "DM2_MONONEUROPATHY",
        "conditions": ["type 2", "diabet", "mononeuropath"],
        "code": "E11.41",
        "desc": "Type 2 diabetes mellitus with diabetic mononeuropathy",
        "suppresses": ["G57", "G58", "E11.9", "E11.40"],
        "covers_groups": ["neuropathy", "diabet", "dm2"]
    },
    {
        "id": "DM2_CKD",
        "conditions": ["type 2", "diabet", "chronic kidney"],
        "code": "E11.22",
        "desc": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
        "suppresses": ["E11.65", "E11.9"],
        "requires_additional_codes": ["N18"],
        "covers_groups": ["ckd", "chronic kidney", "diabet", "dm2"]
    },
    {
        "id": "DM2_NEPHROPATHY",
        "conditions": ["type 2", "diabet", "nephropath"],
        "code": "E11.21",
        "desc": "Type 2 diabetes mellitus with diabetic nephropathy",
        "suppresses": ["E11.9"],
        "covers_groups": ["nephropathy", "diabet", "dm2"]
    },
    {
        "id": "DM2_RETINOPATHY_BACKGROUND",
        "conditions": ["diabet", "retinopathy", "background"],
        "code": "E11.319",
        "desc": "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema",
        "suppresses": ["E11.9", "E11.31", "E11.39"],
        "covers_groups": ["retinopathy", "diabet", "dm2"]
    },
    {
        "id": "DM2_FOOT_ULCER",
        "conditions": ["type 2", "diabet", "foot ulcer"],
        "code": "E11.621",
        "desc": "Type 2 diabetes mellitus with foot ulcer",
        "suppresses": ["E11.9", "E11.62"],
        "covers_groups": ["ulcer", "foot ulcer", "diabet", "dm2"]
    },
    {
        "id": "HF_ACUTE_ON_CHRONIC_SYSTOLIC",
        "conditions": ["acute on chronic systolic", "heart failure"],
        "code": "I50.23",
        "desc": "Acute on chronic systolic (congestive) heart failure",
        "suppresses": ["I50.9", "I50.20", "I50.21", "I50.40", "I50.41", "I50.43"],
        "covers_groups": ["heart failure", "chf"]
    },
    {
        "id": "HF_ACUTE_ON_CHRONIC_DIASTOLIC",
        "conditions": ["acute on chronic diastolic", "heart failure"],
        "code": "I50.33",
        "desc": "Acute on chronic diastolic (congestive) heart failure",
        "suppresses": ["I50.9", "I50.30", "I50.31", "I50.40", "I50.41"],
        "covers_groups": ["heart failure", "chf"]
    },
    {
        "id": "HF_ACUTE_SYSTOLIC",
        "conditions": ["acute systolic", "heart failure"],
        "code": "I50.21",
        "desc": "Acute systolic (congestive) heart failure",
        "suppresses": ["I50.9", "I50.20", "I50.40", "I50.43"],
        "covers_groups": ["heart failure", "chf"]
    },
    {
        "id": "HF_CHRONIC_SYSTOLIC",
        "conditions": ["chronic systolic", "heart failure"],
        "code": "I50.22",
        "desc": "Chronic systolic (congestive) heart failure",
        "suppresses": ["I50.9", "I50.20"],
        "covers_groups": ["heart failure", "chf"]
    },
    {
        "id": "CKD_STAGE_3B",
        "conditions": ["stage 3b", "chronic kidney"],
        "code": "N18.32",
        "desc": "Chronic kidney disease, stage 3b",
        "suppresses": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27", "N18.9", "N18.30", "N18.3", "N18.31"],
        "covers_groups": ["ckd", "chronic kidney", "renal disease"]
    },
    {
        "id": "CKD_STAGE_3A",
        "conditions": ["stage 3a", "chronic kidney"],
        "code": "N18.31",
        "desc": "Chronic kidney disease, stage 3a",
        "suppresses": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27", "N18.9", "N18.30", "N18.3", "N18.32"],
        "covers_groups": ["ckd", "chronic kidney", "renal disease"]
    },
    {
        "id": "CKD_STAGE_3",
        "conditions": ["stage 3", "chronic kidney"],
        "code": "N18.3",
        "desc": "Chronic kidney disease, stage 3 unspecified",
        "suppresses": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27", "N18.9", "N18.30"],
        "covers_groups": ["ckd", "chronic kidney", "renal disease"]
    },
    {
        "id": "CKD_STAGE_4",
        "conditions": ["stage 4", "chronic kidney"],
        "code": "N18.4",
        "desc": "Chronic kidney disease, stage 4",
        "suppresses": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27", "N18.9", "N18.3"],
        "covers_groups": ["ckd", "chronic kidney", "renal disease"]
    },
    {
        "id": "CKD_STAGE_5_ESRD",
        "conditions": ["esrd"],
        "code": "N18.6",
        "desc": "End stage renal disease",
        "suppresses": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27", "N18.9", "N18.3", "N18.4", "N18.5"],
        "covers_groups": ["ckd", "chronic kidney", "esrd", "renal disease"]
    },
    {
        "id": "HTN_WITH_HEART_FAILURE",
        "conditions": ["hypertension", "heart failure"],
        "code": "I11.0",
        "desc": "Hypertensive heart disease with heart failure",
        "suppresses": ["I11.9"],
        "covers_groups": ["hypertension", "htn", "heart failure", "chf"]
    },
    {
        "id": "HTN_WITH_CKD",
        "conditions": ["hypertension", "chronic kidney"],
        "code": "I12.9",
        "desc": "Hypertensive chronic kidney disease with stage 1-4 or unspecified CKD",
        "suppresses": ["I10"],
        "covers_groups": ["hypertension", "htn", "ckd", "chronic kidney"]
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL EXCLUSIVITY RULES
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_EXCLUSIVITY_RULES: list[dict] = [
    {
        "id": "DM_TYPE2_EXCLUSIVITY",
        "note_signals": ["type 2 diabetes", "t2dm", "dm2", "dm 2", "type ii diabetes"],
        "exclude_prefixes": ["E10", "E13", "E08", "E09"],
        "exclude_exact": [],
        "description": "Only E11.* allowed when Type 2 diabetes is confirmed",
    },
    {
        "id": "DM_TYPE1_EXCLUSIVITY",
        "note_signals": ["type 1 diabetes", "t1dm", "dm1", "type i diabetes"],
        "exclude_prefixes": ["E11", "E13", "E08", "E09"],
        "exclude_exact": [],
        "description": "Only E10.* allowed when Type 1 diabetes is confirmed",
    },
    {
        "id": "HTN_ESSENTIAL_EXCLUSIVITY",
        "note_signals": ["hypertension", "htn", "high blood pressure"],
        "anti_signals": ["secondary hypertension", "renovascular", "renal hypertension", "endocrine hypertension", "I15"],
        "exclude_prefixes": ["I15"],
        "exclude_exact": [],
        "description": "I15 (secondary HTN) blocked unless secondary HTN explicitly stated",
    },
    {
        "id": "METABOLIC_NOISE_FILTER",
        "note_signals": ["*"],
        "anti_signals": ["metabolic syndrome", "e88", "metabolic disorder"],
        "exclude_prefixes": ["E88"],
        "exclude_exact": [],
        "description": "E88.x removed unless metabolic syndrome explicitly in note",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PREFIX SUPPRESSION OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────
CROSS_PREFIX_SUPPRESS: dict[str, list[str]] = {
    "E11.42": ["P:G57", "P:G58", "P:G59", "P:G60", "P:G61", "P:G62", "P:G63", "P:G64"],
    "E11.41": ["P:G57", "P:G58", "P:G56"],
    "E11.319": ["EXACT:H35.00"],
    "E11.3519": ["EXACT:H35.31"],
    "E11.22": ["EXACT:N18.9"],
    "I12.9":  ["EXACT:I10"],
    "I13.0":  ["EXACT:I10", "EXACT:I11.9", "EXACT:I50.9"],
    "I13.2":  ["EXACT:I10", "EXACT:I11.9", "EXACT:I50.9"],
    "I11.0":  ["EXACT:I10"],
}

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-HIERARCHY SUPPRESSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
HIERARCHY_SUPPRESSION: dict[str, list[str]] = {
    "E11.42": ["G62", "G60", "G61", "G56"],
    "E11.40": ["G62", "G60", "G61", "G56"],
    "E11.22": ["N18"],
    "I12": ["I10", "N18"],
    "I13": ["I10", "I11", "I12", "I50", "N18"]
}

# ─────────────────────────────────────────────────────────────────────────────
# MANDATORY GROUPS COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────
MANDATORY_GROUPS = {
    "diabetes":          {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
    "hypertension":      {"code": "I10", "description": "Essential (primary) hypertension"},
    "chronic kidney":    {"code": "N18.9", "description": "Chronic kidney disease, unspecified"},
    "ckd":               {"code": "N18.9", "description": "Chronic kidney disease, unspecified"},
    "heart failure":     {"code": "I50.9", "description": "Heart failure, unspecified"},
    "neuropathy":        {"code": "G62.9", "description": "Polyneuropathy, unspecified"},
    "obesity":           {"code": "E66.9", "description": "Obesity, unspecified"},
    "lipid":             {"code": "E78.5", "description": "Hyperlipidemia, unspecified"},
    "sepsis":            {"code": "A41.9", "description": "Sepsis, unspecified"},
    "pneumonia":         {"code": "J18.9", "description": "Pneumonia, unspecified"},
}

# ─────────────────────────────────────────────────────────────────────────────
# HARD-REJECT CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────
HARD_REJECT_PREFIXES: dict[str, list[str]] = {
    "O":  ["pregnan", "obstetric", "maternal", "antepartum", "postpartum", "labour", "labor", "gestation", "trimester"],
    "V":  [],
    "P":  ["neonat", "perinatal", "newborn", "premature", "infant"],
    "Q":  ["congenital", "malformation", "anomaly", "genetic"],
}

ALWAYS_REJECT_PREFIXES: set[str] = {"V"}

RENAL_SYNDROME_PREFIXES: set[str] = {
    "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N25", "N26", "N27",
}

CKD_ENTITY_SIGNALS: list[str] = [
    "chronic kidney", "ckd", "renal disease", "nephropathy", "kidney disease", "renal failure", "kidney failure",
]

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY-TO-PREFIX COVERAGE MAP
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_PREFIX_MAP: dict[str, list[str]] = {
    "diabet": ["E10", "E11", "E13", "E08", "E09"],
    "dm2": ["E11"], "dm1": ["E10"],
    "t2dm": ["E11"], "t1dm": ["E10"],
    "chronic kidney": ["N18"], "ckd": ["N18"],
    "renal failure": ["N18", "N17", "N19"],
    "kidney disease": ["N18", "N17"],
    "nephropathy": ["N18", "E11", "E10"], "aki": ["N17"],
    "heart failure": ["I50"], "cardiac failure": ["I50"],
    "heart attack": ["I21"], "ami": ["I21"],
    "myocardial infarct": ["I21", "I22"],
    "stemi": ["I21"], "nstemi": ["I21"],
    "cad": ["I25"], "coronary": ["I25"], "angina": ["I20"],
    "atrial fibrillation": ["I48"], "afib": ["I48"], "a-fib": ["I48"],
    "hypertension": ["I10", "I11", "I12", "I13"],
    "htn": ["I10", "I11", "I12", "I13"], "high blood pressure": ["I10"],
    "neuropath": ["G60", "G61", "G62", "G63", "E11", "E10"],
    "peripheral neuropath": ["G62", "E11"],
    "obesi": ["E66"], "overweight": ["E66"], "bmi": ["Z68"],
    "hyperlipid": ["E78"], "dyslipid": ["E78"], "cholesterol": ["E78"], "triglyc": ["E78"],
    "gout": ["M10"], "hyperuricemia": ["E79"],
    "cirrhosis": ["K74"], "hepatitis": ["K72", "B18", "B19"],
    "fatty liver": ["K76"], "nash": ["K75"],
    "copd": ["J44"], "emphysema": ["J43"], "asthma": ["J45"], "bronchospasm": ["J45"],
    "pneumonia": ["J18", "J15", "J13"], "respiratory failure": ["J96"],
    "pulmonary embolism": ["I26"], "pe": ["I26"],
    "dvt": ["I82"], "deep vein thrombosis": ["I82"],
    "bacteremia": ["A41"],
    # Anemia
    "anemia": ["D50", "D51", "D63", "D64"],
    "iron deficiency": ["D50"],
    # Stroke
    "stroke": ["I63", "I64"], "cerebral infarct": ["I63"],
    "tia": ["G45"],
    # GI
    "cholelith": ["K80"], "gallstone": ["K80"],
    "cholecystitis": ["K81"],
    "appendicitis": ["K37"],
    "pancreatitis": ["K85", "K86"],
    # Skin
    "cellulitis": ["L03"],
    "pressure ulcer": ["L89"],
    # MSK
    "back pain": ["M54"], "lumbalgia": ["M54"],
    "arthritis": ["M05", "M06", "M15", "M16", "M17"],
    "osteoporosis": ["M80", "M81"],
    "fracture": ["S", "M80"],
    "fall": ["W"],
    "displaced": ["S"],
    "nondisplaced": ["S"],
    "femoral neck fracture": ["S72"],
    "intertrochanteric fracture": ["S72"],
    "hip fracture": ["S72", "M80"],
    "internal fixation": ["CPT", "S"],
    "fixation": ["CPT", "S"],
    "open fracture": ["S"],
    "closed fracture": ["S"],
    # Mental
    "depression": ["F32", "F33"],
    "anxiety": ["F41"],
    "bipolar": ["F31"],
    "schizophrenia": ["F20"],
    # Oncology (generic)
    "cancer": ["C"],
    "malignant": ["C"],
    "tumor": ["C", "D"],
    "neoplasm": ["C", "D"],
    # Pediatric
    "dehydration": ["E86"],
    "electrolyte": ["E87"],
    # Procedures (CPT — entity validation bypassed for CPT codes)
}

# ─────────────────────────────────────────────────────────────────────────────
# RELATIONSHIP VALIDATION RULES (NEW — FIX 2)
# Ensure combination codes only appear when all underlying conditions are present.
# ─────────────────────────────────────────────────────────────────────────────

RELATIONSHIP_VALIDATION_RULES: list[dict] = [
    {
        "id": "HYPERTENSIVE_CKD",
        "target_prefixes": ["I12", "I13"],
        "required_entities": ["hypertension", "chronic kidney"],
        "description": "Hypertensive CKD requires mention of both HTN and CKD",
    },
    {
        "id": "HYPERTENSIVE_HEART_DISEASE",
        "target_prefixes": ["I11", "I13"],
        "required_entities": ["hypertension", "heart failure"],
        "description": "Hypertensive heart disease requires mention of both HTN and HF",
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC BOOSTS (TASK 85)
# Target weak domains with high-confidence grounding triggers.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_SPECIFIC_BOOSTS = {
    "orthopedics": {
        "prefixes": ["S72", "S82", "S42", "S52", "S32"],
        "triggers": ["displaced", "nondisplaced", "comminuted", "oblique", "transverse", "spiral", "impacted", "angulated"],
        "laterality_required": True,
        "boost_amount": 0.15
    },
    "endocrine": {
        "prefixes": ["E11", "E10"],
        "triggers": ["retinopathy", "neuropathy", "nephropathy", "ulcer", "gangrene", "gastroparesis", "polyneuropathy"],
        "boost_amount": 0.20
    },
    "cardiology": {
        "prefixes": ["I50", "I21", "I25"],
        "triggers": ["systolic", "diastolic", "nstemi", "stemi", "exacerbation", "decompensated"],
        "boost_amount": 0.10
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN MERGE RULES (TASK 85)
# Logical combinations for high-impact specialties.
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_MERGE_RULES = [
    {
        "id": "CARDIOLOGY_HTN_HF_MERGE",
        "domain": "cardiology",
        "members": ["I10", "I50"],
        "target": "I11.0",
        "requires_all": True
    },
    {
        "id": "NEPHROLOGY_HTN_CKD_MERGE",
        "domain": "nephrology",
        "members": ["I10", "N18"],
        "target": "I12.9",
        "requires_all": True
    },
    {
        "id": "ENDOCRINE_DM_CKD_MERGE",
        "domain": "endocrine",
        "members": ["E11", "N18"],
        "target": "E11.22",
        "requires_all": True
    },
    {
        "id": "SEPSIS_AKI_ASSOCIATION",
        "domain": "sepsis",
        "members": ["A41", "N17"],
        "target": "A41.9", # Sepsis remains primary but AKI is protected
        "requires_all": True,
        "protect_members": True
    }
]
