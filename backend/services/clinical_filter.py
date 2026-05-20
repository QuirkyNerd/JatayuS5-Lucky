"""
services/clinical_filter.py - Hybrid Entity Classification + Pre-RAG Clinical Filter (v14)

TWO DISTINCT ROLES:
  1. EntityClassifier — tags each extracted entity with a clinical intent class
     BEFORE it enters the RAG query pipeline. Uses ontology + synonym + abbreviation map.
  2. ClinicalRelevanceFilter — post-selection cap that enforces hard output limits
     (max 10 ICD codes, symptom suppression when diagnosis exists).

Architecture:
  Extract → EntityClassifier → ClinicalEntityFilter → RAG → SelectionEngine
  → ClinicalRelevanceFilter → LLM Explanation
"""

from typing import List, Dict, Set
import re

try:
    from utils.logging import get_logger
except ImportError:
    from utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ABBREVIATION → CANONICAL MAP (clinical shorthand the ontology may miss)
# ─────────────────────────────────────────────────────────────────────────────
ABBREVIATION_MAP: dict[str, str] = {
    "sob": "dyspnea",
    "doe": "dyspnea on exertion",
    "cp": "chest pain",
    "n/v": "nausea and vomiting",
    "ha": "headache",
    "loc": "loss of consciousness",
    "brbpr": "bright red blood per rectum",
    "gi bleed": "gastrointestinal hemorrhage",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "abd pain": "abdominal pain",
    "jvd": "jugular venous distension",
    "lle": "left lower extremity",
    "rle": "right lower extremity",
    "bka": "below knee amputation",
    "aka": "above knee amputation",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "cva": "stroke",
    "tia": "transient ischemic attack",
    "sz": "seizure",
    "ams": "altered mental status",
    "r/o": "rule out",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "chf": "heart failure",
    "mi": "myocardial infarction",
    "afib": "atrial fibrillation",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "aki": "acute kidney injury",
    "cad": "coronary artery disease",
    "bph": "benign prostatic hyperplasia",
    "gerd": "gastroesophageal reflux disease",
    "osa": "obstructive sleep apnea",
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY CLASS ONTOLOGY — keyword → entity_class mapping
# These define the CLINICAL INTENT of an entity, not its ICD mapping.
# ─────────────────────────────────────────────────────────────────────────────
_SYMPTOM_KEYWORDS: set[str] = {
    "pain", "ache", "dyspnea", "shortness of breath", "cough", "fever",
    "fatigue", "weakness", "dizziness", "nausea", "vomiting", "diarrhea",
    "constipation", "headache", "chest pain", "abdominal pain", "edema",
    "swelling", "palpitations", "syncope", "malaise", "chills", "rigors",
    "wheezing", "hemoptysis", "hematuria", "dysuria", "polyuria", "polydipsia",
    "weight loss", "weight gain", "anorexia", "insomnia", "tremor",
    "numbness", "tingling", "paresthesia", "pruritus", "rash", "bruising",
    "bleeding", "epistaxis", "tinnitus", "vertigo", "diplopia", "blurred vision",
    "dysphagia", "odynophagia", "hoarseness", "stridor", "orthopnea",
    "jugular venous distension", "altered mental status", "confusion",
    "lethargy", "somnolence", "agitation", "diaphoresis", "claudication",
    "myalgia", "arthralgia", "back pain", "flank pain",
}

_LAB_KEYWORDS: set[str] = {
    "troponin", "bnp", "creatinine", "bun", "gfr", "hemoglobin", "hematocrit",
    "wbc", "platelet", "inr", "ptt", "d-dimer", "lactate", "albumin",
    "bilirubin", "ast", "alt", "alkaline phosphatase", "lipase", "amylase",
    "sodium", "potassium", "chloride", "bicarbonate", "glucose", "hba1c",
    "a1c", "tsh", "t4", "t3", "ferritin", "iron", "tibc", "b12", "folate",
    "urine culture", "blood culture", "sputum culture", "csf analysis",
    "urinalysis", "cbc", "bmp", "cmp", "abg", "vbg", "coagulation",
    "procalcitonin", "crp", "esr", "ck", "ldh", "pro-bnp",
}

_OBSERVATION_KEYWORDS: set[str] = {
    "x-ray", "ct scan", "mri", "ultrasound", "echocardiogram", "ekg",
    "electrocardiogram", "chest x-ray", "ct abdomen", "ct head", "ct chest",
    "mri brain", "doppler", "angiogram", "pet scan", "bone scan",
    "colonoscopy finding", "endoscopy finding", "biopsy result",
    "imaging", "radiology", "nuclear medicine",
}

_PROCEDURE_KEYWORDS: set[str] = {
    "surgery", "arthroplasty", "cholecystectomy", "appendectomy", "cabg",
    "percutaneous coronary intervention", "catheterization", "intubation",
    "ventilation", "dialysis", "transfusion", "thoracentesis", "paracentesis",
    "lumbar puncture", "biopsy", "excision", "resection", "repair",
    "replacement", "implantation", "removal", "drainage", "debridement",
    "amputation", "transplant", "stent", "pacemaker", "defibrillator",
    "endoscopy", "colonoscopy", "bronchoscopy", "tracheostomy", "gastrostomy",
    "colostomy", "ileostomy", "mastectomy", "hysterectomy", "prostatectomy",
    "thyroidectomy", "cesarean section", "laparoscopic", "open",
    "echocardiogram", "ekg", "electrocardiogram",
    "orif", "fixation", "reduction", "intramedullary", "osteotomy", "debridement",
    "hardware", "fusion", "reconstruction"
}

_MEDICATION_KEYWORDS: set[str] = {
    "metformin", "insulin", "glipizide", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atorvastatin", "aspirin", "warfarin", "heparin", "enoxaparin",
    "furosemide", "spironolactone", "digoxin", "amiodarone", "levothyroxine",
    "prednisone", "albuterol", "fluticasone", "omeprazole", "pantoprazole",
    "ceftriaxone", "vancomycin", "piperacillin", "azithromycin", "ciprofloxacin",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION PRIORITY SCORES (higher = more clinically significant)
# ─────────────────────────────────────────────────────────────────────────────
SECTION_PRIORITY: dict[str, int] = {
    "principal_diagnosis": 10,
    "primary_diagnosis": 10,
    "admitting_diagnosis": 9,
    "secondary_diagnosis": 7,
    "secondary_diagnoses": 7,
    "procedures": 6,
    "procedure": 6,
    "operative": 6,
    "comorbidities": 5,
    "past_medical_history": 3,
    "history_of_present_illness": 4,
    "hospital_course": 2,
    "assessment_and_plan": 5,
    "assessment": 5,
    "plan": 4,
    "medications": 1,
    "labs": 1,
    "vitals": 1,
    "review_of_systems": 2,
    "symptoms": 1,
    "chief_complaint": 3,
    "default": 3,
}

# Common diagnostic CPT codes to filter from final output
DIAGNOSTIC_CPTS: set[str] = {
    "71045", "71046", "71047", "71048",  # Chest X-rays
    "93306", "93307", "93308",           # Echocardiograms
    "85025", "85027", "80053", "80048",  # CBC, CMP, BMP
    "93000", "93005", "93010",           # EKG/ECG
    "74176", "74177", "74178",           # CT Abdomen
    "70450", "70460", "70470",           # CT Head
}


# ═══════════════════════════════════════════════════════════════════════════════
# EntityClassifier — HYBRID classification (ontology + synonym + abbreviation)
# ═══════════════════════════════════════════════════════════════════════════════
class EntityClassifier:
    """
    Classifies each extracted entity into one of:
      diagnosis | symptom | procedure | lab | observation | medication

    Uses a 3-tier hybrid approach:
      1. Abbreviation expansion (SOB → dyspnea → symptom)
      2. Keyword ontology match (longest match wins)
      3. ICD code prefix heuristic (R* → symptom, Z* → screening)
    """

    @staticmethod
    def classify(entity_text: str, section: str = "default", icd_code: str = "") -> str:
        """Return entity_class for a given entity string."""
        text_lower = entity_text.lower().strip()

        # --- Tier 1: Abbreviation expansion ---
        for abbr, expansion in ABBREVIATION_MAP.items():
            if text_lower == abbr or text_lower == expansion:
                # Re-classify the expanded form
                text_lower = expansion
                break

        # --- Tier 2: Keyword ontology (longest match first) ---
        best_class = None
        best_len = 0

        for kw in _PROCEDURE_KEYWORDS:
            if kw in text_lower and len(kw) > best_len:
                best_class = "procedure"
                best_len = len(kw)

        for kw in _LAB_KEYWORDS:
            if kw in text_lower and len(kw) > best_len:
                best_class = "lab"
                best_len = len(kw)

        for kw in _OBSERVATION_KEYWORDS:
            if kw in text_lower and len(kw) > best_len:
                best_class = "observation"
                best_len = len(kw)

        for kw in _MEDICATION_KEYWORDS:
            if kw in text_lower and len(kw) > best_len:
                best_class = "medication"
                best_len = len(kw)

        for kw in _SYMPTOM_KEYWORDS:
            if kw in text_lower and len(kw) > best_len:
                best_class = "symptom"
                best_len = len(kw)

        if best_class:
            return best_class

        # --- Tier 3: ICD code prefix heuristic ---
        code_upper = icd_code.upper().strip()
        if code_upper.startswith("R"):
            return "symptom"
        if code_upper.startswith("Z"):
            return "observation"  # screening / contact codes

        # Default: if in a diagnosis section and not classified, assume diagnosis
        return "diagnosis"

    @staticmethod
    def get_section_priority(section: str) -> int:
        """Return numeric priority for a clinical document section."""
        key = section.lower().strip().replace(" ", "_").replace("-", "_")
        return SECTION_PRIORITY.get(key, SECTION_PRIORITY["default"])


# ═══════════════════════════════════════════════════════════════════════════════
# ClinicalEntityFilter — PRE-RAG filter (prunes entities before RAG queries)
# ═══════════════════════════════════════════════════════════════════════════════
class ClinicalEntityFilter:
    """
    Applied BEFORE RAG to prune the entity/rag_query list.
    This prevents symptom, lab, and observation entities from ever reaching RAG,
    which is the root cause of code overcounting (18 codes → 6-10).
    """

    @staticmethod
    def filter_entities(
        entities: list[dict],
        rag_queries: list[str],
        deterministic_codes: list[dict],
    ) -> tuple[list[dict], list[str], list[dict]]:
        """
        Prune entities + rag_queries + deterministic_codes based on clinical intent.

        Rules:
          1. If ANY diagnosis-class entity exists, DROP all symptom entities.
          2. Always DROP lab/observation/medication entities (they are not coded).
          3. Entities from low-priority sections (hospital_course, symptoms) are deprioritized.

        Returns: (filtered_entities, filtered_rag_queries, filtered_det_codes)
        """
        # Classify each entity
        for ent in entities:
            ent["entity_class"] = EntityClassifier.classify(
                ent.get("entity", ""),
                ent.get("section", "default"),
                ent.get("code", ""),
            )
            ent["section_priority"] = EntityClassifier.get_section_priority(
                ent.get("section", "default")
            )

        # Check if any diagnosis-class entity exists
        has_diagnosis = any(e["entity_class"] == "diagnosis" for e in entities)

        filtered_entities = []
        filtered_queries = []

        for ent in entities:
            ec = ent["entity_class"]

            # SOFT CALIBRATION: Stop dropping symptom entities here. 
            # We want them to reach RAG to find related codes. 
            # SelectionEngine and PostFilter will handle suppression if redundant.
            
            # Rule 2: allow labs, observations, medications to reach RAG 
            # (they ground specific diagnoses like CKD or MI)
            
            filtered_entities.append(ent)
            if ent.get("rag_query"):
                filtered_queries.append(ent["rag_query"])

        # 🚨 TASK 14: PRESERVE EMERGENCY QUERIES
        if not filtered_queries and rag_queries:
            logger.info("PreRAGFilter: no entity queries, preserving %d emergency queries", len(rag_queries))
            filtered_queries = rag_queries

        # Also filter deterministic codes whose entity_class would be symptom/lab/obs
        filtered_det = []
        for code_dict in deterministic_codes:
            code_str = code_dict.get("code", "").upper()
            # Suppress R-codes (symptoms) if diagnosis exists
            if code_str.startswith("R") and has_diagnosis:
                logger.info(
                    "PreRAGFilter: dropped symptom code %s from deterministic pool", code_str
                )
                continue
            # Suppress diagnostic CPTs
            if code_str in DIAGNOSTIC_CPTS:
                logger.info(
                    "PreRAGFilter: dropped diagnostic CPT %s from deterministic pool", code_str
                )
                continue
            filtered_det.append(code_dict)

        logger.info(
            "PreRAGFilter: %d→%d entities, %d→%d queries, %d→%d det codes",
            len(entities), len(filtered_entities),
            len(rag_queries), len(filtered_queries),
            len(deterministic_codes), len(filtered_det),
        )

        return filtered_entities, filtered_queries, filtered_det


# ═══════════════════════════════════════════════════════════════════════════════
# ClinicalRelevanceFilter — POST-SELECTION final cap (unchanged role, improved)
# ═══════════════════════════════════════════════════════════════════════════════
class ClinicalRelevanceFilter:
    """
    Final output gatekeeper. Applied AFTER SelectionEngine to enforce:
      1. No symptom R-codes if ANY definitive diagnosis exists
      2. No diagnostic CPTs in final output
      3. Hard cap of 10 ICD codes maximum
      4. Sort by section_priority + confidence for cap decisions
    """

    @staticmethod
    def _has_definitive_diagnosis(codes: List[Dict]) -> bool:
        definitive_prefixes = tuple("ABCDEFGHIJKLMN")
        for c in codes:
            if c.get("type", "").upper() == "ICD-10":
                code_str = c.get("code", "").upper()
                if code_str.startswith(definitive_prefixes):
                    return True
        return False

    @staticmethod
    def filter_codes(candidates: List[Dict], note_text: str = "") -> List[Dict]:
        has_definitive = ClinicalRelevanceFilter._has_definitive_diagnosis(candidates)
        filtered = []

        for code_dict in candidates:
            code_type = code_dict.get("type", "").upper()
            code_str = code_dict.get("code", "").upper()

            # Early urology protection
            is_urology = any(code_str.startswith(pre) for pre in ["N13", "52332", "74176"])
            if is_urology:
                code_dict["protected"] = True

            # Rule 1: reject symptom codes if definitive diagnosis present
            if code_type == "ICD-10" and code_str.startswith("R") and has_definitive and not code_dict.get("protected"):
                logger.info("PostFilter: suppressed symptom %s", code_str)
                continue

            # Rule 2: reject diagnostic CPTs
            if code_type == "CPT" and code_str in DIAGNOSTIC_CPTS and not code_dict.get("protected"):
                logger.info("PostFilter: suppressed diagnostic CPT %s", code_str)
                continue

            filtered.append(code_dict)

        # Rule 3: sort by priority then confidence, cap at 10 ICD codes
        def sort_key(c):
            pri = int(c.get("section_priority", 3))
            conf = float(c.get("confidence", 0))
            is_protected = 1 if c.get("protected", False) else 0
            return (is_protected, pri, conf)

        filtered.sort(key=sort_key, reverse=True)

        icd_count = 0
        final_list = []
        for c in filtered:
            if c.get("type", "").upper() == "ICD-10":
                if icd_count < 10:
                    final_list.append(c)
                    icd_count += 1
                else:
                    logger.info("PostFilter: dropped %s (10 ICD cap)", c.get("code"))
            else:
                final_list.append(c)  # CPT codes pass through

        return final_list


# ═══════════════════════════════════════════════════════════════════════════════
# ClinicalGroundingEngine — AFTER RAG, BEFORE SelectionEngine (v7 — NEW)
# ═══════════════════════════════════════════════════════════════════════════════
class ClinicalGroundingEngine:
    """
    CLINICAL TRUTH VALIDATION LAYER (v7)

    This is the MISSING LAYER the user identified.
    It runs AFTER RAG retrieval but BEFORE SelectionEngine.

    Purpose:
      1. EVIDENCE VALIDATION: every RAG candidate must have a supporting entity in the note
      2. ENTITY-TYPE ALIGNMENT: symptom entity cannot produce diagnosis code and vice versa
      3. CLINICAL PRIORITY: diagnoses > procedures > symptoms
      4. REJECT UNGROUNDED CODES: if no entity supports a code, it is REMOVED

    This layer is NON-NEGOTIABLE for clinical safety.
    """

    # Entity types that cannot produce certain code types
    _TYPE_ALIGNMENT_RULES: dict[str, set[str]] = {
        # SOFT CALIBRATION: Allow labs/obs/meds to ground diagnoses. 
        # They are often critical evidence for specific codes (e.g. Troponin for MI).
        "symptom": set(),
        "lab": set(),
        "observation": set(),
        "medication": set(),
    }

    @staticmethod
    def ground_candidates(
        rag_candidates: List[Dict],
        note_entities: List[str],
        entity_classes: Dict[str, str],
        note_text: str = "",
    ) -> Dict[str, List[Dict]]:
        """
        Filter RAG candidates to only include codes with clinical grounding.

        Returns: { "grounded": [...], "rejected": [...] }
        """
        note_lower = note_text.lower()
        entity_set = {e.lower() for e in note_entities}
        grounded: List[Dict] = []
        rejected: List[Dict] = []
        rejected_count = 0

        for candidate in rag_candidates:
            code = candidate.get("code", "").strip().upper()
            desc = candidate.get("description", "").lower()
            code_type = candidate.get("type", "ICD-10").upper()
            source = candidate.get("source", "rag")
            confidence = float(candidate.get("confidence", 0))

            # Rule 0: deterministic codes always pass (clinically pre-validated)
            if source == "deterministic" or candidate.get("protected", False):
                candidate["grounding"] = "deterministic"
                candidate["grounding_entity"] = "pre-validated"
                grounded.append(candidate)
                continue

            # Rule 1: CPT procedural codes pass if procedure entity exists
            if code_type == "CPT":
                has_proc_entity = any(
                    entity_classes.get(e, "") == "procedure" for e in entity_classes
                )
                if has_proc_entity:
                    candidate["grounding"] = "procedure_entity"
                    grounded.append(candidate)
                else:
                    # Check if procedure keyword is in note text
                    proc_in_note = any(kw in note_lower for kw in _PROCEDURE_KEYWORDS)
                    if proc_in_note:
                        candidate["grounding"] = "procedure_keyword"
                        grounded.append(candidate)
                    else:
                        rejected_count += 1
                        candidate["rejection_reason"] = "no_procedure_context"
                        rejected.append(candidate)
                        logger.debug("Grounding: rejected CPT %s (no procedure context)", code)
                continue

            # Rule 2: EVIDENCE VALIDATION — code must have supporting entity
            supporting_entity = None
            for entity in entity_set:
                # Check if entity appears in code description (semantic match)
                if entity in desc:
                    supporting_entity = entity
                    break
                # Check if description keywords overlap with entity
                entity_words = {w for w in entity.split() if len(w) > 3}
                desc_words = {w for w in desc.split() if len(w) > 3}
                if entity_words and entity_words & desc_words:
                    supporting_entity = entity
                    break
            
            # 🚨 TASK 12: SOFT EVIDENCE FUSION
            # If RAG score is high (semantic confidence), allow without literal grounding
            if not supporting_entity and confidence > 0.80:
                supporting_entity = "semantic_match"
                candidate["grounding"] = "high_conf_semantic"
                candidate["grounding_entity"] = "implicit"

            if not supporting_entity:
                # Fallback: check if code's prefix matches any entity's expected ICD family
                try:
                    try:
                        from services.clinical_rules_config import ENTITY_PREFIX_MAP
                    except ImportError:
                        from services.clinical_rules_config import ENTITY_PREFIX_MAP

                    code_pfx = code.split(".")[0] if "." in code else code[:3]
                    for entity_kw, allowed_pfxs in ENTITY_PREFIX_MAP.items():
                        if entity_kw in note_lower:
                            if any(code_pfx.upper().startswith(p.upper()) for p in allowed_pfxs):
                                supporting_entity = entity_kw
                                break
                except ImportError:
                    pass

            if not supporting_entity:
                rejected_count += 1
                candidate["rejection_reason"] = "no_supporting_entity"
                rejected.append(candidate)
                logger.debug("Grounding: rejected %s '%s' (no supporting entity)", code, desc[:50])
                continue

            # Rule 3: ENTITY-TYPE ALIGNMENT — check entity_class vs code chapter
            entity_class = entity_classes.get(supporting_entity, "diagnosis")
            if entity_class in ClinicalGroundingEngine._TYPE_ALIGNMENT_RULES:
                blocked_chapters = ClinicalGroundingEngine._TYPE_ALIGNMENT_RULES[entity_class]
                code_chapter = code[0].upper() if code else ""
                if code_chapter in blocked_chapters:
                    rejected_count += 1
                    candidate["rejection_reason"] = f"type_alignment_mismatch ({entity_class} vs chapter {code_chapter})"
                    rejected.append(candidate)
                    logger.debug(
                        "Grounding: rejected %s (entity '%s' class='%s' cannot produce chapter '%s')",
                        code, supporting_entity, entity_class, code_chapter,
                    )
                    continue

            # Rule 4: Evidence text check
            evidence_found = supporting_entity in note_lower
            if not evidence_found and len(supporting_entity) > 8:
                 evidence_found = any(w in note_lower for w in supporting_entity.split() if len(w) > 5)

            # Add grounding metadata
            candidate["grounding"] = "entity_confirmed" if evidence_found else "entity_inferred"
            candidate["grounding_entity"] = supporting_entity
            candidate["evidence_found"] = evidence_found
            grounded.append(candidate)

        # 🚨 TASK 12: GROUNDING TRACE
        logger.info(
            "GROUNDING_TRACE: pool=%d -> grounded=%d (rejected=%d)",
            len(rag_candidates), len(grounded), rejected_count
        )

        return {
            "grounded": grounded,
            "rejected": rejected
        }

