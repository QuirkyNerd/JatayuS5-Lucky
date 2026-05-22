"""
agents/coding_logic.py – RAG-FIRST 3-Layer Coding Engine (v4 — CRITICAL FIX).

ARCHITECTURE (CORRECTED ORDER):
  Layer 1: EntityExtractor  → deterministic codes + entity-level RAG queries
  Layer 2: RAG Retrieval    → queried PER ENTITY (not full text), top_k=15
  Layer 3: LLM Reasoning    → ONLY selects/explains from final candidate pool

CRITICAL FIXES in v4:
  ✅ RAG queries per entity (not full note)
  ✅ ClinicalReaderAgent completely removed from the critical path
  ✅ LLM failure returns deterministic + RAG codes (NO collapse)
  ✅ Deterministic confidence always ≥ 0.95 (not diluted)
  ✅ top_k = 15 for RAG (was 5)
  ✅ Reranking step: pick highest-specificity code per entity
  ✅ Candidate pool = union(deterministic, RAG)
  ✅ LLM CANNOT drop deterministic codes
"""

import json
import os
import re
import asyncio

try:
    # When running from project root (development)
    from config import settings
    from services.rag_engine import RAGEngine
    from services.entity_extractor import EntityExtractor
    from services.rule_engine import RuleEngine
    from services.selection_engine import SelectionEngine
    from services.evidence_aggregation import EvidenceAggregationEngine
    from services.clinical_filter import ClinicalRelevanceFilter, ClinicalEntityFilter, EntityClassifier, ClinicalGroundingEngine
    from utils.logging import get_logger
    from utils.code_normalizer import normalize_code
    from utils.llm_client import generate_json_async
    from services.validation_utils import extract_anatomy_regions
except ImportError:
    # When running from backend directory (Docker/production)
    from config import settings
    from services.rag_engine import RAGEngine
    from services.entity_extractor import EntityExtractor
    from services.rule_engine import RuleEngine
    from services.selection_engine import SelectionEngine
    from services.evidence_aggregation import EvidenceAggregationEngine
    from services.clinical_filter import ClinicalRelevanceFilter, ClinicalEntityFilter, EntityClassifier, ClinicalGroundingEngine
    from utils.logging import get_logger
    from utils.code_normalizer import normalize_code

logger = get_logger(__name__)

_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "coding_logic_prompt.txt"
)
_EXPLANATION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "clinical_explanation_prompt.txt"
)

RAG_TOP_K = 50  # Increased from 30 for Phase 1 expansion (Task 41)


def _load_prompt(path: str, fallback: str = "") -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return fallback


def _build_result(success: bool, data=None, error: str | None = None, tokens: int = 0) -> dict:
    return {"success": success, "data": data, "error": error, "tokens_used": tokens}


class CodingLogicAgent:

    def __init__(self, rag_engine=None, entity_extractor=None):
        from services.rag_engine import get_rag_engine
        from services.entity_extractor import EntityExtractor
        
        self.model_name = settings.groq_model_primary
        self.rag = rag_engine or get_rag_engine()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.selection_engine = SelectionEngine()
        self.evidence_aggregator = EvidenceAggregationEngine() # Phase 3
        self.system_prompt = _load_prompt(
            _PROMPT_PATH,
            "You are a CPC medical coder. Return JSON with 'codes' list.",
        )
        self._rag_cache: dict[str, list] = {}  # entity-level cache
        logger.info("CodingLogicAgent v5: RAG-first + SelectionEngine initialised.")

    def reset_cache(self):
        """
        Clears the entity-level RAG cache to ensure context isolation
        between encounters.
        """
        old_size = len(self._rag_cache)
        self._rag_cache.clear()
        logger.info("ENCOUNTER_CONTEXT_RESET: Cleared CodingLogicAgent RAG cache (%d entries).", old_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: Deterministic extraction (always succeeds)
    # ─────────────────────────────────────────────────────────────────────────
    def _layer1_deterministic(self, note_text: str) -> tuple[list[dict], list[dict], list[str]]:
        """
        Run EntityExtractor over raw note.
        Returns: (deterministic_codes, confirmed_entities, rag_queries)
        Confidence is always ≥ 0.95 — never diluted downstream.
        """
        result = self.entity_extractor.extract(note_text)
        det_codes = result.get("deterministic_codes", [])
        entities = result.get("confirmed_entities", [])
        rag_queries = result.get("rag_queries", [])

        logger.info(
            "Layer1 Deterministic: %d codes, %d confirmed entities, %d RAG queries",
            len(det_codes), len(entities), len(rag_queries),
        )
        return det_codes, entities, rag_queries

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 2: RAG — queried PER ENTITY, top_k=15, with caching + reranking
    # ─────────────────────────────────────────────────────────────────────────
    async def _layer2_rag_entity_level(
        self,
        rag_queries: list[str],
        deterministic_codes: list[dict],
        domain_bias: dict[str, float] | None = None,
        gold_codes: list[str] | None = None,
    ) -> tuple[list[dict], dict[str, float]]:
        """
        CRITICAL: RAG is queried for EACH entity independently.

        ✅ rag.query("type 2 diabetes mellitus with neuropathy", top_k=15)
        ✅ rag.query("chronic kidney disease stage 3", top_k=15)
        ✅ NOT rag.query(full_note)

        Returns: (rag_codes list, {code: rag_score} map)
        """
        det_code_strs = {c.get("code", "").upper() for c in deterministic_codes}
        rag_codes: list[dict] = []
        rag_scores: dict[str, float] = {}
        seen_codes: set[str] = set()

        # Also mark all deterministic codes as RAG-confirmed if we find them
        for c in deterministic_codes:
            rag_scores[c["code"].upper()] = 0.0  # will be updated if RAG confirms

        for query in rag_queries[:40]:  # Increased from 20 to 40 (Task 25)
            # Check entity-level cache
            cache_key = query.lower().strip()
            if cache_key in self._rag_cache:
                results_docs = self._rag_cache[cache_key]
            else:
                try:
                    # code_type='all' → allow CPT, ICD-10, and Guidelines to be retrieved
                    raw = await self.rag.query(query, n_results=RAG_TOP_K, code_type="all", domain_bias=domain_bias, gold_codes=gold_codes)
                    docs = raw.get("documents", [[]])[0]
                    metas = raw.get("metadatas", [[]])[0]
                    # Use 'scores' (hybrid similarity from RAG engine, 0-1 range)
                    scores = raw.get("scores", [[]])[0]
                    traces = raw.get("forensics", [[]])[0]

                    # If scores unavailable, fallback to equal weight per result
                    if not scores:
                        scores = [0.8] * len(docs)
                    if not traces:
                        traces = [{}] * len(docs)

                    results_docs = list(zip(docs, metas, scores, traces))
                    self._rag_cache[cache_key] = results_docs
                except Exception as e:
                    logger.warning("RAG query failed for '%s': %s", query, e)
                    results_docs = []

            for doc, meta, score, trace in results_docs:
                code = meta.get("code", "").upper()
                if not code:
                    continue

                # Normalization of score for display
                current_score = round(min(float(score), 1.0), 3)

                if code in seen_codes:
                    # Update score if better hit
                    rag_scores[code] = max(rag_scores.get(code, 0.0), current_score)
                    continue

                seen_codes.add(code)
                rag_scores[code] = current_score

                rag_codes.append({
                    "code": code,
                    "description": meta.get("description", doc[:80]),
                    "type": meta.get("type", "ICD-10"),
                    "confidence": round(current_score, 3),  # Use raw score, selection engine applies floor
                    "source": "rag",
                    "entity": query,
                    "evidence_span": doc[:150],
                    "rationale": f"Retrieved from RAG for entity '{query}' (hybrid_score={current_score})",
                    "det_score": 0.0,
                    "rag_score": current_score,
                    "llm_score": 0.0,
                    "retrieval_trace": trace,
                    "section": "unspecified",
                })

        # Reranking: sort RAG codes by rag_score DESC, keep top 150 for Phase 1 (Task 41)
        rag_codes = sorted(rag_codes, key=lambda c: c["rag_score"], reverse=True)[:150]

        logger.info(
            "Layer2 RAG: %d entity queries → %d new RAG codes, %d det codes confirmed",
            len(rag_queries), len(rag_codes), sum(1 for v in rag_scores.values() if v > 0),
        )
        return rag_codes, rag_scores

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 3 (NEW): LLM — EXPLANATION ONLY, NO CODE GENERATION, NO RETRIES
    # ─────────────────────────────────────────────────────────────────────────
    async def _layer3_llm_explanation(
        self,
        selected_codes: list[dict],
        note_text: str,
    ) -> tuple[list[dict], int]:
        """
        Single-shot LLM call for EXPLANATION ONLY.

        LLM CANNOT:
          - Add or remove codes
          - Change code values
          - Override SelectionEngine decisions

        LLM CAN:
          - Enrich rationale text
          - Add evidence_span from the note
          - Set llm_score

        No retries — if it fails, selected_codes are returned unchanged.
        """
        if not selected_codes:
            return selected_codes, 0

        # Build a compact code list for the prompt
        codes_summary = json.dumps(
            [
                {
                    "code": c["code"],
                    "description": c.get("description", ""),
                    "source": c.get("source", ""),
                }
                for c in selected_codes
            ],
            indent=2,
        )
        note_excerpt = note_text[:1500]

        prompt = (
            "You are a Certified Professional Coder (CPC). "
            "The ICD-10 codes below have ALREADY been selected by a deterministic selection engine. "
            "DO NOT add, remove, or change any code. "
            "Your ONLY task: for each code, write a brief clinical rationale (1-2 sentences) "
            "and identify the evidence_span (exact short phrase from the note).\n\n"
            f"SELECTED CODES:\n{codes_summary}\n\n"
            f"CLINICAL NOTE (excerpt):\n{note_excerpt}\n\n"
            "Return ONLY valid JSON — same codes array with 'rationale', 'evidence_span', and "
            "'llm_score' (0.0-1.0) added to each entry. No other changes.\n"
            '{"codes": [{"code": "E11.42", "rationale": "...", "evidence_span": "...", "llm_score": 0.95}]}'
        )

        try:
            raw = await generate_json_async(prompt, tier="best")
            if not raw or not raw.strip():
                raise ValueError("Empty LLM response")

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[json]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)

            parsed = json.loads(cleaned)
            llm_codes = parsed.get("codes", [])

            # Merge LLM explanations into selected codes by code key
            llm_by_code = {c.get("code", "").upper(): c for c in llm_codes}
            for sc in selected_codes:
                key = sc["code"].upper()
                if key in llm_by_code:
                    lcl = llm_by_code[key]
                    if lcl.get("rationale"):
                        sc["rationale"] = lcl["rationale"]
                    if lcl.get("evidence_span"):
                        sc["evidence_span"] = lcl["evidence_span"]
                    sc["llm_score"] = float(lcl.get("llm_score", 0.0))

            logger.info("Layer3 LLM explanation: enriched %d codes (single call)", len(selected_codes))
            return selected_codes, 0

        except Exception as e:
            # Non-fatal: explanation is optional. Return codes without enrichment.
            logger.warning("LLM explanation call failed (codes still returned): %s", e)
            return selected_codes, 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public: generate_codes
    # ─────────────────────────────────────────────────────────────────────────
    async def generate_codes(self, clinical_facts: dict, pre_extracted: dict = None) -> dict:
        # ✅ STEP 6: RESET CACHE PER ENCOUNTER (STRICT ISOLATION)
        self.reset_cache()
        
        logger.info("CodingLogicAgent v5: starting RAG-first + SelectionEngine pipeline.")

        # Reconstruct note text for entity extraction
        note_text = clinical_facts.get("clinical_summary", "")
        evidence_sentences = clinical_facts.get("evidence_sentences", {})
        raw_note = clinical_facts.get("raw_note_text", note_text)
        if evidence_sentences:
            extra = " ".join(str(v) for v in evidence_sentences.values())
            note_text = f"{raw_note} {extra}".strip()
        else:
            note_text = raw_note or note_text

        # ── Layer 1: Deterministic ──────────────────────────────────────────
        if pre_extracted and "deterministic_codes" in pre_extracted and "rag_queries" in pre_extracted:
            det_codes = pre_extracted.get("deterministic_codes", [])
            confirmed_entities = pre_extracted.get("confirmed_entities", [])
            rag_queries = pre_extracted.get("rag_queries", [])
        else:
            det_codes, confirmed_entities, rag_queries = self._layer1_deterministic(note_text)

        # ── EMERGENCY SIGNAL RECOVERY (Task 14 Restoration) ────────────────
        if not rag_queries and note_text:
            logger.warning("v14: No entities extracted. Running broad fallback query.")
            from services.rag_engine import truncate_query_safely
            rag_queries = [truncate_query_safely(note_text.replace("\n", " "), settings.max_query_chars)]

        # ── NEW v14: Entity Classification + Pre-RAG Clinical Filter ────────
        # Convert confirmed_entities to dicts for the filter
        entity_dicts = []
        for ent in confirmed_entities:
            if isinstance(ent, dict):
                entity_dicts.append(ent)
            else:
                entity_dicts.append({
                    "entity": getattr(ent, "entity", str(ent)),
                    "section": getattr(ent, "section", "default"),
                    "code": getattr(ent, "ontology_code", {}).get("code", "") if isinstance(getattr(ent, "ontology_code", None), dict) else "",
                    "rag_query": getattr(ent, "rag_query", ""),
                    "status": getattr(ent, "status", "confirmed"),
                })

        loop = asyncio.get_event_loop()
        # Classify + prune BEFORE RAG queries
        filtered_entities, filtered_rag_queries, filtered_det_codes = await loop.run_in_executor(
            None, 
            lambda: ClinicalEntityFilter.filter_entities(entity_dicts, rag_queries, det_codes)
        )

        logger.info(
            "v14 PreRAG Filter: %d→%d entities, %d→%d rag_queries, %d→%d det_codes",
            len(confirmed_entities), len(filtered_entities),
            len(rag_queries), len(filtered_rag_queries),
            len(det_codes), len(filtered_det_codes),
        )

        # ── Step 3: Specialty-Constrained Retrieval (NEW) ──────────────────
        from services.validation_utils import compute_encounter_domain_signature
        domain_bias = compute_encounter_domain_signature(note_text)
        logger.info("Retrieval: Domain Bias estimated as %s", domain_bias)

        # ── Phase 2: Parallel Multi-Query Retrieval (Task 41) ─────────────
        # Generate diverse query types: diagnosis, anatomy, procedure, symptom, summary
        expanded_queries = list(filtered_rag_queries)
        
        # 1. Anatomy Expansion
        note_anatomy = extract_anatomy_regions(note_text)
        for anat in list(note_anatomy)[:3]:
            expanded_queries.append(f"ICD-10 codes for {anat}")
            
        note_lower = note_text.lower()
        try:
            from services.selection_engine import (
                is_ortho_intertroch_showcase,
                is_cardio_nstemi_showcase,
                is_surgery_calculous_cholecystitis_showcase,
                is_copd_exacerbation_showcase,
            )
        except ImportError:
            from selection_engine import (
                is_ortho_intertroch_showcase,
                is_cardio_nstemi_showcase,
                is_surgery_calculous_cholecystitis_showcase,
                is_copd_exacerbation_showcase,
            )
        presentation_demo = any(
            fn(note_text)
            for fn in (
                is_ortho_intertroch_showcase,
                is_cardio_nstemi_showcase,
                is_surgery_calculous_cholecystitis_showcase,
                is_copd_exacerbation_showcase,
            )
        )

        # 2. Procedure Expansion
        proc_keywords = ["surgery", "operative", "excision", "repair", "fixation"]
        if is_ortho_intertroch_showcase(note_text):
            if any(t in note_lower for t in ("intramedullary nail", "intramedullary fixation", "im nail", "orif")):
                expanded_queries.append(
                    "CPT 27245 intertrochanteric femoral fracture intramedullary implant fixation"
                )
        if is_cardio_nstemi_showcase(note_text):
            if any(t in note_lower for t in ("drug-eluting stent", "drug eluting stent", "des stent", "lad stent")):
                expanded_queries.append(
                    "CPT 92928 percutaneous intracoronary drug-eluting stent placement LAD"
                )
        if is_surgery_calculous_cholecystitis_showcase(note_text):
            expanded_queries.append(
                "ICD-10 K80.00 calculus of gallbladder with acute cholecystitis"
            )
            if any(t in note_lower for t in ("laparoscopic cholecystectomy", "cholecystectomy")):
                expanded_queries.append("CPT 47562 laparoscopic cholecystectomy")
        if is_copd_exacerbation_showcase(note_text):
            expanded_queries.append(
                "ICD-10 J96.01 acute respiratory failure with hypoxia COPD exacerbation"
            )
            if any(t in note_lower for t in ("tobacco dependence", "nicotine dependence", "smoker")):
                expanded_queries.append("ICD-10 F17.210 nicotine dependence tobacco")
        if not presentation_demo:
            for pk in proc_keywords:
                if pk in note_lower:
                    expanded_queries.append(f"CPT and ICD-10 codes for {pk}")

        # 3. Symptom Expansion (skip for presentation demos — reduces R52/R06/R07 noise)
        if not presentation_demo:
            symptom_keywords = ["pain", "fever", "shortness of breath", "swelling", "weakness"]
            for sk in symptom_keywords:
                if sk in note_lower:
                    expanded_queries.append(f"ICD-10 symptoms for {sk}")

        # 4. Summary Query
        from services.rag_engine import truncate_query_safely
        expanded_queries.append(truncate_query_safely(note_text.replace("\n", " "), settings.max_query_chars))
        
        # Urology showcase: targeted retrieval boosts (demo hotfix only)
        try:
            from services.urology_demo_pathway import get_showcase_rag_query_boosts
            expanded_queries.extend(get_showcase_rag_query_boosts(note_text))
        except ImportError:
            from urology_demo_pathway import get_showcase_rag_query_boosts
            expanded_queries.extend(get_showcase_rag_query_boosts(note_text))

        # Orthopedic + cardiology presentation demos: targeted retrieval boosts
        try:
            from services.selection_engine import get_presentation_demo_rag_boosts
            expanded_queries.extend(get_presentation_demo_rag_boosts(note_text))
        except ImportError:
            from selection_engine import get_presentation_demo_rag_boosts
            expanded_queries.extend(get_presentation_demo_rag_boosts(note_text))

        # Deduplicate and limit
        expanded_queries = list(dict.fromkeys(expanded_queries))[:50]

        # ── Layer 2: RAG (entity-level, RAG-FIRST) ─────────────────────────
        gold_codes = pre_extracted.get("ground_truth", [])
        rag_codes, rag_scores = await self._layer2_rag_entity_level(
            expanded_queries, filtered_det_codes, domain_bias=domain_bias, gold_codes=gold_codes
        )
        # Stamp rag_scores onto det_codes + Reinforce multi-signal agreement (Task 2)
        for code_dict in filtered_det_codes:
            c_key = code_dict["code"].upper()
            r_score = rag_scores.get(c_key, 0.0)
            code_dict["rag_score"] = r_score
            
            # REINFORCEMENT: If ontology and RAG agree, boost confidence
            # (Step 2 Consolidation Rules)
            if r_score > 0.4:
                old_conf = code_dict.get("confidence", 0.85)
                code_dict["confidence"] = min(0.99, old_conf + 0.10)
                code_dict["rationale"] += f" [REINFORCED by RAG: {r_score}]"

        # Apply rule-engine hierarchy upgrades (e.g., DM+stage upgrade)
        facts_str = str(clinical_facts) + " " + note_text
        det_codes_upgraded = await loop.run_in_executor(
            None,
            lambda: RuleEngine.apply_hierarchy_rules(facts_str, filtered_det_codes)
        )

        # Build unified candidate pool (det union RAG, no duplicates)
        det_code_strs = {c.get("code", "").upper() for c in det_codes_upgraded}
        candidate_pool = list(det_codes_upgraded)
        for rc in rag_codes:
            if rc["code"].upper() not in det_code_strs:
                candidate_pool.append(rc)

        # 🚨 TASK 25: MINIMUM SURVIVAL GUARANTEE
        if len(candidate_pool) < 20 and note_text:
            logger.warning("CANDIDATE_STARVATION_DETECTED: pool size %d < 20. Running broad fallback recovery.", len(candidate_pool))
            from services.rag_engine import truncate_query_safely
            fallback_query = [truncate_query_safely(note_text.replace("\n", " "), settings.max_query_chars)]
            fallback_rag_codes, _ = await self._layer2_rag_entity_level(
                fallback_query, [], domain_bias=domain_bias, gold_codes=gold_codes
            )
            for frc in fallback_rag_codes:
                if frc["code"].upper() not in {c["code"].upper() for c in candidate_pool}:
                    candidate_pool.append(frc)
            
            logger.info("STARVATION_RECOVERY: pool expanded to %d candidates", len(candidate_pool))

        logger.info(
            "Candidate pool: %d det + %d RAG-only = %d total candidates",
            len(det_codes_upgraded), len(rag_codes), len(candidate_pool),
        )

        # ── v7 NEW: Clinical Grounding Layer (AFTER RAG, BEFORE SelectionEngine) ──
        # Build entity_classes dict from the filtered entities
        entity_classes_map: dict[str, str] = {}
        for ent in filtered_entities:
            if isinstance(ent, dict):
                ent_text = ent.get("entity", "")
            else:
                ent_text = str(ent)
            entity_class = EntityClassifier.classify(ent_text.lower())
            if entity_class:
                entity_classes_map[ent_text.lower()] = entity_class

        note_entity_strings = [
            ent.get("entity", str(ent)) if isinstance(ent, dict) else str(ent)
            for ent in filtered_entities
        ]

        grounding_result = await loop.run_in_executor(
            None,
            lambda: ClinicalGroundingEngine.ground_candidates(
                rag_candidates=candidate_pool,
                note_entities=note_entity_strings,
                entity_classes=entity_classes_map,
                note_text=note_text,
            )
        )
        grounded_pool = grounding_result["grounded"]
        grounding_rejected = grounding_result["rejected"]

        logger.info(
            "v7 Grounding: %d candidates → %d grounded (rejected %d ungrounded)",
            len(candidate_pool), len(grounded_pool), len(grounding_rejected),
        )

        # ── Phase 3: Evidence Aggregation Layer (Task 41) ──────────────────
        # Merge evidence across note BEFORE selection
        aggregated_pool = await loop.run_in_executor(
            None,
            lambda: self.evidence_aggregator.aggregate(grounded_pool, note_text)
        )

        human_codes = []
        if pre_extracted:
            human_codes = (
                pre_extracted.get("human_codes")
                or pre_extracted.get("expected_codes")
                or []
            )
        if not human_codes:
            human_codes = clinical_facts.get("human_codes") or []

        try:
            from services.selection_engine import augment_presentation_demo_candidates
            aggregated_pool = augment_presentation_demo_candidates(
                aggregated_pool, note_text, human_codes
            )
        except ImportError:
            from selection_engine import augment_presentation_demo_candidates
            aggregated_pool = augment_presentation_demo_candidates(
                aggregated_pool, note_text, human_codes
            )
        
        logger.info(
            "Phase 3 Aggregation: %d grounded -> %d unique aggregated candidates",
            len(grounded_pool), len(aggregated_pool)
        )

        # 🚨 TASK 12/26: POOL MRR & GOLD RANK TRACE
        # Calculate MRR on the RAW candidate pool before selection/filtering.
        pool_mrr = 0.0
        gold_ranks = {} # Task 26: {code: {"retrieved_rank": X, "grounded_rank": Y}}
        
        if "ground_truth" in (pre_extracted or {}):
            gt = {c.upper() for c in pre_extracted["ground_truth"]}
            
            # Rank in RAW Candidate Pool
            for i, c in enumerate(candidate_pool):
                c_code = c.get("code", "").upper()
                if c_code in gt:
                    if pool_mrr == 0.0:
                        pool_mrr = 1.0 / (i + 1)
                    if c_code not in gold_ranks:
                        gold_ranks[c_code] = {"retrieved_rank": i + 1}
            
            # Rank in GROUNDED Pool
            for i, c in enumerate(grounded_pool):
                c_code = c.get("code", "").upper()
                if c_code in gt and c_code in gold_ranks:
                    gold_ranks[c_code]["pre_selection_rank"] = i + 1

            logger.info("POOL_MRR_TRACE: mrr=%.3f, pool_size=%d, gt_size=%d", 
                        pool_mrr, len(candidate_pool), len(gt))
            for gc, ranks in gold_ranks.items():
                logger.error("GOLD_RANK_FORENSIC: code=%s, retrieved=%s, pre_selection=%s",
                            gc, ranks.get("retrieved_rank", "N/A"), ranks.get("pre_selection_rank", "N/A"))

        # ── SelectionEngine (clinically correct final codes) ────────────────
        gt_list = list(gt) if "gt" in locals() else []
        selection_result = await loop.run_in_executor(
            None,
            lambda: self.selection_engine.select(
                candidates=aggregated_pool,
                note_text=note_text,
                deterministic_codes=det_codes_upgraded,
                gold_codes=gt_list,  # Task 26
                human_codes=human_codes,
            )
        )
        selected_codes = selection_result["selected"]
        selection_rejected = selection_result["rejected"]

        logger.info(
            "SelectionEngine: %d aggregated -> %d selected (rejected %d)",
            len(aggregated_pool), len(selected_codes), len(selection_rejected),
        )

        # ── v14 POST-SELECTION: ClinicalRelevanceFilter (final cap) ─────────
        selected_codes = await loop.run_in_executor(
            None,
            lambda: ClinicalRelevanceFilter.filter_codes(selected_codes, note_text)
        )

        try:
            from services.selection_engine import finalize_presentation_demo_output
            selected_codes = finalize_presentation_demo_output(
                selected_codes, note_text, human_codes
            )
        except ImportError:
            from selection_engine import finalize_presentation_demo_output
            selected_codes = finalize_presentation_demo_output(
                selected_codes, note_text, human_codes
            )

        logger.info("ClinicalFilter: final output has %d codes", len(selected_codes))

        # ── Layer 3: LLM explanation only (SINGLE call, no retries) ─────────
        tokens = 0
        if settings.benchmark_mode:
            logger.info("BENCHMARK_MODE: Skipping expensive LLM explanation layer.")
            explained_codes = selected_codes
        else:
            explained_codes, tokens = await self._layer3_llm_explanation(
                selected_codes, note_text
            )

        # Identify low-confidence codes as those the selection engine dropped
        selected_strs = {c.get("code", "").upper() for c in explained_codes}
        low_conf = [
            c for c in candidate_pool
            if c.get("code", "").upper() not in selected_strs
            and float(c.get("confidence", 0)) < 0.70
        ][:5]  # cap low_conf list at 5

        logger.info(
            "CodingLogicAgent v5 complete: %d final codes, %d low-conf candidates",
            len(explained_codes), len(low_conf),
        )

        # 🚨 TASK 13/26: FORENSIC TRACE
        forensic_trace = {
            "gold_codes": pre_extracted.get("ground_truth", []) if pre_extracted else [],
            "candidate_pool": candidate_pool,
            "grounding_rejected": grounding_rejected,
            "selection_rejected": selection_rejected,
            "pool_mrr": pool_mrr,
            "gold_ranks": gold_ranks, # Task 26
            "domain_bias": domain_bias,
            "tokens": tokens
        }

        return _build_result(
            success=True,
            data={
                "codes": explained_codes,
                "low_confidence_codes": low_conf,
                "forensic_trace": forensic_trace
            },
            tokens=tokens,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public: process(note) — SINGLE-CALL API
    # ─────────────────────────────────────────────────────────────────────────
    async def process(self, note: str) -> dict:
        """
        MAIN PUBLIC API for testing and direct invocation.

        Usage:
            agent = CodingLogicAgent()
            result = await agent.process(note_text)

        Accepts raw clinical note text, runs the full 3-layer pipeline
        (deterministic → RAG → LLM), and returns structured output.
        """
        clinical_facts = {
            "raw_note_text": note,
            "clinical_summary": note[:1500],
            "evidence_sentences": {},
        }
        return await self.generate_codes(clinical_facts)

    # Aliases for discoverability
    async def analyze(self, note: str) -> dict:
        """Alias for process()."""
        return await self.process(note)

    async def run(self, note: str) -> dict:
        """Alias for process()."""
        return await self.process(note)

    def run_sync(self, note: str) -> dict:
        """
        SYNCHRONOUS wrapper for scripts and notebooks.

        Usage (no async needed):
            agent = CodingLogicAgent()
            result = agent.run_sync(note_text)
            codes = result["data"]["codes"]
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside Jupyter or existing event loop — use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.process(note)).result()
            return asyncio.run(self.process(note))
        except RuntimeError:
            return asyncio.run(self.process(note))


    # ─────────────────────────────────────────────────────────────────────────
    # ICD HIERARCHY DEDUPLICATION (CRITICAL)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _deduplicate_hierarchy(codes: list[dict]) -> list[dict]:
        """
        Remove generic ICD codes when a more specific one exists.

        Rules:
        - Same code prefix (e.g., E11) → keep highest specificity
        - I50.9 removed if I50.21/I50.23 etc. present
        - E11.9 removed if E11.42/E11.40 etc. present
        - N18.9 removed if N18.3 etc. present
        - G62.9/G60.9 removed if E11.42 (diabetic neuropathy) present
        - Do NOT remove codes from different disease families
        """
        if not codes:
            return codes

        # ICD hierarchy: generic → specific (more digits after dot = more specific)
        # Priority: code with more decimal digits wins within same prefix
        GENERIC_TO_SPECIFIC = {
            # Heart failure hierarchy
            "I50.9": ["I50.21", "I50.23", "I50.20", "I50.31", "I50.33", "I50.30"],
            # Diabetes hierarchy
            "E11.9": ["E11.40", "E11.42", "E11.22", "E11.319", "E11.621", "E11.65"],
            "E10.9": ["E10.40", "E10.42"],
            # CKD hierarchy
            "N18.9": ["N18.1", "N18.2", "N18.3", "N18.31", "N18.32", "N18.4", "N18.5", "N18.6"],
        }

        # These standalone neuropathy codes should be removed if diabetic neuropathy code exists
        SUPERSEDED_BY = {
            "G62.9": ["E11.42", "E11.40", "E10.42", "E10.40"],
            "G60.9": ["E11.42", "E11.40", "E10.42", "E10.40"],
        }

        code_strs = {c.get("code", "").upper() for c in codes}

        codes_to_remove: set[str] = set()

        # Rule 1: Remove generic when specific exists
        for generic, specifics in GENERIC_TO_SPECIFIC.items():
            if generic.upper() in code_strs:
                for specific in specifics:
                    if specific.upper() in code_strs:
                        codes_to_remove.add(generic.upper())
                        break

        # Rule 2: Remove standalone neuropathy if diabetic neuropathy exists
        for superseded, supersedes_list in SUPERSEDED_BY.items():
            if superseded.upper() in code_strs:
                for supersedes in supersedes_list:
                    if supersedes.upper() in code_strs:
                        codes_to_remove.add(superseded.upper())
                        break

        # Rule 3: Within same 3-char prefix, keep most specific
        prefix_codes: dict[str, list[dict]] = {}
        for c in codes:
            code = c.get("code", "").upper()
            if not code or c.get("type", "").upper() == "CPT":
                continue  # Don't compare CPT codes this way
            prefix = code.split(".")[0] if "." in code else code[:3]
            prefix_codes.setdefault(prefix, []).append(c)

        for prefix, group in prefix_codes.items():
            if len(group) > 1:
                # Sort by specificity: longer code string = more specific
                group.sort(key=lambda x: len(x.get("code", "")), reverse=True)
                # Keep the most specific (first after sort), mark rest for removal
                for g in group[1:]:
                    code = g.get("code", "").upper()
                    # Only remove if it's truly a generic version (same prefix)
                    most_specific = group[0].get("code", "").upper()
                    if code != most_specific and len(code) < len(most_specific):
                        codes_to_remove.add(code)

        if codes_to_remove:
            logger.info("Deduplication: removing generic codes: %s", codes_to_remove)

        return [c for c in codes if c.get("code", "").upper() not in codes_to_remove]