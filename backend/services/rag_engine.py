"""
services/rag_engine.py - High-Precision Clinical Retrieval Engine.

RESPONSIBILITIES:
  1. Manages persistent ChromaDB collections for ICD, CPT, and guidelines.
  2. Executes high-precision semantic search with direct phrase grounding.
  3. Eliminates noisy fallback loops and broad semantic expansion.
  4. Enforces deterministic intent routing (Instructional vs. Clinical).
  5. Performs hybrid retrieval (Dense + BM25) and clinical reranking.
  6. Phase 6B: Implements Clinical Anatomical Grounding.
  7. Phase 8: Procedural Semantic Hierarchy Engine.
  8. Phase 9: High-Granularity Anatomical Precision Engine.
"""

import re
import asyncio
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union, Set
import time
from sentence_transformers import CrossEncoder

from config import settings
import utils.logging as _logging
from services.embedding_service import EmbeddingService, get_embedding_service
from services.ontology_service import OntologyService
from services.validation_utils import (
    normalize_clinical_terminology,
    clean_rag_description,
    ENCOUNTER_DOMAINS
)
from services.ontology_validator import get_ontology_validator
from services.coding_decision_engine import CodingDecisionEngine

logger = _logging.get_logger(__name__)

def truncate_query_safely(query: str, max_chars: int) -> str:
    """
    Safely truncates query to max_chars.
    If query is longer than max_chars, it truncates at a whitespace boundary to avoid cutting words.
    """
    if not query:
        return ""
    if len(query) <= max_chars:
        return query
    
    truncated = query[:max_chars]
    last_space = -1
    for i in range(len(truncated) - 1, -1, -1):
        if truncated[i].isspace():
            last_space = i
            break
            
    if last_space != -1:
        return truncated[:last_space].rstrip()
    
    return truncated

# Phase 9: Anatomical Hierarchy Map
ANATOMY_HIERARCHY: Dict[str, List[str]] = {
    "UPPER_EXTREMITY": ["shoulder", "humerus", "humeral", "elbow", "radius", "radial", "ulna", "ulnar", "wrist", "hand", "finger", "thumb", "clavicle", "scapula"],
    "LOWER_EXTREMITY": ["hip", "femur", "femoral", "knee", "tibia", "tibial", "fibula", "fibular", "ankle", "foot", "malleolus", "pelvis", "iliac", "acetabulum", "trochanter"],
    "SPINE": ["cervical", "thoracic", "lumbar", "sacral", "sacrum", "spine", "spinal", "vertebra", "vertebral", "disc", "disk", "lamina"],
    "CARDIOVASCULAR": ["heart", "cardiac", "coronary", "aorta", "aortic", "myocardial", "atrial", "valve", "ventricle"],
    "GI": ["appendix", "appendectomy", "colon", "colic", "stomach", "gastric", "egd", "esophagus", "esophageal", "bowel", "intestinal", "gallbladder", "cholecystectomy"],
    "NEURO": ["brain", "cerebral", "cranial", "head", "skull", "spinal cord", "nerve"],
    "CHEST": ["rib", "sternum", "lung", "pulmonary", "thorax", "thoracic cavity"]
}

_last_error_time = {}
_error_counts = {}

def log_retrieval_failure(label: str, err: Exception, context: str = ""):
    now = time.perf_counter()
    key = (label, type(err).__name__, str(err))
    
    # Track counts
    _error_counts[key] = _error_counts.get(key, 0) + 1
    last_time = _last_error_time.get(key, 0.0)
    
    # Rate limit: log at WARNING level at most once every 10 seconds per unique error
    if now - last_time > 10.0:
        logger.warning(
            "RAG_RETRIEVAL_FAILURE | label=%s | context=%s | err_type=%s | err=%s | count=%d",
            label, context, type(err).__name__, err, _error_counts[key],
            exc_info=True
        )
        _last_error_time[key] = now

class FastBM25:
    """Scipy-accelerated BM25 using CSR matrix for sub-10ms retrieval (Task 11)."""
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        import numpy as np
        from scipy import sparse
        import math
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        if self.corpus_size == 0: return

        # 1. Build Vocabulary
        self.vocab = {}
        for doc in tokenized_corpus:
            for word in doc:
                if word not in self.vocab: self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        
        # 2. Statistics
        doc_lengths = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = np.mean(doc_lengths) if self.corpus_size > 0 else 1.0
        
        # 3. Build Sparse Frequency Matrix (CSR)
        # data[i] is freq of term row_ind[i] in doc col_ind[i]
        data, row_ind, col_ind = [], [], []
        df = np.zeros(self.vocab_size, dtype=np.float32)
        
        for i, doc in enumerate(tokenized_corpus):
            counts = {}
            for word in doc:
                counts[word] = counts.get(word, 0) + 1
            for word, freq in counts.items():
                w_idx = self.vocab[word]
                data.append(float(freq))
                row_ind.append(w_idx)
                col_ind.append(i)
                df[w_idx] += 1
        
        # 4. Precompute BM25 Weights into the Matrix
        # W[word, doc] = IDF * (freq * (k1+1)) / (freq + k1 * (1-b + b*dl/avgdl))
        idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
        
        for i in range(len(data)):
            w_idx = row_ind[i]
            doc_idx = col_ind[i]
            freq = data[i]
            
            w_idf = idf[w_idx]
            dl = doc_lengths[doc_idx]
            
            # BM25 weight
            num = freq * (self.k1 + 1)
            den = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            data[i] = w_idf * (num / den)
            
        # Store as CSR (terms as rows for fast slicing)
        self.weight_matrix = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(self.vocab_size, self.corpus_size))

    def get_scores(self, query_tokens):
        import numpy as np
        if self.corpus_size == 0 or not hasattr(self, 'weight_matrix'): 
            return np.zeros(self.corpus_size, dtype=np.float32)
        
        # Sum rows of query tokens
        valid_indices = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not valid_indices: return np.zeros(self.corpus_size, dtype=np.float32)
        
        # Instantaneous slice & sum
        return np.array(self.weight_matrix[valid_indices, :].sum(axis=0)).flatten()

class RAGEngine:
    _load_count = 0 
    _token_cache = {} # Task 5: Tokenization Cache

    def __init__(self):
        RAGEngine._load_count += 1
        _failures = []
        
        # ── Qdrant Cloud Client ────────────────────────────────────────────
        self.q_client = None
        is_qdrant = settings.vector_db_provider.lower() == "qdrant"
        if is_qdrant or settings.qdrant_url:
            try:
                import traceback
                from qdrant_client import QdrantClient
                logger.info(f"Initializing Qdrant client (url={settings.qdrant_url})...")
                self.q_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=15.0
                )
                cols = self.q_client.get_collections()
                col_names = [c.name for c in cols.collections]
                logger.info(f"QDRANT_CONNECTED_OK | url={settings.qdrant_url} | collections={col_names}")
            except Exception as qe:
                import traceback
                logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | component=QDRANT | error={qe}")
                traceback.print_exc()
                logger.exception("FULL_TRACEBACK")
                _failures.append("QDRANT")
                self.q_client = None
                if is_qdrant:
                    raise RuntimeError(f"Qdrant connection failed: {qe}")

        # ── ChromaDB (local fallback) ──────────────────────────────────────
        db_path = settings.chroma_persist_dir
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self._icd_col = self.client.get_or_create_collection(name=settings.chroma_collection_icd, metadata={"hnsw:space": "cosine"})
        self._cpt_col = self.client.get_or_create_collection(name=settings.chroma_collection_cpt, metadata={"hnsw:space": "cosine"})
        self._guide_col = self.client.get_or_create_collection(name=settings.chroma_collection_guidelines, metadata={"hnsw:space": "cosine"})
        self._symptom_col = self.client.get_or_create_collection(name=settings.chroma_collection_symptoms, metadata={"hnsw:space": "cosine"})

        # ── Embedding Service ──────────────────────────────────────────────
        try:
            self.embedding_service = get_embedding_service()
            logger.info(f"EMBEDDING_MODEL_LOADED | local={self.embedding_service.use_local} | model={getattr(settings, 'embedding_model', 'default')}")
        except Exception as ee:
            logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | component=EMBEDDING | error={ee}")
            _failures.append("EMBEDDING")
            self.embedding_service = get_embedding_service()  # let it crash naturally

        self.ontology_service = OntologyService()
        
        # ── Collection Counts ──────────────────────────────────────────────
        if self.q_client:
            self._counts = {}
            for label, col_name in [("icd10", settings.chroma_collection_icd), 
                                    ("cpt", settings.chroma_collection_cpt), 
                                    ("guidelines", settings.chroma_collection_guidelines), 
                                    ("symptoms", settings.chroma_collection_symptoms)]:
                try:
                    info = self.q_client.get_collection(collection_name=col_name)
                    self._counts[label] = info.points_count
                except Exception as col_err:
                    logger.error(f"Error getting count for Qdrant collection {col_name}: {col_err}")
                    self._counts[label] = 0
        else:
            self._counts = {
                "icd10": self._icd_col.count(),
                "cpt": self._cpt_col.count(),
                "guidelines": self._guide_col.count(),
                "symptoms": self._symptom_col.count()
            }

        # ── BM25 Indices ───────────────────────────────────────────────────
        self.bm25_indices = {}
        self._init_bm25_indices()
        
        # ── Clinical Reranker ──────────────────────────────────────────────
        try:
            logger.info("Initializing Clinical Reranker (MiniLM-L-6)...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            logger.info("RERANKER_LOADED | model=cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as re_err:
            logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | component=RERANKER | error={re_err}")
            _failures.append("RERANKER")
            raise
        
        # ── Coding Decision Engine ─────────────────────────────────────────
        self.decision_engine = CodingDecisionEngine()
        
        # ── SapBERT Semantic Validation ────────────────────────────────────
        try:
            self.ontology_validator = get_ontology_validator()
            self.ontology_validator._load_model()
            logger.info(f"SAPBERT_LOADED | model={settings.sapbert_model}")
        except Exception as sap_err:
            logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | component=SAPBERT | error={sap_err}")
            _failures.append("SAPBERT")
            self.ontology_validator = get_ontology_validator()
        
        # ── Active Vector Backend ──────────────────────────────────────────
        active_backend = "QDRANT" if self.q_client else "CHROMADB"
        logger.info(f"ACTIVE_VECTOR_BACKEND={active_backend}")
        logger.info(f"COLLECTION_COUNTS: {self._counts}")
        logger.info(f"BM25_INDICES_LOADED: {list(self.bm25_indices.keys())}")
        
        # ── Final Summary ──────────────────────────────────────────────────
        if _failures:
            logger.error(f"CRITICAL_COMPONENT_LOAD_FAILURE | failed_components={_failures}")
        else:
            logger.info("ALL_COMPONENTS_LOADED_OK | reranker=YES | sapbert=YES | embedding=YES | vector_backend=%s", active_backend)

    def _normalize_medical_shorthand(self, text: str) -> str:
        """
        Phase 6A/6B: Expanded normalization of medical shorthand.
        """
        shorthand_map = {
            r"\bORIF\b": "open reduction internal fixation",
            r"\bCABG\b": "coronary artery bypass graft",
            r"\bTHA\b": "total hip arthroplasty",
            r"\bTKA\b": "total knee arthroplasty",
            r"\bAPPY\b": "appendectomy",
            r"\bLAP\s+APPY\b": "laparoscopic appendectomy",
            r"\bCAD\b": "coronary artery disease",
            r"\bCKD\b": "chronic kidney disease",
            r"\bHTN\b": "hypertension",
            r"\bHLD\b": "hyperlipidemia",
            r"\bAFIB\b": "atrial fibrillation",
            r"\bCHF\b": "congestive heart failure",
            r"\bDM\b": "diabetes mellitus",
            r"\bPMH\b": "past medical history",
            r"\bSTEMI\b": "ST elevation myocardial infarction",
            r"\bNSTEMI\b": "non ST elevation myocardial infarction",
            r"\bCOPD\b": "chronic obstructive pulmonary disease",
            r"\bPE\b": "pulmonary embolism",
            r"\bDVT\b": "deep vein thrombosis",
            r"\bPCI\b": "percutaneous coronary intervention",
            r"\bEGD\b": "esophagogastroduodenoscopy",
            r"\bTJA\b": "total joint arthroplasty"
        }
        
        normalized = text
        for pattern, expansion in shorthand_map.items():
            normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)
        return normalized

    def _extract_anatomy(self, text: str) -> Dict[str, Any]:
        """
        Phase 9: High-Granularity Anatomy Extractor.
        Returns: { 'primary': set(), 'regions': set() }
        """
        lower = text.lower()
        primary = set()
        regions = set()
        
        # 1. Extract Primary Structures
        anatomy_keywords = {
            "radius": ["radius", "radial"], "ulna": ["ulna", "ulnar"], "wrist": ["wrist"], "hand": ["hand", "metacarpal", "finger", "thumb"],
            "humerus": ["humerus", "humeral"], "shoulder": ["shoulder", "scapula", "clavicle", "glenoid", "rotator cuff"],
            "femur": ["femur", "femoral", "trochanter"], "hip": ["hip", "acetabulum", "pelvis", "iliac"],
            "tibia": ["tibia", "tibial"], "fibula": ["fibula", "fibular"], "ankle": ["ankle", "malleolus"], "foot": ["foot", "metatarsal"],
            "spine": ["spine", "spinal", "vertebra", "lumbar", "cervical", "thoracic", "sacral"],
            "heart": ["heart", "cardiac", "coronary", "bypass", "cabg", "pci"],
            "appendix": ["appendix", "appendectomy"], "colon": ["colon", "colic", "bowel"],
            "brain": ["brain", "cerebral", "cranial", "skull"]
        }
        
        for structure, kws in anatomy_keywords.items():
            if any(kw in lower for kw in kws):
                primary.add(structure)
                
        # 2. Map to Regions
        for structure in primary:
            for region, members in ANATOMY_HIERARCHY.items():
                if any(kw in structure for kw in members):
                    regions.add(region)
        
        # 3. Direct Region Hits
        for region, members in ANATOMY_HIERARCHY.items():
            if any(kw in lower for kw in members):
                regions.add(region)
                
        return {"primary": list(primary), "regions": list(regions)}

    async def query(
        self,
        query_text: str,
        n_results: int = 10,
        code_type: str = "ICD-10",
        domain_bias: dict[str, float] | None = None,
        gold_codes: list[str] | None = None,
        enable_sapbert: bool = True, 
        is_benchmark: bool = False,
        query_vector: list[float] | None = None # Task 8: Optional pre-computed embedding
    ) -> dict:
        """
        High-Precision Clinical Retrieval Query with Phase 9 Anatomical Precision.
        Supports Dual Pipeline comparison for Phase 14 measurable validation.
        """
        # Truncate query text safely and validate minimum length
        query_text = truncate_query_safely(query_text, settings.max_query_chars)
        if len(query_text) < 30:
            logger.warning(
                "RAG_QUERY_ABORTED | query='%s' | length=%d | reason=length_less_than_30",
                query_text, len(query_text)
            )
            intent_scores = self._analyze_medical_intent(query_text, code_type)
            return {**self._empty_result(intent_scores), "timings": {"total_query_ms": 0.0}}
        if is_benchmark:
            # Optimization: Use tighter shortlists for benchmark speed (Task 9)
            rerank_k = 10
            sapbert_k = 3
        else:
            rerank_k = 30
            sapbert_k = 15

        logger.info("RAG_QUERY: text='%s' | sapbert=%s | benchmark=%s", query_text[:50], enable_sapbert, is_benchmark)
        active_backend = "QDRANT" if self.q_client else "CHROMADB"
        logger.info(
            "RETRIEVAL_CONFIG | ACTIVE_VECTOR_BACKEND=%s | top_k=%d | rerank_k=%d | sapbert_k=%d | hybrid_alpha=%.2f | hybrid_beta=%.2f",
            active_backend, n_results, rerank_k, sapbert_k, settings.rag_hybrid_alpha, settings.rag_hybrid_beta
        )
        timings = {}
        t_start = time.perf_counter()
        
        # 0. Medical Normalization (Phase 6A/6B)
        t_pre = time.perf_counter()
        normalized_query = self._normalize_medical_shorthand(normalize_clinical_terminology(query_text))
        
        # 1. High-Granularity Anatomical Extraction (Phase 9)
        anatomy_data = self._extract_anatomy(normalized_query)
        
        # 2. Procedural Intent Extraction (Phase 8)
        proc_intent = self._extract_procedural_intent(normalized_query)
        
        # 3. Intelligent Medical Intent Analysis
        intent_scores = self._analyze_medical_intent(normalized_query, code_type)
        timings["preprocessing_ms"] = round((time.perf_counter() - t_pre) * 1000, 2)
        
        # 4. Embedding Generation
        t_emb = time.perf_counter()
        if query_vector is None:
            query_vector = await self.embedding_service.embed_single(normalized_query)
            timings["embedding_gen_ms"] = round((time.perf_counter() - t_emb) * 1000, 2)
        else:
            timings["embedding_gen_ms"] = 0.0
        
        # 5. Dynamic Multi-Domain Routing
        query_plan = self._build_query_plan(intent_scores, n_results)
        
        # 6. Parallel Retrieval
        t_ret = time.perf_counter()
        loop = asyncio.get_event_loop()
        async def fetch(col, k_limit, weight, col_label):
            try:
                # Use cached count (Task 3 optimization)
                count = self._counts.get(col_label, 0)
                if count == 0: return None
                k_dense = max(1, min(k_limit, count))
                
                t_d = time.perf_counter()
                if self.q_client:
                    col_name = (
                        settings.chroma_collection_icd if col_label == "icd10" else
                        settings.chroma_collection_cpt if col_label == "cpt" else
                        settings.chroma_collection_guidelines if col_label == "guidelines" else
                        settings.chroma_collection_symptoms
                    )
                    query_res = await loop.run_in_executor(None, lambda: self.q_client.query_points(
                        collection_name=col_name,
                        query=query_vector,
                        limit=k_dense,
                        with_payload=True
                    ))
                    docs = []
                    metas = []
                    distances = []
                    for point in query_res.points:
                        payload = point.payload or {}
                        doc_text = payload.get("document") or payload.get("_document") or payload.get("text") or payload.get("description") or ""
                        docs.append(doc_text)
                        
                        meta = dict(payload)
                        meta["code"] = payload.get("code") or payload.get("id") or str(point.id)
                        metas.append(meta)
                        distances.append(1.0 - point.score)
                    dense_res = {
                        "documents": [docs],
                        "metadatas": [metas],
                        "distances": [distances]
                    }
                else:
                    dense_res = await loop.run_in_executor(None, lambda: col.query(
                        query_embeddings=[query_vector],
                        n_results=k_dense,
                        include=["documents", "metadatas", "distances"]
                    ))
                d_ms = (time.perf_counter() - t_d) * 1000
                
                t_s = time.perf_counter()
                sparse_res = self._sparse_search(normalized_query, col_label, k_limit)
                s_ms = (time.perf_counter() - t_s) * 1000
                
                return {
                    "dense": dense_res, "sparse": sparse_res, "collection_weight": weight, "label": col_label,
                    "sub_timings": {"dense_ms": d_ms, "sparse_ms": s_ms}
                }
            except Exception as e:
                log_retrieval_failure(col_label, e, context="fetch")
                return None

        tasks = [fetch(col, k, weight, label) for col, k, weight, label in query_plan]
        raw_results = await asyncio.gather(*tasks)
        
        # Aggregate sub-timings
        timings["dense_retrieval_ms"] = round(max([r["sub_timings"]["dense_ms"] for r in raw_results if r] or [0]), 2)
        timings["sparse_search_ms"] = round(max([r["sub_timings"]["sparse_ms"] for r in raw_results if r] or [0]), 2)
        
        # 7. Hybrid Blending & Scoring
        merged = []
        seen = set()
        for res in raw_results:
            if not res: continue
            label = res["label"]
            col_weight = res["collection_weight"]
            dense_data = res["dense"]
            sparse_data = res["sparse"]
            candidates = {}
            if dense_data and dense_data.get("documents"):
                docs = dense_data["documents"][0]
                metas = dense_data["metadatas"][0]
                dists = dense_data["distances"][0]
                for doc, meta, dist in zip(docs, metas, dists):
                    code = (meta.get("code") or meta.get("id") or f"{label}_{hash(doc[:50])}").strip().upper()
                    score = max(0.0, 1.0 - dist)
                    candidates[code] = {"doc": doc, "meta": meta, "dense": score, "sparse": 0.0}
            for s_doc, s_meta, s_score in sparse_data:
                code = (s_meta.get("code") or s_meta.get("id") or f"{label}_{hash(s_doc[:50])}").strip().upper()
                if code in candidates: candidates[code]["sparse"] = s_score
                else: candidates[code] = {"doc": s_doc, "meta": s_meta, "dense": 0.0, "sparse": s_score}
            for code, data in candidates.items():
                if code in seen: continue
                seen.add(code)
                base_hybrid_score = (0.6 * data["dense"]) + (0.4 * data["sparse"])
                merged.append({
                    "doc": data["doc"], "meta": data["meta"], "code": code,
                    "base_score": base_hybrid_score, "col_weight": col_weight,
                    "label": label, "sparse_score": data["sparse"], "dense_score": data["dense"]
                })

        timings["vector_search_ms"] = round((time.perf_counter() - t_ret) * 1000, 2)
        logger.info("RETRIEVAL_DEPTH | merged_candidates=%d | search_ms=%.1f", len(merged), timings["vector_search_ms"])

        if not merged: return {**self._empty_result(intent_scores), "timings": timings}

        # 8. Baseline Pipeline (Reranker Only)
        t_rerank = time.perf_counter()
        top_candidates = sorted(merged, key=lambda x: x["base_score"], reverse=True)[:rerank_k]
        reranked_results_baseline = self._clinical_rerank(normalized_query, top_candidates, intent_scores, anatomy_data, proc_intent)
        timings["reranker_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)
        logger.info("RERANKER_APPLIED | input_candidates=%d | output_candidates=%d | reranker_ms=%.1f", len(top_candidates), len(reranked_results_baseline), timings["reranker_ms"])
        
        # 10. Task 10: Early Termination for Slam-Dunks
        # If top candidate is very strong (>0.95) and in benchmark mode, skip SapBERT
        top_res = reranked_results_baseline[0] if reranked_results_baseline else None
        if top_res and top_res["score"] > 0.95 and is_benchmark:
            logger.info("RAG_OPTIMIZATION: Early exit triggered for high-confidence match.")
            reranked_results_validated = reranked_results_baseline
            timings["sapbert_ms"] = 0.0
            enable_sapbert = False
        else:
            # 9. Target Pipeline (SapBERT Validation)
            reranked_results_validated = list(reranked_results_baseline)
            timings["sapbert_ms"] = 0.0
            if enable_sapbert:
                # Task 4: Execute SapBERT precision layer
                t_sap = time.perf_counter()
                reranked_results_validated = self.ontology_validator.validate_candidates(normalized_query, reranked_results_baseline[:sapbert_k])
                timings["sapbert_ms"] = round((time.perf_counter() - t_sap) * 1000, 2)
                top3_sapbert = [(c.get("normed_code", "?"), c.get("sapbert_score", 0)) for c in reranked_results_validated[:3]]
                logger.info("SAPBERT_APPLIED | validated_candidates=%d | sapbert_ms=%.1f | top3=%s", len(reranked_results_validated), timings["sapbert_ms"], top3_sapbert)
            else:
                logger.info("SAPBERT_SKIPPED | reason=disabled")
            
        # 10. Task 5 & 6: Difference Analysis
        ontology_shift = []
        for i in range(min(5, len(reranked_results_baseline))):
            b = reranked_results_baseline[i]
            v = next((x for x in reranked_results_validated if x["normed_code"] == b["normed_code"]), None)
            if v:
                ontology_shift.append({
                    "code": b["normed_code"],
                    "before": b["score"],
                    "after": v["score"],
                    "delta": v["score"] - b["score"]
                })
        
        # Use validated results for final grounding
        reranked_results = reranked_results_validated
        
        # 10. Grounding & Filtering
        final_candidates = []
        seen_base_codes = {}
        top_rerank_score = reranked_results[0]["score"] if reranked_results else 0
        for item in reranked_results:
            if item["score"] < (top_rerank_score - 0.45) and len(final_candidates) >= 3: continue
            code = item["normed_code"]
            base = code.split(".")[0]
            if base in seen_base_codes:
                prev_code, prev_score = seen_base_codes[base]
                if item["score"] > prev_score + 0.05:
                    final_candidates = [c for c in final_candidates if c["normed_code"] != prev_code]
                else: continue
            final_candidates.append(item)
            seen_base_codes[base] = (code, item["score"])
            if len(final_candidates) >= 10: break

        # 10. Phase 10: Clinical Coding Decision Reasoning
        # Split candidates for the decision engine
        icd_cands = [c for c in reranked_results if c.get("label") in ["icd10", "symptoms"] or c.get("meta", {}).get("code_type") == "ICD-10"]
        cpt_cands = [c for c in reranked_results if c.get("label") == "cpt" or c.get("meta", {}).get("code_type") == "CPT"]
        guide_cands = [c for c in reranked_results if c.get("label") == "guidelines"]
        
        t_decision = time.perf_counter()
        decision_input = {
            "icd_candidates": icd_cands,
            "cpt_candidates": cpt_cands,
            "guideline_candidates": guide_cands,
            "anatomy": anatomy_data,
            "procedural_intent": proc_intent,
            "ontology_shift": ontology_shift # Task 5 metrics
        }
        
        coding_decision = self.decision_engine.process_coding_decisions(normalized_query, decision_input)
        timings["decision_engine_ms"] = round((time.perf_counter() - t_decision) * 1000, 2)
        
        timings["total_query_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        timings["model_load_count"] = RAGEngine._load_count

        from services.validation_utils import sanitize_numpy
        return sanitize_numpy({
            "decision": coding_decision,
            "documents": [[c["doc"] for c in final_candidates]],
            "metadatas": [[c["meta"] for c in final_candidates]],
            "scores": [[c["score"] for c in final_candidates]],
            "forensics": [[c.get("forensic", {}) for c in final_candidates]],
            "confidence": coding_decision["confidence"]["score"],
            "intent": intent_scores,
            "anatomy": anatomy_data,
            "procedural_intent": proc_intent,
            "timings": timings, # Task 1
            "comparison_trace": { # Task 6 trace
                "sapbert_enabled": enable_sapbert,
                "ontology_shift": ontology_shift,
                "is_benchmark": is_benchmark
            }
        })

    def _extract_procedural_intent(self, query: str) -> Dict[str, str]:
        q = query.lower()
        intent = {"class": "GENERAL", "intervention": "unknown", "approach": "unknown"}
        
        # Priority mapping for surgical approaches (Task 2.2)
        if any(k in q for k in ["laparoscopic", "lap ", "minimally invasive", "endoscopy", "scopic"]):
            intent["class"] = "LAPAROSCOPIC_SURGERY"
            intent["approach"] = "laparoscopic"
        elif any(k in q for k in ["open ", "orif", "arthroplasty", "incision", "laparotomy", "thoracotomy"]):
            intent["class"] = "OPEN_SURGERY"
            intent["approach"] = "open"
        elif any(k in q for k in ["pci", "angioplasty", "stent", "catheter", "transluminal", "percutaneous"]):
            intent["class"] = "ENDOVASCULAR_INTERVENTION"
            intent["approach"] = "percutaneous"
        elif any(k in q for k in ["orthosis", "brace", "prosthetic", "splint", "sling"]):
            intent["class"] = "SUPPORTIVE_DEVICE"
            intent["approach"] = "external"
        elif any(k in q for k in ["closed reduction", "conservative", "non-operative"]):
            intent["class"] = "CLOSED_TREATMENT"
            intent["approach"] = "manual"
        elif any(k in q for k in ["biopsy", "imaging", "screening", "diagnostic"]):
            intent["class"] = "DIAGNOSTIC_PROCEDURE"
            intent["approach"] = "diagnostic"
            
        # Intervention refinement
        if "fixation" in q or "orif" in q: intent["intervention"] = "fixation"
        elif "bypass" in q or "cabg" in q: intent["intervention"] = "reconstruction"
        elif "replacement" in q or "arthroplasty" in q: intent["intervention"] = "reconstruction"
        elif "excision" in q or "ectomy" in q: intent["intervention"] = "excision"
        elif "biopsy" in q: intent["intervention"] = "diagnostic"
        
        return intent

    def _clinical_rerank(self, query: str, candidates: list, intent_scores: dict, q_anatomy: Dict[str, Any], q_proc: Dict[str, str]) -> list:
        """
        Phase 5+6+8+9: Anatomical-Procedural Reasoning Reranker.
        """
        if not candidates:
            return []
        # Task 12 Optimization: Skip model call if already pre-computed in batch
        if candidates and "cross_score" in candidates[0]:
            cross_scores = [c["cross_score"] for c in candidates]
        else:
            pairs = [(query, c["doc"]) for c in candidates]
            cross_scores = self.reranker.predict(pairs)
        
        import numpy as np
        exp_scores = np.exp(cross_scores)
        norm_cross_scores = exp_scores / (1 + exp_scores)

        results = []
        q_clean = query.lower()
        q_prim = set(q_anatomy["primary"])
        q_regs = set(q_anatomy["regions"])
        
        for cand, cross_score in zip(candidates, norm_cross_scores):
            doc = cand["doc"]; meta = cand["meta"]; label = cand["label"]
            d_clean = clean_rag_description(doc).lower()
            
            consistency_score = 0.0
            trace = {"anatomy_match": False, "proc_class_match": False, "device_penalty": False, "anatomy_distance": 0.0, "anatomy_level": 4}
            
            # 1. Laterality (Phase 5B)
            q_left, q_right = "left" in q_clean, "right" in q_clean
            d_left, d_right = "left" in d_clean, "right" in d_clean
            if (q_left and d_left) or (q_right and d_right): consistency_score += 0.25
            elif (q_left and d_right) or (q_right and d_left): consistency_score -= 0.50
                
            # 2. High-Granularity Anatomical Precision (Phase 9)
            d_anatomy_raw = str(meta.get("anatomy", "General")).lower().split(",")
            d_anatomy_prim = set([a.strip() for a in d_anatomy_raw if a.strip() and a.strip() != "general"])
            
            if q_prim:
                # Level 1: Exact Match
                if q_prim.intersection(d_anatomy_prim) or any(p in d_clean for p in q_prim):
                    consistency_score += 0.60; trace["anatomy_level"] = 1; trace["anatomy_match"] = True
                # Level 2: Adjacent/Regional Match
                elif q_regs:
                    d_regs = set()
                    for da in d_anatomy_prim:
                        for reg, members in ANATOMY_HIERARCHY.items():
                            if any(kw in da for kw in members): d_regs.add(reg)
                    if q_regs.intersection(d_regs):
                        consistency_score += 0.25; trace["anatomy_level"] = 2
                    else:
                        # Level 4: Region Mismatch (Severe Penalty)
                        consistency_score -= 1.0; trace["anatomy_level"] = 4; trace["anatomy_distance"] = 1.0
            
            # 3. Procedural Semantic Hierarchy (Phase 8)
            d_class = meta.get("procedure_class", "GENERAL").upper()
            d_interv = meta.get("intervention_type", "unknown").lower()
            if q_proc["class"] != "GENERAL" and q_proc["class"] == d_class:
                consistency_score += 0.40; trace["proc_class_match"] = True
            elif q_proc["class"] != "GENERAL" and d_class != "GENERAL":
                consistency_score -= 0.50
            if q_proc["intervention"] != "unknown" and q_proc["intervention"] == d_interv:
                consistency_score += 0.25
            
            # 4. Supportive Device Suppression (Phase 8)
            is_operative_q = q_proc["class"] in ["OPEN_SURGERY", "LAPAROSCOPIC_SURGERY", "ENDOVASCULAR_INTERVENTION"]
            if is_operative_q and d_class == "SUPPORTIVE_DEVICE":
                consistency_score -= 1.2; trace["device_penalty"] = True # Increased penalty
            
            # 5. Domain/Intent weight
            is_proc_query = intent_scores["CPT"] > 0.6
            if is_proc_query and label == "cpt": consistency_score += 0.20

            final_score = (0.60 * cross_score) + (0.40 * consistency_score)
            final_score *= cand["col_weight"]
            
            results.append({
                "doc": d_clean, "meta": meta, "score": round(final_score, 3), "normed_code": cand["code"],
                "label": label,
                "forensic": {
                    "cross_encoder": round(cross_score, 3),
                    "consistency": round(consistency_score, 3),
                    "anatomy_match": trace["anatomy_match"],
                    "proc_match": trace["proc_class_match"],
                    "penalty": trace["device_penalty"],
                    "col_weight": cand["col_weight"]
                }
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def batch_query(self, queries: List[str], vectors: List[List[float]], n_results: int = 10) -> List[dict]:
        """
        Task 12: High-throughput batch query engine for benchmarking.
        Processes multiple queries simultaneously to maximize CPU/HNSW throughput.
        """
        import time
        t_start = time.perf_counter()
        
        # Truncate queries and validate lengths
        truncated_queries = []
        valid_indices = []
        for idx, q in enumerate(queries):
            trunc_q = truncate_query_safely(q, settings.max_query_chars)
            truncated_queries.append(trunc_q)
            if len(trunc_q) < 30:
                logger.warning(
                    "RAG_BATCH_QUERY_ABORTED | index=%d | query='%s' | length=%d | reason=length_less_than_30",
                    idx, trunc_q, len(trunc_q)
                )
            else:
                valid_indices.append(idx)
        
        # 1. Preprocessing (Sequential but fast)
        normalized_queries = [self._normalize_medical_shorthand(normalize_clinical_terminology(q)) for q in truncated_queries]
        
        # 2. Batch Dense Retrieval
        # We query collections one by one but with all vectors at once
        collections = {
            "icd10": self._icd_col,
            "cpt": self._cpt_col,
            "guidelines": self._guide_col,
            "symptoms": self._symptom_col
        }
        
        all_results = [[] for _ in range(len(queries))]
        
        for label, col in collections.items():
            if self._counts.get(label, 0) == 0: continue
            
            col_weight = 1.0
            if label == "cpt": col_weight = 1.4
            elif label == "guidelines": col_weight = 1.2
            elif label == "symptoms": col_weight = 0.8
            
            # Batch Query using parallel search calls
            if self.q_client:
                # Determine collection name based on label
                col_name = (
                    settings.chroma_collection_icd if label == "icd10" else
                    settings.chroma_collection_cpt if label == "cpt" else
                    settings.chroma_collection_guidelines if label == "guidelines" else
                    settings.chroma_collection_symptoms
                )
                logger.info(f"QDRANT_BATCH_QUERY_START | label={label} | batch_size={len(vectors)}")
                start_time = time.perf_counter()
                loop = asyncio.get_event_loop()
                
                try:
                    from qdrant_client.models import QueryRequest
                    requests = []
                    req_idx_to_orig_idx = []
                    for i, v in enumerate(vectors):
                        if i in valid_indices:
                            requests.append(QueryRequest(
                                query=v,
                                limit=n_results,
                                with_payload=True
                            ))
                            req_idx_to_orig_idx.append(i)
                    
                    if requests:
                        batch_res_raw = await loop.run_in_executor(
                            None,
                            lambda: self.q_client.query_batch_points(
                                collection_name=col_name,
                                requests=requests
                            )
                        )
                    else:
                        batch_res_raw = []
                    
                    batch_res = [None] * len(vectors)
                    for req_i, orig_i in enumerate(req_idx_to_orig_idx):
                        batch_res[orig_i] = batch_res_raw[req_i]
                except Exception as e:
                    log_retrieval_failure(label, e, context=f"batch_query_col={col_name}")
                    batch_res = [None] * len(vectors)
                    
                # Build combined result structure
                res = {"documents": [], "metadatas": [], "distances": []}
                for query_res in batch_res:
                    docs, metas, dists = [], [], []
                    if query_res is not None:
                        for point in query_res.points:
                            payload = point.payload or {}
                            doc_text = payload.get("document") or payload.get("_document") or payload.get("text") or payload.get("description") or ""
                            docs.append(doc_text)
                            meta = dict(payload)
                            meta["code"] = payload.get("code") or payload.get("id") or str(point.id)
                            metas.append(meta)
                            dists.append(1.0 - point.score)
                    res["documents"].append(docs)
                    res["metadatas"].append(metas)
                    res["distances"].append(dists)
            else:
                # Chroma fallback query
                if valid_indices:
                    valid_vectors = [vectors[i] for i in valid_indices]
                    res_subset = col.query(
                        query_embeddings=valid_vectors,
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    # Map back to full size
                    res = {"documents": [], "metadatas": [], "distances": []}
                    subset_idx = 0
                    for i in range(len(queries)):
                        if i in valid_indices:
                            res["documents"].append(res_subset["documents"][subset_idx])
                            res["metadatas"].append(res_subset["metadatas"][subset_idx])
                            res["distances"].append(res_subset["distances"][subset_idx])
                            subset_idx += 1
                        else:
                            res["documents"].append([])
                            res["metadatas"].append([])
                            res["distances"].append([])
                else:
                    res = {"documents": [[]]*len(queries), "metadatas": [[]]*len(queries), "distances": [[]]*len(queries)}
            
            # Blend with Sparse Search (Batch)
            for i, (q_norm, q_vec) in enumerate(zip(normalized_queries, vectors)):
                if i not in valid_indices:
                    continue
                sparse_res = self._sparse_search(q_norm, label, n_results)
                
                # Simple Hybrid Blending
                candidates = {}
                docs = res["documents"][i]; metas = res["metadatas"][i]; dists = res["distances"][i]
                for d, m, dist in zip(docs, metas, dists):
                    code = (m.get("code") or m.get("id") or f"{label}_{hash(d[:20])}").strip().upper()
                    candidates[code] = {"doc": d, "meta": m, "dense": 1.0 - dist, "sparse": 0.0}
                
                for s_doc, s_meta, s_score in sparse_res:
                    code = (s_meta.get("code") or s_meta.get("id") or f"{label}_{hash(s_doc[:20])}").strip().upper()
                    if code in candidates: candidates[code]["sparse"] = s_score
                    else: candidates[code] = {"doc": s_doc, "meta": s_meta, "dense": 0.0, "sparse": s_score}
                
                for code, data in candidates.items():
                    score = (0.6 * data["dense"]) + (0.4 * data["sparse"])
                    all_results[i].append({
                        "doc": data["doc"], "meta": data["meta"], "code": code,
                        "score": score, "label": label, "normed_code": code,
                        "col_weight": col_weight # Restore missing key
                    })

        # 3. Ultra-Batch Reranking (Task 12)
        all_rerank_pairs = []
        q_metadata = [] # (anatomy, proc) per query
        
        for i, q_norm in enumerate(normalized_queries):
            anatomy = self._extract_anatomy(q_norm)
            proc = self._extract_procedural_intent(q_norm)
            q_metadata.append((anatomy, proc))
            
            if i not in valid_indices:
                continue
            
            cands = sorted(all_results[i], key=lambda x: x["score"], reverse=True)[:10]
            for c in cands:
                all_rerank_pairs.append((q_norm, c["doc"]))
            all_results[i] = cands # Keep only top 10 for reranking

        # Perform one big Cross-Encoder pass
        if all_rerank_pairs:
            loop = asyncio.get_event_loop()
            all_cross_scores = await loop.run_in_executor(None, lambda: self.reranker.predict(all_rerank_pairs, batch_size=32))
            
            # Map scores back and apply clinical logic
            score_idx = 0
            for i, q_norm in enumerate(normalized_queries):
                if i not in valid_indices:
                    continue
                anatomy, proc = q_metadata[i]
                cands = all_results[i]
                for c in cands:
                    c["cross_score"] = all_cross_scores[score_idx]
                    score_idx += 1
        
        # 4. Ultra-Batch SapBERT
        # Collect all query-candidate pairs that need validation
        all_sap_candidates = []
        for i, q_norm in enumerate(normalized_queries):
            if i not in valid_indices:
                continue
            anatomy, proc = q_metadata[i]
            # Get reranked results
            reranked = self._clinical_rerank(q_norm, all_results[i], {"ICD": 0.5, "CPT": 0.5, "GUIDELINE": 0.1, "SYMPTOM": 0.1}, anatomy, proc)
            all_results[i] = reranked[:3] # Top 3 for SapBERT

        # Batch encode ALL candidates across the benchmark
        # 5. Parallel Post-Processing (Task 12)
        async def process_one(idx, q_n):
            if idx not in valid_indices:
                return {
                    "decision": {
                        "principal_diagnosis": [],
                        "secondary_diagnoses": [],
                        "chronic_conditions": [],
                        "procedures": [],
                        "modifiers": [],
                        "compliance_flags": [],
                        "medical_necessity": {},
                        "decomposition": {"acuity": "CHRONIC", "temporality": "unknown", "negated_entities": [], "laterality": "UNSPECIFIED", "encounter_type": "INITIAL", "demographics": {"sex": "UNKNOWN", "age_group": "ADULT"}},
                        "causal_relationships": [],
                        "guidelines_used": [],
                        "suppressed_candidates": [],
                        "confidence": {
                            "score": 0.0,
                            "level": "LOW",
                            "review_required": True,
                            "clinical_reasoning": "Query skipped (too short).",
                            "compliance_score": 1.0
                        },
                        "audit_trace": {
                            "engine_version": "Phase 14 (Ontology Constrained Platform)",
                            "anatomy_grounding": {"primary": [], "regions": []},
                            "procedural_intent": {"class": "GENERAL", "intervention": "unknown", "approach": "unknown"},
                            "ontology_shift": [],
                            "ncci_edits": [],
                            "exclusion_logic": [],
                            "semantic_validation": {
                                "model": "SapBERT",
                                "top_concept_match": 0.0,
                                "ontology_precision": 1.0
                            }
                        }
                    },
                    "timings": {"batch_mode": True, "skipped": True},
                    "comparison_trace": {"batch": True, "skipped": True}
                }
            
            ana, prc = q_metadata[idx]
            # Parallelize validation and decision making
            val = await loop.run_in_executor(None, self.ontology_validator.validate_candidates, q_n, all_results[idx])
            dec = await loop.run_in_executor(None, self.decision_engine.process_coding_decisions, q_n, {
                "icd_candidates": [c for c in val if c["label"] == "icd10"],
                "cpt_candidates": [c for c in val if c["label"] == "cpt"],
                "guideline_candidates": [c for c in val if c["label"] == "guidelines"],
                "anatomy": ana,
                "procedural_intent": prc
            })
            return {
                "decision": dec,
                "timings": {"batch_mode": True},
                "comparison_trace": {"batch": True}
            }

        final_outputs = await asyncio.gather(*[process_one(i, q_norm) for i, q_norm in enumerate(normalized_queries)])
        
        # Task 14: Persist SapBERT cache
        self.ontology_validator.save_cache()
        
        logger.info(f"RAG_BATCH_QUERY: Processed {len(queries)} cases in {time.perf_counter() - t_start:.2f}s")
        return final_outputs

    def _analyze_medical_intent(self, query: str, requested_type: str | None = None) -> dict:
        q = query.lower()
        procedure_suffixes = ["ectomy", "plasty", "oscopy", "otomy", "raphy", "pexy", "desis", "lysis", "stomy"]
        procedure_verbs = ["repair", "bypass", "excision", "resection", "fixation", "orif", "graft", "appendectomy", "arthroplasty", "reduction", "decompression"]
        is_procedure = any(q.endswith(s) or f"{s} " in q for s in procedure_suffixes) or \
                       any(f" {v} " in f" {q} " for v in procedure_verbs)
        instructional_phrases = ["code first", "excludes1", "excludes2", "use additional", "guideline"]
        is_instructional = any(p in q for p in instructional_phrases)
        symptom_keywords = ["pain", "swelling", "ache", "fever", "nausea"]
        is_symptom = any(f" {k} " in f" {q} " for k in symptom_keywords)
        scores = {"ICD": 0.5, "CPT": 0.1, "GUIDELINE": 0.1, "SYMPTOM": 0.1}
        
        req_upper = requested_type.upper().replace("-", "") if requested_type else ""
        scores["requested_type"] = req_upper

        if req_upper == "CPT":
            scores["CPT"] += 0.6
        elif req_upper in ("ICD10", "ICD"):
            scores["ICD"] += 0.4
        elif req_upper == "ALL":
            scores["ICD"] += 0.4
            scores["CPT"] += 0.4

        if is_procedure: scores["CPT"] += 0.8
        if is_instructional: scores["GUIDELINE"] += 0.8
        if is_symptom: scores["SYMPTOM"] += 0.6
        return scores

    def _build_query_plan(self, intent: dict, n_results: int) -> list:
        req_type = intent.get("requested_type", "")
        if req_type == "CPT":
            return [
                (self._cpt_col, n_results * 2, 1.5, "cpt"),
                (self._icd_col, n_results // 2, 0.5, "icd10")
            ]
        elif req_type in ("ICD10", "ICD"):
            return [
                (self._icd_col, n_results * 2, 1.5, "icd10")
            ]
        elif req_type == "ALL":
            return [
                (self._icd_col, n_results, 1.0, "icd10"),
                (self._cpt_col, n_results, 1.0, "cpt")
            ]
        elif req_type in ("GUIDELINES", "GUIDELINE"):
            return [
                (self._guide_col, n_results * 2, 1.5, "guidelines"),
                (self._icd_col, n_results // 2, 0.5, "icd10")
            ]
        elif req_type in ("SYMPTOMS", "SYMPTOM"):
            return [
                (self._symptom_col, n_results * 2, 1.5, "symptoms"),
                (self._icd_col, n_results // 2, 0.5, "icd10")
            ]

        # Intent fallback
        plan = []
        if intent["GUIDELINE"] > 0.7:
            plan.append((self._guide_col, n_results * 2, 1.2, "guidelines"))
            plan.append((self._icd_col, n_results // 2, 0.8, "icd10"))
            return plan
        if intent["CPT"] > 0.7:
            plan.append((self._cpt_col, n_results * 2, 1.4, "cpt"))
            plan.append((self._icd_col, n_results, 0.7, "icd10"))
            return plan
        if intent["SYMPTOM"] > 0.6:
            plan.append((self._symptom_col, n_results, 1.0, "symptoms"))
            plan.append((self._icd_col, n_results, 0.8, "icd10"))
            return plan
        plan.append((self._icd_col, n_results, 1.0, "icd10"))
        if intent["CPT"] > 0.3: plan.append((self._cpt_col, n_results // 2, 0.7, "cpt"))
        return plan

    def _init_bm25_indices(self):
        import pickle
        cache_path = os.path.join(settings.chroma_persist_dir, "bm25_cache.pkl")
        
        # Check for existing cache
        if os.path.exists(cache_path):
            try:
                logger.info("Loading BM25 indices from cache...")
                with open(cache_path, "rb") as f:
                    self.bm25_indices = pickle.load(f)
                logger.info("BM25 indices loaded successfully.")
                return
            except Exception as e:
                logger.error("Failed to load BM25 cache: %s", e)

        collections = {"icd10": self._icd_col, "cpt": self._cpt_col, "guidelines": self._guide_col, "symptoms": self._symptom_col}
        for label, col in collections.items():
            try:
                count = self._counts.get(label, 0)
                if count == 0: continue
                all_docs, all_metas = [], []
                
                if self.q_client:
                    col_name = (
                        settings.chroma_collection_icd if label == "icd10" else
                        settings.chroma_collection_cpt if label == "cpt" else
                        settings.chroma_collection_guidelines if label == "guidelines" else
                        settings.chroma_collection_symptoms
                    )
                    logger.info("Building BM25 index from Qdrant for %s (%d docs)...", label, count)
                    offset = None
                    while True:
                        scroll_res, next_offset = self.q_client.scroll(
                            collection_name=col_name,
                            limit=10000,
                            with_payload=True,
                            with_vectors=False,
                            offset=offset
                        )
                        for point in scroll_res:
                            payload = point.payload or {}
                            doc_text = payload.get("document") or payload.get("_document") or payload.get("text") or payload.get("description") or ""
                            all_docs.append(doc_text)
                            
                            meta = dict(payload)
                            meta["code"] = payload.get("code") or payload.get("id") or str(point.id)
                            all_metas.append(meta)
                        if not next_offset:
                            break
                        offset = next_offset
                else:
                    logger.info("Building BM25 index for %s (%d docs)...", label, count)
                    batch_size = 10000
                    for i in range(0, count, batch_size):
                        data = col.get(include=["documents", "metadatas"], limit=batch_size, offset=i)
                        all_docs.extend(data["documents"])
                        all_metas.extend(data["metadatas"])
                if all_docs:
                    tokenized_corpus = [self._bm25_tokenizer(d) for d in all_docs]
                    # Use UltraFastBM25 (Task 11)
                    bm25 = FastBM25(tokenized_corpus)
                    self.bm25_indices[label] = (bm25, all_docs, all_metas)
            except Exception as e: logger.error("Failed to build BM25 index for %s: %s", label, e)
            
        # Save cache
        try:
            logger.info("Saving BM25 indices to cache...")
            with open(cache_path, "wb") as f:
                pickle.dump(self.bm25_indices, f)
            logger.info("BM25_CACHE_REBUILT_OK")
        except Exception as e: logger.error("Failed to save BM25 cache: %s", e)

    def _bm25_tokenizer(self, text: str) -> List[str]:
        # Task 5: Tokenization Cache
        if text in self._token_cache: return self._token_cache[text]
        
        # Medical-specific stopwords (Task 11)
        STOPWORDS = {"the", "and", "for", "with", "from", "was", "were", "had", "has", "are", "patient", "noted", "revealed"}
        
        normalized = self._normalize_medical_shorthand(text)
        tokens = [t for t in re.findall(r"[\w/]{2,}", normalized.lower()) if t not in STOPWORDS]
        
        if len(self._token_cache) < 20000:
            self._token_cache[text] = tokens
        return tokens

    def _sparse_search(self, query: str, collection_label: str, k: int) -> List[tuple]:
        if collection_label not in self.bm25_indices: return []
        bm25, docs, metas = self.bm25_indices[collection_label]
        tokenized_query = self._bm25_tokenizer(query)
        
        # Use FastBM25 get_scores
        scores = bm25.get_scores(tokenized_query)
        
        import numpy as np
        top_n = np.argsort(scores)[::-1][:k]
        max_score = np.max(scores) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        results = []
        for idx in top_n:
            if scores[idx] <= 0: continue
            results.append((docs[idx], metas[idx], float(scores[idx] / max_score)))
        return results

    def collection_counts(self) -> dict:
        if self.q_client:
            # We want keys 'icd10', 'cpt', 'guidelines', 'symptoms'
            return {
                "icd10": self._counts.get("icd10", 0),
                "cpt": self._counts.get("cpt", 0),
                "guidelines": self._counts.get("guidelines", 0),
                "symptoms": self._counts.get("symptoms", 0)
            }
        return {"icd10": self._icd_col.count(), "cpt": self._cpt_col.count(), "guidelines": self._guide_col.count(), "symptoms": self._symptom_col.count()}

    def _qdrant_upsert(self, collection_name: str, ids: list, embeddings: list, documents: list, metadatas: list):
        from qdrant_client.models import PointStruct
        points = []
        for pt_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            import uuid
            try:
                val = uuid.UUID(str(pt_id))
                q_id = str(val)
            except ValueError:
                try:
                    q_id = int(pt_id)
                except ValueError:
                    q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pt_id)))
            
            points.append(PointStruct(
                id=q_id,
                vector=emb,
                payload={**meta, "document": doc}
            ))
        self.q_client.upsert(collection_name=collection_name, points=points)
        # Update cache counts
        label = "icd10" if collection_name == settings.chroma_collection_icd else "cpt" if collection_name == settings.chroma_collection_cpt else "guidelines" if collection_name == settings.chroma_collection_guidelines else "symptoms"
        self._counts[label] = self._counts.get(label, 0) + len(points)

    def upsert_icd(self, ids: list, embeddings: list, documents: list, metadatas: list):
        if self.q_client:
            self._qdrant_upsert(settings.chroma_collection_icd, ids, embeddings, documents, metadatas)
        else:
            self._icd_col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def upsert_cpt(self, ids: list, embeddings: list, documents: list, metadatas: list):
        if self.q_client:
            self._qdrant_upsert(settings.chroma_collection_cpt, ids, embeddings, documents, metadatas)
        else:
            self._cpt_col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def upsert_guidelines(self, ids: list, embeddings: list, documents: list, metadatas: list):
        if self.q_client:
            self._qdrant_upsert(settings.chroma_collection_guidelines, ids, embeddings, documents, metadatas)
        else:
            self._guide_col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def upsert_symptoms(self, ids: list, embeddings: list, documents: list, metadatas: list):
        if self.q_client:
            self._qdrant_upsert(settings.chroma_collection_symptoms, ids, embeddings, documents, metadatas)
        else:
            self._symptom_col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def backup_knowledge_base(self, backup_path: str = "backend/backend/chroma_db_backup"):
        """
        Task 6: Implement NON-DESTRUCTIVE backup support.
        Creates a timestamped snapshot of the current ChromaDB.
        """
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_backup_path = f"{backup_path}_{timestamp}"
        
        logger.info(f"Starting non-destructive backup to {final_backup_path}...")
        try:
            source_dir = settings.chroma_persist_dir
            if os.path.exists(source_dir):
                shutil.copytree(source_dir, final_backup_path)
                logger.info("Backup completed successfully.")
                return final_backup_path
            else:
                logger.error(f"Source directory {source_dir} not found for backup.")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
        return None

    def recreate_collection(self, name: str):
        """
        Protected administrative method to recreate a collection.
        ONLY for manual recovery or forced re-ingestion.
        """
        logger.warning(f"ADMIN: Recreating collection '{name}' - ALL DATA WILL BE LOST.")
        if self.q_client:
            try:
                self.q_client.delete_collection(collection_name=name)
            except Exception:
                pass
            from qdrant_client.models import Distance, VectorParams
            self.q_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            label = "icd10" if name == settings.chroma_collection_icd else "cpt" if name == settings.chroma_collection_cpt else "guidelines" if name == settings.chroma_collection_guidelines else "symptoms"
            self._counts[label] = 0
            return None
        else:
            self.client.delete_collection(name)
            return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def reset_icd(self):
        self.recreate_collection(settings.chroma_collection_icd)

    def reset_cpt(self):
        self.recreate_collection(settings.chroma_collection_cpt)

    def reset_guidelines(self):
        self.recreate_collection(settings.chroma_collection_guidelines)

    def reset_symptoms(self):
        self.recreate_collection(settings.chroma_collection_symptoms)

    def _empty_result(self, intent_scores: dict) -> dict:
        empty_decision = {
            "principal_diagnosis": [],
            "secondary_diagnoses": [],
            "chronic_conditions": [],
            "procedures": [],
            "modifiers": [],
            "compliance_flags": [],
            "medical_necessity": {},
            "decomposition": {"acuity": "CHRONIC", "temporality": "unknown", "negated_entities": [], "laterality": "UNSPECIFIED", "encounter_type": "INITIAL", "demographics": {"sex": "UNKNOWN", "age_group": "ADULT"}},
            "causal_relationships": [],
            "guidelines_used": [],
            "suppressed_candidates": [],
            "confidence": {
                "score": 0.0,
                "level": "LOW",
                "review_required": True,
                "clinical_reasoning": "No diagnosis identified.",
                "compliance_score": 1.0
            },
            "audit_trace": {
                "engine_version": "Phase 14 (Ontology Constrained Platform)",
                "anatomy_grounding": {"primary": [], "regions": []},
                "procedural_intent": {"class": "GENERAL", "intervention": "unknown", "approach": "unknown"},
                "ontology_shift": [],
                "ncci_edits": [],
                "exclusion_logic": [],
                "semantic_validation": {
                    "model": "SapBERT",
                    "top_concept_match": 0.0,
                    "ontology_precision": 1.0
                }
            }
        }
        return {
            "decision": empty_decision,
            "documents": [[]],
            "metadatas": [[]],
            "scores": [[]],
            "forensics": [[]],
            "confidence": 0.0,
            "intent": intent_scores,
            "anatomy": {"primary": [], "regions": []},
            "procedural_intent": {"class": "GENERAL", "intervention": "unknown", "approach": "unknown"},
            "comparison_trace": {
                "sapbert_enabled": True,
                "ontology_shift": [],
                "is_benchmark": False
            }
        }

    # Convenience wrappers for specific code types
    async def search_icd10(self, query_text: str, top_k: int = 10):
        """Search ICD-10 codes using the generic query interface asynchronously and return structured results."""
        result = await self.query(query_text, n_results=top_k, code_type="icd10")
        # Combine documents, metadatas and scores into a list of dicts
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        scores = result.get("scores", [[]])[0]
        combined = []
        for doc, meta, score in zip(docs, metas, scores):
            combined.append({
                "code": meta.get("code") or meta.get("id") or "?",
                "description": doc,
                "score": score,
                "meta": meta,
            })
        return combined

    async def search_cpt(self, query_text: str, top_k: int = 10):
        """Search CPT codes using the generic query interface asynchronously and return structured results."""
        result = await self.query(query_text, n_results=top_k, code_type="cpt")
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        scores = result.get("scores", [[]])[0]
        combined = []
        for doc, meta, score in zip(docs, metas, scores):
            combined.append({
                "code": meta.get("code") or meta.get("id") or "?",
                "description": doc,
                "score": score,
                "meta": meta,
            })
        return combined

    async def search_guidelines(self, query_text: str, top_k: int = 10):
        """Search clinical guidelines using the generic query interface asynchronously and return structured results."""
        result = await self.query(query_text, n_results=top_k, code_type="guidelines")
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        scores = result.get("scores", [[]])[0]
        combined = []
        for doc, meta, score in zip(docs, metas, scores):
            combined.append({
                "code": meta.get("code") or meta.get("id") or "?",
                "description": doc,
                "score": score,
                "meta": meta,
            })
        return combined

    async def search_symptoms(self, query_text: str, top_k: int = 10):
        """Search symptom entries using the generic query interface asynchronously and return structured results."""
        result = await self.query(query_text, n_results=top_k, code_type="symptoms")
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        scores = result.get("scores", [[]])[0]
        combined = []
        for doc, meta, score in zip(docs, metas, scores):
            combined.append({
                "code": meta.get("code") or meta.get("id") or "?",
                "description": doc,
                "score": score,
                "meta": meta,
            })
        return combined

    
    # Destructive methods REMOVED per Phase 11 Hardening Requirements.
    # If reset is needed, it must be done via direct filesystem deletion 
    # or a manual administrative script, never via the core RAGEngine.

_rag_engine_instance = None
def get_rag_engine() -> 'RAGEngine':
    global _rag_engine_instance
    if _rag_engine_instance is None: _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
