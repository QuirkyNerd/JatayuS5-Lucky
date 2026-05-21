
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from config import settings
import utils.logging as _logging

logger = _logging.get_logger(__name__)

class SemanticOntologyValidator:
    """
    Phase 14: SapBERT Semantic Validation & Ontology Precision Engine.
    Disambiguates medical concepts and suppresses same-family hallucinations.
    """
    def __init__(self):
        self.model_name = settings.sapbert_model
        self._model = None
        self._embedding_cache = {} # doc_text -> embedding

    def _load_model(self):
        import os, pickle
        if self._model is None:
            # Task 14: Load Persistent Cache
            cache_path = os.path.join(settings.chroma_persist_dir, "sapbert_cache.pkl")
            if os.path.exists(cache_path):
                try:
                    logger.info("SemanticOntologyValidator: Loading embedding cache...")
                    with open(cache_path, "rb") as f:
                        self._embedding_cache = pickle.load(f)
                    logger.info("SemanticOntologyValidator: Cache loaded (%d terms).", len(self._embedding_cache))
                except Exception as e:
                    logger.error("Failed to load SapBERT cache: %s", e)

            logger.info("SemanticOntologyValidator: Loading SapBERT model (%s)...", self.model_name)
            # SapBERT is specialized for clinical concept alignment.
            # Using CPU for laptop/Docker compatibility.
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def save_cache(self):
        """Task 14: Save SapBERT embedding cache to disk."""
        import os, pickle
        cache_path = os.path.join(settings.chroma_persist_dir, "sapbert_cache.pkl")
        try:
            logger.info("SemanticOntologyValidator: Saving embedding cache (%d terms)...", len(self._embedding_cache))
            with open(cache_path, "wb") as f:
                pickle.dump(self._embedding_cache, f)
        except Exception as e:
            logger.error("Failed to save SapBERT cache: %s", e)

    def validate_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs post-rerank medical semantic validation.
        """
        if not candidates:
            return []

        model = self._load_model()
        
        # 1. Batch Embed Query and non-cached Candidates (Task 7)
        texts_to_encode = [query]
        cache_miss_indices = []
        
        for i, cand in enumerate(candidates):
            doc_text = cand.get("doc", "")
            if doc_text not in self._embedding_cache:
                cache_miss_indices.append(i)
                texts_to_encode.append(doc_text)
        
        # Perform Batch Encoding
        all_embeddings = model.encode(texts_to_encode, convert_to_tensor=True)
        query_embedding = all_embeddings[0]
        
        # Update Cache
        if cache_miss_indices:
            for i, emb_idx in enumerate(range(1, len(all_embeddings))):
                cand_idx = cache_miss_indices[i]
                doc_text = candidates[cand_idx].get("doc", "")
                if len(self._embedding_cache) < 10000:
                    self._embedding_cache[doc_text] = all_embeddings[emb_idx]
        
        validated_results = []
        for cand in candidates:
            doc_text = cand.get("doc", "")
            doc_embedding = self._embedding_cache.get(doc_text)
            
            # 3. Calculate Clinical Concept Similarity (SapBERT Score)
            sap_score = float(util.cos_sim(query_embedding, doc_embedding)[0][0])
            ontology_distance = 1.0 - sap_score
            
            # 4. Same-Family Hallucination Suppression (Task 4)
            # If the clinical similarity is weak (< 0.4), apply a heavy family penalty
            family_penalty = 0.0
            if sap_score < 0.45:
                # This catches cases like "femur" vs "radius" which BGE might cluster
                family_penalty = 0.35
                
            # 5. CPT Precision Refinement (Task 5)
            # Detect operative approach mismatches
            precision_boost = 0.0
            q_lower = query.lower()
            d_lower = doc_text.lower()
            if ("laparoscopic" in q_lower and "open" in d_lower) or ("open" in q_lower and "laparoscopic" in d_lower):
                family_penalty += 0.40 # Severe penalty for approach mismatch
            
            # 6. Adjust Final Score
            # SapBERT score is weighted into the final decision
            cand["sapbert_score"] = round(sap_score, 3)
            cand["ontology_distance"] = round(ontology_distance, 3)
            
            # Penalize the existing rerank score
            current_score = cand.get("score", 0.5)
            new_score = current_score - family_penalty
            if sap_score > 0.85:
                new_score += 0.10 # Boost high-precision ontology matches
                
            cand["score"] = round(new_score, 3)
            # Unified Forensic Trace (Task 1)
            if cand.get("forensic") is None:
                cand["forensic"] = {}
            cand["forensic"].update({
                "sapbert_score": round(sap_score, 3),
                "ontology_penalty": round(family_penalty, 3),
                "ontology_boost": round(precision_boost, 3),
                "sapbert_pass": sap_score > 0.40
            })
            
            validated_results.append(cand)
            
        # Re-sort based on updated ontological scores
        return sorted(validated_results, key=lambda x: x["score"], reverse=True)

_validator_instance = None
def get_ontology_validator() -> SemanticOntologyValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SemanticOntologyValidator()
    return _validator_instance
