import asyncio
from typing import List, Dict, Any

from services.rag_engine import RAGEngine
from services.llm_service import LLMService
from utils.logging import get_logger

logger = get_logger(__name__)

class CodingLogicAgent:
    """
    Production-grade Medical Coding Assistant.
    Orchestrates high-precision RAG retrieval and grounded LLM synthesis.
    """

    def __init__(self):
        self.rag = RAGEngine()
        self.llm = LLMService()

    async def answer_coding_query(self, query: str, code_type: str = "all") -> Dict[str, Any]:
        """
        High-level API for generating a grounded coding answer.
        """
        logger.info(f"CodingLogicAgent: Answering query '{query}'")

        # Step 1: Retrieval (using existing stable RAGEngine)
        # Note: rag.query returns {documents, metadatas, scores, traces}
        raw_retrieval = await self.rag.query(query, n_results=10, code_type=code_type)
        
        # Flatten results for easier processing
        retrieved_chunks = []
        docs = raw_retrieval.get("documents", [[]])[0]
        metas = raw_retrieval.get("metadatas", [[]])[0]
        scores = raw_retrieval.get("scores", [[]])[0]
        traces = raw_retrieval.get("traces", [[]])[0]

        for doc, meta, score, trace in zip(docs, metas, scores, traces):
            retrieved_chunks.append({
                "doc": doc,
                "meta": meta,
                "score": score,
                "trace": trace
            })

        # Step 2: Synthesis (using LLMService)
        synthesis = await self.llm.synthesize_grounded_answer(query, retrieved_chunks)

        # Step 3: Package final response
        return {
            "answer": synthesis.get("answer", "Insufficient grounded evidence found."),
            "reasoning": synthesis.get("reasoning", ""),
            "retrieved_sources": synthesis.get("retrieved_sources", []),
            "confidence": synthesis.get("confidence", 0.0),
            "query_type": code_type,
            "retrieval_trace": traces[:3] # Include top traces for auditability
        }

    async def process(self, note: str) -> Dict[str, Any]:
        """
        Processes a full clinical note for coding guidance.
        """
        # For now, treat the note as a large query or extract key sections.
        # Given the objective is synthesis, we use the note as context.
        return await self.answer_coding_query(note[:2000]) # Cap for synthesis
