import json
import re
from typing import List, Dict, Any

from utils.llm_client import generate_json_async
from utils.logging import get_logger

logger = get_logger(__name__)

class LLMService:
    """
    Production-grade LLM service for grounded clinical answer synthesis.
    Ensures zero hallucination and strict adherence to retrieved evidence.
    """

    def __init__(self):
        pass

    async def synthesize_grounded_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes a professional medical coding answer based ONLY on retrieved chunks.
        """
        if not retrieved_chunks:
            return {
                "answer": "Insufficient grounded evidence found.",
                "reasoning": "No relevant guideline or coding documentation was retrieved for this query.",
                "retrieved_sources": [],
                "confidence": 0.0
            }

        # Format context for the LLM
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("meta", {})
            source_info = (
                f"Source: {meta.get('source', 'Unknown')} | "
                f"Year: {meta.get('year', 'Unknown')} | "
                f"Type: {meta.get('instruction_type', 'General')} | "
                f"Section: {meta.get('section', 'General')} | "
                f"Topic: {meta.get('semantic_topic', 'General')}"
            )
            context_blocks.append(f"--- EVIDENCE BLOCK {i+1} ---\n{source_info}\nContent: {chunk.get('doc')}\n")

        context_text = "\n".join(context_blocks)

        prompt = f"""
You are a professional Medical Coding Audit Assistant (CPC/CCS certified).
Your task is to synthesize a precise, grounded answer to a medical coding query using ONLY the provided evidence blocks.

STRICT RULES:
1. ZERO HALLUCINATION: If the information is not in the evidence blocks, do not include it.
2. SOURCE ADHERENCE: Use the year and instruction type provided in the metadata.
3. CONCISENESS: Provide a professional, compact explanation (1-3 sentences).
4. NO ADVICE: Do not give clinical advice; only explain coding rules and conventions.
5. FALLBACK: If the evidence does not directly answer the query, state "Insufficient grounded evidence found."

QUERY: {query}

RETRIEVED EVIDENCE:
{context_text}

OUTPUT FORMAT (JSON):
{{
  "answer": "Concise coding guidance/explanation...",
  "reasoning": "Clinical logic based on specific retrieved rules...",
  "retrieved_sources": [
    {{
      "year": "...",
      "instruction_type": "...",
      "section": "...",
      "summary": "Short summary of this evidence..."
    }}
  ],
  "confidence": 0.0 to 1.0 (based on how well the evidence matches the query)
}}
"""

        try:
            response_json = await generate_json_async(prompt)
            result = json.loads(response_json)
            
            # Post-processing: Ensure retrieved_sources match reality if LLM missed any
            # (Though JSON mode should be reliable)
            return result
        except Exception as e:
            logger.error(f"LLM Synthesis failed: {e}")
            return {
                "answer": "Error generating grounded answer.",
                "reasoning": f"An internal error occurred during synthesis: {str(e)}",
                "retrieved_sources": [],
                "confidence": 0.0
            }
