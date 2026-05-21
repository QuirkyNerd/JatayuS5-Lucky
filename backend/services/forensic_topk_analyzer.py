# -*- coding: utf-8 -*-
"""
forensic_topk_analyzer.py – Helper utility to inspect the top‑K retrieval results
produced by the RAGEngine. It is primarily used during development and debugging to
verify that the forensic metadata (score components, boosts, and optional logging)
behaves as expected.

The script defines a single public function ``analyze_top_k`` which accepts the list
of ``results`` returned by ``RAGEngine._clinical_rerank`` and an integer ``k``.
It returns a JSON‑serialisable list containing the top‑k entries together with the
relevant forensic fields (when ``FORENSIC_LOGGING`` is enabled).  If forensic data
is omitted, the function synthesises a minimal view so that downstream consumers
do not raise ``KeyError``.

Typical usage::

    from services.forensic_topk_analyzer import analyze_top_k
    top_k = analyze_top_k(rerank_results, k=10)
    print(json.dumps(top_k, indent=2))

The helper is deliberately lightweight and has no external dependencies beyond the
standard library.
"""
import os
from typing import List, Dict, Any


def _safe_forensic(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable forensic payload.

    If the ``forensic`` key is ``None`` (the default when ``FORENSIC_LOGGING`` is
    disabled) we construct a minimal placeholder that mirrors the structure used
    when logging is enabled.  This makes downstream analysis code tolerant to the
    environment variable.
    """
    forensic = entry.get("forensic")
    if forensic is None:
        # Create a placeholder with the same keys but ``None`` values.
        placeholder_keys = [
            "original_score",
            "anatomy_bonus",
            "laterality_bonus",
            "encounter_bonus",
            "fracture_subtype_bonus",
            "specificity_depth_bonus",
            "family_penalty",
            "icd_prefix_bonus",
            "consistency_base",
            "final_score",
            "col_weight",
        ]
        forensic = {k: None for k in placeholder_keys}
    return forensic


def analyze_top_k(results: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    """Return the top‑k results enriched with deterministic forensic data.

    Parameters
    ----------
    results: List[Dict]
        The list produced by ``RAGEngine._clinical_rerank`` (already sorted by
        ``score`` in descending order).
    k: int, optional
        Number of top entries to return. Defaults to ``10``.

    Returns
    -------
    List[Dict]
        A list of dictionaries each containing the most relevant fields for
        debugging:
        ``code``, ``label``, ``score``, ``forensic`` (guaranteed to be a dict).
    """
    top = results[:k]
    enriched = []
    for entry in top:
        enriched.append(
            {
                "code": entry.get("normed_code"),
                "label": entry.get("label"),
                "score": entry.get("score"),
                "forensic": _safe_forensic(entry),
            }
        )
    return enriched

# When the module is executed directly, provide a tiny demo that prints JSON.
if __name__ == "__main__":
    import json, sys
    try:
        # Expect a JSON array on stdin.
        payload = json.load(sys.stdin)
    except Exception:
        print("Provide a JSON array of RAGEngine results on stdin.")
        sys.exit(1)
    result = analyze_top_k(payload, k=int(os.getenv("TOP_K", "10")))
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
