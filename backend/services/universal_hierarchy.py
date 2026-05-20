"""
services/universal_hierarchy.py — UNIVERSAL ICD-10 HIERARCHY ENGINE (v7)

HYBRID MODEL: Seed config rules (PRIMARY) + Structural inference (SECONDARY)

This engine works across ALL ICD-10 chapters (A00-Z99) without hardcoding
any disease-specific logic. It uses the ICD-10 code STRUCTURE to derive:

1. Parent-child relationships (prefix nesting)
2. Ancestor suppression (more specific code suppresses less specific)
3. Cross-prefix suppression (combination code suppresses component codes)
4. Stage/severity override (higher stage suppresses lower stage)

CRITICAL DESIGN PRINCIPLE:
  - SEED rules from hierarchy_config.py fire FIRST (clinically validated)
  - Structural inference fills gaps ONLY for codes not covered by seed rules
  - Dynamic compound detection requires dual entity confirmation in note text
  - NO description-only inference (clinically unsafe)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import seed configs (primary rules)
try:
    from services.clinical_rules_config import CROSS_PREFIX_SUPPRESS, HIERARCHY_SUPPRESSION
except ImportError:
    try:
        from services.clinical_rules_config import CROSS_PREFIX_SUPPRESS, HIERARCHY_SUPPRESSION
    except ImportError:
        CROSS_PREFIX_SUPPRESS = {}
        HIERARCHY_SUPPRESSION = {}


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 STRUCTURAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_BASED_FAMILIES = {
    "N18": "ckd_stage",
    "I50": "hf_severity",
    "K70": "liver_stage",
    "K74": "fibrosis_stage",
    "C77": "lymph_stage",
    "C78": "mets_stage",
    "C79": "mets_stage",
}

_CHAPTER_DOMAINS: dict[str, str] = {
    "A": "infectious",    "B": "infectious",
    "C": "neoplasm",      "D": "neoplasm_blood",
    "E": "endocrine",     "F": "mental",
    "G": "nervous",       "H": "eye_ear",
    "I": "circulatory",   "J": "respiratory",
    "K": "digestive",     "L": "skin",
    "M": "musculoskeletal", "N": "genitourinary",
    "O": "pregnancy",     "P": "perinatal",
    "Q": "congenital",    "R": "symptom",
    "S": "injury",        "T": "injury",
    "U": "special",       "V": "external",
    "W": "external",      "X": "external",
    "Y": "external",      "Z": "factors",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _prefix3(code: str) -> str:
    return code.split(".")[0].upper() if "." in code else code[:3].upper()

def _specificity(code: str) -> int:
    return len(code.replace(".", ""))

def _get_chapter_domain(code: str) -> str:
    if not code:
        return "unknown"
    return _CHAPTER_DOMAINS.get(code[0].upper(), "unknown")

def _extract_numeric_suffix(code: str) -> Optional[int]:
    parts = code.replace(".", "")
    suffix = parts[3:] if len(parts) > 3 else ""
    if suffix and suffix.isdigit():
        return int(suffix)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL HIERARCHY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class UniversalHierarchyEngine:
    """
    Hybrid hierarchy engine: seed config (primary) + structural inference (secondary).
    """

    @staticmethod
    @lru_cache(maxsize=2048)
    def is_ancestor(ancestor: str, descendant: str) -> bool:
        a = ancestor.upper().replace(".", "")
        d = descendant.upper().replace(".", "")
        if a == d:
            return False
        return d.startswith(a) and len(d) > len(a)

    @staticmethod
    def suppress_ancestors(codes: set[str]) -> set[str]:
        """
        ICD-10 Rule: Code to highest specificity.
        If N18.32 exists, suppress N18.3, N18.9.
        Works universally for ANY ICD-10 chapter.
        """
        to_suppress: set[str] = set()
        code_list = sorted(codes, key=lambda c: _specificity(c), reverse=True)

        for i, specific_code in enumerate(code_list):
            for j in range(i + 1, len(code_list)):
                general_code = code_list[j]
                if UniversalHierarchyEngine.is_ancestor(general_code, specific_code):
                    to_suppress.add(general_code)

            pfx3 = _prefix3(specific_code)
            unspec = f"{pfx3}.9"
            if unspec in codes and unspec != specific_code:
                to_suppress.add(unspec)

        return to_suppress

    @staticmethod
    def suppress_by_stage_severity(codes: set[str]) -> set[str]:
        """Suppress lower stage codes when higher stage exists. Universal for registered families."""
        to_suppress: set[str] = set()

        families: dict[str, list[tuple[str, int]]] = {}
        for code in codes:
            pfx3 = _prefix3(code)
            if pfx3 in _STAGE_BASED_FAMILIES:
                family = _STAGE_BASED_FAMILIES[pfx3]
                sev = _extract_numeric_suffix(code)
                if sev is not None:
                    families.setdefault(family, []).append((code, sev))

        for family, members in families.items():
            if len(members) <= 1:
                continue
            members.sort(key=lambda x: x[1], reverse=True)
            best_code, best_sev = members[0]
            for code, sev in members[1:]:
                if sev < best_sev:
                    to_suppress.add(code)
                    logger.debug("UHE: stage suppress '%s' (sev=%d) by '%s' (sev=%d)",
                                 code, sev, best_code, best_sev)

        return to_suppress

    @staticmethod
    def apply_seed_cross_prefix_suppression(codes: set[str]) -> set[str]:
        """Apply clinically validated seed suppressions from hierarchy_config.py. PRIMARY priority."""
        to_suppress: set[str] = set()
        codes_upper = {c.upper() for c in codes}

        for trigger_code, suppress_list in CROSS_PREFIX_SUPPRESS.items():
            if trigger_code.upper() not in codes_upper:
                continue
            for suppress_rule in suppress_list:
                if suppress_rule.startswith("P:"):
                    pfx = suppress_rule[2:].upper()
                    for c in codes:
                        if c.upper().startswith(pfx) and c.upper() != trigger_code.upper():
                            to_suppress.add(c)
                elif suppress_rule.startswith("EXACT:"):
                    exact = suppress_rule[6:].upper()
                    if exact in codes_upper:
                        to_suppress.add(exact)

        for trigger_code, suppress_prefixes in HIERARCHY_SUPPRESSION.items():
            trigger_upper = trigger_code.upper()
            if not any(c.upper().startswith(trigger_upper) for c in codes):
                continue
            for pfx in suppress_prefixes:
                pfx_upper = pfx.upper()
                for c in codes:
                    if c.upper().startswith(pfx_upper) and not c.upper().startswith(trigger_upper):
                        to_suppress.add(c)

        return to_suppress

    @staticmethod
    def infer_structural_cross_prefix(
        codes: set[str],
        code_descriptions: dict[str, str],
        note_entities: list[str],
    ) -> set[str]:
        """
        SECONDARY inference (entity-confirmed only).

        ONLY fires when:
          1. Code NOT covered by seed rules
          2. Both conditions confirmed as entities in the note
          3. Description contains combination pattern ("with", "due to", "in")

        This is NOT description-only guessing — it requires dual entity confirmation.
        """
        to_suppress: set[str] = set()
        seed_covered = {k.upper() for k in CROSS_PREFIX_SUPPRESS} | {k.upper() for k in HIERARCHY_SUPPRESSION}

        for code in codes:
            code_upper = code.upper()
            if code_upper in seed_covered:
                continue

            desc = code_descriptions.get(code, "").lower()
            if not desc:
                continue

            if " with " not in desc and " due to " not in desc and " in " not in desc:
                continue

            pfx3 = _prefix3(code)

            for other_code in codes:
                other_pfx3 = _prefix3(other_code)
                if other_pfx3 == pfx3 or other_code.upper() == code_upper:
                    continue

                other_desc = code_descriptions.get(other_code, "").lower()

                # STRICT: dual entity confirmation required
                code_entity_confirmed = False
                other_entity_confirmed = False
                for entity in note_entities:
                    ent_lower = entity.lower()
                    if ent_lower in desc:
                        code_entity_confirmed = True
                    if ent_lower in other_desc:
                        other_entity_confirmed = True

                if not (code_entity_confirmed and other_entity_confirmed):
                    continue

                if _specificity(code) > _specificity(other_code):
                    other_keywords = {w for w in other_desc.split() if len(w) > 3}
                    desc_keywords = {w for w in desc.split() if len(w) > 3}
                    if other_keywords & desc_keywords:
                        to_suppress.add(other_code)
                        logger.debug("UHE: structural infer '%s' suppresses '%s'", code, other_code)

        return to_suppress

    def get_all_suppressions(
        self,
        codes: set[str],
        code_descriptions: dict[str, str],
        note_entities: list[str],
        protected: set[str],
    ) -> set[str]:
        """
        Master method. Priority order:
        1. Seed cross-prefix (PRIMARY)
        2. Ancestor suppression (structural, universally safe)
        3. Stage/severity (structural, universally safe)
        4. Structural cross-prefix inference (SECONDARY, entity-confirmed only)

        Protected codes are NEVER suppressed.
        """
        all_suppress: set[str] = set()

        seed = self.apply_seed_cross_prefix_suppression(codes)
        all_suppress |= seed
        if seed:
            logger.info("UHE: seed suppression: %d codes: %s", len(seed), seed)

        ancestor = self.suppress_ancestors(codes - all_suppress)
        all_suppress |= ancestor
        if ancestor:
            logger.info("UHE: ancestor suppression: %d codes: %s", len(ancestor), ancestor)

        stage = self.suppress_by_stage_severity(codes - all_suppress)
        all_suppress |= stage
        if stage:
            logger.info("UHE: stage suppression: %d codes: %s", len(stage), stage)

        remaining = codes - all_suppress
        if code_descriptions and note_entities:
            inferred = self.infer_structural_cross_prefix(remaining, code_descriptions, note_entities)
            all_suppress |= inferred
            if inferred:
                logger.info("UHE: structural inference: %d codes: %s", len(inferred), inferred)

        final = all_suppress - protected
        logger.info("UHE: total suppressions=%d (protected saved=%d)", len(final), len(all_suppress - final))
        return final
