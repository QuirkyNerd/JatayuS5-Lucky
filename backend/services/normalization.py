
import re

class CodeNormalizer:
    @staticmethod
    def normalize(code: str) -> str:
        """
        Canonical internal format: No dots, uppercase.
        Example: 'S72.141A' -> 'S72141A'
        """
        if not code:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', str(code)).upper()

    @staticmethod
    def format_display(code: str) -> str:
        """
        External display format: Dotted if it looks like ICD-10.
        Example: 'S72141A' -> 'S72.141A'
        """
        clean = CodeNormalizer.normalize(code)
        # ICD-10 CM heuristic: 3 chars, dot, remainder
        if len(clean) > 3 and not clean.isdigit():
            # If it already has dots, return as is (normalized first then formatted)
            return f"{clean[:3]}.{clean[3:]}"
        return clean

    @staticmethod
    def is_cpt(code: str) -> bool:
        """True if code looks like a CPT code (5 digits)."""
        clean = CodeNormalizer.normalize(code)
        return clean.isdigit() and len(clean) == 5

    @staticmethod
    def is_icd10(code: str) -> bool:
        """True if code looks like an ICD-10 code (Letter followed by digits)."""
        clean = CodeNormalizer.normalize(code)
        return len(clean) >= 3 and clean[0].isalpha()
