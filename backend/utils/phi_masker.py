"""
utils/phi_masker.py - Lightweight PHI Masking utility.
Masks dates, phone numbers, and SSNs.
"""
import re

class PHIMasker:
    # Basic regex patterns for masking
    PHONE_REGEX = re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b')
    SSN_REGEX = re.compile(r'\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b')
    # Simple date catching (MM/DD/YYYY, YYYY-MM-DD, etc.)
    DATE_REGEX = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b')

    @classmethod
    def mask(cls, text: str) -> str:
        """
        Mask sensitive information in clinical notes.
        """
        if not text:
            return text
        
        text = cls.PHONE_REGEX.sub("[PHONE REDACTED]", text)
        text = cls.SSN_REGEX.sub("[SSN REDACTED]", text)
        text = cls.DATE_REGEX.sub("[DATE REDACTED]", text)
        
        return text
