"""
utils/phi_encryptor.py – Fernet-based PHI encryption for database writes.

Encrypts note_text before persisting to the database.
Decrypts on read. Falls back to plaintext if key is not configured (dev mode).

Usage:
    from utils.phi_encryptor import PHIEncryptor
    encrypted = PHIEncryptor.encrypt("clinical note text")
    original  = PHIEncryptor.decrypt(encrypted)
"""

import base64
import logging

from config import settings

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False
    logger.warning("cryptography package not installed – PHI encryption disabled.")


def _get_fernet():
    """Return a Fernet instance if a key is configured, else None."""
    if not _FERNET_AVAILABLE:
        return None
    key = settings.phi_encryption_key.strip()
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except Exception as exc:
        logger.error("Invalid PHI_ENCRYPTION_KEY: %s – encryption disabled.", exc)
        return None


class PHIEncryptor:
    """
    Symmetric encryption for clinical note text at rest.

    • If PHI_ENCRYPTION_KEY is set → encrypts/decrypts using Fernet (AES-128-CBC + HMAC).
    • If key is absent or cryptography is not installed → no-op passthrough (dev mode).
    """

    @staticmethod
    def encrypt(plain_text: str) -> str:
        """Encrypt text for DB storage. Returns ciphertext prefixed with 'ENC:' or plain text."""
        fernet = _get_fernet()
        if fernet is None:
            return plain_text
        try:
            token = fernet.encrypt(plain_text.encode("utf-8"))
            return "ENC:" + token.decode("utf-8")
        except Exception as exc:
            logger.error("PHI encryption failed: %s – storing plaintext.", exc)
            return plain_text

    @staticmethod
    def decrypt(stored_text: str) -> str:
        """Decrypt stored text. Handles both encrypted ('ENC:' prefix) and legacy plaintext."""
        if not stored_text.startswith("ENC:"):
            return stored_text  # Legacy plaintext row – return as-is
        fernet = _get_fernet()
        if fernet is None:
            logger.warning("PHI_ENCRYPTION_KEY not set – cannot decrypt stored PHI.")
            return stored_text
        try:
            token = stored_text[4:].encode("utf-8")
            return fernet.decrypt(token).decode("utf-8")
        except Exception as exc:
            logger.error("PHI decryption failed: %s", exc)
            return stored_text  # Safe fallback – return as-is
