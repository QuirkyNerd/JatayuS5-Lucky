"""
api/file_parser.py - Handles incoming file uploads (PDF, DOCX, Image) and extracts plain text.
Optimised for benchmark stability (removing Gemini Vision dependency).
"""
import io
import pdfplumber
import docx
from PIL import Image

from fastapi import UploadFile, HTTPException
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class FileParser:
    @staticmethod
    async def parse_file(file: UploadFile) -> str:
        """
        Extract text from PDF, DOCX, TXT, or Image.
        """
        content = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text
            
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            # OCR is temporarily disabled following the Groq migration
            # as Groq models are primarily text-based reasoning.
            logger.error("OCR attempted but Vision provider (Gemini) has been removed.")
            raise HTTPException(
                status_code=400,
                detail="Image parsing (OCR) is temporarily unavailable during LLM infrastructure migration."
            )
            
        else:
            # Assume plain text
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1", errors="ignore")
