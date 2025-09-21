"""
Document processors package.

Provides document processing capabilities for various file formats
including PDF, DOCX, images, and audio files.
"""

from .base import DocumentProcessor, ProcessingError, UnsupportedFormatError, FileValidationError
from .router import DocumentRouter
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .image_processor import ImageProcessor
from .ocr_engine import OCREngine, OCRResult

__all__ = [
    'DocumentProcessor',
    'DocumentRouter',
    'PDFProcessor',
    'DOCXProcessor',
    'ImageProcessor',
    'OCREngine',
    'OCRResult',
    'ProcessingError',
    'UnsupportedFormatError',
    'FileValidationError'
]