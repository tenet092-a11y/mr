"""
Simplified document router without libmagic dependency.

Provides file validation and format detection using only built-in Python
libraries and file extensions for maximum compatibility.
"""

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import DocumentProcessor, UnsupportedFormatError, FileValidationError
from ..models import ValidationResult, ContentType
from ..config import ProcessingConfig


class SimpleDocumentRouter:
    """
    Simplified document router that avoids external system dependencies.
    
    Uses file extensions and basic validation for format detection
    without requiring libmagic or other system libraries.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._processors: Dict[str, Type[DocumentProcessor]] = {}
        self._format_mappings = self._build_format_mappings()
    
    def register_processor(self, processor_class: Type[DocumentProcessor], formats: List[str]) -> None:
        """Register a processor for specific file formats."""
        for fmt in formats:
            self._processors[fmt.lower()] = processor_class
    
    def get_supported_formats(self) -> List[str]:
        """Get list of all supported file formats."""
        return list(self._processors.keys())
    
    def detect_file_format(self, file_path: str) -> Optional[str]:
        """
        Detect file format using file extension and basic validation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file format (extension without dot) or None if unknown
        """
        path = Path(file_path)
        
        # Method 1: File extension
        extension = path.suffix.lower().lstrip('.')
        known_formats = ['pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a']
        if extension in known_formats:
            # Validate with basic file signature check
            if self._validate_file_signature(file_path, extension):
                return extension
        
        # Method 2: MIME type detection (fallback)
        mime_format = self._detect_by_mime_type(file_path)
        if mime_format and mime_format in known_formats:
            return mime_format
        
        return None
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Comprehensive file validation without external dependencies.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        path = Path(file_path)
        warnings = []
        
        # Check if file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                error_message=f"File does not exist: {file_path}"
            )
        
        # Check if it's a file (not directory)
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                error_message=f"Path is not a file: {file_path}"
            )
        
        # Check file size
        file_size = path.stat().st_size
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error_message=f"File is empty: {file_path}",
                file_size=file_size
            )
        
        if file_size > max_size_bytes:
            return ValidationResult(
                is_valid=False,
                error_message=f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)",
                file_size=file_size
            )
        
        # Detect file format
        detected_format = self.detect_file_format(file_path)
        
        if not detected_format:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported file format: {file_path}",
                file_size=file_size
            )
        
        # Check if format is in supported list
        if detected_format not in self.config.supported_formats:
            return ValidationResult(
                is_valid=False,
                error_message=f"File format '{detected_format}' is not enabled in configuration",
                file_format=detected_format,
                file_size=file_size
            )
        
        return ValidationResult(
            is_valid=True,
            file_format=detected_format,
            file_size=file_size,
            warnings=warnings
        )
    
    def route_document(self, file_path: str) -> DocumentProcessor:
        """
        Route document to appropriate processor.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Instantiated processor for the file type
            
        Raises:
            UnsupportedFormatError: If file format is not supported
            FileValidationError: If file validation fails
        """
        # Validate file first
        validation = self.validate_file(file_path)
        if not validation.is_valid:
            raise FileValidationError(validation.error_message, file_path)
        
        # Get processor class
        processor_class = self._processors.get(validation.file_format)
        if not processor_class:
            raise UnsupportedFormatError(
                f"No processor available for format: {validation.file_format}",
                file_path
            )
        
        # Instantiate and return processor
        return processor_class(self.config)
    
    def _build_format_mappings(self) -> Dict[str, str]:
        """Build MIME type to format mappings."""
        return {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/wave': 'wav',
            'audio/x-wav': 'wav',
            'audio/mp4': 'm4a',
            'audio/x-m4a': 'm4a',
        }
    
    def _detect_by_mime_type(self, file_path: str) -> Optional[str]:
        """Detect format using Python's built-in mimetypes module."""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                return self._format_mappings.get(mime_type)
        except Exception:
            pass
        return None
    
    def _validate_file_signature(self, file_path: str, expected_format: str) -> bool:
        """
        Validate file using basic signature checking.
        
        Args:
            file_path: Path to the file
            expected_format: Expected file format
            
        Returns:
            True if signature matches expected format
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # Basic signature validation
            if expected_format == 'pdf':
                return header.startswith(b'%PDF')
            elif expected_format == 'png':
                return header.startswith(b'\x89PNG\r\n\x1a\n')
            elif expected_format in ['jpg', 'jpeg']:
                return header.startswith(b'\xff\xd8\xff')
            elif expected_format == 'docx':
                # DOCX files are ZIP archives
                return header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06')
            elif expected_format == 'mp3':
                return header.startswith(b'ID3') or header.startswith(b'\xff\xfb')
            elif expected_format == 'wav':
                return header.startswith(b'RIFF') and b'WAVE' in header
            else:
                # For other formats, assume extension is correct
                return True
                
        except Exception:
            # If we can't read the file, assume extension is correct
            return True