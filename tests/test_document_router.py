"""
Unit tests for DocumentRouter class.

Tests file validation, format detection, and processor routing
functionality with various file types and edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.processors.router import DocumentRouter
from src.processors.base import DocumentProcessor, UnsupportedFormatError, FileValidationError
from src.models import ValidationResult, ContentType
from src.config import ProcessingConfig


class MockProcessor(DocumentProcessor):
    """Mock processor for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.supported_formats = ['txt']
    
    def validate_file(self, file_path: str) -> ValidationResult:
        return ValidationResult(is_valid=True)
    
    def extract_content(self, file_path: str) -> str:
        return "mock content"
    
    def extract_metadata(self, file_path: str):
        return {}
    
    def chunk_content(self, content: str, document_id: str, file_path: str):
        return []
    
    def get_content_type(self) -> ContentType:
        return ContentType.TEXT


class TestDocumentRouter:
    """Test cases for DocumentRouter."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig(
            max_file_size_mb=10,
            supported_formats=['pdf', 'docx', 'png', 'jpg', 'mp3', 'wav']
        )
    
    @pytest.fixture
    def router(self, config):
        """Create DocumentRouter instance."""
        return DocumentRouter(config)
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_init(self, config):
        """Test DocumentRouter initialization."""
        router = DocumentRouter(config)
        
        assert router.config == config
        assert isinstance(router._processors, dict)
        assert isinstance(router._format_mappings, dict)
    
    def test_register_processor(self, router):
        """Test processor registration."""
        router.register_processor(MockProcessor, ['txt', 'text'])
        
        assert 'txt' in router._processors
        assert 'text' in router._processors
        assert router._processors['txt'] == MockProcessor
        assert router._processors['text'] == MockProcessor
    
    def test_get_supported_formats(self, router):
        """Test getting supported formats."""
        router.register_processor(MockProcessor, ['txt'])
        
        formats = router.get_supported_formats()
        assert 'txt' in formats
        assert isinstance(formats, list)
    
    def test_detect_file_format_by_extension(self, router, temp_file):
        """Test format detection by file extension."""
        # Test PDF extension
        pdf_path = temp_file + '.pdf'
        os.rename(temp_file, pdf_path)
        
        format_detected = router.detect_file_format(pdf_path)
        assert format_detected == 'pdf'
        
        os.rename(pdf_path, temp_file)  # Restore for cleanup
    
    def test_detect_file_format_case_insensitive(self, router, temp_file):
        """Test case-insensitive format detection."""
        pdf_path = temp_file + '.PDF'
        os.rename(temp_file, pdf_path)
        
        format_detected = router.detect_file_format(pdf_path)
        assert format_detected == 'pdf'
        
        os.rename(pdf_path, temp_file)
    
    @patch('src.processors.router.magic.Magic')
    def test_detect_file_format_by_mime_type(self, mock_magic, router, temp_file):
        """Test format detection by MIME type."""
        # Mock magic to return PDF MIME type
        mock_magic_instance = Mock()
        mock_magic_instance.from_file.return_value = 'application/pdf'
        mock_magic.return_value = mock_magic_instance
        
        # Create router with mocked magic
        router._magic = mock_magic_instance
        
        format_detected = router.detect_file_format(temp_file)
        assert format_detected == 'pdf'
    
    def test_detect_file_format_by_signature_pdf(self, router, temp_file):
        """Test PDF detection by file signature."""
        # Write PDF header to file
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\n')
        
        format_detected = router.detect_file_format(temp_file)
        assert format_detected == 'pdf'
    
    def test_detect_file_format_by_signature_png(self, router, temp_file):
        """Test PNG detection by file signature."""
        # Write PNG header to file
        with open(temp_file, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        
        format_detected = router.detect_file_format(temp_file)
        assert format_detected == 'png'
    
    def test_detect_file_format_by_signature_jpeg(self, router, temp_file):
        """Test JPEG detection by file signature."""
        # Write JPEG header to file
        with open(temp_file, 'wb') as f:
            f.write(b'\xff\xd8\xff\xe0')
        
        format_detected = router.detect_file_format(temp_file)
        assert format_detected == 'jpg'
    
    def test_detect_file_format_unknown(self, router, temp_file):
        """Test unknown format detection."""
        # File with unknown extension and content
        unknown_path = temp_file + '.unknown'
        os.rename(temp_file, unknown_path)
        
        format_detected = router.detect_file_format(unknown_path)
        assert format_detected is None
        
        os.rename(unknown_path, temp_file)
    
    def test_validate_file_success(self, router, temp_file):
        """Test successful file validation."""
        # Add PDF extension and header
        pdf_path = temp_file + '.pdf'
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\ntest content')
        os.rename(temp_file, pdf_path)
        
        result = router.validate_file(pdf_path)
        
        assert result.is_valid is True
        assert result.file_format == 'pdf'
        assert result.file_size > 0
        assert result.error_message is None
        
        os.rename(pdf_path, temp_file)
    
    def test_validate_file_not_exists(self, router):
        """Test validation of non-existent file."""
        result = router.validate_file('/nonexistent/file.pdf')
        
        assert result.is_valid is False
        assert "does not exist" in result.error_message
    
    def test_validate_file_is_directory(self, router):
        """Test validation when path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = router.validate_file(temp_dir)
            
            assert result.is_valid is False
            assert "not a file" in result.error_message
    
    def test_validate_file_empty(self, router):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = router.validate_file(temp_path)
            
            assert result.is_valid is False
            assert "empty" in result.error_message.lower()
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_too_large(self, router):
        """Test validation of file that's too large."""
        # Create file larger than max size (10MB in test config)
        large_content = b'x' * (11 * 1024 * 1024)  # 11MB
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            result = router.validate_file(temp_path)
            
            assert result.is_valid is False
            assert "too large" in result.error_message.lower()
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_unsupported_format(self, router, temp_file):
        """Test validation of unsupported format."""
        # Use extension not in supported formats
        unsupported_path = temp_file + '.xyz'
        os.rename(temp_file, unsupported_path)
        
        result = router.validate_file(unsupported_path)
        
        assert result.is_valid is False
        assert "unsupported" in result.error_message.lower()
        
        os.rename(unsupported_path, temp_file)
    
    def test_validate_file_format_not_enabled(self, router, temp_file):
        """Test validation when format is detected but not enabled."""
        # Create config without PDF support
        config = ProcessingConfig(supported_formats=['docx'])
        router_no_pdf = DocumentRouter(config)
        
        # Create PDF file
        pdf_path = temp_file + '.pdf'
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\ntest')
        os.rename(temp_file, pdf_path)
        
        result = router_no_pdf.validate_file(pdf_path)
        
        assert result.is_valid is False
        assert "not enabled" in result.error_message
        
        os.rename(pdf_path, temp_file)
    
    def test_route_document_success(self, router, temp_file):
        """Test successful document routing."""
        # Register mock processor
        router.register_processor(MockProcessor, ['pdf'])
        
        # Create PDF file
        pdf_path = temp_file + '.pdf'
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\ntest')
        os.rename(temp_file, pdf_path)
        
        processor = router.route_document(pdf_path)
        
        assert isinstance(processor, MockProcessor)
        
        os.rename(pdf_path, temp_file)
    
    def test_route_document_validation_failure(self, router):
        """Test routing with validation failure."""
        with pytest.raises(FileValidationError):
            router.route_document('/nonexistent/file.pdf')
    
    def test_route_document_no_processor(self, router, temp_file):
        """Test routing when no processor is registered."""
        # Create PDF file but don't register processor
        pdf_path = temp_file + '.pdf'
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\ntest')
        os.rename(temp_file, pdf_path)
        
        with pytest.raises(UnsupportedFormatError):
            router.route_document(pdf_path)
        
        os.rename(pdf_path, temp_file)
    
    def test_validate_pdf_valid(self, router, temp_file):
        """Test PDF-specific validation with valid file."""
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-1.4\ntest content')
        
        result = router._validate_pdf(temp_file)
        assert result.is_valid is True
    
    def test_validate_pdf_invalid_header(self, router, temp_file):
        """Test PDF validation with invalid header."""
        with open(temp_file, 'wb') as f:
            f.write(b'NOT A PDF')
        
        result = router._validate_pdf(temp_file)
        assert result.is_valid is False
        assert "invalid pdf header" in result.error_message.lower()
    
    def test_validate_pdf_unknown_version(self, router, temp_file):
        """Test PDF validation with unknown version."""
        with open(temp_file, 'wb') as f:
            f.write(b'%PDF-9.9\ntest')
        
        result = router._validate_pdf(temp_file)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "unknown pdf version" in result.warnings[0].lower()
    
    @patch('zipfile.ZipFile')
    def test_validate_docx_valid(self, mock_zipfile, router, temp_file):
        """Test DOCX validation with valid file."""
        # Mock zipfile to simulate valid DOCX
        mock_zip = Mock()
        mock_zip.namelist.return_value = ['[Content_Types].xml', '_rels/.rels', 'word/document.xml']
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        result = router._validate_docx(temp_file)
        assert result.is_valid is True
    
    @patch('zipfile.ZipFile')
    def test_validate_docx_missing_files(self, mock_zipfile, router, temp_file):
        """Test DOCX validation with missing required files."""
        # Mock zipfile to simulate invalid DOCX
        mock_zip = Mock()
        mock_zip.namelist.return_value = ['some_file.xml']
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        result = router._validate_docx(temp_file)
        assert result.is_valid is False
        assert "invalid docx structure" in result.error_message.lower()
    
    @patch('zipfile.ZipFile')
    def test_validate_docx_bad_zip(self, mock_zipfile, router, temp_file):
        """Test DOCX validation with corrupted ZIP."""
        import zipfile
        mock_zipfile.side_effect = zipfile.BadZipFile("Bad ZIP file")
        
        result = router._validate_docx(temp_file)
        assert result.is_valid is False
        assert "not a valid zip" in result.error_message.lower()
    
    def test_check_docx_signature_valid(self, router):
        """Test DOCX signature detection."""
        # Create a mock ZIP file that looks like DOCX
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as f:
            import zipfile
            with zipfile.ZipFile(f, 'w') as zf:
                zf.writestr('[Content_Types].xml', 'mock content')
            temp_path = f.name
        
        try:
            result = router._check_docx_signature(temp_path)
            assert result == 'docx'
        finally:
            os.unlink(temp_path)
    
    def test_check_docx_signature_invalid(self, router, temp_file):
        """Test DOCX signature detection with regular ZIP."""
        # Create a regular ZIP file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as f:
            import zipfile
            with zipfile.ZipFile(f, 'w') as zf:
                zf.writestr('regular_file.txt', 'content')
            temp_path = f.name
        
        try:
            result = router._check_docx_signature(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)
    
    @patch('PIL.Image.open')
    def test_validate_image_valid(self, mock_image_open, router, temp_file):
        """Test image validation with valid image."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (800, 600)
        mock_img.verify.return_value = None
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        result = router._validate_image(temp_file)
        assert result.is_valid is True
    
    @patch('PIL.Image.open')
    def test_validate_image_large_dimensions(self, mock_image_open, router, temp_file):
        """Test image validation with very large dimensions."""
        # Mock PIL Image with large dimensions
        mock_img = Mock()
        mock_img.size = (15000, 12000)
        mock_img.verify.return_value = None
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        result = router._validate_image(temp_file)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "very large" in result.warnings[0].lower()
    
    @patch('PIL.Image.open')
    def test_validate_image_corrupted(self, mock_image_open, router, temp_file):
        """Test image validation with corrupted image."""
        mock_image_open.side_effect = Exception("Corrupted image")
        
        result = router._validate_image(temp_file)
        assert result.is_valid is False
        assert "image validation error" in result.error_message.lower()
    
    def test_validate_audio_normal_file(self, router, temp_file):
        """Test audio validation with normal file."""
        result = router._validate_audio(temp_file)
        assert result.is_valid is True
    
    def test_validate_audio_large_file(self, router):
        """Test audio validation with large file."""
        # Create large file (>100MB)
        large_content = b'x' * (101 * 1024 * 1024)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(large_content)
            temp_path = f.name
        
        try:
            result = router._validate_audio(temp_path)
            assert result.is_valid is True
            assert len(result.warnings) > 0
            assert "large audio file" in result.warnings[0].lower()
        finally:
            os.unlink(temp_path)
    
    @patch('librosa.get_duration')
    def test_validate_audio_long_duration(self, mock_get_duration, router, temp_file):
        """Test audio validation with long duration."""
        # Mock librosa to return long duration (>1 hour)
        mock_get_duration.return_value = 4000  # seconds
        
        result = router._validate_audio(temp_file)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "longer than 1 hour" in result.warnings[0].lower()


if __name__ == '__main__':
    pytest.main([__file__])