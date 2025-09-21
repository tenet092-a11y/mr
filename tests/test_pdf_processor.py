"""
Unit tests for PDF processor functionality.

Tests PDF text extraction, metadata extraction, content chunking,
and error handling with various PDF scenarios.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Import the classes we're testing
from src.processors.pdf_processor import PDFProcessor
from src.config import ProcessingConfig
from src.models import ContentType, ValidationResult, DocumentMetadata, ContentChunk


class TestPDFProcessor:
    """Test suite for PDFProcessor class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ProcessingConfig(
            chunk_size=100,
            chunk_overlap=20,
            max_file_size_mb=10
        )
    
    @pytest.fixture
    def processor(self, config):
        """Create a PDFProcessor instance for testing."""
        return PDFProcessor(config)
    
    @pytest.fixture
    def mock_pdf_reader(self):
        """Create a mock PDF reader for testing."""
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = []
        mock_reader.metadata = {}
        return mock_reader
    
    def test_initialization(self, processor):
        """Test PDFProcessor initialization."""
        assert processor.supported_formats == ['pdf']
        assert processor.get_content_type() == ContentType.PDF
        assert isinstance(processor.config, ProcessingConfig)
    
    def test_get_supported_formats(self, processor):
        """Test getting supported file formats."""
        formats = processor.get_supported_formats()
        assert formats == ['pdf']
        assert isinstance(formats, list)
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_validate_file_success(self, mock_stat, mock_exists, mock_file, mock_reader, processor):
        """Test successful file validation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1024  # 1KB file
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [Mock()]  # One page
        mock_reader_instance.pages[0].extract_text.return_value = "Sample text"
        mock_reader.return_value = mock_reader_instance
        
        # Test validation
        result = processor.validate_file("test.pdf")
        
        assert result.is_valid is True
        assert result.file_format == "pdf"
        assert result.file_size == 1024
        assert result.error_message is None
    
    @patch('pathlib.Path.exists')
    def test_validate_file_not_found(self, mock_exists, processor):
        """Test validation with non-existent file."""
        mock_exists.return_value = False
        
        result = processor.validate_file("nonexistent.pdf")
        
        assert result.is_valid is False
        assert "File not found" in result.error_message
    
    def test_validate_file_wrong_extension(self, processor):
        """Test validation with wrong file extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = processor.validate_file(tmp_path)
            assert result.is_valid is False
            assert "Invalid file extension" in result.error_message
            assert result.file_format == ".txt"
        finally:
            os.unlink(tmp_path)
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_validate_file_encrypted(self, mock_stat, mock_exists, mock_file, mock_reader, processor):
        """Test validation with encrypted PDF."""
        # Setup mocks
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1024
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = True
        mock_reader.return_value = mock_reader_instance
        
        result = processor.validate_file("encrypted.pdf")
        
        assert result.is_valid is False
        assert "encrypted" in result.error_message.lower()
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_validate_file_no_pages(self, mock_stat, mock_exists, mock_file, mock_reader, processor):
        """Test validation with PDF containing no pages."""
        # Setup mocks
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1024
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = []  # No pages
        mock_reader.return_value = mock_reader_instance
        
        result = processor.validate_file("empty.pdf")
        
        assert result.is_valid is False
        assert "no pages" in result.error_message.lower()
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    def test_extract_content_success(self, mock_file, mock_reader, processor):
        """Test successful content extraction."""
        # Setup mock pages
        page1 = Mock()
        page1.extract_text.return_value = "This is page 1 content."
        page2 = Mock()
        page2.extract_text.return_value = "This is page 2 content."
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [page1, page2]
        mock_reader.return_value = mock_reader_instance
        
        content = processor.extract_content("test.pdf")
        
        assert "[PAGE 1]" in content
        assert "[PAGE 2]" in content
        assert "This is page 1 content." in content
        assert "This is page 2 content." in content
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    def test_extract_content_encrypted(self, mock_file, mock_reader, processor):
        """Test content extraction from encrypted PDF."""
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = True
        mock_reader.return_value = mock_reader_instance
        
        with pytest.raises(Exception) as exc_info:
            processor.extract_content("encrypted.pdf")
        
        assert "encrypted" in str(exc_info.value).lower()
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    def test_extract_content_empty_pages(self, mock_file, mock_reader, processor):
        """Test content extraction with empty pages."""
        # Setup mock pages - some empty, some with content
        page1 = Mock()
        page1.extract_text.return_value = ""  # Empty page
        page2 = Mock()
        page2.extract_text.return_value = "This is page 2 content."
        page3 = Mock()
        page3.extract_text.return_value = "   "  # Whitespace only
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [page1, page2, page3]
        mock_reader.return_value = mock_reader_instance
        
        content = processor.extract_content("test.pdf")
        
        # Should only include page 2 (non-empty)
        assert "[PAGE 2]" in content
        assert "This is page 2 content." in content
        assert "[PAGE 1]" not in content
        assert "[PAGE 3]" not in content
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    def test_extract_metadata_success(self, mock_file, mock_reader, processor):
        """Test successful metadata extraction."""
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [Mock(), Mock()]  # 2 pages
        mock_reader_instance.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author',
            '/CreationDate': 'D:20231201120000+00\'00\''
        }
        mock_reader_instance.pdf_header = '%PDF-1.4'
        mock_reader.return_value = mock_reader_instance
        
        with patch.object(processor, '_get_file_size', return_value=2048):
            metadata = processor.extract_metadata("test.pdf")
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.creation_date == "2023-12-01"
        assert metadata.page_count == 2
        assert metadata.file_size == 2048
        assert metadata.format_version == '%PDF-1.4'
        assert metadata.confidence_score == 1.0
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf data')
    def test_extract_metadata_minimal(self, mock_file, mock_reader, processor):
        """Test metadata extraction with minimal information."""
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [Mock()]  # 1 page
        mock_reader_instance.metadata = {}  # No metadata
        mock_reader.return_value = mock_reader_instance
        
        with patch.object(processor, '_get_file_size', return_value=1024):
            metadata = processor.extract_metadata("test.pdf")
        
        assert metadata.title == "test"  # Filename without extension
        assert metadata.author is None
        assert metadata.creation_date is None
        assert metadata.page_count == 1
        assert metadata.file_size == 1024
        assert metadata.confidence_score == 1.0
    
    def test_chunk_content_simple(self, processor):
        """Test content chunking with simple text."""
        content = "[PAGE 1]\nThis is a simple test document with some content."
        
        chunks = processor.chunk_content(content, "doc123", "test.pdf")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "doc123"
        assert chunk.content_type == ContentType.PDF
        assert chunk.source_location.file_path == "test.pdf"
        assert chunk.source_location.page_number == 1
        assert "This is a simple test document" in chunk.content
        assert chunk.metadata['page_number'] == 1
        assert chunk.metadata['chunk_index'] == 0
    
    def test_chunk_content_multiple_pages(self, processor):
        """Test content chunking with multiple pages."""
        content = (
            "[PAGE 1]\nThis is page one content.\n\n"
            "[PAGE 2]\nThis is page two content with more text."
        )
        
        chunks = processor.chunk_content(content, "doc123", "test.pdf")
        
        assert len(chunks) == 2
        
        # Check first chunk (page 1)
        chunk1 = chunks[0]
        assert chunk1.source_location.page_number == 1
        assert "This is page one content." in chunk1.content
        
        # Check second chunk (page 2)
        chunk2 = chunks[1]
        assert chunk2.source_location.page_number == 2
        assert "This is page two content" in chunk2.content
    
    def test_chunk_content_long_text(self, processor):
        """Test content chunking with text longer than chunk size."""
        # Create content longer than chunk_size (100 characters)
        long_text = "This is a very long piece of text that should be split into multiple chunks. " * 3
        content = f"[PAGE 1]\n{long_text}"
        
        chunks = processor.chunk_content(content, "doc123", "test.pdf")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # All chunks should be from page 1
        for chunk in chunks:
            assert chunk.source_location.page_number == 1
            assert chunk.document_id == "doc123"
            assert chunk.content_type == ContentType.PDF
    
    def test_process_content_cleaning(self, processor):
        """Test content processing and cleaning."""
        raw_content = (
            "[PAGE 1]\nThis   has    excessive   whitespace.\n\n"
            "[PAGE 2]\nThis.Has.Punctuation.Issues."
        )
        
        processed = processor.process_content(raw_content)
        
        # Page markers should be removed
        assert "[PAGE 1]" not in processed
        assert "[PAGE 2]" not in processed
        
        # Excessive whitespace should be cleaned
        assert "excessive   whitespace" not in processed
        assert "excessive whitespace" in processed
    
    def test_split_by_pages(self, processor):
        """Test splitting content by page markers."""
        content = (
            "[PAGE 1]\nFirst page content\nMore first page\n"
            "[PAGE 2]\nSecond page content\n"
            "[PAGE 5]\nFifth page content"
        )
        
        pages = processor._split_by_pages(content)
        
        assert len(pages) == 3
        assert 1 in pages
        assert 2 in pages
        assert 5 in pages
        assert "First page content" in pages[1]
        assert "Second page content" in pages[2]
        assert "Fifth page content" in pages[5]
    
    def test_create_text_chunks_simple(self, processor):
        """Test creating text chunks from simple text."""
        text = "Short text"
        chunks = processor._create_text_chunks(text, 100, 20)
        
        assert len(chunks) == 1
        assert chunks[0] == "Short text"
    
    def test_create_text_chunks_long(self, processor):
        """Test creating text chunks from long text."""
        text = "This is a long piece of text that needs to be split into multiple chunks for processing."
        chunks = processor._create_text_chunks(text, 30, 10)
        
        assert len(chunks) > 1
        
        # Check that chunks have proper overlap
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) <= 30
            # There should be some overlap between consecutive chunks
            # (This is a simplified check - actual overlap depends on word boundaries)
    
    def test_clean_pdf_text(self, processor):
        """Test PDF text cleaning functionality."""
        dirty_text = "This   has    excessive   whitespace  .  And  weird   spacing !"
        
        cleaned = processor._clean_pdf_text(dirty_text)
        
        # Should have normalized whitespace
        assert "   " not in cleaned
        assert "  ." not in cleaned
        assert ". And" in cleaned or ".And" in cleaned
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_extract_content_file_not_found(self, mock_file, mock_reader, processor):
        """Test content extraction with missing file."""
        with pytest.raises(Exception) as exc_info:
            processor.extract_content("missing.pdf")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_chunk_content_empty_content(self, processor):
        """Test chunking with empty content."""
        content = "[PAGE 1]\n   \n\n[PAGE 2]\n\n"
        
        chunks = processor.chunk_content(content, "doc123", "test.pdf")
        
        # Should return empty list for empty content
        assert len(chunks) == 0
    
    def test_chunk_id_generation(self, processor):
        """Test chunk ID generation."""
        content = "[PAGE 1]\nFirst chunk\n[PAGE 2]\nSecond chunk"
        
        chunks = processor.chunk_content(content, "doc123", "test.pdf")
        
        assert len(chunks) == 2
        assert chunks[0].chunk_id == "doc123_chunk_0000"
        assert chunks[1].chunk_id == "doc123_chunk_0001"


if __name__ == "__main__":
    pytest.main([__file__])