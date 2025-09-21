"""
Integration tests for PDF processor with document router.

Tests the integration between PDFProcessor and DocumentRouter
to ensure proper file routing and processing.
"""

import pytest
from unittest.mock import Mock, patch

from src.processors.pdf_processor import PDFProcessor
from src.processors.router import DocumentRouter
from src.config import ProcessingConfig
from src.models import ContentType


class TestPDFIntegration:
    """Integration tests for PDF processing."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig(
            chunk_size=100,
            chunk_overlap=20,
            max_file_size_mb=10
        )
    
    @pytest.fixture
    def router(self, config):
        """Create document router for testing."""
        return DocumentRouter(config)
    
    def test_router_identifies_pdf_processor(self, router):
        """Test that router correctly identifies PDF files."""
        processor = router.route_document("test.pdf")
        assert isinstance(processor, PDFProcessor)
        assert processor.get_content_type() == ContentType.PDF
    
    def test_pdf_processor_in_supported_formats(self, router):
        """Test that PDF is in router's supported formats."""
        supported = router.get_supported_formats()
        assert 'pdf' in supported
    
    @patch('src.processors.pdf_processor.PyPDF2.PdfReader')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_end_to_end_pdf_processing(self, mock_stat, mock_exists, mock_file, mock_reader, router):
        """Test end-to-end PDF processing through router."""
        # Setup mocks
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1024
        
        # Mock PDF content
        page1 = Mock()
        page1.extract_text.return_value = "This is test content from a PDF document."
        
        mock_reader_instance = Mock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [page1]
        mock_reader_instance.metadata = {'/Title': 'Test PDF'}
        mock_reader.return_value = mock_reader_instance
        
        # Process document through router
        processor = router.route_document("test.pdf")
        result = processor.process_document("test.pdf", "test_doc_123")
        
        # Verify processing result
        assert result.success is True
        assert result.document_content is not None
        assert result.document_content.document_id == "test_doc_123"
        assert result.document_content.content_type == ContentType.PDF
        assert len(result.document_content.chunks) > 0
        
        # Verify chunk content
        chunk = result.document_content.chunks[0]
        assert "This is test content" in chunk.content
        assert chunk.source_location.page_number == 1
        assert chunk.content_type == ContentType.PDF


if __name__ == "__main__":
    pytest.main([__file__])