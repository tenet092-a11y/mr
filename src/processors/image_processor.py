"""
Image processor for extracting text and generating embeddings from images.

Combines OCR text extraction with CLIP embeddings for multimodal content processing.
Supports image metadata extraction and region detection for precise citations.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image, ExifTags
import numpy as np
import torch
import clip

from .base import DocumentProcessor, ProcessingError, ContentExtractionError
from .ocr_engine import OCREngine, OCRResult
from ..models import (
    DocumentContent, ContentChunk, ValidationResult, 
    DocumentMetadata, SourceLocation, ContentType
)
from ..config import ProcessingConfig

logger = logging.getLogger(__name__)


class ImageProcessor(DocumentProcessor):
    """
    Processor for image files that combines OCR text extraction with CLIP embeddings.
    
    Extracts text using OCR, generates visual embeddings using CLIP,
    and provides detailed metadata including EXIF data and image regions
    for precise citation mapping.
    """
    
    def __init__(self, config: ProcessingConfig, 
                 clip_model_name: str = "ViT-B/32",
                 ocr_config: Optional[Dict[str, Any]] = None):
        """
        Initialize image processor.
        
        Args:
            config: Processing configuration
            clip_model_name: CLIP model variant to use
            ocr_config: Configuration for OCR engine
        """
        super().__init__(config)
        self.supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
        
        # Initialize OCR engine
        ocr_config = ocr_config or {}
        self.ocr_engine = OCREngine(**ocr_config)
        
        # Initialize CLIP model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            logger.info(f"CLIP model {clip_model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise ProcessingError(f"CLIP model initialization failed: {e}")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate image file for processing.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ValidationResult with validation status
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Check file extension
            file_extension = path.suffix.lower().lstrip('.')
            if file_extension not in self.supported_formats:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported format: {file_extension}. Supported: {self.supported_formats}"
                )
            
            # Check file size
            if self._is_file_too_large(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large. Maximum size: {self.config.max_file_size_mb}MB"
                )
            
            # Try to open image
            try:
                with Image.open(file_path) as img:
                    # Basic image validation
                    img.verify()
                    
                return ValidationResult(
                    is_valid=True,
                    file_format=file_extension,
                    file_size=self._get_file_size(file_path)
                )
                
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid image file: {str(e)}"
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract text content from image using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text content
            
        Raises:
            ContentExtractionError: If text extraction fails
        """
        try:
            ocr_result = self.ocr_engine.extract_text(file_path)
            
            if not ocr_result.text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return ""
            
            logger.info(f"Extracted {len(ocr_result.text)} characters from {file_path} "
                       f"with confidence {ocr_result.confidence_score:.1f}%")
            
            return ocr_result.text
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise ContentExtractionError(f"Failed to extract text: {e}", file_path, e)
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract comprehensive metadata from image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            DocumentMetadata with image information
        """
        try:
            with Image.open(file_path) as img:
                # Basic image properties
                width, height = img.size
                file_size = self._get_file_size(file_path)
                format_name = img.format
                
                # Extract EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                
                # Get creation date from EXIF if available
                creation_date = None
                if 'DateTime' in exif_data:
                    creation_date = str(exif_data['DateTime'])
                elif 'DateTimeOriginal' in exif_data:
                    creation_date = str(exif_data['DateTimeOriginal'])
                
                # Get camera/software info
                camera_make = exif_data.get('Make', '')
                camera_model = exif_data.get('Model', '')
                software = exif_data.get('Software', '')
                
                title = f"{Path(file_path).stem}"
                if camera_make and camera_model:
                    title += f" ({camera_make} {camera_model})"
                
                return DocumentMetadata(
                    title=title,
                    author=software if software else None,
                    creation_date=creation_date,
                    file_size=file_size,
                    dimensions=(width, height),
                    format_version=format_name,
                    confidence_score=1.0
                )
                
        except Exception as e:
            logger.error(f"Metadata extraction failed for {file_path}: {e}")
            # Return basic metadata even if EXIF extraction fails
            try:
                file_size = self._get_file_size(file_path)
                with Image.open(file_path) as img:
                    width, height = img.size
                    return DocumentMetadata(
                        title=Path(file_path).stem,
                        file_size=file_size,
                        dimensions=(width, height),
                        confidence_score=0.5
                    )
            except:
                return DocumentMetadata(
                    title=Path(file_path).stem,
                    confidence_score=0.1
                )
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Create content chunks from extracted text and image regions.
        
        Args:
            content: Extracted text content
            document_id: Document identifier
            file_path: Original file path
            
        Returns:
            List of ContentChunk objects with text and region information
        """
        chunks = []
        
        try:
            # Get detailed text regions from OCR
            text_regions = self.ocr_engine.extract_text_regions(file_path)
            
            if not text_regions and content.strip():
                # Fallback: create single chunk for entire image if no regions detected
                chunk_id = self._create_chunk_id(document_id, 0)
                source_location = SourceLocation(
                    file_path=file_path,
                    image_coordinates=(0, 0, 0, 0)  # Full image
                )
                
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    content_type=ContentType.IMAGE,
                    source_location=source_location,
                    metadata={
                        'ocr_confidence': 0.0,
                        'region_type': 'full_image',
                        'text_length': len(content)
                    }
                )
                chunks.append(chunk)
                
            else:
                # Create chunks for each detected text region
                for i, region in enumerate(text_regions):
                    if not region['text'].strip():
                        continue
                        
                    chunk_id = self._create_chunk_id(document_id, i)
                    bbox = region['bbox']
                    
                    source_location = SourceLocation(
                        file_path=file_path,
                        image_coordinates=bbox
                    )
                    
                    chunk = ContentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=region['text'],
                        content_type=ContentType.IMAGE,
                        source_location=source_location,
                        metadata={
                            'ocr_confidence': region['confidence'],
                            'region_type': 'text_region',
                            'bbox': bbox,
                            'ocr_level': region['level'],
                            'text_length': len(region['text'])
                        },
                        confidence_score=region['confidence'] / 100.0
                    )
                    chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks for image {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            # Fallback: create single chunk with available content
            if content.strip():
                chunk_id = self._create_chunk_id(document_id, 0)
                source_location = SourceLocation(file_path=file_path)
                
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    content_type=ContentType.IMAGE,
                    source_location=source_location,
                    metadata={'error': str(e)},
                    confidence_score=0.1
                )
                return [chunk]
            
            return []
    
    def generate_clip_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate CLIP embedding for image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            CLIP embedding as numpy array
            
        Raises:
            ProcessingError: If embedding generation fails
        """
        try:
            with Image.open(image_path) as image:
                # Preprocess image for CLIP
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # Generate embedding
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    # Normalize embedding
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"CLIP embedding generation failed for {image_path}: {e}")
            raise ProcessingError(f"Failed to generate CLIP embedding: {e}", image_path, e)
    
    def generate_image_description(self, image_path: str) -> str:
        """
        Generate textual description of image content using CLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated description text
        """
        try:
            # Predefined categories for image classification
            categories = [
                "a photo of a document", "a photo of text", "a photo of a chart",
                "a photo of a graph", "a photo of a table", "a photo of a diagram",
                "a photo of a person", "a photo of people", "a photo of a building",
                "a photo of nature", "a photo of an object", "a photo of artwork",
                "a screenshot", "a scan of a document", "handwritten text"
            ]
            
            with Image.open(image_path) as image:
                # Preprocess image
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # Tokenize text categories
                text_tokens = clip.tokenize(categories).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # Calculate similarities
                    similarities = (image_features @ text_features.T).softmax(dim=-1)
                    
                    # Get top predictions
                    top_probs, top_indices = similarities[0].topk(3)
                    
                    descriptions = []
                    for prob, idx in zip(top_probs, top_indices):
                        if prob > 0.1:  # Only include confident predictions
                            descriptions.append(f"{categories[idx]} ({prob:.2f})")
                    
                    return "; ".join(descriptions) if descriptions else "image content"
                    
        except Exception as e:
            logger.warning(f"Image description generation failed for {image_path}: {e}")
            return "image content"
    
    def get_content_type(self) -> ContentType:
        """Get the content type handled by this processor."""
        return ContentType.IMAGE
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw OCR content.
        
        Args:
            raw_content: Raw text extracted from image
            
        Returns:
            Cleaned and processed text
        """
        if not raw_content:
            return ""
        
        # Basic text cleaning
        processed = raw_content.strip()
        
        # Remove excessive whitespace
        processed = ' '.join(processed.split())
        
        # Remove very short "words" that are likely OCR artifacts
        words = processed.split()
        cleaned_words = [word for word in words if len(word) > 1 or word.isalnum()]
        
        return ' '.join(cleaned_words)
    
    def extract_image_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract detailed image regions with text and coordinates.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of region dictionaries with text, coordinates, and metadata
        """
        try:
            # Get OCR regions
            ocr_regions = self.ocr_engine.extract_text_regions(image_path)
            
            # Enhance with additional metadata
            enhanced_regions = []
            for region in ocr_regions:
                enhanced_region = {
                    'text': region['text'],
                    'confidence': region['confidence'],
                    'bbox': region['bbox'],
                    'ocr_level': region['level'],
                    'region_type': 'text',
                    'area': region['bbox'][2] * region['bbox'][3]  # width * height
                }
                enhanced_regions.append(enhanced_region)
            
            # Sort by confidence and area (larger, more confident regions first)
            enhanced_regions.sort(
                key=lambda x: (x['confidence'], x['area']), 
                reverse=True
            )
            
            return enhanced_regions
            
        except Exception as e:
            logger.error(f"Region extraction failed for {image_path}: {e}")
            return []