"""
OCR Engine for text extraction from images using Tesseract.

Provides OCR functionality with confidence scoring and image preprocessing
for improved text extraction accuracy.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR text extraction."""
    text: str
    confidence_score: float
    word_confidences: List[float]
    bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, width, height)
    preprocessing_applied: List[str]


class OCREngine:
    """
    OCR engine that wraps Tesseract functionality with image preprocessing.
    
    Provides text extraction from images with confidence scoring and
    automatic image enhancement for better OCR accuracy.
    """
    
    def __init__(self, 
                 tesseract_config: str = '--oem 3 --psm 6',
                 min_confidence: float = 30.0,
                 enable_preprocessing: bool = True):
        """
        Initialize OCR engine.
        
        Args:
            tesseract_config: Tesseract configuration string
            min_confidence: Minimum confidence threshold for text extraction
            enable_preprocessing: Whether to apply image preprocessing
        """
        self.tesseract_config = tesseract_config
        self.min_confidence = min_confidence
        self.enable_preprocessing = enable_preprocessing
        
        # Test Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise RuntimeError(f"Tesseract not available: {e}")
    
    def extract_text(self, image_path: str) -> OCRResult:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            OCRResult with extracted text and metadata
            
        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If OCR extraction fails
        """
        try:
            # Load image
            image = Image.open(image_path)
            original_image = image.copy()
            
            preprocessing_steps = []
            
            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                image, steps = self._preprocess_image(image)
                preprocessing_steps.extend(steps)
            
            # Convert to OpenCV format for detailed analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with basic config
            text = pytesseract.image_to_string(
                image, 
                config=self.tesseract_config
            ).strip()
            
            # Process confidence data
            confidences = [float(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            word_confidences = [conf for conf in confidences if conf >= self.min_confidence]
            
            # Extract bounding boxes for words with sufficient confidence
            bounding_boxes = []
            for i, conf in enumerate(ocr_data['conf']):
                if float(conf) >= self.min_confidence:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    bounding_boxes.append((x, y, w, h))
            
            # Calculate overall confidence
            overall_confidence = np.mean(word_confidences) if word_confidences else 0.0
            
            return OCRResult(
                text=text,
                confidence_score=overall_confidence,
                word_confidences=word_confidences,
                bounding_boxes=bounding_boxes,
                preprocessing_applied=preprocessing_steps
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Tuple of (preprocessed_image, applied_steps)
        """
        steps_applied = []
        processed_image = image.copy()
        
        # Convert to grayscale if not already
        if processed_image.mode != 'L':
            processed_image = processed_image.convert('L')
            steps_applied.append('grayscale_conversion')
        
        # Resize if image is too small (OCR works better on larger images)
        width, height = processed_image.size
        if width < 300 or height < 300:
            scale_factor = max(300 / width, 300 / height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
            steps_applied.append('upscaling')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)
        steps_applied.append('contrast_enhancement')
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(processed_image)
        processed_image = enhancer.enhance(1.2)
        steps_applied.append('sharpness_enhancement')
        
        # Apply slight blur to reduce noise
        processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))
        steps_applied.append('noise_reduction')
        
        return processed_image, steps_applied
    
    def extract_text_regions(self, image_path: str) -> List[Dict]:
        """
        Extract text regions with detailed location information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text regions and their coordinates
        """
        try:
            image = Image.open(image_path)
            
            if self.enable_preprocessing:
                image, _ = self._preprocess_image(image)
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            regions = []
            current_text = ""
            current_box = None
            
            for i in range(len(ocr_data['text'])):
                conf = float(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if conf >= self.min_confidence and text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    regions.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': (x, y, w, h),
                        'level': ocr_data['level'][i]
                    })
            
            return regions
            
        except Exception as e:
            logger.error(f"Text region extraction failed for {image_path}: {e}")
            return []
    
    def get_text_confidence(self, image_path: str) -> float:
        """
        Get overall text confidence for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Overall confidence score (0-100)
        """
        try:
            result = self.extract_text(image_path)
            return result.confidence_score
        except Exception:
            return 0.0