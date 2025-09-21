"""
Audio processor for multimodal RAG system.

Handles audio file processing including transcription, metadata extraction,
and content chunking with timestamp-based citations. Integrates with
Whisper for speech-to-text and provides quality assessment.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from .base import DocumentProcessor, ProcessingError, ContentExtractionError
from .whisper_model import WhisperModel, TranscriptionResult, AudioSegment
from ..models import (
    DocumentMetadata, ContentChunk, ValidationResult, ContentType, SourceLocation
)
from ..config import ProcessingConfig


logger = logging.getLogger(__name__)


class AudioQualityAssessment:
    """Assessment of audio quality and transcription confidence."""
    
    def __init__(
        self,
        overall_confidence: float,
        duration: float,
        sample_rate: int,
        signal_to_noise_ratio: Optional[float] = None,
        speech_ratio: Optional[float] = None
    ):
        self.overall_confidence = overall_confidence
        self.duration = duration
        self.sample_rate = sample_rate
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.speech_ratio = speech_ratio
    
    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score (0.0 to 1.0).
        
        Combines transcription confidence with audio quality metrics.
        """
        score = self.overall_confidence
        
        # Adjust based on sample rate (higher is better)
        if self.sample_rate >= 44100:
            score *= 1.0
        elif self.sample_rate >= 22050:
            score *= 0.95
        elif self.sample_rate >= 16000:
            score *= 0.9
        else:
            score *= 0.8
        
        # Adjust based on duration (very short or very long files may have issues)
        if 10 <= self.duration <= 3600:  # 10 seconds to 1 hour is optimal
            score *= 1.0
        elif 5 <= self.duration < 10 or 3600 < self.duration <= 7200:
            score *= 0.95
        else:
            score *= 0.85
        
        return min(1.0, max(0.0, score))
    
    @property
    def quality_category(self) -> str:
        """Get quality category as string."""
        score = self.quality_score
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"


class AudioProcessor(DocumentProcessor):
    """
    Audio processor that transcribes audio files and extracts metadata.
    
    Supports multiple audio formats, provides timestamp-based chunking,
    and includes quality assessment for transcription confidence.
    """
    
    def __init__(self, config: ProcessingConfig, whisper_model_size: str = "base"):
        """
        Initialize AudioProcessor.
        
        Args:
            config: Processing configuration
            whisper_model_size: Whisper model size (tiny, base, small, medium, large)
        """
        super().__init__(config)
        self.supported_formats = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac']
        self.whisper_model = WhisperModel(config, whisper_model_size)
        
        # Audio processing settings
        self.segment_duration = getattr(config, 'audio_segment_duration', 30.0)
        self.segment_overlap = getattr(config, 'audio_segment_overlap', 5.0)
        self.min_confidence_threshold = getattr(config, 'min_audio_confidence', 0.3)
        
        logger.info(f"AudioProcessor initialized with Whisper model: {whisper_model_size}")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate audio file for processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            ValidationResult with validation status
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Check file extension
            file_extension = file_path_obj.suffix.lower().lstrip('.')
            if file_extension not in self.supported_formats:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported audio format: {file_extension}. "
                                f"Supported formats: {', '.join(self.supported_formats)}"
                )
            
            # Check file size
            if self._is_file_too_large(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large. Maximum size: {self.config.max_file_size_mb}MB"
                )
            
            # Check if Whisper can handle the format
            if not self.whisper_model.is_supported_format(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Audio format not supported by Whisper: {file_extension}"
                )
            
            # Try to get basic audio info
            try:
                audio_info = self.whisper_model.get_audio_info(file_path)
                warnings = []
                
                # Check duration
                if audio_info['duration'] > 7200:  # 2 hours
                    warnings.append("Very long audio file may take significant time to process")
                elif audio_info['duration'] < 1:  # Less than 1 second
                    warnings.append("Very short audio file may not transcribe well")
                
                # Check sample rate
                if audio_info['sample_rate'] < 8000:
                    warnings.append("Low sample rate may affect transcription quality")
                
                return ValidationResult(
                    is_valid=True,
                    file_format=file_extension,
                    file_size=audio_info['file_size'],
                    warnings=warnings
                )
                
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Failed to read audio file: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract text content from audio file via transcription.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Transcribed text content
            
        Raises:
            ContentExtractionError: If transcription fails
        """
        try:
            logger.info(f"Starting audio transcription for: {file_path}")
            
            # Transcribe audio
            transcription_result = self.whisper_model.transcribe_audio(
                file_path,
                language=None,  # Auto-detect
                task="transcribe",
                word_timestamps=True
            )
            
            # Store transcription result for later use in chunking
            self._last_transcription = transcription_result
            
            if not transcription_result.text.strip():
                logger.warning(f"No speech detected in audio file: {file_path}")
                return ""
            
            logger.info(f"Transcription completed. Language: {transcription_result.language}, "
                       f"Duration: {transcription_result.duration:.2f}s, "
                       f"Confidence: {transcription_result.confidence:.3f}")
            
            return transcription_result.text
            
        except Exception as e:
            logger.error(f"Failed to extract content from audio file {file_path}: {e}")
            raise ContentExtractionError(
                f"Audio transcription failed: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            DocumentMetadata with audio-specific information
        """
        try:
            # Get basic audio info
            audio_info = self.whisper_model.get_audio_info(file_path)
            
            # Use transcription result if available
            transcription_result = getattr(self, '_last_transcription', None)
            
            metadata = DocumentMetadata(
                title=Path(file_path).stem,
                file_size=audio_info['file_size'],
                duration=audio_info['duration'],
                format_version=audio_info['format'],
                language=transcription_result.language if transcription_result else None,
                confidence_score=transcription_result.confidence if transcription_result else 1.0
            )
            
            # Add audio-specific metadata
            metadata.metadata = {
                'sample_rate': audio_info['sample_rate'],
                'channels': audio_info['channels'],
                'samples': audio_info['samples'],
                'segments_count': len(transcription_result.segments) if transcription_result else 0,
                'whisper_model_size': self.whisper_model.model_size
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from audio file {file_path}: {e}")
            return DocumentMetadata(
                title=Path(file_path).stem,
                confidence_score=0.0
            )
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Create content chunks with timestamp-based citations.
        
        Args:
            content: Transcribed text content
            document_id: Unique document identifier
            file_path: Original audio file path
            
        Returns:
            List of ContentChunk objects with timestamp information
        """
        chunks = []
        
        try:
            # Use stored transcription result
            transcription_result = getattr(self, '_last_transcription', None)
            if not transcription_result:
                logger.warning("No transcription result available for chunking")
                return chunks
            
            # Create time-based segments for chunking
            time_segments = self.whisper_model.segment_audio_by_time(
                transcription_result,
                segment_duration=self.segment_duration,
                overlap_duration=self.segment_overlap
            )
            
            for i, segment in enumerate(time_segments):
                if not segment.text.strip():
                    continue
                
                chunk_id = self._create_chunk_id(document_id, i)
                
                # Create source location with timestamp information
                source_location = SourceLocation(
                    file_path=file_path,
                    timestamp_start=segment.start_time,
                    timestamp_end=segment.end_time
                )
                
                # Create content chunk
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=segment.text.strip(),
                    content_type=ContentType.AUDIO,
                    source_location=source_location,
                    confidence_score=segment.confidence,
                    metadata={
                        'duration': segment.duration,
                        'language': segment.language,
                        'segment_index': i,
                        'timestamp_formatted': f"{self._format_timestamp(segment.start_time)} - {self._format_timestamp(segment.end_time)}"
                    }
                )
                
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} audio chunks from {len(time_segments)} segments")
            
        except Exception as e:
            logger.error(f"Failed to chunk audio content: {e}")
            # Fallback: create single chunk with entire content
            if content.strip():
                chunk_id = self._create_chunk_id(document_id, 0)
                source_location = SourceLocation(file_path=file_path)
                
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content.strip(),
                    content_type=ContentType.AUDIO,
                    source_location=source_location,
                    confidence_score=0.5,  # Lower confidence for fallback
                    metadata={'fallback_chunk': True}
                )
                chunks.append(chunk)
        
        return chunks
    
    def assess_audio_quality(self, file_path: str, transcription_result: Optional[TranscriptionResult] = None) -> AudioQualityAssessment:
        """
        Assess audio quality and transcription confidence.
        
        Args:
            file_path: Path to the audio file
            transcription_result: Optional transcription result
            
        Returns:
            AudioQualityAssessment with quality metrics
        """
        try:
            # Get audio info
            audio_info = self.whisper_model.get_audio_info(file_path)
            
            # Use provided transcription or the last one
            if transcription_result is None:
                transcription_result = getattr(self, '_last_transcription', None)
            
            overall_confidence = transcription_result.confidence if transcription_result else 0.0
            
            # Calculate speech ratio (segments with speech vs total duration)
            speech_ratio = None
            if transcription_result and transcription_result.segments:
                total_speech_time = sum(seg.duration for seg in transcription_result.segments)
                speech_ratio = total_speech_time / transcription_result.duration if transcription_result.duration > 0 else 0.0
            
            return AudioQualityAssessment(
                overall_confidence=overall_confidence,
                duration=audio_info['duration'],
                sample_rate=audio_info['sample_rate'],
                speech_ratio=speech_ratio
            )
            
        except Exception as e:
            logger.error(f"Failed to assess audio quality for {file_path}: {e}")
            return AudioQualityAssessment(
                overall_confidence=0.0,
                duration=0.0,
                sample_rate=0
            )
    
    def get_content_type(self) -> ContentType:
        """Get the content type handled by this processor."""
        return ContentType.AUDIO
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw transcribed content.
        
        Args:
            raw_content: Raw transcribed text
            
        Returns:
            Processed content with basic cleaning
        """
        # Basic text cleaning for transcribed content
        processed = raw_content.strip()
        
        # Remove excessive whitespace
        processed = ' '.join(processed.split())
        
        # Remove common transcription artifacts
        artifacts = ['[MUSIC]', '[NOISE]', '[INAUDIBLE]', '[SILENCE]']
        for artifact in artifacts:
            processed = processed.replace(artifact, '')
        
        # Clean up extra spaces after artifact removal
        processed = ' '.join(processed.split())
        
        return processed
    
    def get_segment_at_timestamp(self, timestamp: float) -> Optional[AudioSegment]:
        """
        Get the audio segment at a specific timestamp.
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            AudioSegment if found, None otherwise
        """
        transcription_result = getattr(self, '_last_transcription', None)
        if not transcription_result:
            return None
        
        for segment in transcription_result.segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment
        
        return None
    
    def get_segments_in_range(self, start_time: float, end_time: float) -> List[AudioSegment]:
        """
        Get all segments that overlap with a time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            
        Returns:
            List of overlapping AudioSegment objects
        """
        transcription_result = getattr(self, '_last_transcription', None)
        if not transcription_result:
            return []
        
        return transcription_result.get_segments_in_range(start_time, end_time)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp as MM:SS or HH:MM:SS.
        
        Args:
            seconds: Timestamp in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            self.whisper_model.cleanup_temp_files()
            # Clear stored transcription result
            if hasattr(self, '_last_transcription'):
                delattr(self, '_last_transcription')
        except Exception as e:
            logger.warning(f"Error during AudioProcessor cleanup: {e}")
    
    def __del__(self):
        """Cleanup when processor is destroyed."""
        self.cleanup()