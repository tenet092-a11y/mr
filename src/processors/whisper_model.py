"""
Whisper model wrapper for offline speech-to-text transcription.

This module provides a wrapper around OpenAI's Whisper model for
offline audio transcription with timestamp preservation and
support for multiple audio formats.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

import whisper
import librosa
import soundfile as sf
from whisper.utils import format_timestamp

from ..models import DocumentMetadata, SourceLocation
from ..config import ProcessingConfig


logger = logging.getLogger(__name__)


class AudioSegment:
    """Represents a segment of audio with transcription and timing information."""
    
    def __init__(
        self,
        start_time: float,
        end_time: float,
        text: str,
        confidence: float = 1.0,
        language: Optional[str] = None
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.confidence = confidence
        self.language = language
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end_time - self.start_time
    
    def __repr__(self) -> str:
        return f"AudioSegment({self.start_time:.2f}-{self.end_time:.2f}s: '{self.text[:50]}...')"


class TranscriptionResult:
    """Complete transcription result with segments and metadata."""
    
    def __init__(
        self,
        text: str,
        segments: List[AudioSegment],
        language: str,
        confidence: float,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.segments = segments
        self.language = language
        self.confidence = confidence
        self.duration = duration
        self.metadata = metadata or {}
    
    def get_text_at_time(self, timestamp: float) -> Optional[str]:
        """Get the text being spoken at a specific timestamp."""
        for segment in self.segments:
            if segment.start_time <= timestamp <= segment.end_time:
                return segment.text
        return None
    
    def get_segments_in_range(self, start_time: float, end_time: float) -> List[AudioSegment]:
        """Get all segments that overlap with the given time range."""
        overlapping_segments = []
        for segment in self.segments:
            if (segment.start_time <= end_time and segment.end_time >= start_time):
                overlapping_segments.append(segment)
        return overlapping_segments


class WhisperModel:
    """
    Wrapper class for OpenAI Whisper model providing offline speech-to-text capabilities.
    
    Supports multiple audio formats, timestamp preservation, and audio segmentation
    for precise citation generation.
    """
    
    SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac']
    
    def __init__(self, config: ProcessingConfig, model_size: str = "base"):
        """
        Initialize Whisper model wrapper.
        
        Args:
            config: Processing configuration
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.config = config
        self.model_size = model_size
        self._model = None
        self._temp_dir = Path(config.temp_directory) / "audio"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Whisper model wrapper with size: {model_size}")
    
    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        return self._model
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Check if the audio file format is supported.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if format is supported
        """
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return file_extension in self.SUPPORTED_FORMATS
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic audio file information.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio metadata
        """
        try:
            # Load audio to get basic info
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = len(audio_data) / sample_rate
            
            # Get file size
            file_size = Path(file_path).stat().st_size
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0],
                'file_size': file_size,
                'format': Path(file_path).suffix.lower().lstrip('.'),
                'samples': len(audio_data)
            }
        except Exception as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
            raise
    
    def preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file for optimal Whisper transcription.
        
        Args:
            file_path: Path to the original audio file
            
        Returns:
            Path to the preprocessed audio file
        """
        try:
            # Load audio with librosa (automatically converts to mono, 16kHz)
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Create temporary file for preprocessed audio
            temp_file = self._temp_dir / f"preprocessed_{Path(file_path).stem}.wav"
            
            # Save preprocessed audio
            sf.write(str(temp_file), audio_data, sample_rate)
            
            logger.debug(f"Preprocessed audio saved to: {temp_file}")
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio {file_path}: {e}")
            # Return original file if preprocessing fails
            return file_path
    
    def transcribe_audio(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio file with timestamp preservation.
        
        Args:
            file_path: Path to the audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            task: 'transcribe' or 'translate'
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        try:
            logger.info(f"Starting transcription of: {file_path}")
            
            # Preprocess audio for better results
            processed_file = self.preprocess_audio(file_path)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                processed_file,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                verbose=False
            )
            
            # Extract segments with timing information
            segments = []
            for segment_data in result.get('segments', []):
                segment = AudioSegment(
                    start_time=segment_data['start'],
                    end_time=segment_data['end'],
                    text=segment_data['text'].strip(),
                    confidence=segment_data.get('avg_logprob', 0.0),
                    language=result.get('language')
                )
                segments.append(segment)
            
            # Calculate overall confidence
            if segments:
                overall_confidence = np.mean([seg.confidence for seg in segments])
            else:
                overall_confidence = 0.0
            
            # Get audio duration
            audio_info = self.get_audio_info(file_path)
            
            transcription_result = TranscriptionResult(
                text=result['text'].strip(),
                segments=segments,
                language=result.get('language', 'unknown'),
                confidence=overall_confidence,
                duration=audio_info['duration'],
                metadata={
                    'model_size': self.model_size,
                    'task': task,
                    'word_timestamps': word_timestamps,
                    'audio_info': audio_info
                }
            )
            
            # Clean up temporary file if it was created
            if processed_file != file_path and Path(processed_file).exists():
                Path(processed_file).unlink()
            
            logger.info(f"Transcription completed. Language: {transcription_result.language}, "
                       f"Segments: {len(segments)}, Confidence: {overall_confidence:.3f}")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio {file_path}: {e}")
            raise
    
    def segment_audio_by_time(
        self,
        transcription_result: TranscriptionResult,
        segment_duration: float = 30.0,
        overlap_duration: float = 5.0
    ) -> List[AudioSegment]:
        """
        Create time-based segments from transcription for chunking.
        
        Args:
            transcription_result: Original transcription result
            segment_duration: Target duration for each segment in seconds
            overlap_duration: Overlap between segments in seconds
            
        Returns:
            List of AudioSegment objects with time-based chunking
        """
        time_segments = []
        current_time = 0.0
        segment_id = 0
        
        while current_time < transcription_result.duration:
            end_time = min(current_time + segment_duration, transcription_result.duration)
            
            # Get all original segments that overlap with this time window
            overlapping_segments = transcription_result.get_segments_in_range(
                current_time, end_time
            )
            
            if overlapping_segments:
                # Combine text from overlapping segments
                combined_text = " ".join([seg.text for seg in overlapping_segments])
                
                # Calculate average confidence
                avg_confidence = np.mean([seg.confidence for seg in overlapping_segments])
                
                time_segment = AudioSegment(
                    start_time=current_time,
                    end_time=end_time,
                    text=combined_text.strip(),
                    confidence=avg_confidence,
                    language=transcription_result.language
                )
                
                time_segments.append(time_segment)
            
            # Move to next segment with overlap
            current_time += segment_duration - overlap_duration
            segment_id += 1
        
        return time_segments
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            DocumentMetadata with audio-specific information
        """
        try:
            audio_info = self.get_audio_info(file_path)
            
            return DocumentMetadata(
                title=Path(file_path).stem,
                file_size=audio_info['file_size'],
                duration=audio_info['duration'],
                format_version=audio_info['format'],
                confidence_score=1.0,  # Will be updated after transcription
                language=None  # Will be detected during transcription
            )
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return DocumentMetadata(
                title=Path(file_path).stem,
                confidence_score=0.0
            )
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary audio files."""
        try:
            for temp_file in self._temp_dir.glob("preprocessed_*.wav"):
                temp_file.unlink()
            logger.debug("Cleaned up temporary audio files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup_temp_files()