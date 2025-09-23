"""
Advanced citation generator for multimodal RAG responses.

Generates precise, accurate citations with detailed source information
including page numbers, timestamps, and confidence scores for different
content types (text, images, audio).
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..models import RetrievalResult, Citation, ContentType, SourceLocation

logger = logging.getLogger(__name__)


@dataclass
class CitationMatch:
    """Represents a match between response text and source content."""
    citation_id: int
    source_result: RetrievalResult
    matched_text: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class CitationStyle:
    """Configuration for citation formatting."""
    format_type: str = "numbered"  # "numbered", "author_year", "footnote"
    include_page_numbers: bool = True
    include_timestamps: bool = True
    include_confidence: bool = False
    max_excerpt_length: int = 150
    show_content_type: bool = True


class CitationGenerator:
    """
    Advanced citation generator for multimodal content.
    
    Features:
    - Precise citation matching using text analysis
    - Support for multiple content types (text, image, audio)
    - Confidence scoring for citation accuracy
    - Flexible citation formatting styles
    - Source location tracking with page/timestamp info
    - Duplicate citation detection and merging
    """
    
    def __init__(self, citation_style: Optional[CitationStyle] = None):
        """
        Initialize citation generator.
        
        Args:
            citation_style: Citation formatting configuration
        """
        self.citation_style = citation_style or CitationStyle()
        
        # Citation matching patterns
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\((\d+)\)',  # (1), (2), etc.
            r'source\s+(\d+)',  # source 1, source 2, etc.
            r'reference\s+(\d+)',  # reference 1, reference 2, etc.
        ]
        
        logger.info("CitationGenerator initialized")
    
    def generate_citations(
        self, 
        response_text: str, 
        context_results: List[RetrievalResult],
        min_confidence: float = 0.3
    ) -> List[Citation]:
        """
        Generate citations for a response based on context results.
        
        Args:
            response_text: Generated response text
            context_results: Source results used for generation
            min_confidence: Minimum confidence threshold for citations
            
        Returns:
            List of Citation objects with source information
        """
        if not response_text or not context_results:
            return []
        
        try:
            # Find citation matches in response
            citation_matches = self._find_citation_matches(response_text, context_results)
            
            # Filter by confidence
            high_confidence_matches = [
                match for match in citation_matches 
                if match.confidence >= min_confidence
            ]
            
            # Convert to Citation objects
            citations = []
            for match in high_confidence_matches:
                citation = self._create_citation(match)
                citations.append(citation)
            
            # Remove duplicates and sort by citation ID
            citations = self._deduplicate_citations(citations)
            citations.sort(key=lambda c: c.citation_id)
            
            logger.info(f"Generated {len(citations)} citations from {len(context_results)} sources")
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []
    
    def _find_citation_matches(
        self, 
        response_text: str, 
        context_results: List[RetrievalResult]
    ) -> List[CitationMatch]:
        """Find matches between response text and source content."""
        matches = []
        
        # First, find explicit citation markers in the response
        explicit_citations = self._extract_explicit_citations(response_text)
        
        # Map explicit citations to sources
        for citation_id, positions in explicit_citations.items():
            if citation_id <= len(context_results):
                source_result = context_results[citation_id - 1]  # 1-indexed
                
                for start_pos, end_pos in positions:
                    # Extract text around citation for context
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(response_text), end_pos + 50)
                    matched_text = response_text[context_start:context_end]
                    
                    # Calculate confidence based on content similarity
                    confidence = self._calculate_citation_confidence(
                        matched_text, source_result.content
                    )
                    
                    match = CitationMatch(
                        citation_id=citation_id,
                        source_result=source_result,
                        matched_text=matched_text.strip(),
                        confidence=confidence,
                        start_pos=start_pos,
                        end_pos=end_pos
                    )
                    matches.append(match)
        
        # Also find implicit citations based on content similarity
        implicit_matches = self._find_implicit_citations(response_text, context_results)
        matches.extend(implicit_matches)
        
        return matches
    
    def _extract_explicit_citations(self, text: str) -> Dict[int, List[Tuple[int, int]]]:
        """Extract explicit citation markers from text."""
        citations = {}
        
        for pattern in self.citation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    citation_id = int(match.group(1))
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    if citation_id not in citations:
                        citations[citation_id] = []
                    citations[citation_id].append((start_pos, end_pos))
                    
                except (ValueError, IndexError):
                    continue
        
        return citations
    
    def _find_implicit_citations(
        self, 
        response_text: str, 
        context_results: List[RetrievalResult]
    ) -> List[CitationMatch]:
        """Find implicit citations based on content similarity."""
        matches = []
        
        # Split response into sentences for analysis
        sentences = self._split_into_sentences(response_text)
        
        for sentence_idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            # Find best matching source for this sentence
            best_match = None
            best_similarity = 0.0
            
            for idx, result in enumerate(context_results):
                similarity = self._calculate_text_similarity(sentence, result.content)
                
                if similarity > best_similarity and similarity > 0.4:  # Threshold for implicit match
                    best_similarity = similarity
                    best_match = (idx + 1, result)  # 1-indexed citation ID
            
            if best_match:
                citation_id, source_result = best_match
                
                # Find sentence position in original text
                sentence_start = response_text.find(sentence)
                sentence_end = sentence_start + len(sentence)
                
                match = CitationMatch(
                    citation_id=citation_id,
                    source_result=source_result,
                    matched_text=sentence.strip(),
                    confidence=best_similarity * 0.8,  # Lower confidence for implicit
                    start_pos=sentence_start,
                    end_pos=sentence_end
                )
                matches.append(match)
        
        return matches
    
    def _calculate_citation_confidence(self, response_text: str, source_content: str) -> float:
        """Calculate confidence score for a citation match."""
        # Text similarity
        similarity = self._calculate_text_similarity(response_text, source_content)
        
        # Length factor (longer matches are more confident)
        length_factor = min(1.0, len(response_text) / 100)
        
        # Keyword overlap
        response_words = set(response_text.lower().split())
        source_words = set(source_content.lower().split())
        
        if response_words:
            keyword_overlap = len(response_words.intersection(source_words)) / len(response_words)
        else:
            keyword_overlap = 0.0
        
        # Combine factors
        confidence = (similarity * 0.5 + length_factor * 0.2 + keyword_overlap * 0.3)
        
        return min(1.0, confidence)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity (could be enhanced with embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_citation(self, match: CitationMatch) -> Citation:
        """Create Citation object from CitationMatch."""
        # Create excerpt from source content
        excerpt = self._create_excerpt(match.source_result.content)
        
        # Get preview data for images/audio
        preview_data = None
        if match.source_result.content_type == ContentType.IMAGE:
            # For images, we could include thumbnail data
            preview_data = self._get_image_preview(match.source_result.source_location.file_path)
        elif match.source_result.content_type == ContentType.AUDIO:
            # For audio, we could include waveform or transcript snippet
            preview_data = self._get_audio_preview(match.source_result)
        
        return Citation(
            citation_id=match.citation_id,
            source_file=match.source_result.source_location.file_path,
            location=match.source_result.source_location,
            excerpt=excerpt,
            relevance_score=match.confidence,
            content_type=match.source_result.content_type,
            preview_data=preview_data
        )
    
    def _create_excerpt(self, content: str) -> str:
        """Create excerpt from source content."""
        if len(content) <= self.citation_style.max_excerpt_length:
            return content
        
        # Try to find a good breaking point
        excerpt = content[:self.citation_style.max_excerpt_length]
        
        # Break at sentence boundary if possible
        last_sentence_end = max(
            excerpt.rfind('.'),
            excerpt.rfind('!'),
            excerpt.rfind('?')
        )
        
        if last_sentence_end > self.citation_style.max_excerpt_length * 0.7:
            excerpt = excerpt[:last_sentence_end + 1]
        else:
            # Break at word boundary
            last_space = excerpt.rfind(' ')
            if last_space > self.citation_style.max_excerpt_length * 0.8:
                excerpt = excerpt[:last_space]
            excerpt += "..."
        
        return excerpt.strip()
    
    def _get_image_preview(self, image_path: str) -> Optional[bytes]:
        """Get preview data for image citations."""
        # In a full implementation, this would generate thumbnail data
        # For now, return None
        return None
    
    def _get_audio_preview(self, result: RetrievalResult) -> Optional[bytes]:
        """Get preview data for audio citations."""
        # In a full implementation, this could include waveform data
        # For now, return None
        return None
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations and merge similar ones."""
        if not citations:
            return citations
        
        # Group by source file and location
        citation_groups = {}
        
        for citation in citations:
            key = (
                citation.source_file,
                citation.location.page_number,
                citation.location.paragraph_index,
                citation.location.timestamp_start
            )
            
            if key not in citation_groups:
                citation_groups[key] = []
            citation_groups[key].append(citation)
        
        # Keep best citation from each group
        deduplicated = []
        for group in citation_groups.values():
            # Sort by relevance score and keep the best one
            best_citation = max(group, key=lambda c: c.relevance_score)
            deduplicated.append(best_citation)
        
        return deduplicated
    
    def format_citations(self, citations: List[Citation]) -> List[str]:
        """
        Format citations according to the specified style.
        
        Args:
            citations: List of Citation objects
            
        Returns:
            List of formatted citation strings
        """
        formatted = []
        
        for citation in citations:
            formatted_citation = self._format_single_citation(citation)
            formatted.append(formatted_citation)
        
        return formatted
    
    def _format_single_citation(self, citation: Citation) -> str:
        """Format a single citation."""
        parts = []
        
        # Citation ID
        if self.citation_style.format_type == "numbered":
            parts.append(f"[{citation.citation_id}]")
        
        # Source file
        file_name = Path(citation.source_file).name
        parts.append(file_name)
        
        # Content type
        if self.citation_style.show_content_type:
            parts.append(f"({citation.content_type.value})")
        
        # Location information
        location_parts = []
        
        if citation.location.page_number and self.citation_style.include_page_numbers:
            location_parts.append(f"p. {citation.location.page_number}")
        
        if citation.location.paragraph_index:
            location_parts.append(f"¶{citation.location.paragraph_index}")
        
        if (citation.location.timestamp_start and 
            self.citation_style.include_timestamps and
            citation.content_type == ContentType.AUDIO):
            timestamp = self._format_timestamp(citation.location.timestamp_start)
            location_parts.append(f"at {timestamp}")
        
        if location_parts:
            parts.append(", ".join(location_parts))
        
        # Confidence score
        if self.citation_style.include_confidence:
            parts.append(f"(confidence: {citation.relevance_score:.2f})")
        
        return " ".join(parts)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def validate_citations(
        self, 
        citations: List[Citation], 
        context_results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Validate citations for accuracy and completeness.
        
        Args:
            citations: Generated citations
            context_results: Original context results
            
        Returns:
            Validation report with metrics and issues
        """
        report = {
            'total_citations': len(citations),
            'total_sources': len(context_results),
            'coverage_rate': 0.0,
            'average_confidence': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        if not citations:
            report['issues'].append("No citations generated")
            return report
        
        # Calculate coverage rate
        cited_sources = {c.source_file for c in citations}
        available_sources = {r.source_location.file_path for r in context_results}
        
        if available_sources:
            report['coverage_rate'] = len(cited_sources) / len(available_sources)
        
        # Calculate average confidence
        if citations:
            report['average_confidence'] = sum(c.relevance_score for c in citations) / len(citations)
        
        # Check for issues
        if report['coverage_rate'] < 0.5:
            report['issues'].append("Low source coverage - many sources not cited")
        
        if report['average_confidence'] < 0.5:
            report['issues'].append("Low average citation confidence")
        
        # Check for missing high-relevance sources
        high_relevance_sources = [
            r for r in context_results 
            if r.similarity_score > 0.8
        ]
        
        cited_high_relevance = [
            c for c in citations 
            if any(c.source_file == r.source_location.file_path for r in high_relevance_sources)
        ]
        
        if len(cited_high_relevance) < len(high_relevance_sources):
            report['issues'].append("Some high-relevance sources not cited")
        
        # Generate recommendations
        if report['coverage_rate'] < 0.7:
            report['recommendations'].append("Consider citing more diverse sources")
        
        if report['average_confidence'] < 0.6:
            report['recommendations'].append("Review citation matching algorithm parameters")
        
        return report