"""
Semantic Analyzer Module
Uses NLP/ML to identify headings based on semantic content.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from src.pdf_parser import TextElement, PageInfo
from src.formatting_analyzer import HeadingCandidate
import os

class SemanticAnalyzer:
    """Analyzes text semantically to identify headings."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Initialize models (lazy loading)
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = None
        
        # Known heading patterns
        self.heading_patterns = [
            r'^chapter\s+\d+',
            r'^section\s+\d+',
            r'^part\s+\d+',
            r'^\d+\.\s+',
            r'^\d+\.\d+\s+',
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z\s]+:$',  # Title Case with colon
        ]
        
        # Common heading words
        self.heading_keywords = [
            'introduction', 'background', 'methodology', 'results', 'discussion',
            'conclusion', 'abstract', 'summary', 'overview', 'chapter', 'section',
            'part', 'appendix', 'references', 'bibliography', 'acknowledgments',
            'table', 'figure', 'analysis', 'findings', 'implications', 'limitations'
        ]
    
    def _load_models(self):
        """Lazy load ML models from local /app/models directory."""
        if self.tokenizer is None:
            try:
                model_dir = os.environ.get('MODEL_DIR', './models')
                # Try XLM-Roberta for multilingual, fallback to distilbert
                if os.path.exists(os.path.join(model_dir, 'xlm-roberta-base')):
                    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, 'xlm-roberta-base'))
                    self.model = AutoModel.from_pretrained(os.path.join(model_dir, 'xlm-roberta-base'))
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, 'distilbert-base-uncased'))
                    self.model = AutoModel.from_pretrained(os.path.join(model_dir, 'distilbert-base-uncased'))
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                if self.debug:
                    self.logger.info(f"Loaded semantic models from {model_dir}")
            except Exception as e:
                self.logger.error(f"Failed to load semantic models: {e}")
                raise
    
    def analyze_semantic_headings(self, pages: List[PageInfo]) -> List[HeadingCandidate]:
        """
        Analyze text semantically to identify heading candidates.
        
        Args:
            pages: List of parsed pages
            
        Returns:
            List of heading candidates identified by semantic analysis
        """
        all_elements = self._get_all_elements(pages)
        
        # Step 1: Pattern-based detection
        pattern_candidates = self._detect_pattern_headings(all_elements)
        
        # Step 2: Keyword-based detection
        keyword_candidates = self._detect_keyword_headings(all_elements)
        
        # Step 3: ML-based detection (if models are available)
        ml_candidates = []
        try:
            self._load_models()
            ml_candidates = self._detect_ml_headings(all_elements)
        except Exception as e:
            self.logger.warning(f"ML-based detection failed: {e}")
        
        # Step 4: Combine and deduplicate candidates
        all_candidates = pattern_candidates + keyword_candidates + ml_candidates
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        # Step 5: Score candidates
        scored_candidates = self._score_semantic_candidates(unique_candidates)
        
        if self.debug:
            self.logger.info(f"Found {len(scored_candidates)} semantic heading candidates")
        
        return scored_candidates
    
    def _get_all_elements(self, pages: List[PageInfo]) -> List[TextElement]:
        """Get all text elements from all pages."""
        all_elements = []
        for page in pages:
            all_elements.extend(page.text_elements)
        return all_elements
    
    def _detect_pattern_headings(self, elements: List[TextElement]) -> List[HeadingCandidate]:
        """Detect headings using regex patterns."""
        candidates = []
        
        for elem in elements:
            text = elem.text.strip()
            
            # Check against heading patterns
            for pattern in self.heading_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    level = self._determine_level_from_pattern(text, pattern)
                    
                    candidate = HeadingCandidate(
                        text=text,
                        level=level,
                        page_number=elem.page_number,
                        confidence=0.7,  # High confidence for pattern matches
                        font_size=elem.font_size,
                        is_bold=elem.is_bold,
                        is_centered=elem.is_centered,
                        position=(elem.x0, elem.y0, elem.x1, elem.y1)
                    )
                    
                    candidates.append(candidate)
                    break
        
        return candidates
    
    def _detect_keyword_headings(self, elements: List[TextElement]) -> List[HeadingCandidate]:
        """Detect headings using keyword matching."""
        candidates = []
        
        for elem in elements:
            text = elem.text.strip().lower()
            
            # Skip very long text (likely body text)
            if len(text) > 100:
                continue
            
            # Check for heading keywords
            heading_score = 0
            for keyword in self.heading_keywords:
                if keyword in text:
                    heading_score += 1
            
            # Check for title case
            if elem.text.istitle():
                heading_score += 0.5
            
            # Check for short, concise text
            if 5 <= len(text) <= 50:
                heading_score += 0.3
            
            # Create candidate if score is high enough
            if heading_score > 0.5:
                level = self._determine_level_from_keywords(text)
                
                candidate = HeadingCandidate(
                    text=elem.text.strip(),
                    level=level,
                    page_number=elem.page_number,
                    confidence=min(heading_score * 0.4, 0.8),  # Convert to confidence
                    font_size=elem.font_size,
                    is_bold=elem.is_bold,
                    is_centered=elem.is_centered,
                    position=(elem.x0, elem.y0, elem.x1, elem.y1)
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _detect_ml_headings(self, elements: List[TextElement]) -> List[HeadingCandidate]:
        """Detect headings using ML models."""
        candidates = []
        
        # Prepare texts for analysis
        texts = []
        valid_elements = []
        
        for elem in elements:
            text = elem.text.strip()
            if 3 <= len(text) <= 200:  # Reasonable length for headings
                texts.append(text)
                valid_elements.append(elem)
        
        if not texts:
            return candidates
        
        try:
            # Use TF-IDF to find distinctive text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate heading probability based on TF-IDF scores
            for i, (text, elem) in enumerate(zip(texts, valid_elements)):
                tfidf_score = np.mean(tfidf_matrix[i].toarray()[0])
                
                # Higher TF-IDF score might indicate distinctive text (potential heading)
                if tfidf_score > 0.1:  # Threshold for distinctive text
                    level = self._determine_level_from_context(text, elem)
                    
                    candidate = HeadingCandidate(
                        text=text,
                        level=level,
                        page_number=elem.page_number,
                        confidence=min(tfidf_score * 2, 0.9),
                        font_size=elem.font_size,
                        is_bold=elem.is_bold,
                        is_centered=elem.is_centered,
                        position=(elem.x0, elem.y0, elem.x1, elem.y1)
                    )
                    
                    candidates.append(candidate)
        
        except Exception as e:
            self.logger.error(f"ML-based heading detection failed: {e}")
        
        return candidates
    
    def _deduplicate_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Remove duplicate candidates based on text similarity."""
        if not candidates:
            return candidates
        
        unique_candidates = []
        seen_texts = set()
        
        for candidate in candidates:
            # Simple deduplication based on exact text match
            if candidate.text not in seen_texts:
                seen_texts.add(candidate.text)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _score_semantic_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Score semantic candidates and filter low-confidence ones."""
        scored_candidates = []
        
        for candidate in candidates:
            # Additional semantic scoring
            semantic_score = self._calculate_semantic_score(candidate)
            
            # Combine with existing confidence
            final_confidence = (candidate.confidence + semantic_score) / 2
            candidate.confidence = final_confidence
            
            # Filter out low-confidence candidates
            if final_confidence >= 0.4:
                scored_candidates.append(candidate)
        
        # Sort by confidence descending
        scored_candidates.sort(key=lambda x: -x.confidence)
        
        return scored_candidates
    
    def _calculate_semantic_score(self, candidate: HeadingCandidate) -> float:
        """Calculate semantic score for a heading candidate."""
        score = 0.5  # Base score
        text = candidate.text.lower()
        
        # Length factor
        if 10 <= len(candidate.text) <= 80:
            score += 0.1
        elif len(candidate.text) > 150:
            score -= 0.2
        
        # Capitalization patterns
        if candidate.text.isupper():
            score += 0.15
        elif candidate.text.istitle():
            score += 0.1
        
        # Numbering patterns
        if re.match(r'^\d+\.', candidate.text):
            score += 0.2
        elif re.match(r'^\w+\s+\d+', candidate.text):
            score += 0.15
        
        # Keyword presence
        keyword_count = sum(1 for keyword in self.heading_keywords if keyword in text)
        score += min(keyword_count * 0.1, 0.3)
        
        # Avoid common body text indicators
        body_indicators = ['the', 'and', 'or', 'but', 'however', 'therefore', 'this', 'that']
        body_count = sum(1 for indicator in body_indicators if indicator in text)
        score -= min(body_count * 0.05, 0.2)
        
        # Punctuation patterns
        if candidate.text.endswith(':'):
            score += 0.1
        elif candidate.text.endswith('.') and len(candidate.text) > 50:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _determine_level_from_pattern(self, text: str, pattern: str) -> int:
        """Determine heading level based on regex pattern."""
        if 'chapter' in pattern.lower():
            return 1
        elif 'section' in pattern.lower():
            return 2
        elif r'^\d+\.\d+' in pattern:
            return 3
        elif r'^\d+\.' in pattern:
            return 2
        else:
            return 2  # Default to H2
    
    def _determine_level_from_keywords(self, text: str) -> int:
        """Determine heading level based on keywords."""
        h1_keywords = ['introduction', 'abstract', 'conclusion', 'chapter']
        h2_keywords = ['background', 'methodology', 'results', 'discussion']
        h3_keywords = ['analysis', 'findings', 'table', 'figure']
        
        text_lower = text.lower()
        
        for keyword in h1_keywords:
            if keyword in text_lower:
                return 1
        
        for keyword in h2_keywords:
            if keyword in text_lower:
                return 2
        
        for keyword in h3_keywords:
            if keyword in text_lower:
                return 3
        
        return 2  # Default to H2
    
    def _determine_level_from_context(self, text: str, element: TextElement) -> int:
        """Determine heading level based on context and formatting."""
        # Use font size as primary indicator
        if element.font_size > 18:
            return 1
        elif element.font_size > 14:
            return 2
        else:
            return 3
    
    def enhance_formatting_candidates(self, 
                                    formatting_candidates: List[HeadingCandidate],
                                    pages: List[PageInfo]) -> List[HeadingCandidate]:
        """
        Enhance formatting-based candidates with semantic analysis.
        
        Args:
            formatting_candidates: Candidates from formatting analysis
            pages: Original page data
            
        Returns:
            Enhanced candidates with improved confidence scores
        """
        enhanced_candidates = []
        
        for candidate in formatting_candidates:
            # Calculate semantic score
            semantic_score = self._calculate_semantic_score(candidate)
            
            # Combine formatting and semantic scores
            combined_confidence = (candidate.confidence + semantic_score) / 2
            
            # Apply semantic boost for strong semantic indicators
            if semantic_score > 0.7:
                combined_confidence = min(combined_confidence + 0.1, 1.0)
            
            # Update candidate
            candidate.confidence = combined_confidence
            enhanced_candidates.append(candidate)
        
        return enhanced_candidates
    
    def get_semantic_statistics(self, pages: List[PageInfo]) -> Dict:
        """Get semantic analysis statistics for debugging."""
        all_elements = self._get_all_elements(pages)
        
        if not all_elements:
            return {}
        
        # Analyze text patterns
        pattern_matches = 0
        keyword_matches = 0
        
        for elem in all_elements:
            text = elem.text.strip()
            
            # Check patterns
            for pattern in self.heading_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
                    break
            
            # Check keywords
            text_lower = text.lower()
            for keyword in self.heading_keywords:
                if keyword in text_lower:
                    keyword_matches += 1
                    break
        
        return {
            'total_elements': len(all_elements),
            'pattern_matches': pattern_matches,
            'keyword_matches': keyword_matches,
            'heading_patterns_count': len(self.heading_patterns),
            'heading_keywords_count': len(self.heading_keywords),
            'avg_text_length': sum(len(elem.text) for elem in all_elements) / len(all_elements)
        } 