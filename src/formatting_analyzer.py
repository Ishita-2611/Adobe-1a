"""
Formatting Analyzer Module
Analyzes font sizes, styles, and layout to identify headings.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.pdf_parser import TextElement, PageInfo

@dataclass
class HeadingCandidate:
    """Represents a potential heading with confidence score."""
    text: str
    level: int  # 1=H1, 2=H2, 3=H3
    page_number: int
    confidence: float
    font_size: float
    is_bold: bool
    is_centered: bool
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            'text': self.text,
            'level': self.level,
            'page': self.page_number,
            'confidence': round(self.confidence, 3),
            'font_size': self.font_size,
            'is_bold': self.is_bold,
            'is_centered': self.is_centered,
            'position': self.position
        }

class FontCluster:
    """Represents a cluster of similar font sizes."""
    
    def __init__(self, font_size: float, count: int, elements: List[TextElement]):
        self.font_size = font_size
        self.count = count
        self.elements = elements
        self.is_heading_cluster = False
        self.heading_level = None

class FormattingAnalyzer:
    """Analyzes document formatting to identify headings."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def analyze_formatting(self, pages: List[PageInfo]) -> List[HeadingCandidate]:
        """
        Analyze document formatting to identify heading candidates.
        
        Args:
            pages: List of parsed pages
            
        Returns:
            List of heading candidates with confidence scores
        """
        all_elements = self._get_all_elements(pages)
        
        # Step 1: Cluster font sizes
        font_clusters = self._cluster_font_sizes(all_elements)
        
        # Step 2: Identify heading clusters
        heading_clusters = self._identify_heading_clusters(font_clusters)
        
        # Step 3: Extract heading candidates
        candidates = self._extract_heading_candidates(heading_clusters)
        
        # Step 4: Score and filter candidates
        scored_candidates = self._score_candidates(candidates)
        
        if self.debug:
            self.logger.info(f"Found {len(scored_candidates)} heading candidates")
            
        return scored_candidates
    
    def _get_all_elements(self, pages: List[PageInfo]) -> List[TextElement]:
        """Get all text elements from all pages."""
        all_elements = []
        for page in pages:
            all_elements.extend(page.text_elements)
        return all_elements
    
    def _cluster_font_sizes(self, elements: List[TextElement]) -> List[FontCluster]:
        """Cluster elements by font size."""
        if not elements:
            return []
        
        # Count font sizes
        font_size_counts = Counter(elem.font_size for elem in elements)
        
        # Group elements by font size
        font_groups = {}
        for elem in elements:
            if elem.font_size not in font_groups:
                font_groups[elem.font_size] = []
            font_groups[elem.font_size].append(elem)
        
        # Create clusters
        clusters = []
        for font_size, count in font_size_counts.items():
            cluster = FontCluster(
                font_size=font_size,
                count=count,
                elements=font_groups[font_size]
            )
            clusters.append(cluster)
        
        # Sort by font size (descending)
        clusters.sort(key=lambda x: x.font_size, reverse=True)
        
        return clusters
    
    def _identify_heading_clusters(self, font_clusters: List[FontCluster]) -> List[FontCluster]:
        """Identify which font clusters likely represent headings."""
        if not font_clusters:
            return []
        
        # Calculate statistics
        all_sizes = [cluster.font_size for cluster in font_clusters]
        max_size = max(all_sizes)
        avg_size = sum(all_sizes) / len(all_sizes)
        
        # Identify heading clusters
        heading_clusters = []
        heading_level = 1
        
        for cluster in font_clusters:
            # Skip if too many elements (likely body text)
            if cluster.count > 50:
                continue
                
            # Check if significantly larger than average
            if cluster.font_size > avg_size * 1.2:
                cluster.is_heading_cluster = True
                cluster.heading_level = heading_level
                heading_clusters.append(cluster)
                
                if heading_level < 3:  # Limit to H1, H2, H3
                    heading_level += 1
            
            # Also check for bold text with reasonable size
            bold_count = sum(1 for elem in cluster.elements if elem.is_bold)
            if bold_count > cluster.count * 0.5 and cluster.font_size > avg_size * 0.9:
                if not cluster.is_heading_cluster:
                    cluster.is_heading_cluster = True
                    cluster.heading_level = min(heading_level, 3)
                    heading_clusters.append(cluster)
                    
                    if heading_level < 3:
                        heading_level += 1
        
        return heading_clusters
    
    def _extract_heading_candidates(self, heading_clusters: List[FontCluster]) -> List[HeadingCandidate]:
        """Extract heading candidates from heading clusters."""
        candidates = []
        
        for cluster in heading_clusters:
            for elem in cluster.elements:
                # Filter out very short or very long text
                text_length = len(elem.text.strip())
                if text_length < 3 or text_length > 200:
                    continue
                
                # Create heading candidate
                candidate = HeadingCandidate(
                    text=elem.text.strip(),
                    level=cluster.heading_level,
                    page_number=elem.page_number,
                    confidence=0.5,  # Initial confidence, will be updated
                    font_size=elem.font_size,
                    is_bold=elem.is_bold,
                    is_centered=elem.is_centered,
                    position=(elem.x0, elem.y0, elem.x1, elem.y1)
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _score_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Score heading candidates based on various factors."""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_confidence_score(candidate)
            candidate.confidence = score
            
            # Filter out low-confidence candidates
            if score >= 0.3:
                scored_candidates.append(candidate)
        
        # Sort by page number, then by confidence
        scored_candidates.sort(key=lambda x: (x.page_number, -x.confidence))
        
        return scored_candidates
    
    def _calculate_confidence_score(self, candidate: HeadingCandidate) -> float:
        """Calculate confidence score for a heading candidate."""
        score = 0.5  # Base score
        
        # Font size factor
        if candidate.font_size > 16:
            score += 0.2
        elif candidate.font_size > 14:
            score += 0.1
        
        # Bold text factor
        if candidate.is_bold:
            score += 0.15
        
        # Centered text factor
        if candidate.is_centered:
            score += 0.1
        
        # Text length factor (headings are usually concise)
        text_length = len(candidate.text)
        if 10 <= text_length <= 80:
            score += 0.1
        elif text_length > 150:
            score -= 0.2
        
        # Capitalization factor
        if candidate.text.isupper():
            score += 0.1
        elif candidate.text.istitle():
            score += 0.05
        
        # Avoid common body text patterns
        if any(word in candidate.text.lower() for word in ['the', 'and', 'or', 'but']):
            score -= 0.1
        
        # Punctuation factor (headings usually don't end with periods)
        if candidate.text.endswith('.'):
            score -= 0.05
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def get_font_statistics(self, pages: List[PageInfo]) -> Dict:
        """Get detailed font statistics for debugging."""
        all_elements = self._get_all_elements(pages)
        
        if not all_elements:
            return {}
        
        font_sizes = [elem.font_size for elem in all_elements]
        font_size_counts = Counter(font_sizes)
        
        return {
            'total_elements': len(all_elements),
            'unique_font_sizes': len(set(font_sizes)),
            'font_size_distribution': dict(font_size_counts.most_common(10)),
            'size_range': {
                'min': min(font_sizes),
                'max': max(font_sizes),
                'avg': sum(font_sizes) / len(font_sizes),
                'std': np.std(font_sizes)
            },
            'formatting_stats': {
                'bold_count': sum(1 for elem in all_elements if elem.is_bold),
                'italic_count': sum(1 for elem in all_elements if elem.is_italic),
                'centered_count': sum(1 for elem in all_elements if elem.is_centered)
            }
        } 