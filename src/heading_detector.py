"""
Main Heading Detector Module
Orchestrates the multi-strategy heading detection pipeline.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from src.pdf_parser import PDFParser, PageInfo
from src.formatting_analyzer import FormattingAnalyzer, HeadingCandidate
from src.semantic_analyzer import SemanticAnalyzer
from src.cross_page_analyzer import CrossPageAnalyzer

@dataclass
class DetectionResult:
    """Result of heading detection process."""
    title: Optional[str]
    headings: List[HeadingCandidate]
    metadata: Dict
    processing_time: float
    method_used: str

class HeadingDetector:
    """Main heading detection orchestrator."""
    
    def __init__(self, 
                 semantic_model: str = "distilbert-base-uncased",
                 debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pdf_parser = PDFParser(debug=debug)
        self.formatting_analyzer = FormattingAnalyzer(debug=debug)
        self.semantic_analyzer = SemanticAnalyzer(model_name=semantic_model, debug=debug)
        self.cross_page_analyzer = CrossPageAnalyzer(debug=debug)
        
        # Detection thresholds
        self.min_confidence = 0.3
        self.max_headings_per_page = 10
        
    def detect_headings(self, pdf_path: str) -> DetectionResult:
        """
        Main heading detection pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DetectionResult with headings and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse PDF
            if self.debug:
                self.logger.info(f"Parsing PDF: {pdf_path}")
            
            pages = self.pdf_parser.parse_pdf(pdf_path)
            
            if not pages:
                raise ValueError("No pages found in PDF")
            
            # Step 2: Extract title
            title = self.pdf_parser.get_document_title(pages)
            
            # Step 3: Multi-strategy heading detection
            all_candidates = self._run_detection_strategies(pages)
            
            # Step 4: Cross-page analysis and validation
            validated_candidates = self.cross_page_analyzer.analyze_cross_page_headings(
                all_candidates, pages
            )
            
            # Step 5: Final filtering and ranking
            final_headings = self._finalize_headings(validated_candidates)
            
            # Step 6: Generate metadata
            processing_time = time.time() - start_time
            metadata = self._generate_metadata(pages, final_headings, processing_time)
            
            # Determine primary method used
            method_used = self._determine_primary_method(final_headings)
            
            result = DetectionResult(
                title=title,
                headings=final_headings,
                metadata=metadata,
                processing_time=processing_time,
                method_used=method_used
            )
            
            if self.debug:
                self.logger.info(f"Detection completed in {processing_time:.2f}s")
                self.logger.info(f"Found {len(final_headings)} headings using {method_used}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in heading detection: {e}")
            raise
    
    def _run_detection_strategies(self, pages: List[PageInfo]) -> List[HeadingCandidate]:
        """Run multiple detection strategies and combine results."""
        all_candidates = []
        
        # Strategy 1: Formatting-based detection
        if self.debug:
            self.logger.info("Running formatting-based detection...")
        
        formatting_candidates = self.formatting_analyzer.analyze_formatting(pages)
        
        # Strategy 2: Semantic-based detection
        if self.debug:
            self.logger.info("Running semantic-based detection...")
        
        semantic_candidates = self.semantic_analyzer.analyze_semantic_headings(pages)
        
        # Strategy 3: Enhance formatting candidates with semantic analysis
        if self.debug:
            self.logger.info("Enhancing formatting candidates with semantic analysis...")
        
        enhanced_formatting = self.semantic_analyzer.enhance_formatting_candidates(
            formatting_candidates, pages
        )
        
        # Combine all candidates
        all_candidates.extend(enhanced_formatting)
        all_candidates.extend(semantic_candidates)
        
        # Remove duplicates
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        if self.debug:
            self.logger.info(f"Combined strategies: {len(unique_candidates)} unique candidates")
        
        return unique_candidates
    
    def _deduplicate_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Remove duplicate candidates based on text and position."""
        if not candidates:
            return candidates
        
        unique_candidates = []
        seen_combinations = set()
        
        for candidate in candidates:
            # Create unique key based on text and approximate position
            key = (
                candidate.text.lower().strip(),
                candidate.page_number,
                round(candidate.position[1], -1)  # Round Y position to nearest 10
            )
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_candidates.append(candidate)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing in enumerate(unique_candidates):
                    if (existing.text.lower().strip() == candidate.text.lower().strip() and
                        existing.page_number == candidate.page_number):
                        if candidate.confidence > existing.confidence:
                            unique_candidates[i] = candidate
                        break
        
        return unique_candidates
    
    def _finalize_headings(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply final filtering and ranking to heading candidates."""
        
        # Filter by confidence
        filtered_candidates = [
            c for c in candidates 
            if c.confidence >= self.min_confidence
        ]
        
        # Sort by page number, then by position (top to bottom)
        filtered_candidates.sort(key=lambda x: (x.page_number, -x.position[1]))
        
        # Apply per-page limits
        final_headings = self._apply_per_page_limit(filtered_candidates)
        return final_headings
    
    def _apply_per_page_limit(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Limit the number of headings per page to avoid noise."""
        from collections import defaultdict
        page_groups = defaultdict(list)
        for c in candidates:
            page_groups[c.page_number].append(c)
        limited = []
        for page, group in page_groups.items():
            group.sort(key=lambda x: -x.confidence)
            limited.extend(group[:self.max_headings_per_page])
        # Resort by page and position
        limited.sort(key=lambda x: (x.page_number, -x.position[1]))
        return limited
    
    def _generate_metadata(self, pages: List[PageInfo], headings: List[HeadingCandidate], processing_time: float) -> Dict:
        """Generate metadata for the detection result."""
        return {
            "total_pages": len(pages),
            "total_headings": len(headings),
            "processing_time": processing_time,
        }
    
    def _determine_primary_method(self, headings: List[HeadingCandidate]) -> str:
        """Determine which method contributed most headings."""
        # Placeholder: could be improved to track source per candidate
        return "hybrid" 