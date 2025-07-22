"""
Cross-Page Analyzer Module
Analyzes headings across page boundaries and validates context.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
from src.pdf_parser import PageInfo
from src.formatting_analyzer import HeadingCandidate

@dataclass
class CrossPageContext:
    """Context information for cross-page analysis."""
    page_number: int
    previous_headings: List[HeadingCandidate]
    next_headings: List[HeadingCandidate]
    page_transition: bool  # True if heading spans page break

class CrossPageAnalyzer:
    """Analyzes headings across page boundaries."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    def analyze_cross_page_headings(self, 
                                  candidates: List[HeadingCandidate],
                                  pages: List[PageInfo]) -> List[HeadingCandidate]:
        """
        Analyze and validate headings across page boundaries.
        
        Args:
            candidates: Initial heading candidates
            pages: Original page data
            
        Returns:
            Validated and enhanced heading candidates
        """
        if not candidates:
            return candidates
        
        # Step 1: Group candidates by page
        page_groups = self._group_candidates_by_page(candidates)
        
        # Step 2: Detect split headings
        merged_candidates = self._detect_and_merge_split_headings(page_groups, pages)
        
        # Step 3: Validate heading hierarchy
        validated_candidates = self._validate_heading_hierarchy(merged_candidates)
        
        # Step 4: Enhance with cross-page context
        enhanced_candidates = self._enhance_with_context(validated_candidates, pages)
        
        # Step 5: Filter and deduplicate
        final_candidates = self._final_filtering(enhanced_candidates)
        
        if self.debug:
            self.logger.info(f"Cross-page analysis: {len(candidates)} -> {len(final_candidates)} candidates")
        
        return final_candidates
    
    def _group_candidates_by_page(self, candidates: List[HeadingCandidate]) -> Dict[int, List[HeadingCandidate]]:
        """Group heading candidates by page number."""
        page_groups = defaultdict(list)
        
        for candidate in candidates:
            page_groups[candidate.page_number].append(candidate)
        
        # Sort candidates within each page by position
        for page_num in page_groups:
            page_groups[page_num].sort(key=lambda x: (-x.position[1], x.position[0]))  # Top to bottom, left to right
        
        return dict(page_groups)
    
    def _detect_and_merge_split_headings(self, 
                                       page_groups: Dict[int, List[HeadingCandidate]],
                                       pages: List[PageInfo]) -> List[HeadingCandidate]:
        """Detect and merge headings that are split across pages."""
        merged_candidates = []
        page_numbers = sorted(page_groups.keys())
        
        for i, page_num in enumerate(page_numbers):
            current_candidates = page_groups[page_num]
            
            # Check for split headings with next page
            if i < len(page_numbers) - 1:
                next_page_num = page_numbers[i + 1]
                next_candidates = page_groups[next_page_num]
                
                # Look for potential continuations
                for current_candidate in current_candidates:
                    merged_candidate = self._check_heading_continuation(
                        current_candidate, next_candidates, pages
                    )
                    
                    if merged_candidate:
                        merged_candidates.append(merged_candidate)
                        # Remove the continuation from next page candidates
                        next_candidates = [c for c in next_candidates if c.text != merged_candidate.text.split()[-1]]
                        page_groups[next_page_num] = next_candidates
                    else:
                        merged_candidates.append(current_candidate)
            else:
                # Last page, add all candidates
                merged_candidates.extend(current_candidates)
        
        return merged_candidates
    
    def _check_heading_continuation(self, 
                                  current_candidate: HeadingCandidate,
                                  next_candidates: List[HeadingCandidate],
                                  pages: List[PageInfo]) -> Optional[HeadingCandidate]:
        """Check if a heading continues on the next page."""
        
        # Look for incomplete headings (ending with incomplete words or punctuation)
        current_text = current_candidate.text.strip()
        
        # Skip if current heading seems complete
        if current_text.endswith('.') or current_text.endswith(':') or len(current_text) > 80:
            return None
        
        # Look for potential continuation in next page candidates
        for next_candidate in next_candidates:
            if self._is_likely_continuation(current_candidate, next_candidate):
                # Merge the candidates
                merged_text = f"{current_text} {next_candidate.text.strip()}"
                
                # Create merged candidate
                merged_candidate = HeadingCandidate(
                    text=merged_text,
                    level=current_candidate.level,
                    page_number=current_candidate.page_number,
                    confidence=min(current_candidate.confidence, next_candidate.confidence),
                    font_size=current_candidate.font_size,
                    is_bold=current_candidate.is_bold,
                    is_centered=current_candidate.is_centered,
                    position=current_candidate.position
                )
                
                return merged_candidate
        
        return None
    
    def _is_likely_continuation(self, 
                              current: HeadingCandidate,
                              next_candidate: HeadingCandidate) -> bool:
        """Check if next candidate is likely a continuation of current."""
        
        # Check if they have similar formatting
        font_size_similar = abs(current.font_size - next_candidate.font_size) < 2
        style_similar = current.is_bold == next_candidate.is_bold
        
        # Check if next candidate is at the top of the page
        next_at_top = next_candidate.position[1] > 700  # Adjust based on page height
        
        # Check if combined text makes sense
        combined_length = len(current.text) + len(next_candidate.text)
        reasonable_length = 10 <= combined_length <= 150
        
        return font_size_similar and style_similar and next_at_top and reasonable_length
    
    def _validate_heading_hierarchy(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Validate and adjust heading hierarchy."""
        if not candidates:
            return candidates
        
        # Sort by page number and position
        sorted_candidates = sorted(candidates, key=lambda x: (x.page_number, -x.position[1]))
        
        validated_candidates = []
        previous_level = 0
        
        for candidate in sorted_candidates:
            # Adjust level based on hierarchy rules
            adjusted_level = self._adjust_heading_level(candidate, previous_level)
            
            # Create new candidate with adjusted level
            adjusted_candidate = HeadingCandidate(
                text=candidate.text,
                level=adjusted_level,
                page_number=candidate.page_number,
                confidence=candidate.confidence,
                font_size=candidate.font_size,
                is_bold=candidate.is_bold,
                is_centered=candidate.is_centered,
                position=candidate.position
            )
            
            validated_candidates.append(adjusted_candidate)
            previous_level = adjusted_level
        
        return validated_candidates
    
    def _adjust_heading_level(self, candidate: HeadingCandidate, previous_level: int) -> int:
        """Adjust heading level based on hierarchy rules."""
        current_level = candidate.level
        
        # Don't allow skipping levels (e.g., H1 -> H3)
        if previous_level > 0 and current_level > previous_level + 1:
            return previous_level + 1
        
        # Ensure level is within valid range
        return max(1, min(current_level, 3))
    
    def _enhance_with_context(self, 
                            candidates: List[HeadingCandidate],
                            pages: List[PageInfo]) -> List[HeadingCandidate]:
        """Enhance candidates with cross-page context information."""
        enhanced_candidates = []
        
        for i, candidate in enumerate(candidates):
            # Get context
            context = self._get_candidate_context(candidate, candidates, i)
            
            # Enhance confidence based on context
            enhanced_confidence = self._calculate_context_confidence(candidate, context)
            
            # Create enhanced candidate
            enhanced_candidate = HeadingCandidate(
                text=candidate.text,
                level=candidate.level,
                page_number=candidate.page_number,
                confidence=enhanced_confidence,
                font_size=candidate.font_size,
                is_bold=candidate.is_bold,
                is_centered=candidate.is_centered,
                position=candidate.position
            )
            
            enhanced_candidates.append(enhanced_candidate)
        
        return enhanced_candidates
    
    def _get_candidate_context(self, 
                             candidate: HeadingCandidate,
                             all_candidates: List[HeadingCandidate],
                             index: int) -> CrossPageContext:
        """Get context information for a candidate."""
        
        # Get previous headings (within reasonable distance)
        previous_headings = []
        for i in range(max(0, index - 5), index):
            if all_candidates[i].page_number >= candidate.page_number - 2:
                previous_headings.append(all_candidates[i])
        
        # Get next headings (within reasonable distance)
        next_headings = []
        for i in range(index + 1, min(len(all_candidates), index + 6)):
            if all_candidates[i].page_number <= candidate.page_number + 2:
                next_headings.append(all_candidates[i])
        
        # Check if this is a page transition
        page_transition = any(h.page_number != candidate.page_number for h in previous_headings + next_headings)
        
        return CrossPageContext(
            page_number=candidate.page_number,
            previous_headings=previous_headings,
            next_headings=next_headings,
            page_transition=page_transition
        )
    
    def _calculate_context_confidence(self, 
                                    candidate: HeadingCandidate,
                                    context: CrossPageContext) -> float:
        """Calculate confidence based on cross-page context."""
        base_confidence = candidate.confidence
        
        # Context-based adjustments
        context_adjustment = 0.0
        
        # Consistency with nearby headings
        if context.previous_headings:
            level_consistency = self._check_level_consistency(candidate, context.previous_headings)
            context_adjustment += level_consistency * 0.1
        
        # Spacing consistency
        spacing_consistency = self._check_spacing_consistency(candidate, context)
        context_adjustment += spacing_consistency * 0.05
        
        # Avoid over-adjusting
        context_adjustment = max(-0.2, min(0.2, context_adjustment))
        
        return max(0.0, min(1.0, base_confidence + context_adjustment))
    
    def _check_level_consistency(self, 
                               candidate: HeadingCandidate,
                               previous_headings: List[HeadingCandidate]) -> float:
        """Check if heading level is consistent with previous headings."""
        if not previous_headings:
            return 0.0
        
        # Get recent previous levels
        recent_levels = [h.level for h in previous_headings[-3:]]
        
        # Check if current level makes sense
        if candidate.level == 1:
            return 0.5  # H1 is always reasonable
        elif candidate.level == 2:
            return 0.3 if any(level <= 2 for level in recent_levels) else -0.2
        elif candidate.level == 3:
            return 0.2 if any(level <= 3 for level in recent_levels) else -0.3
        
        return 0.0
    
    def _check_spacing_consistency(self, 
                                 candidate: HeadingCandidate,
                                 context: CrossPageContext) -> float:
        """Check if heading spacing is consistent."""
        # This is a simplified check - in practice, you'd analyze actual spacing
        # For now, just check if headings are reasonably distributed
        
        if not context.previous_headings:
            return 0.0
        
        # Check if there's reasonable spacing between headings
        last_heading = context.previous_headings[-1]
        
        # If on same page, check position difference
        if last_heading.page_number == candidate.page_number:
            position_diff = abs(candidate.position[1] - last_heading.position[1])
            if position_diff > 50:  # Reasonable spacing
                return 0.1
            elif position_diff < 20:  # Too close
                return -0.1
        
        return 0.0
    
    def _final_filtering(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Apply final filtering to remove low-quality candidates."""
        filtered_candidates = []
        
        for candidate in candidates:
            # Filter by confidence threshold
            if candidate.confidence < 0.3:
                continue
            
            # Filter by text quality
            if not self._is_valid_heading_text(candidate.text):
                continue
            
            # Filter duplicates (case-insensitive)
            if not self._is_duplicate(candidate, filtered_candidates):
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _is_valid_heading_text(self, text: str) -> bool:
        """Check if text is valid for a heading."""
        text = text.strip()
        
        # Length check
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Avoid common false positives
        false_positives = [
            'page', 'figure', 'table', 'see', 'www', 'http', 'email'
        ]
        
        text_lower = text.lower()
        for fp in false_positives:
            if fp in text_lower and len(text) < 20:
                return False
        
        # Check for reasonable character composition
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False
        
        return True
    
    def _is_duplicate(self, candidate: HeadingCandidate, existing_candidates: List[HeadingCandidate]) -> bool:
        """Check if candidate is a duplicate of existing candidates."""
        for existing in existing_candidates:
            # Exact match (case-insensitive)
            if candidate.text.lower().strip() == existing.text.lower().strip():
                return True
            
            # Very similar text (simple similarity check)
            if self._text_similarity(candidate.text, existing.text) > 0.9:
                return True
        
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(min(text1.count(c), text2.count(c)) for c in set(text1 + text2))
        total_chars = len(text1) + len(text2)
        
        return (2 * common_chars) / total_chars if total_chars > 0 else 0.0 