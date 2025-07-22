"""
PDF Parser Module
Extracts text, layout, and formatting information from PDF files.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LTPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
import re

@dataclass
class TextElement:
    """Represents a text element with formatting and position info."""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def is_centered(self) -> bool:
        """Rough estimation if text is centered on page."""
        return self.x0 > 100 and self.x1 < 400  # Adjust based on typical page width

@dataclass
class PageInfo:
    """Represents a page with its text elements."""
    page_number: int
    width: float
    height: float
    text_elements: List[TextElement]

class PDFParser:
    """Main PDF parsing class."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf(self, pdf_path: str) -> List[PageInfo]:
        """
        Parse PDF and extract text elements with formatting info.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PageInfo objects containing text elements
        """
        pages = []
        
        try:
            for page_num, page in enumerate(extract_pages(pdf_path), 1):
                page_info = self._parse_page(page, page_num)
                pages.append(page_info)
                
                if self.debug:
                    self.logger.info(f"Parsed page {page_num}: {len(page_info.text_elements)} elements")
                    
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
            
        return pages
    
    def _parse_page(self, page: LTPage, page_num: int) -> PageInfo:
        """Parse a single page and extract text elements."""
        text_elements = []
        
        for element in page:
            if isinstance(element, LTTextBox):
                for line in element:
                    if isinstance(line, LTTextLine):
                        text_element = self._parse_text_line(line, page_num)
                        if text_element and text_element.text.strip():
                            text_elements.append(text_element)
        
        return PageInfo(
            page_number=page_num,
            width=page.width,
            height=page.height,
            text_elements=text_elements
        )
    
    def _parse_text_line(self, line: LTTextLine, page_num: int) -> Optional[TextElement]:
        """Parse a text line and extract formatting information."""
        text = line.get_text().strip()
        if not text:
            return None
            
        # Get font information from the first character
        font_info = self._get_font_info(line)
        
        return TextElement(
            text=text,
            font_size=font_info['size'],
            font_name=font_info['name'],
            is_bold=font_info['is_bold'],
            is_italic=font_info['is_italic'],
            x0=line.x0,
            y0=line.y0,
            x1=line.x1,
            y1=line.y1,
            page_number=page_num
        )
    
    def _get_font_info(self, line: LTTextLine) -> Dict:
        """Extract font information from a text line."""
        font_sizes = []
        font_names = []
        
        for char in line:
            if isinstance(char, LTChar):
                font_sizes.append(char.height)
                font_names.append(char.fontname)
        
        if not font_sizes:
            return {
                'size': 12.0,
                'name': 'unknown',
                'is_bold': False,
                'is_italic': False
            }
        
        # Use the most common font size and name
        avg_size = sum(font_sizes) / len(font_sizes)
        most_common_font = max(set(font_names), key=font_names.count)
        
        return {
            'size': avg_size,
            'name': most_common_font,
            'is_bold': self._is_bold_font(most_common_font),
            'is_italic': self._is_italic_font(most_common_font)
        }
    
    def _is_bold_font(self, font_name: str) -> bool:
        """Determine if font is bold based on name."""
        bold_indicators = ['bold', 'black', 'heavy', 'demi']
        return any(indicator in font_name.lower() for indicator in bold_indicators)
    
    def _is_italic_font(self, font_name: str) -> bool:
        """Determine if font is italic based on name."""
        italic_indicators = ['italic', 'oblique', 'slant']
        return any(indicator in font_name.lower() for indicator in italic_indicators)
    
    def get_document_title(self, pages: List[PageInfo]) -> Optional[str]:
        """
        Extract document title from the first page.
        
        Args:
            pages: List of parsed pages
            
        Returns:
            Document title if found, None otherwise
        """
        if not pages:
            return None
            
        first_page = pages[0]
        
        # Find the largest font size on the first page
        max_font_size = max(
            (elem.font_size for elem in first_page.text_elements),
            default=0
        )
        
        # Get text elements with the largest font size
        title_candidates = [
            elem for elem in first_page.text_elements
            if elem.font_size >= max_font_size - 1  # Allow small variance
        ]
        
        if title_candidates:
            # Sort by position (top to bottom, left to right)
            title_candidates.sort(key=lambda x: (-x.y0, x.x0))
            
            # Combine text from title candidates
            title_text = ' '.join(elem.text for elem in title_candidates[:3])  # Max 3 elements
            
            # Clean up title
            title_text = re.sub(r'\s+', ' ', title_text).strip()
            
            return title_text if len(title_text) > 3 else None
        
        return None
    
    def get_text_statistics(self, pages: List[PageInfo]) -> Dict:
        """Get statistics about text elements in the document."""
        all_elements = []
        for page in pages:
            all_elements.extend(page.text_elements)
        
        if not all_elements:
            return {}
        
        font_sizes = [elem.font_size for elem in all_elements]
        
        return {
            'total_elements': len(all_elements),
            'total_pages': len(pages),
            'font_sizes': {
                'min': min(font_sizes),
                'max': max(font_sizes),
                'avg': sum(font_sizes) / len(font_sizes),
                'unique': len(set(font_sizes))
            },
            'font_names': list(set(elem.font_name for elem in all_elements)),
            'bold_count': sum(1 for elem in all_elements if elem.is_bold),
            'italic_count': sum(1 for elem in all_elements if elem.is_italic)
        } 