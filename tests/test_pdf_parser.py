import os
from src.pdf_parser import PDFParser

def test_parse_sample_pdf():
    parser = PDFParser(debug=True)
    sample_pdf = os.path.join(os.path.dirname(__file__), 'sample_pdfs', 'sample.pdf')
    if not os.path.exists(sample_pdf):
        assert True  # Skip if no sample
        return
    pages = parser.parse_pdf(sample_pdf)
    assert len(pages) > 0
    assert all(hasattr(p, 'text_elements') for p in pages) 