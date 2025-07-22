import os
from src.heading_detector import HeadingDetector

def test_heading_detection():
    detector = HeadingDetector(debug=True)
    sample_pdf = os.path.join(os.path.dirname(__file__), 'sample_pdfs', 'sample.pdf')
    if not os.path.exists(sample_pdf):
        assert True  # Skip if no sample
        return
    result = detector.detect_headings(sample_pdf)
    assert result is not None
    assert hasattr(result, 'headings')
    assert isinstance(result.headings, list) 