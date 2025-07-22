# PDF Heading Detection System

## Directory Structure

```
pdf-heading-detector/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── pdf_parser.py           # PDF text/layout extraction
│   ├── heading_detector.py     # Main heading detection logic
│   ├── formatting_analyzer.py  # Font clustering & formatting analysis
│   ├── semantic_analyzer.py    # NLP/ML heading detection
│   ├── cross_page_analyzer.py  # Cross-page validation
│   └── utils.py                # Helper functions
├── models/                     # Downloaded ML models (≤200MB)
├── requirements.txt
├── Dockerfile
├── README.md
└── tests/
    ├── __init__.py
    ├── test_pdf_parser.py
    ├── test_heading_detector.py
    └── sample_pdfs/
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main pipeline: `python src/main.py`

## Approach

- [ ] Add approach details here

## Usage

- [ ] Add usage instructions here 