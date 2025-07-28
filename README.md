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
        ├── sample.pdf
        └── sample.json
```

## Approach

This system extracts a structured outline (Title, H1, H2, H3 headings with page numbers) from PDFs using a hybrid approach:

- **Formatting Analysis:** Clusters font sizes, boldness, and centering to find heading candidates. This is robust to most document layouts.
- **Semantic Analysis:** Uses rules, keywords, and a small transformer model (e.g., DistilBERT or XLM-Roberta for multilingual) to semantically identify headings, even when formatting is inconsistent.
- **Cross-Page Analysis:** Deduplicates and validates headings across pages for accuracy.
- **Output:** For each PDF, outputs a JSON with the required structure.

**Why this approach?**
- Font size alone is unreliable; combining formatting and semantics increases accuracy.
- ML model is small (≤200MB), CPU-only, and loaded locally for offline use.
- Modular design for easy extension and reuse.

## Usage

### Local Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place PDFs in the `input/` directory.
3. Run:
   ```bash
   python src/main.py
   ```
4. Output JSONs will appear in the `output/` directory.

### Docker Run
1. Build the image:
   ```bash
   docker build --platform linux/amd64 -t pdf-heading-detector:latest .
   ```
2. Run the container:
   ```bash
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-heading-detector:latest
   ```

### Input/Output
- **Input:** Place PDFs in `/app/input` (or `input/` locally).
- **Output:** JSONs will be written to `/app/output` (or `output/` locally), one per PDF.

## Models & Multilingual
- Uses a small transformer model (e.g., DistilBERT or XLM-Roberta) stored in `/models`.
- Model is downloaded at build time and loaded locally (no internet required at runtime).
- Supports multilingual heading detection if XLM-Roberta is used.

## Tests
- Run tests with:
  ```bash
  pytest tests/
  ```

## Constraints
- CPU-only, ≤200MB model, ≤10s for 50-page PDF, no internet at runtime.

