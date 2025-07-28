import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from src.heading_detector import HeadingDetector

def process_all_pdfs(input_dir: str, output_dir: str, debug: bool = False):
    # Set model directory for semantic analyzer
    model_dir = os.environ.get('MODEL_DIR', '/app/models')
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Please ensure the model is downloaded and available.", file=sys.stderr)
        sys.exit(1)
    os.environ['MODEL_DIR'] = model_dir
    detector = HeadingDetector(debug=debug)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_path.glob('*.pdf'):
        try:
            result = detector.detect_headings(str(pdf_file))
            output_json = {
                "title": result.title,
                "outline": [
                    {"level": f"H{h.level}", "text": h.text, "page": h.page_number} for h in result.headings
                ]
            }
            out_file = output_path / (pdf_file.stem + ".json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            if debug:
                print(f"Processed {pdf_file.name} -> {out_file.name}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Use local folders for local runs, /app folders for Docker
    input_dir = os.environ.get("PDF_INPUT_DIR", "input")
    output_dir = os.environ.get("PDF_OUTPUT_DIR", "output")
    debug = os.environ.get("DEBUG", "0") == "1"
    process_all_pdfs(input_dir, output_dir, debug=debug) 