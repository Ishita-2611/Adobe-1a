FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Download model at build time (example: distilbert-base-uncased)
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./models'); AutoModel.from_pretrained('distilbert-base-uncased', cache_dir='./models')"

# Set environment variables for input/output
ENV PDF_INPUT_DIR=/app/input
ENV PDF_OUTPUT_DIR=/app/output

# Create input/output dirs
RUN mkdir -p /app/input /app/output

# Entrypoint
CMD ["python", "src/main.py"] 