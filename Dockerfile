# Customer Support Analyzer - Simplified & Fixed
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including Rust compiler
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cargo \
    rustc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python packages with prebuilt wheels (no compilation needed)
RUN pip install --default-timeout=200 \
    --prefer-binary \
    --no-cache-dir \
    -r requirements.txt

# Copy application files
COPY 4_streamlit_app.py .
COPY data/ ./data/
COPY models/ ./models/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit - FIXED FILENAME
CMD ["streamlit", "run", "4_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
