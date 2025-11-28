# Dockerfile for reproducible Streamlit app environment
# Use the official Python slim image for compatibility with SciPy wheels

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install system packages used by SciPy and other compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy app
COPY . /app

EXPOSE 8501

# Use Streamlit's recommended command (headless)
CMD ["streamlit", "run", "ICR_analysis_tool.py", "--server.headless=true", "--server.port=8501"]
