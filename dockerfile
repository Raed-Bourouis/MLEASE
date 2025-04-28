# Dockerfile

FROM python:3.11-slim

# Install system dependencies needed by Prophet, XGBoost, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libjpeg-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
