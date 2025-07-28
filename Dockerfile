# Use official NVIDIA CUDA base image with Python 3.11 support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    libffi-dev libbz2-dev libuuid1 libssl-dev \
    zlib1g-dev libncurses5-dev libsqlite3-dev \
    libreadline-dev libtk8.6-dev xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose a port if needed (optional)
# EXPOSE 8000

# Default command
CMD ["python", "main.py"]
