# Use NVIDIA's PyTorch image with Python 3.8
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install tzdata and set the time zone
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    timidity \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-venv \
    python3.8-distutils && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# Create and activate a virtual environment with Python 3.8
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install wheel (necessary for building some packages)
RUN pip install --no-cache-dir wheel

# Install numpy first
RUN pip install --no-cache-dir numpy

# Install chord-extractor without build isolation
RUN pip install --no-cache-dir --no-build-isolation chord-extractor

# Install PyTorch packages from the PyTorch wheel index
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu113

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . .

# (Optional) Expose ports if necessary
# EXPOSE 8000

# (Optional) Set the entry point for your application
# CMD ["python", "your_script.py"]
