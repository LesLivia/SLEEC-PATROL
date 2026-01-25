# Start with the Python 3.11 slim base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Marabou and its Python bindings
# This includes build tools, mathematical libraries, and atomic operation support.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgmp-dev \
    libmpfr-dev \
    libglpk-dev \
    zlib1g-dev \
    pkg-config \
    wget \
    unzip \
    libpython3-dev \
    libatomic1 \
    libgcc-s1 && \
    # Clean up apt caches to keep the image size small
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# numpy is a direct dependency of your script
# maraboupy is the core library. Pinning to 1.0.0 as requested.
# If you decide to try a newer maraboupy version, change this line.
RUN pip install --no-cache-dir numpy tensorflow==2.19.1 onnx==1.17.0 onnxruntime torch==2.8.0 tf2onnx==1.16.1 maraboupy==1.0.0

# Copy your script and the ONNX model into the container
# Make sure your script (e.g., verify_model.py) and the ONNX model
# (carebot_decision_model_v2.onnx) are in the same directory as your Dockerfile
COPY .. /app

# Command to run your script when the container starts
# For this example, let's assume the script is named 'verify_model.py'
CMD ["python", "marabou_test.py"]