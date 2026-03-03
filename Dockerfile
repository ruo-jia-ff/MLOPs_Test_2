# Stage 1: Build stage
FROM python:3.11-slim as build-stage

# Install dependencies to fix OpenCV error (libGL, libsm6, libxext6, and libglib2.0-0)
RUN apt-get update && apt-get install -y \
    git \
    bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/ruo-jia-ff/MLOPs_TOOLS.git /MLOPs_TOOLS
RUN git clone https://github.com/ruo-jia-ff/MLOPs_Test_2.git /MLOPs_TEST

# Stage 2: Final stage
FROM python:3.11-slim

# Install dependencies to fix OpenCV error (libGL, libsm6, libxext6, and libglib2.0-0)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    bash \
    vim \
    nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

COPY --from=build-stage /MLOPs_TOOLS /app
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=build-stage /MLOPs_TEST /app
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the entrypoint for the container to execute the `run_everything.py` script
CMD ["python", "run_everything.py"]
