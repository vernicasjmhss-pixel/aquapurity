# Aquapurity - container image
# ---------------------------------------------------------------------------
# Runs the FastAPI backend with the STGNN models on CPU and serves the static
# dashboard from the same origin. Designed to fit a free 512 MB instance.
#
#   docker build -t aquapurity .
#   docker run -p 8000:8000 aquapurity      ->  http://localhost:8000

FROM python:3.12-slim

# - libgomp1: OpenMP runtime that the CPU PyTorch wheel links against
# - no pip cache / no .pyc keeps the image and runtime footprint down
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/aquapurity

# requirements first so the (large) torch layer is cached across code changes
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# application: source, pre-trained checkpoints/data, and the dashboard
COPY src  ./src
COPY app  ./app
COPY data ./data

EXPOSE 8000

# hosts such as Render/Cloud Run inject $PORT; fall back to 8000 locally
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
