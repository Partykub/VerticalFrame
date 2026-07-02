FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ffmpeg \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements-gpu.txt \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements-gpu.txt

COPY . .

RUN python3 scripts/download_models.py

ENTRYPOINT ["python3", "auto_reframe.py"]
CMD ["--help"]
