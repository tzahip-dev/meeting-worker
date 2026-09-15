# Meeting Capture Worker — RunPod Serverless
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --quiet -r /tmp/requirements.txt

# Reinstall torchvision with CUDA 12.1 kernel (pyannote needs nms)
# --no-deps prevents upgrading torch (which pulls CUDA 13 runtime)
RUN pip install --no-cache-dir --quiet --no-deps torchvision==0.17.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY vast_worker.py /vast_worker.py
COPY runpod_handler.py /runpod_handler.py

ENV MODELS_DIR=/models

CMD ["python", "-u", "/runpod_handler.py"]