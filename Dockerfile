FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY constraints.txt requirements.txt /tmp/

# Step 1: pin torch family to CUDA 12.1 (must come first)
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: install remaining deps, constrained against the pinned torch family
RUN pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt

# Step 3: build-time sanity check — fail early, not on the GPU
RUN python -c "\
import torch, torchvision; \
assert torch.__version__.startswith('2.2.0'), f'torch={torch.__version__}'; \
assert torchvision.__version__.startswith('0.17.0'), f'tvm={torchvision.__version__}'; \
assert torch.version.cuda == '12.1', f'cuda={torch.version.cuda}'; \
from torchvision.ops import nms; \
print('OK', torch.__version__, torchvision.__version__, torch.version.cuda)"

COPY vast_worker.py /vast_worker.py
COPY runpod_handler.py /runpod_handler.py

ENV MODELS_DIR=/models

CMD ["python", "-u", "/runpod_handler.py"]
