# Meeting Capture Worker — RunPod Serverless
#
# GPU stack is intentionally coherent:
#   base image : CUDA 12.8.1 + cuDNN 9.8 + torch 2.8.0 + Python 3.12 (pre-baked)
#   ctranslate2: >=4.6.3  (first release with official CUDA 12.8 support;
#                           >=4.5.0 requires cuDNN 9, which is what the base ships)
#   pyannote    : 4.0.7   (community-1 pipeline needs pyannote.audio >= 4.0.0 + PLDA)
#   torchcodec  : 0.7.0   (the release built against torch 2.8 — ABI matched)
#
# Do NOT relax the torch pins: an unconstrained `pip install` upgrades torch to a
# newer CUDA build and re-breaks ctranslate2 with a silent segfault at ASR load.

FROM runpod/pytorch:1.3.0-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    WORK_DIR=/worker_work

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/asr /models/diarization /models/embedding /worker_work

COPY constraints.txt requirements.txt /tmp/

# 1) torch family pinned to the base image's CUDA 12.8 wheels
RUN python -m pip install --no-cache-dir --timeout 900 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

# 2) torchcodec is ABI-locked to torch; install alone with --no-deps so it can
#    never drag in a different torch
RUN python -m pip install --no-cache-dir --no-deps torchcodec==0.7.0

# 3) application dependencies (torch family constrained)
RUN python -m pip install --no-cache-dir --timeout 900 \
        -r /tmp/requirements.txt -c /tmp/constraints.txt

# 4) runpod SDK. Its dependency chain pulls `cryptography`, and the base image's
#    cryptography is Debian-managed with no RECORD file — pip cannot uninstall it
#    and aborts the transaction. --ignore-installed sidesteps the uninstall.
RUN python -m pip install --no-cache-dir --timeout 900 \
        --ignore-installed cryptography runpod

# 5) build-time gate — fail the build, not the job
RUN python - <<'PY'
import inspect
import torch, torchvision, torchaudio, torchcodec
import ctranslate2, faster_whisper, speechbrain
import runpod
import pyannote.audio as pa
from pyannote.audio.pipelines import SpeakerDiarization as SD

sig = str(inspect.signature(SD.__init__))
src = inspect.getsource(SD.__init__)
assert "plda" in sig or "plda" in src, "pyannote SpeakerDiarization rejects 'plda'"
assert torch.__version__.startswith("2.8"), f"torch={torch.__version__}"
assert torch.version.cuda == "12.8", f"torch cuda={torch.version.cuda}"
assert pa.__version__.startswith("4."), f"pyannote.audio={pa.__version__}"
assert tuple(int(x) for x in ctranslate2.__version__.split(".")[:2]) >= (4, 6), \
    f"ctranslate2={ctranslate2.__version__} (needs >=4.6.3 for CUDA 12.8)"
print("BUILD-OK torch=%s cuda=%s tvm=%s ta=%s tc=%s pyannote=%s ct2=%s runpod=%s" % (
    torch.__version__, torch.version.cuda, torchvision.__version__,
    torchaudio.__version__, torchcodec.__version__, pa.__version__,
    ctranslate2.__version__, runpod.__version__))
PY

COPY vast_worker.py /vast_worker.py
COPY runpod_handler.py /runpod_handler.py

CMD ["python", "-u", "/runpod_handler.py"]