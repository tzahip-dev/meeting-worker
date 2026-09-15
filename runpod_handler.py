#!/usr/bin/env python3
"""RunPod Serverless handler — Meeting Capture Worker (production).

Reports a single boot line to the VPS receiver so the container's GPU/dependency
state is observable even though RunPod serverless does not expose worker stdout.
"""
import os
import sys
import time
import json
import traceback

sys.path.insert(0, "/")

VPS_URL = os.environ.get("VPS_URL", "").rstrip("/")
VPS_SECRET = os.environ.get("VPS_SECRET", "")

import runpod
import vast_worker as vw


def report(status: str, detail: str = "") -> None:
    """Best-effort boot report to the VPS receiver (schema_version 0)."""
    if not VPS_URL:
        return
    try:
        import requests
        import urllib3

        urllib3.disable_warnings()
        rid = f"boot_{int(time.time())}_{status}"
        requests.post(
            f"{VPS_URL}/",
            json={
                "job_id": rid,
                "run_id": rid,
                "status": status,
                "detail": detail[:1200],
                "schema_version": 0,
            },
            headers={"X-Worker-Secret": VPS_SECRET},
            verify=False,
            timeout=10,
        )
    except Exception:
        pass


def boot_check() -> None:
    try:
        import inspect

        import torch
        import torchvision  # noqa: F401  (keeps torchvision ABI loaded early)
        import torchcodec
        import ctranslate2
        import speechbrain  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401
        import pyannote.audio as pa
        from pyannote.audio.pipelines import SpeakerDiarization as SD

        has_plda = "plda" in str(inspect.signature(SD.__init__))
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "none"
        msg = (
            f"torch={torch.__version__} cuda={torch.version.cuda} avail={cuda} "
            f"gpu={gpu} cudnn={torch.backends.cudnn.version()} "
            f"ct2={ctranslate2.__version__} pyannote={pa.__version__} "
            f"plda={has_plda} torchcodec={torchcodec.__version__}"
        )
        print(f"[boot] {msg}", flush=True)
        report("boot_ok", msg)
    except Exception as e:  # noqa: BLE001
        print(f"[boot] FAILED: {e}", flush=True)
        report("boot_fail", f"{e} | {traceback.format_exc()[:600]}")

    hf_probe()


# ── HuggingFace download probe ──────────────────────────────────────────────
# A job that dies during ASR model download leaves no trace in RunPod serverless
# (worker stdout is not exposed), so probe the download path at boot and ship the
# result to the VPS receiver.
HF_RELEVANT_ENV = (
    "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_XET", "HF_XET_HIGH_PERFORMANCE",
    "HF_HUB_ENABLE_HF_TRANSFER", "HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
)


def _env_report() -> str:
    bits = []
    for k in HF_RELEVANT_ENV:
        v = os.environ.get(k)
        if v is None:
            continue
        if "TOKEN" in k:
            v = f"set(len={len(v)})"
        bits.append(f"{k}={v}")
    return " ".join(bits) or "(none set)"


def hf_probe() -> None:
    """Try a small and a large HuggingFace download; report what happens."""
    lines = [f"env: {_env_report()}"]
    try:
        from huggingface_hub import hf_hub_download

        for fname in ("config.json", "model.bin"):
            t0 = time.time()
            try:
                p = hf_hub_download(
                    "ivrit-ai/whisper-large-v3-turbo-ct2",
                    fname,
                    cache_dir=str(vw.MODELS_DIR / "asr"),
                )
                lines.append(f"{fname}: OK ({time.time() - t0:.1f}s)")
                del p
            except Exception as e:  # noqa: BLE001
                lines.append(f"{fname}: FAIL {type(e).__name__}: {repr(e)[:400]}")
                lines.append(f"  trace: {traceback.format_exc()[-500:]}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"probe setup FAIL: {e}")

    detail = "\n".join(lines)
    print(f"[hfprobe]\n{detail}", flush=True)
    report("hf_probe", detail)


def handler(event):
    t_start = time.time()
    inp = event.get("input", {}) or {}
    sha256_hex = inp.get("sha256", "")
    if not sha256_hex:
        return {"status": "failed", "error": "missing sha256"}

    wav_path = vw.WORK_DIR / f"{sha256_hex}.wav"

    try:
        if not wav_path.exists():
            vw.download_wav(sha256_hex)
        vw.log(f"   WAV: {wav_path.name} ({fmt_size(wav_path.stat().st_size)}b)")

        duration_s = vw.get_audio_duration(wav_path)
        vw.log(f"   Duration: {duration_s:.0f}s ({duration_s / 60:.1f}min)")

        source = vw.generate_source_info(wav_path, sha256_hex, inp)
        success = vw.process_one(wav_path, sha256_hex, source, duration_s)

        elapsed = time.time() - t_start
        vw.log(f" Job {'OK' if success else 'FAILED'} in {elapsed:.0f}s")
        return {"status": "ok" if success else "failed", "elapsed_s": round(elapsed, 1)}
    except Exception as e:  # noqa: BLE001
        vw.log(f"❌ Handler error: {e}\n{traceback.format_exc()}")
        return {"status": "failed", "error": str(e)[:500]}


def fmt_size(n: int) -> str:
    for unit in ["", "K", "M", "G"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}T"


if __name__ == "__main__":
    print("🚀 RunPod handler starting (production, baked image)", flush=True)
    boot_check()
    runpod.serverless.start({"handler": handler})