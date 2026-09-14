#!/usr/bin/env python3
"""RunPod Serverless handler — Meeting Capture Worker (production)."""
import os, sys, time, json, traceback, hashlib
from pathlib import Path

sys.path.insert(0, "/")

import runpod
import vast_worker as vw
from datetime import datetime, timezone


def handler(event):
    t_start = time.time()
    inp = event.get("input", {})
    sha256_hex = inp.get("sha256", "")
    if not sha256_hex:
        return {"status": "failed", "error": "missing sha256"}

    wav_path = vw.WORK_DIR / f"{sha256_hex}.wav"

    try:
        # Download WAV from VPS
        if not wav_path.exists():
            vw.download_wav(sha256_hex)
        vw.log(f"   WAV: {wav_path.name} ({vwfmt(wav_path.stat().st_size)}b)")

        # Get audio duration
        duration_s = vw.get_audio_duration(wav_path)
        vw.log(f"   Duration: {duration_s:.0f}s ({duration_s/60:.1f}min)")

        # Build source info
        source = vw.generate_source_info(wav_path, sha256_hex)

        # Process
        success = vw.process_one(wav_path, sha256_hex, source, duration_s)

        elapsed = time.time() - t_start
        vw.log(f"🏁 Job {'OK' if success else 'FAILED'} in {elapsed:.0f}s")
        return {"status": "ok" if success else "failed", "elapsed_s": round(elapsed, 1)}
    except Exception as e:
        tb = traceback.format_exc()
        vw.log(f"❌ Handler error: {e}")
        return {"status": "failed", "error": str(e)[:500]}


def vwfmt(n):
    for unit in ["", "K", "M", "G"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}T"


if __name__ == "__main__":
    # The worker code is on the image — no need to download
    vw.log("🚀 RunPod handler starting (production)")
    runpod.serverless.start({"handler": handler})