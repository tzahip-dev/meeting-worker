#!/usr/bin/env python3
"""
Meeting Capture Worker 2.0 — runs on Vast.ai (Tesla T4, 16GB VRAM).

Models:
  ASR:        ivrit-ai/whisper-large-v3-turbo-ct2         (CTranslate2)
  Diarization: pyannote/speaker-diarization-community-1    (DEFAULT, CC-BY-4.0)
  Embedding:  iic/speech_eres2netv2_sv_zh-cn_16k-common    (ModelScope, ALWAYS separate)

DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md-v2) is available as an
opt-in experiment only — set DIARIZEN_EXPERIMENT=1 in env.

Flow:
  1. Poll VPS GET /jobs → get next WAV sha256
  2. GET /wav/<sha256> → download WAV
  3. GET /config/glossary → taxonomy for Whisper initial_prompt
  4. ASR → segments
  5. Diarization (pyannote community-1) → speaker turns
  6. Embedding (ERes2NetV2, separate process) → embeddings.json
  7. Merge segments → turns.jsonl
  8. Quality analysis → quality.json
  9. Write meta.yaml + done.json
  10. POST to VPS receiver
  11. Clean up

Requires: torch, ctranslate2, faster-whisper, pyannote.audio, modelscope, requests

Environment:
  VPS_URL       — https://<VPS_IP>:8645
  VPS_SECRET    — X-Worker-Secret
  HF_TOKEN      — HuggingFace token (for gated models)
  MODELS_DIR    — where to cache models (default: ~/models)
"""

import hashlib, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

# ── Config from environment ─────────────────────────────────────────────────
VPS_URL = os.environ.get("VPS_URL", "").rstrip("/")
VPS_SECRET = os.environ.get("VPS_SECRET", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")))
WORK_DIR = Path(os.environ.get("WORK_DIR", os.path.expanduser("~/worker_work")))
# On Vast.ai Docker instances, ~ might resolve to / — use workspace if available
if str(WORK_DIR) in ("/worker_work", "//worker_work", "/root/worker_work") or not WORK_DIR.exists():
    _alt = Path("/workspace/worker_work")
    if _alt.parent.exists():
        WORK_DIR = _alt
    else:
        # Ultimate fallback: use PWD
        WORK_DIR = Path(os.environ.get("PWD", "/tmp")) / "worker_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
# Log WORK_DIR for debugging
print(f"   📂 WORK_DIR: {WORK_DIR}")
WORKER_VERSION = "0.4.0"

# ── Model IDs ───────────────────────────────────────────────────────────────
ASR_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DIARIZATION_DEFAULT = "pyannote/speaker-diarization-community-1"
DIARIZATION_EXPERIMENT = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# ── Experiment flags ────────────────────────────────────────────────────────
DIARIZEN_EXPERIMENT = os.environ.get("DIARIZEN_EXPERIMENT", "0") == "1"

# ── Fix functions ──────────────────────────────────────────────────────────

def pyify(o, round_digits=2):
    """המרה רקורסיבית של טיפוסי numpy לטיפוסי פייתון."""
    import numpy as np
    if isinstance(o, dict):
        return {k: pyify(v, round_digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [pyify(v, round_digits) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return round(float(o), round_digits)
    if isinstance(o, np.ndarray):
        return [pyify(v, round_digits) for v in o.tolist()]
    return o


def join_words(ws):
    """Fix 1: חבר רשימת מילים לטקסט עם רווחים תקינים, נקי מתווי BIDI."""
    import re
    # Clean bidirectional control characters (U+200E-U+200F, U+202A-U+202E, U+2066-U+2069)
    BIDI = dict.fromkeys(range(0x200E, 0x2010))
    BIDI.update(dict.fromkeys(range(0x202A, 0x202F)))
    BIDI.update(dict.fromkeys(range(0x2066, 0x206A)))
    txt = " ".join(w["text"].translate(BIDI).strip() for w in ws if w["text"].translate(BIDI).strip())
    txt = re.sub(r"\s+([,.!?;:״”\)\]])", r"\1", txt)
    txt = re.sub(r"([\(\[“])\s+", r"\1", txt)
    return re.sub(r"\s{2,}", " ", txt).strip()


def recording_date(filename, mtime_iso):
    """Fix 5: נסה תאריך משם קובץ, נופל ל-mtime."""
    import re
    from datetime import datetime as _dt
    m = re.search(r"(\d{6})_(\d{6})", filename)
    if m:
        try:
            dt = _dt.strptime(m.group(1) + m.group(2), "%y%m%d%H%M%S")
            return dt.date().isoformat(), "filename"
        except ValueError:
            pass
    return mtime_iso[:10], "mtime"


# ── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt_time(seconds: float) -> str:
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"


def get_audio_duration(path: Path) -> float:
    """Get duration of WAV file in seconds via ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


# ── HTTP helpers ────────────────────────────────────────────────────────────

def vps_get(path: str) -> tuple:
    """GET from VPS. Returns (status_code, data)."""
    import requests
    resp = requests.get(
        f"{VPS_URL}{path}",
        headers={"X-Worker-Secret": VPS_SECRET},
        timeout=30,
        verify=False,  # self-signed cert
    )
    if path.startswith("/wav/"):
        return resp.status_code, resp.content
    return resp.status_code, resp.json()


def vps_post(payload: dict) -> dict:
    """POST results to VPS receiver."""
    import requests
    resp = requests.post(
        f"{VPS_URL}/",
        headers={
            "X-Worker-Secret": VPS_SECRET,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        verify=False,
    )
    return resp.json()


# ── Job polling ─────────────────────────────────────────────────────────────

def fetch_job() -> dict | None:
    """Poll VPS for a pending job."""
    status, data = vps_get("/jobs")
    if status != 200:
        log(f"⚠ VPS /jobs returned {status}")
        return None
    jobs = data.get("jobs", [])
    if not jobs:
        log("📭 No pending jobs")
        return None
    job = jobs[0]
    log(f"📦 Got job: {job['sha8']} ({job['bytes']} bytes)")
    return job


# ── WAV download ────────────────────────────────────────────────────────────

def download_wav(sha256_hex: str) -> Path | None:
    """Download WAV from VPS. Returns local path."""
    wav_path = WORK_DIR / f"{sha256_hex}.wav"
    if wav_path.exists():
        log(f"♻ WAV already cached: {wav_path.name}")
        return wav_path

    status, data = vps_get(f"/wav/{sha256_hex}")
    if status != 200:
        log(f"⚠ VPS /wav/{sha256_hex[:16]} returned {status}")
        return None

    wav_path.write_bytes(data)
    log(f"⬇ Downloaded: {wav_path.name} ({len(data)} bytes)")
    return wav_path


# ── Glossary ────────────────────────────────────────────────────────────────

def fetch_glossary() -> str:
    """Fetch glossary terms from VPS for Whisper initial_prompt."""
    try:
        status, data = vps_get("/config/glossary")
        if status == 200:
            text = data.get("glossary", "")
            if text:
                log(f"📖 Glossary loaded ({len(text)} chars)")
                return text
    except Exception as e:
        log(f"⚠ Glossary fetch: {e}")
    return ""


# ── ASR (CTranslate2 faster-whisper) ────────────────────────────────────────

def run_asr(audio_path: Path, glossary: str = "") -> tuple:
    """
    Transcribe with faster-whisper via CTranslate2.
    Returns (segments, words) where:
      - segments: list of dicts {start, end, text} (for debug files)
      - words: list of dicts {start, end, text, p} (word-level for speaker assignment)
    """
    log("🎤 ASR: loading model...")
    t0 = time.time()

    from faster_whisper import WhisperModel
    
    # Try GPU first, fall back to CPU on CUDA issues
    devices_to_try = ["cuda", "cpu"]
    model = None
    last_error = None
    for dev in devices_to_try:
        try:
            model = WhisperModel(
                ASR_MODEL,
                device=dev,
                compute_type="float16" if dev == "cuda" else "int8",
                download_root=str(MODELS_DIR / "asr"),
            )
            log(f"   ASR loaded on {dev}")
            break
        except Exception as e:
            last_error = e
            log(f"   ⚠ ASR on {dev} failed: {e}")
            continue
    
    if model is None:
        raise RuntimeError(f"ASR failed on all devices: {last_error}")

    segments_info, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,          # 🔴 critical for word-level assignment
        initial_prompt=glossary if glossary else None,
        language="he",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    words = []
    for seg in segments_info:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        })
        # Extract word-level timestamps
        if seg.words:
            for w in seg.words:
                wt = w.word.strip()
                if not wt:
                    continue
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "text": wt,
                    "p": w.probability,   # ASR confidence per word
                })

    words.sort(key=lambda w: w["start"])
    dur_audio = info.duration if info else 0
    elapsed = time.time() - t0
    log(f"   ASR done: {len(segments)} segments, {len(words)} words in {elapsed:.0f}s")
    return segments, words


# ── Diarization (DEFAULT: pyannote community-1) ─────────────────────────────

def run_diarization(audio_path: Path, num_speakers: int = 0) -> tuple[list, str]:
    """
    Run speaker diarization.

    DEFAULT: pyannote/speaker-diarization-community-1 (CC-BY-4.0, pip install).
    Used for ALL production runs.

    EXPERIMENT (DIARIZEN_EXPERIMENT=1): BUT-FIT/diarizen-wavlm-large-s80-md-v2.
    Runs in parallel on the same audio — results are saved but NOT used for
    the main output path.

    Returns (turns_list, model_id_used).
    """
    log("🗣 Diarization: starting...")

    # Always run pyannote community-1 as default
    turns, model_used = _diarize_pyannote(audio_path, num_speakers)

    # If experiment flag is set, ALSO run DiariZen and compare
    if DIARIZEN_EXPERIMENT:
        try:
            log("   🔬 EXPERIMENT: running DiariZen v2 in parallel...")
            exp_turns, _ = _diarize_diarizen(audio_path, num_speakers)
            # Save experiment results alongside main output
            exp_path = WORK_DIR / "_diarizen_experiment.json"
            exp_path.write_text(json.dumps(exp_turns, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"   ✅ DiariZen experiment saved ({len(exp_turns)} turns vs {len(turns)} pyannote)")
        except Exception as e:
            log(f"   ⚠ DiariZen experiment failed: {e}")

    return turns, model_used


def _diarize_pyannote(audio_path: Path, num_speakers: int) -> tuple[list, str]:
    """pyannote community-1 diarization. Returns (turns, model_id)."""
    t0 = time.time()
    log("   Loading pyannote community-1...")
    import torch
    import torchvision  # 👈 MUST import before pyannote to avoid circular import
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_DEFAULT,
        cache_dir=str(MODELS_DIR / "diarization"),
    )
    try:
        pipeline.to(torch.device("cuda"))
        log("   Diarization on CUDA")
    except Exception:
        pipeline.to(torch.device("cpu"))
        log("   ⚠ Diarization on CPU (CUDA failed)")

    params = {}
    if num_speakers > 0:
        params["num_speakers"] = num_speakers

    try:
        result = pipeline(str(audio_path), **params)
    except RuntimeError as e:
        err = str(e)[:60]
        log(f"   Diarization fell back to CPU ({err})")
        pipeline.to(torch.device("cpu"))
        result = pipeline(str(audio_path), **params)
    turns = []

    # Handle both standard pyannote Annotation and community-1 DiarizeOutput
    if hasattr(result, "itertracks"):
        # Standard pyannote Annotation
        for seg, _, spk in result.itertracks(yield_label=True):
            turns.append({
                "speaker": spk,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
            })
    elif hasattr(result, "_fields") and hasattr(result, "segments"):
        # DiarizeOutput (NamedTuple) — iterate over segments attribute
        for seg in result.segments:
            if isinstance(seg, dict):
                turns.append({
                    "speaker": seg.get("speaker", seg.get("label", "UNKNOWN")),
                    "start": round(seg.get("start", 0), 2),
                    "end": round(seg.get("end", 0), 2),
                })
            elif hasattr(seg, "_fields"):  # nested namedtuple
                turns.append({
                    "speaker": seg.speaker if hasattr(seg, "speaker") else "UNKNOWN",
                    "start": round(getattr(seg, "start", 0), 2),
                    "end": round(getattr(seg, "end", 0), 2),
                })
    elif hasattr(result, "segments") or (isinstance(result, dict) and "segments" in result):
        # DiarizeOutput or dict-style
        segments = result.segments if hasattr(result, "segments") else result["segments"]
        for seg in segments:
            turns.append({
                "speaker": seg.get("speaker", seg.get("label", "UNKNOWN")),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
            })
    elif isinstance(result, list):
        # Already a list of turns
        for item in result:
            if isinstance(item, dict):
                turns.append({
                    "speaker": item.get("speaker", item.get("label", "UNKNOWN")),
                    "start": round(item.get("start", 0), 2),
                    "end": round(item.get("end", 0), 2),
                })
    else:
        # Debug: print all attributes and try common patterns
        log(f"   ⚠ Unknown diarization output type: {type(result).__name__}")
        attrs = [x for x in dir(result) if not x.startswith('_')]
        log(f"   🐛 dir: {attrs}")
        # DiarizeOutput has: speaker_diarization (Annotation), speaker_embeddings
        for key in ['speaker_diarization', 'exclusive_speaker_diarization']:
            obj = getattr(result, key, None)
            if obj is not None and hasattr(obj, 'itertracks'):
                for seg, _, spk in obj.itertracks(yield_label=True):
                    turns.append({
                        "speaker": spk,
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                    })
                log(f"   ✅ Extracted {len(turns)} turns from result.{key}")
                break
        if not turns:
            log(f"   → Falling back to single speaker")
            return [], DIARIZATION_DEFAULT

    elapsed = time.time() - t0
    log(f"   pyannote done: {len(turns)} turns in {elapsed:.0f}s")
    return turns, DIARIZATION_DEFAULT


def _diarize_diarizen(audio_path: Path, num_speakers: int) -> tuple[list, str]:
    """DiariZen v2 diarization. Returns (turns, model_id)."""
    t0 = time.time()
    log("   Loading DiariZen v2...")

    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained(
        DIARIZATION_EXPERIMENT,
        cache_dir=str(MODELS_DIR / "diarization"),
    )

    diar_kwargs = {}
    if num_speakers > 0:
        diar_kwargs["num_speakers"] = num_speakers
    else:
        diar_kwargs["min_speakers"] = 1
        diar_kwargs["max_speakers"] = 8

    annotation = pipeline(str(audio_path), **diar_kwargs)
    turns = []
    for seg, _, spk in annotation.itertracks(yield_label=True):
        turns.append({
            "speaker": spk,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
        })

    elapsed = time.time() - t0
    log(f"   DiariZen v2 done: {len(turns)} turns in {elapsed:.0f}s")
    return turns, DIARIZATION_EXPERIMENT


# ── Embedding (ALWAYS ERes2NetV2, separate from diarizer) ──────────────────

def run_embedding(audio_path: Path, diarization: list) -> dict:
    """
    Compute speaker embeddings.
    Uses speechbrain ECAPA (stable). ERes2NetV2 attempts are skipped due to
    modelscope compatibility issues on Vast.ai Docker.
    """
    log("🔬 Embedding: starting (separate from diarizer)...")
    t0 = time.time()

    try:
        return _embed_speechbrain(audio_path, diarization)
    except Exception as e:
        log(f"   ❌ Speechbrain failed: {e}")
        return {"model": "speechbrain/spkrec-ecapa-voxceleb", "dim": 0, "normalized": True, "speakers": []}


def _compute_speaker_stats(diarization: list) -> dict:
    """Compute per-speaker stats + clean segments for embedding sampling."""
    speaker_turns = {}
    for turn in diarization:
        spk = turn["speaker"]
        if spk not in speaker_turns:
            speaker_turns[spk] = []
        speaker_turns[spk].append(turn)

    speaker_stats = {}
    for spk, turns in speaker_turns.items():
        total_sec = sum(t["end"] - t["start"] for t in turns)
        clean_segments = []
        for t in turns:
            dur = t["end"] - t["start"]
            if dur < 2.0:
                continue
            # Check overlap with other speakers
            overlap = False
            for other_spk, other_turns in speaker_turns.items():
                if other_spk == spk:
                    continue
                for ot in other_turns:
                    if max(t["start"], ot["start"]) < min(t["end"], ot["end"]):
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                clean_segments.append({"start": t["start"], "end": t["end"], "dur": dur})

        # Sort by duration descending, take up to 90s / 15 segments
        clean_segments.sort(key=lambda x: x["dur"], reverse=True)
        sampled = []
        total = 0.0
        for cs in clean_segments:
            if total >= 90.0 or len(sampled) >= 15:
                break
            sampled.append(cs)
            total += cs["dur"]

        speaker_stats[spk] = {
            "speech_sec": round(total_sec, 2),
            "clean_sec": round(total, 2),
            "clean_segments": sampled,
        }

    return speaker_stats


def _embed_speechbrain(audio_path: Path, diarization: list) -> dict:
    """Compute embeddings using speechbrain ECAPA."""
    t0 = time.time()
    log("   Loading speechbrain ECAPA...")

    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS_DIR / "embedding"),
        run_opts={"device": "cuda"},
    )

    audio_data, sr = sf.read(str(audio_path))
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    speaker_stats = _compute_speaker_stats(diarization)

    speakers_out = []
    for spk, stats in speaker_stats.items():
        clean_segs = stats["clean_segments"]
        if not clean_segs or stats["clean_sec"] < 20.0:
            log(f"   ⏭ {spk}: only {stats['clean_sec']}s clean speech, skipping embedding")
            continue

        vectors = []
        for cs in clean_segs:
            start_s = int(cs["start"] * sr)
            end_s = int(cs["end"] * sr)
            chunk = audio_data[start_s:end_s]
            if len(chunk) < sr * 0.5:
                continue

            try:
                signal = torch.from_numpy(chunk).float().unsqueeze(0).to("cuda")
                emb = classifier.encode_batch(signal)
                emb_np = emb.squeeze().cpu().numpy()
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np = emb_np / norm
                vectors.append(emb_np)
            except Exception as e:
                log(f"   ⚠ Embedding chunk failed: {e}")

        if vectors:
            mean_vec = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            mean_vec = np.round(mean_vec, 4)
            speakers_out.append({
                "label": spk,
                "speech_sec": stats["speech_sec"],
                "sampled_sec": round(stats["clean_sec"], 2),
                "segments_used": len(vectors),
                "vector": mean_vec.tolist(),
            })

    dim = len(speakers_out[0]["vector"]) if speakers_out else 0
    elapsed = time.time() - t0
    log(f"   Embedding done: {len(speakers_out)} speakers, dim={dim} ({elapsed:.0f}s)")
    return {"model": "speechbrain/spkrec-ecapa-voxceleb", "dim": dim, "normalized": True, "speakers": speakers_out}


# ── Word→Speaker assignment (fix 1: word-level) ──────────────────────────

def assign_speakers_to_words(words: list, dia: list) -> list:
    """
    Assign each word to the most-overlapping speaker using two-pointer scan O(n+m).
    Adds 's' (speaker) and 'c' (confidence) to each word.
    Fix 6: words in gaps fall back to nearest speaker within 2s.
    """
    j = 0
    for w in words:
        while j < len(dia) and dia[j]["end"] < w["start"]:
            j += 1
        best, best_ov, k = "UNKNOWN", 0.0, j
        while k < len(dia) and dia[k]["start"] < w["end"]:
            ov = min(w["end"], dia[k]["end"]) - max(w["start"], dia[k]["start"])
            if ov > best_ov:
                best_ov, best = ov, dia[k]["speaker"]
            k += 1
        # Fix 6: fallback to nearest speaker within 2s
        if best == "UNKNOWN":
            near = []
            for d in dia:
                dist = min(abs(w["start"] - d["end"]), abs(d["start"] - w["end"]))
                if dist < 2.0:
                    near.append((dist, d["speaker"]))
            if near:
                best, best_ov = min(near, key=lambda x: x[0])[1], 0.0
        dur = max(w["end"] - w["start"], 1e-6)
        w["s"] = best
        w["c"] = min(1.0, best_ov / dur)
    return words


def build_turns_from_words(words: list) -> list:
    """
    Merge consecutive words with same speaker into turns, per spec section 4.1.
    Gap < 1.5s to merge, split at 90s.
    Fix 3: also split when speaker confidence drops below AMBIG.
    """
    MERGE_GAP, MAX_TURN, AMBIG = 1.5, 90.0, 0.65
    turns, cur, prev_w = [], None, None
    for w in words:
        start_new = (
            cur is None
            or w["s"] != cur["s"]
            or (w["c"] < AMBIG and prev_w is not None and prev_w["c"] >= AMBIG)
            or w["start"] - cur["_end"] > MERGE_GAP
            or cur["_end"] - cur["t"] > MAX_TURN
        )
        if start_new:
            if cur:
                turns.append(cur)
            cur = {"t": w["start"], "s": w["s"], "_end": w["end"], "_w": []}
        cur["_w"].append(w)
        cur["_end"] = w["end"]
        prev_w = w
    if cur:
        turns.append(cur)

    result = []
    for i, c in enumerate(turns, 1):
        ws = c["_w"]
        tot = sum(w["end"] - w["start"] for w in ws) or 1e-6
        weighted_avg = lambda key: sum(w[key] * (w["end"] - w["start"]) for w in ws) / tot
        result.append({
            "i": i,
            "t": round(c["t"], 2),
            "d": round(c["_end"] - c["t"], 2),
            "s": c["s"],
            "c": round(weighted_avg("c"), 2),
            "a": round(weighted_avg("p"), 2),
            "o": False,
            "x": join_words(ws),  # Fix 1: proper spacing
        })

    return result


def mark_overlap_flags(turns: list, dia: list, words: list = None):
    """Fix 4: תור מסומן overlap רק אם ≥15% מהמילים בו חופפות."""
    for turn in turns:
        ts, te = turn["t"], turn["t"] + turn["d"]
        ov_words = 0.0
        total = 0.0
        for w in (words or []):
            if w["start"] < ts or w["end"] > te:
                continue
            total += w["end"] - w["start"]
            for d in dia:
                if d["speaker"] == turn["s"]:
                    continue
                if min(w["end"], d["end"]) - max(w["start"], d["start"]) > 0.05:
                    ov_words += w["end"] - w["start"]
                    break
        if total > 0 and ov_words / total >= 0.15:
            turn["o"] = True


def compute_overlap_seconds(dia: list) -> float:
    """Sweep-line overlap: time where ≥2 speakers active (fix 3)."""
    events = []
    for d in dia:
        events.append((d["start"], 1))
        events.append((d["end"], -1))
    events.sort()
    active, prev, total = 0, None, 0.0
    for t, delta in events:
        if active >= 2 and prev is not None:
            total += t - prev
        active += delta
        prev = t
    return total


# ── Quality analysis ────────────────────────────────────────────────────────

def analyze_quality(turns: list) -> dict:
    """
    Generate quality.json per spec section 3.3.
    """
    log("📊 Quality analysis...")

    INVALID_HEB_RANGE = set(range(0x05EB, 0x05F0))  # U+05EB–U+05EF

    flagged = []
    for turn in turns:
        reasons = []
        if turn["c"] < 0.7:
            reasons.append("low_speaker_confidence")
        if turn["a"] < 0.5:
            reasons.append("low_asr")
        if turn["o"]:
            reasons.append("overlap")
        if turn["s"] == "UNKNOWN":
            reasons.append("unknown_speaker")
        for ch in turn["x"]:
            if ord(ch) in INVALID_HEB_RANGE:
                reasons.append("invalid_unicode")
                break

        if reasons:
            flagged.append({
                "i": turn["i"],
                "t": turn["t"],
                "s": turn["s"],
                "reasons": reasons,
                "x": turn["x"][:200],
            })

    total = len(turns)
    flagged_count = len(flagged)

    # Limit to 40 longest flagged turns
    flagged_with_dur = [(f, next((t["d"] for t in turns if t["i"] == f["i"]), 0)) for f in flagged]
    flagged_with_dur.sort(key=lambda x: x[1], reverse=True)
    flagged_limited = [f for f, _ in flagged_with_dur[:40]]

    flagged_pct = round(100 * flagged_count / total, 1) if total > 0 else 0
    mean_c = round(sum(t["c"] for t in turns) / total, 2) if total > 0 else 0
    mean_a = round(sum(t["a"] for t in turns) / total, 2) if total > 0 else 0

    if flagged_pct < 15:
        verdict = "ok"
    elif flagged_pct <= 30:
        verdict = "review"
    else:
        verdict = "poor"
    if mean_c < 0.6:
        verdict = "poor"

    return {
        "run_id": "",
        "verdict": verdict,
        "flagged_turns": flagged_count,
        "total_turns": total,
        "flagged_pct": flagged_pct,
        "mean_speaker_confidence": mean_c,
        "mean_asr_confidence": mean_a,
        "flagged": flagged_limited,
    }


# ── Write output files ─────────────────────────────────────────────────────

def write_outputs(run_dir: Path, run_id: str, source: dict,
                  segments: list, diarization: list, diar_model: str,
                  turns: list, embeddings: dict, quality: dict,
                  duration_s: float, overlap_s: float,
                  peak_vram_mb: int, total_sec: float,
                  diarizen_experiment: bool = False):
    """Write 5 output files + debug files (fixes 2, 4, 5, 6)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"📝 Writing output files to {run_dir.name}/")

    import yaml

    # Fix 5: single source of truth for speech_sec — computed from turns
    speech_sec = {}
    for t in turns:
        spk = t["s"]
        speech_sec[spk] = round(speech_sec.get(spk, 0.0) + t["d"], 2)
    if "UNKNOWN" in speech_sec:
        del speech_sec["UNKNOWN"]

    # Fix 4: fix floating point serialization in embeddings
    _emb = embeddings
    if _emb.get("speakers"):
        for s in _emb["speakers"]:
            s["vector"] = [round(float(v), 4) for v in s["vector"]]

    # Speaker data (from turns, single source)
    speakers_in_turns = sorted(set(t["s"] for t in turns if t["s"] != "UNKNOWN"))
    speaker_data = {}
    for spk in speakers_in_turns:
        spk_turns = [t for t in turns if t["s"] == spk]
        first_seen = min(t["t"] for t in spk_turns) if spk_turns else 0
        has_emb = any(s["label"] == spk for s in _emb.get("speakers", []))
        speaker_data[spk] = {
            "speech_sec": speech_sec.get(spk, 0),
            "turn_count": len(spk_turns),
            "share_pct": round(100 * speech_sec.get(spk, 0) / duration_s, 1) if duration_s > 0 else 0,
            "first_seen": round(first_seen, 2),
            "has_embedding": has_emb,
        }

    # ── 1. meta.yaml ──
    date_val, date_source = recording_date(
        source.get("original_filename", ""),
        source.get("original_mtime", "")
    )
    meta = {
        "run_id": run_id,
        "schema_version": 2,
        "date": date_val,
        "date_source": date_source,
        "duration_min": round(duration_s / 60, 1),
        "language": "he",
        "models": {
            "asr": ASR_MODEL,
            "asr_revision": None,
            "diarization": diar_model,
            "diarization_revision": None,
            "embedding": _emb.get("model", EMBEDDING_MODEL),
            "embedding_revision": None,
            "embedding_dim": _emb.get("dim", 0),
        },
        "speakers": [
            {"label": spk, **data}
            for spk, data in speaker_data.items()
        ],
        "stats": {
            "turn_count": len(turns),
            "segment_count": len(segments),
            "mean_asr_confidence": round(sum(t["a"] for t in turns) / len(turns), 2) if turns else 0,
            "mean_speaker_confidence": round(sum(t["c"] for t in turns) / len(turns), 2) if turns else 0,
            "overlap_sec": round(overlap_s, 2),
            "overlap_pct": round(100 * overlap_s / duration_s, 1) if duration_s > 0 else 0,
            "unknown_speaker_turns": sum(1 for t in turns if t["s"] == "UNKNOWN"),
        },
        "processing": {
            "total_sec": round(total_sec, 1),
            "device": "cuda:0",
            "peak_vram_mb": peak_vram_mb,
        },
    }

    if diarizen_experiment:
        meta["models"]["diarization_experiment"] = DIARIZATION_EXPERIMENT

    # ── Write files in order: meta → turns → quality → embeddings → [debug] → done (LAST) ──
    meta_path = run_dir / "meta.yaml"
    meta_path_tmp = run_dir / "meta.yaml.tmp"
    with open(meta_path_tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(pyify(meta), f, allow_unicode=True, sort_keys=False)
    meta_path_tmp.rename(meta_path)

    turns_path = run_dir / "turns.jsonl"
    turns_path_tmp = run_dir / "turns.jsonl.tmp"
    with open(turns_path_tmp, "w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(pyify(turn), ensure_ascii=False) + "\n")
    turns_path_tmp.rename(turns_path)

    quality["run_id"] = run_id
    qual_path = run_dir / "quality.json"
    qual_path_tmp = run_dir / "quality.json.tmp"
    with open(qual_path_tmp, "w", encoding="utf-8") as f:
        json.dump(pyify(quality), f, ensure_ascii=False, separators=(",", ":"))
    qual_path_tmp.rename(qual_path)

    emb_path = run_dir / "embeddings.json"
    emb_path_tmp = run_dir / "embeddings.json.tmp"
    # Fix 4: vectors rounded to 4 digits, rest normal
    _emb_fixed = pyify(_emb, round_digits=4)
    with open(emb_path_tmp, "w", encoding="utf-8") as f:
        json.dump(_emb_fixed, f, ensure_ascii=False, separators=(",", ":"))
    emb_path_tmp.rename(emb_path)

    # ── Debug files (before done.json — agent ignores them) ──
    (run_dir / "transcript.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [f"# Transcript — {len(segments)} segments\n"]
    for s in segments:
        md_lines.append(s["text"])
    (run_dir / "transcript.md").write_text("\n".join(md_lines), encoding="utf-8")
    (run_dir / "diarization.json").write_text(json.dumps(diarization, ensure_ascii=False, indent=2), encoding="utf-8")
    spk_lines = [f"# Meeting — {len(turns)} turns\n"]
    for turn in turns:
        spk_lines.append(f"[{fmt_time(turn['t'])}] {turn['s']}: {turn['x']}")
    (run_dir / "speakers.md").write_text("\n".join(spk_lines), encoding="utf-8")

    if diarizen_experiment:
        exp_src = WORK_DIR / "_diarizen_experiment.json"
        if exp_src.exists():
            shutil.copy(exp_src, run_dir / "diarizen_experiment.json")
            exp_src.unlink()
            log(f"   🔬 Copied DiariZen experiment results")

    # ── 5. done.json (LAST — signal to cowork agent) ──
    done = {
        "run_id": run_id,
        "schema_version": 2,
        "worker_version": WORKER_VERSION,
        "status": "ok",
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "source": source,
        "files": [
            {"name": "meta.yaml", "bytes": meta_path.stat().st_size, "sha256": sha256_file(meta_path)},
            {"name": "turns.jsonl", "bytes": turns_path.stat().st_size, "sha256": sha256_file(turns_path)},
            {"name": "quality.json", "bytes": qual_path.stat().st_size, "sha256": sha256_file(qual_path)},
            {"name": "embeddings.json", "bytes": emb_path.stat().st_size, "sha256": sha256_file(emb_path)},
        ],
    }
    done_path = run_dir / "done.json"
    done_path_tmp = run_dir / "done.json.tmp"
    with open(done_path_tmp, "w", encoding="utf-8") as f:
        json.dump(pyify(done), f, ensure_ascii=False, separators=(",", ":"))
    done_path_tmp.rename(done_path)

    log(f"   ✅ {len(turns)} turns, {len(segments)} segments, {len(speaker_data)} speakers")
    return done


def write_failure_output(run_dir: Path, run_id: str, source: dict,
                         stage: str, message: str, traceback_tail: str = ""):
    """Write a failure done.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    done = {
        "run_id": run_id,
        "schema_version": 2,
        "worker_version": WORKER_VERSION,
        "status": "failed",
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "source": source,
        "error": {
            "stage": stage,
            "message": str(message)[:500],
            "traceback_tail": traceback_tail[-1000:],
        },
    }
    done_path = run_dir / "done.json"
    done_path_tmp = run_dir / "done.json.tmp"
    with open(done_path_tmp, "w", encoding="utf-8") as f:
        json.dump(pyify(done), f, ensure_ascii=False, separators=(",", ":"))
    done_path_tmp.rename(done_path)
    log(f"   ❌ Failure written: {stage} — {message}")


# ── Push results to VPS ────────────────────────────────────────────────────

def push_results(run_dir: Path, run_id: str, source: dict, status: str = "ok"):
    """POST results to VPS receiver via _files mechanism."""
    log(f"☁️ Pushing results to VPS...")

    payload = {
        "run_id": run_id,
        "schema_version": 2,
        "worker_version": WORKER_VERSION,
        "status": status,
        "source": source,
        "_files": {},
    }

    error_path = run_dir / "done.json"
    if error_path.exists():
        done = json.loads(error_path.read_text(encoding="utf-8"))
        if done.get("status") == "failed":
            payload["error"] = done["error"]

    for fname in ["meta.yaml", "turns.jsonl", "quality.json", "embeddings.json", "done.json",
                   "transcript.json", "transcript.md", "diarization.json", "speakers.md",
                   "diarizen_experiment.json"]:
        fp = run_dir / fname
        if fp.exists():
            payload["_files"][fname] = fp.read_text(encoding="utf-8")

    if not payload["_files"]:
        log("   ⚠ No files to push!")
        return False

    try:
        result = vps_post(payload)
        if result.get("status") == "ok" or result.get("error") is True:
            log(f"   ✅ Push accepted: {run_id}")
            return True
        else:
            log(f"   ⚠ Push returned: {result}")
            return False
    except Exception as e:
        log(f"   ❌ Push failed: {e}")
        return False


# ── Main processing loop ───────────────────────────────────────────────────

def process_one(audio_path: Path, sha256_hex: str, source: dict, duration_s: float) -> bool:
    """Process a single audio file end-to-end."""
    run_id = f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{sha256_hex[:8]}"
    run_dir = WORK_DIR / run_id
    t_start = time.time()
    peak_vram_mb = 0
    diar_model_used = DIARIZATION_DEFAULT

    log(f"\n{'='*60}")
    log(f"🎬 Processing: {run_id}")
    log(f"   File: {source.get('original_filename', '?')} ({duration_s:.0f}s)")

    try:
        def check_vram():
            nonlocal peak_vram_mb
            try:
                import subprocess
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                used = int(out.stdout.strip().split("\n")[0])
                peak_vram_mb = max(peak_vram_mb, used)
            except Exception:
                pass

        check_vram()

        # Glossary
        glossary = fetch_glossary()

        # ASR (now returns segments + words with word_timestamps)
        check_vram()
        segments, words = run_asr(audio_path, glossary)
        check_vram()
        if not segments:
            raise RuntimeError("ASR produced no segments")

        # Diarization (pyannote community-1)
        check_vram()
        diarization, diar_model_used = run_diarization(audio_path)
        check_vram()
        if not diarization:
            log("   ⚠ Diarization produced no turns, using single speaker")
            diarization = [{"speaker": "SPEAKER_00", "start": 0, "end": duration_s}]

        # Embedding (always ECAPA, separate from diarizer)
        check_vram()
        embeddings = run_embedding(audio_path, diarization)
        check_vram()

        # Fix 1: word-level speaker assignment
        if words:
            assign_speakers_to_words(words, diarization)
            turns = build_turns_from_words(words)
            mark_overlap_flags(turns, diarization, words)
        else:
            # Fallback if no word timestamps
            log("   ⚠ No word timestamps — using single speaker fallback")
            turns = [{"i": 1, "t": 0, "d": duration_s, "s": "SPEAKER_00",
                       "c": 1.0, "a": 0.5, "o": False, "x": segments[0]["text"]}]

        # Fix 3: proper overlap
        overlap_s = compute_overlap_seconds(diarization)

        # Quality
        quality = analyze_quality(turns)

        # Write files (with all fixes)
        total_sec = time.time() - t_start
        write_outputs(
            run_dir=run_dir,
            run_id=run_id,
            source=source,
            segments=segments,
            diarization=diarization,
            diar_model=diar_model_used,
            turns=turns,
            embeddings=embeddings,
            quality=quality,
            duration_s=duration_s,
            overlap_s=overlap_s,
            peak_vram_mb=peak_vram_mb,
            total_sec=total_sec,
            diarizen_experiment=DIARIZEN_EXPERIMENT,
        )

        # Push
        pushed = push_results(run_dir, run_id, source, status="ok")

        elapsed = time.time() - t_start
        log(f"✅ Done: {run_id} ({elapsed:.0f}s total)")
        return True

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"❌ Processing failed: {e}")

        stage = "unknown"
        msg = str(e)
        if "ASR" in msg or "whisper" in msg.lower():
            stage = "asr"
        elif "diar" in msg.lower():
            stage = "diarization"
        elif "embed" in msg.lower():
            stage = "embedding"

        write_failure_output(run_dir, run_id, source, stage, msg, tb)
        push_results(run_dir, run_id, source, status="failed")
        return False

    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def generate_source_info(audio_path: Path, sha256_hex: str, job: dict = None) -> dict:
    mtime = datetime.fromtimestamp(audio_path.stat().st_mtime, tz=timezone.utc)
    # Use original metadata from job if available (fix 2)
    orig_name = audio_path.name
    orig_mtime = mtime.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    if job:
        orig_name = job.get("original_filename", audio_path.name)
        job_mtime = job.get("original_mtime")
        if job_mtime:
            orig_mtime = job_mtime[:19] + "+00:00"
    return {
        "original_filename": orig_name,
        "original_mtime": orig_mtime,
        "sha256": sha256_hex,
        "bytes": audio_path.stat().st_size,
    }


def main():
    log(f"🚀 Meeting Capture Worker v{WORKER_VERSION}")
    log(f"   VPS: {VPS_URL}")
    log(f"   Diarization: pyannote community-1 (default)")
    log(f"   Embedding:   speechbrain ECAPA")

    if DIARIZEN_EXPERIMENT:
        log(f"   🔬 DiariZen v2 experiment: ENABLED (parallel run)")

    if not VPS_URL or not VPS_SECRET:
        log("❌ VPS_URL and VPS_SECRET must be set")
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            log(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB VRAM)")
        else:
            log("⚠ No GPU detected!")
    except ImportError:
        log("⚠ torch not found")

    # Poll for job
    job = fetch_job()
    if not job:
        log("📭 No work — exiting (instance will auto-stop)")
        _auto_stop()
        return

    sha256_hex = job["sha256"]
    log(f"📦 Processing: {sha256_hex[:16]}... ({job['bytes']} bytes)")

    wav_path = download_wav(sha256_hex)
    if not wav_path:
        log("❌ Failed to download WAV")
        sys.exit(1)

    duration_s = get_audio_duration(wav_path)
    source = generate_source_info(wav_path, sha256_hex, job)
    log(f"   Duration: {duration_s:.0f}s ({duration_s/60:.1f}min)")

    success = process_one(wav_path, sha256_hex, source, duration_s)

    if wav_path.exists():
        wav_path.unlink()
        log(f"   🗑️ Deleted WAV")

    log(f"\n{'✅ Success' if success else '❌ Failed'} — worker finished")

    # Auto-stop the instance
    _auto_stop()


def _auto_stop():
    """Stop the Vast.ai instance to avoid paying for idle time."""
    import requests, json
    try:
        instance_id = os.environ.get("VAST_INSTANCE_ID", "50858558")
        api_key = os.environ.get("VAST_API_KEY", "")
        if not api_key:
            log("   ⚠ No VAST_API_KEY — instance will keep running")
            return
        log("   🛑 Stopping Vast.ai instance...")
        resp = requests.put(
            f"https://console.vast.ai/api/v0/instances/{instance_id}/",
            json={"state": "stopped"},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            log("   ✅ Instance stopping")
        else:
            log(f"   ⚠ Stop returned {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        log(f"   ⚠ Auto-stop failed: {e}")


if __name__ == "__main__":
    main()