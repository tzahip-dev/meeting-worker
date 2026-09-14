# Meeting Capture Worker

RunPod Serverless worker for the Meeting Capture pipeline.

- **ASR:** `ivrit-ai/whisper-large-v3-turbo-ct2` (faster-whisper/CTranslate2)
- **Diarization:** `pyannote/speaker-diarization-community-1`
- **Embedding:** `speechbrain/spkrec-ecapa-voxceleb`
- **Env:** `VPS_URL`, `VPS_SECRET`, `MODELS_DIR`, `HF_TOKEN`

## Build

Pushing to `main` triggers GitHub Actions → builds → pushes `tzahip/meeting-worker:v3` to Docker Hub.

## Deploy

```bash
runpodctl template create --name meeting-worker --serverless \
  --image tzahip/meeting-worker:v3 --container-disk-in-gb 25 \
  --env '{"VPS_URL":"https://<vps>:8645","MODELS_DIR":"/models"}'

runpodctl serverless create --template-id <id> --name meeting-worker \
  --gpu-id "NVIDIA GeForce RTX 3090" --workers-min 0 --workers-max 1
```

## Trigger

VPS cron `runpod_trigger.py` POSTs `{"input":{"sha256":"<wav-hash>"}}` to `/run`.