#!/usr/bin/env python3
"""Echo handler — minimal test for RunPod serverless dispatch."""
import runpod
runpod.serverless.start({"handler": lambda job: {"echo": job.get("input"), "status": "ok"}})