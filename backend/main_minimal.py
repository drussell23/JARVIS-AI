#!/usr/bin/env python3
"""
Minimal JARVIS backend for testing - bypasses import errors
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="JARVIS Minimal Backend", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "JARVIS Minimal Backend Running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "jarvis-minimal"}


@app.get("/voice/jarvis/status")
async def voice_status():
    return {
        "status": "available",
        "mode": "minimal",
        "message": "Voice system in minimal mode",
    }


@app.get("/audio/ml/config")
async def audio_config():
    return {"sample_rate": 16000, "channels": 1, "format": "int16"}


@app.post("/voice/jarvis/activate")
async def activate_jarvis():
    return {"success": True, "message": "JARVIS activated in minimal mode"}


@app.post("/audio/ml/predict")
async def predict_audio():
    return {"prediction": "normal", "confidence": 0.9}


@app.websocket("/audio/ml/stream")
async def audio_ml_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back minimal response
            await websocket.send_json(
                {
                    "type": "response",
                    "status": "minimal_mode",
                    "message": "Audio ML in minimal mode",
                }
            )
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8010")))
    args = parser.parse_args()

    logger.info(f"Starting JARVIS Minimal Backend on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
