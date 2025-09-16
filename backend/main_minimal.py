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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import signal
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to upgrader
_upgrader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _upgrader
    
    logger.info("Starting JARVIS Minimal Backend with self-healing upgrader...")
    
    # Start the minimal to full upgrader
    try:
        from minimal_to_full_upgrader import get_upgrader
        _upgrader = get_upgrader()
        await _upgrader.start()
        logger.info("🔄 Minimal to Full Mode upgrader started - monitoring for upgrade opportunities")
    except Exception as e:
        logger.warning(f"Could not start upgrader: {e}")
        _upgrader = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down minimal backend...")
    if _upgrader:
        await _upgrader.stop()


# Create FastAPI app
app = FastAPI(
    title="JARVIS Minimal Backend", 
    version="1.0.0",
    lifespan=lifespan
)

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
    response = {
        "status": "healthy", 
        "service": "jarvis-minimal",
        "mode": "minimal",
        "components": {
            "vision": False,
            "memory": False,
            "voice": False,
            "tools": False,
            "rust": False
        }
    }
    
    # Add upgrader status if available
    if _upgrader:
        response["upgrader"] = {
            "monitoring": _upgrader._running,
            "attempts": _upgrader._upgrade_attempts,
            "max_attempts": _upgrader._max_attempts
        }
        
    return response


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


@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    """Gracefully shutdown the minimal backend."""
    
    def shutdown_server():
        """Shutdown server after response is sent."""
        time.sleep(1)  # Give time for response
        os.kill(os.getpid(), signal.SIGTERM)
    
    background_tasks.add_task(shutdown_server)
    
    return {
        "success": True,
        "message": "Minimal backend shutting down gracefully"
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8010")))
    args = parser.parse_args()

    logger.info(f"Starting JARVIS Minimal Backend on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
