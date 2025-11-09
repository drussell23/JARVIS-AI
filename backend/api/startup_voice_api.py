"""
Startup Voice Announcement API
Provides voice announcement for system startup completion
Uses the same macOS voice as JARVIS for consistency
"""

import subprocess
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/startup-voice", tags=["startup-voice"])


@router.post("/announce-online")
async def announce_system_online():
    """
    Play JARVIS startup announcement using macOS Daniel voice
    Called by loading page when system reaches 100%
    """
    try:
        message = "JARVIS is online. Ready for your command."

        # Use macOS 'say' command with Daniel voice (same as JARVIS voice)
        # -v Daniel: UK male voice (deeper, professional)
        # -r 175: Speech rate (words per minute)
        subprocess.Popen(
            ["say", "-v", "Daniel", "-r", "175", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info(f"[Startup Voice] Announced: {message}")

        return JSONResponse({
            "status": "success",
            "message": "Voice announcement triggered",
            "voice": "Daniel (UK)",
            "text": message
        })

    except Exception as e:
        logger.error(f"[Startup Voice] Error: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )


@router.get("/test")
async def test_voice():
    """Test endpoint to verify voice is working"""
    try:
        subprocess.Popen(
            ["say", "-v", "Daniel", "Voice test successful"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return JSONResponse({"status": "success", "message": "Test voice played"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
