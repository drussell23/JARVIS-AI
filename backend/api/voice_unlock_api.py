"""
Voice Unlock API Router
======================

FastAPI endpoints for voice unlock enrollment and authentication.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
import asyncio
import json
import numpy as np
import base64
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from voice_unlock import (
        VoiceEnrollmentManager, 
        VoiceAuthenticator, 
        MacUnlockService
    )
    from voice_unlock.config import get_config
    from voice_unlock.services.keychain_service import KeychainService
    from voice_unlock.services.screensaver_integration import ScreensaverManager
    VOICE_UNLOCK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice unlock modules not available: {e}")
    VOICE_UNLOCK_AVAILABLE = False
    # Dummy classes
    class VoiceEnrollmentManager: pass
    class VoiceAuthenticator: pass
    class MacUnlockService: pass
    class KeychainService: pass
    class ScreensaverManager: pass
    def get_config(): return {}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice-unlock", tags=["voice_unlock"])

# Global instances
enrollment_manager = None
authenticator = None
unlock_service = None
keychain_service = None
screensaver_manager = None


def initialize_voice_unlock():
    """Initialize voice unlock components"""
    global enrollment_manager, authenticator, unlock_service, keychain_service, screensaver_manager
    
    if not VOICE_UNLOCK_AVAILABLE:
        logger.warning("Voice unlock modules not available, using daemon connector instead")
        # Initialize the voice_unlock_integration for JARVIS command handling
        try:
            from . import voice_unlock_integration
            logger.info("Voice unlock integration loaded for JARVIS commands")
        except ImportError:
            logger.error("Could not import voice_unlock_integration")
        return True
    
    try:
        # Initialize services
        enrollment_manager = VoiceEnrollmentManager()
        authenticator = VoiceAuthenticator()
        keychain_service = KeychainService()
        
        # Initialize system integration
        from voice_unlock.services.mac_unlock_service import MacUnlockService
        unlock_service = MacUnlockService(authenticator)
        
        # Initialize screensaver integration
        screensaver_manager = ScreensaverManager()
        
        logger.info("✅ Voice Unlock services initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Voice Unlock: {e}")
        return False


@router.get("/status")
async def get_voice_unlock_status():
    """Get voice unlock system status"""
    config = get_config()
    
    status = {
        "initialized": enrollment_manager is not None,
        "config": {
            "integration_mode": config.system.integration_mode,
            "min_samples": config.enrollment.min_samples,
            "anti_spoofing_level": config.security.anti_spoofing_level,
            "adaptive_thresholds": config.authentication.adaptive_thresholds,
        },
        "services": {
            "enrollment": enrollment_manager is not None,
            "authentication": authenticator is not None,
            "keychain": keychain_service is not None,
            "screensaver": screensaver_manager is not None and screensaver_manager.integration.monitoring
        }
    }
    
    # Get enrolled users count
    if keychain_service:
        try:
            voiceprints = keychain_service.list_voiceprints()
            status["enrolled_users"] = len(voiceprints)
        except:
            status["enrolled_users"] = 0
    
    return status


@router.post("/enrollment/start")
async def start_enrollment(user_id: str, metadata: Optional[Dict[str, Any]] = None):
    """Start new enrollment session"""
    if not enrollment_manager:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        session_id = enrollment_manager.start_enrollment(user_id, metadata)
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id,
            "message": f"Enrollment session started for {user_id}"
        }
    except Exception as e:
        logger.error(f"Failed to start enrollment: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.websocket("/enrollment/ws/{session_id}")
async def enrollment_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time enrollment with audio streaming"""
    await websocket.accept()
    
    if not enrollment_manager:
        await websocket.send_json({
            "type": "error",
            "message": "Voice unlock not initialized"
        })
        await websocket.close()
        return
    
    # Validate session
    session_status = enrollment_manager.get_session_status(session_id)
    if not session_status:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid session ID"
        })
        await websocket.close()
        return
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": session_status
        })
        
        # Add visualization callback
        def audio_visualization(audio_chunk: np.ndarray):
            """Send audio visualization data"""
            try:
                # Downsample for visualization
                viz_data = audio_chunk[::10].tolist()
                asyncio.create_task(websocket.send_json({
                    "type": "audio_visualization",
                    "data": {
                        "waveform": viz_data,
                        "energy": float(np.sqrt(np.mean(audio_chunk ** 2)))
                    }
                }))
            except:
                pass
        
        enrollment_manager.audio_capture.add_callback(audio_visualization)
        
        while True:
            # Receive commands or audio data
            data = await websocket.receive_json()
            
            if data["type"] == "collect_sample":
                # Get current phrase
                session = enrollment_manager.sessions[session_id]
                phrase = enrollment_manager._get_enrollment_phrase(session)
                
                # Send phrase to user
                await websocket.send_json({
                    "type": "phrase",
                    "data": {
                        "phrase": phrase,
                        "sample_number": len(session.samples) + 1
                    }
                })
                
                # Collect sample
                if "audio_data" in data:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(data["audio_data"])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    success, message = await enrollment_manager.collect_sample(
                        session_id,
                        phrase=phrase,
                        audio_data=audio_array
                    )
                else:
                    # Use microphone
                    success, message = await enrollment_manager.collect_sample(
                        session_id,
                        phrase=phrase
                    )
                
                # Send result
                await websocket.send_json({
                    "type": "sample_result",
                    "data": {
                        "success": success,
                        "message": message,
                        "status": enrollment_manager.get_session_status(session_id)
                    }
                })
                
                # Check if enrollment complete
                session = enrollment_manager.sessions.get(session_id)
                if session and session.status.value == "completed":
                    await websocket.send_json({
                        "type": "enrollment_complete",
                        "data": {
                            "user_id": session.user_id,
                            "quality_score": np.mean([s.quality_score for s in session.samples])
                        }
                    })
                    break
                    
            elif data["type"] == "cancel":
                enrollment_manager.cancel_enrollment(session_id)
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Enrollment cancelled"
                })
                break
                
    except WebSocketDisconnect:
        logger.info(f"Enrollment WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Enrollment WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Clean up
        if enrollment_manager and hasattr(enrollment_manager.audio_capture, 'callbacks'):
            enrollment_manager.audio_capture.callbacks.clear()


@router.post("/authenticate")
async def authenticate_voice(
    user_id: Optional[str] = None,
    audio_file: Optional[UploadFile] = File(None)
):
    """Authenticate user with voice"""
    if not authenticator:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        # Process audio file if provided
        audio_data = None
        if audio_file:
            audio_bytes = await audio_file.read()
            # Convert to numpy array (assuming WAV format)
            import wave
            import io
            
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Perform authentication
        result, details = await authenticator.authenticate(
            user_id=user_id,
            audio_data=audio_data
        )
        
        return {
            "success": result.value == "success",
            "result": result.value,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/unlock/toggle")
async def toggle_voice_unlock(enable: bool):
    """Enable or disable voice unlock monitoring"""
    if not screensaver_manager:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        if enable:
            screensaver_manager.setup()
            message = "Voice unlock monitoring enabled"
        else:
            screensaver_manager.shutdown()
            message = "Voice unlock monitoring disabled"
            
        return {
            "success": True,
            "enabled": enable,
            "message": message
        }
    except Exception as e:
        logger.error(f"Failed to toggle voice unlock: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/users")
async def list_enrolled_users():
    """List all enrolled users"""
    if not keychain_service:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        voiceprints = keychain_service.list_voiceprints()
        
        return {
            "success": True,
            "users": voiceprints,
            "count": len(voiceprints)
        }
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user_voiceprint(user_id: str):
    """Delete user's voiceprint"""
    if not keychain_service:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        success = keychain_service.delete_voiceprint(user_id)
        
        if success:
            return {
                "success": True,
                "message": f"Voiceprint deleted for user {user_id}"
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
            
    except Exception as e:
        logger.error(f"Failed to delete voiceprint: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/backup/export")
async def export_backup(password: Optional[str] = None):
    """Export encrypted backup of all voiceprints"""
    if not keychain_service:
        raise HTTPException(status_code=503, detail="Voice unlock not initialized")
    
    try:
        from pathlib import Path
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.vubak', delete=False) as tmp:
            backup_path = Path(tmp.name)
            
        success = keychain_service.export_backup(backup_path, password)
        
        if success:
            # Read backup data
            with open(backup_path, 'rb') as f:
                backup_data = f.read()
                
            # Clean up
            backup_path.unlink()
            
            # Return as base64
            return {
                "success": True,
                "backup_data": base64.b64encode(backup_data).decode(),
                "encrypted": password is not None
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create backup")
            
    except Exception as e:
        logger.error(f"Backup export error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config")
async def get_configuration():
    """Get current voice unlock configuration"""
    config = get_config()
    
    return {
        "audio": config.audio.__dict__,
        "enrollment": config.enrollment.__dict__,
        "authentication": config.authentication.__dict__,
        "security": {
            k: v for k, v in config.security.__dict__.items()
            if k not in ['encryption_key', 'master_key']  # Don't expose sensitive
        },
        "system": config.system.__dict__,
        "performance": config.performance.__dict__
    }


@router.post("/config/update")
async def update_configuration(section: str, updates: Dict[str, Any]):
    """Update voice unlock configuration"""
    try:
        config = get_config()
        
        if not hasattr(config, section):
            raise HTTPException(status_code=400, detail=f"Invalid config section: {section}")
            
        # Update configuration
        config.update_from_dict({section: updates})
        
        # Save to file
        config.save_to_file()
        
        return {
            "success": True,
            "message": f"Configuration section '{section}' updated"
        }
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Initialize on import if in main JARVIS context
try:
    from fastapi import FastAPI
    # Only initialize if we're being imported by the main app
    if "voice_unlock" not in globals():
        voice_unlock_initialized = initialize_voice_unlock()
except:
    pass