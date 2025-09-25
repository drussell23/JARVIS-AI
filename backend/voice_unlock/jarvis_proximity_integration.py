"""
JARVIS Proximity + Voice Authentication Integration
==================================================

Integrates the advanced proximity + voice authentication system with JARVIS.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import sys

logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "proximity_voice_auth"))
sys.path.insert(0, str(Path(__file__).parent / "proximity_voice_auth" / "python"))

from auth_engine.dual_factor_auth import DualFactorAuthEngine

# Import VoiceUnlockConnector from WebSocket server
try:
    from objc.server.websocket_server import VoiceUnlockConnector
except ImportError:
    # Fallback to None if not available
    VoiceUnlockConnector = None
    # Only log to logger, not to stdout/stderr
    logger.debug("VoiceUnlockConnector not available - this is normal if proximity auth is not being used")


class ProximityVoiceAuthManager:
    """Manages the proximity + voice authentication system for JARVIS."""
    
    def __init__(self):
        self.auth_engine: Optional[DualFactorAuthEngine] = None
        self.proximity_service: Optional[subprocess.Popen] = None
        self.voice_connector: Optional[VoiceUnlockConnector] = None
        self.is_initialized = False
        self.user_id = "jarvis_user"
        
        # Paths
        self.base_dir = Path(__file__).parent / "proximity_voice_auth"
        self.swift_binary = self.base_dir / "bin" / "ProximityService"
        
    async def initialize(self) -> bool:
        """Initialize the proximity + voice auth system."""
        try:
            logger.info("🚀 Initializing Proximity + Voice Auth System...")
            
            # 1. Start proximity service
            if not await self._start_proximity_service():
                return False
            
            # 2. Initialize auth engine
            self.auth_engine = DualFactorAuthEngine()
            await self.auth_engine.initialize(self.user_id)
            
            # 3. Initialize voice unlock connector
            self.voice_connector = VoiceUnlockConnector()
            
            self.is_initialized = True
            logger.info("✅ Proximity + Voice Auth System initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize proximity auth: {e}")
            return False
    
    async def _start_proximity_service(self) -> bool:
        """Start the Swift proximity detection service."""
        try:
            if not self.swift_binary.exists():
                logger.error(f"Proximity service binary not found at {self.swift_binary}")
                logger.info("Run ./build_swift.sh to build the service")
                return False
            
            # Start the service
            self.proximity_service = subprocess.Popen(
                [str(self.swift_binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for service to start
            await asyncio.sleep(1)
            
            # Check if running
            if self.proximity_service.poll() is not None:
                stdout, stderr = self.proximity_service.communicate()
                logger.error(f"Proximity service failed to start: {stderr.decode()}")
                return False
            
            logger.info("✅ Proximity service started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting proximity service: {e}")
            return False
    
    async def enroll_voice(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        """
        Enroll user voice for authentication.
        
        Args:
            audio_data: Voice sample audio data
            sample_rate: Audio sample rate
            
        Returns:
            Enrollment result
        """
        if not self.auth_engine:
            return {
                "success": False,
                "reason": "Auth engine not initialized"
            }
        
        try:
            # Convert bytes to numpy array for voice authenticator
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Enroll through voice authenticator
            result = self.auth_engine.voice_authenticator.enroll_voice(
                audio_array, 
                sample_rate
            )
            
            if result['success']:
                logger.info(f"Voice enrolled successfully for user {self.user_id}")
            else:
                logger.warning(f"Voice enrollment failed: {result.get('reason')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Voice enrollment error: {e}")
            return {
                "success": False,
                "reason": str(e)
            }
    
    async def authenticate(self, audio_data: bytes = None, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform proximity + voice authentication.
        
        Args:
            audio_data: Optional voice data for active authentication
            sample_rate: Audio sample rate
            
        Returns:
            Authentication result
        """
        if not self.auth_engine:
            return {
                "success": False,
                "reason": "Auth engine not initialized",
                "action": None
            }
        
        try:
            # If no audio provided, check proximity only
            if audio_data is None:
                proximity_score = await self.auth_engine._get_proximity_score()
                
                if proximity_score >= self.auth_engine.proximity_threshold:
                    # High proximity - could allow certain actions
                    return {
                        "success": True,
                        "reason": "Proximity verified",
                        "action": "proximity_unlock",
                        "proximity_score": proximity_score
                    }
                else:
                    return {
                        "success": False,
                        "reason": "Not in proximity",
                        "action": None,
                        "proximity_score": proximity_score
                    }
            
            # Full dual-factor authentication
            result = await self.auth_engine.authenticate(audio_data, sample_rate)
            
            # Convert auth result to JARVIS response
            response = {
                "success": result.success,
                "reason": result.reason,
                "proximity_score": result.proximity_score,
                "voice_score": result.voice_score,
                "combined_score": result.combined_score,
                "action": None
            }
            
            if result.success:
                response["action"] = "full_unlock"
                # Trigger unlock through voice unlock system
                if self.voice_connector:
                    await self.voice_connector.send_command("unlock_screen", {
                        "source": "proximity_voice_auth",
                        "authenticated": True
                    })
            
            return response
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                "success": False,
                "reason": str(e),
                "action": None
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = {
            "initialized": self.is_initialized,
            "proximity_service_running": False,
            "auth_engine_ready": False,
            "voice_enrolled": False
        }
        
        if self.proximity_service:
            status["proximity_service_running"] = self.proximity_service.poll() is None
        
        if self.auth_engine:
            status["auth_engine_ready"] = True
            engine_status = self.auth_engine.get_status()
            status.update({
                "voice_enrolled": engine_status.get("voice_model_ready", False),
                "proximity_connected": engine_status.get("proximity_connected", False),
                "thresholds": engine_status.get("config", {})
            })
        
        return status
    
    async def shutdown(self):
        """Shutdown the proximity + voice auth system."""
        logger.info("Shutting down Proximity + Voice Auth System...")
        
        # Shutdown auth engine
        if self.auth_engine:
            await self.auth_engine.shutdown()
        
        # Stop proximity service
        if self.proximity_service:
            self.proximity_service.terminate()
            self.proximity_service.wait()
        
        self.is_initialized = False
        logger.info("✅ Proximity + Voice Auth System shut down")


# Singleton instance
_proximity_auth_manager = None


async def get_proximity_auth_manager() -> ProximityVoiceAuthManager:
    """Get the global proximity auth manager instance."""
    global _proximity_auth_manager
    
    if _proximity_auth_manager is None:
        _proximity_auth_manager = ProximityVoiceAuthManager()
        await _proximity_auth_manager.initialize()
    
    return _proximity_auth_manager


# Integration with JARVIS commands
async def handle_proximity_voice_command(command: str, audio_data: bytes = None) -> Dict[str, Any]:
    """
    Handle proximity + voice authentication commands from JARVIS.
    
    Args:
        command: Command type (enroll, authenticate, status, etc.)
        audio_data: Optional audio data for voice commands
        
    Returns:
        Command result
    """
    manager = await get_proximity_auth_manager()
    
    if command == "enroll_voice":
        if audio_data:
            return await manager.enroll_voice(audio_data, 16000)
        else:
            return {
                "success": False,
                "reason": "No audio data provided for enrollment"
            }
    
    elif command == "authenticate":
        return await manager.authenticate(audio_data, 16000)
    
    elif command == "check_proximity":
        # Proximity-only check
        return await manager.authenticate(audio_data=None)
    
    elif command == "status":
        return await manager.get_status()
    
    elif command == "shutdown":
        await manager.shutdown()
        return {"success": True, "message": "System shut down"}
    
    else:
        return {
            "success": False,
            "reason": f"Unknown command: {command}"
        }