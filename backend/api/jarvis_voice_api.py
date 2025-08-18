"""
JARVIS Voice API - Voice endpoints with Iron Man-style personality
Integrates JARVIS personality with the web application
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import json
import base64
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Import JARVIS voice components with error handling
try:
    from voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality, VoiceCommand
    JARVIS_IMPORTS_AVAILABLE = True
except (ImportError, OSError, AttributeError) as e:
    logger.warning(f"Failed to import JARVIS voice components: {e}")
    JARVIS_IMPORTS_AVAILABLE = False
    # Create stub classes to prevent NameError
    class EnhancedJARVISVoiceAssistant:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("JARVIS voice components not available")
    
    class EnhancedJARVISPersonality:
        pass
    
    class VoiceCommand:
        pass


class JARVISCommand(BaseModel):
    """Voice command request"""
    text: str
    audio_data: Optional[str] = None  # Base64 encoded audio


class JARVISConfig(BaseModel):
    """JARVIS configuration update"""
    user_name: Optional[str] = None
    humor_level: Optional[str] = None  # low, moderate, high
    work_hours: Optional[tuple] = None
    break_reminder: Optional[bool] = None


class JARVISVoiceAPI:
    """API for JARVIS voice interaction"""
    
    def __init__(self):
        """Initialize JARVIS Voice API"""
        self.router = APIRouter()
        
        # Initialize JARVIS if API key and imports are available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and JARVIS_IMPORTS_AVAILABLE:
            try:
                self.jarvis = EnhancedJARVISVoiceAssistant(api_key)
                self.jarvis_available = True
                logger.info("JARVIS Voice System initialized")
            except Exception as e:
                self.jarvis_available = False
                logger.error(f"Failed to initialize JARVIS: {e}")
        else:
            self.jarvis = None
            self.jarvis_available = False
            logger.warning("JARVIS Voice System not available - ANTHROPIC_API_KEY not set")
        
        self._register_routes()
        
    def _register_routes(self):
        """Register JARVIS-specific routes"""
        # Status and control
        self.router.add_api_route("/jarvis/status", self.get_status, methods=["GET"])
        self.router.add_api_route("/jarvis/activate", self.activate, methods=["POST"])
        self.router.add_api_route("/jarvis/deactivate", self.deactivate, methods=["POST"])
        
        # Command processing
        self.router.add_api_route("/jarvis/command", self.process_command, methods=["POST"])
        self.router.add_api_route("/jarvis/speak", self.speak, methods=["POST"])
        
        # Configuration
        self.router.add_api_route("/jarvis/config", self.get_config, methods=["GET"])
        self.router.add_api_route("/jarvis/config", self.update_config, methods=["POST"])
        
        # Personality
        self.router.add_api_route("/jarvis/personality", self.get_personality, methods=["GET"])
        
        # WebSocket for real-time interaction
        self.router.add_api_websocket_route("/jarvis/stream", self.jarvis_stream)
        
    async def get_status(self) -> Dict:
        """Get JARVIS system status"""
        if not self.jarvis_available:
            return {
                "status": "offline",
                "message": "JARVIS system not available - API key required",
                "features": []
            }
            
        return {
            "status": "online" if self.jarvis.running else "standby",
            "message": "JARVIS at your service" if self.jarvis.running else "JARVIS in standby mode",
            "user_name": self.jarvis.personality.user_preferences['name'],
            "features": [
                "voice_activation",
                "natural_conversation",
                "contextual_awareness",
                "personality_system",
                "break_reminders",
                "humor_and_wit"
            ],
            "wake_words": self.jarvis.wake_words,
            "voice_engine": {
                "calibrated": hasattr(self.jarvis.voice_engine, 'recognizer'),
                "listening": self.jarvis.running
            }
        }
        
    async def activate(self) -> Dict:
        """Activate JARVIS voice system"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        if self.jarvis.running:
            return {
                "status": "already_active",
                "message": "JARVIS is already online, sir"
            }
            
        # Start JARVIS in background
        asyncio.create_task(self.jarvis.start())
        
        return {
            "status": "activating",
            "message": "JARVIS coming online. All systems operational."
        }
        
    async def deactivate(self) -> Dict:
        """Deactivate JARVIS voice system"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        if not self.jarvis.running:
            return {
                "status": "already_inactive",
                "message": "JARVIS is already in standby mode"
            }
            
        await self.jarvis._shutdown()
        
        return {
            "status": "deactivated",
            "message": "JARVIS going into standby mode. Call when you need me."
        }
        
    async def process_command(self, command: JARVISCommand) -> Dict:
        """Process a JARVIS command"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        try:
            # Process command with JARVIS personality
            # Create a VoiceCommand object for the personality
            voice_command = VoiceCommand(
                raw_text=command.text,
                confidence=0.9,  # High confidence for text commands
                intent="conversation",
                needs_clarification=False
            )
            response = await self.jarvis.personality.process_voice_command(voice_command)
            
            # Get contextual info
            context = self.jarvis.personality._get_context_info()
            
            return {
                "command": command.text,
                "response": response,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "user_name": self.jarvis.personality.user_preferences['name']
            }
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def speak(self, request: Dict[str, str]) -> Response:
        """Make JARVIS speak the given text"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
            
        try:
            # Use JARVIS voice engine to speak
            # In a real implementation, this would return audio data
            self.jarvis.voice_engine.speak(text)
            
            return Response(
                content=json.dumps({
                    "status": "success",
                    "message": "Speech synthesized",
                    "text": text
                }),
                media_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def get_config(self) -> Dict:
        """Get JARVIS configuration"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        return {
            "preferences": self.jarvis.personality.user_preferences,
            "wake_words": self.jarvis.wake_words,
            "context_history_size": len(self.jarvis.personality.context),
            "special_commands": list(self.jarvis.special_commands.keys())
        }
        
    async def update_config(self, config: JARVISConfig) -> Dict:
        """Update JARVIS configuration"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        updates = []
        
        if config.user_name:
            self.jarvis.personality.user_preferences['name'] = config.user_name
            updates.append(f"User designation updated to {config.user_name}")
            
        if config.humor_level:
            self.jarvis.personality.user_preferences['humor_level'] = config.humor_level
            updates.append(f"Humor level adjusted to {config.humor_level}")
            
        if config.work_hours:
            self.jarvis.personality.user_preferences['work_hours'] = config.work_hours
            updates.append(f"Work hours updated to {config.work_hours[0]}-{config.work_hours[1]}")
            
        if config.break_reminder is not None:
            self.jarvis.personality.user_preferences['break_reminder'] = config.break_reminder
            updates.append(f"Break reminders {'enabled' if config.break_reminder else 'disabled'}")
            
        return {
            "status": "updated",
            "updates": updates,
            "message": f"Configuration updated, {self.jarvis.personality.user_preferences['name']}."
        }
        
    async def get_personality(self) -> Dict:
        """Get JARVIS personality information"""
        if not self.jarvis_available:
            raise HTTPException(status_code=503, detail="JARVIS not available")
            
        return {
            "personality_traits": [
                "Professional yet personable",
                "British accent and sophisticated vocabulary",
                "Dry humor and wit",
                "Protective and loyal",
                "Anticipates user needs",
                "Contextually aware"
            ],
            "example_responses": [
                "Of course, sir. Shall I also cancel your 3 o'clock?",
                "The weather is partly cloudy, 72 degrees. Perfect for flying, if I may say so, sir.",
                "Sir, your heart rate suggests you haven't taken a break in 3 hours.",
                "I've taken the liberty of ordering your usual coffee, sir.",
                "Might I suggest the Mark 42? It's a personal favorite."
            ],
            "current_context": self.jarvis.personality._get_context_info(),
            "humor_level": self.jarvis.personality.user_preferences['humor_level']
        }
        
    async def jarvis_stream(self, websocket: WebSocket):
        """WebSocket endpoint for real-time JARVIS interaction"""
        await websocket.accept()
        
        if not self.jarvis_available:
            await websocket.send_json({
                "type": "error",
                "message": "JARVIS not available - API key required"
            })
            await websocket.close()
            return
            
        try:
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "message": f"JARVIS online. How may I assist you, {self.jarvis.personality.user_preferences['name']}?",
                "timestamp": datetime.now().isoformat()
            })
            
            while True:
                # Receive data from client
                data = await websocket.receive_json()
                
                if data.get("type") == "command":
                    # Process voice command
                    command_text = data.get("text", "")
                    logger.info(f"WebSocket received command: '{command_text}'")
                    
                    # Handle activation command specially
                    if command_text.lower() == "activate":
                        # Speak activation response on backend
                        if hasattr(self.jarvis, 'voice_engine') and hasattr(self.jarvis.voice_engine, 'speak'):
                            try:
                                await asyncio.to_thread(self.jarvis.voice_engine.speak, "Yes, sir?")
                            except Exception as e:
                                logger.warning(f"Backend activation speech failed: {e}")
                    
                    # Send acknowledgment immediately
                    await websocket.send_json({
                        "type": "processing",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Process with JARVIS - FAST
                    voice_command = VoiceCommand(
                        raw_text=command_text,
                        confidence=0.9,
                        intent="conversation",
                        needs_clarification=False
                    )
                    
                    # Process command and get context in parallel
                    response_task = asyncio.create_task(
                        self.jarvis.personality.process_voice_command(voice_command)
                    )
                    context_task = asyncio.create_task(
                        asyncio.to_thread(self.jarvis.personality._get_context_info)
                    )
                    
                    response = await response_task
                    context = await context_task
                    
                    # Send response immediately
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "command": command_text,
                        "context": context,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Also speak on backend if on macOS
                    if hasattr(self.jarvis, 'voice_engine') and hasattr(self.jarvis.voice_engine, 'speak'):
                        try:
                            await asyncio.to_thread(self.jarvis.voice_engine.speak, response)
                        except Exception as e:
                            logger.warning(f"Backend speech failed: {e}")
                    
                elif data.get("type") == "audio":
                    # Handle audio data (base64 encoded)
                    audio_data = base64.b64decode(data.get("data", ""))
                    
                    # In a real implementation, process audio through speech recognition
                    # For now, acknowledge receipt
                    await websocket.send_json({
                        "type": "audio_received",
                        "size": len(audio_data),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                elif data.get("type") == "ping":
                    # Heartbeat
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
        except WebSocketDisconnect:
            logger.info("JARVIS WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in JARVIS WebSocket: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })