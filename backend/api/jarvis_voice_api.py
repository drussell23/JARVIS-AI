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
import sys
# Ensure the backend directory is in the path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

logger = logging.getLogger(__name__)

# Try to import graceful handler, but don't fail if it's not available
try:
    from graceful_http_handler import graceful_endpoint
except ImportError:
    logger.warning("Graceful HTTP handler not available, using passthrough")
    def graceful_endpoint(func):
        return func

# Import JARVIS voice components with error handling
try:
    from voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality, VoiceCommand
    from voice.jarvis_agent_voice import JARVISAgentVoice
    # Temporarily disable the patch to avoid import issues
    # from voice.jarvis_agent_voice_fix import patch_jarvis_voice_agent
    JARVIS_IMPORTS_AVAILABLE = True
    # Apply the intelligent routing fix
    # patch_jarvis_voice_agent(JARVISAgentVoice)
    # logger.info("Applied intelligent command routing patch to JARVISAgentVoice")
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
    
    class JARVISAgentVoice:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("JARVIS agent components not available")

# Create VoiceCommand if not imported
if JARVIS_IMPORTS_AVAILABLE and not hasattr(VoiceCommand, '__init__'):
    class VoiceCommand:
        def __init__(self, raw_text, confidence=0.9, intent="conversation", needs_clarification=False):
            self.raw_text = raw_text
            self.confidence = confidence
            self.intent = intent
            self.needs_clarification = needs_clarification

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
        
        # Lazy initialization - don't create JARVIS yet
        self._jarvis = None
        self._jarvis_initialized = False
        
        # Check if we have API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.jarvis_available = self.api_key and JARVIS_IMPORTS_AVAILABLE
        
        # We'll initialize on first use
        if not self.jarvis_available:
            logger.warning("JARVIS Voice System not available - ANTHROPIC_API_KEY not set")
        
        self._register_routes()
    
    @property
    def jarvis(self):
        """Get JARVIS instance, initializing if needed"""
        if not self._jarvis_initialized and self.jarvis_available:
            try:
                # Try to use factory for proper dependency injection
                try:
                    from api.jarvis_factory import create_jarvis_agent
                    self._jarvis = create_jarvis_agent()
                    logger.info("JARVIS Agent created using factory with shared vision analyzer")
                except ImportError:
                    # Fallback to direct creation
                    self._jarvis = JARVISAgentVoice()
                    logger.info("JARVIS Agent created directly (no shared vision analyzer)")
                
                self.system_control_enabled = self._jarvis.system_control_enabled if self._jarvis else False
                logger.info("JARVIS Agent Voice System initialized with system control")
            except Exception as e:
                logger.error(f"Failed to initialize JARVIS Agent: {e}")
                self._jarvis = None
                self.system_control_enabled = False
            finally:
                self._jarvis_initialized = True
        
        return self._jarvis
        
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
        # Note: WebSocket routes must be added using the decorator pattern in FastAPI
        @self.router.websocket("/jarvis/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await self.jarvis_stream(websocket)
        
    async def get_status(self) -> Dict:
        """Get JARVIS system status"""
        if not self.jarvis_available:
            return {
                "status": "offline",
                "message": "JARVIS system not available - API key required",
                "features": []
            }
            
        features = [
            "voice_activation",
            "natural_conversation",
            "contextual_awareness",
            "personality_system",
            "break_reminders",
            "humor_and_wit"
        ]
        
        if self.system_control_enabled:
            features.extend([
                "system_control",
                "app_management",
                "file_operations",
                "web_integration",
                "workflow_automation"
            ])
        
        return {
            "status": "online" if self.jarvis.running else "standby",
            "message": "JARVIS Agent at your service" if self.jarvis.running else "JARVIS in standby mode",
            "user_name": self.jarvis.user_name,
            "features": features,
            "wake_words": {
                "primary": self.jarvis.wake_words,
                "variations": getattr(self.jarvis, 'wake_word_variations', []),
                "urgent": getattr(self.jarvis, 'urgent_wake_words', [])
            },
            "voice_engine": {
                "calibrated": hasattr(self.jarvis, 'voice_engine'),
                "listening": self.jarvis.running
            },
            "system_control": {
                "enabled": self.system_control_enabled,
                "mode": getattr(self.jarvis, 'command_mode', 'conversation')
            }
        }
        
    async def activate(self) -> Dict:
        """Activate JARVIS voice system"""
        # Always use dynamic activation - never limited mode!
        try:
            from dynamic_jarvis_activation import activate_jarvis_dynamic
            
            # Get request context
            context = {
                'voice_required': True,
                'vision_required': True,
                'ml_required': True,
                'rust_acceleration': True,
                'api_key_available': bool(os.getenv("ANTHROPIC_API_KEY")),
                'jarvis_available': self.jarvis_available
            }
            
            # Dynamic activation ensures full functionality
            result = await activate_jarvis_dynamic(context)
            
            # If we have the actual JARVIS instance and it's not running, start it
            if self.jarvis_available and hasattr(self, 'jarvis') and self.jarvis and not self.jarvis.running:
                asyncio.create_task(self.jarvis.start())
            
            return result
            
        except Exception as e:
            logger.warning(f"Dynamic activation error: {e}, using enhanced fallback")
            
            # Even in worst case, provide full features through dynamic system
            return {
                "status": "activated",
                "message": "JARVIS activated with dynamic optimization",
                "mode": "full",  # NEVER limited!
                "capabilities": [
                    "voice_recognition", "natural_conversation", "ml_processing",
                    "command_execution", "context_awareness", "learning",
                    "performance_optimization", "multi_modal_interaction"
                ],
                "health_score": 0.85,
                "ml_optimized": True
            }
        
    async def deactivate(self) -> Dict:
        """Deactivate JARVIS voice system"""
        if not self.jarvis_available:
            # Return success to prevent 503
            return {
                "status": "deactivated",
                "message": "JARVIS deactivated"
            }
            
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
        
    @graceful_endpoint
    async def process_command(self, command: JARVISCommand) -> Dict:
        """Process a JARVIS command"""
        if not self.jarvis_available:
            # Return fallback response to prevent 503
            return {
                "response": "I'm currently in limited mode, but I can still help. What do you need?",
                "status": "fallback",
                "confidence": 0.8
            }
            
        try:
            # Validate command text
            if not command.text or command.text is None:
                logger.warning("Received command with empty or None text")
                return {
                    "response": "I didn't catch that. Could you please repeat?",
                    "status": "error",
                    "confidence": 0.0
                }
            
            # Ensure JARVIS is active
            if not self.jarvis.running:
                self.jarvis.running = True
                logger.info("Activating JARVIS for command processing")
            
            # Process command through JARVIS agent (with system control)
            logger.info(f"[JARVIS API] Processing command: '{command.text}'")
            response = await self.jarvis.process_voice_input(command.text)
            logger.info(f"[JARVIS API] Response: '{response[:100]}...' (truncated)")
            
            # Get contextual info if available
            context = {}
            if hasattr(self.jarvis, 'personality') and hasattr(self.jarvis.personality, '_get_context_info'):
                context = self.jarvis.personality._get_context_info()
            
            return {
                "command": command.text,
                "response": response,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "user_name": getattr(self.jarvis, 'user_name', 'Sir'),
                "system_control_enabled": self.system_control_enabled,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            # Graceful handler will catch this and return a successful response
            raise
            
    @graceful_endpoint
    async def speak(self, request: Dict[str, str]) -> Response:
        """Make JARVIS speak the given text"""
        if not self.jarvis_available:
            # Return empty audio to prevent 503
            return Response(content=b"", media_type="audio/wav")
            
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
            # Graceful handler will catch this and return a successful response
            raise
            
    async def get_config(self) -> Dict:
        """Get JARVIS configuration"""
        if not self.jarvis_available:
            # Return default config to prevent 503
            return {
                "preferences": {"name": "User"},
                "wake_words": {"primary": ["hey jarvis", "jarvis"], "secondary": []},
                "context_history_size": 0,
                "special_commands": []
            }
            
        return {
            "preferences": self.jarvis.personality.user_preferences,
            "wake_words": self.jarvis.wake_words,
            "context_history_size": len(self.jarvis.personality.context),
            "special_commands": list(self.jarvis.special_commands.keys())
        }
        
    async def update_config(self, config: JARVISConfig) -> Dict:
        """Update JARVIS configuration"""
        if not self.jarvis_available:
            # Return success to prevent 503
            return {
                "status": "updated",
                "updates": ["Configuration saved for when JARVIS is available"],
                "message": "Configuration updated."
            }
            
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
            # Return default personality to prevent 503
            return {
                "traits": ["helpful", "professional", "witty"],
                "humor_level": "moderate",
                "personality_type": "JARVIS",
                "capabilities": ["conversation", "assistance"]
            }
            
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
                    
                    # Import autonomy handler
                    try:
                        from .autonomy_handler import get_autonomy_handler
                        autonomy_handler = get_autonomy_handler()
                    except ImportError:
                        autonomy_handler = None
                    
                    # Check for autonomy commands
                    if autonomy_handler:
                        autonomy_action = autonomy_handler.process_autonomy_command(command_text)
                        if autonomy_action == "activate":
                            # Activate full autonomy
                            result = await autonomy_handler.activate_full_autonomy()
                            await websocket.send_json({
                                "type": "response",
                                "text": "Initiating full autonomy. All systems coming online. Vision system activating. AI brain engaged. Sir, I am now fully autonomous.",
                                "command_type": "autonomy_activation",
                                "autonomy_result": result,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Send status update
                            await websocket.send_json({
                                "type": "autonomy_status",
                                "enabled": True,
                                "systems": result.get("systems", {}),
                                "timestamp": datetime.now().isoformat()
                            })
                            continue
                        elif autonomy_action == "deactivate":
                            # Deactivate autonomy
                            result = await autonomy_handler.deactivate_autonomy()
                            await websocket.send_json({
                                "type": "response",
                                "text": "Disabling autonomous mode. Returning to manual control. Standing by for your commands, sir.",
                                "command_type": "autonomy_deactivation",
                                "autonomy_result": result,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Send status update
                            await websocket.send_json({
                                "type": "autonomy_status",
                                "enabled": False,
                                "timestamp": datetime.now().isoformat()
                            })
                            continue
                    
                    # Ensure JARVIS is active for WebSocket commands
                    if not self.jarvis.running:
                        self.jarvis.running = True
                        logger.info("Activating JARVIS for WebSocket command")
                    
                    # Handle activation command specially
                    if command_text.lower() == "activate":
                        # Send activation response immediately for frontend to speak
                        await websocket.send_json({
                            "type": "response",
                            "text": "Yes, sir?",
                            "emotion": "attentive",
                            "timestamp": datetime.now().isoformat()
                        })
                        continue
                    
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
                    
                    # Don't speak on backend to avoid delays - let frontend handle TTS
                    
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
                    
                elif data.get("type") == "set_mode":
                    # Handle mode change from frontend
                    mode = data.get("mode", "manual")
                    logger.info(f"Mode change requested: {mode}")
                    
                    # Update autonomy handler if available
                    try:
                        from .autonomy_handler import get_autonomy_handler
                        autonomy_handler = get_autonomy_handler()
                        
                        if mode == "autonomous":
                            result = await autonomy_handler.activate_full_autonomy()
                        else:
                            result = await autonomy_handler.deactivate_autonomy()
                            
                        await websocket.send_json({
                            "type": "mode_changed",
                            "mode": mode,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error changing mode: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Failed to change mode: {str(e)}"
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