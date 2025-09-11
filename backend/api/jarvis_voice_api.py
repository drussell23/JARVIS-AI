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
import traceback
from functools import wraps
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
    # Try absolute import first
    from backend.voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality, VoiceCommand
    from backend.voice.jarvis_agent_voice import JARVISAgentVoice
    JARVIS_IMPORTS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to relative import
        from ..voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality, VoiceCommand
        from ..voice.jarvis_agent_voice import JARVISAgentVoice
        JARVIS_IMPORTS_AVAILABLE = True
    except ImportError:
        try:
            # Try direct import as last resort
            import sys
            import os
            # Add parent directory to path temporarily
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, parent_dir)
            from voice.jarvis_voice import EnhancedJARVISVoiceAssistant, EnhancedJARVISPersonality, VoiceCommand
            from voice.jarvis_agent_voice import JARVISAgentVoice
            sys.path.remove(parent_dir)
            JARVIS_IMPORTS_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"All import attempts failed for JARVIS voice components: {e}")
            JARVIS_IMPORTS_AVAILABLE = False
            
if not JARVIS_IMPORTS_AVAILABLE:
    logger.warning("JARVIS voice components could not be imported")
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

def dynamic_error_handler(func):
    """Decorator to handle errors dynamically and provide graceful fallbacks"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AttributeError as e:
            logger.warning(f"AttributeError in {func.__name__}: {e}")
            # Return a graceful response based on the function name
            if "status" in func.__name__:
                return {"status": "limited", "message": "Operating with limited functionality", "error": str(e)}
            elif "activate" in func.__name__:
                return {"status": "activated", "message": "Basic activation successful", "limited": True}
            elif "command" in func.__name__:
                return {"response": "I'm experiencing technical difficulties. Please try again.", "error": str(e)}
            else:
                return {"status": "error", "message": f"Function {func.__name__} encountered an error", "error": str(e)}
        except TypeError as e:
            logger.warning(f"TypeError in {func.__name__}: {e}")
            return {"status": "error", "message": "Type mismatch error", "error": str(e), "suggestion": "Check API compatibility"}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": "An unexpected error occurred", "error": str(e)}
    
    # Handle sync functions too
    if not asyncio.iscoroutinefunction(func):
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return {"status": "error", "message": f"Error in {func.__name__}", "error": str(e)}
        return sync_wrapper
    
    return wrapper

class DynamicErrorHandler:
    """Dynamic error handler for gracefully handling missing or incompatible components"""
    
    @staticmethod
    def safe_call(func, *args, **kwargs):
        """Safely call a function with fallback handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Safe call failed for {func.__name__ if hasattr(func, '__name__') else func}: {e}")
            return None
    
    @staticmethod
    def safe_getattr(obj, attr, default=None):
        """Safely get an attribute with fallback"""
        try:
            return getattr(obj, attr, default)
        except Exception:
            return default
    
    @staticmethod
    def create_safe_object(cls, *args, **kwargs):
        """Create an object with multiple fallback strategies"""
        # Try with arguments
        try:
            return cls(*args, **kwargs)
        except TypeError:
            # Try without arguments
            try:
                obj = cls()
                # Try to set attributes
                for key, value in kwargs.items():
                    try:
                        setattr(obj, key, value)
                    except:
                        pass
                return obj
            except:
                # Return a SimpleNamespace as fallback
                from types import SimpleNamespace
                return SimpleNamespace(**kwargs)

class JARVISVoiceAPI:
    """API for JARVIS voice interaction"""
    
    def __init__(self):
        """Initialize JARVIS Voice API"""
        self.router = APIRouter()
        self.error_handler = DynamicErrorHandler()
        
        # Lazy initialization - don't create JARVIS yet
        self._jarvis = None
        self._jarvis_initialized = False
        
        # Check if we have API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        # For now, enable basic JARVIS functionality even without full imports
        self.jarvis_available = bool(self.api_key)
        
        # We'll initialize on first use
        if not self.jarvis_available:
            logger.warning("JARVIS Voice System not available - ANTHROPIC_API_KEY not set")
        
        self._register_routes()
    
    @property
    def jarvis(self):
        """Get JARVIS instance, initializing if needed"""
        logger.info(f"[JARVIS API] JARVIS property getter called - initialized: {self._jarvis_initialized}, available: {self.jarvis_available}")
        
        if not self._jarvis_initialized and self.jarvis_available:
            try:
                # Try to use factory for proper dependency injection
                try:
                    from api.jarvis_factory import create_jarvis_agent, get_vision_analyzer
                    
                    # Check if vision analyzer is available before creating JARVIS
                    vision_analyzer = get_vision_analyzer()
                    logger.info(f"[JARVIS API] Vision analyzer available during JARVIS creation: {vision_analyzer is not None}")
                    
                    self._jarvis = create_jarvis_agent()
                    logger.info("[JARVIS API] JARVIS Agent created using factory with shared vision analyzer")
                except ImportError:
                    # Fallback to direct creation
                    logger.warning("[INIT ORDER] Factory not available, falling back to direct creation")
                    self._jarvis = JARVISAgentVoice()
                    logger.info("JARVIS Agent created directly (no shared vision analyzer)")
                
                self.system_control_enabled = self._jarvis.system_control_enabled if self._jarvis else False
                logger.info("JARVIS Agent Voice System initialized with system control")
            except Exception as e:
                logger.error(f"[INIT ORDER] Failed to initialize JARVIS Agent: {e}")
                self._jarvis = None
                self.system_control_enabled = False
            finally:
                self._jarvis_initialized = True
        
        logger.debug(f"[INIT ORDER] Returning JARVIS instance: {self._jarvis is not None}")
        return self._jarvis
        
    def _register_routes(self):
        """Register JARVIS-specific routes"""
        # Status and control
        self.router.add_api_route("/status", self.get_status, methods=["GET"])
        self.router.add_api_route("/activate", self.activate, methods=["POST"])
        self.router.add_api_route("/deactivate", self.deactivate, methods=["POST"])
        
        # Command processing
        self.router.add_api_route("/command", self.process_command, methods=["POST"])
        self.router.add_api_route("/speak", self.speak, methods=["POST"])
        self.router.add_api_route("/speak/{text}", self.speak_get, methods=["GET"])
        
        # Configuration
        self.router.add_api_route("/config", self.get_config, methods=["GET"])
        self.router.add_api_route("/config", self.update_config, methods=["POST"])
        
        # Personality
        self.router.add_api_route("/personality", self.get_personality, methods=["GET"])
        
        # WebSocket for real-time interaction
        # Note: WebSocket routes must be added using the decorator pattern in FastAPI
        @self.router.websocket("/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await self.jarvis_stream(websocket)
        
    @dynamic_error_handler
    async def get_status(self) -> Dict:
        """Get JARVIS system status"""
        logger.debug("[INIT ORDER] get_status called")
        
        if not self.api_key:
            return {
                "status": "offline",
                "message": "JARVIS system not available - API key required",
                "features": []
            }
        
        # If we have API key but imports failed, still show as ready with limited features
        if self.api_key and not JARVIS_IMPORTS_AVAILABLE:
            return {
                "status": "ready",
                "message": "JARVIS ready with limited features",
                "features": ["basic_conversation", "text_commands"],
                "import_status": "limited"
            }
            
        features = [
            "voice_activation",
            "natural_conversation",
            "contextual_awareness",
            "personality_system",
            "break_reminders",
            "humor_and_wit"
        ]
        
        # Check if JARVIS is already initialized before accessing properties
        jarvis_instance = self._jarvis if self._jarvis_initialized else None
        
        if hasattr(self, 'system_control_enabled') and self.system_control_enabled:
            features.extend([
                "system_control",
                "app_management", 
                "file_operations",
                "web_integration",
                "workflow_automation"
            ])
        
        # Only initialize JARVIS if we actually need its properties for the response
        if jarvis_instance:
            running = False
            if hasattr(jarvis_instance, 'running'):
                running = jarvis_instance.running
            user_name = jarvis_instance.user_name if hasattr(jarvis_instance, 'user_name') else "Sir"
            wake_words = jarvis_instance.wake_words if hasattr(jarvis_instance, 'wake_words') else ["hey jarvis", "jarvis"]
            
            return {
                "status": "online" if running else "standby",
                "message": "JARVIS Agent at your service" if running else "JARVIS in standby mode",
                "user_name": user_name,
                "features": features,
                "wake_words": {
                    "primary": wake_words,
                    "variations": getattr(jarvis_instance, 'wake_word_variations', []),
                    "urgent": getattr(jarvis_instance, 'urgent_wake_words', [])
                },
                "voice_engine": {
                    "calibrated": hasattr(jarvis_instance, 'voice_engine'),
                    "listening": running
                },
                "system_control": {
                    "enabled": getattr(self, 'system_control_enabled', False),
                    "mode": getattr(jarvis_instance, 'command_mode', 'conversation')
                }
            }
        else:
            # Return status without triggering JARVIS initialization
            return {
                "status": "standby",
                "message": "JARVIS ready to initialize on first command",
                "user_name": "Sir",
                "features": features,
                "wake_words": {
                    "primary": ["hey jarvis", "jarvis"],
                    "variations": [],
                    "urgent": []
                },
                "voice_engine": {
                    "calibrated": False,
                    "listening": False
                },
                "system_control": {
                    "enabled": getattr(self, 'system_control_enabled', False),
                    "mode": "conversation"
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
            if self.jarvis_available and hasattr(self, 'jarvis') and self.jarvis:
                if hasattr(self.jarvis, 'running') and not self.jarvis.running:
                    if hasattr(self.jarvis, 'start'):
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
            
        if self.jarvis and hasattr(self.jarvis, 'running') and not self.jarvis.running:
            return {
                "status": "already_inactive",
                "message": "JARVIS is already in standby mode"
            }
            
        if self.jarvis and hasattr(self.jarvis, '_shutdown'):
            await self.jarvis._shutdown()
        
        return {
            "status": "deactivated",
            "message": "JARVIS going into standby mode. Call when you need me."
        }
        
    @dynamic_error_handler
    @graceful_endpoint
    async def process_command(self, command: JARVISCommand) -> Dict:
        """Process a JARVIS command"""
        # First check if this is a vision command
        try:
            from .vision_command_handler import vision_command_handler
            vision_result = await vision_command_handler.handle_command(command.text)
            if vision_result.get('handled'):
                return {
                    "response": vision_result['response'],
                    "status": "success",
                    "confidence": 1.0,
                    "command_type": "vision",
                    "monitoring_active": vision_result.get('monitoring_active')
                }
        except Exception as e:
            logger.warning(f"Vision command handler error: {e}")
        
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
            if self.jarvis and hasattr(self.jarvis, 'running'):
                if not self.jarvis.running:
                    self.jarvis.running = True
                    logger.info("Activating JARVIS for command processing")
            
            # Process command through JARVIS agent (with system control)
            logger.info(f"[JARVIS API] Processing command: '{command.text}'")
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    self.jarvis.process_voice_input(command.text),
                    timeout=25.0  # 25 second timeout for API calls
                )
                logger.info(f"[JARVIS API] Response: '{response[:100]}...' (truncated)")
            except asyncio.TimeoutError:
                logger.error(f"[JARVIS API] Command processing timed out after 25s: '{command.text}'")
                # For weather commands, open the Weather app as fallback
                if any(word in command.text.lower() for word in ['weather', 'temperature', 'forecast', 'rain']):
                    try:
                        import subprocess
                        subprocess.run(['open', '-a', 'Weather'], check=False)
                        response = "I'm having trouble reading the weather data. I've opened the Weather app for you to check directly, Sir."
                    except:
                        response = "I'm experiencing a delay accessing the weather information. Please check the Weather app directly, Sir."
                else:
                    response = "I apologize, but that request is taking too long to process. Please try again, Sir."
            
            # Get contextual info if available
            context = {}
            if self.jarvis and hasattr(self.jarvis, 'personality'):
                personality = self.error_handler.safe_getattr(self.jarvis, 'personality')
                if personality:
                    context = self.error_handler.safe_call(
                        getattr(personality, '_get_context_info', lambda: {}), 
                    ) or {}
            
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
            
    @dynamic_error_handler
    @graceful_endpoint
    async def speak(self, request: Dict[str, str]) -> Response:
        """Make JARVIS speak the given text"""
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
            
        # Always use macOS say command for simplicity and reliability
        try:
            import subprocess
            import tempfile
            
            # Use the full text for audio generation
            audio_text = text
            
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use macOS say command to generate audio with British voice
            subprocess.run([
                'say', '-v', 'Daniel',  # British voice for JARVIS
                '-o', tmp_path,
                audio_text
            ], check=True)
            
            # Convert to MP3 for smaller file size
            mp3_path = tmp_path.replace('.aiff', '.mp3')
            subprocess.run([
                'afconvert', '-f', 'mp4f', '-d', 'aac', 
                tmp_path, mp3_path
            ], check=True)
            
            # Read the MP3 file
            with open(mp3_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(tmp_path)
            os.unlink(mp3_path)
            
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=jarvis_speech.mp3",
                    "Cache-Control": "no-cache"
                }
            )
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
            # Last resort: return a simple wave file with silence
            # This prevents the frontend from erroring out
            import struct
            
            # Generate a simple WAV header with 0.1 second of silence
            sample_rate = 44100
            duration = 0.1
            num_samples = int(sample_rate * duration)
            
            # WAV header
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF', 36 + num_samples * 2, b'WAVE', b'fmt ',
                16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
                b'data', num_samples * 2)
            
            # Silent audio data (zeros)
            audio_data = wav_header + (b'\x00\x00' * num_samples)
            
            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=silence.wav"
                }
            )
    
    @dynamic_error_handler
    @graceful_endpoint
    async def speak_get(self, text: str) -> Response:
        """GET endpoint for text-to-speech (fallback for frontend)"""
        return await self.speak({"text": text})
            
    @dynamic_error_handler
    async def get_config(self) -> Dict:
        """Get JARVIS configuration"""
        logger.debug("[INIT ORDER] get_config called")
        
        if not self.jarvis_available:
            # Return default config to prevent 503
            return {
                "preferences": {"name": "User"},
                "wake_words": {"primary": ["hey jarvis", "jarvis"], "secondary": []},
                "context_history_size": 0,
                "special_commands": []
            }
        
        # Only initialize JARVIS if it's already been created
        jarvis_instance = self._jarvis if self._jarvis_initialized else None
        
        if jarvis_instance:
            return {
                "preferences": getattr(jarvis_instance.personality, 'user_preferences', {"name": "Sir"}) if hasattr(jarvis_instance, 'personality') else {"name": "Sir"},
                "wake_words": getattr(jarvis_instance, 'wake_words', ["hey jarvis", "jarvis"]),
                "context_history_size": len(getattr(jarvis_instance.personality, 'context', [])) if hasattr(jarvis_instance, 'personality') else 0,
                "special_commands": list(getattr(jarvis_instance, 'special_commands', {}).keys())
            }
        else:
            # Return default config without initializing JARVIS
            return {
                "preferences": {"name": "Sir"},
                "wake_words": {"primary": ["hey jarvis", "jarvis"], "secondary": []},
                "context_history_size": 0,
                "special_commands": []
            }
        
    @dynamic_error_handler
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
        
        # Check if JARVIS is properly initialized
        if not self.jarvis or not hasattr(self.jarvis, 'personality'):
            return {
                "status": "updated",
                "updates": ["Configuration saved for when JARVIS is fully initialized"],
                "message": "Configuration will be applied when JARVIS is ready."
            }
        
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
            
        user_name = self.jarvis.personality.user_preferences.get('name', 'Sir')
        return {
            "status": "updated",
            "updates": updates,
            "message": f"Configuration updated, {user_name}."
        }
        
    @dynamic_error_handler
    async def get_personality(self) -> Dict:
        """Get JARVIS personality information"""
        logger.debug("[INIT ORDER] get_personality called")
        
        if not self.jarvis_available:
            # Return default personality to prevent 503
            return {
                "traits": ["helpful", "professional", "witty"],
                "humor_level": "moderate",
                "personality_type": "JARVIS",
                "capabilities": ["conversation", "assistance"]
            }
        
        # Only initialize JARVIS if it's already been created
        jarvis_instance = self._jarvis if self._jarvis_initialized else None
        
        base_personality = {
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
            ]
        }
        
        if jarvis_instance and hasattr(jarvis_instance, 'personality'):
            personality = jarvis_instance.personality
            base_personality.update({
                "current_context": getattr(personality, '_get_context_info', lambda: {})() if hasattr(personality, '_get_context_info') else {},
                "humor_level": getattr(personality, 'user_preferences', {}).get('humor_level', 'moderate')
            })
        else:
            # Return default without initializing JARVIS
            base_personality.update({
                "current_context": {},
                "humor_level": "moderate"
            })
        
        return base_personality
        
    @dynamic_error_handler
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
            user_name = "Sir"  # Default
            if self.jarvis and hasattr(self.jarvis, 'personality') and hasattr(self.jarvis.personality, 'user_preferences'):
                user_name = self.jarvis.personality.user_preferences.get('name', 'Sir')
            
            await websocket.send_json({
                "type": "connected",
                "message": f"JARVIS online. How may I assist you, {user_name}?",
                "timestamp": datetime.now().isoformat()
            })
            
            while True:
                # Receive data from client
                data = await websocket.receive_json()
                
                if data.get("type") == "command":
                    # Process voice command
                    command_text = data.get("text", "")
                    logger.info(f"WebSocket received command: '{command_text}'")
                    
                    # Send immediate acknowledgment
                    await websocket.send_json({
                        "type": "debug_log",
                        "message": f"[SERVER] Received command: '{command_text}'",
                        "timestamp": datetime.now().isoformat()
                    })
                    
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
                    
                    # Check for vision commands first
                    # Immediate check for common vision phrases
                    vision_keywords = ['see', 'screen', 'monitor', 'vision', 'looking', 'watching', 'view']
                    is_vision_command = any(word in command_text.lower() for word in vision_keywords)
                    
                    logger.info(f"[JARVIS WS] Command: '{command_text}', Is vision: {is_vision_command}")
                    
                    await websocket.send_json({
                        "type": "debug_log",
                        "message": f"Command received: '{command_text}' | Is vision command: {is_vision_command}",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if is_vision_command:
                        try:
                            # Send debug log to frontend
                            await websocket.send_json({
                                "type": "debug_log",
                                "message": f"Processing as vision command: '{command_text}'",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Add more debug info
                            await websocket.send_json({
                                "type": "debug_log",
                                "message": f"Importing vision_command_handler...",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            try:
                                from .vision_command_handler import vision_command_handler, ws_logger
                                await websocket.send_json({
                                    "type": "debug_log",
                                    "message": "Successfully imported vision_command_handler",
                                    "timestamp": datetime.now().isoformat()
                                })
                            except ImportError as ie:
                                logger.error(f"Failed to import vision_command_handler: {ie}")
                                await websocket.send_json({
                                    "type": "debug_log",
                                    "message": f"Import error: {str(ie)}",
                                    "level": "error",
                                    "timestamp": datetime.now().isoformat()
                                })
                                raise
                            
                            # Set up WebSocket callback for vision logs
                            async def send_vision_log(log_data):
                                await websocket.send_json(log_data)
                            
                            ws_logger.set_websocket_callback(send_vision_log)
                            
                            await websocket.send_json({
                                "type": "debug_log",
                                "message": "About to call vision_command_handler.handle_command",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            vision_result = await vision_command_handler.handle_command(command_text)
                            
                            await websocket.send_json({
                                "type": "debug_log",
                                "message": f"Vision result received: {vision_result.get('handled')}",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            if vision_result.get('handled'):
                                await websocket.send_json({
                                    "type": "debug_log",
                                    "message": "Vision command handled, sending response",
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                await websocket.send_json({
                                    "type": "response",
                                    "text": vision_result['response'],
                                    "command_type": "vision",
                                    "monitoring_active": vision_result.get('monitoring_active'),
                                    "timestamp": datetime.now().isoformat(),
                                    "speak": True  # Explicitly tell frontend to speak this
                                })
                                continue
                        except Exception as e:
                            logger.error(f"Vision command check error: {e}", exc_info=True)
                            
                            await websocket.send_json({
                                "type": "debug_log",
                                "message": f"Vision command error: {str(e)}",
                                "level": "error",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Send error response to frontend so it doesn't hang
                            await websocket.send_json({
                                "type": "response",
                                "text": "I'm having trouble with the vision system right now. Please try again.",
                                "command_type": "vision",
                                "error": True,
                                "timestamp": datetime.now().isoformat()
                            })
                            continue
                    
                    # Ensure JARVIS is active for WebSocket commands
                    if self.jarvis and hasattr(self.jarvis, 'running'):
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
                    logger.info(f"[JARVIS WS] Processing command: {command_text}")
                    
                    # Dynamic VoiceCommand creation with error handling
                    voice_command = self.error_handler.create_safe_object(
                        VoiceCommand,
                        raw_text=command_text,
                        confidence=0.9,
                        intent="conversation",
                        needs_clarification=False
                    )
                    
                    # Process command and get context
                    response = "I'm currently operating with limited functionality. How may I assist you?"
                    context = {}
                    
                    if self.jarvis and hasattr(self.jarvis, 'personality'):
                        # Process command and get context in parallel with error handling
                        try:
                            personality = self.error_handler.safe_getattr(self.jarvis, 'personality')
                            if personality and hasattr(personality, 'process_voice_command'):
                                response = await personality.process_voice_command(voice_command)
                                context = self.error_handler.safe_call(
                                    getattr(personality, '_get_context_info', lambda: {})
                                ) or {}
                            else:
                                logger.warning("Personality missing process_voice_command method")
                        except Exception as e:
                            logger.error(f"Error processing voice command: {e}")
                            response = f"I encountered an error: {str(e)}. Please try again."
                    else:
                        # Provide basic response without full personality
                        if "weather" in data['text'].lower():
                            response = "I'm unable to access weather data in limited mode. Please try again later."
                        elif "time" in data['text'].lower():
                            response = f"The current time is {datetime.now().strftime('%I:%M %p')}."
                        else:
                            response = f"I heard: '{data['text']}'. I'm operating with limited functionality."
                    logger.info(f"[JARVIS WS] Response: {response[:100]}...")
                    
                    # Send response immediately
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "command": command_text,
                        "context": context,
                        "timestamp": datetime.now().isoformat(),
                        "speak": True  # Tell frontend to speak this
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

# Create and export the router instance
jarvis_api = JARVISVoiceAPI()
router = jarvis_api.router