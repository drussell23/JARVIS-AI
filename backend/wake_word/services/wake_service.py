"""
Wake Word Service
=================

Main service for managing wake word detection and JARVIS activation.
"""

import asyncio
import time
import random
import logging
import json
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from dataclasses import dataclass
import threading

from ..config import get_config
from ..core.audio_processor import AudioProcessor, AudioFrame
from ..core.detector import WakeWordDetector, Detection, DetectionState

logger = logging.getLogger(__name__)


class ServiceState(str, Enum):
    """Service state"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class ActivationEvent:
    """Represents a wake word activation event"""
    detection: Detection
    response: str
    timestamp: float
    user_command: Optional[str] = None
    success: bool = True


class WakeWordService:
    """
    Main service orchestrating wake word detection and JARVIS activation.
    """
    
    def __init__(self):
        """Initialize wake word service"""
        self.config = get_config()
        
        # Components
        self.audio_processor: Optional[AudioProcessor] = None
        self.detector: Optional[WakeWordDetector] = None
        
        # State
        self.state = ServiceState.STOPPED
        self.is_listening_for_command = False
        self.command_timeout_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.activation_callback: Optional[Callable[[str], Any]] = None
        self.state_callback: Optional[Callable[[ServiceState], None]] = None
        
        # History
        self.activation_history: List[ActivationEvent] = []
        
        # Event system
        self.event_queue = asyncio.Queue(maxsize=self.config.integration.event_queue_size)
        
        logger.info("Wake word service initialized")
    
    async def start(self, activation_callback: Callable[[str], Any]) -> bool:
        """Start the wake word service"""
        if self.state != ServiceState.STOPPED:
            logger.warning(f"Cannot start service in state: {self.state}")
            return False
        
        try:
            self._set_state(ServiceState.STARTING)
            
            # Store callback
            self.activation_callback = activation_callback
            
            # Initialize components
            self.audio_processor = AudioProcessor(callback=self._on_audio_frame)
            self.detector = WakeWordDetector()
            
            # Set detector callbacks
            self.detector.set_callbacks(
                detection_callback=self._on_detection,
                state_callback=self._on_detector_state_change
            )
            
            # Calibrate noise
            logger.info("Calibrating noise floor...")
            if self.audio_processor.calibrate_noise():
                logger.info("Noise calibration complete")
            
            # Start audio processing
            if not self.audio_processor.start():
                raise Exception("Failed to start audio processor")
            
            self._set_state(ServiceState.RUNNING)
            
            # Start event processor
            asyncio.create_task(self._process_events())
            
            logger.info("Wake word service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start wake word service: {e}")
            self._set_state(ServiceState.ERROR)
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the wake word service"""
        logger.info("Stopping wake word service...")
        
        # Cancel timeout task if running
        if self.command_timeout_task:
            self.command_timeout_task.cancel()
        
        # Stop audio processor
        if self.audio_processor:
            self.audio_processor.stop()
            self.audio_processor = None
        
        # Cleanup detector
        if self.detector:
            self.detector.cleanup()
            self.detector = None
        
        self._set_state(ServiceState.STOPPED)
        logger.info("Wake word service stopped")
    
    def _on_audio_frame(self, frame: AudioFrame):
        """Handle audio frame from processor"""
        if self.detector and self.state == ServiceState.RUNNING:
            # Process for wake word
            self.detector.process_audio(frame.data, frame.timestamp)
    
    def _on_detection(self, detection: Detection):
        """Handle wake word detection"""
        logger.info(f"Wake word detected: {detection.wake_word} (confidence: {detection.confidence:.2f})")
        
        # Add to event queue
        asyncio.create_task(self._handle_activation(detection))
    
    def _on_detector_state_change(self, state: DetectionState):
        """Handle detector state changes"""
        logger.debug(f"Detector state: {state}")
        
        if state == DetectionState.ACTIVATED:
            self._set_state(ServiceState.LISTENING)
    
    async def _handle_activation(self, detection: Detection):
        """Handle wake word activation"""
        try:
            # Play activation sound if configured
            if self.config.response.play_activation_sound:
                await self._play_activation_sound()
            
            # Get response
            response = self._get_activation_response()
            
            # Create activation event
            event = ActivationEvent(
                detection=detection,
                response=response,
                timestamp=time.time()
            )
            
            # Send response
            if self.config.response.use_voice_response:
                await self._speak_response(response)
            
            # Notify via callback
            if self.activation_callback:
                # The callback should trigger the UI to show visual feedback
                await self.activation_callback({
                    'type': 'wake_word_activated',
                    'response': response,
                    'wake_word': detection.wake_word,
                    'confidence': detection.confidence
                })
            
            # Set listening state
            self.is_listening_for_command = True
            self._set_state(ServiceState.LISTENING)
            
            # Start timeout for command
            self.command_timeout_task = asyncio.create_task(
                self._command_timeout(self.config.detection.wake_word_timeout)
            )
            
            # Confirm activation for learning
            self.detector.confirm_activation()
            
            # Add to history
            self.activation_history.append(event)
            
        except Exception as e:
            logger.error(f"Error handling activation: {e}")
    
    def _get_activation_response(self) -> str:
        """Get activation response based on configuration"""
        responses = self.config.response.activation_responses.copy()
        
        # Add time-based greetings if enabled
        if self.config.response.use_time_based_greetings:
            hour = time.localtime().tm_hour
            if 5 <= hour < 12:
                responses.append("Good morning Sir. How may I assist you?")
            elif 12 <= hour < 17:
                responses.append("Good afternoon Sir. What can I do for you?")
            elif 17 <= hour < 22:
                responses.append("Good evening Sir. How can I help?")
            else:
                responses.append("Hello Sir. Still working late I see. How can I help?")
        
        # Select random response
        return random.choice(responses)
    
    async def _speak_response(self, response: str):
        """Speak the response using TTS"""
        # This would integrate with your existing TTS system
        logger.info(f"Speaking: {response}")
        # TODO: Integrate with JARVISVoiceAPI
    
    async def _play_activation_sound(self):
        """Play activation sound effect"""
        # TODO: Implement sound playback
        logger.debug("Playing activation sound")
    
    async def _command_timeout(self, timeout: float):
        """Handle command timeout"""
        await asyncio.sleep(timeout)
        
        if self.is_listening_for_command:
            logger.info("Command timeout - returning to idle")
            self.is_listening_for_command = False
            self._set_state(ServiceState.RUNNING)
            
            # Could speak a timeout message
            if self.config.response.use_voice_response:
                await self._speak_response("Standing by, Sir.")
    
    async def handle_command_received(self):
        """Called when a command is received after wake word"""
        self.is_listening_for_command = False
        
        # Cancel timeout
        if self.command_timeout_task:
            self.command_timeout_task.cancel()
            self.command_timeout_task = None
        
        self._set_state(ServiceState.PROCESSING)
    
    async def handle_command_complete(self):
        """Called when command processing is complete"""
        self._set_state(ServiceState.RUNNING)
    
    def report_false_positive(self):
        """Report false positive detection"""
        if self.detector:
            self.detector.report_false_positive()
    
    def _set_state(self, state: ServiceState):
        """Set service state"""
        if self.state != state:
            logger.debug(f"Service state: {self.state} -> {state}")
            self.state = state
            
            if self.state_callback:
                self.state_callback(state)
    
    async def _process_events(self):
        """Process events from the event queue"""
        while self.state != ServiceState.STOPPED:
            try:
                # This would handle various events
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'enabled': self.config.enabled,
            'state': self.state,
            'is_listening': self.is_listening_for_command,
            'engines': self.detector.get_statistics() if self.detector else {},
            'activation_count': len(self.activation_history),
            'last_activation': self.activation_history[-1].timestamp if self.activation_history else None,
            'wake_words': self.config.detection.wake_words,
            'sensitivity': self.config.detection.sensitivity
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration dynamically"""
        # Update wake words
        if 'wake_words' in updates:
            self.config.detection.wake_words = updates['wake_words']
            # Reinitialize detector if needed
        
        # Update sensitivity
        if 'sensitivity' in updates:
            self.config.detection.sensitivity = updates['sensitivity']
            self.config.detection.threshold = self.config.get_sensitivity_value()
        
        logger.info(f"Configuration updated: {updates}")


class WakeWordAPI:
    """API wrapper for wake word service"""
    
    def __init__(self, service: WakeWordService):
        self.service = service
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return self.service.get_status()
    
    async def enable(self) -> Dict[str, Any]:
        """Enable wake word detection"""
        if self.service.state == ServiceState.STOPPED:
            # Service needs to be started by main app
            return {
                'success': False,
                'message': 'Wake word service not initialized'
            }
        
        self.service.config.enabled = True
        return {
            'success': True,
            'message': 'Wake word detection enabled'
        }
    
    async def disable(self) -> Dict[str, Any]:
        """Disable wake word detection"""
        self.service.config.enabled = False
        return {
            'success': True,
            'message': 'Wake word detection disabled'
        }
    
    async def test_activation(self) -> Dict[str, Any]:
        """Test activation response"""
        response = self.service._get_activation_response()
        
        if self.service.config.response.use_voice_response:
            await self.service._speak_response(response)
        
        return {
            'success': True,
            'response': response
        }
    
    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update wake word settings"""
        self.service.update_config(settings)
        return {
            'success': True,
            'message': 'Settings updated',
            'current_settings': self.service.get_status()
        }