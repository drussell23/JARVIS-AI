"""
Wake Word Service
=================

Main service for managing wake word detection and JARVIS activation.

This module provides the core wake word detection service that orchestrates
audio processing, wake word detection, and activation responses. It manages
the complete lifecycle from audio input to user command processing.

Classes:
    ServiceState: Enumeration of service states
    ActivationEvent: Data class for activation events
    WakeWordService: Main service class
    WakeWordAPI: API wrapper for the service

Example:
    >>> service = WakeWordService()
    >>> await service.start(activation_callback)
    >>> # Service now listening for wake words
    >>> await service.stop()
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
    """Enumeration of wake word service states.
    
    Attributes:
        STOPPED: Service is not running
        STARTING: Service is initializing
        RUNNING: Service is active and monitoring
        LISTENING: Service detected wake word and listening for command
        PROCESSING: Service is processing a user command
        ERROR: Service encountered an error
    """
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class ActivationEvent:
    """Represents a wake word activation event.
    
    Attributes:
        detection: The wake word detection that triggered this event
        response: The activation response given to the user
        timestamp: Unix timestamp when the activation occurred
        user_command: Optional command received after activation
        success: Whether the activation was successful
    """
    detection: Detection
    response: str
    timestamp: float
    user_command: Optional[str] = None
    success: bool = True


class WakeWordService:
    """Main service orchestrating wake word detection and JARVIS activation.
    
    This service manages the complete wake word detection pipeline including
    audio processing, detection engines, activation responses, and command
    listening states. It provides callbacks for integration with the main
    JARVIS system.
    
    Attributes:
        config: Configuration object for wake word settings
        audio_processor: Audio processing component
        detector: Wake word detection component
        state: Current service state
        is_listening_for_command: Whether waiting for user command
        command_timeout_task: Async task for command timeout
        activation_callback: Callback for wake word activations
        state_callback: Callback for state changes
        activation_history: List of recent activation events
        event_queue: Queue for processing events
    """
    
    def __init__(self):
        """Initialize wake word service.
        
        Sets up the service with default configuration and initializes
        all components in a stopped state.
        """
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
        """Start the wake word service.
        
        Initializes all components, calibrates audio, and begins monitoring
        for wake words. The service must be in STOPPED state to start.
        
        Args:
            activation_callback: Callback function to handle wake word activations.
                                Should accept a dict with activation details.
        
        Returns:
            True if service started successfully, False otherwise.
        
        Raises:
            Exception: If audio processor fails to start or other initialization errors.
        
        Example:
            >>> async def handle_activation(data):
            ...     print(f"Wake word: {data['wake_word']}")
            >>> success = await service.start(handle_activation)
        """
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
        """Stop the wake word service.
        
        Cleanly shuts down all components, cancels running tasks, and
        transitions to STOPPED state.
        """
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
        """Handle audio frame from processor.
        
        Processes incoming audio frames through the wake word detector
        when the service is in RUNNING state.
        
        Args:
            frame: Audio frame containing PCM data and timestamp
        """
        if self.detector and self.state == ServiceState.RUNNING:
            # Process for wake word
            self.detector.process_audio(frame.data, frame.timestamp)
    
    def _on_detection(self, detection: Detection):
        """Handle wake word detection.
        
        Called when the detector identifies a wake word. Triggers the
        activation handling process asynchronously.
        
        Args:
            detection: Detection object containing wake word details
        """
        logger.info(f"Wake word detected: {detection.wake_word} (confidence: {detection.confidence:.2f})")
        
        # Add to event queue
        asyncio.create_task(self._handle_activation(detection))
    
    def _on_detector_state_change(self, state: DetectionState):
        """Handle detector state changes.
        
        Updates service state based on detector state transitions.
        
        Args:
            state: New detector state
        """
        logger.debug(f"Detector state: {state}")
        
        if state == DetectionState.ACTIVATED:
            self._set_state(ServiceState.LISTENING)
    
    async def _handle_activation(self, detection: Detection):
        """Handle wake word activation.
        
        Orchestrates the complete activation process including playing sounds,
        generating responses, notifying callbacks, and setting up command
        listening state.
        
        Args:
            detection: The wake word detection that triggered activation
        
        Raises:
            Exception: If activation handling fails
        """
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
    
    def _get_activation_response(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Get context-aware activation response based on configuration.

        Generates intelligent, contextually appropriate responses based on
        time of day, user activity patterns, workspace context, and Phase 4
        proactive intelligence features.

        Args:
            context: Optional context containing:
                - proactive_mode: bool - Phase 4 proactive intelligence active
                - workspace: dict - Current workspace/app context
                - last_interaction: float - Timestamp of last interaction
                - user_focus_level: str - deep_work, focused, casual, idle

        Returns:
            Contextually appropriate activation response string.
            
        Example:
            >>> context = {'user_focus_level': 'deep_work', 'proactive_mode': True}
            >>> response = service._get_activation_response(context)
            >>> print(response)  # "Yes? I'll keep this brief."
        """
        context = context or {}
        responses = []

        # Get time context
        hour = time.localtime().tm_hour
        time_context = (
            'morning' if 5 <= hour < 12 else
            'afternoon' if 12 <= hour < 17 else
            'evening' if 17 <= hour < 22 else
            'night'
        )

        # Check for quick return (< 2 minutes)
        last_interaction = context.get('last_interaction', 0)
        is_quick_return = (time.time() - last_interaction) < 120 if last_interaction else False

        # Get Phase 4 context
        proactive_mode = context.get('proactive_mode', False)
        workspace = context.get('workspace', {})
        focus_level = context.get('user_focus_level', 'casual')

        # Priority 1: Quick return responses (casual, brief)
        if is_quick_return and random.random() < 0.7:
            return random.choice([
                "Yes?",
                "I'm here.",
                "Go ahead.",
                "Listening.",
                "Yes, Sir?",
                "What's next?",
                "Ready."
            ])

        # Priority 2: Phase 4 Proactive Mode (intelligent, aware)
        if proactive_mode and random.random() < 0.5:
            return random.choice([
                "Yes, Sir? I've been monitoring your workspace.",
                "I'm here. I have some suggestions when you're ready.",
                "At your service. I noticed a few patterns worth discussing.",
                "Listening. I've been keeping an eye on things.",
                "Yes? I'm tracking your workflow and ready to optimize."
            ])

        # Priority 3: Focus-aware responses (respect deep work)
        if focus_level == 'deep_work':
            return random.choice([
                "Yes? I'll keep this brief.",
                "I'm here. What do you need?",
                "Listening.",
                "Go ahead - I know you're focused."
            ])
        elif focus_level == 'focused':
            return random.choice([
                "Yes, Sir?",
                "I'm listening.",
                "Ready.",
                "What can I do for you?"
            ])

        # Priority 4: Workspace-aware responses
        focused_app = workspace.get('focused_app')
        if focused_app and random.random() < 0.3:
            return random.choice([
                f"Yes, Sir? I see you're working in {focused_app}.",
                f"I'm here. {focused_app} still open?",
                f"At your service. Need help with {focused_app}?"
            ])

        # Priority 5: Time-based responses (standard)
        time_responses = {
            'morning': [
                "Yes, Sir. How may I assist you this morning?",
                "Good morning. I'm listening.",
                "At your service, Sir.",
                "Morning. What can I do for you?",
                "Ready and listening, Sir."
            ],
            'afternoon': [
                "Yes, Sir. How can I help?",
                "At your service.",
                "I'm here. What do you need?",
                "Listening, Sir.",
                "Ready for your command."
            ],
            'evening': [
                "Yes, Sir. How may I assist you this evening?",
                "Good evening. I'm listening.",
                "At your service, Sir.",
                "Evening. What can I do for you?",
                "Ready and standing by."
            ],
            'night': [
                "Yes, Sir. Burning the midnight oil?",
                "I'm here. What do you need?",
                "Listening, Sir.",
                "Ready for your command, even at this late hour.",
                "At your service, Sir."
            ]
        }

        responses.extend(time_responses.get(time_context, time_responses['afternoon']))

        # Add idle-specific responses if user is idle
        if focus_level == 'idle':
            responses.extend([
                "Finally! What can I do for you?",
                "Welcome back. What would you like to tackle?",
                "Yes, Sir. Ready for action.",
                "I'm here. Let's get productive."
            ])

        # Add personality variations (20% chance)
        if random.random() < 0.2:
            responses.extend([
                "Systems nominal. How may I help?",
                "Neural pathways ready. What's the task?",
                "All systems operational. Your command?",
                "Ready to optimize your workflow, Sir.",
                "Standing by for instructions."
            ])

        # Add original configured responses
        responses.extend(self.config.response.activation_responses)

        # Select random response
        return random.choice(responses)
    
    async def _speak_response(self, response: str):
        """Speak the response using TTS.
        
        Integrates with the text-to-speech system to vocalize the
        activation response.
        
        Args:
            response: Text response to speak
        """
        # This would integrate with your existing TTS system
        logger.info(f"Speaking: {response}")
        # TODO: Integrate with JARVISVoiceAPI
    
    async def _play_activation_sound(self):
        """Play activation sound effect.
        
        Plays a configured sound effect to indicate wake word activation.
        """
        # TODO: Implement sound playback
        logger.debug("Playing activation sound")
    
    async def _command_timeout(self, timeout: float):
        """Handle command timeout.
        
        Waits for the specified timeout period and returns the service
        to idle state if no command is received.
        
        Args:
            timeout: Timeout duration in seconds
        """
        await asyncio.sleep(timeout)
        
        if self.is_listening_for_command:
            logger.info("Command timeout - returning to idle")
            self.is_listening_for_command = False
            self._set_state(ServiceState.RUNNING)
            
            # Could speak a timeout message
            if self.config.response.use_voice_response:
                await self._speak_response("Standing by, Sir.")
    
    async def handle_command_received(self):
        """Called when a command is received after wake word.
        
        Transitions the service from listening to processing state
        and cancels the command timeout.
        """
        self.is_listening_for_command = False
        
        # Cancel timeout
        if self.command_timeout_task:
            self.command_timeout_task.cancel()
            self.command_timeout_task = None
        
        self._set_state(ServiceState.PROCESSING)
    
    async def handle_command_complete(self):
        """Called when command processing is complete.
        
        Returns the service to the running state, ready for the next
        wake word detection.
        """
        self._set_state(ServiceState.RUNNING)
    
    def report_false_positive(self):
        """Report false positive detection.
        
        Notifies the detector that the last activation was a false positive
        to improve future detection accuracy through machine learning.
        """
        if self.detector:
            self.detector.report_false_positive()
    
    def _set_state(self, state: ServiceState):
        """Set service state.
        
        Updates the internal state and notifies registered callbacks
        of state changes.
        
        Args:
            state: New service state
        """
        if self.state != state:
            logger.debug(f"Service state: {self.state} -> {state}")
            self.state = state
            
            if self.state_callback:
                self.state_callback(state)
    
    async def _process_events(self):
        """Process events from the event queue.
        
        Continuously processes events while the service is running.
        This is a placeholder for future event handling functionality.
        
        Raises:
            Exception: If event processing encounters errors
        """
        while self.state != ServiceState.STOPPED:
            try:
                # This would handle various events
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status.
        
        Returns comprehensive status information about the service
        including configuration, state, and statistics.
        
        Returns:
            Dictionary containing:
                - enabled: Whether wake word detection is enabled
                - state: Current service state
                - is_listening: Whether listening for command
                - engines: Detection engine statistics
                - activation_count: Number of activations
                - last_activation: Timestamp of last activation
                - wake_words: Configured wake words
                - sensitivity: Current sensitivity setting
        """
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
        """Update configuration dynamically.
        
        Applies configuration changes without requiring a service restart.
        Some changes may require component reinitialization.
        
        Args:
            updates: Dictionary of configuration updates containing:
                - wake_words: List of wake words to detect
                - sensitivity: Detection sensitivity level
                - Other configuration parameters
        """
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
    """API wrapper for wake word service.
    
    Provides a clean API interface for external components to interact
    with the wake word service. Handles request validation and response
    formatting.
    
    Attributes:
        service: The WakeWordService instance to wrap
    """
    
    def __init__(self, service: WakeWordService):
        """Initialize API wrapper.
        
        Args:
            service: WakeWordService instance to wrap
        """
        self.service = service
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status.
        
        Returns:
            Dictionary containing current service status and configuration.
        """
        return self.service.get_status()
    
    async def enable(self) -> Dict[str, Any]:
        """Enable wake word detection.
        
        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - message: Status message
        """
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
        """Disable wake word detection.
        
        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - message: Status message
        """
        self.service.config.enabled = False
        return {
            'success': True,
            'message': 'Wake word detection disabled'
        }
    
    async def test_activation(self) -> Dict[str, Any]:
        """Test activation response.
        
        Generates and optionally speaks a test activation response
        without requiring an actual wake word detection.
        
        Returns:
            Dictionary containing:
                - success: Whether the test succeeded
                - response: The generated response text
        """
        response = self.service._get_activation_response()
        
        if self.service.config.response.use_voice_response:
            await self.service._speak_response(response)
        
        return {
            'success': True,
            'response': response
        }
    
    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update wake word settings.
        
        Args:
            settings: Dictionary of settings to update
        
        Returns:
            Dictionary containing:
                - success: Whether the update succeeded
                - message: Status message
                - current_settings: Updated service status
        """
        self.service.update_config(settings)
        return {
            'success': True,
            'message': 'Settings updated',
            'current_settings': self.service.get_status()
        }