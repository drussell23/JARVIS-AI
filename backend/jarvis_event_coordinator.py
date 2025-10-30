#!/usr/bin/env python3
"""
JARVIS Event Coordinator - Central Hub for Event-Driven Architecture

This module provides the central coordination system for JARVIS using an event-driven
architecture. It orchestrates all components through the event bus for seamless
integration and manages complex workflows across voice, vision, memory, and control
systems.

The coordinator handles:
- Component lifecycle management
- Event-driven workflow orchestration
- Error recovery and system resilience
- Memory pressure coordination
- Performance monitoring and debugging

Example:
    >>> coordinator = JARVISEventCoordinator(user_name="Sir")
    >>> await coordinator.start()
    >>> # JARVIS is now running with all components coordinated
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Core Event System
from core.event_bus import Event, EventPriority, get_event_bus
from core.event_types import (
    EventTypes, EventBuilder, SystemEvents, VoiceEvents, 
    VisionEvents, ControlEvents, MemoryEvents,
    subscribe_to, subscribe_to_pattern
)
from core.event_metrics import get_metrics_collector
from core.event_web_ui import EventWebUI
from core.event_replay import get_event_replayer, get_event_debugger

# Memory Management
from core.memory_controller_v2 import get_memory_controller

# Import component interfaces (v2 versions with events)
from voice.ml_enhanced_voice_system_v2 import MLEnhancedVoiceSystem
from vision.intelligent_vision_integration_v2 import IntelligentJARVISVision
from system_control.macos_controller_v2 import MacOSController

# Additional components
from voice.jarvis_personality_adapter import PersonalityAdapter as JARVISPersonalityAdapter
from autonomy.autonomous_behaviors import AutonomousBehaviorManager as AutonomousBehaviorEngine

logger = logging.getLogger(__name__)

@dataclass
class CoordinatorState:
    """State tracking for the JARVIS coordinator.
    
    Maintains the current operational state of the coordinator including
    session status, user information, and component health.
    
    Attributes:
        is_running: Whether the coordinator is currently active
        active_session: Whether there's an active user interaction session
        current_user: Name of the current user
        last_activity: Timestamp of the last user interaction
        active_workflows: List of currently executing workflow IDs
        component_status: Status of each system component
    """
    is_running: bool = False
    active_session: bool = False
    current_user: str = "Sir"
    last_activity: Optional[datetime] = None
    active_workflows: List[str] = field(default_factory=list)
    component_status: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class WorkflowContext:
    """Context information for multi-step workflows.
    
    Tracks the progress and state of complex operations that span
    multiple components and require coordination.
    
    Attributes:
        workflow_id: Unique identifier for the workflow
        workflow_type: Type/category of the workflow
        started_at: When the workflow was initiated
        steps_completed: List of completed workflow steps
        pending_steps: List of remaining workflow steps
        context_data: Additional data needed for workflow execution
    """
    workflow_id: str
    workflow_type: str
    started_at: datetime
    steps_completed: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    
class JARVISEventCoordinator:
    """Central coordinator for JARVIS using event-driven architecture.
    
    This class manages all JARVIS components through loose coupling via events,
    orchestrating complex workflows and ensuring system resilience. It serves
    as the central hub that coordinates between voice, vision, memory, and
    control systems.
    
    The coordinator handles:
    - Component initialization and lifecycle management
    - Event-driven workflow orchestration
    - Error recovery and system health monitoring
    - Memory pressure coordination
    - Performance metrics collection
    
    Attributes:
        user_name: Name of the current user
        state: Current operational state of the coordinator
        event_bus: Central event bus for component communication
        event_builder: Builder for creating and publishing events
        metrics_collector: Collects performance metrics
        replayer: Event replay system for debugging
        debugger: Event debugging utilities
        memory_controller: Memory management controller
        voice_system: Voice processing component
        vision_system: Computer vision component
        control_system: System control component
        personality: Personality adaptation component
        autonomy: Autonomous behavior component
        web_ui: Web-based debugging interface
        active_workflows: Currently executing workflows
    """
    
    def __init__(self, user_name: str = "Sir", enable_web_ui: bool = True):
        """Initialize the JARVIS Event Coordinator.
        
        Args:
            user_name: Name to use when addressing the user
            enable_web_ui: Whether to enable the web-based debugging UI
            
        Example:
            >>> coordinator = JARVISEventCoordinator(
            ...     user_name="Alice",
            ...     enable_web_ui=True
            ... )
        """
        self.user_name = user_name
        self.state = CoordinatorState(current_user=user_name)
        
        # Event System
        self.event_bus = get_event_bus()
        self.event_builder = EventBuilder()
        self.metrics_collector = get_metrics_collector()
        self.replayer = get_event_replayer()
        self.debugger = get_event_debugger()
        
        # Memory Controller (singleton)
        self.memory_controller = get_memory_controller()
        
        # Component instances (will be initialized on start)
        self.voice_system = None
        self.vision_system = None
        self.control_system = None
        self.personality = None
        self.autonomy = None
        
        # Web UI for debugging
        self.web_ui = None
        if enable_web_ui:
            self.web_ui = EventWebUI()
            
        # Workflow management
        self.active_workflows: Dict[str, WorkflowContext] = {}
        
        # Setup core event subscriptions
        self._setup_core_subscriptions()
        
        logger.info(f"JARVIS Event Coordinator initialized for {user_name}")
        
    def _setup_core_subscriptions(self) -> None:
        """Setup core event subscriptions for coordination.
        
        Establishes event handlers for system-wide coordination including
        error handling, wake word detection, command processing, and
        component status monitoring.
        """
        
        # System-wide error handling
        @subscribe_to(EventTypes.SYSTEM_ERROR, priority=EventPriority.CRITICAL)
        async def handle_system_error(event: Event):
            """Handle system-wide errors with component-specific recovery."""
            error = event.payload.get("error", "Unknown error")
            source = event.source
            logger.error(f"System error from {source}: {error}")
            
            # Attempt recovery based on source
            if "voice" in source:
                await self._recover_voice_system()
            elif "vision" in source:
                await self._recover_vision_system()
            elif "control" in source:
                await self._recover_control_system()
                
        # Wake word detection starts a session
        @subscribe_to(EventTypes.VOICE_WAKE_WORD_DETECTED)
        async def handle_wake_word(event: Event):
            """Handle wake word detection to start interaction session."""
            confidence = event.payload.get("confidence", 0)
            if confidence > 0.7:
                logger.info(f"Wake word detected with confidence {confidence}")
                self.state.active_session = True
                self.state.last_activity = datetime.now()
                
                # Prepare all systems
                await self._prepare_for_interaction()
                
        # Voice commands trigger workflows
        @subscribe_to(EventTypes.VOICE_COMMAND_RECEIVED)
        async def handle_voice_command(event: Event):
            """Handle voice commands by triggering appropriate workflows."""
            command = event.payload.get("command", "")
            confidence = event.payload.get("confidence", 0)
            
            if confidence > 0.6:
                logger.info(f"Processing command: {command}")
                await self._process_command(command, confidence)
                
        # Vision analysis results
        @subscribe_to(EventTypes.VISION_ANALYSIS_COMPLETE)
        async def handle_vision_complete(event: Event):
            """Handle vision analysis completion for workflow continuation."""
            analysis_id = event.payload.get("results", {}).get("analysis_id")
            
            # Check if this is part of a workflow
            for workflow_id, context in self.active_workflows.items():
                if context.context_data.get("vision_analysis_id") == analysis_id:
                    await self._continue_workflow(workflow_id, "vision_complete", event.payload)
                    break
                    
        # Memory pressure coordination
        @subscribe_to(EventTypes.MEMORY_PRESSURE_CHANGED)
        async def handle_memory_pressure(event: Event):
            """Handle memory pressure changes with coordinated response."""
            new_level = event.payload.get("new_level")
            logger.warning(f"Memory pressure changed to: {new_level}")
            
            if new_level == "critical":
                # Coordinate emergency response
                await self._handle_critical_memory()
                
        # Component status updates
        @subscribe_to_pattern("*.system_ready")
        async def handle_component_ready(event: Event):
            """Handle component ready notifications."""
            component = event.source
            self.state.component_status[component] = "ready"
            logger.info(f"Component ready: {component}")
            
            # Check if all components are ready
            if self._all_components_ready():
                await self._on_all_components_ready()
                
    async def start(self) -> None:
        """Start the JARVIS coordinator and all components.
        
        Initializes all system components, starts monitoring services,
        and announces system readiness.
        
        Raises:
            RuntimeError: If coordinator is already running
            MemoryError: If insufficient memory for initialization
            
        Example:
            >>> coordinator = JARVISEventCoordinator()
            >>> await coordinator.start()
            # JARVIS is now fully operational
        """
        if self.state.is_running:
            logger.warning("Coordinator already running")
            return
            
        logger.info("Starting JARVIS Event Coordinator...")
        
        # Publish startup event
        SystemEvents.startup(
            source="jarvis_coordinator",
            version="2.0",
            config={
                "user": self.user_name,
                "event_driven": True,
                "components": ["voice", "vision", "memory", "control"]
            }
        )
        
        # Start Web UI if enabled
        if self.web_ui:
            await self.web_ui.start()
            logger.info(f"Event Web UI available at http://localhost:{self.web_ui.port}")
            
        # Initialize components
        await self._initialize_components()
        
        # Start memory monitoring
        await self.memory_controller.start_monitoring()
        
        # Enable event tracing in debug mode
        if os.getenv("JARVIS_DEBUG", "").lower() == "true":
            self.debugger.enable_trace()
            
        self.state.is_running = True
        logger.info("JARVIS Event Coordinator started successfully")
        
        # Announce readiness
        await self._announce_ready()
        
    async def _initialize_components(self) -> None:
        """Initialize all JARVIS components.
        
        Loads and initializes each system component with proper error
        handling and memory management.
        
        Raises:
            MemoryError: If insufficient memory for component initialization
        """
        logger.info("Initializing components...")
        
        # Check memory before loading
        can_load, reason = self.memory_controller.should_load_model(2048)  # 2GB estimate
        if not can_load:
            logger.error(f"Cannot initialize components: {reason}")
            SystemEvents.error(
                source="jarvis_coordinator",
                error="Insufficient memory for initialization",
                details={"reason": reason}
            )
            return
            
        # Initialize voice system
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.voice_system = MLEnhancedVoiceSystem(api_key)
                await self.voice_system.start()
                self.state.component_status["voice"] = "initializing"
            else:
                logger.warning("No Anthropic API key - voice system limited")
        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            self.state.component_status["voice"] = "error"
            
        # Initialize vision system
        try:
            self.vision_system = IntelligentJARVISVision()
            self.state.component_status["vision"] = "ready"
        except Exception as e:
            logger.error(f"Failed to initialize vision: {e}")
            self.state.component_status["vision"] = "error"
            
        # Initialize control system
        try:
            self.control_system = MacOSController()
            self.state.component_status["control"] = "ready"
        except Exception as e:
            logger.error(f"Failed to initialize control: {e}")
            self.state.component_status["control"] = "error"
            
        # Initialize personality
        try:
            self.personality = JARVISPersonalityAdapter(self.user_name)
            self.state.component_status["personality"] = "ready"
        except Exception as e:
            logger.error(f"Failed to initialize personality: {e}")
            self.state.component_status["personality"] = "error"
            
        # Initialize autonomy
        try:
            self.autonomy = AutonomousBehaviorEngine()
            self.state.component_status["autonomy"] = "ready"
        except Exception as e:
            logger.error(f"Failed to initialize autonomy: {e}")
            self.state.component_status["autonomy"] = "error"
            
    def _all_components_ready(self) -> bool:
        """Check if all critical components are ready.
        
        Returns:
            bool: True if all critical components are operational
        """
        critical_components = ["voice", "vision", "control"]
        return all(
            self.state.component_status.get(comp) == "ready" 
            for comp in critical_components
        )
        
    async def _on_all_components_ready(self) -> None:
        """Called when all components are ready.
        
        Publishes system ready event and performs final initialization.
        """
        logger.info("All components ready - JARVIS fully operational")
        
        # Publish system ready event
        self.event_builder.publish(
            "jarvis.system_ready",
            source="jarvis_coordinator",
            payload={
                "components": self.state.component_status,
                "timestamp": time.time()
            }
        )
        
    async def _announce_ready(self) -> None:
        """Announce JARVIS is ready for interaction.
        
        Generates and publishes a ready message through the voice system.
        """
        if self.voice_system:
            # Generate ready message
            response = f"All systems operational, {self.user_name}. How may I assist you?"
            
            # Publish response event
            VoiceEvents.response_generated(
                source="jarvis_coordinator",
                response=response,
                processing_time=0.0
            )
            
    async def _prepare_for_interaction(self) -> None:
        """Prepare systems for user interaction.
        
        Captures current context and prepares components for
        optimal response to user commands.
        """
        # Capture current screen for context
        if self.vision_system:
            asyncio.create_task(
                self.vision_system.capture_and_analyze_with_events(
                    "Capture current screen context for conversation",
                    context={"session_start": True}
                )
            )
            
    async def _process_command(self, command: str, confidence: float) -> None:
        """Process a voice command and coordinate response.
        
        Analyzes the command, determines the appropriate workflow,
        and coordinates execution across relevant components.
        
        Args:
            command: The voice command text
            confidence: Confidence score of the voice recognition
        """
        command_lower = command.lower()
        
        # Create workflow context
        workflow_id = f"cmd_{int(time.time() * 1000)}"
        workflow_type = self._determine_workflow_type(command_lower)
        
        context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            started_at=datetime.now(),
            context_data={
                "command": command,
                "confidence": confidence
            }
        )
        
        self.active_workflows[workflow_id] = context
        
        # Publish workflow start
        ControlEvents.workflow_started(
            source="jarvis_coordinator",
            workflow_name=workflow_type,
            components=self._get_workflow_components(workflow_type)
        )
        
        # Route to appropriate workflow
        if workflow_type == "vision_query":
            await self._handle_vision_workflow(workflow_id, command)
        elif workflow_type == "app_control":
            await self._handle_app_workflow(workflow_id, command)
        elif workflow_type == "system_control":
            await self._handle_system_workflow(workflow_id, command)
        elif workflow_type == "web_search":
            await self._handle_web_workflow(workflow_id, command)
        else:
            await self._handle_general_workflow(workflow_id, command)
            
    def _determine_workflow_type(self, command: str) -> str:
        """Determine the type of workflow needed for a command.
        
        Analyzes command text to categorize the type of operation
        required and route to appropriate handlers.
        
        Args:
            command: The command text to analyze
            
        Returns:
            str: The workflow type identifier
        """
        vision_keywords = ["see", "look", "screen", "show", "analyze", "what's on"]
        app_keywords = ["open", "close", "launch", "quit", "switch"]
        system_keywords = ["volume", "brightness", "wifi", "screenshot", "sleep"]
        web_keywords = ["search", "google", "browse", "website"]
        
        if any(keyword in command for keyword in vision_keywords):
            return "vision_query"
        elif any(keyword in command for keyword in app_keywords):
            return "app_control"
        elif any(keyword in command for keyword in system_keywords):
            return "system_control"
        elif any(keyword in command for keyword in web_keywords):
            return "web_search"
        else:
            return "general"
            
    def _get_workflow_components(self, workflow_type: str) -> List[str]:
        """Get components involved in a workflow type.
        
        Args:
            workflow_type: The type of workflow
            
        Returns:
            List[str]: List of component names involved in the workflow
        """
        components_map = {
            "vision_query": ["voice", "vision"],
            "app_control": ["voice", "control"],
            "system_control": ["voice", "control"],
            "web_search": ["voice", "control"],
            "general": ["voice", "personality"]
        }
        return components_map.get(workflow_type, ["voice"])
        
    async def _handle_vision_workflow(self, workflow_id: str, command: str) -> None:
        """Handle vision-related workflows.
        
        Coordinates vision analysis operations including screen capture,
        image analysis, and response generation.
        
        Args:
            workflow_id: Unique identifier for the workflow
            command: The vision-related command
        """
        context = self.active_workflows[workflow_id]
        context.pending_steps = ["capture", "analyze", "respond"]
        
        # Vision system will handle and publish events
        response = await self.vision_system.handle_intelligent_command(command)
        
        # Complete workflow
        context.steps_completed = context.pending_steps
        context.pending_steps = []
        
        await self._complete_workflow(workflow_id, success=True, response=response)
        
    async def _handle_app_workflow(self, workflow_id: str, command: str) -> None:
        """Handle application control workflows.
        
        Coordinates application launch, control, and management operations.
        
        Args:
            workflow_id: Unique identifier for the workflow
            command: The application control command
        """
        context = self.active_workflows[workflow_id]
        
        # Let system control handle via events
        # The macos_controller will process voice commands automatically
        
        # Wait a bit for command execution
        await asyncio.sleep(1)
        
        # Check for any errors via events
        # Complete workflow
        await self._complete_workflow(workflow_id, success=True)
        
    async def _handle_system_workflow(self, workflow_id: str, command: str) -> None:
        """Handle system control workflows.
        
        Coordinates system-level operations like volume control,
        brightness adjustment, and system settings.
        
        Args:
            workflow_id: Unique identifier for the workflow
            command: The system control command
        """
        # Similar to app workflow
        await self._handle_app_workflow(workflow_id, command)
        
    async def _handle_web_workflow(self, workflow_id: str, command: str) -> None:
        """Handle web search workflows.
        
        Coordinates web search operations including query extraction
        and browser automation.
        
        Args:
            workflow_id: Unique identifier for the workflow
            command: The web search command
        """
        context = self.active_workflows[workflow_id]
        
        # Extract search query
        query = command.lower().replace("search for", "").replace("google", "").strip()
        
        if self.control_system:
            success, message = self.control_system.web_search(query)
            
            response = f"I've searched for {query} for you, {self.user_name}." if success else f"Failed to search: {message}"
            
            await self._complete_workflow(workflow_id, success=success, response=response)
            
    async def _handle_general_workflow(self, workflow_id: str, command: str) -> None:
        """Handle general conversation workflows.
        
        Processes general conversational commands using the personality
        system for natural responses.
        
        Args:
            workflow_id: Unique identifier for the workflow
            command: The general command or query
        """
        context = self.active_workflows[workflow_id]
        
        # Use personality for response
        if self.personality:
            response = self.personality.generate_response(
                command,
                context={"confidence": context.context_data.get("confidence", 1.0)}
            )
        else:
            response = "I understand your request, but I'm having trouble processing it right now."
            
        await self._complete_workflow(workflow_id, success=True, response=response)
        
    async def _continue_workflow(self, workflow_id: str, step: str, data: Dict[str, Any]) -> None:
        """Continue a multi-step workflow.
        
        Updates workflow progress and checks for completion.
        
        Args:
            workflow_id: Unique identifier for the workflow
            step: The completed step name
            data: Data from the completed step
        """
        if workflow_id not in self.active_workflows:
            return
            
        context = self.active_workflows[workflow_id]
        context.steps_completed.append(step)
        
        # Store step data
        context.context_data[f"{step}_data"] = data
        
        # Check if workflow is complete
        if not context.pending_steps or all(
            step in context.steps_completed for step in context.pending_steps
        ):
            await self._complete_workflow(workflow_id, success=True)
            
    async def _complete_workflow(self, workflow_id: str, success: bool, 
                               response: Optional[str] = None, error: Optional[str] = None) -> None:
        """Complete a workflow and clean up resources.
        
        Publishes completion events, generates responses, and performs cleanup.
        
        Args:
            workflow_id: Unique identifier for the workflow
            success: Whether the workflow completed successfully
            response: Optional response message to generate
            error: Optional error message if workflow failed
        """
        if workflow_id not in self.active_workflows:
            return
            
        context = self.active_workflows[workflow_id]
        
        # Publish workflow complete event
        ControlEvents.workflow_completed(
            source="jarvis_coordinator",
            workflow_name=context.workflow_type,
            success=success,
            results=context.steps_completed,
            error=error
        )
        
        # Generate response if needed
        if response:
            VoiceEvents.response_generated(
                source="jarvis_coordinator",
                response=response,
                processing_time=(datetime.now() - context.started_at).total_seconds()
            )
            
        # Clean up
        del self.active_workflows[workflow_id]
        
    async def _handle_critical_memory(self) -> None:
        """Handle critical memory pressure situations.
        
        Coordinates emergency response to critical memory conditions
        including pausing operations and unloading models.
        """
        logger.warning("Handling critical memory pressure")
        
        # Pause non-essential operations
        self.event_builder.publish(
            "jarvis.pause_operations",
            source="jarvis_coordinator",
            payload={
                "reason": "memory_critical",
                "components": ["vision", "autonomy"]
            },
            priority=EventPriority.CRITICAL
        )
        
        # Clear all caches
        if self.vision_system:
            self.vision_system.clear_cache()
            
        # Request model unloading
        self.event_builder.publish(
            "jarvis.unload_models",
            source="jarvis_coordinator",
            payload={
                "priority": "non_essential",
                "reason": "memory_critical"
            },
            priority=EventPriority.CRITICAL
        )
        
    async def _recover_voice_system(self) -> None:
        """Attempt to recover voice system after error.
        
        Reinitializes the voice system with proper error handling.
        
        Raises:
            Exception: If voice system recovery fails
        """
        logger.info("Attempting voice system recovery")
        
        try:
            # Reinitialize voice
            if self.voice_system:
                await self.voice_system.stop()
                
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.voice_system = MLEnhancedVoiceSystem(api_key)
                await self.voice_system.start()
                
            logger.info("Voice system recovered")
        except Exception as e:
            logger.error(f"Failed to recover voice system: {e}")
            
    async def _recover_vision_system(self) -> None:
        """Attempt to recover vision system after error.
        
        Reinitializes the vision system with proper error handling.
        
        Raises:
            Exception: If vision system recovery fails
        """
        logger.info("Attempting vision system recovery")
        
        try:
            # Reinitialize vision
            self.vision_system = IntelligentJARVISVision()
            logger.info("Vision system recovered")
        except Exception as e:
            logger.error(f"Failed to recover vision system: {e}")
            
    async def _recover_control_system(self) -> None:
        """Attempt to recover control system after error.
        
        Reinitializes the control system with proper error handling.
        
        Raises:
            Exception: If control system recovery fails
        """
        logger.info("Attempting control system recovery")
        
        try:
            # Reinitialize control
            self.control_system = MacOSController()
            logger.info("Control system recovered")
        except Exception as e:
            logger.error(f"Failed to recover control system: {e}")
            
    async def stop(self) -> None:
        """Stop the coordinator and all components.
        
        Performs graceful shutdown of all components and services.
        """
        logger.info("Stopping JARVIS Event Coordinator...")
        
        # Publish shutdown event
        SystemEvents.shutdown(
            source="jarvis_coordinator",
            reason="user_requested"
        )
        
        # Stop components
        if self.voice_system:
            await self.voice_system.stop()
            
        # Stop memory monitoring
        await self.memory_controller.stop_monitoring()
        
        # Stop web UI
        if self.web_ui:
            await self.web_ui.stop()
            
        self.state.is_running = False
        logger.info("JARVIS Event Coordinator stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status.
        
        Returns:
            Dict[str, Any]: Comprehensive status information including
                component states, workflow counts, and system metrics
                
        Example:
            >>> status = coordinator.get_status()
            >>> print(f"Running: {status['is_running']}")
            >>> print(f"Components: {status['components']}")
        """
        return {
            "is_running": self.state.is_running,
            "active_session": self.state.active_session,
            "current_user": self.state.current_user,
            "last_activity": self.state.last_activity.isoformat() if self.state.last_activity else None,
            "components": self.state.component_status,
            "active_workflows": len(self.active_workflows),
            "event_stats": self.event_bus.get_stats(),
            "memory_stats": self.memory_controller.get_memory_stats()
        }
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Dict[str, Any]: Detailed performance metrics including
                coordinator status, event metrics, and component performance
                
        Example:
            >>> report = coordinator.get_performance_report()
            >>> print(f"Event throughput: {report['event_metrics']['throughput']}")
        """
        metrics_report = self.metrics_collector.get_performance_report(duration_minutes=5)
        
        return {
            "coordinator_status": self.get_status(),
            "event_metrics": metrics_report,
            "component_metrics": {
                "voice": self.voice_system.get_performance_metrics() if self.voice_system else None,
                "memory": self.memory_controller.get_memory_stats()
            }
        }


async def main() -> None:
    """Main entry point for JARVIS with event-driven architecture.
    
    Sets up logging, creates the coordinator, and runs the main event loop
    with proper error handling and graceful shutdown.
    
    Environment Variables:
        ANTHROPIC_API_KEY: API key for voice processing (optional)
        JARVIS_USER: User name for personalization (default: "Sir")
        JARVIS_DEBUG: Enable debug mode if set to "true"
        
    Example:
        >>> # Set environment variables
        >>> os.environ["JARVIS_USER"] = "Alice"
        >>> os.environ["JARVIS_DEBUG"] = "true"