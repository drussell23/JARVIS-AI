#!/usr/bin/env python3
"""
Intelligent Workspace Automation for JARVIS.

This module provides autonomous workflow execution and cross-application automation
capabilities. It learns from user patterns, executes complex multi-step workflows,
and provides intelligent automation suggestions based on context and behavior analysis.

The system integrates with vision navigation, decision engines, and workspace monitoring
to create a comprehensive automation platform that can handle meeting preparation,
focus modes, development environments, and other common workflows.

Example:
    >>> from autonomy.workspace_automation import WorkspaceAutomation
    >>> from autonomy.vision_navigation_system import VisionNavigationSystem
    >>> 
    >>> nav_system = VisionNavigationSystem()
    >>> automation = WorkspaceAutomation(nav_system)
    >>> await automation.start_automation()
    >>> await automation.execute_workflow("meeting_prep")
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# Import navigation and vision systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomy.vision_navigation_system import (
    VisionNavigationSystem, NavigationAction, WorkspaceLayout,
    WorkspaceElement, WorkspaceMap
)
from autonomy.autonomous_decision_engine import (
    AutonomousDecisionEngine, AutonomousAction, ActionPriority
)
from vision.enhanced_monitoring import EnhancedWorkspaceMonitor

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Types of automated workflows.
    
    Defines the different categories of workflows that can be automated,
    each with specific characteristics and use cases.
    
    Attributes:
        MEETING_PREP: Workflow for preparing meeting environments
        FOCUS_MODE: Deep work mode with distraction elimination
        RESEARCH: Research-oriented workspace setup
        DEVELOPMENT: Development environment configuration
        COMMUNICATION: Communication-focused workflows
        BREAK_TIME: Break and rest period workflows
        END_OF_DAY: End-of-workday cleanup and preparation
        CUSTOM: User-defined custom workflows
    """
    MEETING_PREP = "meeting_prep"
    FOCUS_MODE = "focus_mode"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    COMMUNICATION = "communication"
    BREAK_TIME = "break_time"
    END_OF_DAY = "end_of_day"
    CUSTOM = "custom"

class AutomationTrigger(Enum):
    """Triggers for automation execution.
    
    Defines the different types of events or conditions that can
    trigger the execution of automated workflows.
    
    Attributes:
        TIME_BASED: Triggered at specific times or intervals
        EVENT_BASED: Triggered by system or application events
        CONTEXT_BASED: Triggered by workspace context changes
        USER_COMMAND: Triggered by explicit user commands
        PATTERN_DETECTED: Triggered by detected behavioral patterns
    """
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONTEXT_BASED = "context_based"
    USER_COMMAND = "user_command"
    PATTERN_DETECTED = "pattern_detected"

@dataclass
class WorkflowStep:
    """Single step in an automation workflow.
    
    Represents an individual action within a larger workflow, including
    execution parameters, conditions, and retry logic.
    
    Attributes:
        action: The action to execute (e.g., 'open_application')
        params: Parameters for the action execution
        condition: Optional condition function that must return True to execute
        timeout: Maximum time to wait for step completion in seconds
        retries: Number of retry attempts if step fails
        description: Human-readable description of the step
    """
    action: str
    params: Dict[str, Any]
    condition: Optional[Callable[[], bool]] = None
    timeout: float = 30.0
    retries: int = 3
    description: str = ""
    
    async def execute(self, context: Dict[str, Any]) -> bool:
        """Execute the workflow step with retry logic.
        
        Args:
            context: Execution context containing executor and other resources
            
        Returns:
            True if step executed successfully, False otherwise
            
        Raises:
            Exception: If critical execution errors occur
        """
        if self.condition and not self.condition():
            logger.info(f"Skipping step: {self.description} (condition not met)")
            return True
            
        # Execute with retries
        for attempt in range(self.retries):
            try:
                # Call the action with params
                if 'executor' in context:
                    result = await context['executor'](self.action, self.params)
                    if result:
                        return True
                        
            except Exception as e:
                logger.error(f"Step failed (attempt {attempt + 1}): {e}")
                
            if attempt < self.retries - 1:
                await asyncio.sleep(1)  # Wait before retry
                
        return False

@dataclass
class Workflow:
    """Complete automation workflow.
    
    Represents a complete multi-step automation workflow with metadata,
    execution tracking, and success rate monitoring.
    
    Attributes:
        id: Unique identifier for the workflow
        name: Human-readable name
        type: Category of workflow from WorkflowType enum
        steps: List of WorkflowStep objects to execute
        triggers: List of triggers that can activate this workflow
        priority: Execution priority level
        enabled: Whether the workflow is currently enabled
        last_executed: Timestamp of last execution
        execution_count: Total number of times executed
        success_rate: Rolling success rate (0.0 to 1.0)
    """
    id: str
    name: str
    type: WorkflowType
    steps: List[WorkflowStep]
    triggers: List[AutomationTrigger] = field(default_factory=list)
    priority: ActionPriority = ActionPriority.MEDIUM
    enabled: bool = True
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_rate: float = 1.0
    
    def add_step(self, action: str, params: Dict[str, Any], **kwargs):
        """Add a step to the workflow.
        
        Args:
            action: The action type to execute
            params: Parameters for the action
            **kwargs: Additional step configuration (timeout, retries, etc.)
        """
        step = WorkflowStep(action=action, params=params, **kwargs)
        self.steps.append(step)
        
    async def execute(self, context: Dict[str, Any]) -> bool:
        """Execute the complete workflow.
        
        Executes all steps in sequence, handling failures and updating
        execution statistics.
        
        Args:
            context: Execution context with resources and configuration
            
        Returns:
            True if all steps completed successfully, False otherwise
        """
        logger.info(f"Executing workflow: {self.name}")
        success_count = 0
        
        for i, step in enumerate(self.steps):
            logger.info(f"Step {i+1}/{len(self.steps)}: {step.description or step.action}")
            
            if await step.execute(context):
                success_count += 1
            else:
                logger.error(f"Workflow step failed: {step.action}")
                # Decide whether to continue or abort
                if step.action in ['critical', 'required']:
                    logger.error("Critical step failed, aborting workflow")
                    break
                    
        # Update statistics
        self.last_executed = datetime.now()
        self.execution_count += 1
        success_ratio = success_count / len(self.steps) if self.steps else 0
        self.success_rate = (self.success_rate * 0.9) + (success_ratio * 0.1)  # Weighted average
        
        return success_count == len(self.steps)

@dataclass
class AutomationPattern:
    """Detected pattern that can trigger automation.
    
    Represents a behavioral pattern detected from user actions that
    could be automated in the future.
    
    Attributes:
        pattern_type: Type/category of the detected pattern
        occurrences: List of timestamps when pattern occurred
        confidence: Confidence level of pattern detection (0.0 to 1.0)
        suggested_workflow: Optional workflow ID that could automate this pattern
    """
    pattern_type: str
    occurrences: List[datetime]
    confidence: float
    suggested_workflow: Optional[str] = None
    
    def is_recurring(self, threshold: int = 3) -> bool:
        """Check if pattern is recurring enough to be actionable.
        
        Args:
            threshold: Minimum number of occurrences to consider recurring
            
        Returns:
            True if pattern has occurred at least threshold times
        """
        return len(self.occurrences) >= threshold

class WorkspaceAutomation:
    """Intelligent workspace automation system.
    
    Main automation system that learns and executes complex workflows
    across applications. Integrates with vision navigation, decision
    engines, and workspace monitoring to provide comprehensive automation.
    
    The system can:
    - Execute predefined workflows (meeting prep, focus mode, etc.)
    - Learn from user behavior patterns
    - Suggest automation opportunities
    - Schedule workflows for future execution
    - Adapt to user preferences over time
    
    Attributes:
        navigation: Vision navigation system for UI interaction
        decision_engine: Autonomous decision making engine
        monitor: Enhanced workspace monitoring system
        workflows: Dictionary of available workflows by ID
        active_workflows: List of currently executing workflow IDs
        workflow_history: History of workflow executions
        detected_patterns: List of detected behavioral patterns
        user_actions: History of user actions for pattern detection
        automation_enabled: Whether automation is currently active
        learning_mode: Whether system is learning from user actions
    """
    
    def __init__(self, navigation_system: VisionNavigationSystem):
        """Initialize the workspace automation system.
        
        Args:
            navigation_system: Vision navigation system for UI interaction
        """
        self.navigation = navigation_system
        self.decision_engine = AutonomousDecisionEngine()
        self.monitor = EnhancedWorkspaceMonitor()
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: List[str] = []
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Pattern detection
        self.detected_patterns: List[AutomationPattern] = []
        self.user_actions: List[Dict[str, Any]] = []
        self.pattern_threshold = 3  # Minimum occurrences to detect pattern
        
        # Automation state
        self.automation_enabled = True
        self.learning_mode = True
        self.execution_context: Dict[str, Any] = {}
        
        # Initialize predefined workflows
        self._initialize_workflows()
        
        # Schedule management
        self.scheduled_tasks: List[Dict[str, Any]] = []
        self.schedule_task = None
        
    def _initialize_workflows(self):
        """Initialize predefined intelligent workflows.
        
        Creates standard workflows for common automation scenarios including
        meeting preparation, focus mode, research setup, development environment,
        and end-of-day routines.
        """
        
        # Meeting preparation workflow
        meeting_prep = Workflow(
            id="meeting_prep",
            name="Prepare for Meeting",
            type=WorkflowType.MEETING_PREP,
            steps=[],
            triggers=[AutomationTrigger.TIME_BASED, AutomationTrigger.EVENT_BASED]
        )
        
        meeting_prep.add_step("analyze_calendar", {}, 
                             description="Check calendar for upcoming meeting")
        meeting_prep.add_step("close_distracting_apps", {
            "keep_apps": ["Zoom", "Calendar", "Notes"]
        }, description="Close distracting applications")
        meeting_prep.add_step("open_application", {
            "app_name": "Zoom"
        }, description="Open video conferencing app")
        meeting_prep.add_step("arrange_windows", {
            "layout": WorkspaceLayout.SPLIT,
            "apps": ["Zoom", "Notes"]
        }, description="Arrange windows for meeting")
        meeting_prep.add_step("mute_notifications", {
            "duration_minutes": 60
        }, description="Mute notifications during meeting")
        meeting_prep.add_step("prepare_notes", {
            "template": "meeting_notes"
        }, description="Create meeting notes document")
        
        self.workflows["meeting_prep"] = meeting_prep
        
        # Focus mode workflow
        focus_mode = Workflow(
            id="focus_mode",
            name="Deep Focus Mode",
            type=WorkflowType.FOCUS_MODE,
            steps=[],
            priority=ActionPriority.HIGH
        )
        
        focus_mode.add_step("analyze_current_task", {},
                           description="Understand current work context")
        focus_mode.add_step("close_all_except_current", {},
                           description="Close all non-essential windows")
        focus_mode.add_step("maximize_current_window", {},
                           description="Maximize working window")
        focus_mode.add_step("block_distracting_sites", {
            "sites": ["social_media", "news", "entertainment"]
        }, description="Block distracting websites")
        focus_mode.add_step("set_status", {
            "status": "Do Not Disturb",
            "duration_minutes": 90
        }, description="Set DND status everywhere")
        focus_mode.add_step("start_focus_timer", {
            "duration_minutes": 25,
            "break_minutes": 5
        }, description="Start Pomodoro timer")
        
        self.workflows["focus_mode"] = focus_mode
        
        # Research workflow
        research_flow = Workflow(
            id="research_flow",
            name="Research Setup",
            type=WorkflowType.RESEARCH,
            steps=[]
        )
        
        research_flow.add_step("open_browser", {
            "app_name": "Safari"
        }, description="Open web browser")
        research_flow.add_step("open_notes", {
            "app_name": "Notes",
            "create_new": True
        }, description="Open note-taking app")
        research_flow.add_step("arrange_windows", {
            "layout": WorkspaceLayout.SPLIT,
            "apps": ["Safari", "Notes"]
        }, description="Arrange for research")
        research_flow.add_step("create_research_template", {
            "template": "research_notes"
        }, description="Create research template")
        research_flow.add_step("enable_reader_mode", {},
                              description="Enable reader mode in browser")
        
        self.workflows["research_flow"] = research_flow
        
        # Development workflow
        dev_flow = Workflow(
            id="dev_flow",
            name="Development Environment",
            type=WorkflowType.DEVELOPMENT,
            steps=[]
        )
        
        dev_flow.add_step("open_ide", {
            "app_name": "Visual Studio Code"
        }, description="Open development IDE")
        dev_flow.add_step("open_terminal", {
            "app_name": "Terminal"
        }, description="Open terminal")
        dev_flow.add_step("open_browser", {
            "app_name": "Safari",
            "url": "localhost:8000"
        }, description="Open browser for testing")
        dev_flow.add_step("arrange_windows", {
            "layout": "custom_dev",
            "arrangement": {
                "ide": {"x": 0, "y": 0, "width": 0.6, "height": 1.0},
                "browser": {"x": 0.6, "y": 0, "width": 0.4, "height": 0.5},
                "terminal": {"x": 0.6, "y": 0.5, "width": 0.4, "height": 0.5}
            }
        }, description="Arrange development workspace")
        
        self.workflows["dev_flow"] = dev_flow
        
        # End of day workflow
        eod_flow = Workflow(
            id="end_of_day",
            name="End of Day Routine",
            type=WorkflowType.END_OF_DAY,
            steps=[]
        )
        
        eod_flow.add_step("save_all_work", {},
                         description="Save all open documents")
        eod_flow.add_step("create_tomorrow_plan", {
            "app": "Notes",
            "template": "daily_plan"
        }, description="Create tomorrow's plan")
        eod_flow.add_step("close_work_apps", {
            "exclude": ["personal_apps"]
        }, description="Close work applications")
        eod_flow.add_step("clear_downloads", {},
                         description="Clean up downloads folder")
        eod_flow.add_step("backup_important_files", {},
                         description="Backup today's work")
        
        self.workflows["end_of_day"] = eod_flow
    
    async def start_automation(self):
        """Start the automation system.
        
        Initializes all automation components including navigation,
        pattern detection, and schedule monitoring.
        
        Raises:
            Exception: If navigation system fails to start
        """
        self.automation_enabled = True
        
        # Start navigation system
        await self.navigation.start_navigation_mode()
        
        # Start pattern detection
        asyncio.create_task(self._pattern_detection_loop())
        
        # Start schedule monitoring
        self.schedule_task = asyncio.create_task(self._schedule_monitor_loop())
        
        logger.info("Workspace automation started")
        
    async def stop_automation(self):
        """Stop the automation system.
        
        Cleanly shuts down all automation components and cancels
        any running tasks.
        """
        self.automation_enabled = False
        
        # Stop navigation
        await self.navigation.stop_navigation_mode()
        
        # Cancel scheduled tasks
        if self.schedule_task:
            self.schedule_task.cancel()
            
        logger.info("Workspace automation stopped")
        
    async def execute_workflow(self, workflow_id: str, 
                             params: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a specific workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            params: Optional parameters to pass to the workflow
            
        Returns:
            True if workflow completed successfully, False otherwise
            
        Raises:
            ValueError: If workflow_id is not found
        """
        if workflow_id not in self.workflows:
            logger.error(f"Unknown workflow: {workflow_id}")
            return False
            
        workflow = self.workflows[workflow_id]
        
        if not workflow.enabled:
            logger.warning(f"Workflow {workflow_id} is disabled")
            return False
            
        # Check if already running
        if workflow_id in self.active_workflows:
            logger.warning(f"Workflow {workflow_id} is already running")
            return False
            
        try:
            self.active_workflows.append(workflow_id)
            
            # Prepare execution context
            context = {
                'executor': self._execute_action,
                'navigation': self.navigation,
                'monitor': self.monitor,
                'params': params or {},
                'workspace_map': self.navigation.current_map
            }
            
            # Execute workflow
            success = await workflow.execute(context)
            
            # Record execution
            self._record_workflow_execution(workflow_id, success)
            
            return success
            
        finally:
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
                
    async def _execute_action(self, action: str, params: Dict[str, Any]) -> bool:
        """Execute a single automation action.
        
        Internal method that handles the execution of individual workflow
        actions by routing them to appropriate handlers.
        
        Args:
            action: The action type to execute
            params: Parameters for the action
            
        Returns:
            True if action completed successfully, False otherwise
            
        Raises:
            Exception: If action execution encounters critical errors
        """
        try:
            # Navigation actions
            if action == "open_application":
                return await self.navigation.navigate_to_application(
                    params.get('app_name')
                )
                
            elif action == "close_application":
                return await self._close_application(params.get('app_name'))
                
            elif action == "arrange_windows":
                layout = params.get('layout', WorkspaceLayout.SPLIT)
                return await self.navigation.arrange_workspace(layout)
                
            elif action == "switch_to_window":
                window_id = params.get('window_id')
                return await self.navigation._switch_to_window(window_id)
                
            # Workspace actions
            elif action == "close_all_except_current":
                return await self._close_all_except_current()
                
            elif action == "close_distracting_apps":
                return await self._close_distracting_apps(
                    params.get('keep_apps', [])
                )
                
            elif action == "maximize_current_window":
                return await self._maximize_current_window()
                
            # System actions
            elif action == "mute_notifications":
                return await self._mute_notifications(
                    params.get('duration_minutes', 60)
                )
                
            elif action == "set_status":
                return await self._set_system_status(
                    params.get('status'),
                    params.get('duration_minutes')
                )
                
            # Content actions
            elif action == "create_document":
                return await self._create_document(
                    params.get('app'),
                    params.get('template')
                )
                
            elif action == "save_all_work":
                return await self._save_all_work()
                
            # Analysis actions
            elif action == "analyze_current_task":
                return await self._analyze_current_task()
                
            elif action == "analyze_calendar":
                return await self._analyze_calendar()
                
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return False
            
    async def _close_application(self, app_name: str) -> bool:
        """Close a specific application.
        
        Args:
            app_name: Name of the application to close
            
        Returns:
            True if application was closed successfully
        """
        # Find windows for the app
        if not self.navigation.current_map:
            return False
            
        app_windows = [
            w for w in self.navigation.current_map.windows
            if w.app_name.lower() == app_name.lower()
        ]
        
        for window in app_windows:
            await self.navigation._close_window(window.window_id)
            
        return True
        
    async def _close_all_except_current(self) -> bool:
        """Close all windows except the currently focused one.
        
        Returns:
            True if windows were closed successfully
        """
        if not self.navigation.current_map:
            return False
            
        active_window = self.navigation.current_map.active_window
        if not active_window:
            return False
            
        for window in self.navigation.current_map.windows:
            if window.window_id != active_window.window_id:
                await self.navigation._close_window(window.window_id)
                
        return True
        
    async def _close_distracting_apps(self, keep_apps: List[str]) -> bool:
        """Close potentially distracting applications.
        
        Args:
            keep_apps: List of app names to keep open
            
        Returns:
            True if distracting apps were closed successfully
        """
        distracting = [
            "Discord", "Slack", "Twitter", "Facebook", 
            "Instagram", "YouTube", "Netflix", "Spotify"
        ]
        
        if not self.navigation.current_map:
            return False
            
        for window in self.navigation.current_map.windows:
            app_name = window.app_name
            
            # Check if it's a distracting app
            if any(d.lower() in app_name.lower() for d in distracting):
                # Check if it should be kept
                if not any(k.lower() in app_name.lower() for k in keep_apps):
                    await self.navigation._close_window(window.window_id)
                    
        return True
        
    async def _maximize_current_window(self) -> bool:
        """Maximize the currently active window.
        
        Returns:
            True if window was maximized successfully
        """
        if not self.navigation.current_map:
            return False
            
        active_window = self.navigation.current_map.active_window
        if active_window:
            return await self.navigation._maximize_window(active_window.window_id)
            
        return False
        
    async def _mute_notifications(self, duration_minutes: int) -> bool:
        """Mute system notifications for specified duration.
        
        Args:
            duration_minutes: How long to mute notifications
            
        Returns:
            True if notifications were muted successfully
        """
        # This would integrate with macOS notification center
        logger.info(f"Muting notifications for {duration_minutes} minutes")
        
        # Store unmute time
        unmute_time = datetime.now() + timedelta(minutes=duration_minutes)
        self.execution_context['unmute_time'] = unmute_time
        
        return True
        
    async def _set_system_status(self, status: str, duration_minutes: int) -> bool:
        """Set system-wide status (DND, Away, etc.).
        
        Args:
            status: Status message to set
            duration_minutes: How long to maintain the status
            
        Returns:
            True if status was set successfully
        """
        logger.info(f"Setting system status to {status} for {duration_minutes} minutes")
        
        # This would integrate with various apps to set status
        # For now, we'll track it internally
        self.execution_context['system_status'] = {
            'status': status,
            'until': datetime.now() + timedelta(minutes=duration_minutes)
        }
        
        return True
        
    async def _create_document(self, app_name: str, template: str) -> bool:
        """Create a new document with specified template.
        
        Args:
            app_name: Application to create document in
            template: Template type to use
            
        Returns:
            True if document was created successfully
        """
        # Open the app
        if not await self.navigation.navigate_to_application(app_name):
            return False
            
        # Create new document (app-specific)
        # This would use keyboard shortcuts or menu navigation
        logger.info(f"Creating {template} document in {app_name}")
        
        return True
        
    async def _save_all_work(self) -> bool:
        """Save all open documents.
        
        Returns:
            True if all documents were saved successfully
        """
        # This would iterate through windows and trigger save
        logger.info("Saving all open work")
        
        # Use keyboard shortcut Cmd+S on each window
        if self.navigation.current_map:
            for window in self.navigation.current_map.windows:
                await self.navigation._switch_to_window(window.window_id)
                # Send Cmd+S
                await asyncio.sleep(0.5)
                
        return True
        
    async def _analyze_current_task(self) -> bool:
        """Analyze what the user is currently working on.
        
        Returns:
            True if analysis completed successfully
        """
        if not self.navigation.current_map:
            return False
            
        # Use vision system to understand current context
        workspace_state = await self.monitor.get_complete_workspace_state()
        
        # Extract task context
        self.execution_context['current_task'] = {
            'active_app': self.navigation.current_map.active_window.app_name
            if self.navigation.current_map.active_window else None,
            'open_apps': [w.app_name for w in self.navigation.current_map.windows],
            'workspace_context': workspace_state.get('analysis', {})
        }
        
        return True
        
    async def _analyze_calendar(self) -> bool:
        """Analyze calendar for upcoming events.
        
        Returns:
            True if calendar analysis completed successfully
        """
        # This would integrate with calendar app
        logger.info("Analyzing calendar for upcoming events")
        
        # For now, simulate finding a meeting
        self.execution_context['upcoming_meeting'] = {
            'time': datetime.now() + timedelta(minutes=5),
            'title': 'Team Standup',
            'requires_video': True
        }
        
        return True
        
    async def learn_from_user_actions(self, action: Dict[str, Any]):
        """Learn patterns from user actions.
        
        Records user actions and analyzes them for automation patterns.
        
        Args:
            action: Dictionary containing action details (type, app, etc.)
        """
        if not self.learning_mode:
            return
            
        # Record action
        action['timestamp'] = datetime.now()
        self.user_actions.append(action)
        
        # Keep history manageable
        if len(self.user_actions) > 1000:
            self.user_actions = self.user_actions[-1000:]
            
        # Detect patterns
        await self._detect_patterns()
        
    async def _detect_patterns(self):
        """Detect patterns in user actions.
        
        Analyzes recorded user actions to identify recurring patterns
        that could be automated.
        """
        # Group actions by type and time
        action_groups = {}
        
        for action in self.user_actions:
            key = f"{action.get('type')}_{action.get('app')}"
            if key not in action_groups:
                action_groups[key] = []
            action_groups[key].append(action['timestamp'])
            
        # Look for recurring patterns
        for key, timestamps in action_groups.items():
            if len(timestamps) >= self.pattern_threshold:
                # Check if actions happen at similar times
                hours = [ts.hour for ts in timestamps]
                
                # If most actions happen in same hour
                most_common_hour = max(set(hours), key=hours.count)
                hour_frequency = hours.count(most_common_hour) / len(hours)
                
                if hour_frequency > 0.6:  # 60% happen in same hour
                    pattern = AutomationPattern(
                        pattern_type=f"daily_{key}",
                        occurrences=timestamps,
                        confidence=hour_frequency,
                        suggested_workflow=self._suggest_workflow_for_pattern(key)
                    )
                    
                    if pattern not in self.detected_patterns:
                        self.detected_patterns.append(pattern)
                        logger.info(f"Detected pattern: {pattern.pattern_type}")
                        
    def _suggest_workflow_for_pattern(self, pattern_key: str) -> Optional[str]:
        """Suggest a workflow based on detected pattern.
        
        Args:
            pattern_key: Key identifying the pattern type
            
        Returns:
            Workflow ID that could automate this pattern, or None
        """
        if