"""
Multi-Space Context Graph - Advanced Context Tracking for JARVIS
=================================================================

This is the foundational system for JARVIS's "workspace intelligence":
- Tracks activity across all macOS Spaces simultaneously
- Preserves temporal context (what happened 3-5 minutes ago)
- Correlates activities across spaces (terminal error + browser research + IDE edits)
- Enables "what does it say?" natural language queries
- No hardcoding - fully dynamic and adaptive

Architecture:
    MultiSpaceContextGraph (Coordinator)
    ├── SpaceContext (Per-space tracking)
    │   ├── ApplicationContext (Per-app state)
    │   │   ├── TerminalContext
    │   │   ├── BrowserContext
    │   │   ├── IDEContext
    │   │   └── GenericAppContext
    │   └── ActivityTimeline (Temporal events)
    ├── CrossSpaceCorrelator (Relationship detection)
    ├── ContextQueryEngine (Natural language queries)
    └── TemporalDecayManager (3-5 minute TTL with smart decay)

Integration Points:
    - MultiSpaceMonitor (vision/multi_space_monitor.py)
    - ContextStore (core/context/memory_store.py)
    - TemporalContextEngine (vision/intelligence/temporal_context_engine.py)
    - FeedbackLearningLoop (core/learning/feedback_loop.py)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# CONTEXT TYPES - Rich, Structured Context for Different Applications
# ============================================================================


class ContextType(Enum):
    """Types of application contexts we track.

    This enum defines the different categories of applications that JARVIS
    can provide specialized context tracking for.
    """

    TERMINAL = "terminal"
    BROWSER = "browser"
    IDE = "ide"
    EDITOR = "editor"
    COMMUNICATION = "communication"
    GENERIC = "generic"


class ActivitySignificance(Enum):
    """How significant is this activity?

    Used to prioritize context and determine what should be preserved
    longer or highlighted in natural language queries.
    """

    CRITICAL = "critical"  # Errors, crashes, important notifications
    HIGH = "high"  # Code changes, command execution, search queries
    NORMAL = "normal"  # Regular interactions, scrolling, reading
    LOW = "low"  # Idle, background activity
    BACKGROUND = "background"  # No user interaction


@dataclass
class TerminalContext:
    """Context for terminal/command-line applications.

    Tracks command execution, output, errors, and working directory
    to understand development workflows and debugging sessions.

    Attributes:
        last_command: Most recently executed command
        last_output: Output from the last command
        errors: List of error messages detected
        warnings: List of warning messages detected
        exit_code: Exit code of the last command
        working_directory: Current working directory
        shell_type: Type of shell (bash, zsh, fish, etc.)
        recent_commands: Deque of recent commands with timestamps
    """

    last_command: Optional[str] = None
    last_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    working_directory: Optional[str] = None
    shell_type: str = "unknown"  # bash, zsh, fish, etc.
    recent_commands: deque = field(default_factory=lambda: deque(maxlen=10))

    def has_error(self) -> bool:
        """Check if this terminal context contains any errors.

        Returns:
            bool: True if there are errors or non-zero exit code
        """
        return len(self.errors) > 0 or (self.exit_code is not None and self.exit_code != 0)


@dataclass
class BrowserContext:
    """Context for web browsers.

    Tracks browsing activity, research patterns, and content to understand
    what the user is reading or researching.

    Attributes:
        active_url: Currently active URL
        page_title: Title of the current page
        tabs: List of open tabs with URLs and titles
        search_query: Most recent search query
        reading_content: OCR extracted text from the page
        is_researching: Whether user appears to be researching
        research_topic: Detected topic of research
    """

    active_url: Optional[str] = None
    page_title: Optional[str] = None
    tabs: List[Dict[str, str]] = field(default_factory=list)  # [{"url": ..., "title": ...}]
    search_query: Optional[str] = None
    reading_content: Optional[str] = None  # OCR extracted text
    is_researching: bool = False  # Detected research behavior
    research_topic: Optional[str] = None


@dataclass
class IDEContext:
    """Context for IDEs (VS Code, IntelliJ, etc.).

    Tracks code editing activity, open files, errors, and debugging
    to understand development workflows.

    Attributes:
        open_files: List of currently open file paths
        active_file: Currently active/focused file
        cursor_position: Current cursor position (line, column)
        recent_edits: List of recent edit operations
        errors_in_file: List of errors in the current file
        warnings_in_file: List of warnings in the current file
        is_debugging: Whether debugger is active
        language: Programming language of active file
    """

    open_files: List[str] = field(default_factory=list)
    active_file: Optional[str] = None
    cursor_position: Optional[Tuple[int, int]] = None  # (line, column)
    recent_edits: List[Dict[str, Any]] = field(default_factory=list)
    errors_in_file: List[str] = field(default_factory=list)
    warnings_in_file: List[str] = field(default_factory=list)
    is_debugging: bool = False
    language: Optional[str] = None


@dataclass
class GenericAppContext:
    """Generic context for applications we don't have special handling for.

    Provides basic context tracking for any application through OCR
    and interaction monitoring.

    Attributes:
        window_title: Title of the application window
        extracted_text: OCR extracted text from the window
        interaction_count: Number of interactions detected
        last_interaction: Timestamp of last interaction
    """

    window_title: Optional[str] = None
    extracted_text: Optional[str] = None  # OCR
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None


# ============================================================================
# APPLICATION CONTEXT - State Tracking for Individual Applications
# ============================================================================


@dataclass
class ApplicationContext:
    """Tracks state for a specific application within a space.

    This is the main container for application-specific context, containing
    both metadata and type-specific context objects.

    Attributes:
        app_name: Name of the application
        context_type: Type of context (terminal, browser, etc.)
        space_id: ID of the space this app is in
        window_id: Optional window ID for the application
        terminal_context: Terminal-specific context if applicable
        browser_context: Browser-specific context if applicable
        ide_context: IDE-specific context if applicable
        generic_context: Generic context for unknown app types
        first_seen: When this app was first detected
        last_activity: When this app last had activity
        activity_count: Total number of activities recorded
        significance: Current significance level of this app
        screenshots: Deque of recent screenshot references
    """

    app_name: str
    context_type: ContextType
    space_id: int
    window_id: Optional[int] = None

    # Type-specific context (only one will be populated)
    terminal_context: Optional[TerminalContext] = None
    browser_context: Optional[BrowserContext] = None
    ide_context: Optional[IDEContext] = None
    generic_context: Optional[GenericAppContext] = None

    # Metadata
    first_seen: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    activity_count: int = 0
    significance: ActivitySignificance = ActivitySignificance.NORMAL

    # Screenshot references
    screenshots: deque = field(default_factory=lambda: deque(maxlen=5))

    def get_typed_context(
        self,
    ) -> Union[TerminalContext, BrowserContext, IDEContext, GenericAppContext]:
        """Get the specific context object for this app type.

        Returns:
            Union[TerminalContext, BrowserContext, IDEContext, GenericAppContext]:
                The appropriate context object for this application type
        """
        if self.context_type == ContextType.TERMINAL:
            return self.terminal_context or TerminalContext()
        elif self.context_type == ContextType.BROWSER:
            return self.browser_context or BrowserContext()
        elif self.context_type == ContextType.IDE:
            return self.ide_context or IDEContext()
        else:
            return self.generic_context or GenericAppContext()

    def update_activity(self, significance: Optional[ActivitySignificance] = None):
        """Record activity in this application.

        Args:
            significance: Optional significance level for this activity
        """
        self.last_activity = datetime.now()
        self.activity_count += 1
        if significance:
            self.significance = significance

    def add_screenshot(self, screenshot_path: str, ocr_text: Optional[str] = None):
        """Add screenshot reference with optional OCR text.

        Args:
            screenshot_path: Path to the screenshot file
            ocr_text: Optional OCR extracted text from the screenshot
        """
        self.screenshots.append(
            {"path": screenshot_path, "timestamp": datetime.now(), "ocr_text": ocr_text}
        )

    def is_recent(self, within_seconds: int = 180) -> bool:
        """Check if this app had recent activity.

        Args:
            within_seconds: Time window to consider as "recent" (default: 3 minutes)

        Returns:
            bool: True if the app had activity within the specified time window
        """
        return (datetime.now() - self.last_activity).total_seconds() <= within_seconds


# ============================================================================
# SPACE CONTEXT - Per-Space Activity Tracking
# ============================================================================


@dataclass
class ActivityEvent:
    """Individual activity event within a space.

    Represents a single event in the activity timeline, such as app launches,
    command executions, errors, etc.

    Attributes:
        event_type: Type of event (e.g., "app_launched", "command_executed")
        timestamp: When the event occurred
        app_name: Name of the application involved (if applicable)
        significance: Significance level of this event
        details: Additional event-specific details
    """

    event_type: str  # "app_launched", "command_executed", "error_detected", etc.
    timestamp: datetime
    app_name: Optional[str] = None
    significance: ActivitySignificance = ActivitySignificance.NORMAL
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the event
        """
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "app_name": self.app_name,
            "significance": self.significance.value,
            "details": self.details,
        }


class SpaceContext:
    """Tracks all activity within a single macOS Space/Desktop.

    This is the per-space view that gets aggregated into the global context graph.
    Each space maintains its own timeline of events and application contexts.

    Attributes:
        space_id: Unique identifier for this space
        created_at: When this space context was created
        last_activity: When this space last had activity
        applications: Dictionary of application contexts in this space
        activity_timeline: Deque of recent activity events
        is_active: Whether this space is currently active
        visit_count: Number of times this space has been activated
        total_time_active: Total time spent in this space
        tags: Set of tags for categorizing this space
    """

    def __init__(self, space_id: int):
        """Initialize space context.

        Args:
            space_id: Unique identifier for this space
        """
        self.space_id = space_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Application contexts within this space
        self.applications: Dict[str, ApplicationContext] = {}

        # Activity timeline (last 5 minutes, then decay)
        self.activity_timeline: deque = deque(maxlen=100)

        # Space metadata
        self.is_active = False  # Is this the current space?
        self.visit_count = 0
        self.total_time_active = timedelta()
        self._last_activated = None

        # Tags for categorization
        self.tags: Set[str] = set()  # e.g., "development", "research", "communication"

    def activate(self):
        """Mark this space as currently active.

        Records activation time and adds an activation event to the timeline.
        """
        self.is_active = True
        self.visit_count += 1
        self._last_activated = datetime.now()
        self.last_activity = datetime.now()

        self.add_event(
            ActivityEvent(
                event_type="space_activated",
                timestamp=datetime.now(),
                significance=ActivitySignificance.NORMAL,
            )
        )

    def deactivate(self):
        """Mark this space as no longer active.

        Calculates session duration and adds a deactivation event.
        """
        if self.is_active and self._last_activated:
            session_duration = datetime.now() - self._last_activated
            self.total_time_active += session_duration

        self.is_active = False

        self.add_event(
            ActivityEvent(
                event_type="space_deactivated",
                timestamp=datetime.now(),
                significance=ActivitySignificance.LOW,
            )
        )

    def add_application(self, app_name: str, context_type: ContextType) -> ApplicationContext:
        """Add or retrieve application context.

        Args:
            app_name: Name of the application
            context_type: Type of context for this application

        Returns:
            ApplicationContext: The application context object
        """
        if app_name not in self.applications:
            self.applications[app_name] = ApplicationContext(
                app_name=app_name, context_type=context_type, space_id=self.space_id
            )

            self.add_event(
                ActivityEvent(
                    event_type="app_added",
                    timestamp=datetime.now(),
                    app_name=app_name,
                    significance=ActivitySignificance.NORMAL,
                    details={"context_type": context_type.value},
                )
            )

        return self.applications[app_name]

    def remove_application(self, app_name: str):
        """Remove application from this space.

        Args:
            app_name: Name of the application to remove
        """
        if app_name in self.applications:
            del self.applications[app_name]

            self.add_event(
                ActivityEvent(
                    event_type="app_removed",
                    timestamp=datetime.now(),
                    app_name=app_name,
                    significance=ActivitySignificance.LOW,
                )
            )

    def add_event(self, event: ActivityEvent):
        """Add activity event to timeline.

        Args:
            event: The activity event to add
        """
        self.activity_timeline.append(event)
        self.last_activity = event.timestamp

    def get_recent_events(self, within_seconds: int = 180) -> List[ActivityEvent]:
        """Get events from the last N seconds.

        Args:
            within_seconds: Time window in seconds (default: 3 minutes)

        Returns:
            List[ActivityEvent]: List of recent events
        """
        cutoff = datetime.now() - timedelta(seconds=within_seconds)
        return [event for event in self.activity_timeline if event.timestamp > cutoff]

    def get_recent_errors(self, within_seconds: int = 300) -> List[Tuple[str, ActivityEvent]]:
        """Get recent errors from any application in this space.

        Args:
            within_seconds: Time window in seconds (default: 5 minutes)

        Returns:
            List[Tuple[str, ActivityEvent]]: List of (app_name, error_event) tuples
        """
        errors = []
        cutoff = datetime.now() - timedelta(seconds=within_seconds)

        # Check terminal errors
        for app_name, app_ctx in self.applications.items():
            if app_ctx.context_type == ContextType.TERMINAL and app_ctx.terminal_context:
                if app_ctx.last_activity > cutoff and app_ctx.terminal_context.has_error():
                    for error in app_ctx.terminal_context.errors:
                        errors.append(
                            (
                                app_name,
                                ActivityEvent(
                                    event_type="terminal_error",
                                    timestamp=app_ctx.last_activity,
                                    app_name=app_name,
                                    significance=ActivitySignificance.CRITICAL,
                                    details={"error": error},
                                ),
                            )
                        )

        return errors

    def infer_tags(self):
        """Automatically infer tags based on applications present.

        Analyzes the applications in this space to automatically categorize
        the space with relevant tags like "development", "research", etc.
        """
        new_tags = set()

        # Development indicators
        dev_apps = {
            "Terminal",
            "iTerm",
            "iTerm2",
            "VS Code",
            "Code",
            "IntelliJ",
            "PyCharm",
            "Sublime",
        }
        if any(app in self.applications for app in dev_apps):
            new_tags.add("development")

        # Research indicators
        browsers = {"Safari", "Chrome", "Firefox", "Arc", "Brave"}
        if any(app in self.applications for app in browsers):
            new_tags.add("research")

        # Communication indicators
        comm_apps = {"Slack", "Discord", "Zoom", "Microsoft Teams", "Messages"}
        if any(app in self.applications for app in comm_apps):
            new_tags.add("communication")

        self.tags.update(new_tags)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the space context
        """
        return {
            "space_id": self.space_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "visit_count": self.visit_count,
            "total_time_active_seconds": self.total_time_active.total_seconds(),
            "tags": list(self.tags),
            "applications": {
                name: {
                    "app_name": ctx.app_name,
                    "context_type": ctx.context_type.value,
                    "last_activity": ctx.last_activity.isoformat(),
                    "activity_count": ctx.activity_count,
                    "significance": ctx.significance.value,
                }
                for name, ctx in self.applications.items()
            },
            "recent_events": [event.to_dict() for event in self.get_recent_events()],
        }


# ============================================================================
# CROSS-SPACE CORRELATION - Detecting Relationships Across Spaces
# ============================================================================


@dataclass
class CrossSpaceRelationship:
    """Represents a detected relationship between activities in different spaces.

    This captures patterns like debugging workflows that span multiple spaces,
    research activities that inform coding, etc.

    Attributes:
        relationship_id: Unique identifier for this relationship
        relationship_type: Type of relationship (e.g., "debugging_workflow")
        involved_spaces: List of space IDs involved in this relationship
        involved_apps: List of (space_id, app_name) tuples
        confidence: Confidence score for this relationship (0.0-1.0)
        first_detected: When this relationship was first detected
        last_detected: When this relationship was last seen
        evidence: List of evidence supporting this relationship
        description: Human-readable description of the relationship
    """

    relationship_id: str
    relationship_type: str  # "debugging_workflow", "research_and_code", "cross_reference"
    involved_spaces: List[int]
    involved_apps: List[Tuple[int, str]]  # [(space_id, app_name), ...]
    confidence: float
    first_detected: datetime
    last_detected: datetime
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""

    def update_detection(self, new_evidence: Dict[str, Any]):
        """Update when we see more evidence of this relationship.

        Args:
            new_evidence: New evidence supporting this relationship
        """
        self.last_detected = datetime.now()
        self.evidence.append(new_evidence)
        # Increase confidence with more evidence (max 1.0)
        self.confidence = min(1.0, self.confidence + 0.1)


class CrossSpaceCorrelator:
    """Detects relationships and patterns across different spaces.

    This is the "intelligence" that understands:
    - Terminal error in Space 1 + Researching docs in Space 3 = Same problem
    - Editing code in Space 2 + Running tests in Space 1 = Development workflow
    - Reading Slack in Space 4 + Editing doc in Space 2 = Responding to request

    Attributes:
        relationships: Dictionary of detected relationships
        relationship_patterns: List of pattern detection functions
    """

    def __init__(self):
        """Initialize the cross-space correlator."""
        self.relationships: Dict[str, CrossSpaceRelationship] = {}
        self.relationship_patterns = self._init_patterns()

    def _init_patterns(self) -> List[Dict[str, Any]]:
        """Initialize relationship detection patterns.

        Returns:
            List[Dict[str, Any]]: List of pattern detection configurations
        """
        return [
            {
                "name": "debugging_workflow",
                "description": "Terminal error + browser research + code editing",
                "detector": self._detect_debugging_workflow,
                "min_confidence": 0.7,
            },
            {
                "name": "research_and_code",
                "description": "Reading documentation while coding",
                "detector": self._detect_research_and_code,
                "min_confidence": 0.6,
            },
            {
                "name": "cross_terminal_workflow",
                "description": "Multiple terminals working on related tasks",
                "detector": self._detect_cross_terminal_workflow,
                "min_confidence": 0.5,
            },
            {
                "name": "documentation_lookup",
                "description": "Quickly checking docs then returning to work",
                "detector": self._detect_documentation_lookup,
                "min_confidence": 0.8,
            },
        ]

    async def analyze_relationships(
        self, spaces: Dict[int, SpaceContext]
    ) -> List[CrossSpaceRelationship]:
        """Analyze all spaces and detect cross-space relationships.

        Args:
            spaces: Dictionary of space contexts to analyze

        Returns:
            List[CrossSpaceRelationship]: New or updated relationships detected
        """
        detected = []

        for pattern in self.relationship_patterns:
            try:
                result = await pattern["detector"](spaces)
                if result and result.confidence >= pattern["min_confidence"]:
                    # Check if this relationship already exists
                    if result.relationship_id in self.relationships:
                        existing = self.relationships[result.relationship_id]
                        existing.update_detection({"pattern": pattern["name"]})
                        detected.append(existing)
                    else:
                        self.relationships[result.relationship_id] = result
                        detected.append(result)
                        logger.info(
                            f"[CROSS-SPACE] Detected new relationship: {result.relationship_type}"
                        )
            except Exception as e:
                logger.error(f"[CROSS-SPACE] Error in pattern '{pattern['name']}': {e}")

        return detected

    async def _detect_debugging_workflow(
        self, spaces: Dict[int, SpaceContext]
    ) -> Optional[CrossSpaceRelationship]:
        """Detect: Terminal error + Browser research + IDE editing.

        This is a very common developer workflow where an error occurs,
        the developer researches solutions, and then implements fixes.

        Args:
            spaces: Dictionary of space contexts to analyze

        Returns:
            Optional[CrossSpaceRelationship]: Detected relationship or None
        """
        terminal_space = None
        terminal_error = None
        browser_space = None
        browser_research = None
        ide_space = None

        # Find terminal with recent error
        for space in spaces.values():
            errors = space.get_recent_errors(within_seconds=300)  # 5 minutes
            if errors:
                for app_name, event in errors:
                    terminal_space = space.space_id
                    terminal_error = event
                    break
                if terminal_error:
                    break

        if not terminal_error:
            return None

        # Find browser with research activity
        for space in spaces.values():
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.BROWSER and app_ctx.is_recent(
                    within_seconds=300
                ):
                    if app_ctx.browser_context and app_ctx.browser_context.is_researching:
                        browser_space = space.space_id
                        browser_research = app_ctx
                        break
            if browser_research:
                break

        # Find IDE with recent activity
        for space in spaces.values():
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.IDE and app_ctx.is_recent(
                    within_seconds=300
                ):
                    ide_space = space.space_id
                    break
            if ide_space:
                break

        # If we found at least 2 of 3, we have a debugging workflow
        found_count = sum(
            [terminal_space is not None, browser_space is not None, ide_space is not None]
        )
        if found_count >= 2:
            involved_spaces = [
                s for s in [terminal_space, browser_space, ide_space] if s is not None
            ]
            relationship_id = f"debug_workflow_{hash(tuple(sorted(involved_spaces)))}"

            return CrossSpaceRelationship(
                relationship_id=relationship_id,
                relationship_type="debugging_workflow",
                involved_spaces=involved_spaces,
                involved_apps=[
                    (terminal_space, "Terminal") if terminal_space else None,
                    (browser_space, "Browser") if browser_space else None,
                    (ide_space, "IDE") if ide_space else None,
                ],
                confidence=0.7 + (0.1 * (found_count - 2)),
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                evidence=[
                    {
                        "terminal_error": (
                            terminal_error.details.get("error") if terminal_error else None
                        ),
                        "browser_research": (
                            browser_research.browser_context.research_topic
                            if browser_research and browser_research.browser_context
                            else None
                        ),
                    }
                ],
                description=f"Debugging workflow across {found_count} spaces: Terminal error → Research → Fixing code",
            )

        return None

    async def _detect_research_and_code(
        self, spaces: Dict[int, SpaceContext]
    ) -> Optional[CrossSpaceRelationship]:
        """Detect: Browser docs + IDE coding happening simultaneously.

        Args:
            spaces: Dictionary of space contexts to analyze

        Returns:
            Optional[CrossSpaceRelationship]: Detected relationship or None
        """
        browser_space = None
        ide_space = None

        for space in spaces.values():
            # Find browser with docs/research
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.BROWSER and app_ctx.is_recent(
                    within_seconds=180
                ):
                    browser_space = space.space_id
                    break

            # Find IDE with activity
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.IDE and app_ctx.is_recent(
                    within_seconds=180
                ):
                    ide_space = space.space_id
                    break

        if browser_space and ide_space and browser_space != ide_space:
            relationship_id = f"research_code_{browser_space}_{ide_space}"
            return CrossSpaceRelationship(
                relationship_id=relationship_id,
                relationship_type="research_and_code",
                involved_spaces=[browser_space, ide_space],
                involved_apps=[(browser_space, "Browser"), (ide_space, "IDE")],
                confidence=0.6,
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                description=f"Reading documentation in Space {browser_space} while coding in Space {ide_space}",
            )

        return None

    async def _detect_cross_terminal_workflow(
        self, spaces: Dict[int, SpaceContext]
    ) -> Optional[CrossSpaceRelationship]:
        """Detect: Multiple terminals with related commands (e.g., dev server + tests).

        Args:
            spaces: Dictionary of space contexts to analyze

        Returns:
            Optional[CrossSpaceRelationship]: Detected relationship or None
        """
        terminal_spaces = []

        for space in spaces.values():
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.TERMINAL and app_ctx.is_recent(
                    within_seconds=300
                ):
                    terminal_spaces.append((space.space_id, app_name, app_ctx))

        if len(terminal_spaces) >= 2:
            involved_spaces = [s[0] for s in terminal_spaces]
            relationship_id = f"multi_terminal_{hash(tuple(sorted(involved_spaces)))}"
            return CrossSpaceRelationship(
                relationship_id=relationship_id,
                relationship_type="cross_terminal_workflow",
                involved_spaces=involved_spaces,
                involved_apps=[(s[0], s[1]) for s in terminal_spaces],
                confidence=0.5,
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                description=f"Multiple terminals across {len(terminal_spaces)} spaces working on related tasks",
            )

        return None

    async def _detect_documentation_lookup(
        self, spaces: Dict[int, SpaceContext]
    ) -> Optional[CrossSpaceRelationship]:
        """Detect: Quick doc lookup (browser opened briefly, then back to work).

        This would look for a pattern of:
        - Space switch to browser
        - Brief activity (< 30 seconds)
        - Switch back to original space

        Args:
            spaces: Dictionary of space contexts to analyze

        Returns:
            List of detected quick doc lookup patterns
        """
        # TODO: Implement quick doc lookup detection
        return []
