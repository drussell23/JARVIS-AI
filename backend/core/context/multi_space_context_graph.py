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
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# CONTEXT TYPES - Rich, Structured Context for Different Applications
# ============================================================================

class ContextType(Enum):
    """Types of application contexts we track"""
    TERMINAL = "terminal"
    BROWSER = "browser"
    IDE = "ide"
    EDITOR = "editor"
    COMMUNICATION = "communication"
    GENERIC = "generic"


class ActivitySignificance(Enum):
    """How significant is this activity?"""
    CRITICAL = "critical"      # Errors, crashes, important notifications
    HIGH = "high"              # Code changes, command execution, search queries
    NORMAL = "normal"          # Regular interactions, scrolling, reading
    LOW = "low"                # Idle, background activity
    BACKGROUND = "background"  # No user interaction


@dataclass
class TerminalContext:
    """Context for terminal/command-line applications"""
    last_command: Optional[str] = None
    last_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    working_directory: Optional[str] = None
    shell_type: str = "unknown"  # bash, zsh, fish, etc.
    recent_commands: deque = field(default_factory=lambda: deque(maxlen=10))

    def has_error(self) -> bool:
        return len(self.errors) > 0 or (self.exit_code is not None and self.exit_code != 0)


@dataclass
class BrowserContext:
    """Context for web browsers"""
    active_url: Optional[str] = None
    page_title: Optional[str] = None
    tabs: List[Dict[str, str]] = field(default_factory=list)  # [{"url": ..., "title": ...}]
    search_query: Optional[str] = None
    reading_content: Optional[str] = None  # OCR extracted text
    is_researching: bool = False  # Detected research behavior
    research_topic: Optional[str] = None


@dataclass
class IDEContext:
    """Context for IDEs (VS Code, IntelliJ, etc.)"""
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
    """Generic context for applications we don't have special handling for"""
    window_title: Optional[str] = None
    extracted_text: Optional[str] = None  # OCR
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None


# ============================================================================
# APPLICATION CONTEXT - State Tracking for Individual Applications
# ============================================================================

@dataclass
class ApplicationContext:
    """Tracks state for a specific application within a space"""
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

    def get_typed_context(self) -> Union[TerminalContext, BrowserContext, IDEContext, GenericAppContext]:
        """Get the specific context object for this app type"""
        if self.context_type == ContextType.TERMINAL:
            return self.terminal_context or TerminalContext()
        elif self.context_type == ContextType.BROWSER:
            return self.browser_context or BrowserContext()
        elif self.context_type == ContextType.IDE:
            return self.ide_context or IDEContext()
        else:
            return self.generic_context or GenericAppContext()

    def update_activity(self, significance: Optional[ActivitySignificance] = None):
        """Record activity in this application"""
        self.last_activity = datetime.now()
        self.activity_count += 1
        if significance:
            self.significance = significance

    def add_screenshot(self, screenshot_path: str, ocr_text: Optional[str] = None):
        """Add screenshot reference with optional OCR text"""
        self.screenshots.append({
            "path": screenshot_path,
            "timestamp": datetime.now(),
            "ocr_text": ocr_text
        })

    def is_recent(self, within_seconds: int = 180) -> bool:
        """Check if this app had recent activity (default: 3 minutes)"""
        return (datetime.now() - self.last_activity).total_seconds() <= within_seconds


# ============================================================================
# SPACE CONTEXT - Per-Space Activity Tracking
# ============================================================================

@dataclass
class ActivityEvent:
    """Individual activity event within a space"""
    event_type: str  # "app_launched", "command_executed", "error_detected", etc.
    timestamp: datetime
    app_name: Optional[str] = None
    significance: ActivitySignificance = ActivitySignificance.NORMAL
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "app_name": self.app_name,
            "significance": self.significance.value,
            "details": self.details
        }


class SpaceContext:
    """
    Tracks all activity within a single macOS Space/Desktop

    This is the per-space view that gets aggregated into the global context graph.
    """

    def __init__(self, space_id: int):
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
        """Mark this space as currently active"""
        self.is_active = True
        self.visit_count += 1
        self._last_activated = datetime.now()
        self.last_activity = datetime.now()

        self.add_event(ActivityEvent(
            event_type="space_activated",
            timestamp=datetime.now(),
            significance=ActivitySignificance.NORMAL
        ))

    def deactivate(self):
        """Mark this space as no longer active"""
        if self.is_active and self._last_activated:
            session_duration = datetime.now() - self._last_activated
            self.total_time_active += session_duration

        self.is_active = False

        self.add_event(ActivityEvent(
            event_type="space_deactivated",
            timestamp=datetime.now(),
            significance=ActivitySignificance.LOW
        ))

    def add_application(self, app_name: str, context_type: ContextType) -> ApplicationContext:
        """Add or retrieve application context"""
        if app_name not in self.applications:
            self.applications[app_name] = ApplicationContext(
                app_name=app_name,
                context_type=context_type,
                space_id=self.space_id
            )

            self.add_event(ActivityEvent(
                event_type="app_added",
                timestamp=datetime.now(),
                app_name=app_name,
                significance=ActivitySignificance.NORMAL,
                details={"context_type": context_type.value}
            ))

        return self.applications[app_name]

    def remove_application(self, app_name: str):
        """Remove application from this space"""
        if app_name in self.applications:
            del self.applications[app_name]

            self.add_event(ActivityEvent(
                event_type="app_removed",
                timestamp=datetime.now(),
                app_name=app_name,
                significance=ActivitySignificance.LOW
            ))

    def add_event(self, event: ActivityEvent):
        """Add activity event to timeline"""
        self.activity_timeline.append(event)
        self.last_activity = event.timestamp

    def get_recent_events(self, within_seconds: int = 180) -> List[ActivityEvent]:
        """Get events from the last N seconds (default: 3 minutes)"""
        cutoff = datetime.now() - timedelta(seconds=within_seconds)
        return [event for event in self.activity_timeline if event.timestamp > cutoff]

    def get_recent_errors(self, within_seconds: int = 300) -> List[Tuple[str, ActivityEvent]]:
        """Get recent errors from any application in this space"""
        errors = []
        cutoff = datetime.now() - timedelta(seconds=within_seconds)

        # Check terminal errors
        for app_name, app_ctx in self.applications.items():
            if app_ctx.context_type == ContextType.TERMINAL and app_ctx.terminal_context:
                if app_ctx.last_activity > cutoff and app_ctx.terminal_context.has_error():
                    for error in app_ctx.terminal_context.errors:
                        errors.append((app_name, ActivityEvent(
                            event_type="terminal_error",
                            timestamp=app_ctx.last_activity,
                            app_name=app_name,
                            significance=ActivitySignificance.CRITICAL,
                            details={"error": error}
                        )))

        return errors

    def infer_tags(self):
        """Automatically infer tags based on applications present"""
        new_tags = set()

        # Development indicators
        dev_apps = {"Terminal", "iTerm", "iTerm2", "VS Code", "Code", "IntelliJ", "PyCharm", "Sublime"}
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
        """Serialize to dictionary"""
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
                    "significance": ctx.significance.value
                }
                for name, ctx in self.applications.items()
            },
            "recent_events": [event.to_dict() for event in self.get_recent_events()]
        }


# ============================================================================
# CROSS-SPACE CORRELATION - Detecting Relationships Across Spaces
# ============================================================================

@dataclass
class CrossSpaceRelationship:
    """Represents a detected relationship between activities in different spaces"""
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
        """Update when we see more evidence of this relationship"""
        self.last_detected = datetime.now()
        self.evidence.append(new_evidence)
        # Increase confidence with more evidence (max 1.0)
        self.confidence = min(1.0, self.confidence + 0.1)


class CrossSpaceCorrelator:
    """
    Detects relationships and patterns across different spaces.

    This is the "intelligence" that understands:
    - Terminal error in Space 1 + Researching docs in Space 3 = Same problem
    - Editing code in Space 2 + Running tests in Space 1 = Development workflow
    - Reading Slack in Space 4 + Editing doc in Space 2 = Responding to request
    """

    def __init__(self):
        self.relationships: Dict[str, CrossSpaceRelationship] = {}
        self.relationship_patterns = self._init_patterns()

    def _init_patterns(self) -> List[Dict[str, Any]]:
        """Initialize relationship detection patterns"""
        return [
            {
                "name": "debugging_workflow",
                "description": "Terminal error + browser research + code editing",
                "detector": self._detect_debugging_workflow,
                "min_confidence": 0.7
            },
            {
                "name": "research_and_code",
                "description": "Reading documentation while coding",
                "detector": self._detect_research_and_code,
                "min_confidence": 0.6
            },
            {
                "name": "cross_terminal_workflow",
                "description": "Multiple terminals working on related tasks",
                "detector": self._detect_cross_terminal_workflow,
                "min_confidence": 0.5
            },
            {
                "name": "documentation_lookup",
                "description": "Quickly checking docs then returning to work",
                "detector": self._detect_documentation_lookup,
                "min_confidence": 0.8
            }
        ]

    async def analyze_relationships(self, spaces: Dict[int, SpaceContext]) -> List[CrossSpaceRelationship]:
        """
        Analyze all spaces and detect cross-space relationships.

        Returns new or updated relationships detected.
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
                        logger.info(f"[CROSS-SPACE] Detected new relationship: {result.relationship_type}")
            except Exception as e:
                logger.error(f"[CROSS-SPACE] Error in pattern '{pattern['name']}': {e}")

        return detected

    async def _detect_debugging_workflow(self, spaces: Dict[int, SpaceContext]) -> Optional[CrossSpaceRelationship]:
        """
        Detect: Terminal error + Browser research + IDE editing
        This is a very common developer workflow.
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
                if app_ctx.context_type == ContextType.BROWSER and app_ctx.is_recent(within_seconds=300):
                    if app_ctx.browser_context and app_ctx.browser_context.is_researching:
                        browser_space = space.space_id
                        browser_research = app_ctx
                        break
            if browser_research:
                break

        # Find IDE with recent activity
        for space in spaces.values():
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.IDE and app_ctx.is_recent(within_seconds=300):
                    ide_space = space.space_id
                    break
            if ide_space:
                break

        # If we found at least 2 of 3, we have a debugging workflow
        found_count = sum([terminal_space is not None, browser_space is not None, ide_space is not None])
        if found_count >= 2:
            involved_spaces = [s for s in [terminal_space, browser_space, ide_space] if s is not None]
            relationship_id = f"debug_workflow_{hash(tuple(sorted(involved_spaces)))}"

            return CrossSpaceRelationship(
                relationship_id=relationship_id,
                relationship_type="debugging_workflow",
                involved_spaces=involved_spaces,
                involved_apps=[
                    (terminal_space, "Terminal") if terminal_space else None,
                    (browser_space, "Browser") if browser_space else None,
                    (ide_space, "IDE") if ide_space else None
                ],
                confidence=0.7 + (0.1 * (found_count - 2)),
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                evidence=[{
                    "terminal_error": terminal_error.details.get("error") if terminal_error else None,
                    "browser_research": browser_research.browser_context.research_topic if browser_research and browser_research.browser_context else None
                }],
                description=f"Debugging workflow across {found_count} spaces: Terminal error → Research → Fixing code"
            )

        return None

    async def _detect_research_and_code(self, spaces: Dict[int, SpaceContext]) -> Optional[CrossSpaceRelationship]:
        """Detect: Browser docs + IDE coding happening simultaneously"""
        browser_space = None
        ide_space = None

        for space in spaces.values():
            # Find browser with docs/research
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.BROWSER and app_ctx.is_recent(within_seconds=180):
                    browser_space = space.space_id
                    break

            # Find IDE with activity
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.IDE and app_ctx.is_recent(within_seconds=180):
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
                description=f"Reading documentation in Space {browser_space} while coding in Space {ide_space}"
            )

        return None

    async def _detect_cross_terminal_workflow(self, spaces: Dict[int, SpaceContext]) -> Optional[CrossSpaceRelationship]:
        """Detect: Multiple terminals with related commands (e.g., dev server + tests)"""
        terminal_spaces = []

        for space in spaces.values():
            for app_name, app_ctx in space.applications.items():
                if app_ctx.context_type == ContextType.TERMINAL and app_ctx.is_recent(within_seconds=300):
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
                description=f"Multiple terminals across {len(terminal_spaces)} spaces working on related tasks"
            )

        return None

    async def _detect_documentation_lookup(self, spaces: Dict[int, SpaceContext]) -> Optional[CrossSpaceRelationship]:
        """Detect: Quick doc lookup (browser opened briefly, then back to work)"""
        # This would look for a pattern of:
        # - Space switch to browser
        # - Brief activity (< 30 seconds)
        # - Switch back to original space
        # Implementation would require more detailed timing analysis
        return None


# ============================================================================
# MULTI-SPACE CONTEXT GRAPH - Main Coordinator
# ============================================================================

class MultiSpaceContextGraph:
    """
    The master context graph that tracks everything happening across all spaces.

    This is JARVIS's "working memory" - it knows:
    - What's in each space right now
    - What happened in the last 3-5 minutes
    - How activities across spaces are related
    - Where to look when you ask "what does it say?"

    Key Features:
    - NO HARDCODING: Dynamically adapts to any workspace configuration
    - TEMPORAL DECAY: Old context fades away (3-5 minutes default)
    - CROSS-SPACE INTELLIGENCE: Understands relationships across spaces
    - NATURAL LANGUAGE READY: Can answer "what's the error?" queries
    """

    def __init__(self,
                 context_store=None,
                 decay_ttl_seconds: int = 300,  # 5 minutes default
                 enable_cross_space_correlation: bool = True):
        """
        Initialize the multi-space context graph.

        Args:
            context_store: Optional ContextStore backend for persistence
            decay_ttl_seconds: How long to keep context before decay (default: 5 minutes)
            enable_cross_space_correlation: Enable cross-space relationship detection
        """
        # Core state
        self.spaces: Dict[int, SpaceContext] = {}
        self.current_space_id: Optional[int] = None

        # Temporal decay configuration
        self.decay_ttl = timedelta(seconds=decay_ttl_seconds)
        self._decay_task: Optional[asyncio.Task] = None

        # Cross-space correlation
        self.enable_correlation = enable_cross_space_correlation
        self.correlator = CrossSpaceCorrelator() if enable_cross_space_correlation else None
        self._correlation_task: Optional[asyncio.Task] = None

        # Integration with existing systems
        self.context_store = context_store

        # Callbacks for external systems
        self.on_critical_event: Optional[Callable] = None  # Notify on critical events
        self.on_relationship_detected: Optional[Callable] = None  # Notify on cross-space patterns

        logger.info("[MULTI-SPACE-GRAPH] Initialized with decay_ttl=%ds, correlation=%s",
                   decay_ttl_seconds, enable_cross_space_correlation)

    async def start(self):
        """Start background tasks (decay, correlation)"""
        if self._decay_task is None or self._decay_task.done():
            self._decay_task = asyncio.create_task(self._decay_loop())
            logger.info("[MULTI-SPACE-GRAPH] Started decay loop")

        if self.enable_correlation and (self._correlation_task is None or self._correlation_task.done()):
            self._correlation_task = asyncio.create_task(self._correlation_loop())
            logger.info("[MULTI-SPACE-GRAPH] Started correlation loop")

    async def stop(self):
        """Stop background tasks"""
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass

        if self._correlation_task:
            self._correlation_task.cancel()
            try:
                await self._correlation_task
            except asyncio.CancelledError:
                pass

        logger.info("[MULTI-SPACE-GRAPH] Stopped all background tasks")

    # ========================================================================
    # SPACE MANAGEMENT
    # ========================================================================

    def get_or_create_space(self, space_id: int) -> SpaceContext:
        """Get existing space context or create new one"""
        if space_id not in self.spaces:
            self.spaces[space_id] = SpaceContext(space_id)
            logger.info(f"[MULTI-SPACE-GRAPH] Created new space context: Space {space_id}")

        return self.spaces[space_id]

    def set_active_space(self, space_id: int):
        """Mark a space as currently active"""
        # Deactivate previous space
        if self.current_space_id is not None and self.current_space_id in self.spaces:
            self.spaces[self.current_space_id].deactivate()

        # Activate new space
        space = self.get_or_create_space(space_id)
        space.activate()
        self.current_space_id = space_id

        logger.debug(f"[MULTI-SPACE-GRAPH] Switched to Space {space_id}")

    def remove_space(self, space_id: int):
        """Remove a space from tracking"""
        if space_id in self.spaces:
            del self.spaces[space_id]
            logger.info(f"[MULTI-SPACE-GRAPH] Removed Space {space_id}")

    # ========================================================================
    # APPLICATION CONTEXT MANAGEMENT
    # ========================================================================

    def update_terminal_context(self,
                                space_id: int,
                                app_name: str,
                                command: Optional[str] = None,
                                output: Optional[str] = None,
                                errors: Optional[List[str]] = None,
                                exit_code: Optional[int] = None,
                                working_dir: Optional[str] = None):
        """
        Update terminal context in a specific space.

        This is called when we detect terminal activity via OCR or system monitoring.
        """
        space = self.get_or_create_space(space_id)
        app_ctx = space.add_application(app_name, ContextType.TERMINAL)

        if app_ctx.terminal_context is None:
            app_ctx.terminal_context = TerminalContext()

        terminal = app_ctx.terminal_context

        if command:
            terminal.last_command = command
            terminal.recent_commands.append((command, datetime.now()))
        if output:
            terminal.last_output = output
        if errors:
            terminal.errors.extend(errors)
            # Critical event - terminal error!
            significance = ActivitySignificance.CRITICAL
            space.add_event(ActivityEvent(
                event_type="terminal_error",
                timestamp=datetime.now(),
                app_name=app_name,
                significance=significance,
                details={"errors": errors, "command": command}
            ))

            if self.on_critical_event:
                asyncio.create_task(self._safe_callback(self.on_critical_event, {
                    "type": "terminal_error",
                    "space_id": space_id,
                    "app_name": app_name,
                    "errors": errors,
                    "command": command
                }))
        if exit_code is not None:
            terminal.exit_code = exit_code
            if exit_code != 0:
                significance = ActivitySignificance.HIGH
            else:
                significance = ActivitySignificance.NORMAL
        else:
            significance = ActivitySignificance.NORMAL
        if working_dir:
            terminal.working_directory = working_dir

        app_ctx.update_activity(significance)
        logger.debug(f"[MULTI-SPACE-GRAPH] Updated terminal context: Space {space_id}, {app_name}")

    def update_browser_context(self,
                              space_id: int,
                              app_name: str,
                              url: Optional[str] = None,
                              title: Optional[str] = None,
                              extracted_text: Optional[str] = None,
                              search_query: Optional[str] = None):
        """Update browser context in a specific space"""
        space = self.get_or_create_space(space_id)
        app_ctx = space.add_application(app_name, ContextType.BROWSER)

        if app_ctx.browser_context is None:
            app_ctx.browser_context = BrowserContext()

        browser = app_ctx.browser_context

        if url:
            browser.active_url = url
        if title:
            browser.page_title = title
        if extracted_text:
            browser.reading_content = extracted_text
            # Detect if this looks like research
            research_indicators = ["documentation", "docs", "stack overflow", "github", "tutorial", "guide"]
            if any(indicator in extracted_text.lower() for indicator in research_indicators):
                browser.is_researching = True
        if search_query:
            browser.search_query = search_query
            browser.is_researching = True

        significance = ActivitySignificance.HIGH if browser.is_researching else ActivitySignificance.NORMAL
        app_ctx.update_activity(significance)

        logger.debug(f"[MULTI-SPACE-GRAPH] Updated browser context: Space {space_id}, {app_name}")

    def update_ide_context(self,
                          space_id: int,
                          app_name: str,
                          active_file: Optional[str] = None,
                          open_files: Optional[List[str]] = None,
                          errors: Optional[List[str]] = None):
        """Update IDE context in a specific space"""
        space = self.get_or_create_space(space_id)
        app_ctx = space.add_application(app_name, ContextType.IDE)

        if app_ctx.ide_context is None:
            app_ctx.ide_context = IDEContext()

        ide = app_ctx.ide_context

        if active_file:
            ide.active_file = active_file
        if open_files:
            ide.open_files = open_files
        if errors:
            ide.errors_in_file.extend(errors)
            significance = ActivitySignificance.HIGH
        else:
            significance = ActivitySignificance.NORMAL

        app_ctx.update_activity(significance)
        logger.debug(f"[MULTI-SPACE-GRAPH] Updated IDE context: Space {space_id}, {app_name}")

    def add_screenshot_reference(self,
                                space_id: int,
                                app_name: str,
                                screenshot_path: str,
                                ocr_text: Optional[str] = None):
        """Add screenshot reference to application context"""
        space = self.get_or_create_space(space_id)
        if app_name in space.applications:
            space.applications[app_name].add_screenshot(screenshot_path, ocr_text)

    def update_generic_context(self, space_id: int, app_name: str, content: str):
        """Update generic application context (fallback for unknown app types)"""
        space = self.get_or_create_space(space_id)
        app_ctx = space.add_application(app_name, ContextType.GENERIC)

        if app_ctx.generic_context is None:
            app_ctx.generic_context = GenericAppContext()

        generic = app_ctx.generic_context
        generic.extracted_text = content[:500]  # Store first 500 chars
        generic.interaction_count += 1
        generic.last_interaction = datetime.now()

        app_ctx.update_activity(ActivitySignificance.LOW)
        logger.debug(f"[MULTI-SPACE-GRAPH] Updated generic context: Space {space_id}, {app_name}")

    # ========================================================================
    # CONTEXT QUERYING - "What does it say?" Natural Language Interface
    # ========================================================================

    def find_most_recent_error(self, within_seconds: int = 300) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        """
        Find the most recent error across all spaces.

        This is what powers "what does it say?" when there's an error on screen.

        Returns: (space_id, app_name, error_details) or None
        """
        most_recent = None
        most_recent_time = None

        for space in self.spaces.values():
            errors = space.get_recent_errors(within_seconds=within_seconds)
            for app_name, event in errors:
                if most_recent_time is None or event.timestamp > most_recent_time:
                    most_recent_time = event.timestamp
                    most_recent = (space.space_id, app_name, event.details)

        return most_recent

    def find_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Find relevant context based on natural language query.

        Examples:
        - "what does it say?" → Find most recent error/message
        - "what's the error?" → Find most recent error
        - "what's happening in the terminal?" → Find terminal context

        This is the foundation for implicit reference resolution.
        """
        query_lower = query.lower()

        # Implicit reference queries - "what does it say?" "what happened?" etc.
        # These should check for the most critical/recent thing first
        implicit_queries = ["what does it say", "what did it say", "what happened", "what's that"]
        if any(q in query_lower for q in implicit_queries):
            # First check for errors (most important)
            error = self.find_most_recent_error()
            if error:
                space_id, app_name, details = error
                return {
                    "type": "error",
                    "space_id": space_id,
                    "app_name": app_name,
                    "details": details,
                    "message": f"The error in {app_name} (Space {space_id}) is: {details.get('error', 'Unknown error')}"
                }

        # Error-related queries
        if any(word in query_lower for word in ["error", "wrong", "failed", "problem"]):
            error = self.find_most_recent_error()
            if error:
                space_id, app_name, details = error
                return {
                    "type": "error",
                    "space_id": space_id,
                    "app_name": app_name,
                    "details": details,
                    "message": f"The error in {app_name} (Space {space_id}) is: {details.get('error', 'Unknown error')}"
                }

        # Terminal-specific queries
        if "terminal" in query_lower:
            # Find most recent terminal activity
            for space in sorted(self.spaces.values(), key=lambda s: s.last_activity, reverse=True):
                for app_name, app_ctx in space.applications.items():
                    if app_ctx.context_type == ContextType.TERMINAL and app_ctx.is_recent():
                        terminal = app_ctx.terminal_context
                        return {
                            "type": "terminal",
                            "space_id": space.space_id,
                            "app_name": app_name,
                            "last_command": terminal.last_command if terminal else None,
                            "last_output": terminal.last_output if terminal else None,
                            "errors": terminal.errors if terminal else []
                        }

        # Current space context
        if self.current_space_id and self.current_space_id in self.spaces:
            current_space = self.spaces[self.current_space_id]
            return {
                "type": "current_space",
                "space_id": current_space.space_id,
                "applications": list(current_space.applications.keys()),
                "recent_events": [e.to_dict() for e in current_space.get_recent_events()]
            }

        return {"type": "no_relevant_context", "message": "I don't see any recent activity to reference."}

    def get_cross_space_summary(self) -> str:
        """
        Generate natural language summary of cross-space activity.

        This is what JARVIS uses to say things like:
        "I see you're debugging an error in Space 1, researching solutions in Space 3,
         and editing the fix in Space 2."
        """
        if not self.correlator or not self.correlator.relationships:
            return "No cross-space relationships detected."

        summaries = []
        for rel in self.correlator.relationships.values():
            if (datetime.now() - rel.last_detected).total_seconds() < 300:  # Recent (5 minutes)
                summaries.append(rel.description)

        if summaries:
            return " ".join(summaries)
        else:
            return "No recent cross-space activity."

    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================

    async def _decay_loop(self):
        """Background task to decay old context"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._apply_decay()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MULTI-SPACE-GRAPH] Error in decay loop: {e}")

    async def _apply_decay(self):
        """Remove or decay old context based on TTL"""
        cutoff = datetime.now() - self.decay_ttl
        spaces_to_remove = []

        for space_id, space in self.spaces.items():
            # If space hasn't been active recently, consider removing it
            if space.last_activity < cutoff and not space.is_active:
                # But only if it has no recent critical events
                recent_critical = any(
                    e.significance == ActivitySignificance.CRITICAL
                    for e in space.get_recent_events(within_seconds=self.decay_ttl.total_seconds())
                )
                if not recent_critical:
                    spaces_to_remove.append(space_id)

        for space_id in spaces_to_remove:
            logger.info(f"[MULTI-SPACE-GRAPH] Decaying Space {space_id} (inactive for {self.decay_ttl})")
            self.remove_space(space_id)

    async def _correlation_loop(self):
        """Background task to detect cross-space correlations"""
        while True:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds
                if self.correlator:
                    relationships = await self.correlator.analyze_relationships(self.spaces)

                    if relationships and self.on_relationship_detected:
                        for rel in relationships:
                            await self._safe_callback(self.on_relationship_detected, rel)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MULTI-SPACE-GRAPH] Error in correlation loop: {e}")

    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback without crashing the graph"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"[MULTI-SPACE-GRAPH] Error in callback: {e}")

    # ========================================================================
    # INSPECTION & DEBUGGING
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current context graph state"""
        return {
            "total_spaces": len(self.spaces),
            "current_space_id": self.current_space_id,
            "active_spaces": [s.space_id for s in self.spaces.values() if s.is_active],
            "spaces": {
                space_id: space.to_dict()
                for space_id, space in self.spaces.items()
            },
            "cross_space_relationships": [
                {
                    "relationship_id": rel.relationship_id,
                    "type": rel.relationship_type,
                    "involved_spaces": rel.involved_spaces,
                    "confidence": rel.confidence,
                    "description": rel.description
                }
                for rel in (self.correlator.relationships.values() if self.correlator else [])
            ] if self.enable_correlation else [],
            "decay_ttl_seconds": self.decay_ttl.total_seconds()
        }

    def export_to_json(self, filepath: Path):
        """Export current state to JSON for debugging/analysis"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"[MULTI-SPACE-GRAPH] Exported state to {filepath}")


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

_global_context_graph: Optional[MultiSpaceContextGraph] = None


def get_context_graph() -> MultiSpaceContextGraph:
    """Get or create the global context graph singleton"""
    global _global_context_graph
    if _global_context_graph is None:
        _global_context_graph = MultiSpaceContextGraph()
    return _global_context_graph


def set_context_graph(graph: MultiSpaceContextGraph):
    """Set the global context graph (for testing or custom configuration)"""
    global _global_context_graph
    _global_context_graph = graph
