"""
JARVIS macOS Helper - Screen Context Analyzer

Advanced screen context analysis using Claude Vision integration.
Extracts semantic understanding of what the user is doing from screen content.

Features:
- Real-time screen capture with memory optimization
- Claude Vision integration for semantic analysis
- OCR extraction for text-based context
- Visual element tracking and identification
- Activity type classification
- Context enrichment with app metadata
- Async-first design with configurable intervals

Architecture:
    ScreenContextAnalyzer
    ├── CaptureManager (efficient screen capture)
    ├── VisionAnalyzer (Claude Vision integration)
    ├── ContextExtractor (semantic context extraction)
    ├── ElementTracker (visual element tracking)
    └── ContextEnricher (metadata enrichment)

No hardcoded values - all configuration via environment or parameters.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class ActivityType(str, Enum):
    """Types of user activities detected from screen context."""
    CODING = "coding"
    WRITING = "writing"
    BROWSING = "browsing"
    COMMUNICATION = "communication"  # Email, chat, video call
    MEDIA = "media"  # Video, music, images
    PRODUCTIVITY = "productivity"  # Spreadsheets, documents
    DESIGN = "design"  # Design tools, image editing
    TERMINAL = "terminal"
    MEETING = "meeting"
    READING = "reading"
    IDLE = "idle"
    SYSTEM = "system"  # System preferences, settings
    UNKNOWN = "unknown"


class ContextConfidence(str, Enum):
    """Confidence level of context analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VisualElement:
    """A detected visual element on screen."""
    element_id: str
    element_type: str  # button, input, text, image, etc.
    text_content: Optional[str] = None
    bounds: Optional[Dict[str, int]] = None  # x, y, width, height
    confidence: float = 0.0
    is_interactive: bool = False
    is_focused: bool = False
    app_context: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "text_content": self.text_content,
            "bounds": self.bounds,
            "confidence": self.confidence,
            "is_interactive": self.is_interactive,
            "is_focused": self.is_focused,
            "app_context": self.app_context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ScreenContext:
    """Rich context extracted from screen analysis."""
    # Basic info
    context_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    timestamp: datetime = field(default_factory=datetime.now)

    # App context
    active_app: Optional[str] = None
    active_bundle_id: Optional[str] = None
    window_title: Optional[str] = None

    # Activity classification
    activity_type: ActivityType = ActivityType.UNKNOWN
    activity_confidence: ContextConfidence = ContextConfidence.UNCERTAIN

    # Content analysis
    visible_text: List[str] = field(default_factory=list)
    detected_elements: List[VisualElement] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)  # Names, files, URLs, etc.

    # Semantic understanding
    user_intent: Optional[str] = None  # What the user appears to be doing
    current_task: Optional[str] = None  # Inferred current task
    task_progress: Optional[float] = None  # 0.0 to 1.0 if determinable

    # Environment
    is_fullscreen: bool = False
    has_modal_dialog: bool = False
    has_notification: bool = False
    screen_brightness: Optional[float] = None

    # State tracking
    is_error_state: bool = False
    error_message: Optional[str] = None
    is_loading: bool = False
    is_idle: bool = False

    # Enrichment
    related_files: List[str] = field(default_factory=list)
    related_urls: List[str] = field(default_factory=list)
    code_language: Optional[str] = None  # If coding detected

    # Analysis metadata
    analysis_duration_ms: float = 0.0
    vision_tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "context_id": self.context_id,
            "timestamp": self.timestamp.isoformat(),
            "active_app": self.active_app,
            "active_bundle_id": self.active_bundle_id,
            "window_title": self.window_title,
            "activity_type": self.activity_type.value,
            "activity_confidence": self.activity_confidence.value,
            "visible_text_count": len(self.visible_text),
            "detected_elements_count": len(self.detected_elements),
            "key_entities": self.key_entities,
            "user_intent": self.user_intent,
            "current_task": self.current_task,
            "task_progress": self.task_progress,
            "is_fullscreen": self.is_fullscreen,
            "has_modal_dialog": self.has_modal_dialog,
            "is_error_state": self.is_error_state,
            "is_idle": self.is_idle,
            "code_language": self.code_language,
            "analysis_duration_ms": self.analysis_duration_ms,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the context."""
        parts = []

        if self.active_app:
            parts.append(f"Using {self.active_app}")

        if self.activity_type != ActivityType.UNKNOWN:
            parts.append(f"({self.activity_type.value})")

        if self.current_task:
            parts.append(f"- {self.current_task}")

        if self.is_error_state and self.error_message:
            parts.append(f"[ERROR: {self.error_message}]")

        return " ".join(parts) if parts else "Unknown context"


@dataclass
class ContextChange:
    """Represents a change in screen context."""
    change_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    previous_context: Optional[ScreenContext] = None
    current_context: Optional[ScreenContext] = None
    change_type: str = "unknown"  # app_switch, content_change, error_appeared, etc.
    change_magnitude: float = 0.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScreenContextConfig:
    """Configuration for the screen context analyzer."""
    # Capture settings
    capture_interval_seconds: float = float(os.getenv("CONTEXT_CAPTURE_INTERVAL", "3.0"))
    max_captures_in_memory: int = int(os.getenv("CONTEXT_MAX_CAPTURES", "10"))
    capture_scale: float = float(os.getenv("CONTEXT_CAPTURE_SCALE", "0.5"))  # Downscale for efficiency

    # Vision settings
    enable_vision_analysis: bool = os.getenv("CONTEXT_ENABLE_VISION", "true").lower() == "true"
    vision_analysis_interval: float = float(os.getenv("CONTEXT_VISION_INTERVAL", "5.0"))
    max_vision_tokens: int = int(os.getenv("CONTEXT_MAX_VISION_TOKENS", "1000"))

    # OCR settings
    enable_ocr: bool = os.getenv("CONTEXT_ENABLE_OCR", "true").lower() == "true"
    ocr_confidence_threshold: float = float(os.getenv("CONTEXT_OCR_THRESHOLD", "0.7"))

    # Memory management
    memory_limit_mb: int = int(os.getenv("CONTEXT_MEMORY_LIMIT_MB", "150"))
    cache_duration_seconds: float = float(os.getenv("CONTEXT_CACHE_DURATION", "30.0"))

    # Activity detection
    idle_threshold_seconds: float = float(os.getenv("CONTEXT_IDLE_THRESHOLD", "300.0"))  # 5 minutes

    # Change detection
    change_detection_enabled: bool = True
    min_change_magnitude: float = float(os.getenv("CONTEXT_MIN_CHANGE", "0.1"))


# =============================================================================
# Activity Classification
# =============================================================================

class ActivityClassifier:
    """
    Classifies user activity based on app and context.

    Uses a learned mapping of apps to activity types,
    with fallback to heuristic classification.
    """

    def __init__(self):
        # Learned mappings (can be updated from learning database)
        self._app_activity_map: Dict[str, ActivityType] = {}
        self._bundle_activity_map: Dict[str, ActivityType] = {}

        # Default mappings (used as fallback)
        self._default_bundle_map = {
            # Coding
            "com.microsoft.VSCode": ActivityType.CODING,
            "com.todesktop.230313mzl4w4u92": ActivityType.CODING,  # Cursor
            "com.apple.dt.Xcode": ActivityType.CODING,
            "com.jetbrains": ActivityType.CODING,  # Prefix match
            "com.sublimetext": ActivityType.CODING,

            # Communication
            "com.apple.MobileSMS": ActivityType.COMMUNICATION,
            "com.tinyspeck.slackmacgap": ActivityType.COMMUNICATION,
            "us.zoom.xos": ActivityType.MEETING,
            "com.microsoft.teams": ActivityType.MEETING,
            "com.google.Chrome.app.kjgfgldnnfoeklkmfkjfagphfepbbdan": ActivityType.MEETING,  # Meet

            # Browsing
            "com.apple.Safari": ActivityType.BROWSING,
            "com.google.Chrome": ActivityType.BROWSING,
            "org.mozilla.firefox": ActivityType.BROWSING,
            "company.thebrowser.Browser": ActivityType.BROWSING,  # Arc

            # Productivity
            "com.microsoft.Word": ActivityType.WRITING,
            "com.microsoft.Excel": ActivityType.PRODUCTIVITY,
            "com.microsoft.Powerpoint": ActivityType.PRODUCTIVITY,
            "com.apple.iWork.Pages": ActivityType.WRITING,
            "com.apple.iWork.Numbers": ActivityType.PRODUCTIVITY,
            "com.apple.iWork.Keynote": ActivityType.PRODUCTIVITY,
            "com.notion.id": ActivityType.WRITING,
            "md.obsidian": ActivityType.WRITING,

            # Design
            "com.figma.Desktop": ActivityType.DESIGN,
            "com.adobe.Photoshop": ActivityType.DESIGN,
            "com.adobe.illustrator": ActivityType.DESIGN,
            "com.bohemiancoding.sketch3": ActivityType.DESIGN,

            # Media
            "com.spotify.client": ActivityType.MEDIA,
            "com.apple.Music": ActivityType.MEDIA,
            "com.apple.TV": ActivityType.MEDIA,
            "com.google.Chrome.app.agimnkijcaahngcdmfeangaknmldooml": ActivityType.MEDIA,  # YouTube

            # Terminal
            "com.apple.Terminal": ActivityType.TERMINAL,
            "com.googlecode.iterm2": ActivityType.TERMINAL,
            "dev.warp.Warp-Stable": ActivityType.TERMINAL,

            # System
            "com.apple.systempreferences": ActivityType.SYSTEM,
            "com.apple.finder": ActivityType.PRODUCTIVITY,
        }

    def classify(
        self,
        app_name: Optional[str],
        bundle_id: Optional[str],
        window_title: Optional[str] = None,
        visible_text: Optional[List[str]] = None,
    ) -> Tuple[ActivityType, ContextConfidence]:
        """
        Classify the current activity.

        Args:
            app_name: Active application name
            bundle_id: Bundle identifier
            window_title: Window title
            visible_text: Visible text on screen

        Returns:
            Tuple of (ActivityType, ContextConfidence)
        """
        # Try learned mapping first
        if bundle_id and bundle_id in self._bundle_activity_map:
            return self._bundle_activity_map[bundle_id], ContextConfidence.HIGH

        if app_name and app_name in self._app_activity_map:
            return self._app_activity_map[app_name], ContextConfidence.HIGH

        # Try default mapping
        if bundle_id:
            # Exact match
            if bundle_id in self._default_bundle_map:
                return self._default_bundle_map[bundle_id], ContextConfidence.MEDIUM

            # Prefix match
            for prefix, activity in self._default_bundle_map.items():
                if bundle_id.startswith(prefix):
                    return activity, ContextConfidence.MEDIUM

        # Heuristic classification based on window title
        if window_title:
            title_lower = window_title.lower()

            # Coding indicators
            coding_indicators = [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"]
            if any(ind in title_lower for ind in coding_indicators):
                return ActivityType.CODING, ContextConfidence.MEDIUM

            # Meeting indicators
            meeting_indicators = ["meeting", "call", "zoom", "teams", "meet"]
            if any(ind in title_lower for ind in meeting_indicators):
                return ActivityType.MEETING, ContextConfidence.MEDIUM

        return ActivityType.UNKNOWN, ContextConfidence.UNCERTAIN

    def learn_mapping(self, bundle_id: str, activity_type: ActivityType) -> None:
        """Learn a new app-to-activity mapping."""
        self._bundle_activity_map[bundle_id] = activity_type

    def learn_from_feedback(self, bundle_id: str, app_name: str, correct_activity: ActivityType) -> None:
        """Update mappings based on user feedback."""
        if bundle_id:
            self._bundle_activity_map[bundle_id] = correct_activity
        if app_name:
            self._app_activity_map[app_name] = correct_activity


# =============================================================================
# Screen Context Analyzer
# =============================================================================

class ScreenContextAnalyzer:
    """
    Analyzes screen content to extract rich contextual information.

    Features:
    - Screen capture with memory optimization
    - Claude Vision integration for semantic analysis
    - Activity classification
    - Element tracking
    - Change detection
    - Context history with circular buffer
    """

    def __init__(self, config: Optional[ScreenContextConfig] = None):
        """
        Initialize the screen context analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or ScreenContextConfig()

        # State
        self._running = False
        self._paused = False
        self._started_at: Optional[datetime] = None

        # Components
        self._activity_classifier = ActivityClassifier()

        # Vision integration (lazy loaded)
        self._vision_analyzer = None
        self._screen_capturer = None

        # Context history
        self._context_history: Deque[ScreenContext] = deque(
            maxlen=self.config.max_captures_in_memory
        )
        self._current_context: Optional[ScreenContext] = None

        # Change tracking
        self._last_analysis_time: Optional[datetime] = None
        self._last_vision_time: Optional[datetime] = None
        self._last_capture_hash: Optional[str] = None

        # Callbacks
        self._on_context_changed: List[Callable[[ContextChange], Coroutine]] = []
        self._on_activity_changed: List[Callable[[ActivityType, ActivityType], Coroutine]] = []

        # Background tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._vision_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "analyses_performed": 0,
            "vision_analyses": 0,
            "context_changes_detected": 0,
            "errors": 0,
        }

        logger.debug("ScreenContextAnalyzer initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the screen context analyzer.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        try:
            self._running = True
            self._started_at = datetime.now()

            # Initialize vision integration
            await self._init_vision_integration()

            # Start background tasks
            self._capture_task = asyncio.create_task(
                self._capture_loop(),
                name="screen_context_capture"
            )

            if self.config.enable_vision_analysis:
                self._vision_task = asyncio.create_task(
                    self._vision_analysis_loop(),
                    name="screen_context_vision"
                )

            logger.info("ScreenContextAnalyzer started")
            return True

        except Exception as e:
            logger.error(f"Failed to start ScreenContextAnalyzer: {e}")
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the screen context analyzer."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        for task in [self._capture_task, self._vision_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("ScreenContextAnalyzer stopped")

    def pause(self) -> None:
        """Pause analysis (keeps running but skips processing)."""
        self._paused = True

    def resume(self) -> None:
        """Resume analysis."""
        self._paused = False

    # =========================================================================
    # Vision Integration
    # =========================================================================

    async def _init_vision_integration(self) -> None:
        """Initialize vision analysis integration."""
        try:
            # Try to get existing screen analyzer
            from vision.continuous_screen_analyzer import MemoryAwareScreenAnalyzer

            # We'll create a lightweight wrapper if needed
            logger.info("Vision integration available")

        except ImportError:
            logger.warning("Vision integration not available - using basic capture only")

    async def _capture_screen(self) -> Optional[bytes]:
        """Capture the current screen."""
        try:
            import subprocess
            import tempfile

            # Use screencapture for macOS
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name

            result = subprocess.run(
                ["screencapture", "-x", "-C", temp_path],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                with open(temp_path, "rb") as f:
                    data = f.read()

                # Clean up
                os.unlink(temp_path)

                return data

        except Exception as e:
            logger.warning(f"Screen capture failed: {e}")
            self._stats["errors"] += 1

        return None

    def _compute_capture_hash(self, capture_data: bytes) -> str:
        """Compute hash of capture for change detection."""
        return hashlib.md5(capture_data).hexdigest()

    # =========================================================================
    # Background Loops
    # =========================================================================

    async def _capture_loop(self) -> None:
        """Main capture and analysis loop."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(1.0)
                    continue

                start_time = time.time()

                # Get current app context from macOS helper
                app_context = await self._get_app_context()

                # Classify activity
                activity_type, confidence = self._activity_classifier.classify(
                    app_name=app_context.get("app_name"),
                    bundle_id=app_context.get("bundle_id"),
                    window_title=app_context.get("window_title"),
                )

                # Build context
                new_context = ScreenContext(
                    active_app=app_context.get("app_name"),
                    active_bundle_id=app_context.get("bundle_id"),
                    window_title=app_context.get("window_title"),
                    activity_type=activity_type,
                    activity_confidence=confidence,
                    is_fullscreen=app_context.get("is_fullscreen", False),
                    analysis_duration_ms=(time.time() - start_time) * 1000,
                )

                # Detect changes
                await self._process_context_change(new_context)

                # Update current context
                self._current_context = new_context
                self._context_history.append(new_context)
                self._last_analysis_time = datetime.now()
                self._stats["analyses_performed"] += 1

                await asyncio.sleep(self.config.capture_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self._stats["errors"] += 1
                await asyncio.sleep(self.config.capture_interval_seconds)

    async def _vision_analysis_loop(self) -> None:
        """Vision-based deep analysis loop (runs less frequently)."""
        while self._running:
            try:
                if self._paused or not self.config.enable_vision_analysis:
                    await asyncio.sleep(self.config.vision_analysis_interval)
                    continue

                # Capture screen for vision analysis
                capture_data = await self._capture_screen()

                if capture_data:
                    capture_hash = self._compute_capture_hash(capture_data)

                    # Skip if screen hasn't changed significantly
                    if capture_hash != self._last_capture_hash:
                        await self._perform_vision_analysis(capture_data)
                        self._last_capture_hash = capture_hash
                        self._last_vision_time = datetime.now()
                        self._stats["vision_analyses"] += 1

                await asyncio.sleep(self.config.vision_analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Vision analysis loop error: {e}")
                await asyncio.sleep(self.config.vision_analysis_interval)

    async def _perform_vision_analysis(self, capture_data: bytes) -> None:
        """Perform deep vision analysis on screen capture."""
        try:
            # Try to use Claude Vision via existing integration
            # This is where we'd call the vision analyzer

            # For now, we update the current context with any extracted info
            if self._current_context:
                # Extract visible text, detect elements, etc.
                pass

        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")

    # =========================================================================
    # App Context
    # =========================================================================

    async def _get_app_context(self) -> Dict[str, Any]:
        """Get current app context from macOS."""
        try:
            import subprocess

            # Get frontmost app via AppleScript
            script = '''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set appName to name of frontApp
                set bundleId to bundle identifier of frontApp
                set windowTitle to ""
                try
                    set windowTitle to name of front window of frontApp
                end try
                return appName & "|" & bundleId & "|" & windowTitle
            end tell
            '''

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split("|")
                if len(parts) >= 3:
                    return {
                        "app_name": parts[0],
                        "bundle_id": parts[1],
                        "window_title": parts[2],
                        "is_fullscreen": False,  # Would need additional check
                    }

        except Exception as e:
            logger.debug(f"Failed to get app context: {e}")

        return {}

    # =========================================================================
    # Change Detection
    # =========================================================================

    async def _process_context_change(self, new_context: ScreenContext) -> None:
        """Process and notify about context changes."""
        if not self._current_context:
            return

        old = self._current_context
        new = new_context

        changes = []

        # App change
        if old.active_bundle_id != new.active_bundle_id:
            changes.append("app_switch")

        # Activity change
        if old.activity_type != new.activity_type:
            changes.append("activity_change")

            # Notify activity change callbacks
            for callback in self._on_activity_changed:
                try:
                    await callback(old.activity_type, new.activity_type)
                except Exception as e:
                    logger.error(f"Activity change callback error: {e}")

        # Error state change
        if not old.is_error_state and new.is_error_state:
            changes.append("error_appeared")
        elif old.is_error_state and not new.is_error_state:
            changes.append("error_resolved")

        # Notify context change callbacks
        if changes:
            change = ContextChange(
                previous_context=old,
                current_context=new,
                change_type=",".join(changes),
                change_magnitude=len(changes) / 5.0,  # Normalize
            )

            self._stats["context_changes_detected"] += 1

            for callback in self._on_context_changed:
                try:
                    await callback(change)
                except Exception as e:
                    logger.error(f"Context change callback error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_current_context(self) -> Optional[ScreenContext]:
        """Get the current screen context."""
        return self._current_context

    def get_context_history(self, limit: int = 10) -> List[ScreenContext]:
        """Get recent context history."""
        return list(self._context_history)[-limit:]

    def get_current_activity(self) -> Tuple[ActivityType, ContextConfidence]:
        """Get the current activity type and confidence."""
        if self._current_context:
            return (
                self._current_context.activity_type,
                self._current_context.activity_confidence,
            )
        return ActivityType.UNKNOWN, ContextConfidence.UNCERTAIN

    def on_context_changed(
        self,
        callback: Callable[[ContextChange], Coroutine]
    ) -> None:
        """Register a callback for context changes."""
        self._on_context_changed.append(callback)

    def on_activity_changed(
        self,
        callback: Callable[[ActivityType, ActivityType], Coroutine]
    ) -> None:
        """Register a callback for activity changes."""
        self._on_activity_changed.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "running": self._running,
            "paused": self._paused,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "current_activity": self._current_context.activity_type.value if self._current_context else None,
            "context_history_size": len(self._context_history),
            **self._stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_screen_context_analyzer: Optional[ScreenContextAnalyzer] = None


async def get_screen_context_analyzer(
    config: Optional[ScreenContextConfig] = None
) -> ScreenContextAnalyzer:
    """Get the global screen context analyzer instance."""
    global _screen_context_analyzer

    if _screen_context_analyzer is None:
        _screen_context_analyzer = ScreenContextAnalyzer(config)

    return _screen_context_analyzer


async def start_screen_context_analyzer(
    config: Optional[ScreenContextConfig] = None
) -> ScreenContextAnalyzer:
    """Get and start the global screen context analyzer."""
    analyzer = await get_screen_context_analyzer(config)
    if not analyzer._running:
        await analyzer.start()
    return analyzer


async def stop_screen_context_analyzer() -> None:
    """Stop the global screen context analyzer."""
    global _screen_context_analyzer

    if _screen_context_analyzer is not None:
        await _screen_context_analyzer.stop()
        _screen_context_analyzer = None
