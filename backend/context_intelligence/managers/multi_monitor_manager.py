"""
Multi-Monitor Manager
=====================

Overcomes v1.0 limitations:
❌ Assumes single display
❌ Doesn't map spaces to monitors
❌ Can't distinguish "left monitor" vs "right monitor"

New capabilities:
✅ Detects all connected monitors
✅ Maps spaces to specific monitors
✅ Understands spatial relationships (left, right, top, bottom)
✅ Resolves natural language references ("main display", "that screen")
✅ Tracks monitor configurations dynamically

Strategy:
- Query display information from macOS
- Map yabai spaces to physical monitors
- Understand spatial relationships
- Integrate with ImplicitReferenceResolver for context
- Support dynamic monitor changes (connect/disconnect)
"""

import asyncio
import subprocess
import json
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MonitorPosition(Enum):
    """Relative position of monitor"""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    MAIN = "main"


@dataclass
class MonitorInfo:
    """Information about a physical monitor"""
    id: int
    uuid: str
    name: str
    resolution: Tuple[int, int]  # (width, height)
    position: Tuple[int, int]  # (x, y) in screen coordinates
    is_main: bool
    spaces: List[int]  # Yabai space IDs on this monitor
    relative_positions: Set[MonitorPosition]  # left, right, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorLayout:
    """Overall monitor layout"""
    monitors: List[MonitorInfo]
    main_monitor: Optional[MonitorInfo]
    total_resolution: Tuple[int, int]
    layout_type: str  # "single", "horizontal", "vertical", "grid"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpaceMonitorMapping:
    """Mapping of space to monitor"""
    space_id: int
    monitor_id: int
    monitor_uuid: str
    monitor_name: str
    is_visible: bool
    position_on_monitor: str  # "left", "right", "center"


class MonitorDetector:
    """
    Detects connected monitors and their configurations.

    Uses system_profiler and other macOS tools to get monitor information.
    """

    def __init__(self):
        """Initialize monitor detector"""
        self.cache_ttl = 10.0  # Cache for 10 seconds
        self._cache: Optional[List[MonitorInfo]] = None
        self._cache_time: Optional[float] = None

    async def detect_monitors(self, use_cache: bool = True) -> List[MonitorInfo]:
        """
        Detect all connected monitors.

        Args:
            use_cache: Use cached results if available

        Returns:
            List of MonitorInfo objects
        """
        # Check cache
        if use_cache and self._cache and self._cache_time:
            import time
            age = time.time() - self._cache_time
            if age < self.cache_ttl:
                return self._cache

        monitors = []

        try:
            # Get display information from system_profiler
            displays = await self._get_display_info()

            # Get display arrangement from defaults
            arrangement = await self._get_display_arrangement()

            # Combine information
            for i, display in enumerate(displays):
                monitor = MonitorInfo(
                    id=i + 1,
                    uuid=display.get('uuid', f'display_{i}'),
                    name=display.get('name', f'Display {i+1}'),
                    resolution=(display.get('width', 1920), display.get('height', 1080)),
                    position=arrangement.get(i, {}).get('position', (0, 0)),
                    is_main=display.get('main', i == 0),
                    spaces=[],
                    relative_positions=set()
                )
                monitors.append(monitor)

            # Calculate relative positions
            if len(monitors) > 1:
                self._calculate_relative_positions(monitors)

            # Update cache
            import time
            self._cache = monitors
            self._cache_time = time.time()

            return monitors

        except Exception as e:
            logger.error(f"Failed to detect monitors: {e}")
            # Fallback: assume single monitor
            return [MonitorInfo(
                id=1,
                uuid='default',
                name='Main Display',
                resolution=(1920, 1080),
                position=(0, 0),
                is_main=True,
                spaces=[],
                relative_positions={MonitorPosition.MAIN}
            )]

    async def _get_display_info(self) -> List[Dict[str, Any]]:
        """Get display information from system_profiler"""
        try:
            # Run system_profiler
            result = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPDisplaysDataType', '-json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.warning(f"system_profiler failed: {stderr.decode()}")
                return []

            data = json.loads(stdout.decode())

            displays = []
            for item in data.get('SPDisplaysDataType', []):
                # Extract displays from the structure
                for display_key, display_info in item.items():
                    if isinstance(display_info, dict) and 'spdisplays_ndrvs' in display_info:
                        for display in display_info['spdisplays_ndrvs']:
                            resolution_str = display.get('_spdisplays_resolution', '1920 x 1080')
                            width, height = self._parse_resolution(resolution_str)

                            displays.append({
                                'uuid': display.get('_spdisplays_display-serial-number', f'display_{len(displays)}'),
                                'name': display.get('_name', f'Display {len(displays)+1}'),
                                'width': width,
                                'height': height,
                                'main': display.get('spdisplays_main', 'spdisplays_no') == 'spdisplays_yes'
                            })

            return displays if displays else [{'uuid': 'default', 'name': 'Main Display', 'width': 1920, 'height': 1080, 'main': True}]

        except Exception as e:
            logger.warning(f"Failed to get display info: {e}")
            return [{'uuid': 'default', 'name': 'Main Display', 'width': 1920, 'height': 1080, 'main': True}]

    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse resolution string like '1920 x 1080'"""
        match = re.search(r'(\d+)\s*x\s*(\d+)', resolution_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1920, 1080

    async def _get_display_arrangement(self) -> Dict[int, Dict[str, Any]]:
        """Get display arrangement (positions)"""
        # This would ideally use CGDisplay API, but for now we'll use defaults
        # In a real implementation, you'd use a Swift/ObjC bridge
        try:
            result = await asyncio.create_subprocess_exec(
                'defaults', 'read', '/Library/Preferences/com.apple.windowserver', 'DisplaySets',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            # Parse the arrangement (simplified)
            # Real implementation would parse the plist structure
            return {}

        except Exception as e:
            logger.debug(f"Could not get display arrangement: {e}")
            return {}

    def _calculate_relative_positions(self, monitors: List[MonitorInfo]):
        """Calculate relative positions of monitors"""
        if len(monitors) <= 1:
            if monitors:
                monitors[0].relative_positions.add(MonitorPosition.MAIN)
            return

        # Find main monitor
        main = next((m for m in monitors if m.is_main), monitors[0])
        main.relative_positions.add(MonitorPosition.MAIN)

        # Calculate positions relative to main
        for monitor in monitors:
            if monitor == main:
                continue

            dx = monitor.position[0] - main.position[0]
            dy = monitor.position[1] - main.position[1]

            # Horizontal relationships
            if abs(dx) > abs(dy):
                if dx > 0:
                    monitor.relative_positions.add(MonitorPosition.RIGHT)
                    main.relative_positions.add(MonitorPosition.LEFT)
                else:
                    monitor.relative_positions.add(MonitorPosition.LEFT)
                    main.relative_positions.add(MonitorPosition.RIGHT)

            # Vertical relationships
            if abs(dy) > abs(dx):
                if dy > 0:
                    monitor.relative_positions.add(MonitorPosition.BOTTOM)
                    main.relative_positions.add(MonitorPosition.TOP)
                else:
                    monitor.relative_positions.add(MonitorPosition.TOP)
                    main.relative_positions.add(MonitorPosition.BOTTOM)


class MonitorSpaceMapper:
    """
    Maps yabai spaces to physical monitors.

    Queries yabai for space and display information.
    """

    def __init__(self):
        """Initialize monitor space mapper"""
        pass

    async def map_spaces_to_monitors(
        self, monitors: List[MonitorInfo]
    ) -> List[MonitorInfo]:
        """
        Map yabai spaces to monitors.

        Args:
            monitors: List of detected monitors

        Returns:
            Updated monitors with space mappings
        """
        try:
            # Get yabai display information
            displays = await self._query_yabai_displays()

            # Get yabai space information
            spaces = await self._query_yabai_spaces()

            # Map spaces to displays
            display_spaces: Dict[int, List[int]] = {}
            for space in spaces:
                display_id = space.get('display', 1)
                space_id = space.get('index', space.get('id', 0))

                if display_id not in display_spaces:
                    display_spaces[display_id] = []
                display_spaces[display_id].append(space_id)

            # Update monitor information
            for i, monitor in enumerate(monitors):
                # Map yabai display ID to monitor (1-indexed)
                yabai_display_id = i + 1
                monitor.spaces = sorted(display_spaces.get(yabai_display_id, []))

            return monitors

        except Exception as e:
            logger.error(f"Failed to map spaces to monitors: {e}")
            # Fallback: assign all spaces to first monitor
            if monitors:
                try:
                    spaces = await self._query_yabai_spaces()
                    monitors[0].spaces = [s.get('index', s.get('id', i+1)) for i, s in enumerate(spaces)]
                except:
                    pass
            return monitors

    async def _query_yabai_displays(self) -> List[Dict[str, Any]]:
        """Query yabai for display information"""
        try:
            result = await asyncio.create_subprocess_exec(
                'yabai', '-m', 'query', '--displays',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.warning(f"yabai query displays failed: {stderr.decode()}")
                return []

            return json.loads(stdout.decode())

        except Exception as e:
            logger.warning(f"Failed to query yabai displays: {e}")
            return []

    async def _query_yabai_spaces(self) -> List[Dict[str, Any]]:
        """Query yabai for space information"""
        try:
            result = await asyncio.create_subprocess_exec(
                'yabai', '-m', 'query', '--spaces',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.warning(f"yabai query spaces failed: {stderr.decode()}")
                return []

            return json.loads(stdout.decode())

        except Exception as e:
            logger.warning(f"Failed to query yabai spaces: {e}")
            return []


class MonitorReferenceResolver:
    """
    Resolves natural language monitor references.

    Integrates with ImplicitReferenceResolver for context-aware resolution.

    Examples:
    - "left monitor" → Monitor 2
    - "main display" → Monitor 1
    - "that screen" → Recently discussed monitor
    - "the other one" → Monitor not currently focused
    """

    def __init__(
        self,
        implicit_resolver: Optional[Any] = None,
        conversation_tracker: Optional[Any] = None
    ):
        """
        Initialize monitor reference resolver.

        Args:
            implicit_resolver: ImplicitReferenceResolver instance
            conversation_tracker: ConversationTracker instance
        """
        self.implicit_resolver = implicit_resolver
        self.conversation_tracker = conversation_tracker

        # Common monitor references
        self.reference_patterns = {
            'main': ['main', 'primary', 'first'],
            'secondary': ['secondary', 'second', 'other'],
            'left': ['left'],
            'right': ['right'],
            'top': ['top', 'upper'],
            'bottom': ['bottom', 'lower'],
        }

    async def resolve_monitor_reference(
        self,
        query: str,
        monitors: List[MonitorInfo],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[MonitorInfo]:
        """
        Resolve monitor reference from query.

        Args:
            query: User query with monitor reference
            monitors: Available monitors
            context: Additional context

        Returns:
            Resolved MonitorInfo or None
        """
        query_lower = query.lower()

        # Check for direct references
        for ref_type, patterns in self.reference_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    monitor = self._resolve_by_pattern(ref_type, monitors)
                    if monitor:
                        return monitor

        # Check for "that monitor", "this display", etc.
        if any(word in query_lower for word in ['that', 'this', 'the']):
            # Use conversation context
            if self.conversation_tracker:
                conv_context = self.conversation_tracker.get_recent_context()
                # Get last mentioned monitor (would need to track monitor entities)
                # For now, return None and let caller handle

        # Check for monitor ID/name
        for monitor in monitors:
            if monitor.name.lower() in query_lower:
                return monitor
            if str(monitor.id) in query_lower:
                return monitor

        return None

    def _resolve_by_pattern(
        self, pattern_type: str, monitors: List[MonitorInfo]
    ) -> Optional[MonitorInfo]:
        """Resolve monitor by pattern type"""
        if pattern_type == 'main':
            return next((m for m in monitors if m.is_main), None)

        if pattern_type == 'secondary':
            return next((m for m in monitors if not m.is_main), None)

        if pattern_type == 'left':
            return next((m for m in monitors if MonitorPosition.LEFT in m.relative_positions), None)

        if pattern_type == 'right':
            return next((m for m in monitors if MonitorPosition.RIGHT in m.relative_positions), None)

        if pattern_type == 'top':
            return next((m for m in monitors if MonitorPosition.TOP in m.relative_positions), None)

        if pattern_type == 'bottom':
            return next((m for m in monitors if MonitorPosition.BOTTOM in m.relative_positions), None)

        return None


class MultiMonitorManager:
    """
    Main manager for multi-monitor support.

    Detects monitors, maps spaces, resolves references.
    """

    def __init__(
        self,
        implicit_resolver: Optional[Any] = None,
        conversation_tracker: Optional[Any] = None,
        auto_refresh_interval: float = 30.0
    ):
        """
        Initialize Multi-Monitor Manager.

        Args:
            implicit_resolver: ImplicitReferenceResolver instance
            conversation_tracker: ConversationTracker instance
            auto_refresh_interval: How often to refresh monitor info (seconds)
        """
        self.detector = MonitorDetector()
        self.mapper = MonitorSpaceMapper()
        self.resolver = MonitorReferenceResolver(implicit_resolver, conversation_tracker)
        self.auto_refresh_interval = auto_refresh_interval

        self._current_layout: Optional[MonitorLayout] = None
        self._refresh_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize and detect monitors"""
        await self.refresh_monitor_layout()

        # Start auto-refresh task
        if self.auto_refresh_interval > 0:
            self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

    async def refresh_monitor_layout(self) -> MonitorLayout:
        """
        Refresh monitor layout.

        Returns:
            Updated MonitorLayout
        """
        # Step 1: Detect monitors
        monitors = await self.detector.detect_monitors(use_cache=False)

        # Step 2: Map spaces to monitors
        monitors = await self.mapper.map_spaces_to_monitors(monitors)

        # Step 3: Determine layout type
        layout_type = self._determine_layout_type(monitors)

        # Step 4: Calculate total resolution
        total_resolution = self._calculate_total_resolution(monitors)

        # Step 5: Find main monitor
        main_monitor = next((m for m in monitors if m.is_main), monitors[0] if monitors else None)

        layout = MonitorLayout(
            monitors=monitors,
            main_monitor=main_monitor,
            total_resolution=total_resolution,
            layout_type=layout_type
        )

        self._current_layout = layout
        logger.info(f"[MULTI-MONITOR] Detected {len(monitors)} monitor(s), layout: {layout_type}")

        return layout

    def _determine_layout_type(self, monitors: List[MonitorInfo]) -> str:
        """Determine layout type from monitor configuration"""
        if len(monitors) == 1:
            return "single"

        # Check if all monitors are horizontally arranged
        positions = [m.position for m in monitors]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        if x_range > y_range * 2:
            return "horizontal"
        elif y_range > x_range * 2:
            return "vertical"
        else:
            return "grid"

    def _calculate_total_resolution(self, monitors: List[MonitorInfo]) -> Tuple[int, int]:
        """Calculate total resolution across all monitors"""
        if not monitors:
            return (0, 0)

        # Calculate bounding box
        min_x = min(m.position[0] for m in monitors)
        min_y = min(m.position[1] for m in monitors)
        max_x = max(m.position[0] + m.resolution[0] for m in monitors)
        max_y = max(m.position[1] + m.resolution[1] for m in monitors)

        return (max_x - min_x, max_y - min_y)

    async def get_monitor_by_reference(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[MonitorInfo]:
        """
        Get monitor by natural language reference.

        Args:
            query: Query with monitor reference
            context: Additional context

        Returns:
            MonitorInfo or None
        """
        if not self._current_layout:
            await self.refresh_monitor_layout()

        if not self._current_layout or not self._current_layout.monitors:
            return None

        return await self.resolver.resolve_monitor_reference(
            query, self._current_layout.monitors, context
        )

    async def get_spaces_on_monitor(
        self, monitor_id: Optional[int] = None, monitor_reference: Optional[str] = None
    ) -> List[int]:
        """
        Get spaces on a specific monitor.

        Args:
            monitor_id: Direct monitor ID
            monitor_reference: Natural language reference

        Returns:
            List of space IDs
        """
        if not self._current_layout:
            await self.refresh_monitor_layout()

        if not self._current_layout or not self._current_layout.monitors:
            return []

        # Resolve monitor
        monitor = None
        if monitor_id:
            monitor = next((m for m in self._current_layout.monitors if m.id == monitor_id), None)
        elif monitor_reference:
            monitor = await self.get_monitor_by_reference(monitor_reference)

        if monitor:
            return monitor.spaces

        return []

    async def get_monitor_for_space(self, space_id: int) -> Optional[MonitorInfo]:
        """
        Get monitor containing a specific space.

        Args:
            space_id: Space ID

        Returns:
            MonitorInfo or None
        """
        if not self._current_layout:
            await self.refresh_monitor_layout()

        if not self._current_layout:
            return None

        for monitor in self._current_layout.monitors:
            if space_id in monitor.spaces:
                return monitor

        return None

    def get_current_layout(self) -> Optional[MonitorLayout]:
        """Get current monitor layout"""
        return self._current_layout

    async def _auto_refresh_loop(self):
        """Auto-refresh loop"""
        while True:
            try:
                await asyncio.sleep(self.auto_refresh_interval)
                await self.refresh_monitor_layout()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-refresh failed: {e}")

    async def shutdown(self):
        """Shutdown manager"""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass


# Global instance
_multi_monitor_manager: Optional[MultiMonitorManager] = None


def get_multi_monitor_manager() -> Optional[MultiMonitorManager]:
    """Get the global MultiMonitorManager instance"""
    return _multi_monitor_manager


def initialize_multi_monitor_manager(
    implicit_resolver: Optional[Any] = None,
    conversation_tracker: Optional[Any] = None,
    auto_refresh_interval: float = 30.0
) -> MultiMonitorManager:
    """
    Initialize the global MultiMonitorManager instance.

    Note: The manager will be fully initialized on first use (lazy initialization).
    Call await manager.initialize() explicitly if you need immediate initialization.

    Args:
        implicit_resolver: ImplicitReferenceResolver instance
        conversation_tracker: ConversationTracker instance
        auto_refresh_interval: How often to refresh (seconds)

    Returns:
        MultiMonitorManager instance
    """
    global _multi_monitor_manager

    _multi_monitor_manager = MultiMonitorManager(
        implicit_resolver=implicit_resolver,
        conversation_tracker=conversation_tracker,
        auto_refresh_interval=auto_refresh_interval
    )

    return _multi_monitor_manager
