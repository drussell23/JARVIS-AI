"""
Multi-Monitor Query Handler
============================

Handles multi-monitor specific queries:
- "What's on my second monitor?"
- "Show me all my displays"
- "Which monitor has the terminal?"
- "Move space 3 to the left monitor"

Architecture:
- Uses MultiMonitorManager for monitor detection and spatial resolution
- Uses CaptureStrategyManager for intelligent capture
- Uses OCRStrategyManager for OCR with fallbacks
- Uses ImplicitReferenceResolver for entity/reference resolution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Import managers
try:
    from context_intelligence.managers import (
        get_multi_monitor_manager,
        get_capture_strategy_manager,
        get_ocr_strategy_manager,
        MultiMonitorManager,
        CaptureStrategyManager,
        OCRStrategyManager,
        MonitorInfo,
        MonitorLayout
    )
    MULTI_MONITOR_AVAILABLE = True
except ImportError:
    MULTI_MONITOR_AVAILABLE = False
    get_multi_monitor_manager = lambda: None
    get_capture_strategy_manager = lambda: None
    get_ocr_strategy_manager = lambda: None
    logger.warning("Multi-monitor managers not available")

try:
    from backend.context_intelligence.resolvers import get_implicit_reference_resolver
    IMPLICIT_RESOLVER_AVAILABLE = True
except ImportError:
    IMPLICIT_RESOLVER_AVAILABLE = False
    get_implicit_reference_resolver = lambda: None
    logger.warning("ImplicitReferenceResolver not available")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MultiMonitorQueryType(Enum):
    """Multi-monitor query types"""
    MONITOR_CONTENT = "monitor_content"  # "What's on my second monitor?"
    LIST_DISPLAYS = "list_displays"      # "Show me all my displays"
    FIND_WINDOW = "find_window"          # "Which monitor has the terminal?"
    MOVE_SPACE = "move_space"            # "Move space 3 to the left monitor"


@dataclass
class SpaceContent:
    """Content of a space on a monitor"""
    space_id: int
    image: Optional[Any] = None
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    windows: List[str] = field(default_factory=list)
    capture_method: str = "unknown"
    ocr_method: str = "unknown"
    error: Optional[str] = None


@dataclass
class MonitorContentResult:
    """Result of monitor content query"""
    monitor: MonitorInfo
    spaces: List[SpaceContent]
    summary: str
    total_spaces: int
    success: bool
    error: Optional[str] = None


@dataclass
class DisplayListResult:
    """Result of display list query"""
    monitors: List[MonitorInfo]
    layout: MonitorLayout
    summary: str
    total_monitors: int
    total_spaces: int
    success: bool


@dataclass
class WindowLocationResult:
    """Result of window/app location query"""
    found: bool
    monitor: Optional[MonitorInfo] = None
    space_id: Optional[int] = None
    window_name: str = ""
    query: str = ""
    summary: str = ""
    error: Optional[str] = None


@dataclass
class MoveSpaceResult:
    """Result of space move operation"""
    success: bool
    space_id: int
    source_monitor: Optional[MonitorInfo] = None
    target_monitor: Optional[MonitorInfo] = None
    command_executed: str = ""
    summary: str = ""
    error: Optional[str] = None


# ============================================================================
# MULTI-MONITOR QUERY HANDLER
# ============================================================================

class MultiMonitorQueryHandler:
    """
    Handler for multi-monitor specific queries.

    Handles 4 query types:
    1. MONITOR_CONTENT: Show content of a specific monitor
    2. LIST_DISPLAYS: List all displays with info
    3. FIND_WINDOW: Find which monitor/space has a window
    4. MOVE_SPACE: Move a space to a different monitor
    """

    def __init__(
        self,
        multi_monitor_manager: Optional[MultiMonitorManager] = None,
        capture_manager: Optional[CaptureStrategyManager] = None,
        ocr_manager: Optional[OCRStrategyManager] = None,
        implicit_resolver: Optional[Any] = None
    ):
        """
        Initialize Multi-Monitor Query Handler.

        Args:
            multi_monitor_manager: Manager for multi-monitor operations
            capture_manager: Manager for intelligent capture
            ocr_manager: Manager for intelligent OCR
            implicit_resolver: Resolver for implicit references
        """
        self.multi_monitor_manager = multi_monitor_manager or get_multi_monitor_manager()
        self.capture_manager = capture_manager or get_capture_strategy_manager()
        self.ocr_manager = ocr_manager or get_ocr_strategy_manager()
        self.implicit_resolver = implicit_resolver or get_implicit_reference_resolver()

        logger.info("[MULTI-MONITOR-HANDLER] Initialized")
        logger.info(f"  Multi-Monitor Manager: {'✅' if self.multi_monitor_manager else '❌'}")
        logger.info(f"  Capture Manager: {'✅' if self.capture_manager else '❌'}")
        logger.info(f"  OCR Manager: {'✅' if self.ocr_manager else '❌'}")
        logger.info(f"  Implicit Resolver: {'✅' if self.implicit_resolver else '❌'}")

    async def process_query(
        self,
        query: str,
        query_type: MultiMonitorQueryType,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Process a multi-monitor query.

        Args:
            query: User query
            query_type: Type of multi-monitor query
            context: Optional context

        Returns:
            Query result based on type
        """
        logger.info(f"[MULTI-MONITOR-HANDLER] Processing {query_type.value} query: {query}")

        # Ensure multi-monitor manager is initialized
        if self.multi_monitor_manager:
            try:
                await self.multi_monitor_manager.initialize()
            except Exception as e:
                logger.warning(f"[MULTI-MONITOR-HANDLER] Could not initialize manager: {e}")

        # Route to appropriate handler
        if query_type == MultiMonitorQueryType.MONITOR_CONTENT:
            return await self.handle_monitor_content(query, context)
        elif query_type == MultiMonitorQueryType.LIST_DISPLAYS:
            return await self.handle_list_displays(query, context)
        elif query_type == MultiMonitorQueryType.FIND_WINDOW:
            return await self.handle_find_window(query, context)
        elif query_type == MultiMonitorQueryType.MOVE_SPACE:
            return await self.handle_move_space(query, context)
        else:
            logger.error(f"[MULTI-MONITOR-HANDLER] Unknown query type: {query_type}")
            return None

    async def handle_monitor_content(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MonitorContentResult:
        """
        Handle "What's on my second monitor?" queries.

        Args:
            query: User query
            context: Optional context

        Returns:
            MonitorContentResult with monitor content
        """
        logger.info(f"[MULTI-MONITOR-HANDLER] Handling monitor content query: {query}")

        try:
            # Step 1: Get monitor layout
            layout = await self.multi_monitor_manager.get_current_layout()

            if not layout.monitors:
                return MonitorContentResult(
                    monitor=None,
                    spaces=[],
                    summary="No monitors detected.",
                    total_spaces=0,
                    success=False,
                    error="No monitors found"
                )

            # Step 2: Resolve monitor reference from query
            monitor = await self.multi_monitor_manager.resolve_monitor_reference(
                query, layout.monitors, context
            )

            if not monitor:
                # Try to extract monitor reference with implicit resolver
                if self.implicit_resolver:
                    resolved = await self._resolve_monitor_with_implicit(query, layout.monitors, context)
                    monitor = resolved if resolved else layout.monitors[0]  # Default to first monitor
                else:
                    monitor = layout.monitors[0]  # Default to first monitor

            # Step 3: Capture all spaces on this monitor
            space_contents = await self._capture_monitor_spaces(monitor)

            # Step 4: Generate summary
            summary = self._generate_monitor_content_summary(monitor, space_contents)

            return MonitorContentResult(
                monitor=monitor,
                spaces=space_contents,
                summary=summary,
                total_spaces=len(space_contents),
                success=True
            )

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error in handle_monitor_content: {e}")
            return MonitorContentResult(
                monitor=None,
                spaces=[],
                summary=f"Error: {str(e)}",
                total_spaces=0,
                success=False,
                error=str(e)
            )

    async def handle_list_displays(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DisplayListResult:
        """
        Handle "Show me all my displays" queries.

        Args:
            query: User query
            context: Optional context

        Returns:
            DisplayListResult with all displays
        """
        logger.info(f"[MULTI-MONITOR-HANDLER] Handling list displays query: {query}")

        try:
            # Get current monitor layout
            layout = await self.multi_monitor_manager.get_current_layout()

            # Count total spaces across all monitors
            total_spaces = sum(len(m.spaces) for m in layout.monitors)

            # Generate summary
            summary = self._generate_display_list_summary(layout)

            return DisplayListResult(
                monitors=layout.monitors,
                layout=layout,
                summary=summary,
                total_monitors=len(layout.monitors),
                total_spaces=total_spaces,
                success=True
            )

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error in handle_list_displays: {e}")
            return DisplayListResult(
                monitors=[],
                layout=None,
                summary=f"Error: {str(e)}",
                total_monitors=0,
                total_spaces=0,
                success=False
            )

    async def handle_find_window(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WindowLocationResult:
        """
        Handle "Which monitor has the terminal?" queries.

        Args:
            query: User query
            context: Optional context

        Returns:
            WindowLocationResult with window location
        """
        logger.info(f"[MULTI-MONITOR-HANDLER] Handling find window query: {query}")

        try:
            # Step 1: Extract window/app name from query
            window_name = await self._extract_window_name(query, context)

            # Step 2: Get monitor layout
            layout = await self.multi_monitor_manager.get_current_layout()

            # Step 3: Search all monitors for the window
            for monitor in layout.monitors:
                for space_id in monitor.spaces:
                    # Capture and OCR the space
                    space_content = await self._capture_single_space(space_id)

                    if space_content and self._window_found_in_space(window_name, space_content):
                        summary = f"Found '{window_name}' on {monitor.name} (Space {space_id})"
                        return WindowLocationResult(
                            found=True,
                            monitor=monitor,
                            space_id=space_id,
                            window_name=window_name,
                            query=query,
                            summary=summary
                        )

            # Not found
            summary = f"Could not find '{window_name}' on any monitor"
            return WindowLocationResult(
                found=False,
                window_name=window_name,
                query=query,
                summary=summary
            )

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error in handle_find_window: {e}")
            return WindowLocationResult(
                found=False,
                query=query,
                summary=f"Error: {str(e)}",
                error=str(e)
            )

    async def handle_move_space(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MoveSpaceResult:
        """
        Handle "Move space 3 to the left monitor" queries.

        Args:
            query: User query
            context: Optional context

        Returns:
            MoveSpaceResult with move operation result
        """
        logger.info(f"[MULTI-MONITOR-HANDLER] Handling move space query: {query}")

        try:
            # Step 1: Extract space ID from query
            space_id = await self._extract_space_id(query, context)

            if space_id is None:
                return MoveSpaceResult(
                    success=False,
                    space_id=0,
                    summary="Could not determine which space to move",
                    error="Space ID not found in query"
                )

            # Step 2: Get monitor layout
            layout = await self.multi_monitor_manager.get_current_layout()

            # Step 3: Find current monitor for space
            source_monitor = None
            for monitor in layout.monitors:
                if space_id in monitor.spaces:
                    source_monitor = monitor
                    break

            # Step 4: Resolve target monitor reference
            target_monitor = await self.multi_monitor_manager.resolve_monitor_reference(
                query, layout.monitors, context
            )

            if not target_monitor:
                return MoveSpaceResult(
                    success=False,
                    space_id=space_id,
                    source_monitor=source_monitor,
                    summary="Could not determine target monitor",
                    error="Target monitor not found in query"
                )

            # Step 5: Execute move command
            result = await self._execute_space_move(space_id, target_monitor)

            return result

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error in handle_move_space: {e}")
            return MoveSpaceResult(
                success=False,
                space_id=0,
                summary=f"Error: {str(e)}",
                error=str(e)
            )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def _capture_monitor_spaces(self, monitor: MonitorInfo) -> List[SpaceContent]:
        """Capture all spaces on a monitor"""
        space_contents = []

        for space_id in monitor.spaces:
            content = await self._capture_single_space(space_id)
            if content:
                space_contents.append(content)

        return space_contents

    async def _capture_single_space(self, space_id: int) -> Optional[SpaceContent]:
        """Capture a single space with intelligent fallback"""
        try:
            # Try capture with fallbacks
            capture_result = await self.capture_manager.capture_with_fallback(
                space_id=space_id,
                max_attempts=3
            )

            if not capture_result.success or not capture_result.image:
                return SpaceContent(
                    space_id=space_id,
                    error="Capture failed"
                )

            # Try OCR with fallbacks
            ocr_result = await self.ocr_manager.ocr_with_fallback(
                image=capture_result.image,
                max_attempts=2
            )

            return SpaceContent(
                space_id=space_id,
                image=capture_result.image,
                ocr_text=ocr_result.text if ocr_result.success else "",
                ocr_confidence=ocr_result.confidence if ocr_result.success else 0.0,
                capture_method=capture_result.method,
                ocr_method=ocr_result.method if ocr_result.success else "none"
            )

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error capturing space {space_id}: {e}")
            return SpaceContent(
                space_id=space_id,
                error=str(e)
            )

    async def _resolve_monitor_with_implicit(
        self,
        query: str,
        monitors: List[MonitorInfo],
        context: Optional[Dict[str, Any]]
    ) -> Optional[MonitorInfo]:
        """Resolve monitor reference using implicit resolver"""
        try:
            # Use implicit resolver to extract monitor reference
            resolved = await self.implicit_resolver.resolve_reference(query, context)

            # Map resolved reference to monitor
            if resolved and 'monitor' in resolved:
                monitor_ref = resolved['monitor']

                # Try to match by name, ID, or position
                for monitor in monitors:
                    if (str(monitor.id) == str(monitor_ref) or
                        monitor.name.lower() == str(monitor_ref).lower()):
                        return monitor

            return None

        except Exception as e:
            logger.warning(f"[MULTI-MONITOR-HANDLER] Could not resolve with implicit resolver: {e}")
            return None

    async def _extract_window_name(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Extract window/app name from query"""
        # Common patterns
        query_lower = query.lower()

        # Try implicit resolver first
        if self.implicit_resolver:
            try:
                resolved = await self.implicit_resolver.resolve_reference(query, context)
                if resolved and 'entity' in resolved:
                    return resolved['entity']
            except Exception as e:
                logger.debug(f"[MULTI-MONITOR-HANDLER] Implicit resolver failed: {e}")

        # Fallback: Extract from patterns
        # "which monitor has the terminal?" -> "terminal"
        # "where is chrome?" -> "chrome"
        patterns = [
            r"has (?:the )?(\w+)",
            r"where is (?:the )?(\w+)",
            r"find (?:the )?(\w+)",
            r"locate (?:the )?(\w+)",
        ]

        import re
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)

        # Default: use last word
        words = query.split()
        return words[-1].strip('?.,!')

    async def _extract_space_id(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Extract space ID from query"""
        import re

        # Pattern: "move space 3", "space 5 to", etc.
        match = re.search(r'space\s+(\d+)', query.lower())
        if match:
            return int(match.group(1))

        return None

    def _window_found_in_space(self, window_name: str, space_content: SpaceContent) -> bool:
        """Check if window is found in space content"""
        if not space_content.ocr_text:
            return False

        window_name_lower = window_name.lower()
        ocr_text_lower = space_content.ocr_text.lower()

        return window_name_lower in ocr_text_lower

    async def _execute_space_move(
        self,
        space_id: int,
        target_monitor: MonitorInfo
    ) -> MoveSpaceResult:
        """Execute yabai command to move space to monitor"""
        try:
            # Get target display ID
            display_id = target_monitor.id

            # Construct yabai command
            command = f"yabai -m space {space_id} --display {display_id}"

            # Execute command
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5.0
            )

            success = result.returncode == 0

            if success:
                summary = f"Successfully moved Space {space_id} to {target_monitor.name}"
            else:
                summary = f"Failed to move space: {result.stderr}"

            return MoveSpaceResult(
                success=success,
                space_id=space_id,
                target_monitor=target_monitor,
                command_executed=command,
                summary=summary,
                error=result.stderr if not success else None
            )

        except Exception as e:
            logger.error(f"[MULTI-MONITOR-HANDLER] Error executing move: {e}")
            return MoveSpaceResult(
                success=False,
                space_id=space_id,
                target_monitor=target_monitor,
                summary=f"Error executing move: {str(e)}",
                error=str(e)
            )

    def _generate_monitor_content_summary(
        self,
        monitor: MonitorInfo,
        spaces: List[SpaceContent]
    ) -> str:
        """Generate summary of monitor content"""
        lines = [
            f"Monitor: {monitor.name}",
            f"Resolution: {monitor.resolution[0]}x{monitor.resolution[1]}",
            f"Spaces: {len(spaces)}",
            ""
        ]

        for space in spaces:
            if space.error:
                lines.append(f"  Space {space.space_id}: ❌ {space.error}")
            else:
                confidence_icon = "✅" if space.ocr_confidence > 0.8 else "⚠️"
                preview = space.ocr_text[:100] + "..." if space.ocr_text and len(space.ocr_text) > 100 else space.ocr_text or ""
                lines.append(f"  Space {space.space_id}: {confidence_icon} {preview}")

        return "\n".join(lines)

    def _generate_display_list_summary(self, layout: MonitorLayout) -> str:
        """Generate summary of all displays"""
        lines = [f"Total Monitors: {len(layout.monitors)}", ""]

        for i, monitor in enumerate(layout.monitors, 1):
            main_indicator = "⭐ MAIN" if monitor.is_main else ""
            lines.append(f"{i}. {monitor.name} {main_indicator}")
            lines.append(f"   Resolution: {monitor.resolution[0]}x{monitor.resolution[1]}")
            lines.append(f"   Position: {monitor.position}")
            lines.append(f"   Spaces: {', '.join(str(s) for s in monitor.spaces)}")

            if monitor.relative_positions:
                positions = ', '.join(p.value for p in monitor.relative_positions)
                lines.append(f"   Relative: {positions}")

            lines.append("")

        return "\n".join(lines)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_handler: Optional[MultiMonitorQueryHandler] = None


def get_multi_monitor_query_handler() -> Optional[MultiMonitorQueryHandler]:
    """Get the global multi-monitor query handler instance"""
    return _global_handler


def initialize_multi_monitor_query_handler(
    multi_monitor_manager: Optional[MultiMonitorManager] = None,
    capture_manager: Optional[CaptureStrategyManager] = None,
    ocr_manager: Optional[OCRStrategyManager] = None,
    implicit_resolver: Optional[Any] = None
) -> MultiMonitorQueryHandler:
    """
    Initialize the global MultiMonitorQueryHandler instance.

    Args:
        multi_monitor_manager: MultiMonitorManager instance
        capture_manager: CaptureStrategyManager instance
        ocr_manager: OCRStrategyManager instance
        implicit_resolver: ImplicitReferenceResolver instance

    Returns:
        MultiMonitorQueryHandler instance
    """
    global _global_handler

    _global_handler = MultiMonitorQueryHandler(
        multi_monitor_manager=multi_monitor_manager,
        capture_manager=capture_manager,
        ocr_manager=ocr_manager,
        implicit_resolver=implicit_resolver
    )

    logger.info("[MULTI-MONITOR-HANDLER] Global instance initialized")
    return _global_handler
