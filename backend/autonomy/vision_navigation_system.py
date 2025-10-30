#!/usr/bin/env python3
"""
Full Screen Vision & Navigation System for JARVIS.

This module provides complete workspace understanding and autonomous navigation capabilities
for the JARVIS AI assistant. It combines computer vision, OCR, window management, and
intelligent navigation to enable full control over the desktop environment.

The system can:
- Map the entire workspace including all windows and UI elements
- Navigate between applications and UI components autonomously
- Learn navigation patterns and optimize workflows
- Arrange windows in predefined layouts
- Execute complex multi-step workflows

Example:
    >>> nav_system = VisionNavigationSystem()
    >>> await nav_system.start_navigation_mode()
    >>> await nav_system.navigate_to_application("Safari")
    >>> await nav_system.arrange_workspace(WorkspaceLayout.SPLIT)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

# System integration imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Vision and workspace imports
from vision.screen_capture_module import ScreenCaptureModule, ScreenCapture
from vision.ocr_processor import OCRProcessor, OCRResult, TextRegion
from vision.window_analysis import WindowAnalyzer, WindowContent, ApplicationCategory
from vision.window_detector import WindowDetector, WindowInfo
from vision.enhanced_monitoring import EnhancedWorkspaceMonitor

# macOS integration
try:
    import Quartz
    import AppKit
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGNullWindowID
    from AppKit import NSWorkspace, NSRunningApplication
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class NavigationAction(Enum):
    """Types of navigation actions that can be performed.
    
    Attributes:
        SWITCH_WINDOW: Switch focus to a different window
        OPEN_APPLICATION: Launch a new application
        CLOSE_WINDOW: Close an existing window
        MINIMIZE_WINDOW: Minimize a window to dock
        MAXIMIZE_WINDOW: Maximize a window to full screen
        MOVE_WINDOW: Move a window to new position
        RESIZE_WINDOW: Resize a window
        CLICK_ELEMENT: Click on a UI element
        TYPE_TEXT: Type text at current cursor position
        SCROLL: Scroll within a window or element
        FOCUS_ELEMENT: Set focus to a specific element
        NAVIGATE_MENU: Navigate through menu structures
        SWITCH_DESKTOP: Switch to different desktop/space
        ARRANGE_WINDOWS: Arrange multiple windows in layout
    """
    SWITCH_WINDOW = "switch_window"
    OPEN_APPLICATION = "open_application"
    CLOSE_WINDOW = "close_window"
    MINIMIZE_WINDOW = "minimize_window"
    MAXIMIZE_WINDOW = "maximize_window"
    MOVE_WINDOW = "move_window"
    RESIZE_WINDOW = "resize_window"
    CLICK_ELEMENT = "click_element"
    TYPE_TEXT = "type_text"
    SCROLL = "scroll"
    FOCUS_ELEMENT = "focus_element"
    NAVIGATE_MENU = "navigate_menu"
    SWITCH_DESKTOP = "switch_desktop"
    ARRANGE_WINDOWS = "arrange_windows"

class WorkspaceLayout(Enum):
    """Predefined workspace layouts for window arrangement.
    
    Attributes:
        FOCUS: Single window maximized for focused work
        SPLIT: Two windows side by side
        GRID: Four windows arranged in 2x2 grid
        CASCADE: Windows cascaded with slight offsets
        CUSTOM: User-defined custom layout
    """
    FOCUS = "focus"  # Single window maximized
    SPLIT = "split"  # Two windows side by side
    GRID = "grid"   # Four windows in grid
    CASCADE = "cascade"  # Windows cascaded
    CUSTOM = "custom"  # User-defined layout

@dataclass
class WorkspaceElement:
    """Represents a UI element in the workspace.
    
    This class encapsulates all information about a UI element including its
    position, type, content, and relationships to other elements.
    
    Attributes:
        id: Unique identifier for the element
        type: Type of element (window, button, menu, text_field, etc.)
        bounds: Element boundaries as (x, y, width, height)
        parent_window: ID of the parent window containing this element
        text: Text content of the element if applicable
        is_interactive: Whether the element can be interacted with
        is_focused: Whether the element currently has focus
        children: List of child elements contained within this element
        properties: Additional properties and metadata
    """
    id: str
    type: str  # window, button, menu, text_field, etc.
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    parent_window: Optional[str] = None
    text: Optional[str] = None
    is_interactive: bool = False
    is_focused: bool = False
    children: List['WorkspaceElement'] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within the element's bounds.
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
            
        Returns:
            True if point is within element bounds, False otherwise
            
        Example:
            >>> element = WorkspaceElement("btn1", "button", (10, 10, 100, 50))
            >>> element.contains_point(50, 30)
            True
        """
        ex, ey, ew, eh = self.bounds
        return ex <= x <= ex + ew and ey <= y <= ey + eh
    
    def center_point(self) -> Tuple[int, int]:
        """Get the center point of the element.
        
        Returns:
            Tuple of (x, y) coordinates for the element's center
            
        Example:
            >>> element = WorkspaceElement("btn1", "button", (10, 10, 100, 50))
            >>> element.center_point()
            (60, 35)
        """
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)

@dataclass
class WorkspaceMap:
    """Complete map of the current workspace state.
    
    This class represents a snapshot of the entire workspace including all
    windows, UI elements, and their relationships at a specific point in time.
    
    Attributes:
        windows: List of all visible windows
        elements: List of all UI elements across all windows
        active_window: Currently focused/active window
        screen_bounds: Screen boundaries as (x, y, width, height)
        desktop_number: Current desktop/space number
        timestamp: When this map was created
    """
    windows: List[WindowInfo]
    elements: List[WorkspaceElement]
    active_window: Optional[WindowInfo] = None
    screen_bounds: Tuple[int, int, int, int] = (0, 0, 1920, 1080)
    desktop_number: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    def find_element_at(self, x: int, y: int) -> Optional[WorkspaceElement]:
        """Find the topmost element at given coordinates.
        
        Args:
            x: X coordinate to search
            y: Y coordinate to search
            
        Returns:
            The topmost element at the coordinates, or None if no element found
            
        Example:
            >>> workspace_map.find_element_at(100, 200)
            WorkspaceElement(id="button_1", type="button", ...)
        """
        for element in reversed(self.elements):  # Check from top to bottom
            if element.contains_point(x, y):
                return element
        return None
    
    def get_window_elements(self, window_id: str) -> List[WorkspaceElement]:
        """Get all elements belonging to a specific window.
        
        Args:
            window_id: ID of the window to get elements for
            
        Returns:
            List of elements that belong to the specified window
            
        Example:
            >>> elements = workspace_map.get_window_elements("window_123")
            >>> len(elements)
            15
        """
        return [e for e in self.elements if e.parent_window == window_id]

@dataclass
class NavigationPath:
    """Represents a path for navigating between UI elements.
    
    This class encapsulates the sequence of actions needed to navigate from
    one element to another, along with metadata about the path's efficiency.
    
    Attributes:
        start: Starting element for navigation
        end: Target element for navigation
        steps: List of navigation steps to execute
        confidence: Confidence score for path success (0.0-1.0)
        estimated_time: Estimated time to complete navigation in seconds
    """
    start: WorkspaceElement
    end: WorkspaceElement
    steps: List[NavigationAction]
    confidence: float = 0.0
    estimated_time: float = 0.0
    
    def add_step(self, action: NavigationAction, params: Dict[str, Any] = None):
        """Add a navigation step to the path.
        
        Args:
            action: The navigation action to perform
            params: Parameters for the action
            
        Example:
            >>> path.add_step(NavigationAction.CLICK_ELEMENT, {"coordinates": (100, 200)})
        """
        self.steps.append({
            'action': action,
            'params': params or {},
            'timestamp': datetime.now()
        })

class VisionNavigationSystem:
    """Full screen vision and navigation system for autonomous desktop control.
    
    This system provides complete workspace understanding and autonomous navigation
    capabilities by combining computer vision, OCR, window management, and intelligent
    navigation algorithms. It can map the entire workspace, navigate between applications,
    learn usage patterns, and execute complex workflows.
    
    The system operates in two modes:
    1. Passive monitoring: Observes workspace without taking actions
    2. Active navigation: Can control mouse, keyboard, and window management
    
    Attributes:
        screen_capture: Module for capturing screen content
        ocr_processor: OCR processor for text extraction
        window_analyzer: Analyzer for window content and structure
        window_detector: Detector for finding and tracking windows
        enhanced_monitor: Enhanced workspace monitoring capabilities
        current_map: Current workspace map snapshot
        element_cache: Cache of UI elements for fast lookup
        window_relationships: Relationships between windows
        navigation_history: History of navigation actions
        current_focus: Currently focused element
        navigation_mode: Whether active navigation is enabled
        navigation_patterns: Learned navigation patterns
        element_interactions: Count of interactions per element
        workflow_sequences: Recorded workflow sequences
        navigation_stats: Performance statistics
    """
    
    def __init__(self):
        """Initialize the vision navigation system.
        
        Sets up all vision components, workspace mapping structures, and
        navigation state tracking.
        """
        # Vision components
        self.screen_capture = ScreenCaptureModule(capture_interval=0.5)  # Faster for navigation
        self.ocr_processor = OCRProcessor()
        self.window_analyzer = WindowAnalyzer()
        self.window_detector = WindowDetector()
        self.enhanced_monitor = EnhancedWorkspaceMonitor()
        
        # Workspace mapping
        self.current_map: Optional[WorkspaceMap] = None
        self.element_cache: Dict[str, WorkspaceElement] = {}
        self.window_relationships: Dict[str, List[str]] = defaultdict(list)
        
        # Navigation state
        self.navigation_history: List[NavigationAction] = []
        self.current_focus: Optional[WorkspaceElement] = None
        self.navigation_mode: bool = False
        
        # Learning system
        self.navigation_patterns: Dict[str, List[NavigationPath]] = defaultdict(list)
        self.element_interactions: Dict[str, int] = defaultdict(int)
        self.workflow_sequences: List[List[NavigationAction]] = []
        
        # Performance tracking
        self.navigation_stats = {
            'total_navigations': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'average_time': 0.0
        }
        
    async def start_navigation_mode(self):
        """Enable navigation mode for active workspace control.
        
        When navigation mode is active, the system can perform actions like
        clicking, typing, and window management. This starts continuous
        workspace mapping for real-time navigation.
        
        Raises:
            RuntimeError: If system lacks necessary permissions for navigation
            
        Example:
            >>> nav_system = VisionNavigationSystem()
            >>> await nav_system.start_navigation_mode()
        """
        self.navigation_mode = True
        logger.info("Navigation mode activated - JARVIS has full workspace control")
        
        # Start continuous workspace mapping
        asyncio.create_task(self._continuous_mapping())
        
    async def stop_navigation_mode(self):
        """Disable navigation mode and stop active control.
        
        Stops continuous mapping and disables the ability to perform
        navigation actions. The system returns to passive monitoring only.
        
        Example:
            >>> await nav_system.stop_navigation_mode()
        """
        self.navigation_mode = False
        logger.info("Navigation mode deactivated")
        
    async def _continuous_mapping(self):
        """Continuously map the workspace while navigation mode is active.
        
        This internal method runs in the background to maintain an up-to-date
        map of the workspace, analyzing window relationships and updating
        the element cache every 500ms.
        
        Raises:
            Exception: Logs errors but continues operation
        """
        while self.navigation_mode:
            try:
                # Update workspace map
                self.current_map = await self.map_full_workspace()
                
                # Analyze relationships
                await self._analyze_window_relationships()
                
                # Update element cache
                self._update_element_cache()
                
                await asyncio.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                logger.error(f"Error in continuous mapping: {e}")
                await asyncio.sleep(1)
    
    async def map_full_workspace(self) -> WorkspaceMap:
        """Create a complete map of the current workspace.
        
        Captures the full screen, detects all windows, extracts UI elements,
        and builds a comprehensive map of the workspace state.
        
        Returns:
            WorkspaceMap containing all windows, elements, and workspace state
            
        Raises:
            Exception: If screen capture or window detection fails
            
        Example:
            >>> workspace_map = await nav_system.map_full_workspace()
            >>> print(f"Found {len(workspace_map.windows)} windows")
        """
        # Capture full screen
        screen_capture = self.screen_capture.capture_screen()
        
        # Get all windows
        windows = self.window_detector.get_all_windows()
        
        # Extract all UI elements
        elements = await self._extract_all_elements(screen_capture, windows)
        
        # Find active window
        active_window = next((w for w in windows if w.is_focused), None)
        
        # Get screen bounds
        screen_bounds = self._get_screen_bounds()
        
        return WorkspaceMap(
            windows=windows,
            elements=elements,
            active_window=active_window,
            screen_bounds=screen_bounds,
            desktop_number=self._get_current_desktop()
        )
    
    async def _extract_all_elements(self, screen_capture: ScreenCapture, 
                                   windows: List[WindowInfo]) -> List[WorkspaceElement]:
        """Extract all UI elements from the screen capture.
        
        Processes each visible window to extract UI elements using OCR and
        computer vision techniques, then sorts elements by z-order.
        
        Args:
            screen_capture: Captured screen image
            windows: List of detected windows
            
        Returns:
            List of all UI elements found across all windows
            
        Raises:
            Exception: If element extraction fails for any window
        """
        all_elements = []
        
        # Process each visible window
        for window in windows:
            if not window.is_visible:
                continue
                
            # Extract elements from window region
            window_elements = await self._extract_window_elements(
                screen_capture, window
            )
            all_elements.extend(window_elements)
            
        # Sort by z-order (approximate by y-coordinate and size)
        all_elements.sort(key=lambda e: (e.bounds[1], -e.bounds[2] * e.bounds[3]))
        
        return all_elements
    
    async def _extract_window_elements(self, screen_capture: ScreenCapture,
                                     window: WindowInfo) -> List[WorkspaceElement]:
        """Extract UI elements from a specific window.
        
        Uses OCR to find text regions and computer vision to detect UI
        components like buttons, menus, and text fields within the window.
        
        Args:
            screen_capture: Captured screen image
            window: Window to extract elements from
            
        Returns:
            List of UI elements found in the window
            
        Raises:
            Exception: If OCR processing or element detection fails
        """
        elements = []
        
        # Define window region
        region = (window.x, window.y, window.width, window.height)
        
        # Run OCR on window region
        ocr_result = await self.ocr_processor.process_image(
            screen_capture.image, region
        )
        
        # Convert text regions to elements
        for text_region in ocr_result.regions:
            element = WorkspaceElement(
                id=f"text_{window.window_id}_{len(elements)}",
                type=text_region.area_type,
                bounds=text_region.bounding_box,
                parent_window=window.window_id,
                text=text_region.text,
                is_interactive=text_region.area_type in ['button', 'menu', 'label'],
                properties={'confidence': text_region.confidence}
            )
            elements.append(element)
            
        # Detect additional UI elements using vision
        ui_elements = await self._detect_ui_elements(screen_capture, window)
        elements.extend(ui_elements)
        
        # Build element hierarchy
        self._build_element_hierarchy(elements)
        
        return elements
    
    async def _detect_ui_elements(self, screen_capture: ScreenCapture,
                                window: WindowInfo) -> List[WorkspaceElement]:
        """Detect UI elements using computer vision techniques.
        
        Uses advanced computer vision to detect UI components that may not
        be captured by OCR, such as buttons, icons, and interactive areas.
        
        Args:
            screen_capture: Captured screen image
            window: Window to analyze for UI elements
            
        Returns:
            List of detected UI elements
            
        Note:
            Currently uses heuristics based on window type. Future versions
            will implement advanced CV techniques for shape and pattern detection.
        """
        elements = []
        
        # This would use advanced CV techniques to detect:
        # - Buttons (by shape and shading)
        # - Text fields (by borders and cursors)
        # - Menus (by structure)
        # - Icons (by patterns)
        # - Scrollbars (by position and appearance)
        
        # For now, we'll use heuristics based on the window type
        if window.app_name in ['Finder', 'Files']:
            # File browser elements
            elements.append(WorkspaceElement(
                id=f"sidebar_{window.window_id}",
                type="sidebar",
                bounds=(window.x, window.y + 50, 200, window.height - 50),
                parent_window=window.window_id,
                is_interactive=True,
                properties={'role': 'navigation'}
            ))
            
        return elements
    
    def _build_element_hierarchy(self, elements: List[WorkspaceElement]):
        """Build parent-child relationships between UI elements.
        
        Analyzes element positions and sizes to determine containment
        relationships, building a hierarchy where larger elements that
        contain smaller ones become parents.
        
        Args:
            elements: List of elements to analyze for hierarchy
            
        Note:
            Modifies elements in-place by adding children to parent elements
        """
        # Sort by area (larger elements are likely parents)
        sorted_elements = sorted(
            elements, 
            key=lambda e: e.bounds[2] * e.bounds[3], 
            reverse=True
        )
        
        # Check containment
        for i, parent in enumerate(sorted_elements):
            for child in sorted_elements[i+1:]:
                if self._contains(parent.bounds, child.bounds):
                    parent.children.append(child)
                    
    def _contains(self, parent_bounds: Tuple[int, int, int, int],
                  child_bounds: Tuple[int, int, int, int]) -> bool:
        """Check if parent bounds completely contain child bounds.
        
        Args:
            parent_bounds: Parent element bounds (x, y, width, height)
            child_bounds: Child element bounds (x, y, width, height)
            
        Returns:
            True if parent completely contains child, False otherwise
        """
        px, py, pw, ph = parent_bounds
        cx, cy, cw, ch = child_bounds
        
        return (cx >= px and cy >= py and 
                cx + cw <= px + pw and cy + ch <= py + ph)
    
    async def navigate_to_element(self, target: WorkspaceElement) -> bool:
        """Navigate to a specific UI element.
        
        Finds the optimal path to the target element and executes all
        necessary navigation steps, including window switching if needed.
        
        Args:
            target: The UI element to navigate to
            
        Returns:
            True if navigation was successful, False otherwise
            
        Raises:
            Exception: If navigation mode is not active or navigation fails
            
        Example:
            >>> button = workspace_map.find_element_at(100, 200)
            >>> success = await nav_system.navigate_to_element(button)
            >>> print(f"Navigation {'succeeded' if success else 'failed'}")
        """
        if not self.navigation_mode:
            logger.warning("Navigation mode not active")
            return False
            
        try:
            # Find optimal path to element
            path = await self._find_navigation_path(target)
            
            if not path:
                logger.error(f"No path found to element {target.id}")
                return False
                
            # Execute navigation steps
            for step in path.steps:
                success = await self._execute_navigation_step(step)
                if not success:
                    logger.error(f"Navigation step failed: {step}")
                    return False
                    
                # Small delay between steps
                await asyncio.sleep(0.1)
                
            # Update focus
            self.current_focus = target
            self.navigation_stats['successful_navigations'] += 1
            
            # Record pattern for learning
            self._record_navigation_pattern(path)
            
            return True
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            self.navigation_stats['failed_navigations'] += 1
            return False
    
    async def _find_navigation_path(self, target: WorkspaceElement) -> Optional[NavigationPath]:
        """Find the optimal path to navigate to a target element.
        
        Analyzes the current state and target element to determine the most
        efficient sequence of actions needed for navigation.
        
        Args:
            target: The element to navigate to
            
        Returns:
            NavigationPath with steps to reach target, or None if no path found
            
        Note:
            Currently implements simple direct navigation. Future versions will
            include complex multi-step paths and obstacle avoidance.
        """
        if not self.current_map:
            return None
            
        # Start from current focus or active window
        start = self.current_focus or self._get_active_element()
        
        if not start:
            return None
            
        path = NavigationPath(start=start, end=target, steps=[])
        
        # Check if target is in same window
        if start.parent_window == target.parent_window:
            # Direct navigation within window
            path.add_step(NavigationAction.CLICK_ELEMENT, {
                'element_id': target.id,
                'coordinates': target.center_point()
            })
        else:
            # Need to switch windows first
            target_window = self._find_window_by_id(target.parent_window)
            if target_window:
                path.add_step(NavigationAction.SWITCH_WINDOW, {
                    'window_id': target.parent_window,
                    'window_title': target_window.window_title
                })
                
                # Then click element
                path.add_step(NavigationAction.CLICK_ELEMENT, {
                    'element_id': target.id,
                    'coordinates': target.center_point()
                })
                
        path.confidence = 0.9  # High confidence for direct paths
        path.estimated_time = len(path.steps) * 0.5  # 0.5s per step estimate
        
        return path
    
    async def _execute_navigation_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single navigation step.
        
        Performs the actual system interaction for a navigation step,
        such as clicking, typing, or window management.
        
        Args:
            step: Dictionary containing action type and parameters
            
        Returns:
            True if step executed successfully, False otherwise
            
        Raises:
            Exception: If step execution encounters system-level errors
        """
        action = step['action']
        params = step['params']
        
        try:
            if action == NavigationAction.SWITCH_WINDOW:
                return await self._switch_to_window(params['window_id'])
                
            elif action == NavigationAction.CLICK_ELEMENT:
                return await self._click_at_coordinates(params['coordinates'])
                
            elif action == NavigationAction.TYPE_TEXT:
                return await self._type_text(params['text'])
                
            elif action == NavigationAction.OPEN_APPLICATION:
                return await self._open_application(params['app_name'])
                
            elif action == NavigationAction.CLOSE_WINDOW:
                return await self._close_window(params['window_id'])
                
            elif action == NavigationAction.ARRANGE_WINDOWS:
                return await self._arrange_windows(params['layout'])
                
            else:
                logger.warning(f"Unknown navigation action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing navigation step: {e}")
            return False
    
    async def _switch_to_window(self, window_id: str) -> bool:
        """Switch focus to a specific window.
        
        Uses macOS APIs to bring the specified window to the foreground
        and give it focus.
        
        Args:
            window_id: ID of the window to switch to
            
        Returns:
            True if window switch was successful, False otherwise
            
        Raises:
            Exception: If macOS APIs are not available or window not found
        """
        if not MACOS_AVAILABLE:
            logger.error("macOS integration not available")
            return False
            
        try:
            # Use macOS APIs to bring window to front
            workspace = NSWorkspace.sharedWorkspace()
            
            # Find the window
            window_info = self._find_window_by_id(window_id)
            if not window_info:
                return False
                
            # Find the app
            apps = workspace.runningApplications()
            for app in apps:
                if app.localizedName() == window_info.app_name:
                    # Activate the app
                    app.activateWithOptions_(NSRunningApplication.ActivationOptions.ActivateIgnoringOtherApps)
                    
                    # Record action
                    self.navigation_history.append(NavigationAction.SWITCH_WINDOW)
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error switching window: {e}")
            return False
    
    async def _click_at_coordinates(self, coordinates: Tuple[int, int]) -> bool:
        """Click at specific screen coordinates.
        
        Generates mouse click events at the specified coordinates using
        macOS Core Graphics APIs.
        
        Args:
            coordinates: Tuple of (x, y) screen coordinates to click
            
        Returns:
            True if click was successful, False otherwise
            
        Raises:
            Exception: If macOS APIs are not available or click fails
        """
        if not MACOS_AVAILABLE:
            logger.error("macOS integration not available")
            return False
            
        try:
            x, y = coordinates
            
            # Create mouse events
            from Quartz import (
                CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap,
                kCGEventLeftMouseDown, kCGEventLeftMouseUp, CGPointMake
            )
            
            point = CGPointMake(x, y)
            
            # Mouse down
            mouse_down = CGEventCreateMouseEvent(
                None, kCGEventLeftMouseDown, point, 0
            )
            CGEventPost(kCGHIDEventTap, mouse_down)
            
            # Small delay
            await asyncio.sleep(0.05)
            
            # Mouse up
            mouse_up = CGEventCreateMouseEvent(
                None, kCGEventLeftMouseUp, point, 0
            )
            CGEventPost(kCGHIDEventTap, mouse_up)
            
            # Record action
            self.navigation_history.append(NavigationAction.CLICK_ELEMENT)
            return True
            
        except Exception as e:
            logger.error(f"Error clicking: {e}")
            return False
    
    async def _type_text(self, text: str) -> bool:
        """Type text at the current cursor position.
        
        Generates keyboard events to type the specified text using
        macOS Core Graphics APIs.
        
        Args:
            text: Text string to type
            
        Returns:
            True if text was typed successfully, False otherwise
            
        Raises:
            Exception: If macOS APIs are not available or typing fails
        """
        if not MACOS_AVAILABLE:
            logger.error("macOS integration not available")
            return False
            
        try:
            from Quartz import (
                CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap,
                CGEventSetUnicodeString
            )
            
            # Create keyboard event
            event = CGEventCreateKeyboardEvent(None, 0, True)
            CGEventSetUnicodeString(event, len(text), text)
            CGEventPost(kCGHIDEventTap, event)
            
            # Record action
            self.navigation_history.append(NavigationAction.TYPE_TEXT)
            return True
            
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return False
    
    async def navigate_to_application(self, app_name: str) -> bool:
        """Navigate to a specific application.
        
        Switches to an existing window of the application if open, or
        launches the application if not currently running.
        
        Args:
            app_name: Name of the application to navigate to