#!/usr/bin/env python3
"""
Workspace Optimizer for JARVIS Multi-Window Intelligence
Suggests optimal window arrangements and workspace improvements
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math

from .window_detector import WindowDetector, WindowInfo
from .window_relationship_detector import WindowRelationshipDetector, WindowGroup
from .smart_query_router import SmartQueryRouter

logger = logging.getLogger(__name__)


@dataclass
class WindowLayout:
    """Represents a suggested window layout"""
    layout_type: str  # 'side_by_side', 'stacked', 'quadrant', 'focus_center'
    positions: Dict[int, Dict[str, int]]  # window_id -> {x, y, width, height}
    description: str
    benefit: str
    confidence: float


@dataclass
class WorkspaceOptimization:
    """Complete workspace optimization recommendation"""
    layout_suggestions: List[WindowLayout]
    missing_tools: List[str]
    focus_improvements: List[str]
    window_cleanup: List[WindowInfo]  # Windows to consider closing
    productivity_score: float  # 0.0 to 1.0
    
    def to_jarvis_message(self) -> str:
        """Convert to JARVIS voice message"""
        messages = []
        
        if self.layout_suggestions:
            messages.append(f"I suggest reorganizing your windows using a {self.layout_suggestions[0].layout_type} layout")
        
        if self.missing_tools:
            messages.append(f"You might benefit from opening {', '.join(self.missing_tools[:2])}")
        
        if self.focus_improvements:
            messages.append(self.focus_improvements[0])
        
        if self.window_cleanup:
            messages.append(f"Consider closing {len(self.window_cleanup)} unused windows")
        
        return "Sir, " + ". ".join(messages) if messages else "Your workspace is well organized, sir"


class WorkspaceOptimizer:
    """Optimizes window arrangements and suggests improvements"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.relationship_detector = WindowRelationshipDetector()
        self.query_router = SmartQueryRouter()
        
        # Screen dimensions (will be detected)
        self.screen_width = 2560  # Default, will update
        self.screen_height = 1600  # Default, will update
        
        # Layout templates
        self.layout_templates = {
            'side_by_side': self._calculate_side_by_side,
            'quadrant': self._calculate_quadrant,
            'focus_center': self._calculate_focus_center,
            'stacked': self._calculate_stacked,
            'coding_layout': self._calculate_coding_layout,
        }
        
        # Ideal tool combinations for different tasks
        self.task_tool_requirements = {
            'coding': {
                'required': ['ide', 'terminal'],
                'recommended': ['browser', 'documentation'],
                'optional': ['communication', 'notes']
            },
            'research': {
                'required': ['browser'],
                'recommended': ['notes', 'documentation'],
                'optional': ['communication']
            },
            'communication': {
                'required': ['communication'],
                'recommended': ['calendar', 'notes'],
                'optional': ['browser']
            },
            'debugging': {
                'required': ['ide', 'terminal', 'browser'],
                'recommended': ['documentation', 'logs'],
                'optional': ['communication']
            }
        }
    
    def analyze_workspace(self, windows: List[WindowInfo] = None) -> WorkspaceOptimization:
        """Analyze workspace and provide optimization suggestions"""
        if windows is None:
            windows = self.window_detector.get_all_windows()
        
        # Update screen dimensions from largest window
        self._update_screen_dimensions(windows)
        
        # Detect window groups and relationships
        relationships = self.relationship_detector.detect_relationships(windows)
        groups = self.relationship_detector.group_windows(windows, relationships)
        
        # Identify current task context
        task_context = self._identify_task_context(windows, groups)
        
        # Generate layout suggestions
        layout_suggestions = self._generate_layout_suggestions(windows, groups, task_context)
        
        # Identify missing tools
        missing_tools = self._identify_missing_tools(windows, task_context)
        
        # Generate focus improvements
        focus_improvements = self._generate_focus_improvements(windows, groups)
        
        # Identify windows to clean up
        window_cleanup = self._identify_cleanup_candidates(windows, groups)
        
        # Calculate productivity score
        productivity_score = self._calculate_productivity_score(
            windows, groups, missing_tools, window_cleanup
        )
        
        return WorkspaceOptimization(
            layout_suggestions=layout_suggestions,
            missing_tools=missing_tools,
            focus_improvements=focus_improvements,
            window_cleanup=window_cleanup,
            productivity_score=productivity_score
        )
    
    def _update_screen_dimensions(self, windows: List[WindowInfo]) -> None:
        """Update screen dimensions based on window positions"""
        if not windows:
            return
        
        # Find maximum window positions to estimate screen size
        max_x = max(w.bounds['x'] + w.bounds['width'] for w in windows)
        max_y = max(w.bounds['y'] + w.bounds['height'] for w in windows)
        
        # Update with some padding
        self.screen_width = max(max_x + 100, 1920)
        self.screen_height = max(max_y + 100, 1080)
    
    def _identify_task_context(self, windows: List[WindowInfo], 
                              groups: List[WindowGroup]) -> str:
        """Identify what task the user is likely doing"""
        # Count window types
        window_types = defaultdict(int)
        
        for window in windows:
            if self._is_ide(window):
                window_types['ide'] += 1
            elif self._is_terminal(window):
                window_types['terminal'] += 1
            elif self._is_browser(window):
                window_types['browser'] += 1
            elif self._is_communication(window):
                window_types['communication'] += 1
            elif self._is_documentation(window):
                window_types['documentation'] += 1
        
        # Determine task based on window composition
        if window_types['ide'] > 0 and window_types['terminal'] > 0:
            if window_types['browser'] > 2:
                return 'debugging'
            else:
                return 'coding'
        elif window_types['browser'] > 3 and window_types['documentation'] > 0:
            return 'research'
        elif window_types['communication'] > 2:
            return 'communication'
        else:
            return 'general'
    
    def _generate_layout_suggestions(self, windows: List[WindowInfo], 
                                   groups: List[WindowGroup],
                                   task_context: str) -> List[WindowLayout]:
        """Generate layout suggestions based on context"""
        suggestions = []
        
        # Get primary group (largest or most important)
        primary_group = groups[0] if groups else None
        
        if task_context == 'coding' and primary_group:
            # Suggest coding layout
            layout = self._calculate_coding_layout(primary_group.windows)
            if layout:
                suggestions.append(layout)
        
        elif task_context == 'research':
            # Suggest quadrant layout for multiple browsers
            browser_windows = [w for w in windows if self._is_browser(w)][:4]
            if len(browser_windows) >= 2:
                layout = self._calculate_quadrant(browser_windows)
                if layout:
                    suggestions.append(layout)
        
        # Always suggest side-by-side for 2 related windows
        if primary_group and len(primary_group.windows) == 2:
            layout = self._calculate_side_by_side(primary_group.windows)
            if layout:
                suggestions.append(layout)
        
        # Suggest focus layout if too many windows
        if len(windows) > 10:
            focused = next((w for w in windows if w.is_focused), None)
            if focused and primary_group:
                layout = self._calculate_focus_center(focused, primary_group.windows)
                if layout:
                    suggestions.append(layout)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _calculate_side_by_side(self, windows: List[WindowInfo]) -> Optional[WindowLayout]:
        """Calculate side-by-side layout for 2 windows"""
        if len(windows) != 2:
            return None
        
        # Simple 50/50 split
        positions = {
            windows[0].window_id: {
                'x': 0,
                'y': 50,  # Leave room for menu bar
                'width': self.screen_width // 2,
                'height': self.screen_height - 50
            },
            windows[1].window_id: {
                'x': self.screen_width // 2,
                'y': 50,
                'width': self.screen_width // 2,
                'height': self.screen_height - 50
            }
        }
        
        return WindowLayout(
            layout_type='side_by_side',
            positions=positions,
            description=f"Side-by-side: {windows[0].app_name} | {windows[1].app_name}",
            benefit="Easy comparison and reference between windows",
            confidence=0.9
        )
    
    def _calculate_quadrant(self, windows: List[WindowInfo]) -> Optional[WindowLayout]:
        """Calculate quadrant layout for up to 4 windows"""
        if not windows or len(windows) > 4:
            return None
        
        positions = {}
        half_width = self.screen_width // 2
        half_height = (self.screen_height - 50) // 2
        
        # Position windows in quadrants
        quadrants = [
            (0, 50),  # Top-left
            (half_width, 50),  # Top-right
            (0, 50 + half_height),  # Bottom-left
            (half_width, 50 + half_height)  # Bottom-right
        ]
        
        for i, window in enumerate(windows[:4]):
            x, y = quadrants[i]
            positions[window.window_id] = {
                'x': x,
                'y': y,
                'width': half_width,
                'height': half_height
            }
        
        return WindowLayout(
            layout_type='quadrant',
            positions=positions,
            description=f"Quadrant layout for {len(windows)} windows",
            benefit="Equal visibility for all windows",
            confidence=0.85
        )
    
    def _calculate_focus_center(self, focus_window: WindowInfo, 
                               related_windows: List[WindowInfo]) -> Optional[WindowLayout]:
        """Calculate layout with focus window in center"""
        if not focus_window:
            return None
        
        positions = {}
        
        # Large center window
        center_width = int(self.screen_width * 0.6)
        center_height = int(self.screen_height * 0.7)
        center_x = (self.screen_width - center_width) // 2
        center_y = (self.screen_height - center_height) // 2 + 50
        
        positions[focus_window.window_id] = {
            'x': center_x,
            'y': center_y,
            'width': center_width,
            'height': center_height
        }
        
        # Arrange other windows around the edges
        other_windows = [w for w in related_windows if w.window_id != focus_window.window_id][:4]
        
        edge_positions = [
            (0, center_y, center_x, center_height),  # Left
            (center_x + center_width, center_y, 
             self.screen_width - center_x - center_width, center_height),  # Right
            (center_x, 50, center_width, center_y - 50),  # Top
            (center_x, center_y + center_height, center_width, 
             self.screen_height - center_y - center_height)  # Bottom
        ]
        
        for i, window in enumerate(other_windows):
            if i < len(edge_positions):
                x, y, w, h = edge_positions[i]
                positions[window.window_id] = {
                    'x': x, 'y': y, 'width': w, 'height': h
                }
        
        return WindowLayout(
            layout_type='focus_center',
            positions=positions,
            description=f"{focus_window.app_name} centered with supporting windows",
            benefit="Maintains focus while keeping context visible",
            confidence=0.8
        )
    
    def _calculate_stacked(self, windows: List[WindowInfo]) -> Optional[WindowLayout]:
        """Calculate vertically stacked layout"""
        if not windows:
            return None
        
        positions = {}
        window_height = (self.screen_height - 50) // len(windows)
        
        for i, window in enumerate(windows):
            positions[window.window_id] = {
                'x': 0,
                'y': 50 + (i * window_height),
                'width': self.screen_width,
                'height': window_height
            }
        
        return WindowLayout(
            layout_type='stacked',
            positions=positions,
            description=f"Stacked layout for {len(windows)} windows",
            benefit="Full width for each window, good for reading",
            confidence=0.7
        )
    
    def _calculate_coding_layout(self, windows: List[WindowInfo]) -> Optional[WindowLayout]:
        """Calculate optimal coding layout"""
        # Find IDE, terminal, and browser
        ide = next((w for w in windows if self._is_ide(w)), None)
        terminal = next((w for w in windows if self._is_terminal(w)), None)
        browser = next((w for w in windows if self._is_browser(w) or self._is_documentation(w)), None)
        
        if not ide:
            return None
        
        positions = {}
        
        if ide and terminal and browser:
            # 3-pane layout: IDE left (60%), terminal bottom-right (40%), browser top-right (40%)
            ide_width = int(self.screen_width * 0.6)
            right_width = self.screen_width - ide_width
            half_height = (self.screen_height - 50) // 2
            
            positions[ide.window_id] = {
                'x': 0, 'y': 50,
                'width': ide_width,
                'height': self.screen_height - 50
            }
            
            positions[browser.window_id] = {
                'x': ide_width, 'y': 50,
                'width': right_width,
                'height': half_height
            }
            
            positions[terminal.window_id] = {
                'x': ide_width, 'y': 50 + half_height,
                'width': right_width,
                'height': half_height
            }
            
            return WindowLayout(
                layout_type='coding_layout',
                positions=positions,
                description="Optimized coding layout: IDE + Terminal + Docs",
                benefit="IDE has focus, terminal visible for commands, docs for reference",
                confidence=0.95
            )
        
        elif ide and terminal:
            # 2-pane: IDE top (70%), terminal bottom (30%)
            ide_height = int((self.screen_height - 50) * 0.7)
            
            positions[ide.window_id] = {
                'x': 0, 'y': 50,
                'width': self.screen_width,
                'height': ide_height
            }
            
            positions[terminal.window_id] = {
                'x': 0, 'y': 50 + ide_height,
                'width': self.screen_width,
                'height': self.screen_height - 50 - ide_height
            }
            
            return WindowLayout(
                layout_type='coding_layout',
                positions=positions,
                description="IDE + Terminal stacked layout",
                benefit="Full width for code, terminal accessible below",
                confidence=0.9
            )
        
        return None
    
    def _identify_missing_tools(self, windows: List[WindowInfo], 
                               task_context: str) -> List[str]:
        """Identify tools that would help with current task"""
        missing = []
        
        if task_context not in self.task_tool_requirements:
            return missing
        
        requirements = self.task_tool_requirements[task_context]
        
        # Check required tools
        current_tools = self._get_current_tools(windows)
        
        for tool in requirements['required']:
            if tool not in current_tools:
                missing.append(self._get_tool_suggestion(tool))
        
        # Check recommended tools (only if no required tools missing)
        if not missing:
            for tool in requirements['recommended']:
                if tool not in current_tools:
                    missing.append(self._get_tool_suggestion(tool))
        
        return missing[:3]  # Return top 3 suggestions
    
    def _get_current_tools(self, windows: List[WindowInfo]) -> Set[str]:
        """Get set of current tool types"""
        tools = set()
        
        for window in windows:
            if self._is_ide(window):
                tools.add('ide')
            elif self._is_terminal(window):
                tools.add('terminal')
            elif self._is_browser(window):
                tools.add('browser')
            elif self._is_communication(window):
                tools.add('communication')
            elif self._is_documentation(window):
                tools.add('documentation')
        
        return tools
    
    def _get_tool_suggestion(self, tool_type: str) -> str:
        """Get friendly suggestion for tool type"""
        suggestions = {
            'ide': 'your code editor',
            'terminal': 'Terminal',
            'browser': 'a browser for documentation',
            'documentation': 'documentation viewer',
            'communication': 'communication app',
            'notes': 'note-taking app'
        }
        return suggestions.get(tool_type, tool_type)
    
    def _generate_focus_improvements(self, windows: List[WindowInfo], 
                                   groups: List[WindowGroup]) -> List[str]:
        """Generate suggestions for improving focus"""
        improvements = []
        
        # Check for too many windows
        if len(windows) > 20:
            improvements.append(
                f"You have {len(windows)} windows open. Consider closing inactive windows to improve focus"
            )
        
        # Check for scattered related windows
        if groups:
            for group in groups:
                if group.group_type == 'project' and len(group.windows) > 1:
                    # Check if windows are far apart
                    if self._are_windows_scattered(group.windows):
                        improvements.append(
                            f"Your {group.group_type} windows are scattered. Group them together for better workflow"
                        )
                        break
        
        # Check for distracting apps while coding
        focused = next((w for w in windows if w.is_focused), None)
        if focused and self._is_ide(focused):
            distracting = [w for w in windows if self._is_distracting(w) and w.is_visible]
            if distracting:
                improvements.append(
                    "Consider minimizing social media or entertainment apps while coding"
                )
        
        return improvements[:2]  # Return top 2 improvements
    
    def _identify_cleanup_candidates(self, windows: List[WindowInfo], 
                                   groups: List[WindowGroup]) -> List[WindowInfo]:
        """Identify windows that could be closed"""
        candidates = []
        
        # Find windows not in any group
        grouped_ids = set()
        for group in groups:
            grouped_ids.update(w.window_id for w in group.windows)
        
        ungrouped = [w for w in windows if w.window_id not in grouped_ids]
        
        # Prioritize invisible, background windows
        for window in ungrouped:
            if not window.is_visible or window.layer < 0:
                candidates.append(window)
        
        # Look for duplicate windows
        app_windows = defaultdict(list)
        for window in windows:
            app_windows[window.app_name].append(window)
        
        for app, app_wins in app_windows.items():
            if len(app_wins) > 3:  # More than 3 windows of same app
                # Keep focused and 2 most recent, suggest closing others
                app_wins.sort(key=lambda w: (not w.is_focused, w.window_id))
                candidates.extend(app_wins[3:])
        
        return candidates[:5]  # Return top 5 candidates
    
    def _calculate_productivity_score(self, windows: List[WindowInfo],
                                    groups: List[WindowGroup],
                                    missing_tools: List[str],
                                    cleanup_candidates: List[WindowInfo]) -> float:
        """Calculate overall workspace productivity score"""
        score = 1.0
        
        # Penalty for too many windows
        if len(windows) > 30:
            score -= 0.2
        elif len(windows) > 20:
            score -= 0.1
        
        # Penalty for missing tools
        score -= len(missing_tools) * 0.1
        
        # Penalty for windows needing cleanup
        score -= len(cleanup_candidates) * 0.05
        
        # Bonus for good grouping
        if groups:
            score += min(len(groups) * 0.05, 0.2)
        
        # Bonus for focused window
        if any(w.is_focused for w in windows):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _are_windows_scattered(self, windows: List[WindowInfo]) -> bool:
        """Check if windows are far apart on screen"""
        if len(windows) < 2:
            return False
        
        # Calculate center points
        centers = []
        for window in windows:
            cx = window.bounds['x'] + window.bounds['width'] // 2
            cy = window.bounds['y'] + window.bounds['height'] // 2
            centers.append((cx, cy))
        
        # Calculate average distance between windows
        total_distance = 0
        count = 0
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                total_distance += distance
                count += 1
        
        if count > 0:
            avg_distance = total_distance / count
            # Consider scattered if average distance > 40% of screen diagonal
            screen_diagonal = math.sqrt(self.screen_width**2 + self.screen_height**2)
            return avg_distance > screen_diagonal * 0.4
        
        return False
    
    def _is_ide(self, window: WindowInfo) -> bool:
        """Check if window is an IDE"""
        ide_apps = ['Visual Studio Code', 'Cursor', 'Xcode', 'IntelliJ', 'PyCharm']
        return any(ide in window.app_name for ide in ide_apps)
    
    def _is_terminal(self, window: WindowInfo) -> bool:
        """Check if window is a terminal"""
        return 'Terminal' in window.app_name or 'iTerm' in window.app_name
    
    def _is_browser(self, window: WindowInfo) -> bool:
        """Check if window is a browser"""
        browsers = ['Chrome', 'Safari', 'Firefox', 'Edge']
        return any(browser in window.app_name for browser in browsers)
    
    def _is_communication(self, window: WindowInfo) -> bool:
        """Check if window is communication app"""
        comm_apps = ['Discord', 'Slack', 'Messages', 'Mail']
        return any(app in window.app_name for app in comm_apps)
    
    def _is_documentation(self, window: WindowInfo) -> bool:
        """Check if window is documentation"""
        if window.window_title:
            doc_keywords = ['docs', 'documentation', 'api', 'reference']
            return any(kw in window.window_title.lower() for kw in doc_keywords)
        return False
    
    def _is_distracting(self, window: WindowInfo) -> bool:
        """Check if window is potentially distracting"""
        distracting_apps = ['Twitter', 'Facebook', 'Instagram', 'TikTok', 
                           'YouTube', 'Netflix', 'Spotify']
        return any(app in window.app_name for app in distracting_apps)


def test_workspace_optimizer():
    """Test workspace optimization"""
    print("🔧 Testing Workspace Optimizer")
    print("=" * 50)
    
    optimizer = WorkspaceOptimizer()
    
    print("\nAnalyzing current workspace...")
    optimization = optimizer.analyze_workspace()
    
    print(f"\n📊 Workspace Analysis:")
    print(f"   Productivity Score: {optimization.productivity_score:.0%}")
    
    if optimization.layout_suggestions:
        print(f"\n📐 Layout Suggestions:")
        for i, layout in enumerate(optimization.layout_suggestions, 1):
            print(f"   {i}. {layout.description}")
            print(f"      Benefit: {layout.benefit}")
            print(f"      Confidence: {layout.confidence:.0%}")
    
    if optimization.missing_tools:
        print(f"\n🔧 Missing Tools:")
        for tool in optimization.missing_tools:
            print(f"   • {tool}")
    
    if optimization.focus_improvements:
        print(f"\n🎯 Focus Improvements:")
        for improvement in optimization.focus_improvements:
            print(f"   • {improvement}")
    
    if optimization.window_cleanup:
        print(f"\n🧹 Windows to Consider Closing:")
        for window in optimization.window_cleanup[:3]:
            print(f"   • {window.app_name} - {window.window_title or 'Untitled'}")
    
    print(f"\n🎙️ JARVIS would say: \"{optimization.to_jarvis_message()}\"")


if __name__ == "__main__":
    test_workspace_optimizer()