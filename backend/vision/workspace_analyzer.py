#!/usr/bin/env python3
"""
Workspace Analyzer for JARVIS Multi-Window Intelligence
Analyzes relationships between windows and provides workspace insights
"""

import os
import base64
import asyncio
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
import json
import cv2
import numpy as np

from .window_detector import WindowDetector, WindowInfo
from .multi_window_capture import MultiWindowCapture, WindowCapture
from .window_relationship_detector import WindowRelationshipDetector
from .smart_query_router import SmartQueryRouter, QueryRoute, QueryIntent
from .privacy_controls import PrivacyControlSystem

# Only import Anthropic if available
try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logging.warning("Anthropic library not available - workspace analysis will be limited")

logger = logging.getLogger(__name__)

@dataclass
class WorkspaceAnalysis:
    """Complete workspace analysis result"""
    focused_task: str
    window_relationships: Dict[str, List[str]]
    workspace_context: str
    suggestions: List[str]
    important_notifications: List[str]
    confidence: float

class WorkspaceAnalyzer:
    """Analyzes multi-window workspace using Claude Vision"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.window_detector = WindowDetector()
        self.capture_system = MultiWindowCapture()
        self.relationship_detector = WindowRelationshipDetector()
        self.query_router = SmartQueryRouter()
        self.privacy_controls = PrivacyControlSystem()
        
        # Initialize Claude if available
        self.claude_client = None
        if CLAUDE_AVAILABLE and (api_key or os.getenv("ANTHROPIC_API_KEY")):
            self.claude_client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            logger.info("Claude Vision initialized for workspace analysis")
        else:
            logger.warning("Running without Claude - using basic analysis only")
    
    async def analyze_workspace(self, query: str = "What am I working on?") -> WorkspaceAnalysis:
        """Analyze entire workspace based on query"""
        
        # Get all windows first
        all_windows = self.window_detector.get_all_windows()
        
        # Apply privacy filtering
        filtered_windows, blocked = self.privacy_controls.filter_windows(all_windows)
        
        # Log if windows were blocked
        if blocked:
            logger.info(f"Privacy: Blocked {len(blocked)} windows from analysis")
        
        # Use dynamic multi-window engine for intelligent window selection
        try:
            from .dynamic_multi_window_engine import get_dynamic_multi_window_engine
            dynamic_engine = get_dynamic_multi_window_engine()
            
            # Get dynamic analysis
            analysis = dynamic_engine.analyze_windows_for_query(query, filtered_windows)
            
            # Capture windows based on dynamic analysis
            captures = []
            
            # Capture primary windows with full resolution
            for window in analysis.primary_windows:
                try:
                    capture = await self.capture_system._async_capture_window(
                        window, 1.0  # Full resolution for primary windows
                    )
                    captures.append(capture)
                except Exception as e:
                    logger.warning(f"Failed to capture primary window {window.app_name}: {e}")
            
            # Capture context windows with reduced resolution
            for window in analysis.context_windows[:3]:  # Limit context windows
                try:
                    capture = await self.capture_system._async_capture_window(
                        window, 0.5  # Half resolution for context
                    )
                    captures.append(capture)
                except Exception as e:
                    logger.warning(f"Failed to capture context window {window.app_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Dynamic analysis failed, using fallback: {e}")
            # Fallback to smart query routing
            route = self.query_router.route_query(query, filtered_windows)
            
            # Capture only the relevant windows
            if route.capture_all:
                # For overview queries, capture representative sample
                captures = await self.capture_system.capture_multiple_windows()
            else:
                # Capture specific windows based on routing
                captures = []
                for window in route.target_windows[:5]:  # Limit to 5
                    try:
                        capture = await self.capture_system._async_capture_window(
                            window, 
                            1.0 if window.is_focused else 0.5
                        )
                        captures.append(capture)
                    except Exception as e:
                        logger.warning(f"Failed to capture {window.app_name}: {e}")
                        # Continue with other windows instead of failing entirely
        
        if not captures:
            # Fall back to basic window info without screenshots
            all_windows = self.window_detector.get_all_windows()
            filtered_windows, _ = self.privacy_controls.filter_windows(all_windows)
            
            # Still try to provide useful info based on window titles
            return self._analyze_from_window_info_only(filtered_windows, query)
        
        # Use Claude for analysis if available
        if self.claude_client and len(captures) > 0:
            return await self._analyze_with_claude(captures, query)
        else:
            return self._analyze_basic(captures)
    
    async def _analyze_with_claude(self, captures: List[WindowCapture], 
                                  query: str) -> WorkspaceAnalysis:
        """Analyze workspace using Claude Vision"""
        
        # Prepare images for Claude
        image_messages = []
        
        # Add focused window first (if exists)
        focused_capture = next((c for c in captures if c.window_info.is_focused), None)
        
        if focused_capture:
            # Encode focused window
            focused_b64 = self._encode_image(focused_capture.image)
            image_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": focused_b64
                }
            })
            
            # Add context about focused window
            window_context = f"PRIMARY WINDOW (In Focus): {focused_capture.window_info.app_name}"
            if focused_capture.window_info.window_title:
                window_context += f" - {focused_capture.window_info.window_title}"
        
        # Create composite of other windows
        other_captures = [c for c in captures if not c.window_info.is_focused][:4]
        
        if other_captures:
            try:
                # Create a grid of other windows
                composite = self._create_context_grid(other_captures)
                composite_b64 = self._encode_image(composite)
            except Exception as e:
                logger.error(f"Error creating context grid: {e}")
                # Skip composite if it fails
                other_captures = []
            
            image_messages.append({
                "type": "text",
                "text": "\nCONTEXT WINDOWS (Background):"
            })
            
            image_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": composite_b64
                }
            })
            
            # List context windows
            context_list = "\n".join([
                f"- {c.window_info.app_name}: {c.window_info.window_title or 'Untitled'}"
                for c in other_captures
            ])
            
            image_messages.append({
                "type": "text",
                "text": context_list
            })
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this multi-window workspace and answer: {query}

Be CONCISE. Start your response directly with what the user is doing. No preambles like "Based on the workspace..." or "I can see that...".

Format your response as:
1. PRIMARY TASK: One sentence about what the user is working on (start with "You're...")
2. CONTEXT: Brief note about supporting windows (if relevant)
3. NOTIFICATIONS: Only mention errors, warnings, or urgent items
4. SUGGESTIONS: Only if critical

Keep the ENTIRE response under 100 words. Focus on the primary window."""

        try:
            # Send to Claude
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,  # Reduced to encourage conciseness
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        *image_messages
                    ]
                }]
            )
            
            # Parse Claude's response
            return self._parse_claude_response(response.content[0].text, captures)
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return self._analyze_basic(captures)
    
    def _analyze_from_window_info_only(self, windows: List[WindowInfo], 
                                       query: str, route: Optional[QueryRoute] = None) -> WorkspaceAnalysis:
        """Analyze workspace using only window titles when captures fail"""
        # Find focused window
        focused_window = next((w for w in windows if w.is_focused), None)
        
        # Determine task based on window info
        if focused_window:
            task = f"You're working in {focused_window.app_name}"
            if focused_window.window_title:
                task += f" on {focused_window.window_title}"
        else:
            task = "Unable to determine current focus"
        
        # Detect window relationships
        relationships = self.relationship_detector.detect_relationships(windows)
        
        # Build context using dynamic analysis
        try:
            from .dynamic_multi_window_engine import get_dynamic_multi_window_engine
            dynamic_engine = get_dynamic_multi_window_engine()
            
            # Get dynamic analysis for context
            analysis = dynamic_engine.analyze_windows_for_query(query, windows)
            
            if analysis.primary_windows:
                primary_apps = [w.app_name for w in analysis.primary_windows]
                context = f"Primary focus on: {', '.join(primary_apps)}"
                if analysis.context_windows:
                    context += f" with {len(analysis.context_windows)} related windows"
            else:
                context = f"{len(windows)} windows open across {len(set(w.app_name for w in windows))} applications"
        except:
            # Fallback context
            context = f"{len(windows)} windows open across {len(set(w.app_name for w in windows))} applications"
        
        # Check for important notifications in window titles
        notifications = []
        for window in windows:
            if window.window_title:
                title_lower = window.window_title.lower()
                if any(word in title_lower for word in ['error', 'warning', 'failed', 'alert']):
                    notifications.append(f"{window.app_name}: {window.window_title}")
        
        return WorkspaceAnalysis(
            focused_task=task,
            window_relationships={"groups": relationships} if relationships else {},
            workspace_context=context,
            suggestions=[],
            important_notifications=notifications[:3],  # Limit to 3
            confidence=0.5  # Lower confidence without screenshots
        )
    
    def _analyze_basic(self, captures: List[WindowCapture]) -> WorkspaceAnalysis:
        """Basic workspace analysis without Claude"""
        
        focused = next((c for c in captures if c.window_info.is_focused), None)
        
        if focused:
            focused_task = f"Working in {focused.window_info.app_name}"
            if focused.window_info.window_title:
                focused_task += f" on '{focused.window_info.window_title}'"
        else:
            focused_task = "No focused application detected"
        
        # Get all windows for relationship analysis
        all_windows = [c.window_info for c in captures]
        
        # Detect relationships using our intelligence layer
        detected_relationships = self.relationship_detector.detect_relationships(all_windows)
        groups = self.relationship_detector.group_windows(all_windows, detected_relationships)
        
        # Build relationships dictionary
        relationships = {}
        for rel in detected_relationships:
            if rel.confidence >= 0.7:  # Only high confidence
                window1 = next(w for w in all_windows if w.window_id == rel.window1_id)
                window2 = next(w for w in all_windows if w.window_id == rel.window2_id)
                rel_key = f"{window1.app_name}_{window2.app_name}"
                relationships[rel_key] = [
                    f"{rel.relationship_type}: {', '.join(rel.evidence)}"
                ]
        
        # Build context from groups
        if groups:
            group = groups[0]  # Primary group
            workspace_context = f"Working on {group.group_type} with {len(group.windows)} related windows"
            if group.common_elements:
                workspace_context += f" ({', '.join(group.common_elements[:2])})"
        else:
            workspace_context = f"Active workspace with {len(captures)} windows"
        
        # Generate suggestions based on workspace
        suggestions = []
        if any('error' in w.window_title.lower() if w.window_title else False for w in all_windows):
            suggestions.append("Errors detected - check terminal output")
        
        return WorkspaceAnalysis(
            focused_task=focused_task,
            window_relationships=relationships,
            workspace_context=workspace_context,
            suggestions=suggestions or ["Enable Claude API for deeper analysis"],
            important_notifications=[],
            confidence=0.7
        )
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 for Claude"""
        # Convert to PNG
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    
    def _create_context_grid(self, captures: List[WindowCapture], 
                           max_width: int = 800) -> np.ndarray:
        """Create a grid of context windows"""
        
        if not captures:
            return np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Resize all images to same height
        target_height = 300
        resized_images = []
        
        for capture in captures[:4]:  # Max 4 windows in grid
            img = capture.image
            if img is None or img.size == 0:
                continue
                
            scale = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            
            # Limit width
            if new_width > max_width // 2:
                new_width = max_width // 2
                scale = new_width / img.shape[1]
                target_height_adj = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_width, target_height_adj))
            else:
                img = cv2.resize(img, (new_width, target_height))
            
            # Add label
            label = capture.window_info.app_name
            cv2.putText(img, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            resized_images.append(img)
        
        if not resized_images:
            return np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Create 2x2 grid
        if len(resized_images) >= 2:
            # Pad images to same width for each row
            row1_width = max(img.shape[1] for img in resized_images[:2])
            row1 = []
            for img in resized_images[:2]:
                if img.shape[1] < row1_width:
                    pad = row1_width - img.shape[1]
                    img = np.pad(img, ((0, 0), (0, pad), (0, 0)), constant_values=240)
                row1.append(img)
            
            row1_combined = np.hstack(row1)
            
            if len(resized_images) >= 3:
                # Check if we have images for row2
                row2_images = resized_images[2:4]
                if row2_images:
                    row2_width = max(img.shape[1] for img in row2_images)
                    row2 = []
                    for img in row2_images:
                        if img.shape[1] < row2_width:
                            pad = row2_width - img.shape[1]
                            img = np.pad(img, ((0, 0), (0, pad), (0, 0)), constant_values=240)
                        row2.append(img)
                    
                    row2_combined = np.hstack(row2) if len(row2) > 1 else row2[0]
                    
                    # Pad rows to same width
                    max_row_width = max(row1_combined.shape[1], row2_combined.shape[1])
                    if row1_combined.shape[1] < max_row_width:
                        pad = max_row_width - row1_combined.shape[1]
                        row1_combined = np.pad(row1_combined, ((0, 0), (0, pad), (0, 0)), constant_values=240)
                    if row2_combined.shape[1] < max_row_width:
                        pad = max_row_width - row2_combined.shape[1]
                        row2_combined = np.pad(row2_combined, ((0, 0), (0, pad), (0, 0)), constant_values=240)
                    
                    grid = np.vstack([row1_combined, row2_combined])
                else:
                    grid = row1_combined
            else:
                grid = row1_combined
        else:
            grid = resized_images[0]
        
        return grid
    
    def _parse_claude_response(self, response: str, 
                             captures: List[WindowCapture]) -> WorkspaceAnalysis:
        """Parse Claude's response into structured analysis"""
        
        # Remove common verbose preambles
        response = response.replace("Based on the information provided in the multi-window workspace,", "")
        response = response.replace("Based on the multi-window workspace analysis,", "")
        response = response.replace("Looking at your workspace,", "")
        response = response.replace("I can see that", "")
        response = response.strip()
        
        # Extract sections from formatted response
        focused_task = ""
        context_info = ""
        notifications = []
        suggestions = []
        
        # Parse the response looking for our format markers
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section markers
            if line.startswith("PRIMARY TASK:") or line.startswith("1. PRIMARY TASK:"):
                current_section = "task"
                # Extract the task after the marker
                task_text = line.replace("PRIMARY TASK:", "").replace("1. PRIMARY TASK:", "").strip()
                if task_text:
                    focused_task = task_text
            elif line.startswith("CONTEXT:") or line.startswith("2. CONTEXT:"):
                current_section = "context"
                context_text = line.replace("CONTEXT:", "").replace("2. CONTEXT:", "").strip()
                if context_text:
                    context_info = context_text
            elif line.startswith("NOTIFICATIONS:") or line.startswith("3. NOTIFICATIONS:"):
                current_section = "notifications"
                notif_text = line.replace("NOTIFICATIONS:", "").replace("3. NOTIFICATIONS:", "").strip()
                if notif_text and notif_text.lower() not in ["none", "n/a", "-"]:
                    notifications.append(notif_text)
            elif line.startswith("SUGGESTIONS:") or line.startswith("4. SUGGESTIONS:"):
                current_section = "suggestions"
                sugg_text = line.replace("SUGGESTIONS:", "").replace("4. SUGGESTIONS:", "").strip()
                if sugg_text and sugg_text.lower() not in ["none", "n/a", "-"]:
                    suggestions.append(sugg_text)
            else:
                # Continue adding to current section if no new marker
                if current_section == "task" and not focused_task:
                    focused_task = line
                elif current_section == "notifications":
                    if line and line.lower() not in ["none", "n/a", "-"]:
                        notifications.append(line)
                elif current_section == "suggestions":
                    if line and line.lower() not in ["none", "n/a", "-"]:
                        suggestions.append(line)
        
        # Clean up any numbered list formatting
        if focused_task:
            # Remove numbered list prefixes like "1. " or "1) "
            focused_task = re.sub(r'^\d+[\.\)]\s*', '', focused_task)
            focused_task = focused_task.strip()
        
        # If we couldn't parse the formatted response, try to extract meaningful content
        if not focused_task:
            # Look for lines starting with "You're" or containing task info
            for line in lines:
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if clean_line.startswith("You're") or clean_line.startswith("Working"):
                    focused_task = clean_line
                    break
            
            # If still no task, create from window info
            if not focused_task and captures:
                focused_capture = next((c for c in captures if c.window_info.is_focused), None)
                if focused_capture:
                    app_name = focused_capture.window_info.app_name
                    title = focused_capture.window_info.window_title
                    if title:
                        focused_task = f"You're working in {app_name} on {title}"
                    else:
                        focused_task = f"You're working in {app_name}"
        
        # Build relationships from captures
        relationships = {}
        if len(captures) > 1:
            focused = next((c for c in captures if c.window_info.is_focused), None)
            if focused:
                other_apps = [c.window_info.app_name for c in captures 
                            if c != focused and c.window_info.app_name != focused.window_info.app_name][:3]
                if other_apps:
                    relationships["supporting_apps"] = [
                        f"Also using: {', '.join(other_apps)}"
                    ]
        
        # Create concise workspace context
        workspace_context = focused_task or f"{len(captures)} windows detected"
        
        return WorkspaceAnalysis(
            focused_task=focused_task or "Analyzing workspace",
            window_relationships=relationships,
            workspace_context=workspace_context,
            suggestions=suggestions[:1] if suggestions else [],  # Max 1 suggestion
            important_notifications=notifications[:1] if notifications else [],  # Max 1 notification
            confidence=0.9
        )

async def test_workspace_analyzer():
    """Test workspace analysis"""
    print("🧠 Testing Workspace Analyzer")
    print("=" * 50)
    
    analyzer = WorkspaceAnalyzer()
    
    # Test different queries
    queries = [
        "What am I working on?",
        "Do I have any messages?",
        "What's the overall state of my workspace?",
        "Are there any errors I should look at?"
    ]
    
    for query in queries:
        print(f"\n❓ Query: '{query}'")
        print("Analyzing workspace...")
        
        analysis = await analyzer.analyze_workspace(query)
        
        print(f"\n📊 Analysis Results:")
        print(f"   Primary Task: {analysis.focused_task}")
        print(f"   Confidence: {analysis.confidence:.0%}")
        
        if analysis.window_relationships:
            print(f"\n   Window Relationships:")
            for rel, details in list(analysis.window_relationships.items())[:3]:
                print(f"   - {rel}: {details[0]}")
        
        if analysis.suggestions:
            print(f"\n   Suggestions:")
            for suggestion in analysis.suggestions[:3]:
                print(f"   • {suggestion}")
        
        if analysis.important_notifications:
            print(f"\n   Notifications:")
            for notification in analysis.important_notifications[:3]:
                print(f"   ⚠️  {notification}")
        
        print("\n" + "-" * 50)
    
    print("\n✅ Workspace analysis test complete!")

if __name__ == "__main__":
    asyncio.run(test_workspace_analyzer())