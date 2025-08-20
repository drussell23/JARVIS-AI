#!/usr/bin/env python3
"""
Workspace Analyzer for JARVIS Multi-Window Intelligence
Analyzes relationships between windows and provides workspace insights
"""

import os
import base64
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
import json
import cv2
import numpy as np

from .window_detector import WindowDetector, WindowInfo
from .multi_window_capture import MultiWindowCapture, WindowCapture

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
        
        # Initialize Claude if available
        self.claude_client = None
        if CLAUDE_AVAILABLE and (api_key or os.getenv("ANTHROPIC_API_KEY")):
            self.claude_client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            logger.info("Claude Vision initialized for workspace analysis")
        else:
            logger.warning("Running without Claude - using basic analysis only")
    
    async def analyze_workspace(self, query: str = "What am I working on?") -> WorkspaceAnalysis:
        """Analyze entire workspace based on query"""
        
        # Capture multiple windows
        captures = await self.capture_system.capture_multiple_windows()
        
        if not captures:
            return WorkspaceAnalysis(
                focused_task="Unable to capture windows",
                window_relationships={},
                workspace_context="No windows detected",
                suggestions=[],
                important_notifications=[],
                confidence=0.0
            )
        
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
    
    def _analyze_basic(self, captures: List[WindowCapture]) -> WorkspaceAnalysis:
        """Basic workspace analysis without Claude"""
        
        focused = next((c for c in captures if c.window_info.is_focused), None)
        
        if focused:
            focused_task = f"Working in {focused.window_info.app_name}"
            if focused.window_info.window_title:
                focused_task += f" on '{focused.window_info.window_title}'"
        else:
            focused_task = "No focused application detected"
        
        # Group windows by app category
        categories = {
            'development': [],
            'browser': [],
            'communication': [],
            'other': []
        }
        
        for capture in captures:
            app = capture.window_info.app_name
            if any(dev in app for dev in ['Code', 'Terminal', 'Xcode']):
                categories['development'].append(app)
            elif any(browser in app for browser in ['Chrome', 'Safari', 'Firefox']):
                categories['browser'].append(app)
            elif any(comm in app for comm in ['Discord', 'Slack', 'Messages']):
                categories['communication'].append(app)
            else:
                categories['other'].append(app)
        
        # Build relationships
        relationships = {}
        if categories['development'] and categories['browser']:
            relationships['development_support'] = [
                f"{dev} likely using {browser} for documentation"
                for dev in categories['development']
                for browser in categories['browser']
            ]
        
        # Build context
        active_categories = [cat for cat, apps in categories.items() if apps]
        workspace_context = f"Active workspace with {len(captures)} windows across {len(active_categories)} categories"
        
        return WorkspaceAnalysis(
            focused_task=focused_task,
            window_relationships=relationships,
            workspace_context=workspace_context,
            suggestions=["Enable Claude API for intelligent workspace analysis"],
            important_notifications=[],
            confidence=0.5
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
            import re
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