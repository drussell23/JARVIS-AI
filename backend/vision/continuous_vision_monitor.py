#!/usr/bin/env python3
"""
Continuous Vision Monitor for JARVIS
Monitors screen continuously in autonomous mode using Claude API
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import base64
from PIL import Image
import io

from vision.screen_vision import ScreenVisionSystem
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
from vision.window_detector import WindowDetector
from vision.workspace_analyzer import WorkspaceAnalyzer

logger = logging.getLogger(__name__)

class ContinuousVisionMonitor:
    """
    Monitors screen continuously and uses Claude to understand what's happening
    """
    
    def __init__(self, claude_api_key: str):
        """Initialize continuous vision monitor"""
        self.monitoring_active = False
        self.monitoring_interval = 2.0  # seconds
        
        # Initialize components
        self.screen_vision = ScreenVisionSystem()
        self.claude_vision = ClaudeVisionAnalyzer(claude_api_key)
        self.window_detector = WindowDetector()
        self.workspace_analyzer = WorkspaceAnalyzer()
        
        # State tracking
        self.last_screen_state = None
        self.notification_history = []
        self.action_callbacks = []
        self.update_callbacks = []
        
        logger.info("Continuous Vision Monitor initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting continuous vision monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        logger.info("Stopping continuous vision monitoring")
    
    def add_update_callback(self, callback: Callable):
        """Add callback for screen updates"""
        self.update_callbacks.append(callback)
    
    def add_action_callback(self, callback: Callable):
        """Add callback for suggested actions"""
        self.action_callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Main monitoring loop with CPU throttling"""
        import psutil
        cpu_threshold = 25.0
        
        # Increase monitoring interval to reduce CPU usage
        self.monitoring_interval = max(self.monitoring_interval, 10.0)  # Min 10 seconds
        
        while self.monitoring_active:
            try:
                # Check CPU before processing
                cpu_usage = psutil.cpu_percent(interval=0.1)
                if cpu_usage > cpu_threshold:
                    logger.debug(f"CPU too high ({cpu_usage}%) - skipping vision monitor cycle")
                    await asyncio.sleep(30)  # Wait 30 seconds when CPU is high
                    continue
                
                # Capture current screen state
                screen_state = await self._capture_screen_state()
                
                # Skip Claude analysis if CPU is still elevated
                if psutil.cpu_percent(interval=0.1) > cpu_threshold:
                    logger.debug("CPU elevated - using cached analysis")
                    analysis = self.last_screen_state if self.last_screen_state else screen_state
                else:
                    # Analyze with Claude only when CPU is low
                    analysis = await self._analyze_screen_state(screen_state)
                
                # Check for significant changes
                if self._has_significant_change(analysis):
                    # Notify update callbacks
                    for callback in self.update_callbacks:
                        try:
                            await callback(analysis)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                # Check for actionable items
                if analysis.get("actionable_items"):
                    for callback in self.action_callbacks:
                        try:
                            await callback(analysis["actionable_items"])
                        except Exception as e:
                            logger.error(f"Action callback error: {e}")
                
                # Update state
                self.last_screen_state = analysis
                
                # Dynamic interval based on CPU
                if cpu_usage < 15:
                    interval = self.monitoring_interval
                elif cpu_usage < 20:
                    interval = self.monitoring_interval * 2
                else:
                    interval = self.monitoring_interval * 4
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 60 seconds on error
    
    async def _capture_screen_state(self) -> Dict[str, Any]:
        """Capture comprehensive screen state"""
        try:
            # Capture main screen
            screenshot = await self.screen_vision.capture_screen()
            
            # Get all windows
            windows = self.window_detector.get_all_windows()
            
            # Get focused window
            focused_window = next((w for w in windows if w.is_focused), None)
            
            # Prepare screen state
            state = {
                "timestamp": datetime.now().isoformat(),
                "screenshot": self._encode_image(screenshot),
                "windows": [
                    {
                        "id": w.window_id,
                        "app": w.app_name,
                        "title": w.window_title,
                        "focused": w.is_focused,
                        "visible": w.is_visible,
                        "bounds": w.bounds
                    }
                    for w in windows[:20]  # Limit to 20 most relevant
                ],
                "focused_app": focused_window.app_name if focused_window else None,
                "window_count": len(windows)
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error capturing screen state: {e}")
            return {}
    
    async def _analyze_screen_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze screen state with Claude"""
        try:
            # Prepare image for Claude
            screenshot_data = state.get("screenshot", "")
            
            # Build comprehensive prompt
            prompt = f"""Analyze this screen capture and provide detailed insights.

Current workspace:
- Focused app: {state.get('focused_app', 'Unknown')}
- Total windows: {state.get('window_count', 0)}
- Top windows: {json.dumps(state.get('windows', [])[:5], indent=2)}

Please analyze:
1. What is the user currently doing?
2. Are there any notifications (WhatsApp, Messages, Discord, Slack)?
3. Are there any errors or issues visible?
4. What actions might be helpful?
5. Is there anything urgent that needs attention?

Provide a comprehensive analysis including:
- Current task/activity
- Detected notifications (app name, count, urgency)
- Potential issues or errors
- Suggested actions
- Context understanding

Respond in JSON format."""

            # Get Claude's analysis
            if screenshot_data:
                analysis = await self.claude_vision.analyze_image_with_prompt(
                    screenshot_data, prompt
                )
            else:
                # Fallback to text-only analysis
                analysis = await self.claude_vision.analyze_workspace_context(state)
            
            # Parse response
            try:
                result = json.loads(analysis) if isinstance(analysis, str) else analysis
            except:
                result = {"raw_analysis": analysis}
            
            # Add metadata
            result.update({
                "timestamp": state["timestamp"],
                "focused_app": state.get("focused_app"),
                "window_count": state.get("window_count")
            })
            
            # Extract notifications
            notifications = self._extract_notifications(result, state)
            result["notifications"] = notifications
            
            # Determine actionable items
            actionable_items = self._determine_actionable_items(result)
            result["actionable_items"] = actionable_items
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing screen state: {e}")
            return state
    
    def _extract_notifications(self, analysis: Dict, state: Dict) -> List[Dict]:
        """Extract notifications from analysis"""
        notifications = []
        
        # Check for messaging app notifications
        messaging_apps = ["WhatsApp", "Messages", "Discord", "Slack", "Teams"]
        
        for window in state.get("windows", []):
            app_name = window.get("app", "")
            title = window.get("title", "")
            
            # Check if it's a messaging app
            if any(app in app_name for app in messaging_apps):
                # Look for unread indicators in title
                if any(indicator in title.lower() for indicator in ["unread", "new", "("]):
                    notifications.append({
                        "app": app_name,
                        "type": "message",
                        "title": title,
                        "urgency": "medium",
                        "detected_at": datetime.now().isoformat()
                    })
        
        # Add Claude-detected notifications
        if "notifications" in analysis:
            notifications.extend(analysis["notifications"])
        
        return notifications
    
    def _determine_actionable_items(self, analysis: Dict) -> List[Dict]:
        """Determine what actions could be taken"""
        actions = []
        
        # Check for notifications that might need response
        for notification in analysis.get("notifications", []):
            if notification.get("urgency") in ["high", "urgent"]:
                actions.append({
                    "type": "respond_to_notification",
                    "app": notification.get("app"),
                    "reason": "Urgent notification detected",
                    "priority": "high"
                })
        
        # Add Claude-suggested actions
        if "suggested_actions" in analysis:
            actions.extend(analysis["suggested_actions"])
        
        return actions
    
    def _has_significant_change(self, current_analysis: Dict) -> bool:
        """Determine if there's a significant change worth reporting"""
        if not self.last_screen_state:
            return True
        
        # Check for new notifications
        current_notifications = set(
            (n["app"], n["type"]) for n in current_analysis.get("notifications", [])
        )
        last_notifications = set(
            (n["app"], n["type"]) for n in self.last_screen_state.get("notifications", [])
        )
        
        if current_notifications != last_notifications:
            return True
        
        # Check for focus change
        if current_analysis.get("focused_app") != self.last_screen_state.get("focused_app"):
            return True
        
        # Check for significant window count change
        current_count = current_analysis.get("window_count", 0)
        last_count = self.last_screen_state.get("window_count", 0)
        
        if abs(current_count - last_count) > 3:
            return True
        
        # Check for new actionable items
        if len(current_analysis.get("actionable_items", [])) > len(
            self.last_screen_state.get("actionable_items", [])
        ):
            return True
        
        return False
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        if not image:
            return ""
        
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    
    async def analyze_current_screen(self) -> Dict[str, Any]:
        """Analyze current screen immediately"""
        state = await self._capture_screen_state()
        return await self._analyze_screen_state(state)
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_interval": self.monitoring_interval,
            "last_analysis": self.last_screen_state.get("timestamp") if self.last_screen_state else None,
            "notifications_detected": len(self.notification_history),
            "callbacks_registered": {
                "update": len(self.update_callbacks),
                "action": len(self.action_callbacks)
            }
        }