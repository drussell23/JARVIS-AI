"""
Real-time Vision Monitor with Automation Triggers
Continuous monitoring with configurable change detection and automation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of automation triggers"""
    TEXT_APPEARS = "text_appears"
    TEXT_DISAPPEARS = "text_disappears"
    COLOR_CHANGE = "color_change"
    MOTION_DETECTED = "motion"
    APP_OPENS = "app_opens"
    APP_CLOSES = "app_closes"
    NOTIFICATION = "notification"
    UPDATE_AVAILABLE = "update"
    IDLE_DETECTED = "idle"
    CUSTOM = "custom"

@dataclass
class TriggerCondition:
    """Condition for triggering automation"""
    trigger_type: TriggerType
    parameters: Dict[str, Any]
    callback: Callable
    active: bool = True
    cooldown_seconds: int = 0
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass 
class MonitoringRegion:
    """Region to monitor on screen"""
    name: str
    x: int
    y: int
    width: int
    height: int
    sensitivity: float = 0.05  # Change threshold
    monitor_index: int = 0  # For multi-monitor support

class ChangeDetector:
    """Detect various types of changes in images"""
    
    @staticmethod
    def calculate_difference(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate normalized difference between images"""
        if img1.shape != img2.shape:
            # Resize to match
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale for comparison
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Normalize to 0-1 range
        return np.mean(diff) / 255.0
    
    @staticmethod
    def detect_motion(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect motion between frames"""
        diff_score = ChangeDetector.calculate_difference(img1, img2)
        
        if diff_score > threshold:
            # Find areas of change
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find contours of changed areas
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_areas = []
            for contour in contours[:5]:  # Top 5 areas
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
            
            return {
                "motion_detected": True,
                "difference_score": float(diff_score),
                "motion_areas": motion_areas
            }
        
        return {
            "motion_detected": False,
            "difference_score": float(diff_score)
        }
    
    @staticmethod
    def detect_color_change(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect significant color changes"""
        # Calculate average color for each image
        avg_color1 = np.mean(img1.reshape(-1, 3), axis=0)
        avg_color2 = np.mean(img2.reshape(-1, 3), axis=0)
        
        # Calculate color difference
        color_diff = np.linalg.norm(avg_color1 - avg_color2) / 255.0
        
        return {
            "color_changed": color_diff > threshold,
            "color_difference": float(color_diff),
            "previous_color": avg_color1.tolist(),
            "current_color": avg_color2.tolist()
        }

class RealtimeVisionMonitor:
    """Real-time vision monitoring with automation triggers"""
    
    def __init__(self, vision_system: Any):
        """Initialize with a vision system instance"""
        self.vision_system = vision_system
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Monitoring configuration
        self.monitoring_interval = 1.0  # seconds
        self.regions: Dict[str, MonitoringRegion] = {}
        self.triggers: List[TriggerCondition] = []
        
        # State tracking
        self.last_screenshots: Dict[str, np.ndarray] = {}
        self.last_analysis: Dict[str, Any] = {}
        self.monitoring_stats = {
            "total_frames": 0,
            "changes_detected": 0,
            "triggers_fired": 0,
            "start_time": None,
            "errors": 0
        }
        
        # Change detector
        self.change_detector = ChangeDetector()
    
    def add_region(self, region: MonitoringRegion):
        """Add a region to monitor"""
        self.regions[region.name] = region
        logger.info(f"Added monitoring region: {region.name}")
    
    def add_trigger(self, trigger: TriggerCondition):
        """Add an automation trigger"""
        self.triggers.append(trigger)
        logger.info(f"Added trigger: {trigger.trigger_type.value}")
    
    def remove_trigger(self, trigger_type: TriggerType):
        """Remove triggers of a specific type"""
        self.triggers = [t for t in self.triggers if t.trigger_type != trigger_type]
    
    async def start_monitoring(self, 
                             interval: float = 1.0,
                             full_screen: bool = True):
        """Start real-time monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_interval = interval
        self.monitoring_stats["start_time"] = datetime.now()
        
        # Add full screen region if requested
        if full_screen and "full_screen" not in self.regions:
            # Get screen dimensions
            try:
                import pyautogui
                width, height = pyautogui.size()
                self.add_region(MonitoringRegion(
                    name="full_screen",
                    x=0, y=0,
                    width=width,
                    height=height,
                    sensitivity=0.05
                ))
            except:
                logger.error("Could not determine screen size")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started real-time monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Calculate runtime
        if self.monitoring_stats["start_time"]:
            runtime = datetime.now() - self.monitoring_stats["start_time"]
            logger.info(f"Monitoring stopped. Runtime: {runtime}, Frames: {self.monitoring_stats['total_frames']}")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.monitoring_stats["total_frames"] += 1
                
                # Monitor each region
                for region_name, region in self.regions.items():
                    await self._monitor_region(region_name, region)
                
                # Check for idle
                await self._check_idle_state()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self.monitoring_stats["errors"] += 1
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_region(self, region_name: str, region: MonitoringRegion):
        """Monitor a specific region"""
        try:
            # Capture region screenshot
            screenshot = await self.vision_system._capture_screenshot({
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height
            })
            
            if screenshot is None:
                return
            
            # Check for changes
            if region_name in self.last_screenshots:
                last_screenshot = self.last_screenshots[region_name]
                
                # Detect motion
                motion_result = self.change_detector.detect_motion(
                    last_screenshot, screenshot, region.sensitivity
                )
                
                if motion_result["motion_detected"]:
                    self.monitoring_stats["changes_detected"] += 1
                    
                    # Analyze the change
                    analysis = await self._analyze_change(
                        region_name, screenshot, motion_result
                    )
                    
                    # Process triggers
                    await self._process_triggers(region_name, analysis)
            
            # Update last screenshot
            self.last_screenshots[region_name] = screenshot
            
        except Exception as e:
            logger.error(f"Error monitoring region {region_name}: {e}")
    
    async def _analyze_change(self, 
                            region_name: str,
                            screenshot: np.ndarray,
                            motion_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what changed in the region"""
        # Use vision system to understand the change
        prompt = "What changed in this image? Focus on: new text, notifications, app changes."
        
        try:
            analysis = await self.vision_system.analyze_screen(
                prompt=prompt,
                analysis_type="activity",
                use_cache=False  # Don't cache for real-time monitoring
            )
            
            # Combine with motion detection results
            analysis["motion_info"] = motion_result
            analysis["region"] = region_name
            analysis["timestamp"] = datetime.now()
            
            # Store last analysis
            self.last_analysis[region_name] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "motion_info": motion_result,
                "region": region_name,
                "timestamp": datetime.now()
            }
    
    async def _process_triggers(self, region_name: str, analysis: Dict[str, Any]):
        """Process automation triggers based on analysis"""
        for trigger in self.triggers:
            if not trigger.active:
                continue
            
            # Check cooldown
            if trigger.last_triggered and trigger.cooldown_seconds > 0:
                if datetime.now() - trigger.last_triggered < timedelta(seconds=trigger.cooldown_seconds):
                    continue
            
            # Check trigger conditions
            should_fire = await self._check_trigger_condition(trigger, region_name, analysis)
            
            if should_fire:
                # Fire trigger
                trigger.trigger_count += 1
                trigger.last_triggered = datetime.now()
                self.monitoring_stats["triggers_fired"] += 1
                
                try:
                    # Execute callback
                    if asyncio.iscoroutinefunction(trigger.callback):
                        await trigger.callback(region_name, analysis)
                    else:
                        trigger.callback(region_name, analysis)
                        
                    logger.info(f"Fired trigger: {trigger.trigger_type.value}")
                    
                except Exception as e:
                    logger.error(f"Trigger callback error: {e}")
    
    async def _check_trigger_condition(self,
                                     trigger: TriggerCondition,
                                     region_name: str,
                                     analysis: Dict[str, Any]) -> bool:
        """Check if a trigger condition is met"""
        params = trigger.parameters
        
        if trigger.trigger_type == TriggerType.TEXT_APPEARS:
            # Check if specific text appeared
            target_text = params.get("text", "").lower()
            description = analysis.get("description", "").lower()
            return target_text in description
        
        elif trigger.trigger_type == TriggerType.APP_OPENS:
            # Check if specific app opened
            target_app = params.get("app", "").lower()
            apps = analysis.get("applications_mentioned", [])
            return any(target_app in app.lower() for app in apps)
        
        elif trigger.trigger_type == TriggerType.NOTIFICATION:
            # Check for notifications
            return "notification" in analysis.get("description", "").lower()
        
        elif trigger.trigger_type == TriggerType.UPDATE_AVAILABLE:
            # Check for updates
            return analysis.get("has_updates", False)
        
        elif trigger.trigger_type == TriggerType.COLOR_CHANGE:
            # Check color change in region
            if region_name in self.last_screenshots and "motion_info" in analysis:
                return analysis["motion_info"].get("difference_score", 0) > params.get("threshold", 0.1)
        
        elif trigger.trigger_type == TriggerType.MOTION_DETECTED:
            # Check for motion
            return analysis.get("motion_info", {}).get("motion_detected", False)
        
        elif trigger.trigger_type == TriggerType.CUSTOM:
            # Custom condition function
            condition_func = params.get("condition")
            if condition_func:
                return condition_func(region_name, analysis)
        
        return False
    
    async def _check_idle_state(self):
        """Check for idle state"""
        # Simple idle detection based on no changes
        if self.monitoring_stats["total_frames"] > 10:  # After initial frames
            recent_changes = self.monitoring_stats["changes_detected"]
            total_frames = self.monitoring_stats["total_frames"]
            
            # If less than 1% changes in last period, consider idle
            if recent_changes / total_frames < 0.01:
                for trigger in self.triggers:
                    if trigger.trigger_type == TriggerType.IDLE_DETECTED:
                        await self._process_triggers("full_screen", {"idle": True})
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        runtime = None
        if self.monitoring_stats["start_time"]:
            runtime = (datetime.now() - self.monitoring_stats["start_time"]).total_seconds()
        
        return {
            "active": self.monitoring_active,
            "runtime_seconds": runtime,
            "total_frames": self.monitoring_stats["total_frames"],
            "changes_detected": self.monitoring_stats["changes_detected"],
            "triggers_fired": self.monitoring_stats["triggers_fired"],
            "errors": self.monitoring_stats["errors"],
            "regions_monitored": len(self.regions),
            "active_triggers": len([t for t in self.triggers if t.active]),
            "fps": self.monitoring_stats["total_frames"] / runtime if runtime else 0
        }
    
    def get_trigger_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all triggers"""
        return [{
            "type": trigger.trigger_type.value,
            "active": trigger.active,
            "trigger_count": trigger.trigger_count,
            "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None,
            "cooldown": trigger.cooldown_seconds
        } for trigger in self.triggers]

# Example automation functions
async def on_notification_detected(region: str, analysis: Dict[str, Any]):
    """Example: Handle notification detection"""
    logger.info(f"Notification detected in {region}: {analysis.get('description', '')}")
    # Could trigger system notification, log, or other action

async def on_app_opened(region: str, analysis: Dict[str, Any]):
    """Example: Handle app opening"""
    apps = analysis.get('applications_mentioned', [])
    logger.info(f"Apps opened: {apps}")
    # Could trigger workspace setup, reminders, etc.

async def on_idle_detected(region: str, analysis: Dict[str, Any]):
    """Example: Handle idle state"""
    logger.info("User appears idle - could trigger screensaver, cleanup, etc.")