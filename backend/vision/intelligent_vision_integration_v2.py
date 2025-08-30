#!/usr/bin/env python3
"""
Intelligent Vision Integration for JARVIS with Event-Driven Architecture
Combines screen capture permissions with Claude's intelligence
Now with loose coupling through event bus
"""

import os
import asyncio
import time
from typing import Optional, Dict, Any, List
import logging

from .screen_vision import ScreenVisionSystem, JARVISVisionIntegration
from .screen_capture_fallback import capture_with_intelligence
from .claude_vision_analyzer import ClaudeVisionAnalyzer

# EVENT BUS INTEGRATION
from core.event_bus import Event, EventPriority, get_event_bus
from core.event_types import (
    EventTypes, EventBuilder, VisionEvents, SystemEvents, MemoryEvents,
    subscribe_to, subscribe_to_pattern
)

logger = logging.getLogger(__name__)

class IntelligentJARVISVision:
    """Enhanced JARVIS vision that combines permissions with intelligence and events"""
    
    def __init__(self):
        self.vision_system = ScreenVisionSystem()
        self.jarvis_vision = JARVISVisionIntegration(self.vision_system)
        
        # EVENT BUS SETUP
        self.event_bus = get_event_bus()
        self.event_builder = EventBuilder()
        
        # Check for Claude API
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.claude_analyzer = None
        if self.api_key:
            self.claude_analyzer = ClaudeVisionAnalyzer(self.api_key)
            
        # Vision state tracking
        self.last_capture_time = 0
        self.capture_cache = {}
        self.active_analyses = {}
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        # Publish startup event
        SystemEvents.startup(
            source="intelligent_vision",
            version="2.0",
            config={
                "claude_available": bool(self.api_key),
                "screen_access": self._check_permissions()
            }
        )
            
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for vision system"""
        
        # Subscribe to voice commands that need vision
        @subscribe_to(EventTypes.VOICE_COMMAND_RECEIVED)
        async def handle_voice_command(event: Event):
            command = event.payload.get("command", "").lower()
            confidence = event.payload.get("confidence", 0)
            
            # Check if command needs vision
            vision_keywords = [
                "see", "look", "screen", "analyze", "show", "what's on",
                "check", "read", "describe", "tell me about"
            ]
            
            if any(keyword in command for keyword in vision_keywords):
                logger.info(f"Vision activated by voice command: {command}")
                
                # Process vision command
                response = await self.handle_intelligent_command(command)
                
                # Publish response event
                self.event_builder.publish(
                    "vision.response_generated",
                    source="intelligent_vision",
                    payload={
                        "command": command,
                        "response": response,
                        "confidence": confidence
                    }
                )
                
        # Subscribe to memory pressure events
        @subscribe_to(EventTypes.MEMORY_PRESSURE_CHANGED)
        async def handle_memory_pressure(event: Event):
            new_level = event.payload.get("new_level")
            
            if new_level == "critical":
                # Clear vision cache
                self.capture_cache.clear()
                logger.info("Cleared vision cache due to critical memory pressure")
                
                # Publish cache cleared event
                MemoryEvents.cache_cleared(
                    source="intelligent_vision",
                    cache_type="capture_cache",
                    size_mb=0  # Would calculate actual size
                )
            elif new_level == "high":
                # Reduce cache size
                if len(self.capture_cache) > 5:
                    # Keep only 5 most recent
                    items = sorted(self.capture_cache.items(), 
                                 key=lambda x: x[1].get("timestamp", 0))
                    self.capture_cache = dict(items[-5:])
                    
        # Subscribe to system errors
        @subscribe_to(EventTypes.SYSTEM_ERROR, priority=EventPriority.HIGH)
        async def handle_system_error(event: Event):
            error = event.payload.get("error", "")
            if "vision" in error.lower() or "screen" in error.lower():
                logger.error(f"Vision-related error: {error}")
                # Clear any stuck analyses
                self.active_analyses.clear()
                
        # Subscribe to workflow events
        @subscribe_to_pattern("control.workflow_*")
        async def handle_workflow_events(event: Event):
            if event.type == EventTypes.CONTROL_WORKFLOW_STARTED:
                workflow = event.payload.get("workflow")
                if "vision" in event.payload.get("components", []):
                    # Prepare for vision operations
                    logger.info(f"Vision preparing for workflow: {workflow}")
                    
    def _check_permissions(self) -> bool:
        """Check if we have screen recording permissions"""
        try:
            # Try a quick capture to test permissions
            result = capture_with_intelligence(
                query="test",
                use_claude=False
            )
            return result.get("success", False)
        except:
            return False
            
    async def handle_intelligent_command(self, command: str) -> str:
        """
        Handle vision commands with enhanced intelligence and event publishing
        
        This method provides intelligent responses while publishing events
        for system-wide coordination.
        """
        start_time = time.time()
        command_lower = command.lower()
        
        # Generate unique analysis ID
        analysis_id = f"analysis_{int(time.time() * 1000)}"
        
        # Special handling for "can you see my screen?"
        if "can you see my screen" in command_lower or "what's on my screen" in command_lower:
            # Publish screen capture start event
            VisionEvents.screen_captured(
                source="intelligent_vision",
                display_id="main",
                capture_metadata={
                    "analysis_id": analysis_id,
                    "command": command,
                    "method": "claude_api" if self.api_key else "fallback"
                }
            )
            
            # If Claude is available, use it directly
            if self.api_key:
                try:
                    # Check cache first
                    cache_key = "screen_analysis"
                    if cache_key in self.capture_cache:
                        cached = self.capture_cache[cache_key]
                        if time.time() - cached["timestamp"] < 5:  # 5 second cache
                            logger.info("Using cached screen analysis")
                            return cached["response"]
                    
                    # Track active analysis
                    self.active_analyses[analysis_id] = {
                        "start_time": start_time,
                        "command": command
                    }
                    
                    # Get Claude's analysis directly
                    claude_result = capture_with_intelligence(
                        query="Please analyze this screenshot and describe what the user appears to be working on. Be specific about the applications open, the content visible, and any relevant details you can see.",
                        use_claude=True
                    )
                    
                    # Remove from active analyses
                    del self.active_analyses[analysis_id]
                    
                    processing_time = time.time() - start_time
                    
                    if claude_result.get("intelligence_used") and claude_result.get("analysis"):
                        response = f"Yes sir, I can see your screen. {claude_result['analysis']}"
                        
                        # Cache the response
                        self.capture_cache[cache_key] = {
                            "response": response,
                            "timestamp": time.time()
                        }
                        
                        # Publish analysis complete event
                        VisionEvents.analysis_complete(
                            source="intelligent_vision",
                            analysis_type="screen_overview",
                            results={
                                "analysis_id": analysis_id,
                                "summary": claude_result['analysis'],
                                "success": True
                            },
                            duration=processing_time
                        )
                        
                        return response
                    elif not claude_result["success"]:
                        # Publish permission denied event
                        VisionEvents.permission_denied(
                            source="intelligent_vision",
                            reason="screen_recording_permission",
                            suggestion="Grant permission in System Preferences"
                        )
                        
                        return (
                            "I'm unable to see your screen at the moment, sir. "
                            "Please grant me screen recording permission in "
                            "System Preferences → Security & Privacy → Privacy → Screen Recording."
                        )
                    else:
                        # Publish error event
                        VisionEvents.error(
                            source="intelligent_vision",
                            error="Analysis failed",
                            details={"analysis_id": analysis_id}
                        )
                        
                        return "I can see your screen but encountered an error analyzing it. Please try again."
                except Exception as e:
                    logger.error(f"Vision analysis error: {e}")
                    
                    # Publish error event
                    SystemEvents.error(
                        source="intelligent_vision",
                        error=f"Vision analysis failed: {str(e)}",
                        details={"analysis_id": analysis_id, "command": command}
                    )
                    
                    return f"I encountered an error accessing your screen: {str(e)}"
            else:
                # No Claude API key
                VisionEvents.error(
                    source="intelligent_vision",
                    error="No API key configured",
                    details={"required": "ANTHROPIC_API_KEY"}
                )
                
                return (
                    "I need an Anthropic API key to analyze your screen. "
                    "Please configure ANTHROPIC_API_KEY in your environment."
                )
            
        # For other vision commands, use enhanced processing
        elif any(phrase in command_lower for phrase in [
            "what's on my screen", "analyze my screen", "what do you see",
            "check for updates", "look for updates",
            "what am i working", "what i'm working", "working on",
            "what are you seeing", "describe what you see",
            "tell me what", "show me what"
        ]):
            # Use the intelligent capture if Claude is available
            if self.api_key:
                # Track active analysis
                self.active_analyses[analysis_id] = {
                    "start_time": start_time,
                    "command": command
                }
                
                # Publish capture event
                VisionEvents.screen_captured(
                    source="intelligent_vision",
                    display_id="main",
                    capture_metadata={
                        "analysis_id": analysis_id,
                        "command": command,
                        "query_type": self._determine_query_type(command_lower)
                    }
                )
                
                # Create a more specific query based on the user's intent
                if "working" in command_lower:
                    query = "Analyze what the user is currently working on based on the open applications, visible windows, and content. Be specific about the applications, files, and tasks visible."
                    analysis_type = "work_context"
                elif "update" in command_lower:
                    query = "Check for any software update notifications, badges, or update-related windows on the screen."
                    analysis_type = "update_check"
                else:
                    query = f"Analyze the screen and respond to this request: {command}"
                    analysis_type = "general_analysis"
                
                result = capture_with_intelligence(
                    query=query,
                    use_claude=True
                )
                
                # Remove from active analyses
                del self.active_analyses[analysis_id]
                
                processing_time = time.time() - start_time
                
                if result.get("intelligence_used") and result.get("analysis"):
                    # Detect objects/applications mentioned
                    objects_detected = self._extract_objects(result['analysis'])
                    
                    # Publish analysis complete event
                    VisionEvents.analysis_complete(
                        source="intelligent_vision",
                        analysis_type=analysis_type,
                        results={
                            "analysis_id": analysis_id,
                            "summary": result['analysis'],
                            "objects_detected": objects_detected,
                            "success": True
                        },
                        duration=processing_time
                    )
                    
                    # Publish object detection events for each detected item
                    for obj in objects_detected:
                        VisionEvents.object_detected(
                            source="intelligent_vision",
                            object_type=obj["type"],
                            object_name=obj["name"],
                            confidence=obj.get("confidence", 0.9),
                            metadata={"analysis_id": analysis_id}
                        )
                    
                    return f"Sir, {result['analysis']}"
                else:
                    VisionEvents.error(
                        source="intelligent_vision",
                        error="Analysis failed",
                        details={
                            "analysis_id": analysis_id,
                            "analysis_type": analysis_type
                        }
                    )
                    
            # Fallback to standard vision handling
            response = await self.jarvis_vision.handle_vision_command(command)
            
            # Publish response event
            self.event_builder.publish(
                "vision.command_handled",
                source="intelligent_vision",
                payload={
                    "command": command,
                    "response": response,
                    "method": "fallback"
                }
            )
            
            return response
            
        # Default to standard vision handling
        return await self.jarvis_vision.handle_vision_command(command)
    
    def _determine_query_type(self, command: str) -> str:
        """Determine the type of vision query"""
        if "working" in command:
            return "work_context"
        elif "update" in command:
            return "update_check"
        elif "read" in command:
            return "text_extraction"
        elif "color" in command or "design" in command:
            return "visual_analysis"
        else:
            return "general"
            
    def _extract_objects(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract mentioned objects/applications from analysis text"""
        objects = []
        
        # Common application names to look for
        app_keywords = [
            "VS Code", "Visual Studio Code", "Chrome", "Safari", "Firefox",
            "Terminal", "Finder", "Slack", "Discord", "Spotify", "Mail",
            "Calendar", "Notes", "Preview", "Xcode", "Docker"
        ]
        
        # UI elements
        ui_keywords = [
            "window", "tab", "button", "menu", "dialog", "notification",
            "sidebar", "toolbar", "panel"
        ]
        
        analysis_lower = analysis_text.lower()
        
        # Check for applications
        for app in app_keywords:
            if app.lower() in analysis_lower:
                objects.append({
                    "type": "application",
                    "name": app,
                    "confidence": 0.9
                })
                
        # Check for UI elements
        for element in ui_keywords:
            if element in analysis_lower:
                objects.append({
                    "type": "ui_element",
                    "name": element,
                    "confidence": 0.8
                })
                
        # Check for file types
        file_extensions = [".py", ".js", ".json", ".yaml", ".md", ".txt"]
        for ext in file_extensions:
            if ext in analysis_lower:
                objects.append({
                    "type": "file",
                    "name": f"{ext} file",
                    "confidence": 0.85
                })
                
        return objects
        
    async def capture_and_analyze_with_events(self, query: str, 
                                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Capture and analyze screen with full event integration
        
        This method is for programmatic use and returns structured data
        while publishing appropriate events.
        """
        analysis_id = f"analysis_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Publish capture start event
        VisionEvents.screen_captured(
            source="intelligent_vision",
            display_id="main",
            capture_metadata={
                "analysis_id": analysis_id,
                "query": query,
                "has_context": bool(context)
            }
        )
        
        try:
            # Perform capture and analysis
            if self.api_key:
                result = capture_with_intelligence(
                    query=query,
                    use_claude=True
                )
                
                processing_time = time.time() - start_time
                
                if result.get("success") and result.get("analysis"):
                    # Extract structured data
                    objects = self._extract_objects(result['analysis'])
                    
                    # Publish analysis complete
                    VisionEvents.analysis_complete(
                        source="intelligent_vision",
                        analysis_type="programmatic",
                        results={
                            "analysis_id": analysis_id,
                            "summary": result['analysis'],
                            "objects_detected": objects,
                            "success": True
                        },
                        duration=processing_time
                    )
                    
                    return {
                        "success": True,
                        "analysis_id": analysis_id,
                        "analysis": result['analysis'],
                        "objects": objects,
                        "processing_time": processing_time
                    }
                else:
                    raise Exception("Analysis failed")
            else:
                raise Exception("No API key configured")
                
        except Exception as e:
            # Publish error event
            VisionEvents.error(
                source="intelligent_vision",
                error=str(e),
                details={
                    "analysis_id": analysis_id,
                    "query": query
                }
            )
            
            return {
                "success": False,
                "analysis_id": analysis_id,
                "error": str(e)
            }
            
    def get_active_analyses(self) -> List[Dict[str, Any]]:
        """Get list of currently active analyses"""
        return [
            {
                "analysis_id": aid,
                "command": info["command"],
                "duration": time.time() - info["start_time"]
            }
            for aid, info in self.active_analyses.items()
        ]
        
    def clear_cache(self):
        """Clear vision cache"""
        size_before = len(self.capture_cache)
        self.capture_cache.clear()
        
        # Publish cache cleared event
        MemoryEvents.cache_cleared(
            source="intelligent_vision",
            cache_type="capture_cache",
            size_mb=0,  # Would calculate actual size
            items_cleared=size_before
        )
        
        logger.info(f"Cleared {size_before} cached items")

async def test_intelligent_vision():
    """Test the intelligent vision responses with event integration"""
    
    print("🧠 Testing Intelligent JARVIS Vision with Events")
    print("=" * 50)
    
    # Get event bus stats before
    event_bus = get_event_bus()
    stats_before = event_bus.get_stats()
    
    vision = IntelligentJARVISVision()
    
    # Test the key command
    test_commands = [
        "Hey JARVIS, can you see my screen?",
        "What's on my screen?",
        "Check for software updates",
        "Analyze what I'm working on"
    ]
    
    for command in test_commands:
        print(f"\n🎤 You: {command}")
        response = await vision.handle_intelligent_command(command)
        print(f"🤖 JARVIS: {response}")
        print("-" * 50)
        
    # Show event bus stats after
    stats_after = event_bus.get_stats()
    print(f"\n📊 Event Bus Statistics:")
    print(f"   Events published: {stats_after['published'] - stats_before['published']}")
    print(f"   Events processed: {stats_after['processed'] - stats_before['processed']}")
    
    # Show active analyses
    active = vision.get_active_analyses()
    if active:
        print(f"\n🔄 Active Analyses: {len(active)}")
        for analysis in active:
            print(f"   - {analysis['analysis_id']}: {analysis['command']}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_vision())