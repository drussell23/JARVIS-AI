#!/usr/bin/env python3
"""
Simplified Vision System using only Claude Vision API
Removes all local ML models for faster performance
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Optional, Any
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path
import os

from utils.centralized_model_manager import model_manager
from vision.screen_vision import ScreenVisionSystem

logger = logging.getLogger(__name__)


class SimplifiedVisionSystem:
    """
    Streamlined vision system using only Claude API
    No local ML models = faster startup and response
    """
    
    def __init__(self):
        """Initialize simplified vision system"""
        # Get Claude Vision from centralized manager
        self.claude_analyzer = model_manager.get_claude_vision_analyzer()
        self.enabled = self.claude_analyzer is not None
        
        # Basic screen capture (no OCR, no ML)
        self.screen_vision = ScreenVisionSystem()
        
        # Track performance
        self.last_response_time = None
        self.total_requests = 0
        self.total_response_time = 0
        
        if self.enabled:
            logger.info("Simplified Vision System initialized with Claude API")
        else:
            logger.warning("Claude Vision not available - vision features disabled")
    
    async def analyze_screen(self, query: str = None) -> Dict[str, Any]:
        """
        Analyze screen using Claude Vision
        
        Args:
            query: Optional specific question about the screen
            
        Returns:
            Analysis results from Claude
        """
        start_time = datetime.now()
        
        if not self.enabled:
            return {
                "success": False,
                "message": "Vision system not available - please configure ANTHROPIC_API_KEY",
                "response_time": 0
            }
        
        try:
            # Capture screenshot
            screenshot = self.screen_vision.capture_screen()
            if screenshot is None:
                return {
                    "success": False,
                    "message": "Unable to capture screen - check permissions",
                    "response_time": 0
                }
            
            # Convert to numpy array for Claude
            screenshot_np = np.array(screenshot)
            
            # Default query if none provided
            if not query:
                query = "Please describe what you see on the screen, including open applications and what the user appears to be working on."
            
            # Send to Claude for analysis
            result = await self.claude_analyzer.analyze_screenshot(screenshot_np, query)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            self.last_response_time = response_time
            self.total_requests += 1
            self.total_response_time += response_time
            
            return {
                "success": True,
                "analysis": result.get("description", ""),
                "confidence": result.get("confidence", 0.9),
                "response_time": response_time,
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def find_element(self, element_description: str) -> Dict[str, Any]:
        """Find UI element on screen using Claude"""
        query = f"Please locate the following element on the screen: {element_description}. Provide its location if visible."
        return await self.analyze_screen(query)
    
    async def read_text(self, area_description: Optional[str] = None) -> Dict[str, Any]:
        """Read text from screen or specific area"""
        if area_description:
            query = f"Please read and transcribe the text in the {area_description} area of the screen."
        else:
            query = "Please read and transcribe all visible text on the screen."
        return await self.analyze_screen(query)
    
    async def check_for_notifications(self) -> Dict[str, Any]:
        """Check for system notifications or updates"""
        query = "Are there any notifications, alerts, or software update prompts visible on the screen?"
        return await self.analyze_screen(query)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_response = self.total_response_time / max(self.total_requests, 1)
        
        return {
            "enabled": self.enabled,
            "total_requests": self.total_requests,
            "average_response_time": f"{avg_response:.2f}s",
            "last_response_time": f"{self.last_response_time:.2f}s" if self.last_response_time else "N/A",
            "backend": "claude_vision_api",
            "local_models": "none",
            "memory_usage": "minimal"
        }
    
    async def capture_and_describe(self) -> str:
        """Quick method to capture and describe screen (for compatibility)"""
        result = await self.analyze_screen()
        
        if result["success"]:
            return f"Yes sir, I can see your screen. {result['analysis']}"
        else:
            return f"I'm having trouble seeing your screen: {result['message']}"


# Global instance for easy access
_vision_system = None


def get_vision_system() -> SimplifiedVisionSystem:
    """Get or create the global vision system instance"""
    global _vision_system
    if _vision_system is None:
        _vision_system = SimplifiedVisionSystem()
    return _vision_system