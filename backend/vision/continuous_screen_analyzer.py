#!/usr/bin/env python3
"""
Continuous Screen Analyzer for JARVIS
Provides real-time screen monitoring with Claude Vision integration
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class ContinuousScreenAnalyzer:
    """
    Continuous screen monitoring system that integrates with Claude Vision
    Enables JARVIS to always see and understand what's on screen
    """
    
    def __init__(self, vision_handler, update_interval: float = 2.0):
        """
        Initialize continuous screen analyzer
        
        Args:
            vision_handler: The vision action handler for Claude Vision
            update_interval: How often to capture screen (in seconds)
        """
        self.vision_handler = vision_handler
        self.update_interval = update_interval
        self.is_monitoring = False
        self._monitoring_task = None
        
        # Screen state tracking
        self.current_screen_state = {
            'last_capture': None,
            'last_analysis': None,
            'current_app': None,
            'visible_elements': [],
            'context': {},
            'timestamp': None
        }
        
        # Callbacks for different events
        self.event_callbacks = {
            'app_changed': [],
            'content_changed': [],
            'weather_visible': [],
            'error_detected': [],
            'user_needs_help': []
        }
        
        # Performance optimization
        self.cache_duration = 5.0  # Cache analysis for 5 seconds
        self._analysis_cache = {}
        
        logger.info("Continuous Screen Analyzer initialized")
    
    async def start_monitoring(self):
        """Start continuous screen monitoring"""
        if self.is_monitoring:
            logger.warning("Screen monitoring is already active")
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started continuous screen monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous screen monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous screen monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Capture and analyze screen
                await self._capture_and_analyze()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _capture_and_analyze(self):
        """Capture screen and analyze with Claude Vision"""
        try:
            # Capture current screen
            capture_result = await self.vision_handler.capture_screen()
            if not capture_result.success:
                return
            
            current_time = time.time()
            
            # Quick analysis to detect changes
            quick_analysis = await self._quick_screen_analysis()
            
            # Determine if we need full analysis
            needs_full_analysis = self._needs_full_analysis(quick_analysis)
            
            if needs_full_analysis:
                # Perform full Claude Vision analysis
                analysis = await self._full_screen_analysis()
                
                # Update screen state
                self._update_screen_state(analysis)
                
                # Trigger relevant callbacks
                await self._process_screen_events(analysis)
            
        except Exception as e:
            logger.error(f"Error capturing/analyzing screen: {e}")
    
    async def _quick_screen_analysis(self) -> Dict[str, Any]:
        """Perform quick analysis to detect major changes"""
        # This could use a lightweight ML model or simple heuristics
        # For now, we'll use Claude Vision with a simple prompt
        params = {
            'query': 'What application is currently in focus? Just name the app, nothing else.'
        }
        
        result = await self.vision_handler.describe_screen(params)
        
        return {
            'current_app': result.description if result.success else 'Unknown',
            'timestamp': time.time()
        }
    
    async def _full_screen_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive screen analysis"""
        # Check cache first
        cache_key = f"full_analysis_{int(time.time() / self.cache_duration)}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Comprehensive analysis prompt
        params = {
            'query': '''Analyze the current screen and provide:
1. Currently active application
2. Key UI elements visible
3. Any weather information if Weather app is visible
4. Any error messages or dialogs
5. What the user appears to be doing
6. Any text content that might be relevant

Be concise but thorough.'''
        }
        
        result = await self.vision_handler.describe_screen(params)
        
        analysis = {
            'success': result.success,
            'description': result.description if result.success else '',
            'timestamp': time.time(),
            'raw_data': result.data if hasattr(result, 'data') else {}
        }
        
        # Cache the result
        self._analysis_cache[cache_key] = analysis
        
        return analysis
    
    def _needs_full_analysis(self, quick_analysis: Dict[str, Any]) -> bool:
        """Determine if full analysis is needed"""
        # Always analyze if no previous state
        if not self.current_screen_state['last_analysis']:
            return True
        
        # Check if app changed
        if quick_analysis.get('current_app') != self.current_screen_state.get('current_app'):
            return True
        
        # Check if enough time has passed
        last_analysis_time = self.current_screen_state.get('timestamp', 0)
        if time.time() - last_analysis_time > 10.0:  # Full analysis every 10 seconds
            return True
        
        return False
    
    def _update_screen_state(self, analysis: Dict[str, Any]):
        """Update internal screen state"""
        self.current_screen_state.update({
            'last_analysis': analysis.get('description', ''),
            'timestamp': analysis.get('timestamp', time.time()),
            'current_app': self._extract_current_app(analysis)
        })
    
    def _extract_current_app(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Extract current application from analysis"""
        description = analysis.get('description', '').lower()
        
        # Common app detection patterns
        apps = {
            'weather': ['weather app', 'weather.app'],
            'safari': ['safari'],
            'chrome': ['chrome', 'google chrome'],
            'vscode': ['vs code', 'visual studio code', 'vscode'],
            'terminal': ['terminal', 'iterm'],
            'finder': ['finder'],
            'mail': ['mail app', 'mail.app'],
            'messages': ['messages', 'imessage']
        }
        
        for app_name, keywords in apps.items():
            if any(keyword in description for keyword in keywords):
                return app_name
        
        return None
    
    async def _process_screen_events(self, analysis: Dict[str, Any]):
        """Process screen events and trigger callbacks"""
        description = analysis.get('description', '').lower()
        
        # Check for weather visibility
        if 'weather' in description and any(word in description for word in ['temperature', 'degrees', '°']):
            await self._trigger_event('weather_visible', {
                'analysis': analysis,
                'weather_info': self._extract_weather_info(description)
            })
        
        # Check for errors
        if any(word in description for word in ['error', 'failed', 'exception', 'crash']):
            await self._trigger_event('error_detected', {
                'analysis': analysis,
                'error_context': description
            })
    
    def _extract_weather_info(self, description: str) -> Optional[str]:
        """Extract weather information from screen description"""
        # Use the weather parser we created
        from utils.weather_response_parser import WeatherResponseParser
        parser = WeatherResponseParser()
        return parser.extract_weather_info(description)
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event callbacks"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for specific events"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
    
    async def get_current_screen_context(self) -> Dict[str, Any]:
        """Get current screen context for queries"""
        # If we have recent analysis, return it
        if self.current_screen_state['last_analysis']:
            age = time.time() - self.current_screen_state['timestamp']
            if age < self.cache_duration:
                return self.current_screen_state
        
        # Otherwise, do a fresh analysis
        analysis = await self._full_screen_analysis()
        self._update_screen_state(analysis)
        return self.current_screen_state
    
    async def query_screen_for_weather(self) -> Optional[str]:
        """
        Query screen specifically for weather information
        This is what we'll use when user asks about weather
        """
        # First check if Weather app is already visible
        context = await self.get_current_screen_context()
        
        if context.get('current_app') == 'weather':
            # Weather app is already open, just read it
            params = {
                'query': 'Read the weather information from the Weather app. What is the temperature, conditions, and forecast?'
            }
        else:
            # Need to open Weather app first
            from system_control import MacOSController
            controller = MacOSController()
            
            # Open Weather app
            controller.open_application("Weather")
            await asyncio.sleep(2.0)  # Wait for it to open
            
            # Now read the weather
            params = {
                'query': 'The Weather app should now be open. Read the weather information: temperature, conditions, and forecast for today.'
            }
        
        result = await self.vision_handler.describe_screen(params)
        
        if result.success:
            # Parse the weather info
            from utils.weather_response_parser import WeatherResponseParser
            parser = WeatherResponseParser()
            weather_info = parser.extract_weather_info(result.description)
            
            # Format nicely
            return parser.format_weather_response(weather_info)
        
        return None