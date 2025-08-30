"""
Optimized Vision System - Combines memory efficiency, circuit breaker, and batch processing
Designed for Phase 0C with 16GB RAM constraints
"""

import asyncio
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
from PIL import Image
import logging

from .memory_efficient_vision_analyzer import MemoryEfficientVisionAnalyzer
from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .screen_vision import ScreenVision

logger = logging.getLogger(__name__)

class OptimizedVisionSystem:
    """Main vision system with all optimizations integrated"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize optimized vision system"""
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("No Anthropic API key found - vision features will be limited")
        
        # Initialize components
        self._init_components()
        
        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "api_analyses": 0,
            "cached_analyses": 0,
            "fallback_analyses": 0,
            "average_response_time": 0,
            "memory_peaks": []
        }
        
        # Real-time monitoring state
        self._monitoring_active = False
        self._monitoring_task = None
        self._last_screenshot = None
        self._change_callbacks = []
    
    def _init_components(self):
        """Initialize all vision components"""
        # Memory-efficient analyzer with caching
        if self.api_key:
            self.analyzer = MemoryEfficientVisionAnalyzer(
                api_key=self.api_key,
                cache_dir="./vision_cache",
                max_cache_size_mb=500,
                max_memory_usage_mb=2048
            )
            
            # Circuit breaker for API protection
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=120,
                expected_exception=Exception,
                half_open_max_calls=2
            )
            
            # Set up circuit breaker callbacks
            self.circuit_breaker.set_callbacks(
                on_state_change=self._on_circuit_state_change,
                on_failure=self._on_api_failure
            )
        else:
            self.analyzer = None
            self.circuit_breaker = None
        
        # Screen capture system (always available)
        try:
            self.screen_capture = ScreenVision()
        except:
            logger.warning("Screen capture initialization failed - using fallback")
            self.screen_capture = None
        
        # Batch processing queue
        self._batch_queue = asyncio.Queue(maxsize=10)
        self._batch_processor_task = None
    
    def _on_circuit_state_change(self, old_state, new_state):
        """Handle circuit breaker state changes"""
        logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
        
        if new_state.value == "open":
            # Switch to fallback mode
            logger.warning("API circuit open - switching to fallback analysis")
    
    def _on_api_failure(self):
        """Handle API failures"""
        logger.error("Vision API call failed")
        self.performance_metrics["fallback_analyses"] += 1
    
    async def analyze_screen(self, 
                           prompt: str = "Describe what you see on the screen",
                           analysis_type: str = "ui",
                           use_cache: bool = True,
                           region: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Main method to analyze screen with all optimizations
        
        Args:
            prompt: Analysis prompt
            analysis_type: Type of analysis (text/ui/activity)
            use_cache: Whether to use caching
            region: Optional region to analyze {x, y, width, height}
            
        Returns:
            Analysis results
        """
        start_time = datetime.now()
        self.performance_metrics["total_analyses"] += 1
        
        try:
            # Capture screenshot
            screenshot = await self._capture_screenshot(region)
            
            if not screenshot:
                return {
                    "error": "Failed to capture screenshot",
                    "description": "Unable to access screen content"
                }
            
            # Try Claude Vision API with circuit breaker
            if self.analyzer and self.circuit_breaker:
                try:
                    result = await self.circuit_breaker.call(
                        self.analyzer.analyze_screenshot,
                        screenshot,
                        prompt,
                        analysis_type,
                        use_cache
                    )
                    self.performance_metrics["api_analyses"] += 1
                    
                    # Check if result was from cache
                    if "cache_hit" in str(self.analyzer.metrics):
                        self.performance_metrics["cached_analyses"] += 1
                    
                except CircuitBreakerError:
                    logger.warning("Circuit breaker open - using fallback analysis")
                    result = await self._fallback_analysis(screenshot, prompt)
                except Exception as e:
                    logger.error(f"Vision API error: {e}")
                    result = await self._fallback_analysis(screenshot, prompt)
            else:
                # No API key - use fallback
                result = await self._fallback_analysis(screenshot, prompt)
            
            # Add performance data
            elapsed = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(elapsed)
            
            result["performance"] = {
                "response_time": elapsed,
                "method": "api" if "error" not in result else "fallback",
                "cached": "cache_hit" in str(result)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Vision system error: {e}")
            return {
                "error": str(e),
                "description": "Vision analysis failed"
            }
    
    async def _capture_screenshot(self, region: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
        """Capture screenshot with fallback methods"""
        # Try primary screen capture
        if self.screen_capture:
            try:
                if region:
                    return self.screen_capture.capture_region(
                        region['x'], region['y'], 
                        region['width'], region['height']
                    )
                else:
                    return self.screen_capture.capture_screen()
            except Exception as e:
                logger.error(f"Primary screen capture failed: {e}")
        
        # Fallback to pyautogui
        try:
            import pyautogui
            if region:
                screenshot = pyautogui.screenshot(region=(
                    region['x'], region['y'],
                    region['width'], region['height']
                ))
            else:
                screenshot = pyautogui.screenshot()
            
            return np.array(screenshot)
        except Exception as e:
            logger.error(f"Fallback screen capture failed: {e}")
            return None
    
    async def _fallback_analysis(self, screenshot: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Fallback analysis when API is unavailable"""
        self.performance_metrics["fallback_analyses"] += 1
        
        # Convert to PIL Image
        if isinstance(screenshot, np.ndarray):
            image = Image.fromarray(screenshot.astype(np.uint8))
        else:
            image = screenshot
        
        # Basic analysis
        width, height = image.size
        
        # Simple color analysis
        colors = image.getcolors(maxcolors=100)
        dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5] if colors else []
        
        return {
            "description": "Basic screen analysis (API unavailable)",
            "screen_info": {
                "width": width,
                "height": height,
                "dominant_colors": dominant_colors
            },
            "fallback_mode": True,
            "prompt": prompt
        }
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics"""
        # Update average response time
        total = self.performance_metrics["total_analyses"]
        current_avg = self.performance_metrics["average_response_time"]
        new_avg = (current_avg * (total - 1) + response_time) / total
        self.performance_metrics["average_response_time"] = new_avg
        
        # Track memory peaks
        if self.analyzer:
            memory_mb = self.analyzer._get_memory_usage() / (1024 * 1024)
            self.performance_metrics["memory_peaks"].append({
                "timestamp": datetime.now(),
                "memory_mb": memory_mb
            })
            
            # Keep only last 100 peaks
            if len(self.performance_metrics["memory_peaks"]) > 100:
                self.performance_metrics["memory_peaks"] = self.performance_metrics["memory_peaks"][-100:]
    
    async def batch_analyze_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch analyze multiple screen regions efficiently"""
        if not self.analyzer:
            return [await self._fallback_analysis(None, r.get('prompt', '')) for r in regions]
        
        # Capture full screenshot once
        screenshot = await self._capture_screenshot()
        if not screenshot:
            return [{"error": "Screenshot capture failed"} for _ in regions]
        
        # Use batch processing from analyzer
        try:
            results = await self.circuit_breaker.call(
                self.analyzer.batch_analyze_regions,
                screenshot,
                regions
            )
            return results
        except (CircuitBreakerError, Exception) as e:
            logger.error(f"Batch analysis failed: {e}")
            return [await self._fallback_analysis(screenshot, r.get('prompt', '')) for r in regions]
    
    async def start_continuous_monitoring(self, 
                                        interval: float = 5.0,
                                        change_threshold: float = 0.05,
                                        on_change: Optional[callable] = None):
        """Start continuous screen monitoring with change detection"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        if on_change:
            self._change_callbacks.append(on_change)
        
        async def monitor():
            while self._monitoring_active:
                try:
                    # Capture current screen
                    current_screenshot = await self._capture_screenshot()
                    
                    if current_screenshot is not None and self._last_screenshot is not None:
                        # Analyze with change detection
                        if self.analyzer:
                            result = await self.analyzer.analyze_with_change_detection(
                                current_screenshot,
                                self._last_screenshot,
                                "Describe any significant changes on the screen",
                                threshold=change_threshold
                            )
                            
                            if result.get("changed", False):
                                # Notify callbacks
                                for callback in self._change_callbacks:
                                    await callback(result)
                    
                    self._last_screenshot = current_screenshot
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                
                await asyncio.sleep(interval)
        
        self._monitoring_task = asyncio.create_task(monitor())
        logger.info(f"Started continuous monitoring with {interval}s interval")
    
    async def stop_continuous_monitoring(self):
        """Stop continuous screen monitoring"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self._change_callbacks.clear()
        logger.info("Stopped continuous monitoring")
    
    async def analyze_multi_monitor(self) -> List[Dict[str, Any]]:
        """Analyze all connected monitors"""
        results = []
        
        # Get monitor information
        try:
            import screeninfo
            monitors = screeninfo.get_monitors()
            
            for i, monitor in enumerate(monitors):
                region = {
                    "x": monitor.x,
                    "y": monitor.y,
                    "width": monitor.width,
                    "height": monitor.height
                }
                
                result = await self.analyze_screen(
                    prompt=f"Analyze content on monitor {i+1}",
                    region=region
                )
                
                result["monitor_info"] = {
                    "index": i,
                    "name": monitor.name,
                    "primary": monitor.is_primary,
                    "resolution": f"{monitor.width}x{monitor.height}"
                }
                
                results.append(result)
                
        except ImportError:
            logger.warning("screeninfo not available - analyzing primary display only")
            result = await self.analyze_screen("Analyze the primary display")
            results.append(result)
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = {
            "performance": self.performance_metrics,
            "circuit_breaker": self.circuit_breaker.get_health_status() if self.circuit_breaker else None,
            "cache": self.analyzer.get_metrics() if self.analyzer else None,
            "monitoring": {
                "active": self._monitoring_active,
                "callbacks_registered": len(self._change_callbacks)
            }
        }
        
        return metrics
    
    async def cleanup(self):
        """Clean up resources and cache"""
        # Stop monitoring
        await self.stop_continuous_monitoring()
        
        # Clean up old cache
        if self.analyzer:
            removed = self.analyzer.cleanup_old_cache(days=1)
            logger.info(f"Cleaned up {removed} old cache entries")
        
        # Cancel batch processor if running
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Vision system cleanup completed")

# Singleton instance
_vision_system_instance = None

def get_vision_system() -> OptimizedVisionSystem:
    """Get or create the singleton vision system instance"""
    global _vision_system_instance
    if _vision_system_instance is None:
        _vision_system_instance = OptimizedVisionSystem()
    return _vision_system_instance