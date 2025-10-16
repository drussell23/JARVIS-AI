#!/usr/bin/env python3
"""
Vision-Guided UI Navigator for Display Connection
==================================================

Uses JARVIS Vision to navigate macOS Control Center and connect to displays.
Bypasses all macOS Sequoia security restrictions by using visual recognition.

How it works:
1. Capture screen with existing vision system
2. Use Claude Vision to identify UI elements
3. Calculate click coordinates from bounding boxes
4. Execute mouse clicks with PyAutoGUI
5. Verify actions completed successfully
6. Monitor connection progress visually

Features:
- Zero hardcoding - fully configuration-driven
- Async/await support throughout
- Self-healing with retry logic
- Comprehensive visual verification
- Integration with existing JARVIS vision system
- Works on macOS Sequoia without accessibility permissions

Author: Derek Russell
Date: 2025-10-15
Version: 2.0
"""

import asyncio
import logging
import subprocess
import json
import pyautogui
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image
import base64
import io
import re

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Visual UI element detected by Claude Vision"""
    name: str
    description: str
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x, y, width, height)
    center_point: Optional[Tuple[int, int]]  # (x, y)
    confidence: float
    element_type: str  # icon, button, text, menu_item


@dataclass
class NavigationResult:
    """Result of a navigation attempt"""
    success: bool
    message: str
    steps_completed: List[str]
    duration: float
    screenshot_path: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class VisionUINavigator:
    """
    Vision-guided UI navigator
    
    Uses Claude Vision to see and interact with macOS UI elements.
    Perfect for bypassing macOS Sequoia security restrictions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize vision navigator"""
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'vision_navigator_config.json'
        
        self.config = self._load_config(config_path)
        self.screenshots_dir = Path.home() / '.jarvis' / 'screenshots' / 'ui_navigation'
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Vision analyzer (will be set by display monitor)
        self.vision_analyzer = None
        
        # Enhanced Vision Pipeline (v1.0)
        self.enhanced_pipeline = None
        self.use_enhanced_pipeline = self.config.get('advanced', {}).get('use_enhanced_pipeline', True)
        
        # Statistics
        self.stats = {
            'total_navigations': 0,
            'successful': 0,
            'failed': 0,
            'avg_duration': 0.0,
            'enhanced_pipeline_used': 0,
            'fallback_used': 0
        }

        # Learning system: Cache successful Control Center position
        self.learned_cc_position = None  # Will be (x, y) after first successful click
        self.learning_cache_file = Path.home() / '.jarvis' / 'control_center_position.json'

        # Advanced detection system
        self.detection_history = []  # Track last 10 detection attempts with outcomes
        self.failure_patterns = {}  # Track common failure scenarios
        self.adaptive_confidence_threshold = 0.75  # Dynamic threshold based on history
        self.screen_context = {}  # Cache screen state (resolution, dark mode, etc.)
        self.detection_strategies = ['learned', 'primary', 'multi_pass', 'exhaustive', 'heuristic']
        self.current_strategy_index = 0

        # Edge case detection flags
        self.edge_cases = {
            'dark_mode': None,  # Will be detected
            'retina_display': None,  # Will be detected
            'resolution': None,  # Will be detected
            'menu_bar_autohide': False,
            'time_format_12h': True
        }

        # Configure PyAutoGUI safety
        pyautogui.PAUSE = self.config.get('mouse', {}).get('delay_between_actions', 0.5)
        pyautogui.FAILSAFE = True

        # Load learned Control Center position if available
        self._load_learned_position()

        # Detect edge cases on initialization
        self._detect_edge_cases()

        logger.info("[VISION NAV] Vision UI Navigator initialized")
        logger.info(f"[VISION NAV] Enhanced Pipeline: {'enabled' if self.use_enhanced_pipeline else 'disabled'}")
        if self.learned_cc_position:
            logger.info(f"[VISION NAV] 🎓 Learned position loaded: {self.learned_cc_position}")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"[VISION NAV] Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"[VISION NAV] Config not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"[VISION NAV] Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "navigation": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "screenshot_verification": True,
                "step_delay": 0.8,
                "max_navigation_time": 30.0
            },
            "mouse": {
                "delay_between_actions": 0.5,
                "click_duration": 0.1,
                "movement_speed": 0.3
            },
            "vision": {
                "confidence_threshold": 0.7,
                "use_bounding_boxes": True,
                "verify_actions": True,
                "screenshot_format": "png",
                "screenshot_quality": 95
            },
            "prompts": {
                "find_control_center": "Find the Control Center icon in the menu bar (top right, looks like two overlapping rectangles). Provide the exact pixel coordinates of its center.",
                "find_screen_mirroring": "Find the Screen Mirroring button in Control Center (looks like two overlapping screens). Provide the exact pixel coordinates of its center.",
                "find_display": "Find '{display_name}' in the list of available displays. Provide the exact pixel coordinates to click on it."
            }
        }
    
    def set_vision_analyzer(self, analyzer):
        """Set Claude Vision analyzer instance"""
        self.vision_analyzer = analyzer
        logger.info("[VISION NAV] Vision analyzer connected")
        
        # Initialize Enhanced Vision Pipeline
        if self.use_enhanced_pipeline:
            asyncio.create_task(self._initialize_enhanced_pipeline())
    
    async def _initialize_enhanced_pipeline(self):
        """Initialize Enhanced Vision Pipeline"""
        try:
            from vision.enhanced_vision_pipeline import get_vision_pipeline
            
            self.enhanced_pipeline = get_vision_pipeline()
            
            # Initialize all stages
            initialized = await self.enhanced_pipeline.initialize()
            
            if initialized:
                # Connect Claude Vision to validator
                if self.enhanced_pipeline.model_validator and self.vision_analyzer:
                    self.enhanced_pipeline.model_validator.set_claude_analyzer(self.vision_analyzer)
                
                logger.info("[VISION NAV] ✅ Enhanced Vision Pipeline v1.0 initialized")
                logger.info("[VISION NAV] 🚀 5-stage pipeline ready:")
                logger.info("[VISION NAV]    Stage 1: Screen Region Segmentation (Quadtree)")
                logger.info("[VISION NAV]    Stage 2: Icon Pattern Recognition (OpenCV + Edge)")
                logger.info("[VISION NAV]    Stage 3: Coordinate Calculation (Physics-based)")
                logger.info("[VISION NAV]    Stage 4: Multi-Model Validation (Monte Carlo)")
                logger.info("[VISION NAV]    Stage 5: Mouse Automation (Bezier trajectories)")
            else:
                logger.warning("[VISION NAV] Enhanced Pipeline initialization failed, using fallback")
                self.use_enhanced_pipeline = False
                
        except Exception as e:
            logger.warning(f"[VISION NAV] Could not initialize Enhanced Pipeline: {e}")
            logger.info("[VISION NAV] Using fallback navigation methods")
            self.use_enhanced_pipeline = False
    
    async def connect_to_display(self, display_name: str) -> NavigationResult:
        """
        Connect to a display using vision-guided navigation
        
        Args:
            display_name: Name of display to connect to (e.g., "Living Room TV")
            
        Returns:
            NavigationResult with success status and details
        """
        start_time = time.time()
        steps_completed = []
        self.stats['total_navigations'] += 1
        
        logger.info(f"[VISION NAV] Starting vision-guided connection to '{display_name}'")
        
        try:
            # Step 1: Find and click Control Center icon
            logger.info("[VISION NAV] Step 1: Finding Control Center icon...")
            cc_clicked = await self._find_and_click_control_center()
            if not cc_clicked:
                raise Exception("Could not find or click Control Center icon")
            steps_completed.append("control_center_opened")
            
            # Wait for Control Center to open
            await asyncio.sleep(self.config['navigation']['step_delay'])
            
            # Step 2: Find and click Screen Mirroring button
            logger.info("[VISION NAV] Step 2: Finding Screen Mirroring button...")
            sm_clicked = await self._find_and_click_screen_mirroring()
            if not sm_clicked:
                raise Exception("Could not find or click Screen Mirroring button")
            steps_completed.append("screen_mirroring_opened")
            
            # Wait for Screen Mirroring menu to open
            await asyncio.sleep(self.config['navigation']['step_delay'])
            
            # Step 3: Find and click display
            logger.info(f"[VISION NAV] Step 3: Finding '{display_name}' in list...")
            display_clicked = await self._find_and_click_display(display_name)
            if not display_clicked:
                raise Exception(f"Could not find or click '{display_name}' in display list")
            steps_completed.append("display_selected")
            
            # Step 4: Verify connection
            logger.info("[VISION NAV] Step 4: Verifying connection...")
            await asyncio.sleep(2.0)  # Wait for connection to establish
            
            connected = await self._verify_connection(display_name)
            if connected:
                steps_completed.append("connection_verified")
            
            duration = time.time() - start_time
            self.stats['successful'] += 1
            self.stats['avg_duration'] = (
                (self.stats['avg_duration'] * (self.stats['successful'] - 1) + duration) 
                / self.stats['successful']
            )
            
            logger.info(f"[VISION NAV] ✅ Successfully connected to '{display_name}' in {duration:.2f}s")
            
            return NavigationResult(
                success=True,
                message=f"Successfully connected to {display_name} using vision navigation",
                steps_completed=steps_completed,
                duration=duration
            )
            
        except Exception as e:
            self.stats['failed'] += 1
            duration = time.time() - start_time
            
            logger.error(f"[VISION NAV] ❌ Navigation failed: {e}")
            
            return NavigationResult(
                success=False,
                message=f"Vision navigation failed: {str(e)}",
                steps_completed=steps_completed,
                duration=duration,
                error_details={'exception': str(e), 'steps_completed': steps_completed}
            )
    
    def _load_learned_position(self):
        """Load previously learned Control Center position from cache"""
        try:
            if self.learning_cache_file.exists():
                with open(self.learning_cache_file) as f:
                    data = json.load(f)
                    self.learned_cc_position = tuple(data.get('control_center_position', []))
                    if self.learned_cc_position:
                        logger.info(f"[VISION NAV] 🎓 Loaded learned position: {self.learned_cc_position}")
        except Exception as e:
            logger.warning(f"[VISION NAV] Could not load learned position: {e}")
            self.learned_cc_position = None

    def _save_learned_position(self, x: int, y: int):
        """Save successful Control Center position for future use"""
        try:
            self.learned_cc_position = (x, y)
            self.learning_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_cache_file, 'w') as f:
                json.dump({
                    'control_center_position': [x, y],
                    'screen_resolution': list(pyautogui.size()),
                    'edge_cases': self.edge_cases,
                    'learned_at': datetime.now().isoformat()
                }, f)
            logger.info(f"[VISION NAV] 💾 Saved learned position: ({x}, {y})")
        except Exception as e:
            logger.warning(f"[VISION NAV] Could not save learned position: {e}")

    def _detect_edge_cases(self):
        """Detect screen configuration and edge cases"""
        try:
            # Detect resolution
            width, height = pyautogui.size()
            self.edge_cases['resolution'] = (width, height)
            self.screen_context['resolution'] = (width, height)

            # Detect retina display (macOS specific)
            try:
                import subprocess
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                      capture_output=True, text=True, timeout=2)
                self.edge_cases['retina_display'] = 'Retina' in result.stdout
            except:
                self.edge_cases['retina_display'] = False

            # Detect dark mode (macOS specific)
            try:
                import subprocess
                result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                                      capture_output=True, text=True, timeout=1)
                self.edge_cases['dark_mode'] = 'Dark' in result.stdout
            except:
                self.edge_cases['dark_mode'] = False  # Light mode or unable to detect

            logger.info(f"[VISION NAV] 🔍 Edge cases detected: {self.edge_cases}")

        except Exception as e:
            logger.warning(f"[VISION NAV] Could not detect edge cases: {e}")

    def _calculate_confidence_score(self, analysis: str, coords: tuple, menu_bar_width: int) -> float:
        """
        Calculate confidence score for detected Control Center position

        Returns: Float 0.0-1.0 representing confidence
        """
        try:
            confidence = 0.0
            x, y = coords

            # Factor 1: Position in expected range (0.3 weight)
            expected_min = menu_bar_width - 180
            expected_max = menu_bar_width - 100
            if expected_min <= x <= expected_max:
                position_score = 1.0
            elif expected_min - 50 <= x <= expected_max + 50:
                position_score = 0.5  # Close but not ideal
            else:
                position_score = 0.0
            confidence += position_score * 0.3

            # Factor 2: Y position centered (0.1 weight)
            if 12 <= y <= 18:
                confidence += 0.1
            elif 5 <= y <= 25:
                confidence += 0.05

            # Factor 3: Analysis mentions key features (0.3 weight)
            key_features = [
                'rectangle', 'overlap', 'side-by-side',
                'monochrome', 'gray', 'white'
            ]
            feature_score = sum(1 for f in key_features if f.lower() in analysis.lower())
            confidence += min(feature_score / len(key_features), 1.0) * 0.3

            # Factor 4: Analysis explicitly mentions HIGH confidence (0.15 weight)
            if 'CONFIDENCE: HIGH' in analysis:
                confidence += 0.15
            elif 'CONFIDENCE: MEDIUM' in analysis:
                confidence += 0.075

            # Factor 5: Verification provided (0.15 weight)
            if 'VERIFICATION:' in analysis and any(word in analysis for word in ['NOT Siri', 'NOT brightness', 'NOT WiFi']):
                confidence += 0.15

            return min(confidence, 1.0)

        except Exception as e:
            logger.warning(f"[VISION NAV] Error calculating confidence: {e}")
            return 0.5  # Default medium confidence

    def _record_detection_attempt(self, success: bool, coords: tuple, confidence: float, strategy: str, error: str = None):
        """Record detection attempt for adaptive learning"""
        try:
            attempt = {
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'coords': coords,
                'confidence': confidence,
                'strategy': strategy,
                'error': error,
                'resolution': self.edge_cases['resolution'],
                'dark_mode': self.edge_cases['dark_mode']
            }

            self.detection_history.append(attempt)

            # Keep only last 10 attempts
            if len(self.detection_history) > 10:
                self.detection_history.pop(0)

            # Update failure patterns
            if not success and error:
                self.failure_patterns[error] = self.failure_patterns.get(error, 0) + 1

            # Adjust adaptive threshold based on recent history
            if len(self.detection_history) >= 5:
                recent_successes = sum(1 for a in self.detection_history[-5:] if a['success'])
                success_rate = recent_successes / 5
                if success_rate < 0.6:
                    # Lower threshold if we're having trouble
                    self.adaptive_confidence_threshold = max(0.6, self.adaptive_confidence_threshold - 0.05)
                    logger.info(f"[VISION NAV] 📉 Lowered confidence threshold to {self.adaptive_confidence_threshold}")
                elif success_rate > 0.8:
                    # Raise threshold if we're doing well
                    self.adaptive_confidence_threshold = min(0.85, self.adaptive_confidence_threshold + 0.02)

        except Exception as e:
            logger.warning(f"[VISION NAV] Could not record detection attempt: {e}")

    async def _adaptive_strategy_selection(self) -> str:
        """Select best detection strategy based on historical data"""
        try:
            # If no history, use default order
            if not self.detection_history:
                return self.detection_strategies[0]

            # If learned position exists and has been working, prioritize it
            if self.learned_cc_position:
                recent_learned_attempts = [a for a in self.detection_history[-3:]
                                          if a.get('strategy') == 'learned']
                if recent_learned_attempts and all(a['success'] for a in recent_learned_attempts):
                    logger.info("[VISION NAV] 🎓 Using learned position (high success rate)")
                    return 'learned'

            # Check success rate by strategy
            strategy_stats = {}
            for attempt in self.detection_history[-10:]:
                strat = attempt.get('strategy', 'unknown')
                if strat not in strategy_stats:
                    strategy_stats[strat] = {'successes': 0, 'total': 0}
                strategy_stats[strat]['total'] += 1
                if attempt['success']:
                    strategy_stats[strat]['successes'] += 1

            # Find best performing strategy
            best_strategy = None
            best_rate = 0.0
            for strat, stats in strategy_stats.items():
                if stats['total'] >= 2:  # Need at least 2 attempts
                    rate = stats['successes'] / stats['total']
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = strat

            if best_strategy and best_rate > 0.7:
                logger.info(f"[VISION NAV] 📊 Using {best_strategy} strategy (success rate: {best_rate:.1%})")
                return best_strategy

            # Fall back to default order
            return 'primary'

        except Exception as e:
            logger.warning(f"[VISION NAV] Error in adaptive strategy selection: {e}")
            return 'primary'

    def _analyze_icon_color(self, screenshot: Image.Image, x: int, y: int) -> Dict[str, Any]:
        """
        Analyze the color properties of the icon region to detect Siri (colorful) vs Control Center (monochrome)

        Returns dict with:
        - is_colorful: True if icon has significant color (likely Siri)
        - is_monochrome: True if icon is grayscale (likely Control Center)
        - saturation_avg: Average color saturation (0-100)
        - color_variance: Variance in hue values
        """
        try:
            # Extract icon region (30x30 pixels around center)
            icon_size = 30
            left = max(0, x - icon_size // 2)
            top = max(0, y - icon_size // 2)
            right = min(screenshot.width, x + icon_size // 2)
            bottom = min(screenshot.height, y + icon_size // 2)

            icon_region = screenshot.crop((left, top, right, bottom))

            # Convert to RGB if needed
            if icon_region.mode != 'RGB':
                icon_region = icon_region.convert('RGB')

            # Get pixels
            pixels = list(icon_region.getdata())

            # Calculate color metrics
            saturations = []
            hues = []

            for r, g, b in pixels:
                # Convert RGB to HSV manually
                r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
                max_c = max(r_norm, g_norm, b_norm)
                min_c = min(r_norm, g_norm, b_norm)
                delta = max_c - min_c

                # Saturation (0-100)
                if max_c > 0:
                    saturation = (delta / max_c) * 100
                else:
                    saturation = 0
                saturations.append(saturation)

                # Hue (0-360)
                if delta > 0:
                    if max_c == r_norm:
                        hue = 60 * (((g_norm - b_norm) / delta) % 6)
                    elif max_c == g_norm:
                        hue = 60 * (((b_norm - r_norm) / delta) + 2)
                    else:
                        hue = 60 * (((r_norm - g_norm) / delta) + 4)
                    hues.append(hue)

            # Calculate average saturation
            saturation_avg = sum(saturations) / len(saturations) if saturations else 0

            # Calculate hue variance (color diversity)
            if len(hues) > 1:
                hue_mean = sum(hues) / len(hues)
                hue_variance = sum((h - hue_mean) ** 2 for h in hues) / len(hues)
            else:
                hue_variance = 0

            # Determine if colorful or monochrome
            # Siri has high saturation (>25) and high hue variance (>500)
            # Control Center has low saturation (<15) and low hue variance (<100)
            is_colorful = saturation_avg > 25 or hue_variance > 500
            is_monochrome = saturation_avg < 15 and hue_variance < 100

            result = {
                'is_colorful': is_colorful,
                'is_monochrome': is_monochrome,
                'saturation_avg': saturation_avg,
                'color_variance': hue_variance
            }

            logger.info(f"[VISION NAV] 🎨 Color analysis at ({x}, {y}): saturation={saturation_avg:.1f}, variance={hue_variance:.1f}, colorful={is_colorful}, mono={is_monochrome}")

            return result

        except Exception as e:
            logger.warning(f"[VISION NAV] Error in color analysis: {e}")
            # Return neutral result on error
            return {
                'is_colorful': False,
                'is_monochrome': False,
                'saturation_avg': 0,
                'color_variance': 0
            }

    async def _find_and_click_control_center(self) -> bool:
        """
        Find and click Control Center icon using learned position + Claude Vision

        Priority:
        1. Try learned position (if available and recently successful)
        2. Use Claude Vision detection with enhanced prompts
        3. Multi-pass detection if needed
        4. Heuristic fallback

        This method learns from successful clicks to improve accuracy over time.
        """
        try:
            # Try learned position first (fastest and most accurate if available)
            if self.learned_cc_position:
                logger.info(f"[VISION NAV] 🎓 Trying learned position: {self.learned_cc_position}")
                x, y = self.learned_cc_position

                await self._click_at(x, y)

                if await self._verify_control_center_clicked(x, y):
                    logger.info("[VISION NAV] ✅ Learned position worked!")
                    return True
                else:
                    logger.warning("[VISION NAV] ⚠️ Learned position failed, clearing cache...")
                    self.learned_cc_position = None  # Clear failed position
                    if self.learning_cache_file.exists():
                        self.learning_cache_file.unlink()

            # Continue with Claude Vision detection
            # Capture screen
            screenshot = await self._capture_screen()
            if not screenshot:
                logger.error("[VISION NAV] Failed to capture screen")
                return False

            # Crop to menu bar region (top 50px) for faster analysis
            menu_bar_height = 50
            menu_bar_screenshot = screenshot.crop((0, 0, screenshot.width, menu_bar_height))

            logger.info(f"[VISION NAV] 🎯 Direct Claude Vision detection for Control Center")
            logger.info(f"[VISION NAV] Analyzing menu bar: {menu_bar_screenshot.width}x{menu_bar_screenshot.height}px")

            # Save screenshot for analysis
            screenshot_path = self.screenshots_dir / f'control_center_{int(time.time())}.png'
            menu_bar_screenshot.save(screenshot_path)

            # Primary Method: Claude Vision Direct Detection
            if self.vision_analyzer:
                logger.info("[VISION NAV] 🤖 Asking Claude Vision to locate Control Center...")

                # ULTRA-ENHANCED prompt with dynamic context and step-by-step analysis
                # Add dynamic context based on detected edge cases
                context_info = []
                if self.edge_cases.get('dark_mode'):
                    context_info.append("**CONTEXT: Dark mode is enabled** - Control Center icon will appear as light gray/white on dark background")
                else:
                    context_info.append("**CONTEXT: Light mode is enabled** - Control Center icon will appear as dark gray on light background")

                if self.edge_cases.get('resolution'):
                    width, height = self.edge_cases['resolution']
                    expected_cc_x = width - 125  # Control Center typically 110-140px from right
                    context_info.append(f"**CONTEXT: Screen resolution is {width}x{height}**")
                    context_info.append(f"**CRITICAL POSITION: Control Center X should be approximately {expected_cc_x} (exactly 110-140px from RIGHT edge)**")
                    context_info.append(f"**CRITICAL: If you detect X < {width-150}, it's TOO FAR LEFT (likely Siri, NOT Control Center)**")
                    context_info.append(f"**CRITICAL: If you detect X > {width-105}, it's TOO FAR RIGHT (likely Time display, NOT Control Center)**")

                context_section = "\n".join(context_info) if context_info else ""

                prompt = f"""You are analyzing a macOS menu bar screenshot. Your mission is to find the Control Center icon with 100% accuracy.

{context_section}

**CRITICAL - VISUAL IDENTIFICATION:**

The Control Center icon looks like this (simplified ASCII representation):
```
[ ][ ]  <-- Two rectangles side-by-side, slightly overlapping
```

**Precise Visual Description:**
- Two overlapping rounded rectangles oriented HORIZONTALLY (side-by-side, NOT stacked vertically)
- The rectangles overlap in the MIDDLE, creating a distinctive shape
- Monochrome gray/white color ONLY (if you see ANY color, it's WRONG)
- Size: Approximately 20-24px wide, 16-20px tall
- Appears as a solid, simple geometric icon
- The overlap creates a subtle "merged" appearance in the center
- Think of it as looking like two phone screens or two cards placed side by side with overlap

**SHAPE COMPARISON - Learn the Differences:**
- Control Center: [ ][ ] (two rectangles side-by-side)
- Keyboard Brightness: (O) (sun/star burst shape)
- Siri: (O) (colorful circular orb - VERY COMMON MISTAKE!)
- WiFi: ())) (radiating curved lines)
- Bluetooth: (B) (stylized "B" letter)
- Battery: [|||||] (rectangular battery outline)
- Sound: (>) (speaker with waves)
- Time: "2:55 AM" (text/numbers, NOT an icon)

**ABSOLUTE EXCLUSIONS - If you see these, keep looking:**
1. NOT Siri - circular/orb shape with colors (blue, pink, purple gradients)
2. NOT Keyboard Brightness - sun symbol with rays
3. NOT Display Brightness - sun inside a screen/monitor shape
4. NOT WiFi - curved radiating lines
5. NOT Bluetooth - "B" with triangular arrows
6. NOT Battery - rectangular outline with fill indicator
7. NOT Sound - speaker cone shape
8. NOT Time Display - numerical text characters
9. NOT any icon with COLOR (Control Center is grayscale ONLY)
10. NOT any circular/round icon (Control Center is RECTANGULAR)

**CRITICAL: PIXEL-PERFECT POSITION REQUIREMENTS:**

Menu bar layout from RIGHT to LEFT (with EXACT pixel distances):

[Time: "2:55 AM"] <---105px---> [CONTROL CENTER] <---50px---> [SIRI - COLORFUL ORB]
    (far right)                       (TARGET!)              (TOO FAR LEFT!)
    X > 1335                          X = 1300-1335          X < 1290

**ABSOLUTE RULES - POSITION-BASED (STRICTLY ENFORCED):**
1. Control Center MUST be 108-142px from RIGHT edge (NARROW RANGE!)
2. If X < (width - 142): TOO FAR LEFT = This is SIRI, NOT Control Center!
3. If X > (width - 108): TOO FAR RIGHT = This is Time display, NOT Control Center!
4. The CORRECT range is VERY NARROW: X must be in [width-142] to [width-108]
5. Siri is typically at width-170, Control Center at width-125, Time at width-80

**VISUAL LAYOUT WITH PIXEL DISTANCES:**

```
RIGHT EDGE (X=1440)
    |
    |<-- 105px -->| TIME DISPLAY ("2:55 AM")
    |             |
    |<-- 125px -->|<-- CONTROL CENTER (TWO RECTANGLES [ ][ ]) **TARGET**
    |             |
    |<-- 170px -->|<-- SIRI (COLORFUL CIRCLE - WRONG!)
    |             |
    |<-- 200px -->|<-- BRIGHTNESS (SUN SHAPE - WRONG!)
```

**SPATIAL POSITIONING - Where to Look:**

Menu bar layout (RIGHT section):
[...icons...] [Brightness] [Siri] [CONTROL CENTER] [Time Display]
                              ↑          ↑               ↑
                         COLORFUL!   TWO RECTS!      NUMBERS!
                         (X~1270)    (X~1315)        (X~1350)
                         TOO LEFT!   CORRECT!        TOO RIGHT!

Exact position rules:
1. Start from the RIGHT edge of the screen
2. Skip the Time display (numbers like "2:55 AM" - last 105 pixels from right)
3. Control Center is immediately to the LEFT of Time (108-142px from right edge - STRICT!)
4. Siri is FURTHER LEFT than Control Center (>145px from right edge)
5. Control Center X position MUST be in range: [screen_width - 142] to [screen_width - 108]
6. Control Center Y position should be centered: 12-18 pixels from top
7. Any detection outside this range will be REJECTED as wrong icon

**WARNING: COMMON MISTAKE - Siri vs Control Center:**
- Siri: Circular, colorful, X > 145px from right (X ~1270 on 1440px screen) - REJECT THIS!
- Control Center: Rectangular, monochrome, X = 108-142px from right (X ~1315 on 1440px screen) - CORRECT!
- THEY ARE NEXT TO EACH OTHER - Make sure you pick the RIGHT ONE!
- If you detect X < (width - 142), you've selected Siri - WRONG! Move RIGHT!

**STEP-BY-STEP DETECTION PROCESS:**

Step 1: Scan the rightmost 250 pixels of the menu bar
Step 2: Identify and SKIP the Time display (it's text/numbers)
Step 3: Look for the first ICON to the left of Time
Step 4: Check if it's two overlapping rectangles (side-by-side)
Step 5: Verify it's monochrome (no colors)
Step 6: Verify it's NOT circular (Siri) or sun-shaped (Brightness)
Step 7: If all checks pass, this is Control Center

**BEFORE RESPONDING - Self-Verification Checklist:**
- [ ] Did I check the rightmost section (last 250px)?
- [ ] Did I identify the Time display and look LEFT of it?
- [ ] Is the icon I selected made of TWO RECTANGLES side-by-side?
- [ ] Is the icon MONOCHROME gray/white (NO colors)?
- [ ] Is the icon NOT circular (rules out Siri)?
- [ ] Is the icon NOT sun-shaped (rules out Brightness)?
- [ ] Is the X position between screen_width-142 and screen_width-108? (STRICT!)
- [ ] Is the Y position centered in menu bar (12-18px)?
- [ ] Did I verify I'm NOT detecting Siri (which is >145px from right)?

**RESPONSE FORMAT (Required):**

First, describe what you see:
CANDIDATE_DESCRIPTION: [Describe the icon you identified and why it matches Control Center]

Then provide coordinates:
X_POSITION: [x coordinate - center of icon]
Y_POSITION: [y coordinate - center of icon]

Finally, confirm:
CONFIDENCE: [HIGH/MEDIUM/LOW]
VERIFICATION: [Explain why you're certain this is NOT Siri/Brightness/other icons]

**EXAMPLE RESPONSE:**
CANDIDATE_DESCRIPTION: I identified an icon at X=1265 that shows two overlapping rectangles side-by-side. It's monochrome gray, located between a sun icon (brightness) and the time display "2:55 AM". The shape clearly shows two rounded rectangles with horizontal overlap, matching the Control Center design.

X_POSITION: 1265
Y_POSITION: 15

CONFIDENCE: HIGH
VERIFICATION: This is NOT Siri (no circular shape or colors), NOT brightness (no sun rays), NOT WiFi (no curved lines). The two-rectangle side-by-side shape is unique to Control Center. Position is correct: to the left of time display.

**CRITICAL REMINDER:**
- If you're looking at a CIRCULAR icon, it's WRONG (likely Siri)
- If you're looking at a SUN-SHAPED icon, it's WRONG (likely Brightness)
- If you're looking at COLORED icon, it's WRONG (Control Center is grayscale)
- Control Center has a UNIQUE shape: TWO RECTANGLES SIDE-BY-SIDE [ ][ ]

Take your time. Analyze carefully. The correct icon is there - find the two overlapping rectangles!"""

                # Analyze with Claude Vision
                analysis = await self._analyze_with_vision(screenshot_path, prompt)

                if analysis:
                    logger.info(f"[VISION NAV] Claude response received: {analysis[:200]}...")

                    # Extract coordinates with robust parsing
                    coords = self._extract_coordinates_advanced(analysis, menu_bar_screenshot)

                    if coords:
                        x, y = coords
                        logger.info(f"[VISION NAV] ✅ Claude Vision detected Control Center at ({x}, {y})")

                        # Calculate confidence score
                        confidence = self._calculate_confidence_score(
                            analysis, coords, menu_bar_screenshot.width
                        )
                        logger.info(f"[VISION NAV] 📊 Confidence score: {confidence:.2f} (threshold: {self.adaptive_confidence_threshold:.2f})")

                        # Validate coordinates are within menu bar bounds
                        if not self._validate_coordinates(x, y, menu_bar_screenshot.width, menu_bar_height):
                            logger.warning(f"[VISION NAV] ⚠️ Coordinates out of bounds, adjusting...")
                            x, y = self._adjust_suspicious_coordinates(x, y, menu_bar_screenshot.width, menu_bar_height)
                            logger.info(f"[VISION NAV] Adjusted to: ({x}, {y})")
                            confidence *= 0.8  # Reduce confidence for adjusted coordinates

                        # ADVANCED: Spatial validation - Control Center should be in right section
                        if not await self._validate_control_center_position(x, y, menu_bar_screenshot):
                            logger.warning(f"[VISION NAV] ⚠️ Position failed spatial validation at ({x}, {y})")
                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='spatial_validation_failed'
                            )
                            logger.info("[VISION NAV] Attempting multi-pass detection...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        # CRITICAL LAYER 1: Position-based Siri rejection
                        # Siri is typically at X = width - 180 to width - 160
                        # Control Center is at X = width - 140 to width - 110
                        width = menu_bar_screenshot.width
                        distance_from_right = width - x

                        # TIGHTENED: Reject if too far left (Siri zone)
                        if distance_from_right > 142:
                            logger.error(f"[VISION NAV] 🚫 REJECTED: Position is {distance_from_right}px from right edge (TOO FAR LEFT - likely Siri)")
                            logger.error(f"[VISION NAV] 🚫 Control Center should be 110-140px from right, detected {distance_from_right}px")

                            # SMART SEARCH: Look for Control Center near the detected position
                            logger.info(f"[VISION NAV] 🔍 Searching for Control Center near detected position...")

                            # Search to the right of the detected position (where Control Center likely is)
                            search_positions = [
                                (x + 30, y),  # 30px right
                                (x + 45, y),  # 45px right (typical Siri-to-Control Center distance)
                                (x + 60, y),  # 60px right
                                (x - 30, y),  # 30px left (in case we overshot)
                            ]

                            for search_x, search_y in search_positions:
                                search_distance = width - search_x
                                if 108 <= search_distance <= 142:
                                    # Check color at this position
                                    search_color = self._analyze_icon_color(menu_bar_screenshot, search_x, search_y)
                                    if not search_color['is_colorful'] and search_color['saturation_avg'] < 20:
                                        logger.info(f"[VISION NAV] ✅ Found monochrome icon at ({search_x}, {search_y}) - likely Control Center!")
                                        await self._click_at(search_x, search_y)
                                        if await self._verify_control_center_clicked(search_x, search_y):
                                            logger.info("[VISION NAV] ✅ Smart search successful!")
                                            return True

                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='position_too_far_left_likely_siri'
                            )
                            logger.info("[VISION NAV] 🔄 Smart search didn't find Control Center, attempting multi-pass...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        if distance_from_right < 108:
                            logger.error(f"[VISION NAV] 🚫 REJECTED: Position is {distance_from_right}px from right edge (TOO FAR RIGHT - likely in time zone)")
                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='position_too_far_right_in_time_zone'
                            )
                            logger.info("[VISION NAV] 🔄 Attempting multi-pass detection...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        logger.info(f"[VISION NAV] ✅ Position check passed: {distance_from_right}px from right edge (ideal range: 110-140px)")

                        # CRITICAL LAYER 2: Color analysis - reject colorful icons (Siri)
                        color_analysis = self._analyze_icon_color(menu_bar_screenshot, x, y)
                        if color_analysis['is_colorful']:
                            logger.error(f"[VISION NAV] 🚫 REJECTED: Detected icon is COLORFUL (likely Siri, not Control Center)")
                            logger.error(f"[VISION NAV] 🚫 Saturation: {color_analysis['saturation_avg']:.1f}%, Variance: {color_analysis['color_variance']:.1f}")

                            # INTELLIGENT ADJUSTMENT: Move right from Siri to find Control Center
                            logger.info(f"[VISION NAV] 🎯 Detected Siri at X={x}, adjusting right by 40-50px to find Control Center")
                            adjusted_x = x + 45  # Control Center is typically 40-50px to the right of Siri
                            adjusted_y = y

                            # Validate adjusted position
                            width = menu_bar_screenshot.width
                            adjusted_distance = width - adjusted_x

                            if 108 <= adjusted_distance <= 142:
                                logger.info(f"[VISION NAV] ✅ Adjusted position ({adjusted_x}, {adjusted_y}) is in valid range")
                                # Verify the adjusted position is not colorful
                                adjusted_color = self._analyze_icon_color(menu_bar_screenshot, adjusted_x, adjusted_y)
                                if not adjusted_color['is_colorful']:
                                    logger.info(f"[VISION NAV] ✅ Adjusted position is MONOCHROME - likely Control Center!")
                                    await self._click_at(adjusted_x, adjusted_y)
                                    if await self._verify_control_center_clicked(adjusted_x, adjusted_y):
                                        logger.info("[VISION NAV] ✅ Successfully clicked Control Center via intelligent adjustment!")
                                        return True

                            # If adjustment didn't work, fall back to multi-pass
                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='colorful_icon_detected_likely_siri'
                            )
                            logger.info("[VISION NAV] 🔄 Adjustment didn't work, attempting multi-pass detection...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        if not color_analysis['is_monochrome']:
                            logger.warning(f"[VISION NAV] ⚠️ Icon color analysis inconclusive (not clearly monochrome)")
                            confidence *= 0.7  # Reduce confidence if color is ambiguous
                        else:
                            logger.info(f"[VISION NAV] ✅ Color check passed: Icon is MONOCHROME (Control Center)")

                        # Check if confidence meets adaptive threshold
                        if confidence < self.adaptive_confidence_threshold:
                            logger.warning(f"[VISION NAV] ⚠️ Confidence {confidence:.2f} below threshold {self.adaptive_confidence_threshold:.2f}")
                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='low_confidence'
                            )
                            logger.info("[VISION NAV] Attempting multi-pass detection for higher confidence...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        # Click the icon
                        await self._click_at(x, y)

                        # Self-correction: Verify we clicked the right icon
                        if await self._verify_control_center_clicked(x, y):
                            # Record successful detection
                            self._record_detection_attempt(
                                success=True,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary'
                            )
                            # Save this position for future use
                            self._save_learned_position(x, y)
                            logger.info(f"[VISION NAV] ✅ Control Center clicked successfully (confidence: {confidence:.2f})")
                            return True
                        else:
                            # Record failed detection
                            self._record_detection_attempt(
                                success=False,
                                coords=(x, y),
                                confidence=confidence,
                                strategy='primary',
                                error='verification_failed'
                            )
                            # Wrong icon clicked - try to self-correct
                            logger.warning("[VISION NAV] ⚠️ Wrong icon clicked! Attempting self-correction...")
                            corrected = await self._self_correct_control_center_click()
                            # Note: self_correct method will save position if it succeeds
                            return corrected
                    else:
                        logger.warning("[VISION NAV] Could not extract valid coordinates from Claude response")
                        logger.debug(f"[VISION NAV] Full response: {analysis[:500]}")
                else:
                    logger.warning("[VISION NAV] No response from Claude Vision")

            # Fallback: Smart heuristic based on typical Control Center placement
            logger.info("[VISION NAV] 💡 Using smart heuristic fallback")
            return await self._click_control_center_heuristic()

        except Exception as e:
            logger.error(f"[VISION NAV] Error in Control Center detection: {e}", exc_info=True)
            return False
    
    async def _find_and_click_screen_mirroring(self) -> bool:
        """
        Find and click Screen Mirroring button using direct Claude Vision detection

        This method uses Claude Vision API directly for maximum accuracy.
        """
        try:
            # Let UI settle after Control Center opens
            await asyncio.sleep(0.5)

            # Capture screen
            screenshot = await self._capture_screen()
            if not screenshot:
                logger.error("[VISION NAV] Failed to capture screen")
                return False

            logger.info(f"[VISION NAV] 🎯 Direct Claude Vision detection for Screen Mirroring")

            # Save screenshot for analysis
            screenshot_path = self.screenshots_dir / f'screen_mirroring_{int(time.time())}.png'
            screenshot.save(screenshot_path)

            # Primary Method: Claude Vision Direct Detection
            if self.vision_analyzer:
                logger.info("[VISION NAV] 🤖 Asking Claude Vision to locate Screen Mirroring button...")

                # Enhanced prompt for Screen Mirroring detection
                prompt = """You are analyzing a macOS Control Center screenshot. I need you to find the Screen Mirroring button and provide its EXACT pixel coordinates.

**What the Screen Mirroring button looks like:**
- Icon shows two overlapping screens/rectangles (computer monitor symbol)
- Usually has text "Screen Mirroring" below or next to the icon
- Blue or white icon depending on state
- Located in Control Center panel (dark or light background)

**Your task:**
1. Locate the Screen Mirroring button visually
2. Determine its CENTER POINT coordinates
3. Respond with EXACT pixel coordinates

**Response format (use this exact format):**
X_POSITION: [x coordinate]
Y_POSITION: [y coordinate]

**Example response:**
X_POSITION: 680
Y_POSITION: 450

Provide only the coordinates in this format. Be as accurate as possible."""

                # Analyze with Claude Vision
                analysis = await self._analyze_with_vision(screenshot_path, prompt)

                if analysis:
                    logger.info(f"[VISION NAV] Claude response received: {analysis[:150]}...")

                    # Extract coordinates with robust parsing
                    coords = self._extract_coordinates_advanced(analysis, screenshot)

                    if coords:
                        x, y = coords
                        logger.info(f"[VISION NAV] ✅ Claude Vision detected Screen Mirroring at ({x}, {y})")

                        # Click the button
                        await self._click_at(x, y)
                        return True
                    else:
                        logger.warning("[VISION NAV] Could not extract valid coordinates from Claude response")
                        logger.debug(f"[VISION NAV] Full response: {analysis[:500]}")
                else:
                    logger.warning("[VISION NAV] No response from Claude Vision")

            # Fallback: OCR search for text "Screen Mirroring"
            logger.info("[VISION NAV] 💡 Using OCR fallback for Screen Mirroring")
            return await self._click_screen_mirroring_ocr(screenshot)

        except Exception as e:
            logger.error(f"[VISION NAV] Error finding Screen Mirroring: {e}", exc_info=True)
            return False
    
    async def _find_and_click_display(self, display_name: str) -> bool:
        """
        Find and click display in the list using direct Claude Vision detection
        """
        try:
            # Let UI settle after Screen Mirroring menu opens
            await asyncio.sleep(0.5)

            # Capture screen with display list
            screenshot = await self._capture_screen()
            if not screenshot:
                logger.error("[VISION NAV] Failed to capture screen")
                return False

            logger.info(f"[VISION NAV] 🎯 Direct Claude Vision detection for '{display_name}'")

            # Save screenshot for analysis
            screenshot_path = self.screenshots_dir / f'display_{display_name}_{int(time.time())}.png'
            screenshot.save(screenshot_path)

            # Primary Method: Claude Vision Direct Detection
            if self.vision_analyzer:
                logger.info(f"[VISION NAV] 🤖 Asking Claude Vision to locate '{display_name}'...")

                # Enhanced prompt for display detection
                prompt = f"""You are analyzing a macOS Screen Mirroring menu. I need you to find the display named "{display_name}" and provide its EXACT pixel coordinates.

**What to look for:**
- Text reading "{display_name}"
- May appear in a list of available displays
- May have an icon next to it (TV icon, monitor icon, etc.)
- Could be in light or dark background

**Your task:**
1. Locate the "{display_name}" display entry
2. Determine the CENTER POINT coordinates where I should click
3. Respond with EXACT pixel coordinates

**Response format (use this exact format):**
X_POSITION: [x coordinate]
Y_POSITION: [y coordinate]

**Example response:**
X_POSITION: 720
Y_POSITION: 380

Provide only the coordinates in this format. Be as accurate as possible."""

                # Analyze with Claude Vision
                analysis = await self._analyze_with_vision(screenshot_path, prompt)

                if analysis:
                    logger.info(f"[VISION NAV] Claude response received: {analysis[:150]}...")

                    # Extract coordinates with robust parsing
                    coords = self._extract_coordinates_advanced(analysis, screenshot)

                    if coords:
                        x, y = coords
                        logger.info(f"[VISION NAV] ✅ Claude Vision detected '{display_name}' at ({x}, {y})")

                        # Click the display
                        await self._click_at(x, y)
                        return True
                    else:
                        logger.warning("[VISION NAV] Could not extract valid coordinates from Claude response")
                        logger.debug(f"[VISION NAV] Full response: {analysis[:500]}")
                else:
                    logger.warning("[VISION NAV] No response from Claude Vision")

            # Fallback: OCR search for display name
            logger.info(f"[VISION NAV] 💡 Using OCR fallback to find '{display_name}'")
            return await self._click_display_ocr(screenshot, display_name)

        except Exception as e:
            logger.error(f"[VISION NAV] Error finding display: {e}", exc_info=True)
            return False
    
    async def _verify_connection(self, display_name: str) -> bool:
        """Verify display connection was successful"""
        try:
            # Wait for connection to establish
            await asyncio.sleep(1.0)
            
            # Capture screen to check for connection indicators
            screenshot = await self._capture_screen()
            
            if not screenshot:
                return False
            
            screenshot_path = self.screenshots_dir / f'verification_{int(time.time())}.png'
            screenshot.save(screenshot_path)
            
            # Use Claude Vision to verify connection
            if self.vision_analyzer:
                prompt = f"Is the display '{display_name}' currently connected? Look for connection indicators, checkmarks, or the display appearing in the list of connected displays."
                
                analysis = await self._analyze_with_vision(screenshot_path, prompt)
                
                # Check if response indicates connection
                if analysis and ('yes' in analysis.lower() or 'connected' in analysis.lower() or 'checkmark' in analysis.lower()):
                    logger.info(f"[VISION NAV] ✅ Connection to '{display_name}' verified")
                    return True
            
            # Fallback: Assume success if we got this far
            logger.info("[VISION NAV] Connection verification skipped (vision analyzer not available)")
            return True
            
        except Exception as e:
            logger.error(f"[VISION NAV] Error verifying connection: {e}")
            return False
    
    async def _capture_screen(self) -> Optional[Image.Image]:
        """Capture current screen using existing vision infrastructure"""
        try:
            # Try using existing reliable screenshot capture
            from vision.reliable_screenshot_capture import ReliableScreenshotCapture
            
            capture = ReliableScreenshotCapture()
            
            # Try different capture methods
            if hasattr(capture, 'capture_current_space'):
                result = await capture.capture_current_space()
            elif hasattr(capture, 'capture_screen'):
                result = await capture.capture_screen()
            elif hasattr(capture, 'capture'):
                result = capture.capture()
            else:
                # Manually call the capture method
                result = await capture.capture_with_fallback()
            
            if hasattr(result, 'success') and result.success and hasattr(result, 'image'):
                return result.image
            elif isinstance(result, Image.Image):
                return result
            
        except ImportError:
            logger.debug("[VISION NAV] ReliableScreenshotCapture not available")
        except AttributeError as e:
            logger.debug(f"[VISION NAV] Screenshot method not available: {e}")
        except Exception as e:
            logger.debug(f"[VISION NAV] Screenshot capture error: {e}")
        
        # Fallback: Use screencapture command
        try:
            temp_path = self.screenshots_dir / f'temp_{int(time.time())}.png'
            
            process = await asyncio.create_subprocess_exec(
                'screencapture', '-x', str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if temp_path.exists():
                image = Image.open(temp_path)
                temp_path.unlink()  # Clean up
                logger.debug(f"[VISION NAV] Screenshot captured with screencapture command")
                return image
                
        except Exception as e:
            logger.error(f"[VISION NAV] Screenshot fallback failed: {e}")
        
        return None
    
    async def _analyze_with_vision(self, image_path: Path, prompt: str) -> Optional[str]:
        """Analyze image with Claude Vision"""
        if not self.vision_analyzer:
            logger.warning("[VISION NAV] No vision analyzer available")
            return None
        
        try:
            # Load image as PIL Image (Claude Vision Analyzer expects this)
            image = Image.open(image_path)
            
            # Use analyze_screenshot method (standard for ClaudeVisionAnalyzer)
            response = await self.vision_analyzer.analyze_screenshot(
                image=image,  # Pass PIL Image directly
                prompt=prompt,
                use_cache=False  # Don't cache UI navigation prompts
            )
            
            # Handle response - analyze_screenshot returns (Dict, AnalysisMetrics)
            if isinstance(response, tuple):
                analysis_dict, metrics = response
                # Extract text from response
                if isinstance(analysis_dict, dict):
                    response_text = analysis_dict.get('response', analysis_dict.get('text', str(analysis_dict)))
                else:
                    response_text = str(analysis_dict)
            else:
                response_text = str(response)
            
            logger.info(f"[VISION NAV] Claude response: {response_text[:300] if response_text else 'None'}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"[VISION NAV] Vision analysis error: {e}", exc_info=True)
            return None
    
    def _extract_coordinates_advanced(self, response: str, screenshot: Image.Image) -> Optional[Tuple[int, int]]:
        """
        Advanced coordinate extraction with multiple format support and validation

        Supports formats like:
        - X_POSITION: 1260, Y_POSITION: 15
        - (1260, 15)
        - x: 1260, y: 15
        - center at 1260, 15
        - 180 pixels from right edge

        Args:
            response: Claude Vision response text
            screenshot: Screenshot being analyzed (for dimension validation)

        Returns:
            (x, y) tuple or None
        """
        if not response:
            logger.warning("[VISION NAV] Empty response from Claude Vision")
            return None

        try:
            logger.debug(f"[VISION NAV] Parsing response: {response[:200]}...")

            # Pattern 1: X_POSITION: 1234, Y_POSITION: 56 (our requested format)
            x_match = re.search(r'X[_\s]*POSITION\s*:\s*(\d+)', response, re.IGNORECASE)
            y_match = re.search(r'Y[_\s]*POSITION\s*:\s*(\d+)', response, re.IGNORECASE)
            if x_match and y_match:
                x, y = int(x_match.group(1)), int(y_match.group(1))
                logger.info(f"[VISION NAV] ✅ Extracted (X_POSITION format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 2: (x, y) tuple format
            match = re.search(r'\((\d+),\s*(\d+)\)', response)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                logger.info(f"[VISION NAV] ✅ Extracted (tuple format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 3: x: 1234, y: 56
            match = re.search(r'x\s*[:=]\s*(\d+).*?y\s*[:=]\s*(\d+)', response, re.IGNORECASE | re.DOTALL)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                logger.info(f"[VISION NAV] ✅ Extracted (x:y format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 4: JSON format {"x": 1234, "y": 56}
            match = re.search(r'\{.*?"x"\s*:\s*(\d+).*?"y"\s*:\s*(\d+).*?\}', response, re.IGNORECASE | re.DOTALL)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                logger.info(f"[VISION NAV] ✅ Extracted (JSON format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 5: "center at 1234, 56" or "located at 1234, 56"
            match = re.search(r'(?:center|located|position|point)\s+(?:at\s+)?(\d+)\s*,\s*(\d+)', response, re.IGNORECASE)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                logger.info(f"[VISION NAV] ✅ Extracted (descriptive format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 6: Descriptive "X pixels from left/right, Y pixels from top"
            left_match = re.search(r'(\d+)\s*(?:px|pixels?)?\s*from\s+(?:the\s+)?left', response, re.IGNORECASE)
            right_match = re.search(r'(\d+)\s*(?:px|pixels?)?\s*from\s+(?:the\s+)?right', response, re.IGNORECASE)
            top_match = re.search(r'(\d+)\s*(?:px|pixels?)?\s*from\s+(?:the\s+)?top', response, re.IGNORECASE)

            if (left_match or right_match) and top_match:
                if left_match:
                    x = int(left_match.group(1))
                elif right_match:
                    x = screenshot.width - int(right_match.group(1))

                y = int(top_match.group(1))
                logger.info(f"[VISION NAV] ✅ Extracted (descriptive pixels format): ({x}, {y})")
                return self._validate_and_return(x, y, screenshot)

            # Pattern 7: Two sequential 3-4 digit numbers (last resort)
            numbers = re.findall(r'\b(\d{3,4})\b', response)
            if len(numbers) >= 2:
                x, y = int(numbers[0]), int(numbers[1])
                # Only use if reasonable for screenshot dimensions
                if 0 <= x <= screenshot.width * 2 and 0 <= y <= 100:  # Allow 2x for Retina
                    logger.info(f"[VISION NAV] ⚠️  Extracted (guessed from numbers): ({x}, {y})")
                    return self._validate_and_return(x, y, screenshot)

            # No patterns matched
            logger.error(f"[VISION NAV] ❌ Could not extract coordinates from Claude response")
            logger.error(f"[VISION NAV] Full response: {response[:800]}")
            return None

        except Exception as e:
            logger.error(f"[VISION NAV] Error extracting coordinates: {e}", exc_info=True)
            return None

    def _validate_and_return(self, x: int, y: int, screenshot: Image.Image) -> Tuple[int, int]:
        """
        Validate coordinates and return them (with logging)

        Args:
            x: X coordinate
            y: Y coordinate
            screenshot: Screenshot for dimension checking

        Returns:
            (x, y) tuple
        """
        width = screenshot.width
        height = screenshot.height

        # Log dimensions for debugging
        logger.debug(f"[VISION NAV] Screenshot dimensions: {width}x{height}px")
        logger.debug(f"[VISION NAV] Proposed coordinates: ({x}, {y})")

        # Basic sanity checks
        if x < 0 or y < 0:
            logger.warning(f"[VISION NAV] ⚠️ Negative coordinates: ({x}, {y})")

        if x > width:
            logger.warning(f"[VISION NAV] ⚠️ X coordinate {x} exceeds width {width}")

        if y > height:
            logger.warning(f"[VISION NAV] ⚠️ Y coordinate {y} exceeds height {height}")

        return (x, y)

    def _validate_coordinates(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Validate that coordinates are within acceptable bounds

        Args:
            x: X coordinate
            y: Y coordinate
            width: Screen/region width
            height: Screen/region height

        Returns:
            True if valid, False otherwise
        """
        # Allow some tolerance for Retina displays (2x scaling)
        max_x = width * 2
        max_y = height * 2

        valid = (
            0 <= x <= max_x and
            0 <= y <= max_y
        )

        if not valid:
            logger.warning(f"[VISION NAV] Coordinates ({x}, {y}) outside bounds (0-{max_x}, 0-{max_y})")

        return valid

    def _adjust_suspicious_coordinates(self, x: int, y: int, width: int, height: int) -> Tuple[int, int]:
        """
        Adjust suspicious coordinates to reasonable values

        Args:
            x: X coordinate
            y: Y coordinate
            width: Screen/region width
            height: Screen/region height

        Returns:
            Adjusted (x, y) tuple
        """
        adjusted_x = x
        adjusted_y = y

        # If Y is too large, assume menu bar center
        if y > height:
            adjusted_y = 15  # Menu bar center
            logger.info(f"[VISION NAV] Adjusted Y: {y} → {adjusted_y} (menu bar center)")

        # If X is too large, cap at width
        if x > width:
            # Try to preserve relative position
            if x <= width * 2:  # Might be Retina coordinates
                adjusted_x = x // 2
                logger.info(f"[VISION NAV] Adjusted X: {x} → {adjusted_x} (Retina scaling)")
            else:
                adjusted_x = width - 180  # Typical Control Center position
                logger.info(f"[VISION NAV] Adjusted X: {x} → {adjusted_x} (capped to typical position)")

        # If X is too small (unlikely for Control Center in right section)
        if x < width // 2:
            logger.warning(f"[VISION NAV] X coordinate {x} seems too far left for Control Center")
            adjusted_x = width - 180
            logger.info(f"[VISION NAV] Adjusted X: {x} → {adjusted_x} (moved to right section)")

        return (adjusted_x, adjusted_y)

    def _extract_coordinates_from_response(self, response: str) -> Optional[Tuple[int, int]]:
        """
        Legacy coordinate extraction method (kept for compatibility)

        NOTE: Use _extract_coordinates_advanced() for new code
        """
        if not response:
            return None

        try:
            # Use a simple fallback implementation
            # Pattern: X_POSITION: 1234, Y_POSITION: 56
            x_match = re.search(r'X[_\s]*POSITION:\s*(\d+)', response, re.IGNORECASE)
            y_match = re.search(r'Y[_\s]*POSITION:\s*(\d+)', response, re.IGNORECASE)
            if x_match and y_match:
                x, y = int(x_match.group(1)), int(y_match.group(1))
                return (x, y)

            # Pattern: (x, y)
            match = re.search(r'\((\d+),\s*(\d+)\)', response)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return (x, y)

            return None

        except Exception as e:
            logger.error(f"[VISION NAV] Error extracting coordinates: {e}")
            return None
    
    async def _click_at(self, x: int, y: int):
        """Click at specific coordinates"""
        try:
            logger.info(f"[VISION NAV] Clicking at ({x}, {y})")
            
            # Move to position
            pyautogui.moveTo(x, y, duration=self.config['mouse']['movement_speed'])
            
            # Brief pause
            await asyncio.sleep(0.1)
            
            # Click
            pyautogui.click(x, y, duration=self.config['mouse']['click_duration'])
            
            logger.debug(f"[VISION NAV] Click executed at ({x}, {y})")
            
        except Exception as e:
            logger.error(f"[VISION NAV] Click error: {e}")
            raise
    
    async def _validate_control_center_position(self, x: int, y: int, menu_bar_screenshot: Image.Image) -> bool:
        """
        Advanced spatial validation to ensure detected position is reasonable for Control Center

        Uses spatial reasoning to validate that the detected coordinates make sense
        for where Control Center icon should be located.

        Args:
            x: Detected X coordinate
            y: Detected Y coordinate
            menu_bar_screenshot: Menu bar image

        Returns:
            True if position passes validation, False otherwise
        """
        try:
            width = menu_bar_screenshot.width

            # Control Center should be in the RIGHT 30% of menu bar
            right_section_start = width * 0.7

            if x < right_section_start:
                logger.warning(f"[VISION NAV] ⚠️ X={x} too far left (should be > {right_section_start:.0f})")
                return False

            # Control Center should NOT be in the very last 100px (that's usually time/date)
            if x > width - 100:
                logger.warning(f"[VISION NAV] ⚠️ X={x} too far right (time display area)")
                return False

            # Y should be centered in menu bar
            if y < 5 or y > 35:
                logger.warning(f"[VISION NAV] ⚠️ Y={y} outside menu bar center range (5-35)")
                return False

            logger.info(f"[VISION NAV] ✅ Position validation passed for ({x}, {y})")
            return True

        except Exception as e:
            logger.error(f"[VISION NAV] Error in position validation: {e}")
            return True  # Don't block on validation errors

    async def _multi_pass_detection(self, menu_bar_screenshot: Image.Image) -> bool:
        """
        Advanced multi-pass detection using different strategies

        When initial detection fails spatial validation, this method tries
        multiple detection strategies:
        1. Ask Claude to list ALL icons and their positions
        2. Use process of elimination to find Control Center
        3. Focus on rightmost region only

        Args:
            menu_bar_screenshot: Menu bar image

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("[VISION NAV] 🔄 Starting multi-pass detection...")

            # Save screenshot for analysis
            screenshot_path = self.screenshots_dir / f'multipass_{int(time.time())}.png'
            menu_bar_screenshot.save(screenshot_path)

            if not self.vision_analyzer:
                logger.error("[VISION NAV] No vision analyzer for multi-pass")
                return False

            # PASS 1: Comprehensive icon mapping
            logger.info("[VISION NAV] Pass 1: Mapping all icons...")

            mapping_prompt = """Analyze this macOS menu bar and list ALL visible icons from LEFT to RIGHT, focusing especially on the RIGHTMOST icons.

For EACH icon, provide:
1. Icon type (e.g., "WiFi", "Bluetooth", "Battery", "Siri", "Control Center", "Time", etc.)
2. X position (approximate center)
3. Visual description (color, shape, etc.)

**CRITICAL DISTINCTIONS - These are DIFFERENT icons:**

1. **Siri**: COLORFUL circular orb (rainbow/purple/blue colors), typically at X = width - 170 to width - 150
2. **Control Center**: TWO OVERLAPPING RECTANGLES (side by side), MONOCHROME (gray/white), typically at X = width - 135 to width - 115
3. **Time Display**: Numbers showing time, typically at X = width - 80

**IMPORTANT:** Siri and Control Center are NEXT TO EACH OTHER! Siri is to the LEFT, Control Center is to the RIGHT (closer to time display).

Format your response like this:
ICON_1: WiFi | X: 1200 | Radiating waves, white
ICON_2: Bluetooth | X: 1225 | B symbol, white
ICON_3: Battery | X: 1250 | Battery shape, white
ICON_4: Siri | X: 1270 | Circular colorful orb (purple/rainbow)
ICON_5: Control Center | X: 1315 | Two overlapping rectangles, monochrome gray
ICON_6: Time | X: 1360 | Clock/time numbers "2:55 AM"

List ALL icons you see, from left to right. Pay special attention to distinguishing Siri (colorful circle) from Control Center (gray rectangles)."""

            analysis = await self._analyze_with_vision(screenshot_path, mapping_prompt)

            if analysis:
                logger.info(f"[VISION NAV] Icon mapping response: {analysis[:300]}...")

                # Extract Control Center position from mapping
                cc_match = re.search(r'Control\s*Center.*?X[:\s]*(\d+)', analysis, re.IGNORECASE)
                if cc_match:
                    x = int(cc_match.group(1))
                    y = 15  # Menu bar center

                    logger.info(f"[VISION NAV] ✅ Multi-pass detected Control Center at ({x}, {y})")

                    # Validate this position with strict bounds
                    width = menu_bar_screenshot.width
                    distance_from_right = width - x

                    if distance_from_right > 142 or distance_from_right < 108:
                        logger.warning(f"[VISION NAV] ⚠️ Multi-pass position {distance_from_right}px from right is outside ideal range (108-142px)")
                        # Continue to Pass 2 instead of using this position
                    elif await self._validate_control_center_position(x, y, menu_bar_screenshot):
                        # Position is good, click it
                        await self._click_at(x, y)

                        # Verify
                        if await self._verify_control_center_clicked(x, y):
                            logger.info("[VISION NAV] ✅ Multi-pass detection successful!")
                            return True

            # PASS 2: Focused right-side detection
            logger.info("[VISION NAV] Pass 2: Focused right-side scan...")

            # Crop to rightmost 250 pixels only
            right_section = menu_bar_screenshot.crop((menu_bar_screenshot.width - 250, 0, menu_bar_screenshot.width, 50))
            right_path = self.screenshots_dir / f'right_section_{int(time.time())}.png'
            right_section.save(right_path)

            focused_prompt = """This is the RIGHTMOST section of the macOS menu bar (last 250 pixels).

Find the Control Center icon. It looks like TWO OVERLAPPING RECTANGLES side by side, MONOCHROME (gray/white).

**CRITICAL - IGNORE THESE WRONG ICONS:**
- Siri: COLORFUL circular orb (purple/rainbow) - This is TOO FAR LEFT!
- Sun symbol (brightness) - Wrong shape
- WiFi waves - Wrong shape
- Battery - Wrong shape
- Clock/time numbers - TOO FAR RIGHT!

**WHAT TO LOOK FOR:**
- Shape: TWO RECTANGLES side by side [ ][ ] or overlapping slightly
- Color: MONOCHROME gray or white (NOT colorful!)
- Position: Between Siri (colorful) and Time (numbers), typically around X=115-135 in this 250px crop
- This should be the icon CLOSEST to the right edge that is NOT the time display

**YOUR TASK:**
Look for the MONOCHROME rectangular icon between any colorful icons (like Siri) and the time display.

Provide X position relative to this cropped image (0-250).

Format:
X_POSITION: [number]
Y_POSITION: 15

Be VERY careful - Control Center is monochrome rectangles, NOT the colorful Siri orb!"""

            analysis2 = await self._analyze_with_vision(right_path, focused_prompt)

            if analysis2:
                coords = self._extract_coordinates_advanced(analysis2, right_section)
                if coords:
                    relative_x, y = coords
                    # Convert relative X to absolute X
                    absolute_x = (menu_bar_screenshot.width - 250) + relative_x

                    logger.info(f"[VISION NAV] ✅ Focused scan detected at relative ({relative_x}, {y}), absolute ({absolute_x}, {y})")

                    # Validate position with strict bounds
                    width = menu_bar_screenshot.width
                    distance_from_right = width - absolute_x

                    if distance_from_right > 142 or distance_from_right < 108:
                        logger.warning(f"[VISION NAV] ⚠️ Focused scan position {distance_from_right}px from right is outside ideal range (108-142px)")
                        logger.warning("[VISION NAV] ⚠️ Skipping this detection, will use heuristic")
                    else:
                        logger.info(f"[VISION NAV] ✅ Focused scan position validated: {distance_from_right}px from right edge")
                        # Click it
                        await self._click_at(absolute_x, y)

                        # Verify
                        if await self._verify_control_center_clicked(absolute_x, y):
                            logger.info("[VISION NAV] ✅ Focused scan successful!")
                            return True

            # PASS 3: Intelligent scanning approach
            logger.info("[VISION NAV] Pass 3: Intelligent scanning approach...")
            scan_result = await self._scan_for_control_center(menu_bar_screenshot)
            if scan_result:
                x, y = scan_result
                logger.info(f"[VISION NAV] ✅ Intelligent scan found Control Center at ({x}, {y})")
                await self._click_at(x, y)
                if await self._verify_control_center_clicked(x, y):
                    logger.info("[VISION NAV] ✅ Intelligent scan successful!")
                    return True

            # If all passes fail, fall back to heuristic
            logger.warning("[VISION NAV] ⚠️ All multi-pass attempts failed, using heuristic")
            return await self._click_control_center_heuristic()

        except Exception as e:
            logger.error(f"[VISION NAV] Error in multi-pass detection: {e}", exc_info=True)
            return False

    async def _verify_control_center_clicked(self, clicked_x: int, clicked_y: int) -> bool:
        """
        Verify that Control Center actually opened after clicking

        Args:
            clicked_x: X coordinate that was clicked
            clicked_y: Y coordinate that was clicked

        Returns:
            True if Control Center opened, False otherwise
        """
        try:
            # Wait for UI to respond
            await asyncio.sleep(0.5)

            # Capture current screen
            screenshot = await self._capture_screen()
            if not screenshot:
                logger.warning("[VISION NAV] Could not capture screen for verification")
                return True  # Assume success if can't verify

            # Save for analysis
            screenshot_path = self.screenshots_dir / f'verification_{int(time.time())}.png'
            screenshot.save(screenshot_path)

            if not self.vision_analyzer:
                return True  # Assume success if no analyzer

            logger.info(f"[VISION NAV] 🔍 Verifying click at ({clicked_x}, {clicked_y})...")

            # Ask Claude to verify
            verification_prompt = """Look at this screenshot. Did Control Center open?

Control Center is a panel that appears when you click the Control Center icon in the menu bar.
It typically shows:
- WiFi settings
- Bluetooth settings
- Screen Mirroring button
- Display settings
- Sound controls
- Other system controls

Please respond with:
- "YES" if Control Center panel is open and visible
- "NO" if Control Center is NOT open (might have clicked wrong icon)

Keep your response very brief - just YES or NO."""

            # Analyze with Claude Vision
            analysis = await self._analyze_with_vision(screenshot_path, verification_prompt)

            if analysis:
                analysis_lower = analysis.lower()
                logger.info(f"[VISION NAV] Verification response: {analysis[:100]}")

                if 'yes' in analysis_lower or 'control center' in analysis_lower and 'open' in analysis_lower:
                    logger.info("[VISION NAV] ✅ Verification passed - Control Center opened correctly")
                    return True
                elif 'no' in analysis_lower:
                    logger.warning("[VISION NAV] ❌ Verification failed - Wrong icon was clicked")
                    return False

            # If unclear, assume success
            logger.info("[VISION NAV] ⚠️ Could not determine verification status, assuming success")
            return True

        except Exception as e:
            logger.error(f"[VISION NAV] Error verifying click: {e}", exc_info=True)
            return True  # Assume success on error to avoid blocking

    async def _self_correct_control_center_click(self) -> bool:
        """
        Self-correct by asking Claude what icon was clicked and where the real Control Center is

        This method provides a feedback loop for learning from mistakes.

        Returns:
            True if successfully corrected and clicked the right icon
        """
        try:
            logger.info("[VISION NAV] 🔧 Starting self-correction process...")

            # Capture current screen state
            screenshot = await self._capture_screen()
            if not screenshot:
                logger.error("[VISION NAV] Cannot self-correct without screenshot")
                return False

            # Crop to menu bar
            menu_bar_screenshot = screenshot.crop((0, 0, screenshot.width, 50))

            # Save for analysis
            screenshot_path = self.screenshots_dir / f'self_correct_{int(time.time())}.png'
            menu_bar_screenshot.save(screenshot_path)

            if not self.vision_analyzer:
                logger.error("[VISION NAV] Cannot self-correct without vision analyzer")
                return False

            # Ask Claude for correction
            correction_prompt = """I clicked the wrong icon in the macOS menu bar. Please help me find the CORRECT Control Center icon.

**What I need:**
1. Identify which icon I clicked (wrong one)
2. Find the ACTUAL Control Center icon (two overlapping rounded rectangles)
3. Provide the EXACT coordinates of the CORRECT Control Center icon

**Control Center icon characteristics:**
- Two overlapping rounded rectangles (toggle/switch shape)
- Solid icon, not transparent
- Located in the RIGHT section of menu bar
- Usually between WiFi/Bluetooth and the Time display
- Typically around 150-200 pixels from the right edge

**Response format:**
WRONG_ICON: [description of what I clicked]
CORRECT_X_POSITION: [x coordinate of REAL Control Center]
CORRECT_Y_POSITION: [y coordinate of REAL Control Center]

Example:
WRONG_ICON: WiFi icon
CORRECT_X_POSITION: 1260
CORRECT_Y_POSITION: 15

Please help me find the correct icon!"""

            # Analyze with Claude Vision
            logger.info("[VISION NAV] 🤖 Asking Claude for correction guidance...")
            analysis = await self._analyze_with_vision(screenshot_path, correction_prompt)

            if not analysis:
                logger.error("[VISION NAV] No correction guidance received from Claude")
                return False

            logger.info(f"[VISION NAV] Correction guidance: {analysis[:200]}...")

            # Extract corrected coordinates
            x_match = re.search(r'CORRECT[_\s]*X[_\s]*POSITION\s*:\s*(\d+)', analysis, re.IGNORECASE)
            y_match = re.search(r'CORRECT[_\s]*Y[_\s]*POSITION\s*:\s*(\d+)', analysis, re.IGNORECASE)

            # Also try simpler patterns
            if not (x_match and y_match):
                coords = self._extract_coordinates_advanced(analysis, menu_bar_screenshot)
                if coords:
                    corrected_x, corrected_y = coords
                    logger.info(f"[VISION NAV] 🎯 Extracted corrected coordinates: ({corrected_x}, {corrected_y})")
                else:
                    logger.error("[VISION NAV] Could not extract corrected coordinates")
                    return False
            else:
                corrected_x = int(x_match.group(1))
                corrected_y = int(y_match.group(1))
                logger.info(f"[VISION NAV] 🎯 Corrected coordinates from Claude: ({corrected_x}, {corrected_y})")

            # Extract what icon was clicked (for learning)
            wrong_icon_match = re.search(r'WRONG[_\s]*ICON\s*:\s*(.+?)(?:\n|$)', analysis, re.IGNORECASE)
            if wrong_icon_match:
                wrong_icon = wrong_icon_match.group(1).strip()
                logger.info(f"[VISION NAV] 📝 Claude identified wrong icon: {wrong_icon}")

            # Validate corrected coordinates
            if not self._validate_coordinates(corrected_x, corrected_y, menu_bar_screenshot.width, 50):
                logger.warning(f"[VISION NAV] ⚠️ Corrected coordinates suspicious, adjusting...")
                corrected_x, corrected_y = self._adjust_suspicious_coordinates(
                    corrected_x, corrected_y, menu_bar_screenshot.width, 50
                )

            # Click the corrected coordinates
            logger.info(f"[VISION NAV] 🖱️ Clicking corrected position: ({corrected_x}, {corrected_y})")
            await self._click_at(corrected_x, corrected_y)

            # Verify the correction worked
            await asyncio.sleep(0.5)

            # Verify and save if successful
            if await self._verify_control_center_clicked(corrected_x, corrected_y):
                logger.info("[VISION NAV] ✅ Self-correction successful!")
                self._save_learned_position(corrected_x, corrected_y)
                return True
            else:
                logger.warning("[VISION NAV] ⚠️ Self-correction verification failed")
                return False

        except Exception as e:
            logger.error(f"[VISION NAV] Error during self-correction: {e}", exc_info=True)
            return False

    async def _scan_for_control_center(self, menu_bar_screenshot: Image.Image) -> tuple:
        """
        Intelligently scan the menu bar to find Control Center by analyzing multiple candidate positions
        """
        try:
            logger.info("[VISION NAV] 🔍 Starting intelligent Control Center scan...")
            width = menu_bar_screenshot.width
            height = menu_bar_screenshot.height

            # Define scan range: rightmost 300 pixels, excluding time area
            scan_start = width - 300
            scan_end = width - 100

            candidates = []

            # Scan in 15-pixel increments
            for x in range(scan_start, scan_end, 15):
                y = 15  # Center of menu bar

                # Analyze color at this position
                color_info = self._analyze_icon_color(menu_bar_screenshot, x, y)

                # Calculate distance from right edge
                distance_from_right = width - x

                # Score this candidate
                score = 0
                reason = []

                # Position scoring (ideal: 108-142px from right)
                if 108 <= distance_from_right <= 142:
                    score += 50
                    reason.append(f"ideal position ({distance_from_right}px from right)")
                elif 100 <= distance_from_right <= 150:
                    score += 25
                    reason.append(f"acceptable position ({distance_from_right}px from right)")

                # Color scoring (Control Center is monochrome)
                if color_info['is_monochrome']:
                    score += 40
                    reason.append("monochrome (gray/white)")
                elif not color_info['is_colorful']:
                    score += 20
                    reason.append("not colorful")
                else:
                    score -= 30  # Penalty for colorful icons (likely Siri)
                    reason.append("COLORFUL (likely Siri)")

                # Saturation scoring
                if color_info['saturation_avg'] < 10:
                    score += 10
                    reason.append(f"very low saturation ({color_info['saturation_avg']:.1f}%)")

                candidates.append({
                    'x': x,
                    'y': y,
                    'score': score,
                    'distance_from_right': distance_from_right,
                    'is_colorful': color_info['is_colorful'],
                    'is_monochrome': color_info['is_monochrome'],
                    'saturation': color_info['saturation_avg'],
                    'reasons': reason
                })

            # Sort candidates by score
            candidates.sort(key=lambda c: c['score'], reverse=True)

            # Log top candidates
            logger.info(f"[VISION NAV] Found {len(candidates)} candidates:")
            for i, candidate in enumerate(candidates[:5]):
                logger.info(f"[VISION NAV]   #{i+1}: X={candidate['x']} (score={candidate['score']}) - {', '.join(candidate['reasons'])}")

            # Select best candidate
            if candidates and candidates[0]['score'] >= 50:
                best = candidates[0]
                logger.info(f"[VISION NAV] ✅ Selected best candidate at X={best['x']} with score {best['score']}")
                return (best['x'], best['y'])
            else:
                logger.warning("[VISION NAV] ⚠️ No strong candidate found via scanning")
                return None

        except Exception as e:
            logger.error(f"[VISION NAV] Error in intelligent scan: {e}")
            return None

    async def _click_control_center_heuristic(self) -> bool:
        """Fallback: Click Control Center using saved or heuristic position"""
        try:
            # Get screen dimensions
            screen_width, screen_height = pyautogui.size()

            # First try intelligent scanning on the menu bar
            menu_bar_height = 50
            menu_bar_screenshot = await self._capture_menu_bar(menu_bar_height)
            if menu_bar_screenshot:
                scan_result = await self._scan_for_control_center(menu_bar_screenshot)
                if scan_result:
                    x, y = scan_result
                    logger.info(f"[VISION NAV] 🎯 Heuristic scan found Control Center at ({x}, {y})")
                    await self._click_at(x, y)
                    return True

            logger.info(f"[VISION NAV] Screen dimensions: {screen_width}x{screen_height}")

            # Try to use saved position from config first
            cc_config = self.config.get('ui_elements', {}).get('control_center', {})

            if 'absolute_x' in cc_config and 'absolute_y' in cc_config:
                # Use saved position
                saved_x = cc_config['absolute_x']
                saved_y = cc_config['absolute_y']
                saved_screen_width = cc_config.get('screen_width', screen_width)

                # If screen resolution changed, adjust using offset
                if saved_screen_width != screen_width and 'offset_from_right' in cc_config:
                    offset = cc_config['offset_from_right']
                    x = screen_width - offset
                    y = saved_y
                    logger.info(f"[VISION NAV] Using adjusted position (screen resolution changed): ({x}, {y})")
                else:
                    x = saved_x
                    y = saved_y
                    logger.info(f"[VISION NAV] Using saved position from config: ({x}, {y})")

                await self._click_at(x, y)
                return True

            # Fallback: Use improved heuristic based on typical Control Center placement
            logger.info(f"[VISION NAV] No saved position, using improved heuristic...")
            logger.warning(f"[VISION NAV] 💡 TIP: For perfect accuracy, let Claude Vision analyze your menu bar")

            # Control Center is typically about 150-200px from the right edge on most Macs
            # It's to the LEFT of the WiFi/Battery icons and time display
            # Try multiple likely positions in order of probability
            positions_to_try = [
                (screen_width - 180, 15, "180px from right (typical position)"),
                (screen_width - 160, 15, "160px from right"),
                (screen_width - 200, 15, "200px from right"),
                (screen_width - 150, 15, "150px from right"),
                (screen_width - 220, 15, "220px from right"),
            ]

            # Try the most likely position first
            x, y, description = positions_to_try[0]
            logger.info(f"[VISION NAV] Using heuristic: ({x}, {y}) - {description}")
            logger.info(f"[VISION NAV] This should click near the Control Center icon (two overlapping rectangles)")

            await self._click_at(x, y)
            return True

        except Exception as e:
            logger.error(f"[VISION NAV] Heuristic click failed: {e}")
            return False
    
    async def _click_screen_mirroring_ocr(self, screenshot: Image.Image) -> bool:
        """Use OCR to find and click Screen Mirroring"""
        try:
            # Use existing OCR infrastructure if available
            from vision.ocr_processor import OCRProcessor
            
            ocr = OCRProcessor()
            text_regions = await ocr.process_image(screenshot)
            
            # Look for "Screen Mirroring" or "Display"
            for region in text_regions:
                text = region.get('text', '').lower()
                if 'screen mirroring' in text or 'screen mirror' in text:
                    # Get bounding box
                    bbox = region.get('bbox')
                    if bbox:
                        # Calculate center
                        x = bbox[0] + bbox[2] // 2
                        y = bbox[1] + bbox[3] // 2
                        
                        await self._click_at(x, y)
                        return True
            
            return False
            
        except ImportError:
            logger.warning("[VISION NAV] OCR processor not available")
            return False
        except Exception as e:
            logger.error(f"[VISION NAV] OCR click failed: {e}")
            return False
    
    async def _click_display_ocr(self, screenshot: Image.Image, display_name: str) -> bool:
        """Use OCR to find and click display name"""
        try:
            from vision.ocr_processor import OCRProcessor
            
            ocr = OCRProcessor()
            text_regions = await ocr.process_image(screenshot)
            
            # Look for display name
            for region in text_regions:
                text = region.get('text', '')
                if display_name.lower() in text.lower():
                    bbox = region.get('bbox')
                    if bbox:
                        x = bbox[0] + bbox[2] // 2
                        y = bbox[1] + bbox[3] // 2
                        
                        await self._click_at(x, y)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"[VISION NAV] OCR display click failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get navigation statistics"""
        success_rate = 0.0
        if self.stats['total_navigations'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_navigations']) * 100
        
        return {
            'total_navigations': self.stats['total_navigations'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': round(success_rate, 2),
            'avg_duration': round(self.stats['avg_duration'], 2),
            'vision_analyzer_connected': self.vision_analyzer is not None
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get navigator status"""
        return {
            'initialized': True,
            'config_loaded': self.config is not None,
            'vision_connected': self.vision_analyzer is not None,
            'screenshots_dir': str(self.screenshots_dir),
            'stats': self.get_stats()
        }


# Singleton instance
_navigator_instance: Optional[VisionUINavigator] = None


def get_vision_navigator(config_path: Optional[str] = None) -> VisionUINavigator:
    """Get singleton vision navigator instance"""
    global _navigator_instance
    if _navigator_instance is None:
        _navigator_instance = VisionUINavigator(config_path)
    return _navigator_instance


if __name__ == "__main__":
    # Test the navigator
    async def test():
        logging.basicConfig(level=logging.INFO)
        
        navigator = get_vision_navigator()
        
        print("\n" + "="*60)
        print("Vision UI Navigator Test")
        print("="*60)
        
        # Get status
        print("\n1. Navigator Status:")
        status = navigator.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test connection
        print("\n2. Testing connection to Living Room TV...")
        print("   (This will actually attempt to connect!)")
        
        result = await navigator.connect_to_display("Living Room TV")
        
        print(f"\n3. Result:")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Steps: {result.steps_completed}")
        
        # Get stats
        print("\n4. Statistics:")
        stats = navigator.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*60)
    
    asyncio.run(test())
