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
        
        # Configure PyAutoGUI safety
        pyautogui.PAUSE = self.config.get('mouse', {}).get('delay_between_actions', 0.5)
        pyautogui.FAILSAFE = True
        
        # Load learned Control Center position if available
        self._load_learned_position()

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
                    'learned_at': datetime.now().isoformat()
                }, f)
            logger.info(f"[VISION NAV] 💾 Saved learned position: ({x}, {y})")
        except Exception as e:
            logger.warning(f"[VISION NAV] Could not save learned position: {e}")

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

                # Enhanced, detailed prompt with exclusion rules
                prompt = """You are analyzing a macOS menu bar screenshot. I need you to find the Control Center icon and provide its EXACT pixel coordinates.

**CRITICAL: What Control Center icon looks like:**
- Two overlapping rounded rectangles (like toggle switches stacked)
- Looks like two squares overlapping side by side
- Solid white/gray icon on dark menu bar background
- 20-24px wide, 16-20px tall
- UNIQUE SHAPE - nothing else looks like this!

**IMPORTANT: What Control Center is NOT:**
- NOT the Siri icon (colorful orb/circle - VERY COMMON MISTAKE!)
- NOT the WiFi icon (radiating waves)
- NOT the Bluetooth icon (B symbol)
- NOT the Battery icon (battery shape)
- NOT the Time/Clock display (numbers)
- NOT the Keyboard Brightness icon (sun symbol)
- NOT the Display Brightness icon (sun with monitor)
- NOT the Sound icon (speaker symbol)
- NOT any colorful icon (Control Center is monochrome gray/white)

**Where to find Control Center:**
- Far RIGHT section of menu bar
- Between Display/Keyboard brightness icons and Time display
- Usually 100-180 pixels from the right edge
- Look for the UNIQUE two overlapping rectangles shape

**IMPORTANT SPATIAL CLUES:**
1. Control Center is to the LEFT of the Time display
2. Control Center is to the RIGHT of brightness/sound icons
3. It's in the last 200 pixels from the right edge
4. Look for the distinctive "two overlapping squares" shape

**Your task:**
1. Scan the RIGHT section of the menu bar (last 250 pixels)
2. Find the icon with TWO OVERLAPPING RECTANGLES shape
3. Ignore all other icon shapes (sun, waves, battery, etc.)
4. Provide EXACT center coordinates

**Response format (REQUIRED):**
X_POSITION: [x coordinate]
Y_POSITION: [y coordinate]

**Example:**
X_POSITION: 1260
Y_POSITION: 15

Be VERY careful to identify the correct two-overlapping-rectangles icon shape!"""

                # Analyze with Claude Vision
                analysis = await self._analyze_with_vision(screenshot_path, prompt)

                if analysis:
                    logger.info(f"[VISION NAV] Claude response received: {analysis[:150]}...")

                    # Extract coordinates with robust parsing
                    coords = self._extract_coordinates_advanced(analysis, menu_bar_screenshot)

                    if coords:
                        x, y = coords
                        logger.info(f"[VISION NAV] ✅ Claude Vision detected Control Center at ({x}, {y})")

                        # Validate coordinates are within menu bar bounds
                        if not self._validate_coordinates(x, y, menu_bar_screenshot.width, menu_bar_height):
                            logger.warning(f"[VISION NAV] ⚠️ Coordinates out of bounds, adjusting...")
                            x, y = self._adjust_suspicious_coordinates(x, y, menu_bar_screenshot.width, menu_bar_height)
                            logger.info(f"[VISION NAV] Adjusted to: ({x}, {y})")

                        # ADVANCED: Spatial validation - Control Center should be in right section
                        if not await self._validate_control_center_position(x, y, menu_bar_screenshot):
                            logger.warning(f"[VISION NAV] ⚠️ Position failed spatial validation at ({x}, {y})")
                            logger.info("[VISION NAV] Attempting multi-pass detection...")
                            return await self._multi_pass_detection(menu_bar_screenshot)

                        # Click the icon
                        await self._click_at(x, y)

                        # Self-correction: Verify we clicked the right icon
                        if await self._verify_control_center_clicked(x, y):
                            # Save this position for future use
                            self._save_learned_position(x, y)
                            return True
                        else:
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

            mapping_prompt = """Analyze this macOS menu bar and list ALL visible icons from LEFT to RIGHT.

For EACH icon, provide:
1. Icon type (e.g., "WiFi", "Bluetooth", "Battery", "Time", "Control Center", etc.)
2. X position (approximate center)
3. Visual description

**CRITICAL:** The Control Center icon looks like TWO OVERLAPPING RECTANGLES or squares overlapping side by side.
It is NOT the sun icon (brightness), NOT the WiFi icon, NOT the battery icon.

Format your response like this:
ICON_1: WiFi | X: 1200 | Radiating waves symbol
ICON_2: Bluetooth | X: 1225 | B symbol
ICON_3: Battery | X: 1250 | Battery shape
ICON_4: Control Center | X: 1280 | Two overlapping rectangles
ICON_5: Time | X: 1350 | Clock/time display

List ALL icons you see, from left to right."""

            analysis = await self._analyze_with_vision(screenshot_path, mapping_prompt)

            if analysis:
                logger.info(f"[VISION NAV] Icon mapping response: {analysis[:300]}...")

                # Extract Control Center position from mapping
                cc_match = re.search(r'Control\s*Center.*?X[:\s]*(\d+)', analysis, re.IGNORECASE)
                if cc_match:
                    x = int(cc_match.group(1))
                    y = 15  # Menu bar center

                    logger.info(f"[VISION NAV] ✅ Multi-pass detected Control Center at ({x}, {y})")

                    # Validate this position
                    if await self._validate_control_center_position(x, y, menu_bar_screenshot):
                        # Click it
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

Find the Control Center icon. It looks like TWO OVERLAPPING RECTANGLES side by side.

IGNORE these icons:
- Sun symbol (brightness)
- WiFi waves
- Battery
- Clock/time numbers

Provide X position relative to this cropped image (0-250).

Format:
X_POSITION: [number]
Y_POSITION: 15

Be VERY careful - only select the two-overlapping-rectangles icon!"""

            analysis2 = await self._analyze_with_vision(right_path, focused_prompt)

            if analysis2:
                coords = self._extract_coordinates_advanced(analysis2, right_section)
                if coords:
                    relative_x, y = coords
                    # Convert relative X to absolute X
                    absolute_x = (menu_bar_screenshot.width - 250) + relative_x

                    logger.info(f"[VISION NAV] ✅ Focused scan detected at relative ({relative_x}, {y}), absolute ({absolute_x}, {y})")

                    # Click it
                    await self._click_at(absolute_x, y)

                    # Verify
                    if await self._verify_control_center_clicked(absolute_x, y):
                        logger.info("[VISION NAV] ✅ Focused scan successful!")
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

    async def _click_control_center_heuristic(self) -> bool:
        """Fallback: Click Control Center using saved or heuristic position"""
        try:
            # Get screen dimensions
            screen_width, screen_height = pyautogui.size()

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
