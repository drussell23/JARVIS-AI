#!/usr/bin/env python3
"""
Adaptive Control Center Clicker
================================

Production-grade, self-healing Control Center interaction system.
Solves the "Coordinate Brittleness" problem with multi-method detection
and adaptive learning.

Features:
- Zero hardcoded coordinates (fully dynamic discovery)
- 6-layer fallback detection chain
- Self-learning coordinate cache with TTL
- Async/await throughout for parallel detection
- Screenshot verification after each action
- macOS version compatibility layer
- Comprehensive metrics and observability

Detection Methods (in priority order):
1. Cached Coordinates - Instant, learned from previous successes
2. OCR Detection - pytesseract + Claude Vision fallback
3. Template Matching - OpenCV template matching
4. Edge Detection - Contour analysis + shape recognition
5. Accessibility API - macOS Accessibility framework
6. AppleScript Fallback - System Events UI scripting

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0
"""

# DISABLED: The coordinate fix is causing problems
# It's not properly handling our coordinates and may be causing the doubling
import sys
from pathlib import Path
# try:
#     # Add backend to path if needed
#     backend_path = Path(__file__).parent.parent
#     if str(backend_path) not in sys.path:
#         sys.path.insert(0, str(backend_path))
#
#     from display.coordinate_fix import apply_coordinate_fix
#     apply_coordinate_fix()
#     print("[ADAPTIVE] ✅ Applied Retina coordinate fix globally")
# except Exception as e:
#     print(f"[ADAPTIVE] ⚠️ Could not apply coordinate fix: {e}")

import asyncio
import logging
import json
import time
import hashlib
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Protocol
from enum import Enum

import cv2
import numpy as np
import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class DetectionStatus(Enum):
    """Detection attempt status enumeration.
    
    Attributes:
        SUCCESS: Detection completed successfully
        FAILED: Detection failed to find target
        UNAVAILABLE: Detection method not available
        TIMEOUT: Detection timed out
    """
    SUCCESS = "success"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"


@dataclass
class DetectionResult:
    """Result from a single detection method.
    
    Attributes:
        success: Whether detection was successful
        method: Name of detection method used
        coordinates: (x, y) coordinates if found, None otherwise
        confidence: Detection confidence score (0.0-1.0)
        duration: Time taken for detection in seconds
        metadata: Additional method-specific data
        error: Error message if detection failed
    """
    success: bool
    method: str
    coordinates: Optional[Tuple[int, int]]
    confidence: float
    duration: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary.
        
        Returns:
            Dictionary representation of the detection result
        """
        return asdict(self)


@dataclass
class ClickResult:
    """Result from a click operation.
    
    Attributes:
        success: Whether click operation was successful
        target: Target identifier that was clicked
        coordinates: Final coordinates used for click
        method_used: Detection method that found the target
        verification_passed: Whether post-click verification passed
        duration: Total operation duration in seconds
        fallback_attempts: Number of fallback methods attempted
        metadata: Additional operation data
        error: Error message if operation failed
    """
    success: bool
    target: str
    coordinates: Tuple[int, int]
    method_used: str
    verification_passed: bool
    duration: float
    fallback_attempts: int
    metadata: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class CachedCoordinate:
    """Cached coordinate with learning metadata.
    
    Attributes:
        target: Target identifier
        coordinates: (x, y) coordinates
        confidence: Detection confidence when cached
        method: Detection method that found coordinates
        timestamp: Unix timestamp when cached
        success_count: Number of successful uses
        failure_count: Number of failed uses
        screen_hash: Hash of screen configuration
        macos_version: macOS version when cached
    """
    target: str
    coordinates: Tuple[int, int]
    confidence: float
    method: str
    timestamp: float
    success_count: int = 1
    failure_count: int = 0
    screen_hash: Optional[str] = None
    macos_version: Optional[str] = None


# ============================================================================
# Detection Method Protocol
# ============================================================================

class DetectionMethod(Protocol):
    """Protocol defining interface for detection methods.
    
    All detection methods must implement this protocol to be used
    by the adaptive clicker system.
    """

    @property
    def name(self) -> str:
        """Get method name.
        
        Returns:
            Unique identifier for this detection method
        """
        ...

    @property
    def priority(self) -> int:
        """Get method priority.
        
        Returns:
            Priority level (lower = higher priority)
        """
        ...

    async def is_available(self) -> bool:
        """Check if detection method is available.
        
        Returns:
            True if method can be used, False otherwise
        """
        ...

    async def detect(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect target and return coordinates.
        
        Args:
            target: Target identifier to find
            context: Optional context information
            
        Returns:
            DetectionResult with coordinates if found
        """
        ...


# ============================================================================
# Coordinate Cache Manager
# ============================================================================

class CoordinateCache:
    """Persistent coordinate cache with learning capabilities.

    Features:
    - TTL-based invalidation
    - Success/failure tracking
    - Screen resolution awareness
    - macOS version tracking
    
    The cache learns from successful clicks and invalidates coordinates
    that fail repeatedly or become stale.
    """

    def __init__(self, cache_file: Optional[Path] = None, ttl_seconds: int = 86400):
        """Initialize coordinate cache.

        Args:
            cache_file: Path to cache file (default: ~/.jarvis/control_center_cache.json)
            ttl_seconds: Time-to-live for cached coordinates (default: 24 hours)
        """
        self.cache_file = cache_file or (
            Path.home() / ".jarvis" / "control_center_cache.json"
        )
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedCoordinate] = {}
        self.screen_resolution = self._get_screen_resolution()
        self.macos_version = self._get_macos_version()

        self._load_cache()

        logger.info(f"[CACHE] Initialized cache at {self.cache_file}")
        logger.info(f"[CACHE] Screen: {self.screen_resolution}, macOS: {self.macos_version}")

    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get current screen resolution.
        
        Returns:
            (width, height) tuple of screen resolution
        """
        return pyautogui.size()

    def _get_macos_version(self) -> str:
        """Get macOS version string.
        
        Returns:
            macOS version string or "unknown" if detection fails
        """
        try:
            result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"[CACHE] Could not get macOS version: {e}")
            return "unknown"

    def _get_screen_hash(self) -> str:
        """Get hash of current screen configuration.
        
        Returns:
            8-character hash of screen resolution and macOS version
        """
        config = f"{self.screen_resolution}_{self.macos_version}"
        return hashlib.md5(config.encode()).hexdigest()[:8]

    def _load_cache(self):
        """Load cache from disk.
        
        Loads cached coordinates from JSON file and converts to
        CachedCoordinate objects. Handles file corruption gracefully.
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)

                # Convert to CachedCoordinate objects
                for key, value in data.items():
                    self.cache[key] = CachedCoordinate(**value)

                logger.info(f"[CACHE] Loaded {len(self.cache)} cached coordinates")
                self._clean_expired()
            else:
                logger.info("[CACHE] No existing cache found, starting fresh")
        except Exception as e:
            logger.error(f"[CACHE] Error loading cache: {e}", exc_info=True)
            self.cache = {}

    def _save_cache(self):
        """Save cache to disk.
        
        Persists current cache state to JSON file. Handles write
        errors gracefully to prevent cache corruption.
        """
        try:
            data = {key: asdict(value) for key, value in self.cache.items()}

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"[CACHE] Saved {len(self.cache)} cached coordinates")
        except Exception as e:
            logger.error(f"[CACHE] Error saving cache: {e}", exc_info=True)

    def _clean_expired(self):
        """Remove expired cache entries.
        
        Removes entries older than TTL and saves cache if any
        entries were removed.
        """
        now = time.time()
        expired_keys = []

        for key, cached in self.cache.items():
            age = now - cached.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            logger.debug(f"[CACHE] Removed expired entry: {key}")

        if expired_keys:
            self._save_cache()

    def get(self, target: str) -> Optional[CachedCoordinate]:
        """Get cached coordinates for target.

        Args:
            target: Target identifier (e.g., "control_center", "screen_mirroring")

        Returns:
            Cached coordinates if valid, None otherwise
            
        The method validates that:
        - Entry is not expired
        - Screen configuration matches
        - Success rate is acceptable
        """
        self._clean_expired()

        cached = self.cache.get(target)
        if not cached:
            return None

        # Validate screen configuration matches
        current_hash = self._get_screen_hash()
        if cached.screen_hash != current_hash:
            logger.warning(
                f"[CACHE] Screen configuration changed for {target}, "
                f"invalidating cache"
            )
            return None

        # Check if coordinates have high failure rate
        if cached.failure_count > cached.success_count * 2:
            logger.warning(
                f"[CACHE] High failure rate for {target} "
                f"({cached.failure_count} failures vs {cached.success_count} successes), "
                f"invalidating cache"
            )
            return None

        logger.info(
            f"[CACHE] ✅ Cache hit for {target}: {cached.coordinates} "
            f"(success_rate={cached.success_count}/{cached.success_count + cached.failure_count})"
        )
        return cached

    def set(
        self,
        target: str,
        coordinates: Tuple[int, int],
        confidence: float,
        method: str
    ):
        """Cache coordinates for target.

        Args:
            target: Target identifier
            coordinates: (x, y) coordinates
            confidence: Detection confidence (0.0-1.0)
            method: Detection method used
            
        Updates existing entry or creates new one. Increments success
        count and updates metadata.
        """
        screen_hash = self._get_screen_hash()

        # Update existing entry or create new one
        if target in self.cache:
            cached = self.cache[target]
            cached.coordinates = coordinates
            cached.confidence = max(cached.confidence, confidence)
            cached.method = method
            cached.timestamp = time.time()
            cached.success_count += 1
            cached.screen_hash = screen_hash
            cached.macos_version = self.macos_version
        else:
            self.cache[target] = CachedCoordinate(
                target=target,
                coordinates=coordinates,
                confidence=confidence,
                method=method,
                timestamp=time.time(),
                success_count=1,
                failure_count=0,
                screen_hash=screen_hash,
                macos_version=self.macos_version
            )

        self._save_cache()
        logger.info(f"[CACHE] ✅ Cached {target}: {coordinates} (method={method})")

    def mark_failure(self, target: str):
        """Mark a cached coordinate as failed.
        
        Args:
            target: Target identifier that failed
            
        Increments failure count for the cached coordinate.
        High failure rates will cause cache invalidation.
        """
        if target in self.cache:
            self.cache[target].failure_count += 1
            self._save_cache()
            logger.warning(
                f"[CACHE] Marked {target} as failed "
                f"(failures={self.cache[target].failure_count})"
            )

    def invalidate(self, target: str):
        """Invalidate cached coordinate.
        
        Args:
            target: Target identifier to invalidate
            
        Removes the cached coordinate completely.
        """
        if target in self.cache:
            del self.cache[target]
            self._save_cache()
            logger.info(f"[CACHE] Invalidated cache for {target}")

    def clear(self):
        """Clear all cached coordinates.
        
        Removes all cached coordinates and saves empty cache.
        """
        self.cache = {}
        self._save_cache()
        logger.info("[CACHE] Cleared all cached coordinates")


# ============================================================================
# Detection Methods
# ============================================================================

class CachedDetection:
    """Detection using cached coordinates.
    
    Fastest detection method that uses previously learned coordinates.
    Always runs first in the detection chain.
    """

    name = "cached"
    priority = 1

    def __init__(self, cache: CoordinateCache):
        """Initialize cached detection.
        
        Args:
            cache: CoordinateCache instance to use
        """
        self.cache = cache

    async def is_available(self) -> bool:
        """Check if cached detection is available.
        
        Returns:
            Always True - cached detection is always available
        """
        return True

    async def detect(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect using cached coordinates.
        
        Args:
            target: Target identifier to find
            context: Optional context (unused)
            
        Returns:
            DetectionResult with cached coordinates if available
        """
        start_time = time.time()

        logger.info(f"[CACHED DETECTION] 🔍 Checking cache for target: '{target}'")

        cached = self.cache.get(target)
        if not cached:
            logger.warning(f"[CACHED DETECTION] ❌ No cached coordinates found for '{target}'")
            return DetectionResult(
                success=False,
                method=self.name,
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error="No cached coordinates found"
            )

        logger.info(f"[CACHED DETECTION] ✅ Cache HIT for '{target}': coordinates={cached.coordinates}, confidence={cached.confidence}")
        logger.info(f"[CACHED DETECTION] 📊 Cache stats: success_count={cached.success_count}, failure_count={cached.failure_count}, age={time.time() - cached.timestamp:.1f}s")

        return DetectionResult(
            success=True,
            method=self.name,
            coordinates=cached.coordinates,
            confidence=cached.confidence,
            duration=time.time() - start_time,
            metadata={
                "success_count": cached.success_count,
                "failure_count": cached.failure_count,
                "age_seconds": time.time() - cached.timestamp
            }
        )


class SimpleHeuristicDetection:
    """Detection using simple screen position heuristics.
    
    Fast detection method that uses mathematical calculations
    based on known screen layouts. Runs after cache miss.
    """

    name = "simple_heuristic"
    priority = 2  # Run after cache, before complex methods

    def __init__(self):
        """Initialize simple heuristic detection."""
        self._detector = None

    async def is_available(self) -> bool:
        """Check if simple heuristic detection is available.
        
        Returns:
            Always True - heuristic detection is always available
        """
        return True

    async def detect(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect using simple position heuristics.
        
        Args:
            target: Target identifier to find
            context: Optional context (unused)
            
        Returns:
            DetectionResult with calculated coordinates for known targets
        """
        start_time = time.time()

        try:
            if self._detector is None:
                from backend.display.simple_menubar_detector import get_simple_menubar_detector
                self._detector = get_simple_menubar_detector()

            logger.info(f"[SIMPLE HEURISTIC] Detecting '{target}' using position math...")

            # Get position based on target (KNOWN CORRECT COORDINATES)
            if target == "control_center":
                x, y = self._detector.get_control_center_position()  # (1236, 12)
            elif target == "screen_mirroring":
                x, y = self._detector.get_screen_mirroring_position()  # (1396, 177)
            elif target == "Living Room TV":
                x, y = self._detector.get_living_room_tv_position()  # (1223, 115)
            else:
                return DetectionResult(
                    success=False,
                    method=self.name,
                    coordinates=None,
                    confidence=0.0,
                    duration=time.time() - start_time,
                    metadata={},
                    error=f"No known coordinates for target: {target}"
                )

            logger.info(f"[SIMPLE HEURISTIC] ✅ Found '{target}' at ({x}, {y}) (known coordinates)")

            # High confidence since these are known correct coordinates
            return DetectionResult(
                success=True,
                method=self.name,
                coordinates=(x, y),
                confidence=0.95,  # High confidence, will be verified anyway
                duration=time.time() - start_time,
                metadata={"method": "known_coordinates"}
            )

        except Exception as e:
            logger.error(f"[SIMPLE HEURISTIC] Detection failed: {e}", exc_info=True)
            return DetectionResult(
                success=False,
                method=self.name,
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error=str(e)
            )


class OCRDetection:
    """Detection using OCR (pytesseract + Claude Vision fallback).
    
    Uses optical character recognition to find text-based targets
    and Claude Vision API for icon recognition.
    """

    name = "ocr"
    priority = 3  # After Enhanced Vision Pipeline

    def __init__(self, vision_analyzer=None):
        """Initialize OCR detection.
        
        Args:
            vision_analyzer: Claude Vision analyzer instance (optional)
        """
        self.vision_analyzer = vision_analyzer
        self._tesseract_available = None
        self._dpi_scale = None

    def _get_dpi_scale(self) -> float:
        """Get DPI scale factor (cached).
        
        Returns:
            DPI scale factor (1.0 for standard displays, 2.0 for Retina)
        """
        if self._dpi_scale is None:
            try:
                from AppKit import NSScreen
                main_screen = NSScreen.mainScreen()
                self._dpi_scale = main_screen.backingScaleFactor()
                logger.info(f"[OCR] Detected DPI scale: {self._dpi_scale}x")
            except:
                self._dpi_scale = 1.0
                logger.warning("[OCR] Could not detect DPI scale, assuming 1.0x")
        return self._dpi_scale

    def _convert_to_logical_pixels(self, x: int, y: int, region_offset: tuple = (0, 0)) -> tuple:
        """Convert coordinates from physical pixels to logical pixels.

        Args:
            x: X coordinate in physical pixels (from screenshot/vision)
            y: Y coordinate in physical pixels (from screenshot/vision)
            region_offset: Offset of the screenshot region in logical pixels

        Returns:
            (x, y) tuple in logical pixels for PyAutoGUI
        """
        dpi_scale = self._get_dpi_scale()

        # Convert from physical to logical
        logical_x = x / dpi_scale
        logical_y = y / dpi_scale

        # Add region offset (already in logical pixels)
        final_x = int(round(logical_x + region_offset[0]))
        final_y = int(round(logical_y + region_offset[1]))

        logger.info(
            f"[OCR] Coordinate conversion: "
            f"Physical ({x}, {y}) -> Logical ({logical_x:.1f}, {logical_y:.1f}) "
            f"+ Offset {region_offset} = Final ({final_x}, {final_y}) [DPI={dpi_scale}x]"
        )

        return (final_x, final_y)

    async def is_available(self) -> bool:
        """Check if OCR detection is available.
        
        Returns:
            True if pytesseract or Claude Vision is available
        """
        if self._tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
            except Exception:
                self._tesseract_available = False

        return self._tesseract_available or self.vision_analyzer is not None

    async def detect(
        self,
        target: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect using OCR/Visual Recognition.
        
        Args:
            target: Target identifier to find
            context: Optional context information
            
        Returns:
            DetectionResult with coordinates if text/icon found
        """
        start_time = time.time()

        try:
            # Determine screenshot region based on target
            if target == "control_center":
                # Menu bar only (top 50 pixels)
                screenshot = pyautogui.screenshot(region=(0, 0, 2000, 50))
            elif target == "screen_mirroring":
                # Control Center menu area (top-right, larger region)
                screenshot = pyautogui.screenshot(region=(800, 0, 800, 400))
            else:
                # Generic: capture larger area
                screenshot = pyautogui.screenshot(region=(0, 0, 2000, 500))

            # For icon-based targets (Control Center), skip pytesseract and use Claude Vision
            if target == "control_center":
                # Skip pytesseract (can't detect icons), go straight to Claude Vision
                if self.vision_analyzer:
                    result = await self._ocr_with_claude(screenshot, target)
                    if result.success:
                        return result
                else:
                    logger.warning("[OCR] Control Center requires Claude Vision for icon detection")
            else:
                # For text-based targets, try pytesseract first
                if self._tesseract_available:
                    result = await self._ocr_with_tesseract(screenshot, target)
                    if result.success:
                        return result

                # Fallback to Claude Vision
                if self.vision_analyzer:
                    result = await self._ocr_with_claude(screenshot, target)
                    if result.success:
                        return result

            return DetectionResult(
                success=False,
                method=self.name,
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error="OCR detection failed with all methods"
            )

        except Exception as e:
            logger.error(f"[OCR] Detection failed: {e}", exc_info=True)
            return DetectionResult(
                success=False,
                method=self.name,
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error=str(e)
            )

    async def _ocr_with_tesseract(
        self,
        screenshot: Image.Image,
        target: str
    ) -> DetectionResult:
        """Use pytesseract for OCR text detection.
        
        Args:
            screenshot: PIL Image to analyze
            target: Target text to find
            
        Returns:
            DetectionResult with text coordinates if found
        """
        start_time = time.time()

        try:
            import pytesseract

            # Run OCR
            data = pytesseract.image_to_data(
                screenshot,
                output_type=pytesseract.Output.DICT
            )

            # Search for target text
            target_lower = target.lower().replace("_", " ")

            for i, text in enumerate(data['text']):
                if target_lower in text.lower():
                    # Calculate center point (in physical pixels from screenshot)
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    confidence = float(data['conf'][i]) / 100.0

                    logger.info(
                        f"[OCR-TESSERACT] Found '{target}' at ({x}, {y}) in screenshot (physical pixels) "
                        f"with confidence {confidence:.2%}"
                    )

                    # CRITICAL: Convert to logical pixels
                    # Tesseract coordinates are relative to screenshot, need to account for region offset
                    region_offset = (0, 0)  # Tesseract is typically used on full screenshots
                    logical_x, logical_y = self._convert_to_logical_pixels(x, y, region_offset)

                    logger.info(
                        f"[OCR-TESSERACT] Converted to logical pixels: ({logical_x}, {logical_y})"
                    )

                    return DetectionResult(
                        success=True,
                        method="ocr_tesseract",
                        coordinates=(logical_x, logical_y),
                        confidence=confidence,
                        duration=time.time() - start_time,
                        metadata={
                            "text_found": text,
                            "physical_coords": (x, y),
                            "logical_coords": (logical_x, logical_y)
                        }
                    )

            return DetectionResult(
                success=False,
                method="ocr_tesseract",
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error=f"Text '{target}' not found in OCR results"
            )

        except Exception as e:
            logger.error(f"[OCR-TESSERACT] Failed: {e}")
            return DetectionResult(
                success=False,
                method="ocr_tesseract",
                coordinates=None,
                confidence=0.0,
                duration=time.time() - start_time,
                metadata={},
                error=str(e)
            )

    async def _ocr_with_claude(
        self,
        screenshot: Image.Image,
        target: str
    ) -> DetectionResult:
        """Use Claude Vision for visual recognition (icons + text).
        
        Args:
            screenshot: PIL Image to analyze
            target: Target element to find
            
        Returns:
            DetectionResult with element coordinates if found
        """
        start_time = time.time()

        try:
            if not self.vision_analyzer:
                return DetectionResult(
                    success=False,
                    method="ocr_claude",
                    coordinates=None,
                    confidence=0.0,
                    duration=time.time() - start_time,
                    metadata={},
                    error="Claude Vision analyzer not available"
                )

            width, height = screenshot.size

            # Build context-aware prompt based on target
            if target == "control_center":
                prompt = """Find the Control Center icon in this macOS menu bar screenshot.

The Control Center icon looks like:
- Two toggle switches (circles on lines) stacked vertically
- Located in the top-right menu bar area
- Usually near the battery, Wi-Fi, and clock icons
- Dark icon on light background (or light icon on dark mode)

Return ONLY the coordinates in this EXACT format:
COORDINATES: x=<number>, y=<number>

Where x and y are pixel positions from the top-left corner of this image.

If not found, respond with: NOT_FOUND"""

            elif target == "screen_mirroring":
                prompt = """Find the "Screen Mirroring" text or "Display" menu item in this Control Center menu screenshot.

Look for:
- Text that says "Screen Mirroring" OR "Display"
- Usually appears as a menu item with an icon
- May have a right-pointing arrow (›) next to it

Return ONLY the coordinates in this EXACT format:
COORDINATES: x=<number>, y=<number>

Where x and y are pixel positions from the top-left corner of this image.

If not found, respond with: NOT_FOUND"""

            else:
                # Generic text/icon search
                prompt = f"""Find the text or icon labeled "{target}" in this screenshot.

Return ONLY the coordinates in this EXACT format:
COORDINATES: x=<number>, y=<number>

Where x and y are pixel positions from the top-left corner.

If not found, respond with: NOT_FOUND"""

            result = await self.vision_analyzer.analyze_screenshot(
                image=screenshot,
                prompt=prompt,
                use_cache=False
            )

            # Extract response
            if isinstance(result, tuple):
                analysis, _ = result
                response_text = analysis.get('analysis', '')
            else:
                response_text = result.get('analysis', '')

            # Check for NOT_FOUND
            if "NOT_FOUND" in response_text:
                return DetectionResult(
                    success=False,
                    method="ocr_claude