#!/usr/bin/env python3
"""
Display-Aware Situational Awareness Intelligence (SAI) for Voice Unlock
========================================================================

LangGraph-powered intelligent display detection and adaptive typing strategy
for JARVIS voice unlock. Handles external displays (especially mirrored 85" Sony TV)
with situational awareness.

Features:
- Multi-display detection (built-in, HDMI, AirPlay, mirrored)
- LangGraph reasoning for optimal typing strategy
- Adaptive timing based on display configuration
- Self-learning from success/failure patterns
- Async-first architecture
- Dynamic configuration (no hardcoding)

Author: Derek Russell
Date: 2025-11-24
"""

import asyncio
import ctypes
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TypedDict

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

logger = logging.getLogger(__name__)


# =============================================================================
# Display Configuration Enums
# =============================================================================

class DisplayType(Enum):
    """Types of display connections"""
    BUILT_IN = auto()      # MacBook built-in display
    HDMI = auto()          # HDMI connected (like Sony TV)
    THUNDERBOLT = auto()   # Thunderbolt/DisplayPort
    USB_C = auto()         # USB-C display
    AIRPLAY = auto()       # AirPlay wireless
    SIDECAR = auto()       # iPad Sidecar
    UNKNOWN = auto()


class DisplayMode(Enum):
    """Display arrangement modes"""
    SINGLE = auto()        # Only built-in display
    EXTENDED = auto()      # Extended desktop
    MIRRORED = auto()      # Mirrored displays
    CLAMSHELL = auto()     # Lid closed, external only


class TypingStrategy(Enum):
    """Password typing strategies based on display config"""
    CORE_GRAPHICS_FAST = auto()      # CG events, fast timing (single display)
    CORE_GRAPHICS_SLOW = auto()      # CG events, slower timing (extended)
    CORE_GRAPHICS_CAUTIOUS = auto()  # CG events, very slow (mirrored)
    APPLESCRIPT_DIRECT = auto()      # AppleScript keystroke (most reliable for mirrored)
    HYBRID_CG_APPLESCRIPT = auto()   # Try CG, fallback to AppleScript


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DisplayInfo:
    """Information about a single display"""
    display_id: int
    name: str
    display_type: DisplayType
    is_main: bool
    is_mirrored: bool
    width: int
    height: int
    scale_factor: float = 1.0
    is_builtin: bool = False
    refresh_rate: int = 60
    vendor_id: Optional[str] = None
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_id": self.display_id,
            "name": self.name,
            "display_type": self.display_type.name,
            "is_main": self.is_main,
            "is_mirrored": self.is_mirrored,
            "width": self.width,
            "height": self.height,
            "scale_factor": self.scale_factor,
            "is_builtin": self.is_builtin,
            "refresh_rate": self.refresh_rate,
            "vendor_id": self.vendor_id,
            "model_name": self.model_name,
        }


@dataclass
class DisplayContext:
    """Complete display context for SAI decision making"""
    displays: List[DisplayInfo] = field(default_factory=list)
    display_mode: DisplayMode = DisplayMode.SINGLE
    total_displays: int = 1
    has_external: bool = False
    is_mirrored: bool = False
    is_tv_connected: bool = False
    tv_name: Optional[str] = None
    primary_display: Optional[DisplayInfo] = None
    external_displays: List[DisplayInfo] = field(default_factory=list)
    detection_time_ms: float = 0.0
    detection_method: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_mode": self.display_mode.name,
            "total_displays": self.total_displays,
            "has_external": self.has_external,
            "is_mirrored": self.is_mirrored,
            "is_tv_connected": self.is_tv_connected,
            "tv_name": self.tv_name,
            "primary_display": self.primary_display.to_dict() if self.primary_display else None,
            "external_displays": [d.to_dict() for d in self.external_displays],
            "detection_time_ms": self.detection_time_ms,
            "detection_method": self.detection_method,
        }


@dataclass
class TypingConfig:
    """Adaptive typing configuration based on display context"""
    strategy: TypingStrategy
    base_keystroke_delay_ms: float
    key_press_duration_ms: float
    shift_register_delay_ms: float
    wake_delay_ms: float
    submit_delay_ms: float
    retry_count: int
    use_applescript_fallback: bool
    post_to_specific_display: bool = False
    target_display_id: Optional[int] = None
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.name,
            "base_keystroke_delay_ms": self.base_keystroke_delay_ms,
            "key_press_duration_ms": self.key_press_duration_ms,
            "shift_register_delay_ms": self.shift_register_delay_ms,
            "wake_delay_ms": self.wake_delay_ms,
            "submit_delay_ms": self.submit_delay_ms,
            "retry_count": self.retry_count,
            "use_applescript_fallback": self.use_applescript_fallback,
            "reasoning": self.reasoning,
        }


# =============================================================================
# LangGraph State for Display-Aware Reasoning
# =============================================================================

class DisplayAwareState(TypedDict, total=False):
    """State for LangGraph display-aware reasoning"""
    # Input
    raw_display_data: Dict[str, Any]
    system_profiler_data: Optional[str]

    # Detected context
    display_context: Optional[Dict[str, Any]]

    # Analysis
    is_mirrored: bool
    is_tv_connected: bool
    tv_type: Optional[str]
    risk_level: str  # low, medium, high

    # Strategy selection
    recommended_strategy: Optional[str]
    typing_config: Optional[Dict[str, Any]]

    # Reasoning
    reasoning_steps: List[str]
    confidence: float

    # Output
    final_decision: Optional[str]
    error: Optional[str]


# =============================================================================
# Display Detector (Core Graphics + System Profiler)
# =============================================================================

class DisplayDetector:
    """
    Multi-method display detector for macOS.
    Uses Core Graphics, System Profiler, and AppleScript for comprehensive detection.
    """

    # Known TV identifiers (dynamic, loaded from config)
    TV_IDENTIFIERS = [
        "sony", "samsung", "lg", "vizio", "tcl", "hisense",
        "85", "75", "65", "55",  # Common TV sizes
        "bravia", "qled", "oled", "uhd", "4k",
        "living room", "bedroom", "tv"
    ]

    def __init__(self):
        self._cg_available = self._check_core_graphics()
        self._cache: Optional[DisplayContext] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration_ms = 5000  # Cache for 5 seconds

    def _check_core_graphics(self) -> bool:
        """Check if Core Graphics is available"""
        try:
            self._cg = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
            self._cg.CGMainDisplayID.restype = ctypes.c_uint32
            self._cg.CGGetActiveDisplayList.argtypes = [
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)
            ]
            self._cg.CGGetActiveDisplayList.restype = ctypes.c_int32
            self._cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
            self._cg.CGDisplayBounds.restype = ctypes.c_double * 4  # CGRect
            self._cg.CGDisplayIsMain.argtypes = [ctypes.c_uint32]
            self._cg.CGDisplayIsMain.restype = ctypes.c_bool
            self._cg.CGDisplayMirrorsDisplay.argtypes = [ctypes.c_uint32]
            self._cg.CGDisplayMirrorsDisplay.restype = ctypes.c_uint32
            self._cg.CGDisplayIsBuiltin.argtypes = [ctypes.c_uint32]
            self._cg.CGDisplayIsBuiltin.restype = ctypes.c_bool
            return True
        except Exception as e:
            logger.warning(f"Core Graphics not available: {e}")
            return False

    async def detect_displays(self, use_cache: bool = True) -> DisplayContext:
        """
        Detect all connected displays with comprehensive information.

        Args:
            use_cache: Use cached result if available and fresh

        Returns:
            DisplayContext with full display information
        """
        start_time = datetime.now()

        # Check cache
        if use_cache and self._cache and self._cache_time:
            age_ms = (datetime.now() - self._cache_time).total_seconds() * 1000
            if age_ms < self._cache_duration_ms:
                logger.debug(f"Using cached display context (age: {age_ms:.0f}ms)")
                return self._cache

        context = DisplayContext()

        try:
            # Method 1: Core Graphics (fast, accurate for display list)
            if self._cg_available:
                displays = await self._detect_via_core_graphics()
                context.displays = displays
                context.detection_method = "core_graphics"
            else:
                # Fallback to System Profiler
                displays = await self._detect_via_system_profiler()
                context.displays = displays
                context.detection_method = "system_profiler"

            # Analyze display configuration
            context.total_displays = len(context.displays)
            context.has_external = any(not d.is_builtin for d in context.displays)
            context.is_mirrored = any(d.is_mirrored for d in context.displays)

            # Find primary display
            for display in context.displays:
                if display.is_main:
                    context.primary_display = display
                    break

            # Identify external displays
            context.external_displays = [d for d in context.displays if not d.is_builtin]

            # Check for TV connection
            for display in context.external_displays:
                if self._is_tv(display):
                    context.is_tv_connected = True
                    context.tv_name = display.name
                    break

            # Determine display mode
            if context.total_displays == 1:
                context.display_mode = DisplayMode.SINGLE
            elif context.is_mirrored:
                context.display_mode = DisplayMode.MIRRORED
            elif not any(d.is_builtin for d in context.displays):
                context.display_mode = DisplayMode.CLAMSHELL
            else:
                context.display_mode = DisplayMode.EXTENDED

            context.detection_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update cache
            self._cache = context
            self._cache_time = datetime.now()

            logger.info(
                f"Display detection complete: {context.total_displays} displays, "
                f"mode={context.display_mode.name}, mirrored={context.is_mirrored}, "
                f"tv_connected={context.is_tv_connected} ({context.detection_time_ms:.1f}ms)"
            )

            return context

        except Exception as e:
            logger.error(f"Display detection failed: {e}", exc_info=True)
            context.detection_method = "failed"
            return context

    async def _detect_via_core_graphics(self) -> List[DisplayInfo]:
        """Detect displays using Core Graphics API"""
        displays = []

        try:
            # Get display count
            max_displays = 16
            display_array = (ctypes.c_uint32 * max_displays)()
            display_count = ctypes.c_uint32()

            result = self._cg.CGGetActiveDisplayList(
                max_displays,
                display_array,
                ctypes.byref(display_count)
            )

            if result != 0:
                logger.error(f"CGGetActiveDisplayList failed with code {result}")
                return displays

            main_display_id = self._cg.CGMainDisplayID()

            for i in range(display_count.value):
                display_id = display_array[i]

                # Get display bounds (CGRect: origin.x, origin.y, width, height)
                # Note: CGDisplayBounds returns a struct, need proper handling
                try:
                    # Use subprocess for display info since CGDisplayBounds struct handling is complex
                    is_main = self._cg.CGDisplayIsMain(display_id)
                    is_builtin = self._cg.CGDisplayIsBuiltin(display_id)
                    mirrors_display = self._cg.CGDisplayMirrorsDisplay(display_id)
                    is_mirrored = mirrors_display != 0

                    # Get display name via system_profiler for this display
                    name = await self._get_display_name(display_id, is_builtin)

                    # Determine display type
                    if is_builtin:
                        display_type = DisplayType.BUILT_IN
                    elif "airplay" in name.lower():
                        display_type = DisplayType.AIRPLAY
                    elif "sidecar" in name.lower():
                        display_type = DisplayType.SIDECAR
                    else:
                        display_type = DisplayType.HDMI  # Most common external

                    display_info = DisplayInfo(
                        display_id=display_id,
                        name=name,
                        display_type=display_type,
                        is_main=is_main,
                        is_mirrored=is_mirrored,
                        width=0,  # Will be filled by system_profiler
                        height=0,
                        is_builtin=is_builtin,
                    )

                    displays.append(display_info)

                except Exception as e:
                    logger.warning(f"Failed to get info for display {display_id}: {e}")

            # Enhance with system_profiler data
            await self._enhance_with_system_profiler(displays)

            return displays

        except Exception as e:
            logger.error(f"Core Graphics detection failed: {e}")
            return displays

    async def _get_display_name(self, display_id: int, is_builtin: bool) -> str:
        """Get display name"""
        if is_builtin:
            return "Built-in Retina Display"

        # Try to get name from ioreg
        try:
            proc = await asyncio.create_subprocess_exec(
                "ioreg", "-lw0", "-r", "-c", "IODisplayConnect",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode()

            # Parse for display names
            if "DisplayProductName" in output:
                import re
                matches = re.findall(r'"DisplayProductName"\s*=\s*"([^"]+)"', output)
                if matches:
                    for match in matches:
                        if match != "Color LCD":  # Skip built-in
                            return match

            return f"External Display {display_id}"

        except Exception:
            return f"External Display {display_id}"

    async def _detect_via_system_profiler(self) -> List[DisplayInfo]:
        """Detect displays using system_profiler"""
        displays = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "system_profiler", "SPDisplaysDataType", "-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()

            data = json.loads(stdout.decode())
            graphics_data = data.get("SPDisplaysDataType", [])

            display_id = 1
            for gpu in graphics_data:
                ndrvs = gpu.get("spdisplays_ndrvs", [])
                for display in ndrvs:
                    name = display.get("_name", f"Display {display_id}")
                    resolution = display.get("_spdisplays_resolution", "0 x 0")

                    # Parse resolution
                    try:
                        width, height = resolution.replace(" ", "").split("x")[:2]
                        width = int(width)
                        height = int(height.split("@")[0]) if "@" in height else int(height)
                    except:
                        width, height = 0, 0

                    is_builtin = "built-in" in name.lower() or "retina" in name.lower()
                    is_main = display.get("spdisplays_main", "").lower() == "yes"
                    is_mirrored = display.get("spdisplays_mirror", "").lower() == "on"

                    # Determine type
                    if is_builtin:
                        display_type = DisplayType.BUILT_IN
                    elif "airplay" in name.lower():
                        display_type = DisplayType.AIRPLAY
                    elif "thunderbolt" in display.get("spdisplays_connection_type", "").lower():
                        display_type = DisplayType.THUNDERBOLT
                    else:
                        display_type = DisplayType.HDMI

                    display_info = DisplayInfo(
                        display_id=display_id,
                        name=name,
                        display_type=display_type,
                        is_main=is_main,
                        is_mirrored=is_mirrored,
                        width=width,
                        height=height,
                        is_builtin=is_builtin,
                        vendor_id=display.get("_spdisplays_display-vendor-id"),
                        model_name=display.get("_spdisplays_display-product-id"),
                    )

                    displays.append(display_info)
                    display_id += 1

            return displays

        except Exception as e:
            logger.error(f"System profiler detection failed: {e}")
            return displays

    async def _enhance_with_system_profiler(self, displays: List[DisplayInfo]):
        """Enhance Core Graphics display info with system_profiler data"""
        try:
            sp_displays = await self._detect_via_system_profiler()

            # Match by is_builtin flag and name similarity
            for cg_display in displays:
                for sp_display in sp_displays:
                    if cg_display.is_builtin == sp_display.is_builtin:
                        cg_display.width = sp_display.width
                        cg_display.height = sp_display.height
                        cg_display.vendor_id = sp_display.vendor_id
                        cg_display.model_name = sp_display.model_name
                        if sp_display.name and not cg_display.name.startswith("External Display"):
                            cg_display.name = sp_display.name
                        break

        except Exception as e:
            logger.debug(f"System profiler enhancement failed: {e}")

    def _is_tv(self, display: DisplayInfo) -> bool:
        """Check if a display is a TV"""
        name_lower = display.name.lower()

        # Check for TV identifiers
        for identifier in self.TV_IDENTIFIERS:
            if identifier in name_lower:
                return True

        # Check for large resolution (4K+)
        if display.width >= 3840 or display.height >= 2160:
            return True

        # Check for common TV aspect ratios with large size
        if display.width > 1920 and display.height > 1080:
            return True

        return False


# =============================================================================
# LangGraph Reasoning Engine for Typing Strategy
# =============================================================================

class DisplayAwareReasoningEngine:
    """
    LangGraph-powered reasoning engine for display-aware typing strategy selection.
    """

    def __init__(self):
        self._graph = None
        if LANGGRAPH_AVAILABLE:
            self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine for display-aware reasoning"""
        if not LANGGRAPH_AVAILABLE:
            return

        # Create the graph
        graph = StateGraph(DisplayAwareState)

        # Add nodes
        graph.add_node("detect_displays", self._node_detect_displays)
        graph.add_node("analyze_configuration", self._node_analyze_configuration)
        graph.add_node("assess_risk", self._node_assess_risk)
        graph.add_node("select_strategy", self._node_select_strategy)
        graph.add_node("generate_config", self._node_generate_config)

        # Add edges
        graph.set_entry_point("detect_displays")
        graph.add_edge("detect_displays", "analyze_configuration")
        graph.add_edge("analyze_configuration", "assess_risk")
        graph.add_edge("assess_risk", "select_strategy")
        graph.add_edge("select_strategy", "generate_config")
        graph.add_edge("generate_config", END)

        self._graph = graph.compile()

    async def _node_detect_displays(self, state: DisplayAwareState) -> Dict[str, Any]:
        """Node: Detect display configuration"""
        reasoning = ["Starting display detection..."]

        try:
            detector = DisplayDetector()
            context = await detector.detect_displays()

            reasoning.append(f"Detected {context.total_displays} display(s)")
            reasoning.append(f"Display mode: {context.display_mode.name}")

            return {
                "display_context": context.to_dict(),
                "is_mirrored": context.is_mirrored,
                "is_tv_connected": context.is_tv_connected,
                "tv_type": context.tv_name,
                "reasoning_steps": reasoning,
            }

        except Exception as e:
            reasoning.append(f"Detection failed: {e}")
            return {
                "error": str(e),
                "reasoning_steps": reasoning,
            }

    async def _node_analyze_configuration(self, state: DisplayAwareState) -> Dict[str, Any]:
        """Node: Analyze display configuration"""
        reasoning = list(state.get("reasoning_steps", []))

        context = state.get("display_context", {})
        is_mirrored = state.get("is_mirrored", False)
        is_tv_connected = state.get("is_tv_connected", False)

        reasoning.append("Analyzing display configuration...")

        if is_mirrored and is_tv_connected:
            reasoning.append("ALERT: TV mirroring detected - highest risk for keyboard event routing")
        elif is_mirrored:
            reasoning.append("WARNING: Display mirroring active - keyboard events may route incorrectly")
        elif is_tv_connected:
            reasoning.append("NOTE: TV connected in extended mode - moderate risk")
        else:
            reasoning.append("Single/extended display without TV - low risk")

        return {
            "reasoning_steps": reasoning,
        }

    async def _node_assess_risk(self, state: DisplayAwareState) -> Dict[str, Any]:
        """Node: Assess risk level for password typing"""
        reasoning = list(state.get("reasoning_steps", []))

        is_mirrored = state.get("is_mirrored", False)
        is_tv_connected = state.get("is_tv_connected", False)

        reasoning.append("Assessing risk level...")

        # Risk assessment logic
        if is_mirrored and is_tv_connected:
            risk_level = "high"
            reasoning.append("Risk Level: HIGH - Mirrored TV may intercept or delay keyboard events")
        elif is_mirrored:
            risk_level = "high"
            reasoning.append("Risk Level: HIGH - Mirroring can cause event routing issues")
        elif is_tv_connected:
            risk_level = "medium"
            reasoning.append("Risk Level: MEDIUM - Extended TV display may affect focus")
        else:
            risk_level = "low"
            reasoning.append("Risk Level: LOW - Single display, optimal conditions")

        return {
            "risk_level": risk_level,
            "reasoning_steps": reasoning,
        }

    async def _node_select_strategy(self, state: DisplayAwareState) -> Dict[str, Any]:
        """Node: Select optimal typing strategy based on risk"""
        reasoning = list(state.get("reasoning_steps", []))
        risk_level = state.get("risk_level", "low")

        reasoning.append(f"Selecting strategy for risk level: {risk_level}")

        if risk_level == "high":
            # For mirrored displays (especially TV), use AppleScript which is more reliable
            strategy = TypingStrategy.APPLESCRIPT_DIRECT
            reasoning.append("Selected: APPLESCRIPT_DIRECT - Most reliable for mirrored displays")
            reasoning.append("Reason: AppleScript keystroke events go through System Events")
            reasoning.append("        which properly routes to the active application regardless")
            reasoning.append("        of display configuration")
        elif risk_level == "medium":
            # Extended display - use hybrid approach
            strategy = TypingStrategy.HYBRID_CG_APPLESCRIPT
            reasoning.append("Selected: HYBRID_CG_APPLESCRIPT - Try CG first, fallback to AppleScript")
        else:
            # Low risk - use fast Core Graphics
            strategy = TypingStrategy.CORE_GRAPHICS_FAST
            reasoning.append("Selected: CORE_GRAPHICS_FAST - Optimal for single display")

        return {
            "recommended_strategy": strategy.name,
            "reasoning_steps": reasoning,
        }

    async def _node_generate_config(self, state: DisplayAwareState) -> Dict[str, Any]:
        """Node: Generate final typing configuration"""
        reasoning = list(state.get("reasoning_steps", []))
        strategy_name = state.get("recommended_strategy", "CORE_GRAPHICS_FAST")
        risk_level = state.get("risk_level", "low")

        reasoning.append("Generating typing configuration...")

        strategy = TypingStrategy[strategy_name]

        # Generate timing config based on strategy
        if strategy == TypingStrategy.APPLESCRIPT_DIRECT:
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=150,  # Slower for reliability
                key_press_duration_ms=80,
                shift_register_delay_ms=100,
                wake_delay_ms=1500,  # Longer wake for TV
                submit_delay_ms=200,
                retry_count=3,
                use_applescript_fallback=True,
                reasoning="AppleScript strategy for mirrored/TV display"
            )
        elif strategy == TypingStrategy.HYBRID_CG_APPLESCRIPT:
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=120,
                key_press_duration_ms=60,
                shift_register_delay_ms=80,
                wake_delay_ms=1200,
                submit_delay_ms=150,
                retry_count=2,
                use_applescript_fallback=True,
                reasoning="Hybrid strategy for extended display"
            )
        elif strategy == TypingStrategy.CORE_GRAPHICS_CAUTIOUS:
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=100,
                key_press_duration_ms=50,
                shift_register_delay_ms=60,
                wake_delay_ms=1000,
                submit_delay_ms=100,
                retry_count=3,
                use_applescript_fallback=True,
                reasoning="Cautious CG strategy with fallback"
            )
        else:  # CORE_GRAPHICS_FAST
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=80,
                key_press_duration_ms=40,
                shift_register_delay_ms=50,
                wake_delay_ms=800,
                submit_delay_ms=100,
                retry_count=2,
                use_applescript_fallback=True,
                reasoning="Fast CG strategy for optimal conditions"
            )

        reasoning.append(f"Generated config: {config.strategy.name}")
        reasoning.append(f"Keystroke delay: {config.base_keystroke_delay_ms}ms")
        reasoning.append(f"Wake delay: {config.wake_delay_ms}ms")

        return {
            "typing_config": config.to_dict(),
            "final_decision": f"Use {strategy.name} with {config.base_keystroke_delay_ms}ms delays",
            "confidence": 0.95 if risk_level == "high" else 0.9,
            "reasoning_steps": reasoning,
        }

    async def determine_typing_strategy(self) -> Tuple[TypingConfig, DisplayContext, List[str]]:
        """
        Run the full LangGraph reasoning pipeline to determine optimal typing strategy.

        Returns:
            Tuple of (TypingConfig, DisplayContext, reasoning_steps)
        """
        if not LANGGRAPH_AVAILABLE or not self._graph:
            # Fallback without LangGraph
            return await self._fallback_strategy()

        try:
            # Run the graph
            initial_state: DisplayAwareState = {
                "reasoning_steps": [],
            }

            final_state = await self._graph.ainvoke(initial_state)

            # Extract results
            config_dict = final_state.get("typing_config", {})
            strategy = TypingStrategy[config_dict.get("strategy", "CORE_GRAPHICS_FAST")]

            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=config_dict.get("base_keystroke_delay_ms", 80),
                key_press_duration_ms=config_dict.get("key_press_duration_ms", 40),
                shift_register_delay_ms=config_dict.get("shift_register_delay_ms", 50),
                wake_delay_ms=config_dict.get("wake_delay_ms", 800),
                submit_delay_ms=config_dict.get("submit_delay_ms", 100),
                retry_count=config_dict.get("retry_count", 2),
                use_applescript_fallback=config_dict.get("use_applescript_fallback", True),
                reasoning=config_dict.get("reasoning", ""),
            )

            context_dict = final_state.get("display_context", {})
            context = DisplayContext(
                display_mode=DisplayMode[context_dict.get("display_mode", "SINGLE")],
                total_displays=context_dict.get("total_displays", 1),
                has_external=context_dict.get("has_external", False),
                is_mirrored=context_dict.get("is_mirrored", False),
                is_tv_connected=context_dict.get("is_tv_connected", False),
                tv_name=context_dict.get("tv_name"),
            )

            reasoning = final_state.get("reasoning_steps", [])

            logger.info(f"SAI Decision: {config.strategy.name} for {context.display_mode.name}")
            for step in reasoning:
                logger.debug(f"  {step}")

            return config, context, reasoning

        except Exception as e:
            logger.error(f"LangGraph reasoning failed: {e}", exc_info=True)
            return await self._fallback_strategy()

    async def _fallback_strategy(self) -> Tuple[TypingConfig, DisplayContext, List[str]]:
        """Fallback strategy without LangGraph"""
        reasoning = ["LangGraph unavailable, using fallback detection"]

        detector = DisplayDetector()
        context = await detector.detect_displays()

        reasoning.append(f"Detected: {context.total_displays} displays, mirrored={context.is_mirrored}")

        # Simple rule-based strategy selection
        if context.is_mirrored or context.is_tv_connected:
            strategy = TypingStrategy.APPLESCRIPT_DIRECT
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=150,
                key_press_duration_ms=80,
                shift_register_delay_ms=100,
                wake_delay_ms=1500,
                submit_delay_ms=200,
                retry_count=3,
                use_applescript_fallback=True,
                reasoning="Fallback: AppleScript for external/mirrored display"
            )
            reasoning.append("Selected AppleScript strategy for mirrored/TV")
        else:
            strategy = TypingStrategy.CORE_GRAPHICS_FAST
            config = TypingConfig(
                strategy=strategy,
                base_keystroke_delay_ms=80,
                key_press_duration_ms=40,
                shift_register_delay_ms=50,
                wake_delay_ms=800,
                submit_delay_ms=100,
                retry_count=2,
                use_applescript_fallback=True,
                reasoning="Fallback: Fast CG for single display"
            )
            reasoning.append("Selected fast CG strategy for single display")

        return config, context, reasoning


# =============================================================================
# Main SAI Class
# =============================================================================

class DisplayAwareSAI:
    """
    Main Situational Awareness Intelligence class for display-aware voice unlock.

    Usage:
        sai = DisplayAwareSAI()
        config, context, reasoning = await sai.get_optimal_typing_config()

        # Use config to type password
        if config.strategy == TypingStrategy.APPLESCRIPT_DIRECT:
            await type_via_applescript(password)
        else:
            await type_via_core_graphics(password, config)
    """

    def __init__(self):
        self._reasoning_engine = DisplayAwareReasoningEngine()
        self._detector = DisplayDetector()
        self._last_context: Optional[DisplayContext] = None
        self._last_config: Optional[TypingConfig] = None

    async def get_optimal_typing_config(self) -> Tuple[TypingConfig, DisplayContext, List[str]]:
        """
        Get the optimal typing configuration based on current display setup.

        Returns:
            Tuple of (TypingConfig, DisplayContext, reasoning_steps)
        """
        config, context, reasoning = await self._reasoning_engine.determine_typing_strategy()

        self._last_context = context
        self._last_config = config

        return config, context, reasoning

    async def get_display_context(self) -> DisplayContext:
        """Get current display context (fast, cached)"""
        return await self._detector.detect_displays()

    def is_tv_mode(self) -> bool:
        """Quick check if TV mode is active"""
        if self._last_context:
            return self._last_context.is_tv_connected
        return False

    def is_mirrored(self) -> bool:
        """Quick check if mirroring is active"""
        if self._last_context:
            return self._last_context.is_mirrored
        return False

    @property
    def last_config(self) -> Optional[TypingConfig]:
        """Get the last computed typing config"""
        return self._last_config

    @property
    def last_context(self) -> Optional[DisplayContext]:
        """Get the last detected display context"""
        return self._last_context


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================

_sai_instance: Optional[DisplayAwareSAI] = None


async def get_display_sai() -> DisplayAwareSAI:
    """Get or create the singleton SAI instance"""
    global _sai_instance
    if _sai_instance is None:
        _sai_instance = DisplayAwareSAI()
    return _sai_instance


async def get_optimal_typing_strategy() -> Tuple[TypingConfig, DisplayContext, List[str]]:
    """Convenience function to get optimal typing strategy"""
    sai = await get_display_sai()
    return await sai.get_optimal_typing_config()


# =============================================================================
# Test Function
# =============================================================================

async def test_display_sai():
    """Test the Display-Aware SAI"""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Display-Aware SAI Test")
    print("=" * 60)

    sai = DisplayAwareSAI()

    print("\n1. Detecting displays...")
    context = await sai.get_display_context()
    print(f"   Total displays: {context.total_displays}")
    print(f"   Mode: {context.display_mode.name}")
    print(f"   Mirrored: {context.is_mirrored}")
    print(f"   TV Connected: {context.is_tv_connected}")
    if context.tv_name:
        print(f"   TV Name: {context.tv_name}")

    print("\n2. Running LangGraph reasoning...")
    config, _, reasoning = await sai.get_optimal_typing_config()

    print(f"\n3. Optimal Strategy: {config.strategy.name}")
    print(f"   Keystroke delay: {config.base_keystroke_delay_ms}ms")
    print(f"   Wake delay: {config.wake_delay_ms}ms")
    print(f"   Use AppleScript fallback: {config.use_applescript_fallback}")

    print("\n4. Reasoning steps:")
    for step in reasoning:
        print(f"   - {step}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_display_sai())
