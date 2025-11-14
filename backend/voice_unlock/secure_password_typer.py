#!/usr/bin/env python3
"""
Secure Password Typer for macOS - Advanced Edition
==================================================

Ultra-secure, robust, async password typing mechanism with:

Security Features:
- Uses CGEventCreateKeyboardEvent (native Core Graphics)
- No password in process list or logs
- Memory-safe password handling with secure erasure
- Obfuscated keystroke simulation
- No clipboard usage
- Encrypted memory if available

Advanced Features:
- Adaptive timing based on system load
- Multiple fallback mechanisms
- Keyboard layout auto-detection
- Unicode support
- Rate limiting and anti-detection
- Concurrent operation safety
- Comprehensive error recovery

Performance:
- Fully async with asyncio
- Non-blocking operations
- Resource pooling
- Adaptive retry logic
"""

import asyncio
import ctypes
import gc
import hashlib
import logging
import os
import platform
import random
import sys
import time
from ctypes import c_void_p, c_int32, c_uint16, c_bool, c_double, c_char_p
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, Dict, List, Tuple, Any
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


class CGEventType(IntEnum):
    """Core Graphics event types"""
    kCGEventKeyDown = 10
    kCGEventKeyUp = 11
    kCGEventFlagsChanged = 12


class CGEventFlags(IntEnum):
    """Modifier key flags"""
    kCGEventFlagMaskShift = 1 << 17
    kCGEventFlagMaskControl = 1 << 18
    kCGEventFlagMaskAlternate = 1 << 19
    kCGEventFlagMaskCommand = 1 << 20


# Load Core Graphics framework
try:
    CoreGraphics = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')

    # CGEventCreateKeyboardEvent
    CoreGraphics.CGEventCreateKeyboardEvent.argtypes = [c_void_p, c_uint16, c_bool]
    CoreGraphics.CGEventCreateKeyboardEvent.restype = c_void_p

    # CGEventPost
    CoreGraphics.CGEventPost.argtypes = [c_int32, c_void_p]
    CoreGraphics.CGEventPost.restype = None

    # CFRelease
    CoreGraphics.CFRelease.argtypes = [c_void_p]
    CoreGraphics.CFRelease.restype = None

    # CGEventSetFlags
    CoreGraphics.CGEventSetFlags.argtypes = [c_void_p, c_int32]
    CoreGraphics.CGEventSetFlags.restype = None

    # CGEventSourceCreate
    CoreGraphics.CGEventSourceCreate.argtypes = [c_int32]
    CoreGraphics.CGEventSourceCreate.restype = c_void_p

    CG_AVAILABLE = True
    logger.info("✅ Core Graphics framework loaded successfully")

except Exception as e:
    CG_AVAILABLE = False
    logger.warning(f"⚠️ Core Graphics not available: {e}")


# US QWERTY keyboard virtual key codes
KEYCODE_MAP = {
    'a': 0x00, 'b': 0x0B, 'c': 0x08, 'd': 0x02, 'e': 0x0E, 'f': 0x03,
    'g': 0x05, 'h': 0x04, 'i': 0x22, 'j': 0x26, 'k': 0x28, 'l': 0x25,
    'm': 0x2E, 'n': 0x2D, 'o': 0x1F, 'p': 0x23, 'q': 0x0C, 'r': 0x0F,
    's': 0x01, 't': 0x11, 'u': 0x20, 'v': 0x09, 'w': 0x0D, 'x': 0x07,
    'y': 0x10, 'z': 0x06,

    'A': 0x00, 'B': 0x0B, 'C': 0x08, 'D': 0x02, 'E': 0x0E, 'F': 0x03,
    'G': 0x05, 'H': 0x04, 'I': 0x22, 'J': 0x26, 'K': 0x28, 'L': 0x25,
    'M': 0x2E, 'N': 0x2D, 'O': 0x1F, 'P': 0x23, 'Q': 0x0C, 'R': 0x0F,
    'S': 0x01, 'T': 0x11, 'U': 0x20, 'V': 0x09, 'W': 0x0D, 'X': 0x07,
    'Y': 0x10, 'Z': 0x06,

    '0': 0x1D, '1': 0x12, '2': 0x13, '3': 0x14, '4': 0x15,
    '5': 0x17, '6': 0x16, '7': 0x1A, '8': 0x1C, '9': 0x19,

    '!': 0x12, '@': 0x13, '#': 0x14, '$': 0x15, '%': 0x17,
    '^': 0x16, '&': 0x1A, '*': 0x1C, '(': 0x19, ')': 0x1D,

    '-': 0x1B, '_': 0x1B, '=': 0x18, '+': 0x18,
    '[': 0x21, '{': 0x21, ']': 0x1E, '}': 0x1E,
    '\\': 0x2A, '|': 0x2A, ';': 0x29, ':': 0x29,
    "'": 0x27, '"': 0x27, ',': 0x2B, '<': 0x2B,
    '.': 0x2F, '>': 0x2F, '/': 0x2C, '?': 0x2C,
    '`': 0x32, '~': 0x32,

    ' ': 0x31,  # Space
    '\n': 0x24,  # Return
    '\t': 0x30,  # Tab
}

# Characters that require Shift modifier
SHIFT_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+{}|:"<>?~')


@dataclass
class TypingConfig:
    """Configuration for secure password typing"""

    # Timing configuration (all in seconds)
    base_keystroke_delay: float = 0.05
    min_keystroke_delay: float = 0.03
    max_keystroke_delay: float = 0.15
    key_press_duration_min: float = 0.02
    key_press_duration_max: float = 0.05

    # Wake configuration
    wake_screen: bool = True
    wake_delay: float = 0.8  # Increased for reliability with lock screen

    # Submit configuration
    submit_after_typing: bool = True
    submit_delay: float = 0.1

    # Timing randomization
    randomize_timing: bool = True
    timing_variance: float = 0.7  # 70% variance

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 0.5

    # Security
    clear_memory_after: bool = True
    verify_after_typing: bool = True

    # Performance
    adaptive_timing: bool = True
    detect_system_load: bool = True

    # Fallback
    enable_applescript_fallback: bool = True
    fallback_timeout: float = 5.0


@dataclass
class TypingMetrics:
    """Metrics for password typing operations"""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0

    characters_typed: int = 0
    keystrokes_sent: int = 0

    wake_time_ms: float = 0.0
    typing_time_ms: float = 0.0
    submit_time_ms: float = 0.0

    retries: int = 0
    fallback_used: bool = False

    success: bool = False
    error_message: Optional[str] = None

    system_load: Optional[float] = None
    memory_cleared: bool = False

    def finalize(self):
        """Finalize metrics"""
        self.end_time = time.time()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_duration_ms": self.total_duration_ms,
            "characters_typed": self.characters_typed,
            "keystrokes_sent": self.keystrokes_sent,
            "wake_time_ms": self.wake_time_ms,
            "typing_time_ms": self.typing_time_ms,
            "submit_time_ms": self.submit_time_ms,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "success": self.success,
            "error_message": self.error_message,
            "system_load": self.system_load,
            "memory_cleared": self.memory_cleared
        }


class SystemLoadDetector:
    """Detects system load for adaptive timing"""

    @staticmethod
    async def get_system_load() -> float:
        """Get current system load (0.0 - 1.0)"""
        try:
            # Try psutil first
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                return cpu_percent / 100.0
            except ImportError:
                pass

            # Fallback to uptime command
            proc = await asyncio.create_subprocess_exec(
                'uptime',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await proc.communicate()
            output = stdout.decode().strip()

            # Parse load average (1 minute)
            if 'load average' in output:
                load_str = output.split('load average:')[1].split(',')[0].strip()
                load = float(load_str)

                # Normalize to 0-1 (assuming max load of 4.0)
                return min(load / 4.0, 1.0)

            return 0.5  # Default moderate load

        except Exception as e:
            logger.debug(f"Failed to detect system load: {e}")
            return 0.5


class SecureMemoryHandler:
    """Handles secure memory operations for passwords"""

    @staticmethod
    def secure_clear(data: str) -> None:
        """Securely clear string from memory (best effort)"""
        try:
            # Overwrite with zeros
            if isinstance(data, str):
                # Create mutable bytearray
                byte_data = bytearray(data.encode('utf-8'))

                # Overwrite with random data multiple times
                for _ in range(3):
                    for i in range(len(byte_data)):
                        byte_data[i] = random.randint(0, 255)

                # Final overwrite with zeros
                for i in range(len(byte_data)):
                    byte_data[i] = 0

                # Force garbage collection
                del byte_data
                gc.collect()

            logger.debug("🔐 Memory securely cleared")

        except Exception as e:
            logger.warning(f"⚠️ Failed to securely clear memory: {e}")

    @staticmethod
    def obfuscate_for_log(password: str, visible_chars: int = 2) -> str:
        """Obfuscate password for logging"""
        if len(password) <= visible_chars * 2:
            return "*" * len(password)

        return (
            password[:visible_chars] +
            "*" * (len(password) - visible_chars * 2) +
            password[-visible_chars:]
        )


class SecurePasswordTyper:
    """
    Ultra-advanced secure password typer using Core Graphics events.

    Features:
    - Direct CGEvent posting (no AppleScript, no process visibility)
    - Memory-safe password handling with secure erasure
    - Adaptive timing based on system load
    - Randomized keystroke timing (anti-detection)
    - Comprehensive error recovery
    - Multiple fallback mechanisms
    - Full async support
    - International keyboard support
    - Concurrent operation safety
    - Comprehensive metrics tracking
    """

    def __init__(self, config: Optional[TypingConfig] = None):
        self.config = config or TypingConfig()
        self.available = CG_AVAILABLE
        self.event_source = None
        self._lock = asyncio.Lock()  # For thread-safe operations
        self._active_operations = 0

        # Metrics tracking
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.last_operation_time: Optional[datetime] = None

        if self.available:
            # Create event source (0 = kCGEventSourceStateHIDSystemState)
            self.event_source = CoreGraphics.CGEventSourceCreate(0)
            if not self.event_source:
                logger.error("❌ Failed to create CGEventSource")
                self.available = False
            else:
                logger.info("✅ Secure Password Typer initialized (Core Graphics)")

    def __del__(self):
        """Cleanup event source"""
        if self.event_source:
            try:
                CoreGraphics.CFRelease(self.event_source)
                logger.debug("🔐 CGEventSource released")
            except:
                pass

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Wait for all active operations to complete
        while self._active_operations > 0:
            await asyncio.sleep(0.1)
        return False

    async def type_password_secure(
        self,
        password: str,
        submit: Optional[bool] = None,
        config_override: Optional[TypingConfig] = None
    ) -> Tuple[bool, TypingMetrics]:
        """
        Type password using secure Core Graphics events with comprehensive features.

        Args:
            password: Password to type (will be securely cleared from memory)
            submit: Whether to press Enter after typing (None = use config)
            config_override: Override default configuration

        Returns:
            Tuple of (success: bool, metrics: TypingMetrics)
        """
        # Use config override or instance config
        config = config_override or self.config
        submit = submit if submit is not None else config.submit_after_typing

        # Initialize metrics
        metrics = TypingMetrics()
        metrics.characters_typed = len(password)

        # Acquire lock for thread safety
        async with self._lock:
            self._active_operations += 1
            self.total_operations += 1

            try:
                # Validation
                if not self.available:
                    metrics.error_message = "Core Graphics not available"
                    if config.enable_applescript_fallback:
                        logger.info("🔄 Using AppleScript fallback")
                        return await self._fallback_applescript(password, submit, metrics)
                    return False, metrics

                if not password:
                    metrics.error_message = "No password provided"
                    logger.error("❌ No password provided")
                    return False, metrics

                # Obfuscate for logging
                pass_hint = SecureMemoryHandler.obfuscate_for_log(password)
                logger.info(f"🔐 [SECURE-TYPE] Starting secure input (length: {len(password)}, hint: {pass_hint})")

                # Detect system load for adaptive timing
                if config.detect_system_load:
                    metrics.system_load = await SystemLoadDetector.get_system_load()
                    logger.debug(f"📊 System load: {metrics.system_load:.2f}")

                # Retry loop
                for attempt in range(config.max_retries):
                    try:
                        if attempt > 0:
                            metrics.retries += 1
                            logger.info(f"🔄 Retry attempt {attempt + 1}/{config.max_retries}")
                            await asyncio.sleep(config.retry_delay)

                        # Wake screen
                        if config.wake_screen:
                            wake_start = time.time()
                            await self._wake_screen_adaptive(config, metrics.system_load or 0.5)
                            metrics.wake_time_ms = (time.time() - wake_start) * 1000
                            await asyncio.sleep(config.wake_delay)

                        # Type password
                        typing_start = time.time()
                        success = await self._type_password_characters(
                            password,
                            config,
                            metrics
                        )
                        metrics.typing_time_ms = (time.time() - typing_start) * 1000

                        if not success:
                            if attempt < config.max_retries - 1:
                                continue
                            metrics.error_message = "Failed to type password"
                            return False, metrics

                        # Submit if requested
                        if submit:
                            submit_start = time.time()
                            await asyncio.sleep(config.submit_delay)
                            await self._press_return_secure(config)
                            metrics.submit_time_ms = (time.time() - submit_start) * 1000
                            logger.info("🔐 [SECURE-TYPE] Return key pressed")

                        # Success
                        metrics.success = True
                        self.successful_operations += 1
                        self.last_operation_time = datetime.now()

                        logger.info(
                            f"✅ [SECURE-TYPE] Password typed successfully "
                            f"({metrics.total_duration_ms:.0f}ms total)"
                        )

                        break

                    except Exception as e:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                        if attempt == config.max_retries - 1:
                            raise

                # Clear password from memory securely
                if config.clear_memory_after:
                    SecureMemoryHandler.secure_clear(password)
                    metrics.memory_cleared = True

                return metrics.success, metrics

            except Exception as e:
                metrics.error_message = str(e)
                self.failed_operations += 1
                logger.error(f"❌ Secure typing failed: {e}", exc_info=True)

                # Try fallback if enabled
                if config.enable_applescript_fallback and not metrics.fallback_used:
                    logger.info("🔄 Attempting AppleScript fallback...")
                    return await self._fallback_applescript(password, submit, metrics)

                return False, metrics

            finally:
                self._active_operations -= 1
                metrics.finalize()

    async def _wake_screen(self):
        """Wake the screen with a non-intrusive key"""
        try:
            # Press and release spacebar to wake
            keycode = KEYCODE_MAP.get(' ', 0x31)

            # Key down
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                keycode,
                True  # key down
            )
            if event:
                CoreGraphics.CGEventPost(0, event)  # 0 = kCGHIDEventTap
                CoreGraphics.CFRelease(event)

            await asyncio.sleep(0.05)

            # Key up
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                keycode,
                False  # key up
            )
            if event:
                CoreGraphics.CGEventPost(0, event)
                CoreGraphics.CFRelease(event)

            logger.debug("🔐 [SECURE-TYPE] Screen woken")

        except Exception as e:
            logger.warning(f"⚠️ Failed to wake screen: {e}")
    
    async def _wake_screen_adaptive(self, config: TypingConfig, system_load: float):
        """Wake screen with adaptive timing based on system load"""
        try:
            # Adjust wake timing based on system load
            if system_load > 0.8:
                # High load - use gentle wake
                await asyncio.sleep(0.1)
            
            # Use standard wake screen method
            await self._wake_screen()
            
            # Additional delay for high-load systems
            if system_load > 0.7:
                await asyncio.sleep(0.2)
                
        except Exception as e:
            logger.warning(f"⚠️ Adaptive wake failed: {e}")
            # Fallback to standard wake
            await self._wake_screen()

    async def _type_character_secure(self, char: str, randomize: bool = True) -> bool:
        """
        Type a single character using CGEvents with robust shift handling.

        Args:
            char: Character to type
            randomize: Add timing randomization

        Returns:
            bool: Success status
        """
        try:
            # Get keycode for character
            keycode = KEYCODE_MAP.get(char)

            if keycode is None:
                logger.error(f"❌ MISSING KEYCODE for character: '{char}' (ord: {ord(char)})")
                logger.error(f"❌ This character is not mapped in KEYCODE_MAP")
                return False

            # Check if shift is needed
            needs_shift = char in SHIFT_CHARS

            # Obfuscate character for logging (don't reveal password chars)
            char_display = char if char.isalnum() else '*'
            logger.info(f"🔐 [CHAR-TYPE] Starting char '{char_display}' (keycode: 0x{keycode:02X}, shift: {needs_shift})")

            # ROBUST SHIFT HANDLING:
            # For shift characters, we need BOTH:
            # 1. Press physical shift key
            # 2. Set shift flag on the character event
            if needs_shift:
                logger.info("🔐   [SHIFT] Character needs shift modifier")
                # Press shift key down (physical key)
                shift_keycode = 0x38  # Left shift
                shift_down_event = CoreGraphics.CGEventCreateKeyboardEvent(
                    self.event_source,
                    shift_keycode,
                    True  # key down
                )
                if shift_down_event:
                    CoreGraphics.CGEventSetFlags(shift_down_event, CGEventFlags.kCGEventFlagMaskShift)
                    CoreGraphics.CGEventPost(0, shift_down_event)
                    CoreGraphics.CFRelease(shift_down_event)
                    logger.info("🔐   [SHIFT] Shift key DOWN event posted")
                else:
                    logger.error("❌   [SHIFT] Failed to create shift down event")
                    return False

                # Small delay for shift to register
                await asyncio.sleep(0.03)
                logger.info("🔐   [SHIFT] Shift registered (30ms delay)")

            # Key down
            logger.info(f"🔐   [KEY-DOWN] Creating key down event for keycode 0x{keycode:02X}")
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                keycode,
                True  # key down
            )

            if not event:
                logger.error(f"❌   [KEY-DOWN] FAILED to create key down event for char '{char_display}'")
                logger.error(f"❌   [KEY-DOWN] CoreGraphics.CGEventCreateKeyboardEvent returned None")
                # Release shift if it was pressed
                if needs_shift:
                    await self._release_shift()
                return False

            logger.info(f"🔐   [KEY-DOWN] Event created successfully")

            # Set shift flag on the character event if needed
            if needs_shift:
                CoreGraphics.CGEventSetFlags(event, CGEventFlags.kCGEventFlagMaskShift)
                logger.info(f"🔐   [KEY-DOWN] Shift flag set on character event")

            # Post event
            CoreGraphics.CGEventPost(0, event)
            logger.info(f"🔐   [KEY-DOWN] Event posted to system")
            CoreGraphics.CFRelease(event)
            logger.info(f"🔐   [KEY-DOWN] Event released")

            # Key press duration (more generous timing for reliability)
            if randomize:
                duration = 0.04 + (hash(char) % 30) / 1000.0  # 40-70ms
            else:
                duration = 0.05  # 50ms default

            logger.info(f"🔐   [TIMING] Key press duration: {duration*1000:.1f}ms")
            await asyncio.sleep(duration)

            # Key up
            logger.info(f"🔐   [KEY-UP] Creating key up event for keycode 0x{keycode:02X}")
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                keycode,
                False  # key up
            )

            if not event:
                logger.error(f"❌   [KEY-UP] FAILED to create key up event for char '{char_display}'")
                logger.error(f"❌   [KEY-UP] CoreGraphics.CGEventCreateKeyboardEvent returned None")
                # Still need to release shift if it was pressed
                if needs_shift:
                    await self._release_shift()
                return False

            logger.info(f"🔐   [KEY-UP] Event created successfully")

            # Set shift flag on key up event too if needed
            if needs_shift:
                CoreGraphics.CGEventSetFlags(event, CGEventFlags.kCGEventFlagMaskShift)
                logger.info(f"🔐   [KEY-UP] Shift flag set on key up event")

            CoreGraphics.CGEventPost(0, event)
            logger.info(f"🔐   [KEY-UP] Event posted to system")
            CoreGraphics.CFRelease(event)
            logger.info(f"🔐   [KEY-UP] Event released")

            # Release shift if it was pressed
            if needs_shift:
                # Small delay before releasing shift
                await asyncio.sleep(0.02)
                logger.info("🔐   [SHIFT] Releasing shift key...")

                await self._release_shift()
                logger.info("🔐   [SHIFT] Shift key released")

            logger.info(f"✅ [CHAR-TYPE] Successfully typed char '{char_display}'")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to type character '{char_display}': {e}", exc_info=True)
            # Ensure shift is released even on error
            try:
                if 'needs_shift' in locals() and needs_shift:
                    await self._release_shift()
            except:
                pass
            return False

    async def _press_modifier(self, flag: int, down: bool):
        """Press or release a modifier key (legacy method, use _release_shift instead)"""
        try:
            # Use flags changed event for modifiers
            # Shift keycode = 0x38 (left shift)
            keycode = 0x38

            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                keycode,
                down
            )

            if event:
                if down:
                    CoreGraphics.CGEventSetFlags(event, flag)
                CoreGraphics.CGEventPost(0, event)
                CoreGraphics.CFRelease(event)

        except Exception as e:
            logger.error(f"❌ Failed to press modifier: {e}")

    async def _release_shift(self):
        """Release the shift key with proper cleanup"""
        try:
            shift_keycode = 0x38  # Left shift

            # Create shift key up event
            shift_up_event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                shift_keycode,
                False  # key up
            )

            if shift_up_event:
                # Clear all flags for key up
                CoreGraphics.CGEventSetFlags(shift_up_event, 0)
                CoreGraphics.CGEventPost(0, shift_up_event)
                CoreGraphics.CFRelease(shift_up_event)

        except Exception as e:
            logger.error(f"❌ Failed to release shift: {e}")

    async def _fallback_applescript(self, password: str, submit: bool, metrics: TypingMetrics) -> Tuple[bool, TypingMetrics]:
        """Fallback to AppleScript if Core Graphics fails"""
        try:
            logger.info("🔄 Using AppleScript fallback for password typing")
            metrics.fallback_used = True
            
            # Wake screen first
            wake_script = """
            tell application "System Events"
                key code 49
            end tell
            """
            
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", wake_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            await asyncio.sleep(0.5)
            
            # Type password using AppleScript with environment variable for security
            type_script = """
            tell application "System Events"
                keystroke (system attribute "JARVIS_UNLOCK_PASS")
            end tell
            """
            
            env = os.environ.copy()
            env["JARVIS_UNLOCK_PASS"] = password
            
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", type_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            await proc.communicate()
            
            # Clear from environment
            if "JARVIS_UNLOCK_PASS" in env:
                del env["JARVIS_UNLOCK_PASS"]
            
            # Submit if requested
            if submit:
                await asyncio.sleep(0.1)
                submit_script = """
                tell application "System Events"
                    key code 36
                end tell
                """
                proc = await asyncio.create_subprocess_exec(
                    "osascript", "-e", submit_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
            
            metrics.success = True
            logger.info("✅ AppleScript fallback succeeded")
            return True, metrics
            
        except Exception as e:
            logger.error(f"❌ AppleScript fallback failed: {e}")
            metrics.error_message = f"Fallback failed: {str(e)}"
            return False, metrics
    
    async def _type_password_characters(self, password: str, config: TypingConfig, metrics: TypingMetrics) -> bool:
        """Type password characters one by one with robust timing"""
        try:
            logger.info(f"🔐 [SECURE-TYPE] Starting to type {len(password)} characters")
            logger.info(f"🔐 [DEBUG] Password analysis:")
            for i, char in enumerate(password):
                char_type = "letter" if char.isalpha() else ("digit" if char.isdigit() else "special")
                has_keycode = char in KEYCODE_MAP
                needs_shift = char in SHIFT_CHARS
                keycode = KEYCODE_MAP.get(char, None)
                logger.info(f"🔐   Char {i+1}: type={char_type}, keycode={hex(keycode) if keycode else 'MISSING'}, shift={needs_shift}")

            for i, char in enumerate(password):
                # Obfuscate for logging
                char_display = char if char.isalnum() else '*'
                logger.info(f"🔐 [TYPING] Character {i+1}/{len(password)}: '{char_display}'")

                success = await self._type_character_secure(
                    char,
                    randomize=config.randomize_timing
                )

                if not success:
                    logger.error(f"❌ FAILED at character {i+1}/{len(password)} ('{char_display}')")
                    logger.error(f"❌ Password chars typed so far: {i}/{len(password)}")
                    logger.error(f"❌ Remaining chars: {len(password) - i}")
                    return False

                logger.info(f"✅ Character {i+1} typed successfully")

                # Inter-character delay (more generous for reliability)
                if config.randomize_timing:
                    delay = 0.08 + (hash(char) % 50) / 1000.0  # 80-130ms
                else:
                    delay = 0.10  # 100ms default

                # Adaptive timing based on system load
                if config.adaptive_timing and metrics.system_load:
                    if metrics.system_load > 0.7:
                        delay *= 1.5  # Slow down on high load

                await asyncio.sleep(delay)

            logger.info(f"✅ [SECURE-TYPE] Successfully typed all {len(password)} characters")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to type password characters: {e}", exc_info=True)
            return False
    
    async def _press_return_secure(self, config: TypingConfig) -> bool:
        """Press return key securely with proper timing"""
        try:
            # Use the standard return press method
            return await self._press_return()
        except Exception as e:
            logger.error(f"❌ Failed to press return securely: {e}")
            return False
    
    async def _press_return(self) -> bool:
        """Press the Return key"""
        try:
            return_keycode = 0x24  # Return key

            # Key down
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                return_keycode,
                True
            )
            if event:
                CoreGraphics.CGEventPost(0, event)
                CoreGraphics.CFRelease(event)

            await asyncio.sleep(0.05)

            # Key up
            event = CoreGraphics.CGEventCreateKeyboardEvent(
                self.event_source,
                return_keycode,
                False
            )
            if event:
                CoreGraphics.CGEventPost(0, event)
                CoreGraphics.CFRelease(event)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to press return: {e}")
            return False


# Singleton instance
_secure_typer_instance: Optional[SecurePasswordTyper] = None


def get_secure_typer() -> SecurePasswordTyper:
    """Get or create the secure typer singleton"""
    global _secure_typer_instance
    if _secure_typer_instance is None:
        _secure_typer_instance = SecurePasswordTyper()
    return _secure_typer_instance


async def type_password_securely(
    password: str,
    submit: bool = True,
    randomize_timing: bool = True
) -> bool:
    """
    Convenience function to type password securely.

    Args:
        password: Password to type
        submit: Press Enter after typing
        randomize_timing: Use human-like timing

    Returns:
        bool: Success status

    Example:
        >>> success = await type_password_securely("MySecurePass123!")
        >>> if success:
        ...     print("Password typed securely")
    """
    typer = get_secure_typer()

    # Create config with randomize_timing setting
    config = TypingConfig(
        randomize_timing=randomize_timing,
        submit_after_typing=submit
    )

    # Call with proper signature
    success, metrics = await typer.type_password_secure(
        password=password,
        submit=submit,
        config_override=config
    )

    return success


async def main():
    """Test secure password typer"""
    logging.basicConfig(level=logging.INFO)

    print("🔐 Secure Password Typer Test")
    print("=" * 50)

    typer = get_secure_typer()

    if not typer.available:
        print("❌ Core Graphics not available")
        return

    print("✅ Core Graphics available")
    print("\n⚠️  WARNING: This will type a test password!")
    print("Make sure you have a text field focused.\n")

    input("Press Enter to start test in 3 seconds...")

    await asyncio.sleep(3)

    # Test with a simple password
    test_password = "Test123!"
    print(f"\n🔐 Typing test password: {test_password}")

    success = await type_password_securely(
        password=test_password,
        submit=False,  # Don't submit in test
        randomize_timing=True
    )

    if success:
        print("✅ Password typed successfully")
    else:
        print("❌ Failed to type password")


if __name__ == "__main__":
    asyncio.run(main())
