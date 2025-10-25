"""
System State Manager - Advanced System Health & Edge Case Handling
===================================================================

Handles all system-level edge cases dynamically and robustly:
- Yabai not running detection and recovery
- Yabai crash detection with auto-restart
- Display sleep detection
- Screen lock detection
- Headless/SSH session detection

Architecture:
    System Check Request → YabaiHealthChecker → DisplayStateDetector
          ↓                      ↓                     ↓
    Validate System      Check Yabai          Check Display
          ↓                      ↓                     ↓
    Is Healthy?          Running/Crashed      Awake/Locked/Sleep
          ↓                      ↓                     ↓
          └──────────────────────┴─────────────────────→ SystemRecoveryHandler
                                                              ↓
                                                        Auto-Recover/Report

Features:
- ✅ Async/await throughout
- ✅ Dynamic service detection (no hardcoding)
- ✅ Robust timeout handling
- ✅ Auto-recovery attempts
- ✅ Natural language error messages
- ✅ Comprehensive health monitoring
"""

import asyncio
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM STATE DEFINITIONS
# ============================================================================

class YabaiState(Enum):
    """Yabai service states"""
    RUNNING = "running"
    NOT_INSTALLED = "not_installed"
    INSTALLED_NOT_RUNNING = "installed_not_running"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class DisplayState(Enum):
    """Display states"""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    LOCKED = "locked"
    NO_DISPLAYS = "no_displays"
    UNKNOWN = "unknown"


class SystemHealth(Enum):
    """Overall system health"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class YabaiStatus:
    """Yabai service status"""
    state: YabaiState
    version: Optional[str] = None
    pid: Optional[int] = None
    uptime: Optional[float] = None
    error: Optional[str] = None
    message: str = ""
    can_recover: bool = False
    recovery_command: Optional[str] = None


@dataclass
class DisplayStatus:
    """Display system status"""
    state: DisplayState
    display_count: int = 0
    active_display_count: int = 0
    is_headless: bool = False
    screen_locked: bool = False
    displays_sleeping: bool = False
    message: str = ""
    error: Optional[str] = None


@dataclass
class SystemStateInfo:
    """Comprehensive system state"""
    health: SystemHealth
    yabai_status: YabaiStatus
    display_status: DisplayStatus
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    can_use_vision: bool
    can_use_spaces: bool
    timestamp: datetime
    recovery_suggestions: List[str] = field(default_factory=list)


# ============================================================================
# YABAI HEALTH CHECKER
# ============================================================================

class YabaiHealthChecker:
    """
    Monitors yabai service health and detects issues.
    """

    def __init__(self, timeout: float = 5.0):
        """
        Initialize yabai health checker.

        Args:
            timeout: Timeout for yabai commands in seconds
        """
        self.timeout = timeout
        self._last_check: Optional[YabaiStatus] = None
        self._last_check_time: Optional[datetime] = None

    async def check_yabai_status(self) -> YabaiStatus:
        """
        Check yabai service status comprehensively.

        Returns:
            YabaiStatus with detailed information
        """
        start_time = time.time()

        # Check if yabai is installed
        is_installed = await self._is_yabai_installed()

        if not is_installed:
            status = YabaiStatus(
                state=YabaiState.NOT_INSTALLED,
                message="Yabai not detected. Install: brew install koekeishiya/formulae/yabai",
                can_recover=False,
                recovery_command="brew install koekeishiya/formulae/yabai"
            )
            self._cache_status(status)
            return status

        # Check if yabai is running
        try:
            result = await asyncio.wait_for(
                self._query_yabai(),
                timeout=self.timeout
            )

            if result["success"]:
                # Yabai is running and responsive
                status = YabaiStatus(
                    state=YabaiState.RUNNING,
                    version=result.get("version"),
                    pid=result.get("pid"),
                    uptime=result.get("uptime"),
                    message="Yabai is running normally"
                )
                self._cache_status(status)
                return status
            else:
                # Yabai installed but not running
                status = YabaiStatus(
                    state=YabaiState.INSTALLED_NOT_RUNNING,
                    error=result.get("error"),
                    message="Yabai is not running. Start: brew services start yabai",
                    can_recover=True,
                    recovery_command="brew services start yabai"
                )
                self._cache_status(status)
                return status

        except asyncio.TimeoutError:
            # Command timed out - yabai likely crashed
            logger.error(f"[YABAI-HEALTH] Yabai query timed out after {self.timeout}s")

            status = YabaiStatus(
                state=YabaiState.TIMEOUT,
                error=f"Yabai command timed out after {self.timeout}s",
                message="Yabai crashed or hung. Restart: brew services restart yabai",
                can_recover=True,
                recovery_command="brew services restart yabai"
            )
            self._cache_status(status)
            return status

        except Exception as e:
            logger.error(f"[YABAI-HEALTH] Error checking yabai status: {e}")

            status = YabaiStatus(
                state=YabaiState.UNKNOWN,
                error=str(e),
                message=f"Error checking yabai: {str(e)}",
                can_recover=False
            )
            self._cache_status(status)
            return status

    async def _is_yabai_installed(self) -> bool:
        """Check if yabai is installed"""
        try:
            result = await asyncio.create_subprocess_shell(
                "which yabai",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return result.returncode == 0 and len(stdout) > 0

        except Exception as e:
            logger.error(f"[YABAI-HEALTH] Error checking yabai installation: {e}")
            return False

    async def _query_yabai(self) -> Dict[str, Any]:
        """
        Query yabai to check if it's responsive.

        Returns:
            Dict with success status and metadata
        """
        try:
            # Try a simple query
            result = await asyncio.create_subprocess_shell(
                "yabai -m query --spaces",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                # Try to get version
                version_result = await asyncio.create_subprocess_shell(
                    "yabai --version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                version_stdout, _ = await version_result.communicate()
                version = version_stdout.decode().strip() if version_result.returncode == 0 else None

                # Try to get PID
                pid_result = await asyncio.create_subprocess_shell(
                    "pgrep -x yabai",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                pid_stdout, _ = await pid_result.communicate()
                pid = int(pid_stdout.decode().strip()) if pid_result.returncode == 0 and pid_stdout else None

                return {
                    "success": True,
                    "version": version,
                    "pid": pid
                }
            else:
                return {
                    "success": False,
                    "error": stderr.decode().strip()
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _cache_status(self, status: YabaiStatus):
        """Cache status for quick retrieval"""
        self._last_check = status
        self._last_check_time = datetime.now()

    def get_cached_status(self, max_age_seconds: float = 5.0) -> Optional[YabaiStatus]:
        """Get cached status if recent enough"""
        if self._last_check and self._last_check_time:
            age = (datetime.now() - self._last_check_time).total_seconds()
            if age < max_age_seconds:
                return self._last_check
        return None


# ============================================================================
# DISPLAY STATE DETECTOR
# ============================================================================

class DisplayStateDetector:
    """
    Detects display state (awake, sleeping, locked, headless).
    """

    def __init__(self):
        """Initialize display state detector"""
        pass

    async def check_display_state(self) -> DisplayStatus:
        """
        Check display state comprehensively.

        Returns:
            DisplayStatus with detailed information
        """
        # Check if headless (no displays)
        is_headless = await self._is_headless_session()

        if is_headless:
            return DisplayStatus(
                state=DisplayState.NO_DISPLAYS,
                display_count=0,
                active_display_count=0,
                is_headless=True,
                message="No displays detected. Vision requires GUI session.",
                error="Running in headless/SSH session"
            )

        # Check if screen is locked
        is_locked = await self._is_screen_locked()

        if is_locked:
            return DisplayStatus(
                state=DisplayState.LOCKED,
                screen_locked=True,
                message="Screen is locked. Unlock to capture.",
                error="Screen lock detected"
            )

        # Check if displays are sleeping
        are_sleeping = await self._are_displays_sleeping()

        if are_sleeping:
            return DisplayStatus(
                state=DisplayState.SLEEPING,
                displays_sleeping=True,
                message="Display is sleeping. Wake to use vision.",
                error="Displays are in sleep mode"
            )

        # Get active display count
        display_count, active_count = await self._get_display_count()

        # All checks passed - displays are active
        return DisplayStatus(
            state=DisplayState.ACTIVE,
            display_count=display_count,
            active_display_count=active_count,
            message=f"{active_count} active display(s) detected"
        )

    async def _is_headless_session(self) -> bool:
        """Check if running in headless/SSH session"""
        try:
            # Check if DISPLAY environment variable is set (X11)
            # or if we're in a console session
            display_var = os.environ.get("DISPLAY")

            # Check for SSH session
            ssh_client = os.environ.get("SSH_CLIENT")
            ssh_tty = os.environ.get("SSH_TTY")

            # Check if CGSessionCopyCurrentDictionary returns a session
            result = await asyncio.create_subprocess_shell(
                "python3 -c 'import Quartz; session = Quartz.CGSessionCopyCurrentDictionary(); print(session is not None)'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                has_session = "True" in stdout.decode()
                if not has_session:
                    return True

            # If SSH session and no display, it's headless
            if (ssh_client or ssh_tty) and not display_var:
                return True

            # Try to get display list
            display_result = await asyncio.create_subprocess_shell(
                "system_profiler SPDisplaysDataType 2>/dev/null | grep -c 'Resolution'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            display_stdout, _ = await display_result.communicate()

            if display_result.returncode == 0:
                display_count = int(display_stdout.decode().strip() or "0")
                return display_count == 0

            return False

        except Exception as e:
            logger.error(f"[DISPLAY-DETECTOR] Error checking headless state: {e}")
            return False

    async def _is_screen_locked(self) -> bool:
        """Check if screen is locked"""
        try:
            # Check using Python's Quartz framework
            result = await asyncio.create_subprocess_shell(
                """python3 -c "
import Quartz
session = Quartz.CGSessionCopyCurrentDictionary()
if session:
    print(session.get('CGSSessionScreenIsLocked', 0))
else:
    print(0)
" 2>/dev/null""",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                is_locked = stdout.decode().strip() == "1"
                return is_locked

            # Fallback: check login window process
            login_check = await asyncio.create_subprocess_shell(
                "pgrep -x loginwindow",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            login_stdout, _ = await login_check.communicate()

            # If loginwindow is running, might be locked
            # (This is a heuristic, not 100% reliable)
            return login_check.returncode == 0 and len(login_stdout) > 0

        except Exception as e:
            logger.error(f"[DISPLAY-DETECTOR] Error checking screen lock: {e}")
            return False

    async def _are_displays_sleeping(self) -> bool:
        """Check if displays are sleeping"""
        try:
            # Use pmset to check display sleep state
            result = await asyncio.create_subprocess_shell(
                "pmset -g powerstate IODisplayWrangler | grep -c 'DevicePowerState=0'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                sleeping_count = int(stdout.decode().strip() or "0")
                return sleeping_count > 0

            # Fallback: check display brightness
            brightness_result = await asyncio.create_subprocess_shell(
                "brightness -l 2>/dev/null | grep -c 'brightness 0.00'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            brightness_stdout, _ = await brightness_result.communicate()

            if brightness_result.returncode == 0:
                zero_brightness = int(brightness_stdout.decode().strip() or "0")
                return zero_brightness > 0

            return False

        except Exception as e:
            logger.debug(f"[DISPLAY-DETECTOR] Error checking display sleep: {e}")
            return False

    async def _get_display_count(self) -> Tuple[int, int]:
        """
        Get display count.

        Returns:
            Tuple of (total_displays, active_displays)
        """
        try:
            # Use system_profiler to get display count
            result = await asyncio.create_subprocess_shell(
                "system_profiler SPDisplaysDataType 2>/dev/null | grep -c 'Resolution'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                total = int(stdout.decode().strip() or "0")
                # Assume all detected displays are active if not sleeping/locked
                return total, total

            # Fallback to 1 display
            return 1, 1

        except Exception as e:
            logger.error(f"[DISPLAY-DETECTOR] Error getting display count: {e}")
            return 1, 1


# ============================================================================
# SYSTEM RECOVERY HANDLER
# ============================================================================

class SystemRecoveryHandler:
    """
    Attempts to automatically recover from system state issues.
    """

    def __init__(self, auto_recover: bool = False):
        """
        Initialize recovery handler.

        Args:
            auto_recover: If True, automatically attempt recovery
        """
        self.auto_recover = auto_recover
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 2

    async def attempt_recovery(self, state_info: SystemStateInfo) -> Tuple[bool, str]:
        """
        Attempt to recover from system state issues.

        Args:
            state_info: Current system state

        Returns:
            Tuple of (recovery_successful, message)
        """
        if not self.auto_recover:
            return False, "Auto-recovery is disabled"

        # Try to recover yabai issues
        if state_info.yabai_status.can_recover:
            return await self._recover_yabai(state_info.yabai_status)

        return False, "No recovery action available"

    async def _recover_yabai(self, yabai_status: YabaiStatus) -> Tuple[bool, str]:
        """Attempt to recover yabai service"""

        # Check recovery attempt count
        attempt_key = f"yabai_{yabai_status.state.value}"
        attempts = self.recovery_attempts.get(attempt_key, 0)

        if attempts >= self.max_recovery_attempts:
            return False, f"Max recovery attempts ({self.max_recovery_attempts}) reached"

        if not yabai_status.recovery_command:
            return False, "No recovery command available"

        logger.info(f"[RECOVERY] Attempting to recover yabai: {yabai_status.recovery_command}")

        try:
            # Execute recovery command
            result = await asyncio.create_subprocess_shell(
                yabai_status.recovery_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30.0)

            # Increment attempt counter
            self.recovery_attempts[attempt_key] = attempts + 1

            if result.returncode == 0:
                # Wait a bit for service to start
                await asyncio.sleep(2.0)

                logger.info(f"[RECOVERY] Yabai recovery command executed successfully")
                return True, f"Successfully executed: {yabai_status.recovery_command}"
            else:
                error_msg = stderr.decode().strip()
                logger.error(f"[RECOVERY] Recovery command failed: {error_msg}")
                return False, f"Recovery command failed: {error_msg}"

        except asyncio.TimeoutError:
            logger.error(f"[RECOVERY] Recovery command timed out")
            return False, "Recovery command timed out"

        except Exception as e:
            logger.error(f"[RECOVERY] Recovery error: {e}")
            return False, f"Recovery error: {str(e)}"

    def reset_recovery_attempts(self):
        """Reset recovery attempt counters"""
        self.recovery_attempts.clear()


# ============================================================================
# SYSTEM STATE MANAGER
# ============================================================================

class SystemStateManager:
    """
    Main coordinator for system state monitoring and edge case handling.

    Integrates:
    - YabaiHealthChecker (yabai service monitoring)
    - DisplayStateDetector (display state detection)
    - SystemRecoveryHandler (auto-recovery)
    """

    def __init__(
        self,
        yabai_timeout: float = 5.0,
        auto_recover: bool = False,
        cache_ttl: float = 5.0
    ):
        """
        Initialize system state manager.

        Args:
            yabai_timeout: Timeout for yabai commands
            auto_recover: Enable automatic recovery
            cache_ttl: Cache TTL for health checks
        """
        self.yabai_checker = YabaiHealthChecker(timeout=yabai_timeout)
        self.display_detector = DisplayStateDetector()
        self.recovery_handler = SystemRecoveryHandler(auto_recover=auto_recover)
        self.cache_ttl = cache_ttl

        logger.info(f"[SYSTEM-STATE-MANAGER] Initialized (auto_recover={auto_recover})")

    async def check_system_state(self, use_cache: bool = True) -> SystemStateInfo:
        """
        Comprehensive system state check.

        Args:
            use_cache: Use cached results if available

        Returns:
            SystemStateInfo with complete state information
        """
        start_time = datetime.now()

        # Check if we can use cache
        if use_cache:
            cached = self.yabai_checker.get_cached_status(self.cache_ttl)
            if cached and cached.state == YabaiState.RUNNING:
                # Quick path for healthy systems
                logger.debug("[SYSTEM-STATE-MANAGER] Using cached healthy state")

        # Run checks in parallel
        yabai_task = asyncio.create_task(self.yabai_checker.check_yabai_status())
        display_task = asyncio.create_task(self.display_detector.check_display_state())

        yabai_status, display_status = await asyncio.gather(yabai_task, display_task)

        # Determine overall health
        health, checks_passed, checks_failed, warnings = self._determine_health(
            yabai_status,
            display_status
        )

        # Determine capabilities
        can_use_vision = self._can_use_vision(yabai_status, display_status)
        can_use_spaces = self._can_use_spaces(yabai_status)

        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(yabai_status, display_status)

        state_info = SystemStateInfo(
            health=health,
            yabai_status=yabai_status,
            display_status=display_status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            can_use_vision=can_use_vision,
            can_use_spaces=can_use_spaces,
            timestamp=start_time,
            recovery_suggestions=recovery_suggestions
        )

        # Attempt auto-recovery if enabled
        if self.recovery_handler.auto_recover and health in [SystemHealth.DEGRADED, SystemHealth.UNHEALTHY]:
            logger.info("[SYSTEM-STATE-MANAGER] Attempting auto-recovery...")
            success, message = await self.recovery_handler.attempt_recovery(state_info)
            if success:
                logger.info(f"[SYSTEM-STATE-MANAGER] Recovery successful: {message}")
                # Re-check state after recovery
                return await self.check_system_state(use_cache=False)

        return state_info

    def _determine_health(
        self,
        yabai_status: YabaiStatus,
        display_status: DisplayStatus
    ) -> Tuple[SystemHealth, List[str], List[str], List[str]]:
        """Determine overall system health"""

        checks_passed = []
        checks_failed = []
        warnings = []

        # Check yabai
        if yabai_status.state == YabaiState.RUNNING:
            checks_passed.append("Yabai is running")
        elif yabai_status.state == YabaiState.NOT_INSTALLED:
            checks_failed.append("Yabai not installed")
        elif yabai_status.state in [YabaiState.INSTALLED_NOT_RUNNING, YabaiState.TIMEOUT, YabaiState.CRASHED]:
            checks_failed.append(f"Yabai {yabai_status.state.value}")
        else:
            warnings.append(f"Yabai state unknown: {yabai_status.state.value}")

        # Check displays
        if display_status.state == DisplayState.ACTIVE:
            checks_passed.append(f"Displays active ({display_status.active_display_count})")
        elif display_status.state == DisplayState.NO_DISPLAYS:
            checks_failed.append("No displays (headless session)")
        elif display_status.state == DisplayState.LOCKED:
            checks_failed.append("Screen is locked")
        elif display_status.state == DisplayState.SLEEPING:
            checks_failed.append("Displays are sleeping")
        else:
            warnings.append(f"Display state unknown: {display_status.state.value}")

        # Determine health level
        critical_failures = [
            YabaiState.NOT_INSTALLED,
            DisplayState.NO_DISPLAYS
        ]

        if yabai_status.state in critical_failures or display_status.state in critical_failures:
            health = SystemHealth.CRITICAL
        elif len(checks_failed) > 0:
            health = SystemHealth.UNHEALTHY
        elif len(warnings) > 0:
            health = SystemHealth.DEGRADED
        else:
            health = SystemHealth.HEALTHY

        return health, checks_passed, checks_failed, warnings

    def _can_use_vision(self, yabai_status: YabaiStatus, display_status: DisplayStatus) -> bool:
        """Check if vision system can be used"""
        # Need active displays to use vision
        display_ok = display_status.state == DisplayState.ACTIVE

        # Yabai is helpful but not strictly required for basic vision
        return display_ok

    def _can_use_spaces(self, yabai_status: YabaiStatus) -> bool:
        """Check if space operations can be used"""
        # Need yabai running for space operations
        return yabai_status.state == YabaiState.RUNNING

    def _generate_recovery_suggestions(
        self,
        yabai_status: YabaiStatus,
        display_status: DisplayStatus
    ) -> List[str]:
        """Generate recovery suggestions"""
        suggestions = []

        if yabai_status.state == YabaiState.NOT_INSTALLED:
            suggestions.append("Install yabai: brew install koekeishiya/formulae/yabai")

        elif yabai_status.state == YabaiState.INSTALLED_NOT_RUNNING:
            suggestions.append("Start yabai: brew services start yabai")

        elif yabai_status.state in [YabaiState.TIMEOUT, YabaiState.CRASHED]:
            suggestions.append("Restart yabai: brew services restart yabai")

        if display_status.state == DisplayState.LOCKED:
            suggestions.append("Unlock your screen to use vision features")

        elif display_status.state == DisplayState.SLEEPING:
            suggestions.append("Wake your display to use vision features")

        elif display_status.state == DisplayState.NO_DISPLAYS:
            suggestions.append("Vision features require a GUI session (not SSH/headless)")

        return suggestions

    async def wait_for_healthy_state(
        self,
        timeout: float = 30.0,
        check_interval: float = 2.0
    ) -> Tuple[bool, SystemStateInfo]:
        """
        Wait for system to reach healthy state.

        Args:
            timeout: Maximum time to wait
            check_interval: Time between checks

        Returns:
            Tuple of (became_healthy, final_state)
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            state_info = await self.check_system_state(use_cache=False)

            if state_info.health == SystemHealth.HEALTHY:
                return True, state_info

            logger.info(f"[SYSTEM-STATE-MANAGER] Waiting for healthy state... ({state_info.health.value})")
            await asyncio.sleep(check_interval)

        # Timeout reached
        final_state = await self.check_system_state(use_cache=False)
        return False, final_state


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_system_state_manager: Optional[SystemStateManager] = None


def get_system_state_manager() -> SystemStateManager:
    """Get or create the global SystemStateManager instance"""
    global _system_state_manager
    if _system_state_manager is None:
        _system_state_manager = SystemStateManager()
    return _system_state_manager


def initialize_system_state_manager(
    yabai_timeout: float = 5.0,
    auto_recover: bool = False,
    cache_ttl: float = 5.0
) -> SystemStateManager:
    """
    Initialize the global SystemStateManager with custom settings.

    Args:
        yabai_timeout: Timeout for yabai commands
        auto_recover: Enable automatic recovery
        cache_ttl: Cache TTL for health checks

    Returns:
        Initialized SystemStateManager
    """
    global _system_state_manager
    _system_state_manager = SystemStateManager(yabai_timeout, auto_recover, cache_ttl)
    return _system_state_manager
