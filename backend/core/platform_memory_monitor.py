"""
Platform-Aware Memory Pressure Monitor

This module provides platform-specific memory pressure monitoring to prevent
unnecessary GCP VM creation by using OS-native memory pressure detection methods.

Features:
- macOS: vm_stat + memory_pressure + page outs
- Linux: PSI (Pressure Stall Information) + /proc/meminfo reclaimable
- Distinguishes between "high usage" (cache) vs "high pressure" (actual OOM risk)
- Prevents false alarms from cached memory

Philosophy:
- macOS: Uses memory pressure levels (normal/warn/critical)
- Linux: Uses PSI metrics + reclaimable memory (cache/buffers can be freed)
- Both: Only trigger GCP when ACTUAL memory pressure exists, not just high %

Example:
    >>> monitor = get_memory_monitor()
    >>> snapshot = await monitor.get_memory_pressure()
    >>> should_create, reason = monitor.should_create_gcp_vm(snapshot)
    >>> print(f"Create VM: {should_create}, Reason: {reason}")
"""

import asyncio
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryPressureSnapshot:
    """Platform-aware memory pressure snapshot containing comprehensive memory metrics.
    
    This dataclass captures both raw memory statistics and platform-specific pressure
    indicators to provide accurate memory pressure assessment across different operating systems.
    
    Attributes:
        timestamp: When this snapshot was taken
        platform: Operating system platform ("darwin" or "linux")
        total_gb: Total system memory in GB
        available_gb: Available memory in GB (OS-reported)
        used_gb: Used memory in GB
        usage_percent: Memory usage percentage
        macos_pressure_level: macOS memory pressure level ("normal", "warn", "critical")
        macos_page_outs: Cumulative page outs from vm_stat
        macos_is_swapping: Whether system is actively swapping to disk
        linux_psi_some_avg10: PSI metric - % time some processes stalled on memory
        linux_psi_full_avg10: PSI metric - % time all processes stalled on memory
        linux_reclaimable_gb: Cache + buffers that can be freed (GB)
        linux_actual_pressure_gb: Real unavailable memory after accounting for cache
        pressure_level: Universal pressure assessment level
        gcp_shift_recommended: Whether GCP VM creation is recommended
        gcp_shift_urgent: Whether GCP VM creation is urgent
        reasoning: Human-readable explanation of the pressure assessment
    """

    timestamp: datetime
    platform: str  # "darwin" or "linux"

    # Raw metrics (all platforms)
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percent: float

    # macOS-specific
    macos_pressure_level: Optional[str] = None  # "normal", "warn", "critical"
    macos_page_outs: Optional[int] = None
    macos_is_swapping: bool = False

    # Linux-specific
    linux_psi_some_avg10: Optional[float] = None  # % time some processes stalled
    linux_psi_full_avg10: Optional[float] = None  # % time all processes stalled
    linux_reclaimable_gb: Optional[float] = None  # Cache + buffers that can be freed
    linux_actual_pressure_gb: Optional[float] = None  # Real unavailable memory

    # Universal pressure assessment
    pressure_level: str = "normal"  # "low", "normal", "elevated", "high", "critical"
    gcp_shift_recommended: bool = False
    gcp_shift_urgent: bool = False
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary format for serialization.
        
        Returns:
            Dictionary representation of the snapshot with rounded numeric values
            and ISO-formatted timestamp.
            
        Example:
            >>> snapshot = MemoryPressureSnapshot(...)
            >>> data = snapshot.to_dict()
            >>> print(data['pressure_level'])
            'normal'
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "platform": self.platform,
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "usage_percent": round(self.usage_percent, 1),
            "macos_pressure_level": self.macos_pressure_level,
            "macos_page_outs": self.macos_page_outs,
            "linux_psi_some": (
                round(self.linux_psi_some_avg10, 2) if self.linux_psi_some_avg10 else None
            ),
            "linux_psi_full": (
                round(self.linux_psi_full_avg10, 2) if self.linux_psi_full_avg10 else None
            ),
            "linux_reclaimable_gb": (
                round(self.linux_reclaimable_gb, 2) if self.linux_reclaimable_gb else None
            ),
            "linux_actual_pressure_gb": (
                round(self.linux_actual_pressure_gb, 2) if self.linux_actual_pressure_gb else None
            ),
            "pressure_level": self.pressure_level,
            "gcp_shift_recommended": self.gcp_shift_recommended,
            "gcp_shift_urgent": self.gcp_shift_urgent,
            "reasoning": self.reasoning,
        }


class PlatformMemoryMonitor:
    """Platform-aware memory pressure monitoring system.
    
    This class provides accurate memory pressure detection across different operating
    systems to prevent unnecessary GCP VM creation. It uses OS-native methods to
    distinguish between high memory usage (which may be cache) and actual memory
    pressure that could lead to OOM conditions.
    
    The monitor uses different strategies per platform:
    - macOS: memory_pressure command + vm_stat page outs
    - Linux: PSI (Pressure Stall Information) + /proc/meminfo analysis
    - Fallback: Conservative percentage-based thresholds
    
    Attributes:
        platform: Current operating system platform
        is_macos: Whether running on macOS
        is_linux: Whether running on Linux
        last_page_outs: Previous page outs count for delta calculation
        last_check_time: Previous check timestamp for rate calculation
        psi_memory_path: Path to Linux PSI memory file
        meminfo_path: Path to Linux meminfo file
        macos_thresholds: macOS-specific threshold configuration
        linux_thresholds: Linux-specific threshold configuration
    """

    def __init__(self) -> None:
        """Initialize the platform memory monitor.
        
        Detects the current platform and sets up platform-specific monitoring
        capabilities including PSI availability on Linux and threshold configuration.
        """
        self.platform = platform.system().lower()
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"

        # macOS tracking
        self.last_page_outs: Optional[int] = None  # Track cumulative for delta calculation
        self.last_check_time: Optional[float] = None

        # Linux PSI paths
        self.psi_memory_path = Path("/proc/pressure/memory")
        self.meminfo_path = Path("/proc/meminfo")

        # Thresholds (conservative to prevent false alarms)
        self.macos_thresholds = {
            "page_outs_per_second": 100,  # 100 pages/sec = active swapping
            "page_outs_delta_threshold": 5000,  # 5000 pages since last check
        }

        self.linux_thresholds = {
            "psi_some_warning": 10.0,  # 10% = some memory pressure
            "psi_some_critical": 30.0,  # 30% = high memory pressure
            "psi_full_warning": 1.0,  # 1% = processes blocked on memory
            "psi_full_critical": 5.0,  # 5% = severe memory stalls
            "min_available_gb": 2.0,  # Minimum available before alarm (after cache)
        }

        logger.info(f"🧠 PlatformMemoryMonitor initialized (platform: {self.platform})")
        if self.is_linux:
            psi_available = self.psi_memory_path.exists()
            logger.info(f"   Linux PSI available: {psi_available}")

    async def get_memory_pressure(self) -> MemoryPressureSnapshot:
        """Get current memory pressure using platform-native methods.
        
        Collects comprehensive memory metrics using the most appropriate method
        for the current platform, then analyzes the data to determine actual
        memory pressure levels.
        
        Returns:
            MemoryPressureSnapshot containing all relevant memory metrics and
            pressure assessment for the current platform.
            
        Example:
            >>> monitor = PlatformMemoryMonitor()
            >>> snapshot = await monitor.get_memory_pressure()
            >>> print(f"Pressure: {snapshot.pressure_level}")
            'normal'
        """
        # Get base metrics (all platforms)
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        usage_percent = mem.percent

        snapshot = MemoryPressureSnapshot(
            timestamp=datetime.now(),
            platform=self.platform,
            total_gb=total_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            usage_percent=usage_percent,
        )

        # Platform-specific pressure detection
        if self.is_macos:
            await self._add_macos_pressure(snapshot)
        elif self.is_linux:
            await self._add_linux_pressure(snapshot)
        else:
            # Fallback for unknown platforms
            self._add_fallback_pressure(snapshot)

        return snapshot

    async def _add_macos_pressure(self, snapshot: MemoryPressureSnapshot) -> None:
        """Add macOS-specific memory pressure detection to snapshot.
        
        Uses macOS-native tools to detect memory pressure:
        1. memory_pressure command for system pressure levels
        2. vm_stat for page out detection (swapping activity)
        3. Rate-based analysis to detect active swapping
        
        Args:
            snapshot: MemoryPressureSnapshot to populate with macOS metrics
        """
        # Method 1: memory_pressure command (most accurate)
        pressure_level = await self._get_macos_memory_pressure_cmd()
        snapshot.macos_pressure_level = pressure_level

        # Method 2: vm_stat page outs (swapping detection)
        page_outs = await self._get_macos_page_outs()
        snapshot.macos_page_outs = page_outs

        # Check if actively swapping (page outs increasing)
        current_time = time.time()

        if self.last_page_outs is not None and self.last_check_time is not None:
            page_outs_delta = page_outs - self.last_page_outs
            time_delta = current_time - self.last_check_time

            # Calculate pages per second
            if time_delta > 0:
                pages_per_second = page_outs_delta / time_delta
                snapshot.macos_is_swapping = (
                    pages_per_second > self.macos_thresholds["page_outs_per_second"]
                    or page_outs_delta > self.macos_thresholds["page_outs_delta_threshold"]
                )
            else:
                snapshot.macos_is_swapping = False
        else:
            # First check - can't determine if swapping yet
            snapshot.macos_is_swapping = False

        self.last_page_outs = page_outs
        self.last_check_time = current_time

        # Determine pressure level and GCP shift recommendation
        if pressure_level == "critical" or snapshot.macos_is_swapping:
            snapshot.pressure_level = "critical"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = True
            snapshot.reasoning = f"macOS reports CRITICAL pressure (level={pressure_level}, swapping={snapshot.macos_is_swapping})"

        elif pressure_level == "warn":
            snapshot.pressure_level = "high"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = False
            snapshot.reasoning = f"macOS reports WARN pressure level"

        elif snapshot.available_gb < 2.0:
            snapshot.pressure_level = "elevated"
            snapshot.gcp_shift_recommended = False
            snapshot.gcp_shift_urgent = False
            snapshot.reasoning = (
                f"Low available memory ({snapshot.available_gb:.1f}GB) but no system pressure yet"
            )

        else:
            snapshot.pressure_level = "normal"
            snapshot.gcp_shift_recommended = False
            snapshot.gcp_shift_urgent = False
            snapshot.reasoning = (
                f"macOS pressure={pressure_level}, {snapshot.available_gb:.1f}GB available"
            )

        logger.debug(
            f"macOS pressure: {snapshot.pressure_level} | "
            f"level={pressure_level} | page_outs={page_outs} | "
            f"available={snapshot.available_gb:.1f}GB"
        )

    async def _add_linux_pressure(self, snapshot: MemoryPressureSnapshot) -> None:
        """Add Linux-specific memory pressure detection using PSI + reclaimable memory.
        
        Uses Linux-specific mechanisms to detect real memory pressure:
        1. PSI (Pressure Stall Information) for actual process stalls
        2. /proc/meminfo analysis to distinguish cache from unavailable memory
        3. Combined analysis to avoid false alarms from cached memory
        
        Args:
            snapshot: MemoryPressureSnapshot to populate with Linux metrics
        """
        # Method 1: PSI (Pressure Stall Information) - kernel 4.20+
        psi_some, psi_full = await self._get_linux_psi()
        snapshot.linux_psi_some_avg10 = psi_some
        snapshot.linux_psi_full_avg10 = psi_full

        # Method 2: /proc/meminfo - calculate reclaimable memory (cache + buffers)
        reclaimable_gb, actual_pressure_gb = await self._get_linux_reclaimable()
        snapshot.linux_reclaimable_gb = reclaimable_gb
        snapshot.linux_actual_pressure_gb = actual_pressure_gb

        # Determine pressure level using PSI + reclaimable memory
        reasons = []

        # PSI metrics indicate actual memory pressure
        if psi_full is not None and psi_full >= self.linux_thresholds["psi_full_critical"]:
            snapshot.pressure_level = "critical"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = True
            reasons.append(f"PSI full={psi_full:.1f}% (processes stalled on memory)")

        elif psi_full is not None and psi_full >= self.linux_thresholds["psi_full_warning"]:
            snapshot.pressure_level = "high"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = False
            reasons.append(f"PSI full={psi_full:.1f}% (some memory stalls)")

        elif psi_some is not None and psi_some >= self.linux_thresholds["psi_some_critical"]:
            snapshot.pressure_level = "high"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = False
            reasons.append(f"PSI some={psi_some:.1f}% (high memory pressure)")

        elif psi_some is not None and psi_some >= self.linux_thresholds["psi_some_warning"]:
            snapshot.pressure_level = "elevated"
            snapshot.gcp_shift_recommended = False
            snapshot.gcp_shift_urgent = False
            reasons.append(f"PSI some={psi_some:.1f}% (moderate pressure)")

        # Check actual available memory (after excluding reclaimable cache)
        elif (
            actual_pressure_gb is not None
            and actual_pressure_gb < self.linux_thresholds["min_available_gb"]
        ):
            snapshot.pressure_level = "elevated"
            snapshot.gcp_shift_recommended = False
            snapshot.gcp_shift_urgent = False
            reasons.append(f"Low actual available: {actual_pressure_gb:.1f}GB (after cache)")

        else:
            snapshot.pressure_level = "normal"
            snapshot.gcp_shift_recommended = False
            snapshot.gcp_shift_urgent = False

            # Explain why high % usage is OK
            if snapshot.usage_percent > 80:
                reasons.append(
                    f"{snapshot.usage_percent:.0f}% used but {reclaimable_gb:.1f}GB is cache "
                    f"(actual pressure: {actual_pressure_gb:.1f}GB available)"
                )
            else:
                reasons.append(f"{snapshot.available_gb:.1f}GB available, no pressure")

        snapshot.reasoning = "; ".join(reasons) if reasons else "Normal operation"

        logger.debug(
            f"Linux pressure: {snapshot.pressure_level} | "
            f"PSI some={psi_some:.1f}% full={psi_full:.1f}% | "
            f"reclaimable={reclaimable_gb:.1f}GB | actual_avail={actual_pressure_gb:.1f}GB"
        )

    def _add_fallback_pressure(self, snapshot: MemoryPressureSnapshot) -> None:
        """Add fallback pressure detection for unknown platforms.
        
        Uses conservative percentage-based thresholds when platform-specific
        monitoring is not available. This provides basic protection but may
        not be as accurate as native platform methods.
        
        Args:
            snapshot: MemoryPressureSnapshot to populate with fallback metrics
        """
        # Use conservative percentage-based thresholds
        if snapshot.available_gb < 1.0:
            snapshot.pressure_level = "critical"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = True
            snapshot.reasoning = f"Less than 1GB available"
        elif snapshot.available_gb < 2.0:
            snapshot.pressure_level = "high"
            snapshot.gcp_shift_recommended = True
            snapshot.gcp_shift_urgent = False
            snapshot.reasoning = f"Less than 2GB available"
        elif snapshot.usage_percent > 90:
            snapshot.pressure_level = "elevated"
            snapshot.gcp_shift_recommended = False
            snapshot.reasoning = (
                f"High usage ({snapshot.usage_percent:.0f}%) but platform-specific pressure unknown"
            )
        else:
            snapshot.pressure_level = "normal"
            snapshot.gcp_shift_recommended = False
            snapshot.reasoning = f"{snapshot.available_gb:.1f}GB available"

    async def _get_macos_memory_pressure_cmd(self) -> str:
        """Get macOS memory pressure level using memory_pressure command.
        
        Executes the macOS memory_pressure command to get the system's
        assessment of current memory pressure levels.
        
        Returns:
            Memory pressure level: "normal", "warn", "critical", or "unknown"
            if the command fails or times out.
            
        Raises:
            No exceptions raised - errors are logged and "unknown" returned.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "memory_pressure", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            output = stdout.decode().lower()

            # Parse output: "System-wide memory free percentage: 25%"
            # Status: normal/warn/critical
            if "status: critical" in output or "critical" in output:
                return "critical"
            elif "status: warn" in output or "warn" in output:
                return "warn"
            else:
                return "normal"

        except asyncio.TimeoutError:
            logger.warning("memory_pressure command timeout")
            return "unknown"
        except FileNotFoundError:
            logger.debug("memory_pressure command not found")
            return "unknown"
        except Exception as e:
            logger.debug(f"Failed to get macOS memory pressure: {e}")
            return "unknown"

    async def _get_macos_page_outs(self) -> int:
        """Get macOS cumulative page outs from vm_stat command.
        
        Executes vm_stat to get the cumulative number of pages that have
        been paged out to disk, which indicates swapping activity.
        
        Returns:
            Cumulative number of pages paged out, or 0 if command fails.
            
        Raises:
            No exceptions raised - errors are logged and 0 returned.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "vm_stat", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            output = stdout.decode()

            # Parse "Pages paged out: 12345."
            for line in output.split("\n"):
                if "Pages paged out:" in line or "Pageouts:" in line:
                    page_outs_str = line.split(":")[1].strip().rstrip(".")
                    return int(page_outs_str)

            return 0

        except Exception as e:
            logger.debug(f"Failed to get page outs: {e}")
            return 0

    async def _get_linux_psi(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Linux PSI (Pressure Stall Information) metrics.
        
        Reads PSI memory metrics from /proc/pressure/memory to get accurate
        information about memory pressure from the kernel's perspective.
        
        Returns:
            Tuple of (psi_some_avg10, psi_full_avg10) where:
            - psi_some: % of time at least one process stalled on memory
            - psi_full: % of time ALL processes stalled on memory (severe)
            Both values are None if PSI is not available.
            
        Raises:
            No exceptions raised - errors are logged and (None, None) returned.
        """
        if not self.psi_memory_path.exists():
            return None, None

        try:
            content = self.psi_memory_path.read_text()

            # Parse PSI format:
            # some avg10=0.00 avg60=0.00 avg300=0.00 total=0
            # full avg10=0.00 avg60=0.00 avg300=0.00 total=0

            psi_some = None
            psi_full = None

            for line in content.strip().split("\n"):
                if line.startswith("some"):
                    # Extract avg10 value
                    for part in line.split():
                        if part.startswith("avg10="):
                            psi_some = float(part.split("=")[1])

                elif line.startswith("full"):
                    for part in line.split():
                        if part.startswith("avg10="):
                            psi_full = float(part.split("=")[1])

            return psi_some, psi_full

        except Exception as e:
            logger.debug(f"Failed to read PSI: {e}")
            return None, None

    async def _get_linux_reclaimable(self) -> Tuple[Optional[float], Optional[float]]:
        """Get Linux reclaimable memory from /proc/meminfo.
        
        Analyzes /proc/meminfo to determine how much memory is actually
        reclaimable (cache/buffers) vs truly unavailable, providing a more
        accurate picture of memory pressure than simple usage percentages.
        
        Returns:
            Tuple of (reclaimable_gb, actual_pressure_gb) where:
            - reclaimable_gb: Cache + buffers that can be instantly freed
            - actual_pressure_gb: Real available memory (MemAvailable - considers reclaimable)
            Both values are None if /proc/meminfo is not available.
            
        Raises:
            No exceptions raised - errors are logged and (None, None) returned.
        """
        if not self.meminfo_path.exists():
            return None, None

        try:
            content = self.meminfo_path.read_text()

            # Parse /proc/meminfo
            mem_available = None
            buffers = None
            cached = None
            sreclaimable = None

            for line in content.split("\n"):
                if line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) / 1024  # KB -> MB
                elif line.startswith("Buffers:"):
                    buffers = int(line.split()[1]) / 1024
                elif line.startswith("Cached:"):
                    cached = int(line.split()[1]) / 1024
                elif line.startswith("SReclaimable:"):
                    sreclaimable = int(line.split()[1]) / 1024

            if mem_available is None:
                return None, None

            # Calculate reclaimable (cache that can be freed)
            reclaimable_mb = (buffers or 0) + (cached or 0) + (sreclaimable or 0)
            reclaimable_gb = reclaimable_mb / 1024

            # MemAvailable already accounts for reclaimable memory
            actual_available_gb = mem_available / 1024

            return reclaimable_gb, actual_available_gb

        except Exception as e:
            logger.debug(f"Failed to read meminfo: {e}")
            return None, None

    def should_create_gcp_vm(self, snapshot: MemoryPressureSnapshot) -> Tuple[bool, str]:
        """Decide if we should create a GCP VM based on memory pressure.
        
        Analyzes the memory pressure snapshot to determine whether GCP VM
        creation is warranted based on the current memory situation.
        
        Args:
            snapshot: MemoryPressureSnapshot containing current memory metrics
            
        Returns:
            Tuple of (should_create, reason) where:
            - should_create: Boolean indicating if VM creation is recommended
            - reason: Human-readable explanation of the decision
            
        Example:
            >>> should_create, reason = monitor.should_create_gcp_vm(snapshot)
            >>> if should_create:
            ...     print(f"Creating VM: {reason}")
        """
        if snapshot.gcp_shift_urgent:
            return True, f"URGENT: {snapshot.reasoning}"

        if snapshot.gcp_shift_recommended:
            return True, f"RECOMMENDED: {snapshot.reasoning}"

        return False, f"NOT NEEDED: {snapshot.reasoning}"

    async def continuous_monitor(
        self, 
        interval_seconds: int = 5, 
        callback: Optional[Callable[[MemoryPressureSnapshot], Any]] = None
    ) -> None:
        """Continuously monitor memory pressure and call callback on changes.
        
        Runs an infinite loop monitoring memory pressure at regular intervals.
        Calls the provided callback function whenever the pressure level changes,
        and logs warnings when GCP VM creation is recommended.
        
        Args:
            interval_seconds: Time between pressure checks in seconds
            callback: Optional async function(snapshot) called on pressure level changes
            
        Raises:
            No exceptions propagated - errors are logged and monitoring continues.
            
        Example:
            >>> async def on_pressure_change(snapshot):
            ...     print(f"Pressure changed to: {snapshot.pressure_level}")
            >>> await monitor.continuous_monitor(interval_seconds=10, callback=on_pressure_change)
        """
        last_pressure_level = None

        while True:
            try:
                snapshot = await self.get_memory_pressure()

                # Call callback on pressure level changes
                if snapshot.pressure_level != last_pressure_level:
                    logger.info(
                        f"🧠 Memory pressure: {snapshot.pressure_level.upper()} | "
                        f"{snapshot.reasoning}"
                    )

                    if callback:
                        await callback(snapshot)

                    last_pressure_level = snapshot.pressure_level

                # Log warnings for high pressure
                if snapshot.gcp_shift_recommended:
                    urgency = "URGENT" if snapshot.gcp_shift_urgent else "RECOMMENDED"
                    logger.warning(f"⚠️  GCP shift {urgency}: {snapshot.reasoning}")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(interval_seconds)


# Global singleton
_monitor: Optional[PlatformMemoryMonitor] = None


def get_memory_monitor() -> PlatformMemoryMonitor:
    """Get global memory monitor instance.
    
    Returns the singleton PlatformMemoryMonitor instance, creating it
    if it doesn't exist. This ensures consistent monitoring across
    the application.
    
    Returns:
        PlatformMemoryMonitor: Global monitor instance
        
    Example:
        >>> monitor = get_memory_monitor()
        >>> snapshot = await monitor.get_memory_pressure()
    """
    global _monitor
    if _monitor is None:
        _monitor = PlatformMemoryMonitor()
    return _monitor


async def test_memory_monitor() -> None:
    """Test the memory monitor functionality.
    
    Comprehensive test function that demonstrates the memory monitor's
    capabilities by taking a snapshot and displaying all relevant metrics
    in a formatted output.
    
    Example:
        >>> await test_memory_monitor()
        ================================================================================
        Platform Memory Monitor Test
        ================================================================================
        
        Platform: darwin
        Total RAM: 16.0GB
        Used RAM: 12.5GB (78.1%)
        Available RAM: 3.5GB
        ...
    """
    monitor = get_memory_monitor()

    print("\n" + "=" * 80)
    print("Platform Memory Monitor Test")
    print("=" * 80 + "\n")

    snapshot = await monitor.get_memory_pressure()

    print(f"Platform: {snapshot.platform}")
    print(f"Total RAM: {snapshot.total_gb:.1f}GB")
    print(f"Used RAM: {snapshot.used_gb:.1f}GB ({snapshot.usage_percent:.1f}%)")
    print(f"Available RAM: {snapshot.available_gb:.1f}GB")
    print()

    if snapshot.platform == "darwin":
        print("macOS Metrics:")
        print(f"  Pressure Level: {snapshot.macos_pressure_level}")
        print(f"  Page Outs: {snapshot.macos_page_outs}")
        print(f"  Is Swapping: {snapshot.macos_is_swapping}")

    elif snapshot.platform == "linux":
        print("Linux Metrics:")
        print(f"  PSI some (avg10): {snapshot.linux_psi_some_avg10}%")
        print(f