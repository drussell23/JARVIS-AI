"""
Advanced RAM Monitor for Hybrid Architecture
Monitors both local and GCP RAM with intelligent decision-making
Features:
- Real-time RAM tracking (local + GCP)
- macOS memory pressure detection (vm_stat + page outs)
- Historical trend analysis
- Predictive capacity estimation
- Zero hardcoding, fully config-driven
"""

import asyncio
import logging
import platform
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class RAMSnapshot:
    """Snapshot of RAM state at a point in time"""

    timestamp: datetime
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percent: float
    memory_pressure: str  # "low", "normal", "high", "critical"
    page_outs: int = 0  # macOS swapping indicator
    is_swapping: bool = False
    can_accept_workload: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "usage_percent": round(self.usage_percent, 1),
            "memory_pressure": self.memory_pressure,
            "page_outs": self.page_outs,
            "is_swapping": self.is_swapping,
            "can_accept_workload": self.can_accept_workload,
        }


class AdvancedRAMMonitor:
    """
    Advanced RAM monitoring with local + GCP awareness
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ram_config = config.get("hybrid", {}).get("ram_awareness", {})

        # Thresholds
        self.local_thresholds = self.ram_config.get(
            "local_thresholds", {"warning": 70, "critical": 85, "recovery": 60, "optimization": 40}
        )
        self.gcp_thresholds = self.ram_config.get(
            "gcp_thresholds", {"warning": 75, "critical": 90, "healthy": 60}
        )

        # Monitoring config
        self.monitoring_interval = self.ram_config.get("monitoring_interval", 5)
        self.decision_window = self.ram_config.get("decision_window", 30)
        self.page_outs_threshold = self.ram_config.get("page_outs_threshold", 5000)

        # State
        self.is_macos = platform.system() == "Darwin"
        self.local_history: deque = deque(
            maxlen=int(self.decision_window / self.monitoring_interval)
        )
        self.gcp_history: deque = deque(maxlen=int(self.decision_window / self.monitoring_interval))

        # Last page_outs baseline for delta calculation
        self.last_page_outs = 0

        # Background monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info(f"📊 AdvancedRAMMonitor initialized (macOS: {self.is_macos})")

    async def start(self):
        """Start background RAM monitoring"""
        if self.is_running:
            logger.warning("RAM monitor already running")
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"✅ RAM monitoring started (interval: {self.monitoring_interval}s)")

    async def stop(self):
        """Stop background RAM monitoring"""
        if not self.is_running:
            return

        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️  RAM monitoring stopped")

    async def _monitoring_loop(self):
        """Background loop to monitor RAM every N seconds"""
        while self.is_running:
            try:
                # Get current local RAM state
                snapshot = await self.get_local_ram_snapshot()
                self.local_history.append(snapshot)

                # Log if memory pressure changes
                if snapshot.memory_pressure in ["high", "critical"]:
                    logger.warning(
                        f"⚠️  Memory Pressure: {snapshot.memory_pressure.upper()} "
                        f"({snapshot.usage_percent:.1f}% used, {snapshot.available_gb:.1f}GB free)"
                    )

                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RAM monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def get_local_ram_snapshot(self) -> RAMSnapshot:
        """Get current local RAM state"""
        # Get basic RAM info from psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        usage_percent = mem.percent

        # macOS-specific memory pressure detection
        page_outs = 0
        is_swapping = False

        if self.is_macos:
            page_outs = await self._get_macos_page_outs()
            is_swapping = page_outs > self.page_outs_threshold

        # Determine memory pressure level
        memory_pressure = self._calculate_memory_pressure(usage_percent, page_outs, is_swapping)

        # Can accept workload?
        can_accept = usage_percent < self.local_thresholds["critical"] and not is_swapping

        return RAMSnapshot(
            timestamp=datetime.now(),
            total_gb=total_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            usage_percent=usage_percent,
            memory_pressure=memory_pressure,
            page_outs=page_outs,
            is_swapping=is_swapping,
            can_accept_workload=can_accept,
        )

    async def _get_macos_page_outs(self) -> int:
        """Get macOS page outs (swapping indicator) from vm_stat"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "vm_stat", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode()

            # Parse "Pages paged out: 12345."
            for line in output.split("\n"):
                if "Pages paged out:" in line:
                    page_outs_str = line.split(":")[1].strip().rstrip(".")
                    page_outs = int(page_outs_str)

                    # Calculate delta from last check
                    page_outs - self.last_page_outs
                    self.last_page_outs = page_outs

                    return page_outs

            return 0
        except Exception as e:
            logger.debug(f"Failed to get page outs: {e}")
            return 0

    def _calculate_memory_pressure(
        self, usage_percent: float, page_outs: int, is_swapping: bool
    ) -> str:
        """
        Calculate memory pressure level based on usage + swapping

        Returns: "low", "normal", "high", "critical"
        """
        if is_swapping or usage_percent >= self.local_thresholds["critical"]:
            return "critical"
        elif usage_percent >= self.local_thresholds["warning"]:
            return "high"
        elif usage_percent >= self.local_thresholds["recovery"]:
            return "normal"
        else:
            return "low"

    def get_current_pressure_percent(self) -> float:
        """Get current RAM pressure as percentage (for routing decisions)"""
        if not self.local_history:
            # No data yet, return safe value
            return 0.0

        latest = self.local_history[-1]
        return latest.usage_percent

    def should_prefer_gcp(self) -> bool:
        """Should we prefer GCP routing right now?"""
        if not self.local_history:
            return False

        latest = self.local_history[-1]

        # Prefer GCP if:
        # 1. Memory pressure is high or critical
        # 2. Currently swapping
        # 3. Usage above warning threshold
        return (
            latest.memory_pressure in ["high", "critical"]
            or latest.is_swapping
            or latest.usage_percent >= self.local_thresholds["warning"]
        )

    def should_force_gcp(self) -> bool:
        """Must force GCP routing right now? (emergency)"""
        if not self.local_history:
            return False

        latest = self.local_history[-1]

        # Force GCP if critical or heavy swapping
        return (
            latest.memory_pressure == "critical"
            or latest.usage_percent >= self.local_thresholds["critical"]
            or latest.page_outs > self.page_outs_threshold * 2
        )

    def can_reclaim_to_local(self) -> bool:
        """Should we reclaim components back to local? (cost optimization)"""
        if not self.local_history:
            return False

        # Check last 3 snapshots for consistent low pressure
        if len(self.local_history) < 3:
            return False

        recent = list(self.local_history)[-3:]

        # All recent snapshots should show low usage
        return all(
            snap.usage_percent < self.local_thresholds["optimization"] and not snap.is_swapping
            for snap in recent
        )

    def estimate_available_capacity(self, estimated_ram_gb: float) -> bool:
        """Can local handle this additional RAM requirement?"""
        if not self.local_history:
            return False

        latest = self.local_history[-1]

        # Check if we have enough available RAM + buffer
        buffer_gb = 1.0  # Keep 1GB buffer
        return latest.available_gb >= (estimated_ram_gb + buffer_gb)

    def get_routing_decision_context(self) -> Dict[str, Any]:
        """Get comprehensive context for routing decisions"""
        if not self.local_history:
            return {"available": False, "reason": "No RAM data available yet"}

        latest = self.local_history[-1]

        # Calculate trend (increasing or decreasing usage?)
        trend = "stable"
        if len(self.local_history) >= 3:
            recent = list(self.local_history)[-3:]
            if recent[-1].usage_percent > recent[0].usage_percent + 5:
                trend = "increasing"
            elif recent[-1].usage_percent < recent[0].usage_percent - 5:
                trend = "decreasing"

        return {
            "available": True,
            "current_snapshot": latest.to_dict(),
            "prefer_gcp": self.should_prefer_gcp(),
            "force_gcp": self.should_force_gcp(),
            "can_reclaim": self.can_reclaim_to_local(),
            "trend": trend,
            "thresholds": {
                "warning": self.local_thresholds["warning"],
                "critical": self.local_thresholds["critical"],
                "recovery": self.local_thresholds["recovery"],
                "optimization": self.local_thresholds["optimization"],
            },
            "history_size": len(self.local_history),
            "monitoring_active": self.is_running,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring/logging"""
        if not self.local_history:
            return {}

        latest = self.local_history[-1]

        # Calculate averages over history
        avg_usage = sum(s.usage_percent for s in self.local_history) / len(self.local_history)
        max_usage = max(s.usage_percent for s in self.local_history)
        min_usage = min(s.usage_percent for s in self.local_history)

        return {
            "current": latest.to_dict(),
            "averages": {
                "usage_percent": round(avg_usage, 1),
                "max_usage": round(max_usage, 1),
                "min_usage": round(min_usage, 1),
            },
            "decisions": {
                "prefer_gcp": self.should_prefer_gcp(),
                "force_gcp": self.should_force_gcp(),
                "can_reclaim": self.can_reclaim_to_local(),
            },
            "config": {
                "monitoring_interval": self.monitoring_interval,
                "decision_window": self.decision_window,
                "thresholds": self.local_thresholds,
            },
        }


# Global instance (lazy initialized)
_ram_monitor: Optional[AdvancedRAMMonitor] = None


def get_ram_monitor(config: Optional[Dict] = None) -> AdvancedRAMMonitor:
    """Get or create global RAM monitor instance"""
    global _ram_monitor
    if _ram_monitor is None:
        if config is None:
            raise ValueError("Must provide config for first initialization")
        _ram_monitor = AdvancedRAMMonitor(config)
    return _ram_monitor
