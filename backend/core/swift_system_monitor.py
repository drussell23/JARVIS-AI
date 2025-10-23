"""
Swift System Monitor - High-performance system monitoring
Provides Python interface to Swift-accelerated system monitoring with fallback
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Swift performance bridge
SWIFT_AVAILABLE = False
try:
    from backend.swift_bridge.performance_bridge import (
        get_system_monitor,
        SWIFT_PERFORMANCE_AVAILABLE,
        SystemMetrics as SwiftSystemMetrics
    )
    SWIFT_AVAILABLE = SWIFT_PERFORMANCE_AVAILABLE
    logger.info("✅ Swift performance bridge available")
except Exception as e:
    logger.warning(f"Swift performance bridge not available, using Python fallback: {e}")

@dataclass
class SystemMetrics:
    """System metrics (unified interface)"""
    cpu_usage_percent: float
    memory_used_mb: int
    memory_available_mb: int
    memory_total_mb: int
    memory_pressure: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class SwiftSystemMonitorWrapper:
    """
    High-performance system monitor with Swift acceleration
    Falls back to Python psutil if Swift is unavailable
    """

    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self._swift_monitor = None
        self._monitoring = False
        self._monitor_task = None
        self._callbacks: List[Callable[[SystemMetrics], None]] = []
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 100

        # Try to initialize Swift monitor
        if SWIFT_AVAILABLE:
            try:
                self._swift_monitor = get_system_monitor()
                if self._swift_monitor:
                    logger.info("✅ Swift system monitor initialized")
                else:
                    logger.warning("Swift monitor creation failed, using Python fallback")
            except Exception as e:
                logger.warning(f"Failed to create Swift monitor: {e}, using Python fallback")
        else:
            logger.info("Using Python fallback for system monitoring")

    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        if self._swift_monitor:
            try:
                # Use Swift for high-performance metrics
                swift_metrics = self._swift_monitor.get_metrics()
                return SystemMetrics(
                    cpu_usage_percent=swift_metrics.cpu_usage_percent,
                    memory_used_mb=swift_metrics.memory_used_mb,
                    memory_available_mb=swift_metrics.memory_available_mb,
                    memory_total_mb=swift_metrics.memory_total_mb,
                    memory_pressure=swift_metrics.memory_pressure,
                    timestamp=swift_metrics.timestamp
                )
            except Exception as e:
                logger.error(f"Swift metrics failed: {e}, falling back to Python")

        # Python fallback using psutil
        return self._get_metrics_python()

    def _get_metrics_python(self) -> SystemMetrics:
        """Get metrics using Python psutil (fallback)"""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        # Get memory pressure (macOS specific)
        memory_pressure = self._get_memory_pressure()

        return SystemMetrics(
            cpu_usage_percent=cpu,
            memory_used_mb=int(mem.used / (1024 ** 2)),
            memory_available_mb=int(mem.available / (1024 ** 2)),
            memory_total_mb=int(mem.total / (1024 ** 2)),
            memory_pressure=memory_pressure,
            timestamp=time.time()
        )

    def _get_memory_pressure(self) -> str:
        """Get system memory pressure (macOS specific)"""
        try:
            import subprocess
            result = subprocess.run(
                ['memory_pressure'],
                capture_output=True,
                text=True,
                timeout=1
            )
            output = result.stdout.lower()

            if 'critical' in output:
                return 'critical'
            elif 'warn' in output:
                return 'warn'
            elif 'normal' in output:
                return 'normal'
            else:
                return 'unknown'
        except:
            # Fallback based on memory percentage
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                return 'critical'
            elif mem.percent > 70:
                return 'warn'
            else:
                return 'normal'

    async def start_monitoring(self, callback: Optional[Callable[[SystemMetrics], None]] = None):
        """Start continuous monitoring"""
        if callback:
            self._callbacks.append(callback)

        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"System monitoring started (interval: {self.update_interval}s)")

    async def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")

    async def _monitor_loop(self):
        """Internal monitoring loop"""
        while self._monitoring:
            try:
                metrics = self.get_metrics()

                # Store in history
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history.pop(0)

                # Call callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(metrics)
                        else:
                            callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(self.update_interval)

    def get_history(self, limit: Optional[int] = None) -> List[SystemMetrics]:
        """Get metrics history"""
        if limit:
            return self._metrics_history[-limit:]
        return self._metrics_history.copy()

    def get_average_metrics(self, seconds: int = 60) -> Optional[SystemMetrics]:
        """Get average metrics over time period"""
        if not self._metrics_history:
            return None

        cutoff_time = time.time() - seconds
        recent = [m for m in self._metrics_history if m.timestamp >= cutoff_time]

        if not recent:
            return None

        return SystemMetrics(
            cpu_usage_percent=sum(m.cpu_usage_percent for m in recent) / len(recent),
            memory_used_mb=int(sum(m.memory_used_mb for m in recent) / len(recent)),
            memory_available_mb=int(sum(m.memory_available_mb for m in recent) / len(recent)),
            memory_total_mb=recent[0].memory_total_mb,
            memory_pressure=recent[-1].memory_pressure,
            timestamp=time.time()
        )

    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        metrics = self.get_metrics()
        return (
            metrics.cpu_usage_percent < 80 and
            metrics.memory_pressure in ['normal', 'warn']
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        metrics = self.get_metrics()
        avg_metrics = self.get_average_metrics(60)

        return {
            'acceleration': 'swift' if self._swift_monitor else 'python',
            'monitoring': self._monitoring,
            'current': metrics.to_dict(),
            'average_60s': avg_metrics.to_dict() if avg_metrics else None,
            'healthy': self.is_healthy(),
            'history_size': len(self._metrics_history)
        }

# Singleton instance
_system_monitor_instance = None

def get_swift_system_monitor(update_interval: float = 5.0) -> SwiftSystemMonitorWrapper:
    """Get singleton system monitor instance"""
    global _system_monitor_instance

    if _system_monitor_instance is None:
        _system_monitor_instance = SwiftSystemMonitorWrapper(update_interval)

    return _system_monitor_instance

if __name__ == "__main__":
    # Test system monitor
    logging.basicConfig(level=logging.INFO)

    async def test():
        monitor = get_swift_system_monitor()

        print("🔍 Swift System Monitor Test")
        print("=" * 50)

        # Get current metrics
        metrics = monitor.get_metrics()
        print(f"\n📊 Current Metrics:")
        print(f"  CPU: {metrics.cpu_usage_percent:.1f}%")
        print(f"  Memory: {metrics.memory_used_mb}MB / {metrics.memory_total_mb}MB")
        print(f"  Available: {metrics.memory_available_mb}MB")
        print(f"  Pressure: {metrics.memory_pressure}")

        # Test monitoring
        print(f"\n⏱️  Testing monitoring (5 samples)...")
        samples = []

        def collect_sample(m: SystemMetrics):
            samples.append(m)
            print(f"  Sample {len(samples)}: CPU={m.cpu_usage_percent:.1f}% Memory={m.memory_pressure}")

        await monitor.start_monitoring(collect_sample)
        await asyncio.sleep(6)  # Collect samples
        await monitor.stop_monitoring()

        # Show status
        status = monitor.get_status()
        print(f"\n🏥 System Health:")
        print(f"  Acceleration: {status['acceleration']}")
        print(f"  Healthy: {status['healthy']}")

    asyncio.run(test())
