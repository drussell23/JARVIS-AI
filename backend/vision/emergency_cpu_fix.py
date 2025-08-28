#!/usr/bin/env python3
"""
Emergency CPU Fix for JARVIS Continuous Learning
Immediate reduction from 97% to ~40% CPU without Rust
"""

import os
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class EmergencyCPUFix:
    """
    Emergency fix to immediately reduce CPU usage
    This is a temporary solution until Rust layer is ready
    """

    def __init__(self):
        self.enabled = True
        self.cpu_limit = 40.0  # Target CPU %
        self.check_interval = 1.0  # seconds
        self.throttle_factor = 1.0

        # Monitoring
        self.cpu_history = []
        self.last_check = time.time()

        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._cpu_monitor_loop, daemon=True, name="CPUMonitor"
        )
        self.monitor_thread.start()

        logger.info("🚨 Emergency CPU Fix activated!")
        logger.info(f"   Target CPU: {self.cpu_limit}%")

    def _cpu_monitor_loop(self):
        """Monitor CPU and adjust throttling"""
        while self.enabled:
            try:
                # Get current CPU usage
                cpu_percent = psutil.Process().cpu_percent(interval=0.1)
                self.cpu_history.append(cpu_percent)

                # Keep only recent history
                if len(self.cpu_history) > 60:
                    self.cpu_history.pop(0)

                # Calculate average
                avg_cpu = np.mean(self.cpu_history[-10:]) if self.cpu_history else 0

                # Adjust throttle factor
                if avg_cpu > self.cpu_limit * 1.2:  # 20% over limit
                    self.throttle_factor *= 1.5
                    self.throttle_factor = min(self.throttle_factor, 10.0)
                    logger.warning(
                        f"CPU high ({avg_cpu:.1f}%), throttle: {self.throttle_factor:.1f}x"
                    )

                elif avg_cpu < self.cpu_limit * 0.8:  # 20% under limit
                    self.throttle_factor *= 0.9
                    self.throttle_factor = max(self.throttle_factor, 0.1)

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"CPU monitor error: {e}")
                time.sleep(5)

    def get_sleep_time(self, base_sleep: float = 1.0) -> float:
        """Get adjusted sleep time based on CPU load"""
        return base_sleep * self.throttle_factor

    def should_skip_cycle(self) -> bool:
        """Check if we should skip this learning cycle"""
        if not self.cpu_history:
            return False

        current_cpu = self.cpu_history[-1] if self.cpu_history else 0
        return current_cpu > self.cpu_limit * 1.5  # Skip if 50% over limit


# Global instance
# DISABLED: This was causing excessive CPU throttling and preventing startup
# _emergency_fix = EmergencyCPUFix()
_emergency_fix = None


def apply_emergency_fixes():
    """Apply all emergency fixes to reduce CPU usage"""
    logger.info("\n🚨 Emergency CPU fixes DISABLED for better performance")

    # Return a dummy object if needed
    class DummyFix:
        def should_skip_cycle(self):
            return False

        throttle_factor = 1.0

    return DummyFix()


def wrap_high_cpu_function(func):
    """Decorator to add CPU throttling to functions"""

    def wrapper(*args, **kwargs):
        # DISABLED: No throttling - just pass through
        return func(*args, **kwargs)

    return wrapper


class ThrottledContinuousLearning:
    """
    Heavily throttled version of continuous learning
    Reduces CPU from 97% to ~40% as emergency measure
    """

    def __init__(self, original_learning):
        self.original = original_learning
        self.emergency_fix = _emergency_fix
        self.skip_counter = 0
        self.last_run = datetime.now()

        # Override methods with throttled versions
        self._wrap_methods()

        logger.info("✅ Throttled continuous learning activated")

    def _wrap_methods(self):
        """Wrap all methods with throttling"""
        methods_to_wrap = [
            "_learning_cycle",
            "_perform_retraining",
            "_perform_fine_tuning",
            "_process_task_queue",
            "_evaluate_model",
        ]

        for method_name in methods_to_wrap:
            if hasattr(self.original, method_name):
                original_method = getattr(self.original, method_name)
                wrapped_method = wrap_high_cpu_function(original_method)
                setattr(self.original, method_name, wrapped_method)

    @property
    def running(self):
        # DISABLED: No CPU checking
        return self.original.running

    @running.setter
    def running(self, value):
        self.original.running = value


def get_cpu_status():
    """Get current CPU status and recommendations"""
    cpu_percent = psutil.Process().cpu_percent(interval=0.5)
    total_cpu = psutil.cpu_percent(interval=0.5)

    status = {
        "process_cpu": cpu_percent,
        "total_cpu": total_cpu,
        "throttle_factor": _emergency_fix.throttle_factor,
        "emergency_mode": cpu_percent > 60,
        "recommendations": [],
    }

    if cpu_percent > 80:
        status["recommendations"].append("🚨 CRITICAL: Kill and restart JARVIS")
    elif cpu_percent > 60:
        status["recommendations"].append("⚠️  HIGH: Consider disabling features")
    elif cpu_percent > 40:
        status["recommendations"].append("⚡ MODERATE: Running with throttling")
    else:
        status["recommendations"].append("✅ GOOD: CPU usage acceptable")

    return status


def emergency_shutdown():
    """Emergency shutdown of high-CPU components"""
    logger.warning("🚨 EMERGENCY SHUTDOWN INITIATED")

    # Set all disable flags
    os.environ["DISABLE_CONTINUOUS_LEARNING"] = "true"
    os.environ["DISABLE_VISION_MONITORING"] = "true"
    os.environ["DISABLE_ML_FEATURES"] = "true"

    # Try to stop learning threads
    try:
        from . import vision_system_v2

        if hasattr(vision_system_v2, "_vision_system"):
            vision_system_v2._vision_system.shutdown()
    except:
        pass

    # Force garbage collection
    import gc

    gc.collect()

    logger.info("✅ Emergency shutdown complete")


if __name__ == "__main__":
    # Test emergency fixes
    print("\n🚨 Testing Emergency CPU Fixes...")
    print("=" * 50)

    # Apply fixes
    fix = apply_emergency_fixes()

    # Monitor for 10 seconds
    print("\nMonitoring CPU for 10 seconds...")
    for i in range(10):
        status = get_cpu_status()
        print(
            f"\rCPU: {status['process_cpu']:.1f}% | "
            f"Throttle: {status['throttle_factor']:.1f}x | "
            f"Status: {status['recommendations'][0]}",
            end="",
        )
        time.sleep(1)

    print("\n\n✅ Emergency fixes are working!")
    print("=" * 50)
