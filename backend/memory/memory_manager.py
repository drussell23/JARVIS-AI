"""
M1-Optimized Memory Manager for AI-Powered Chatbot
Provides proactive memory management to prevent crashes on 16GB M1 MacBook
"""

import os
import psutil
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import numpy as np
from enum import Enum
import gc
import objgraph
import tracemalloc
import weakref
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryState(Enum):
    """Memory pressure states"""

    HEALTHY = "healthy"  # < 70% usage
    WARNING = "warning"  # 70-85% usage
    CRITICAL = "critical"  # 85-95% usage
    EMERGENCY = "emergency"  # > 95% usage

class ComponentPriority(Enum):
    """Component priority levels for memory management"""

    CRITICAL = 1  # Core functionality (chatbot)
    HIGH = 2  # Important features (NLP)
    MEDIUM = 3  # Enhanced features (RAG, Voice)
    LOW = 4  # Optional features (Training, Analytics)

@dataclass
class MemorySnapshot:
    """Point-in-time memory status"""

    timestamp: datetime
    total: int
    available: int
    used: int
    percent: float
    state: MemoryState
    components: Dict[str, int] = field(default_factory=dict)

@dataclass
class ComponentInfo:
    """Information about a managed component"""

    name: str
    priority: ComponentPriority
    estimated_memory: int  # bytes
    actual_memory: Optional[int] = None
    is_loaded: bool = False
    last_used: Optional[datetime] = None
    load_time: Optional[float] = None
    reference: Optional[weakref.ref] = None

class MemoryPredictor:
    """AI-driven memory prediction using historical data"""

    def __init__(self, history_size: int = 100):
        self.history: deque = deque(maxlen=history_size)
        self.component_patterns: Dict[str, List[float]] = {}

    def record_usage(self, component: str, memory_mb: float):
        """Record component memory usage"""
        self.history.append(
            {
                "timestamp": datetime.now(),
                "component": component,
                "memory_mb": memory_mb,
            }
        )

        if component not in self.component_patterns:
            self.component_patterns[component] = []
        self.component_patterns[component].append(memory_mb)

    def predict_memory_need(self, component: str) -> float:
        """Predict memory requirement for a component"""
        if component not in self.component_patterns:
            # Return conservative estimate based on component type
            estimates = {
                "nlp_engine": 1500,  # 1.5GB
                "rag_engine": 3000,  # 3GB
                "voice_engine": 2000,  # 2GB
                "domain_knowledge": 1000,  # 1GB
                "simple_chatbot": 100,  # 100MB
            }
            return estimates.get(component, 500)  # Default 500MB

        # Use historical average with 20% buffer
        pattern = self.component_patterns[component]
        if pattern:
            avg = np.mean(pattern)
            std = np.std(pattern) if len(pattern) > 1 else avg * 0.1
            return avg + (std * 1.2)  # 1.2 standard deviations for safety
        return 500  # Default fallback

class M1MemoryManager:
    """
    Proactive memory management system optimized for M1 Macs
    Prevents crashes by monitoring and managing component lifecycle
    """

    def __init__(self):
        # System configuration
        self.total_ram = psutil.virtual_memory().total
        self.safe_threshold = 0.70  # 70% - start optimizing
        self.warning_threshold = 0.85  # 85% - aggressive cleanup
        self.critical_threshold = 0.95  # 95% - emergency mode

        # Component registry
        self.components: Dict[str, ComponentInfo] = {}
        self.loaded_components: Dict[str, Any] = {}

        # Memory tracking
        self.memory_history: deque = deque(maxlen=1000)
        self.predictor = MemoryPredictor()

        # Monitoring
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 2.0  # seconds
        self.is_monitoring = False

        # Callbacks
        self.state_callbacks: List[Callable] = []

        # M1 specific optimizations
        self.is_m1 = self._detect_m1()
        if self.is_m1:
            logger.info("M1 Mac detected - enabling unified memory optimizations")

        # Initialize memory tracking
        tracemalloc.start()

    def _detect_m1(self) -> bool:
        """Detect if running on M1 Mac"""
        import platform

        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def register_component(
        self, name: str, priority: ComponentPriority, estimated_memory_mb: int
    ):
        """Register a component for memory management"""
        self.components[name] = ComponentInfo(
            name=name,
            priority=priority,
            estimated_memory=estimated_memory_mb * 1024 * 1024,  # Convert to bytes
        )
        logger.info(
            f"Registered component: {name} (Priority: {priority.name}, Est. Memory: {estimated_memory_mb}MB)"
        )

    async def register_intelligent_components(self):
        """Register intelligent chatbot components"""
        try:
            # Register intelligent chatbot
            self.register_component(
                "intelligent_chatbot",
                ComponentPriority.MEDIUM,
                estimated_memory_mb=500  # Estimate for intelligent mode
            )
            
            # Register LangChain chatbot
            self.register_component(
                "langchain_chatbot", 
                ComponentPriority.HIGH,
                estimated_memory_mb=2000  # Estimate for LangChain mode
            )
            
            logger.info("Intelligent components registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register intelligent components: {e}")
            return False

    async def start_monitoring(self):
        """Start the memory monitoring loop"""
        # TEMPORARY FIX: Disable monitoring to prevent infinite loop
        logger.info("Memory monitoring disabled to fix infinite loop issue")
        return
        
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Memory monitoring started")

    async def stop_monitoring(self):
        """Stop the memory monitoring loop"""
        if self.is_monitoring and self.monitor_task:
            self.is_monitoring = False
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Memory monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = await self.get_memory_snapshot()
                self.memory_history.append(snapshot)

                # Check state and take action
                await self._handle_memory_state(snapshot)

                # Log status
                if snapshot.state != MemoryState.HEALTHY:
                    logger.warning(
                        f"Memory {snapshot.state.value}: {snapshot.percent:.1f}% used"
                    )

            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")

            await asyncio.sleep(self.monitor_interval)

    async def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory status"""
        mem = psutil.virtual_memory()

        # Determine state based on usage
        percent = mem.percent / 100.0
        
        # Fix: Only trigger WARNING/CRITICAL states if memory usage is actually high
        # The 0.8% indicates very low usage, should be HEALTHY
        if percent < self.safe_threshold:
            state = MemoryState.HEALTHY
        elif percent < self.warning_threshold:
            # Add safeguard: if percent is suspiciously low (< 5%), force HEALTHY
            if mem.percent < 5.0:  # Less than 5% usage should never be WARNING
                state = MemoryState.HEALTHY
            else:
                state = MemoryState.WARNING
        elif percent < self.critical_threshold:
            state = MemoryState.CRITICAL
        else:
            state = MemoryState.EMERGENCY

        # Get component memory usage
        component_memory = {}
        for name, component in self.loaded_components.items():
            try:
                size = self._estimate_object_size(component)
                component_memory[name] = size
                self.components[name].actual_memory = size
            except Exception as e:
                logger.debug(f"Could not measure {name}: {e}")

        return MemorySnapshot(
            timestamp=datetime.now(),
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=percent,
            state=state,
            components=component_memory,
        )

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate memory size of an object"""
        try:
            # Try to get actual size for known types
            if hasattr(obj, "__sizeof__"):
                return obj.__sizeof__()

            # For PyTorch models
            if hasattr(obj, "parameters"):
                total = 0
                for param in obj.parameters():
                    total += param.nelement() * param.element_size()
                return total

            # Fallback to objgraph
            return len(objgraph.show_most_common_types(objects=[obj]))

        except Exception:
            return 0

    async def _handle_memory_state(self, snapshot: MemorySnapshot):
        """Handle different memory states"""
        if snapshot.state == MemoryState.WARNING:
            await self._optimize_memory()
        elif snapshot.state == MemoryState.CRITICAL:
            await self._aggressive_cleanup()
        elif snapshot.state == MemoryState.EMERGENCY:
            await self._emergency_shutdown()

        # Notify callbacks
        for callback in self.state_callbacks:
            try:
                await callback(snapshot)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")

    async def _optimize_memory(self):
        """Optimize memory usage (WARNING state)"""
        # Prevent infinite optimization loop - add cooldown period
        if hasattr(self, '_last_optimize_time'):
            if datetime.now() - self._last_optimize_time < timedelta(minutes=1):
                logger.debug("Skipping optimization - cooldown period active")
                return
        
        self._last_optimize_time = datetime.now()
        logger.info("Optimizing memory usage...")

        # Force garbage collection
        gc.collect()

        # Clear caches
        if hasattr(self, "_clear_caches"):
            await self._clear_caches()

        # Unload low-priority components not used recently
        cutoff_time = datetime.now() - timedelta(minutes=5)
        for name, info in self.components.items():
            if (
                info.is_loaded
                and info.priority == ComponentPriority.LOW
                and info.last_used
                and info.last_used < cutoff_time
            ):
                await self.unload_component(name)

    async def _aggressive_cleanup(self):
        """Aggressive memory cleanup (CRITICAL state)"""
        logger.warning("Aggressive memory cleanup initiated")

        # Unload all LOW and MEDIUM priority components
        for name, info in self.components.items():
            if info.is_loaded and info.priority.value >= ComponentPriority.MEDIUM.value:
                await self.unload_component(name)

        # Force multiple GC passes
        for _ in range(3):
            gc.collect()

    async def _emergency_shutdown(self):
        """Emergency shutdown of non-critical components"""
        logger.error("EMERGENCY: Shutting down non-critical components")

        # Keep only CRITICAL components
        for name, info in self.components.items():
            if info.is_loaded and info.priority != ComponentPriority.CRITICAL:
                await self.unload_component(name, emergency=True)

        # Aggressive garbage collection
        gc.collect(2)  # Full collection

    async def optimize_memory(self, mode: str = "normal"):
        """Public method to optimize memory
        
        Args:
            mode: "normal", "aggressive", or "emergency"
        """
        if mode == "normal":
            await self._optimize_memory()
        elif mode == "aggressive":
            await self._aggressive_cleanup()
        elif mode == "emergency":
            await self._emergency_shutdown()
        else:
            logger.warning(f"Unknown optimization mode: {mode}, using normal")
            await self._optimize_memory()

    async def can_load_component(self, name: str) -> Tuple[bool, Optional[str]]:
        """Check if a component can be safely loaded"""
        if name not in self.components:
            return False, "Component not registered"

        info = self.components[name]
        if info.is_loaded:
            return True, "Already loaded"

        # Predict memory need
        predicted_mb = self.predictor.predict_memory_need(name)
        predicted_bytes = predicted_mb * 1024 * 1024

        # Get current memory
        snapshot = await self.get_memory_snapshot()

        # Check if we have enough memory
        if snapshot.available < predicted_bytes * 1.2:  # 20% buffer
            return (
                False,
                f"Insufficient memory: need {predicted_mb:.1f}MB, have {snapshot.available / 1024 / 1024:.1f}MB",
            )

        # Check if loading would push us over threshold
        new_percent = (snapshot.used + predicted_bytes) / snapshot.total
        if new_percent > self.safe_threshold:
            return False, f"Would exceed safe threshold ({new_percent * 100:.1f}%)"

        return True, None

    async def load_component(self, name: str, component: Any) -> bool:
        """Load a component with memory safety checks"""
        can_load, reason = await self.can_load_component(name)
        if not can_load:
            logger.warning(f"Cannot load {name}: {reason}")
            return False

        try:
            # Record start time
            start_time = datetime.now()

            # Store component
            self.loaded_components[name] = component
            info = self.components[name]
            info.is_loaded = True
            info.last_used = datetime.now()
            info.reference = weakref.ref(component)

            # Measure actual memory usage
            await asyncio.sleep(0.1)  # Let memory settle
            snapshot = await self.get_memory_snapshot()
            actual_mb = snapshot.components.get(name, 0) / 1024 / 1024

            # Record for prediction
            self.predictor.record_usage(name, actual_mb)

            # Calculate load time
            info.load_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Loaded {name}: {actual_mb:.1f}MB in {info.load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            # Cleanup on failure
            if name in self.loaded_components:
                del self.loaded_components[name]
            info.is_loaded = False
            return False

    async def unload_component(self, name: str, emergency: bool = False) -> bool:
        """Unload a component to free memory"""
        if name not in self.components or not self.components[name].is_loaded:
            return False

        try:
            logger.info(f"Unloading component: {name} (emergency={emergency})")

            # Get component
            component = self.loaded_components.get(name)

            # Call cleanup if available
            if component and hasattr(component, "cleanup"):
                try:
                    if asyncio.iscoroutinefunction(component.cleanup):
                        await component.cleanup()
                    else:
                        component.cleanup()
                except Exception as e:
                    if not emergency:
                        logger.error(f"Error during {name} cleanup: {e}")

            # Remove references
            if name in self.loaded_components:
                del self.loaded_components[name]

            # Update info
            info = self.components[name]
            info.is_loaded = False
            info.actual_memory = None
            info.reference = None

            # Force garbage collection
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Failed to unload {name}: {e}")
            return False

    def get_component(self, name: str) -> Optional[Any]:
        """Get a loaded component and update last used time"""
        if name in self.loaded_components:
            self.components[name].last_used = datetime.now()
            return self.loaded_components[name]
        return None

    def add_state_callback(self, callback: Callable):
        """Add a callback for memory state changes"""
        self.state_callbacks.append(callback)

    async def get_memory_report(self) -> Dict[str, Any]:
        """Generate a detailed memory report"""
        snapshot = await self.get_memory_snapshot()

        # Component summary
        components_summary = []
        for name, info in self.components.items():
            components_summary.append(
                {
                    "name": name,
                    "priority": info.priority.name,
                    "is_loaded": info.is_loaded,
                    "estimated_mb": info.estimated_memory / 1024 / 1024,
                    "actual_mb": (
                        info.actual_memory / 1024 / 1024 if info.actual_memory else None
                    ),
                    "last_used": info.last_used.isoformat() if info.last_used else None,
                    "load_time": info.load_time,
                }
            )

        # History analysis
        if self.memory_history:
            history_data = [h.percent * 100 for h in list(self.memory_history)[-50:]]
            avg_usage = np.mean(history_data)
            max_usage = np.max(history_data)
            trend = "increasing" if history_data[-1] > history_data[0] else "decreasing"
        else:
            avg_usage = max_usage = snapshot.percent * 100
            trend = "stable"

        return {
            "current_state": {
                "state": snapshot.state.value,
                "percent_used": snapshot.percent * 100,
                "available_mb": snapshot.available / 1024 / 1024,
                "used_mb": snapshot.used / 1024 / 1024,
                "total_mb": snapshot.total / 1024 / 1024,
            },
            "components": components_summary,
            "analysis": {
                "average_usage": avg_usage,
                "max_usage": max_usage,
                "trend": trend,
                "is_m1": self.is_m1,
            },
            "thresholds": {
                "safe": self.safe_threshold * 100,
                "warning": self.warning_threshold * 100,
                "critical": self.critical_threshold * 100,
            },
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_monitoring()
        tracemalloc.stop()

# Example usage and testing
if __name__ == "__main__":

    async def test_memory_manager():
        # Create manager
        manager = M1MemoryManager()

        # Register components
        manager.register_component("simple_chatbot", ComponentPriority.CRITICAL, 100)
        manager.register_component("nlp_engine", ComponentPriority.HIGH, 1500)
        manager.register_component("rag_engine", ComponentPriority.MEDIUM, 3000)
        manager.register_component("voice_engine", ComponentPriority.MEDIUM, 2000)

        # Start monitoring
        await manager.start_monitoring()

        # Get initial report
        report = await manager.get_memory_report()
        print(f"Initial memory state: {report['current_state']['state']}")
        print(f"Memory usage: {report['current_state']['percent_used']:.1f}%")

        # Simulate loading a component
        class MockComponent:
            def __init__(self, size_mb):
                self.data = bytearray(size_mb * 1024 * 1024)

        # Try to load NLP engine
        mock_nlp = MockComponent(100)  # Small mock
        success = await manager.load_component("nlp_engine", mock_nlp)
        print(f"NLP Engine loaded: {success}")

        # Wait a bit and check report
        await asyncio.sleep(5)
        report = await manager.get_memory_report()
        print(f"\nMemory Report:")
        print(json.dumps(report, indent=2, default=str))

        # Cleanup
        await manager.cleanup()

    # Run test
    asyncio.run(test_memory_manager())

