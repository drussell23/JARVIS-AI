"""
M1-Optimized Memory Manager for AI-Powered Chatbot

This module provides proactive memory management to prevent crashes on 16GB M1 MacBook.
It includes intelligent component lifecycle management, memory prediction, and automated
cleanup strategies optimized for Apple Silicon unified memory architecture.

The memory manager monitors system memory usage in real-time and takes proactive actions
to prevent out-of-memory conditions by unloading low-priority components and optimizing
memory allocation patterns.

Example:
    >>> manager = M1MemoryManager()
    >>> manager.register_component("nlp_engine", ComponentPriority.HIGH, 1500)
    >>> await manager.start_monitoring()
    >>> success = await manager.load_component("nlp_engine", nlp_instance)
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
    """Memory pressure states for system monitoring.
    
    Attributes:
        HEALTHY: Less than 70% memory usage - normal operation
        WARNING: 70-85% memory usage - start optimization
        CRITICAL: 85-95% memory usage - aggressive cleanup needed
        EMERGENCY: Greater than 95% memory usage - emergency shutdown
    """

    HEALTHY = "healthy"  # < 70% usage
    WARNING = "warning"  # 70-85% usage
    CRITICAL = "critical"  # 85-95% usage
    EMERGENCY = "emergency"  # > 95% usage

class ComponentPriority(Enum):
    """Component priority levels for memory management decisions.
    
    Lower numeric values indicate higher priority. Critical components
    are preserved during memory pressure situations.
    
    Attributes:
        CRITICAL: Core functionality that must remain loaded (chatbot core)
        HIGH: Important features that should be preserved (NLP processing)
        MEDIUM: Enhanced features that can be unloaded (RAG, Voice)
        LOW: Optional features that are first to be unloaded (Training, Analytics)
    """

    CRITICAL = 1  # Core functionality (chatbot)
    HIGH = 2  # Important features (NLP)
    MEDIUM = 3  # Enhanced features (RAG, Voice)
    LOW = 4  # Optional features (Training, Analytics)

@dataclass
class MemorySnapshot:
    """Point-in-time memory status snapshot.
    
    Captures comprehensive memory state including system metrics
    and per-component memory usage for analysis and decision making.
    
    Attributes:
        timestamp: When this snapshot was taken
        total: Total system memory in bytes
        available: Available memory in bytes
        used: Used memory in bytes
        percent: Memory usage as fraction (0.0-1.0)
        state: Current memory pressure state
        components: Per-component memory usage in bytes
    """

    timestamp: datetime
    total: int
    available: int
    used: int
    percent: float
    state: MemoryState
    components: Dict[str, int] = field(default_factory=dict)

@dataclass
class ComponentInfo:
    """Information about a managed component.
    
    Tracks metadata and state for each registered component
    to enable intelligent loading/unloading decisions.
    
    Attributes:
        name: Unique component identifier
        priority: Priority level for memory management
        estimated_memory: Estimated memory usage in bytes
        actual_memory: Measured memory usage in bytes (if loaded)
        is_loaded: Whether component is currently loaded
        last_used: Timestamp of last access
        load_time: Time taken to load component in seconds
        reference: Weak reference to component object
    """

    name: str
    priority: ComponentPriority
    estimated_memory: int  # bytes
    actual_memory: Optional[int] = None
    is_loaded: bool = False
    last_used: Optional[datetime] = None
    load_time: Optional[float] = None
    reference: Optional[weakref.ref] = None

class MemoryPredictor:
    """AI-driven memory prediction using historical data.
    
    Analyzes historical memory usage patterns to predict future
    memory requirements for components, enabling proactive management.
    
    Attributes:
        history: Deque storing recent memory usage records
        component_patterns: Per-component memory usage patterns
    """

    def __init__(self, history_size: int = 100):
        """Initialize the memory predictor.
        
        Args:
            history_size: Maximum number of historical records to maintain
        """
        self.history: deque = deque(maxlen=history_size)
        self.component_patterns: Dict[str, List[float]] = {}

    def record_usage(self, component: str, memory_mb: float) -> None:
        """Record component memory usage for pattern analysis.
        
        Args:
            component: Component name
            memory_mb: Memory usage in megabytes
        """
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
        """Predict memory requirement for a component.
        
        Uses historical patterns with statistical analysis to predict
        future memory needs, including safety buffers.
        
        Args:
            component: Component name to predict for
            
        Returns:
            Predicted memory requirement in megabytes
            
        Example:
            >>> predictor = MemoryPredictor()
            >>> predictor.record_usage("nlp_engine", 1200.0)
            >>> predicted = predictor.predict_memory_need("nlp_engine")
            >>> print(f"Predicted: {predicted}MB")
        """
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
    """Proactive memory management system optimized for M1 Macs.
    
    Prevents crashes by monitoring system memory and managing component
    lifecycle based on priority and usage patterns. Includes M1-specific
    optimizations for unified memory architecture.
    
    The manager operates in multiple phases:
    1. HEALTHY: Normal operation, all components available
    2. WARNING: Start optimization, unload unused low-priority components
    3. CRITICAL: Aggressive cleanup, unload medium-priority components
    4. EMERGENCY: Keep only critical components loaded
    
    Attributes:
        total_ram: Total system RAM in bytes
        safe_threshold: Memory usage threshold to start optimization (0.70)
        warning_threshold: Memory usage threshold for aggressive cleanup (0.85)
        critical_threshold: Memory usage threshold for emergency mode (0.95)
        components: Registry of managed components
        loaded_components: Currently loaded component instances
        memory_history: Historical memory usage data
        predictor: Memory usage prediction engine
        monitor_task: Background monitoring task
        monitor_interval: Monitoring frequency in seconds
        is_monitoring: Whether monitoring is active
        state_callbacks: Callbacks for memory state changes
        is_m1: Whether running on M1 Mac
    """

    def __init__(self):
        """Initialize the M1 Memory Manager.
        
        Sets up monitoring thresholds, component registry, and M1-specific
        optimizations. Starts memory tracking but not active monitoring.
        """
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
        """Detect if running on M1 Mac.
        
        Returns:
            True if running on Apple Silicon (M1/M2), False otherwise
        """
        import platform

        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def register_component(
        self, name: str, priority: ComponentPriority, estimated_memory_mb: int
    ) -> None:
        """Register a component for memory management.
        
        Components must be registered before they can be loaded and managed.
        Registration includes priority level and estimated memory usage.
        
        Args:
            name: Unique component identifier
            priority: Priority level for memory management decisions
            estimated_memory_mb: Estimated memory usage in megabytes
            
        Example:
            >>> manager = M1MemoryManager()
            >>> manager.register_component("nlp_engine", ComponentPriority.HIGH, 1500)
        """
        self.components[name] = ComponentInfo(
            name=name,
            priority=priority,
            estimated_memory=estimated_memory_mb * 1024 * 1024,  # Convert to bytes
        )
        logger.info(
            f"Registered component: {name} (Priority: {priority.name}, Est. Memory: {estimated_memory_mb}MB)"
        )

    async def register_intelligent_components(self) -> bool:
        """Register intelligent chatbot components with appropriate priorities.
        
        Pre-registers common intelligent chatbot components with estimated
        memory usage based on typical model sizes and framework overhead.
        
        Returns:
            True if registration successful, False otherwise
            
        Raises:
            Exception: If component registration fails
        """
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

    async def start_monitoring(self) -> None:
        """Start the memory monitoring loop.
        
        Begins continuous monitoring of system memory usage and automatic
        management actions. Currently disabled to prevent infinite loops.
        
        Note:
            Monitoring is temporarily disabled due to infinite loop issues.
            This will be re-enabled once the loop condition is fixed.
        """
        # TEMPORARY FIX: Disable monitoring to prevent infinite loop
        logger.info("Memory monitoring disabled to fix infinite loop issue")
        return
        
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the memory monitoring loop.
        
        Gracefully stops the background monitoring task and cleans up resources.
        """
        if self.is_monitoring and self.monitor_task:
            self.is_monitoring = False
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Memory monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop for continuous memory management.
        
        Runs continuously while monitoring is enabled, taking snapshots
        and triggering appropriate management actions based on memory state.
        
        Raises:
            Exception: Logs errors but continues monitoring
        """
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
        """Get current memory status snapshot.
        
        Captures comprehensive memory state including system metrics
        and per-component usage for analysis and decision making.
        
        Returns:
            MemorySnapshot containing current memory state
            
        Example:
            >>> manager = M1MemoryManager()
            >>> snapshot = await manager.get_memory_snapshot()
            >>> print(f"Memory usage: {snapshot.percent:.1f}%")
        """
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
        """Estimate memory size of an object.
        
        Uses multiple strategies to estimate object memory usage,
        including built-in methods and framework-specific approaches.
        
        Args:
            obj: Object to measure
            
        Returns:
            Estimated memory usage in bytes
        """
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

    async def _handle_memory_state(self, snapshot: MemorySnapshot) -> None:
        """Handle different memory states with appropriate actions.
        
        Triggers different levels of memory management based on current
        memory pressure state, from gentle optimization to emergency shutdown.
        
        Args:
            snapshot: Current memory snapshot
            
        Raises:
            Exception: Logs callback errors but continues processing
        """
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

    async def _optimize_memory(self) -> None:
        """Optimize memory usage during WARNING state.
        
        Performs gentle memory optimization including garbage collection,
        cache clearing, and unloading of unused low-priority components.
        Includes cooldown period to prevent excessive optimization.
        """
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

    async def _aggressive_cleanup(self) -> None:
        """Aggressive memory cleanup during CRITICAL state.
        
        Unloads all LOW and MEDIUM priority components and performs
        multiple garbage collection passes to free maximum memory.
        """
        logger.warning("Aggressive memory cleanup initiated")

        # Unload all LOW and MEDIUM priority components
        for name, info in self.components.items():
            if info.is_loaded and info.priority.value >= ComponentPriority.MEDIUM.value:
                await self.unload_component(name)

        # Force multiple GC passes
        for _ in range(3):
            gc.collect()

    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown of non-critical components.
        
        Keeps only CRITICAL priority components loaded and performs
        aggressive garbage collection to prevent system crash.
        """
        logger.error("EMERGENCY: Shutting down non-critical components")

        # Keep only CRITICAL components
        for name, info in self.components.items():
            if info.is_loaded and info.priority != ComponentPriority.CRITICAL:
                await self.unload_component(name, emergency=True)

        # Aggressive garbage collection
        gc.collect(2)  # Full collection

    async def optimize_memory(self, mode: str = "normal") -> None:
        """Public method to manually optimize memory.
        
        Allows external code to trigger memory optimization at different
        intensity levels based on current needs.
        
        Args:
            mode: Optimization mode - "normal", "aggressive", or "emergency"
            
        Example:
            >>> await manager.optimize_memory("aggressive")
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
        """Check if a component can be safely loaded.
        
        Analyzes current memory state and predicted component requirements
        to determine if loading would be safe without triggering memory pressure.
        
        Args:
            name: Component name to check
            
        Returns:
            Tuple of (can_load: bool, reason: Optional[str])
            If can_load is False, reason explains why
            
        Example:
            >>> can_load, reason = await manager.can_load_component("nlp_engine")
            >>> if not can_load:
            ...     print(f"Cannot load: {reason}")
        """
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
        """Load a component with memory safety checks.
        
        Performs safety checks before loading and tracks actual memory usage
        for future predictions. Updates component metadata and records timing.
        
        Args:
            name: Component name (must be registered)
            component: Component instance to load
            
        Returns:
            True if loaded successfully, False otherwise
            
        Raises:
            Exception: Logs loading errors but returns False
            
        Example:
            >>> nlp_engine = NLPEngine()
            >>> success = await manager.load_component("nlp_engine", nlp_engine)
            >>> if success:
            ...     print("NLP engine loaded successfully")
        """
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
        """Unload a component to free memory.
        
        Safely unloads a component by calling cleanup methods if available,
        removing references, and forcing garbage collection.
        
        Args:
            name: Component name to unload
            emergency: If True, skip cleanup methods for faster unloading
            
        Returns:
            True if unloaded successfully, False otherwise
            
        Raises:
            Exception: Logs unloading errors but returns False
            
        Example:
            >>> success = await manager.unload_component("nlp_engine")
            >>> if success:
            ...     print("NLP engine unloaded")
        """
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
        """Get a loaded component and update last used time.
        
        Retrieves a loaded component instance and updates its last used
        timestamp for memory management decisions.
        
        Args:
            name: Component name to retrieve
            
        Returns:
            Component instance if loaded, None otherwise
            
        Example:
            >>> nlp_engine = manager.get_component("nlp_engine")
            >>> if nlp_engine:
            ...     result = nlp_engine.process(text)
        """
        if name in self.loaded_components:
            self.components[name].last_used = datetime.now()
            return self.loaded_components[name]
        return None

    def add_state_callback(self, callback: Callable) -> None:
        """Add a callback for memory state changes.
        
        Registers a callback function that will be called whenever
        the memory state changes (HEALTHY -> WARNING, etc.).
        
        Args:
            callback: Async function to call with MemorySnapshot parameter
            
        Example:
            >>> async def on_memory_change(snapshot):
            ...     print(f"Memory state: {snapshot.state.value}")
            >>> manager.add_state_callback(on_memory_change)
        """
        self.state_callbacks.append(callback)

    async def get_memory_report(self) -> Dict[str, Any]:
        """Generate a detailed memory report.
        
        Creates comprehensive report including current state, component status,
        historical analysis, and system configuration for debugging and monitoring.
        
        Returns:
            Dictionary containing detailed memory analysis
            
        Example:
            >>> report = await manager.get_memory_report()
            >>> print(f"Current state: {report['current_state']['state']}")
            >>> print(f"Memory usage: {report['current_state']['percent_used']:.1f}%")
        """
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

    async def cleanup(self) -> None:
        """Cleanup resources and stop monitoring.
        
        Gracefully shuts down the memory manager by stopping monitoring
        and cleaning up tracking resources.
        """
        await self.stop_monitoring()
        tracemalloc.stop()

# Example usage and testing
if __name__ == "__main__":

    async def test_memory_manager():
        """Test the memory manager functionality.
        
        Demonstrates basic usage including component registration,
        monitoring startup, component loading, and report generation.
        """
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
            """Mock component for testing memory management."""
            
            def __init__(self, size_mb: int):
                """Initialize mock component with specified memory usage.
                
                Args: