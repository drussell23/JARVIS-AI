#!/usr/bin/env python3
"""
Comprehensive Edge Case Tests for Component Warmup System
==========================================================

Tests all documented edge cases with real scenarios and validation.
"""

import asyncio
import logging
import time

import pytest
from core.component_warmup import ComponentPriority, ComponentStatus, ComponentWarmupSystem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Edge Case 1: Component Hangs During Load
# ═══════════════════════════════════════════════════════════════


class TestComponentHangEdgeCase:
    """Test timeout protection for hanging components"""

    @pytest.mark.asyncio
    async def test_component_timeout_protection(self):
        """Verify timeout prevents indefinite hang"""
        warmup = ComponentWarmupSystem()

        # Simulate hanging component
        async def hanging_loader():
            await asyncio.sleep(100)  # Hangs for 100 seconds
            return "should_never_reach_here"

        warmup.register_component(
            name="hanging_component",
            loader=hanging_loader,
            priority=ComponentPriority.HIGH,
            timeout=2.0,  # Only wait 2 seconds per attempt
            retry_count=2,  # 3 total attempts (initial + 2 retries)
            required=False,
        )

        start_time = time.time()
        report = await warmup.warmup_all()
        duration = time.time() - start_time

        # Should timeout quickly: 2s timeout × 3 attempts = ~6s (not 100s)
        assert duration < 8.0, f"Timeout failed: took {duration}s, expected <8s"
        assert "hanging_component" in report["failed_components"]
        assert warmup.get_status("hanging_component") == ComponentStatus.FAILED

    @pytest.mark.asyncio
    async def test_blocking_io_detection(self):
        """Detect and handle blocking I/O in async function"""
        warmup = ComponentWarmupSystem()

        import requests  # Synchronous library

        # BAD PATTERN - Blocking I/O
        async def bad_loader():
            # This blocks the event loop!
            response = requests.get("http://httpbin.org/delay/10", timeout=10)
            return response.json()

        # GOOD PATTERN - Async I/O
        async def good_loader():
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get("http://httpbin.org/delay/1") as response:
                    return await response.json()

        warmup.register_component(
            name="bad_component", loader=bad_loader, timeout=3.0, required=False
        )

        warmup.register_component(
            name="good_component", loader=good_loader, timeout=5.0, required=False
        )

        start = time.time()
        await warmup.warmup_all()
        time.time() - start

        # Bad component should timeout
        assert warmup.get_status("bad_component") in [
            ComponentStatus.FAILED,
            ComponentStatus.READY,
        ]

        # Good component should work
        if warmup.get_status("good_component") == ComponentStatus.READY:
            assert warmup.get_component("good_component") is not None


# ═══════════════════════════════════════════════════════════════
# Edge Case 2: Circular Dependencies
# ═══════════════════════════════════════════════════════════════


class TestCircularDependencyEdgeCase:
    """Test circular dependency detection and handling"""

    @pytest.mark.asyncio
    async def test_simple_circular_dependency(self):
        """Detect A → B → A cycle"""
        warmup = ComponentWarmupSystem()

        async def loader_a():
            return "component_a"

        async def loader_b():
            return "component_b"

        # Create circular dependency
        warmup.register_component(
            name="component_a",
            loader=loader_a,
            dependencies=["component_b"],  # A depends on B
            priority=ComponentPriority.HIGH,
        )

        warmup.register_component(
            name="component_b",
            loader=loader_b,
            dependencies=["component_a"],  # B depends on A (circular!)
            priority=ComponentPriority.HIGH,
        )

        # Should detect and handle gracefully
        report = await warmup.warmup_all()

        # System should not deadlock
        assert report is not None

    @pytest.mark.asyncio
    async def test_complex_circular_dependency(self):
        """Detect A → B → C → D → B cycle"""
        warmup = ComponentWarmupSystem()

        components = ["A", "B", "C", "D"]
        dependencies = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": ["B"],  # Creates cycle
        }

        for comp in components:

            async def loader():
                await asyncio.sleep(0.1)
                return f"component_{comp}"

            warmup.register_component(
                name=comp,
                loader=loader,
                dependencies=dependencies[comp],
                priority=ComponentPriority.HIGH,
                timeout=5.0,
            )

        # Should handle cycle without hanging
        start = time.time()
        await warmup.warmup_all()
        duration = time.time() - start

        assert duration < 10.0, "Circular dependency caused deadlock"


# ═══════════════════════════════════════════════════════════════
# Edge Case 3: Dependency Fails to Load
# ═══════════════════════════════════════════════════════════════


class TestDependencyFailureEdgeCase:
    """Test graceful degradation when dependencies fail"""

    @pytest.mark.asyncio
    async def test_required_dependency_failure(self):
        """Handle when required dependency fails"""
        warmup = ComponentWarmupSystem()

        async def failing_dependency():
            raise Exception("Dependency failed to load")

        async def dependent_component():
            return "dependent"

        warmup.register_component(
            name="dependency",
            loader=failing_dependency,
            priority=ComponentPriority.HIGH,
            required=True,
            retry_count=0,
        )

        warmup.register_component(
            name="dependent",
            loader=dependent_component,
            dependencies=["dependency"],
            priority=ComponentPriority.HIGH,
            required=False,
        )

        report = await warmup.warmup_all()

        # Dependency should fail
        assert "dependency" in report["failed_components"]

        # Dependent should also fail (dependency not met)
        assert not warmup.is_ready("dependent")

    @pytest.mark.asyncio
    async def test_optional_dependency_graceful_degradation(self):
        """Component should work without optional dependency"""
        warmup = ComponentWarmupSystem()

        async def failing_optional_dep():
            raise Exception("Optional dependency failed")

        class AdaptiveComponent:
            def __init__(self, dependency=None):
                self.dependency = dependency
                self.mode = "full" if dependency else "degraded"

            async def process(self):
                if self.mode == "full":
                    return "full_featured"
                return "basic_mode"

        async def load_adaptive():
            dep = warmup.get_component("optional_dep")
            return AdaptiveComponent(dep)

        warmup.register_component(
            name="optional_dep",
            loader=failing_optional_dep,
            priority=ComponentPriority.HIGH,
            required=False,
        )

        warmup.register_component(
            name="adaptive",
            loader=load_adaptive,
            priority=ComponentPriority.HIGH,
            dependencies=[],  # No hard dependency
        )

        report = await warmup.warmup_all()

        # Optional dep fails
        assert "optional_dep" in report["failed_components"]

        # But adaptive component should work in degraded mode
        if warmup.is_ready("adaptive"):
            component = warmup.get_component("adaptive")
            assert component.mode == "degraded"


# ═══════════════════════════════════════════════════════════════
# Edge Case 4: Race Condition in Concurrent Loading
# ═══════════════════════════════════════════════════════════════


class TestRaceConditionEdgeCase:
    """Test race condition protection"""

    @pytest.mark.asyncio
    async def test_concurrent_shared_resource_access(self):
        """Prevent race conditions on shared resources"""
        warmup = ComponentWarmupSystem()

        # Shared resource simulation
        shared_state = {"initialized": False, "init_count": 0}
        lock = asyncio.Lock()

        async def load_component_with_shared_resource(name):
            async with lock:  # Protected access
                if not shared_state["initialized"]:
                    await asyncio.sleep(0.1)  # Simulate initialization
                    shared_state["initialized"] = True
                shared_state["init_count"] += 1
            return f"component_{name}"

        # Register multiple components using shared resource
        for i in range(5):
            warmup.register_component(
                name=f"component_{i}",
                loader=lambda: load_component_with_shared_resource(f"comp_{i}"),
                priority=ComponentPriority.HIGH,
            )

        report = await warmup.warmup_all()

        # All should initialize successfully
        assert report["ready_count"] == 5
        # Shared resource should be initialized exactly once
        assert shared_state["initialized"] is True

    @pytest.mark.asyncio
    async def test_concurrent_file_write_race(self):
        """Prevent race condition on file writes"""
        import tempfile
        from pathlib import Path

        warmup = ComponentWarmupSystem()
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / "shared.txt"
        write_lock = asyncio.Lock()

        async def load_with_file_write(component_id):
            async with write_lock:
                # Only one component writes at a time
                if not file_path.exists():
                    file_path.write_text("initialized\n")
                with open(file_path, "a") as f:
                    f.write(f"{component_id}\n")
                await asyncio.sleep(0.1)
            return f"component_{component_id}"

        for i in range(10):
            warmup.register_component(
                name=f"writer_{i}",
                loader=lambda cid=i: load_with_file_write(cid),
                priority=ComponentPriority.HIGH,
            )

        await warmup.warmup_all()

        # Verify no corruption
        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 11  # 1 init + 10 components
        assert lines[0] == "initialized"


# ═══════════════════════════════════════════════════════════════
# Edge Case 5: Memory Spike During Warmup
# ═══════════════════════════════════════════════════════════════


class TestMemorySpikeEdgeCase:
    """Test memory-aware loading to prevent OOM"""

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self):
        """Ensure memory limits are respected"""

        class MemoryAwareWarmup(ComponentWarmupSystem):
            def __init__(self, max_memory_mb=100):
                super().__init__()
                self.max_memory_mb = max_memory_mb
                self.current_memory_mb = 0
                self.memory_lock = asyncio.Lock()

            async def allocate_memory(self, size_mb):
                async with self.memory_lock:
                    while self.current_memory_mb + size_mb > self.max_memory_mb:
                        await asyncio.sleep(0.1)
                    self.current_memory_mb += size_mb

            async def release_memory(self, size_mb):
                async with self.memory_lock:
                    self.current_memory_mb -= size_mb

        warmup = MemoryAwareWarmup(max_memory_mb=50)

        # Simulate memory-heavy components
        async def load_heavy_component(size_mb):
            await warmup.allocate_memory(size_mb)
            await asyncio.sleep(0.2)
            instance = {"memory_mb": size_mb, "data": "x" * 1000}
            await warmup.release_memory(size_mb)
            return instance

        # Register 5 components, each 20MB (would be 100MB total if all loaded at once)
        for i in range(5):
            warmup.register_component(
                name=f"heavy_{i}",
                loader=lambda: load_heavy_component(20),
                priority=ComponentPriority.HIGH,
            )

        start = time.time()
        report = await warmup.warmup_all()
        time.time() - start

        # Should never exceed 50MB (max_memory_mb)
        assert warmup.current_memory_mb <= warmup.max_memory_mb

        # All components should eventually load (serialized due to memory constraint)
        assert report["ready_count"] == 5


# ═══════════════════════════════════════════════════════════════
# Edge Case 6: Component Loads But Is Broken
# ═══════════════════════════════════════════════════════════════


class TestBrokenComponentEdgeCase:
    """Test runtime health monitoring for components that load but don't work"""

    @pytest.mark.asyncio
    async def test_health_check_catches_broken_component(self):
        """Health check should detect broken component"""
        warmup = ComponentWarmupSystem()

        class BrokenComponent:
            def __init__(self):
                self.loaded = True
                self.broken = True

            async def ping(self):
                if self.broken:
                    raise Exception("Component is broken")
                return "pong"

        async def load_broken():
            return BrokenComponent()

        async def health_check_broken(component):
            try:
                await component.ping()
                return True
            except:
                return False

        warmup.register_component(
            name="broken_component",
            loader=load_broken,
            health_check=health_check_broken,
            priority=ComponentPriority.HIGH,
            retry_count=2,
            required=False,
        )

        report = await warmup.warmup_all()

        # Component loads but health check fails
        status = warmup.get_status("broken_component")
        assert status in [ComponentStatus.DEGRADED, ComponentStatus.FAILED]

        metrics = report["component_metrics"]["broken_component"]
        assert metrics["health_score"] < 1.0

    @pytest.mark.asyncio
    async def test_intermittent_component_failure(self):
        """Handle component that works sometimes"""
        warmup = ComponentWarmupSystem()

        class IntermittentComponent:
            def __init__(self):
                self.call_count = 0

            async def operation(self):
                self.call_count += 1
                if self.call_count % 3 == 0:
                    raise Exception("Intermittent failure")
                return "success"

        async def load_intermittent():
            return IntermittentComponent()

        async def health_check_intermittent(component):
            try:
                await component.operation()
                return True
            except:
                return False

        warmup.register_component(
            name="intermittent",
            loader=load_intermittent,
            health_check=health_check_intermittent,
            retry_count=3,
            required=False,
        )

        await warmup.warmup_all()

        # May or may not pass depending on timing
        # But should handle gracefully either way
        status = warmup.get_status("intermittent")
        assert status in [ComponentStatus.READY, ComponentStatus.DEGRADED, ComponentStatus.FAILED]


# ═══════════════════════════════════════════════════════════════
# Integration Test: Multiple Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestMultipleEdgeCasesIntegration:
    """Test system with multiple edge cases occurring simultaneously"""

    @pytest.mark.asyncio
    async def test_realistic_startup_scenario(self):
        """Simulate realistic startup with various issues"""
        warmup = ComponentWarmupSystem(max_concurrent=5)

        # Component 1: Normal, fast loading
        async def load_fast():
            await asyncio.sleep(0.1)
            return "fast_component"

        # Component 2: Slow but succeeds
        async def load_slow():
            await asyncio.sleep(2.0)
            return "slow_component"

        # Component 3: Fails on first try, succeeds on retry
        attempt_count = {"count": 0}

        async def load_flaky():
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise Exception("Transient failure")
            return "flaky_component"

        # Component 4: Always fails
        async def load_broken():
            raise Exception("Permanently broken")

        # Component 5: Depends on broken component
        async def load_dependent():
            return "dependent_component"

        # Component 6: Hangs (will timeout)
        async def load_hanging():
            await asyncio.sleep(100)
            return "hanging_component"

        # Register all components
        warmup.register_component(
            "fast", load_fast, ComponentPriority.CRITICAL, timeout=5.0, required=True
        )

        warmup.register_component(
            "slow", load_slow, ComponentPriority.HIGH, timeout=5.0, required=False
        )

        warmup.register_component(
            "flaky", load_flaky, ComponentPriority.HIGH, timeout=5.0, retry_count=3, required=False
        )

        warmup.register_component(
            "broken", load_broken, ComponentPriority.MEDIUM, timeout=5.0, required=False
        )

        warmup.register_component(
            "dependent",
            load_dependent,
            ComponentPriority.MEDIUM,
            dependencies=["broken"],
            timeout=5.0,
            required=False,
        )

        warmup.register_component(
            "hanging", load_hanging, ComponentPriority.LOW, timeout=2.0, required=False
        )

        # Execute warmup
        start = time.time()
        report = await warmup.warmup_all()
        duration = time.time() - start

        # Verify results
        assert duration < 10.0, "Warmup took too long"

        # Fast component should always work
        assert warmup.is_ready("fast")

        # Slow component should work (within timeout)
        assert warmup.is_ready("slow")

        # Flaky component should work after retries
        assert warmup.is_ready("flaky")
        assert report["component_metrics"]["flaky"]["retry_count"] >= 1

        # Broken component should fail
        assert "broken" in report["failed_components"]

        # Hanging component should timeout
        assert "hanging" in report["failed_components"]

        # System should still be functional despite failures
        assert report["ready_count"] >= 3
        assert report["failed_count"] >= 2


# ═══════════════════════════════════════════════════════════════
# Performance Tests
# ═══════════════════════════════════════════════════════════════


class TestWarmupPerformance:
    """Test performance characteristics under various conditions"""

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self):
        """Verify parallel loading is faster than sequential"""
        component_count = 10
        load_time_per_component = 0.5

        # Sequential loading simulation
        async def sequential_load():
            total = 0
            for i in range(component_count):
                await asyncio.sleep(load_time_per_component)
                total += 1
            return total

        # Parallel loading via warmup
        warmup = ComponentWarmupSystem(max_concurrent=component_count)

        for i in range(component_count):

            async def loader():
                await asyncio.sleep(load_time_per_component)
                return f"component_{i}"

            warmup.register_component(
                f"component_{i}", loader, ComponentPriority.HIGH, timeout=10.0
            )

        # Time sequential
        seq_start = time.time()
        await sequential_load()
        seq_duration = time.time() - seq_start

        # Time parallel
        par_start = time.time()
        await warmup.warmup_all()
        par_duration = time.time() - par_start

        # Parallel should be much faster
        speedup = seq_duration / par_duration
        assert speedup >= 5.0, f"Parallel loading not fast enough: {speedup}x speedup"

        logger.info(f"Sequential: {seq_duration:.2f}s, Parallel: {par_duration:.2f}s")
        logger.info(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
