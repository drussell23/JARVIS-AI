#!/usr/bin/env python3
"""
Memory Profiler and Benchmarking Tool for JARVIS Vision System
Measures memory usage, tracks performance improvements, validates Priority 3 targets
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil

# Ensure vision directory is in path for imports
vision_dir = os.path.dirname(os.path.abspath(__file__))
if vision_dir not in sys.path:
    sys.path.insert(0, vision_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""

    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_gb: float
    pressure: str = "unknown"
    active_components: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""

    name: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    duration_seconds: float
    memory_delta_mb: float
    peak_memory_mb: float
    avg_memory_mb: float
    success: bool
    notes: str = ""


class MemoryProfiler:
    """Memory profiler for JARVIS components"""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.benchmarks: List[BenchmarkResult] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None

    def take_snapshot(self, components: Optional[List[str]] = None) -> MemorySnapshot:
        """Take a memory snapshot"""
        mem_info = self.process.memory_info()
        vm_info = psutil.virtual_memory()

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            percent=self.process.memory_percent(),
            available_gb=vm_info.available / 1024 / 1024 / 1024,
            active_components=components or [],
        )

        self.snapshots.append(snapshot)
        return snapshot

    def set_baseline(self):
        """Set baseline memory snapshot"""
        self.baseline_snapshot = self.take_snapshot(["baseline"])
        logger.info(
            f"📊 Baseline: RSS={self.baseline_snapshot.rss_mb:.1f}MB, "
            f"Available={self.baseline_snapshot.available_gb:.1f}GB"
        )

    def get_memory_delta(self) -> float:
        """Get memory delta from baseline"""
        if not self.baseline_snapshot:
            return 0.0
        current = self.take_snapshot()
        return current.rss_mb - self.baseline_snapshot.rss_mb

    async def profile_component(
        self, name: str, component_func, duration_seconds: float = 60, sample_interval: float = 1.0
    ) -> BenchmarkResult:
        """Profile a component's memory usage over time"""
        logger.info(f"🔍 Profiling {name} for {duration_seconds}s...")

        start_snapshot = self.take_snapshot([name])
        start_time = time.time()

        memory_samples = []

        # Start component
        component_task = asyncio.create_task(component_func())

        # Sample memory periodically
        try:
            while time.time() - start_time < duration_seconds:
                await asyncio.sleep(sample_interval)
                snapshot = self.take_snapshot([name])
                memory_samples.append(snapshot.rss_mb)

        except Exception as e:
            logger.error(f"Error profiling {name}: {e}")

        finally:
            # Stop component
            if not component_task.done():
                component_task.cancel()
                try:
                    await component_task
                except asyncio.CancelledError:
                    pass

        end_snapshot = self.take_snapshot([name])
        duration = time.time() - start_time

        # Calculate statistics
        memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb
        peak_memory = max(memory_samples) if memory_samples else end_snapshot.rss_mb
        avg_memory = (
            sum(memory_samples) / len(memory_samples) if memory_samples else end_snapshot.rss_mb
        )

        result = BenchmarkResult(
            name=name,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            duration_seconds=duration,
            memory_delta_mb=memory_delta,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            success=True,
        )

        self.benchmarks.append(result)
        logger.info(
            f"✅ {name}: Δ={memory_delta:+.1f}MB, Peak={peak_memory:.1f}MB, Avg={avg_memory:.1f}MB"
        )

        return result

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive memory profiling report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "baseline": {
                "rss_mb": self.baseline_snapshot.rss_mb if self.baseline_snapshot else 0,
                "available_gb": (
                    self.baseline_snapshot.available_gb if self.baseline_snapshot else 0
                ),
            },
            "benchmarks": [],
            "summary": {
                "total_components": len(self.benchmarks),
                "total_memory_delta_mb": 0.0,
                "highest_peak_mb": 0.0,
                "priority_3_target_met": False,
            },
        }

        total_delta = 0.0
        highest_peak = 0.0

        for benchmark in self.benchmarks:
            total_delta += benchmark.memory_delta_mb
            highest_peak = max(highest_peak, benchmark.peak_memory_mb)

            report["benchmarks"].append(
                {
                    "name": benchmark.name,
                    "duration_s": benchmark.duration_seconds,
                    "memory_delta_mb": benchmark.memory_delta_mb,
                    "peak_memory_mb": benchmark.peak_memory_mb,
                    "avg_memory_mb": benchmark.avg_memory_mb,
                    "success": benchmark.success,
                    "notes": benchmark.notes,
                }
            )

        report["summary"]["total_memory_delta_mb"] = total_delta
        report["summary"]["highest_peak_mb"] = highest_peak

        # Check Priority 3 target: Reduce 1.2GB → 800MB (33% reduction)
        # Target: Keep multi-space vision under 800MB
        if highest_peak <= 800:
            report["summary"]["priority_3_target_met"] = True
            report["summary"][
                "target_status"
            ] = f"✅ SUCCESS: Peak {highest_peak:.0f}MB ≤ 800MB target"
        else:
            report["summary"]["priority_3_target_met"] = False
            report["summary"][
                "target_status"
            ] = f"❌ EXCEEDED: Peak {highest_peak:.0f}MB > 800MB target"

        # Calculate reduction from baseline 1.2GB
        baseline_mb = 1200
        reduction_percent = 100 * (1 - highest_peak / baseline_mb)
        report["summary"]["reduction_from_baseline"] = {
            "baseline_mb": baseline_mb,
            "current_peak_mb": highest_peak,
            "reduction_mb": baseline_mb - highest_peak,
            "reduction_percent": reduction_percent,
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to {output_file}")

        return report

    def print_summary(self):
        """Print human-readable summary"""
        if not self.baseline_snapshot:
            logger.warning("No baseline set")
            return

        print("\n" + "=" * 80)
        print("JARVIS VISION SYSTEM - MEMORY PROFILING SUMMARY")
        print("=" * 80)

        print(f"\n📊 Baseline Memory:")
        print(f"   RSS: {self.baseline_snapshot.rss_mb:.1f}MB")
        print(f"   Available: {self.baseline_snapshot.available_gb:.1f}GB")

        print(f"\n🔍 Benchmarked Components: {len(self.benchmarks)}")
        for benchmark in self.benchmarks:
            print(f"\n   {benchmark.name}:")
            print(f"      Duration: {benchmark.duration_seconds:.1f}s")
            print(f"      Memory Δ: {benchmark.memory_delta_mb:+.1f}MB")
            print(f"      Peak: {benchmark.peak_memory_mb:.1f}MB")
            print(f"      Average: {benchmark.avg_memory_mb:.1f}MB")

        # Calculate totals
        total_delta = sum(b.memory_delta_mb for b in self.benchmarks)
        highest_peak = max((b.peak_memory_mb for b in self.benchmarks), default=0)

        print(f"\n📈 Overall Statistics:")
        print(f"   Total Memory Δ: {total_delta:+.1f}MB")
        print(f"   Highest Peak: {highest_peak:.1f}MB")

        # Priority 3 target check
        target_mb = 800
        baseline_mb = 1200
        print(f"\n🎯 Priority 3 Target: {baseline_mb}MB → {target_mb}MB (33% reduction)")

        if highest_peak <= target_mb:
            reduction = baseline_mb - highest_peak
            reduction_pct = 100 * (reduction / baseline_mb)
            print(f"   ✅ SUCCESS: Peak {highest_peak:.0f}MB ≤ {target_mb}MB")
            print(f"   💰 Reduction: {reduction:.0f}MB ({reduction_pct:.1f}%)")
        else:
            excess = highest_peak - target_mb
            print(f"   ❌ EXCEEDED: Peak {highest_peak:.0f}MB > {target_mb}MB")
            print(f"   ⚠️  Over target by: {excess:.0f}MB")

        print("\n" + "=" * 80 + "\n")


async def benchmark_multi_space_vision():
    """Benchmark multi-space vision system"""
    try:
        from macos_memory_manager import initialize_memory_manager
        from multi_space_capture_engine import (
            CaptureQuality,
            MultiSpaceCaptureEngine,
            SpaceCaptureRequest,
        )

        # Initialize memory manager
        memory_manager = await initialize_memory_manager()

        # Initialize capture engine with memory manager
        engine = MultiSpaceCaptureEngine(memory_manager=memory_manager)

        # Run captures for 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            request = SpaceCaptureRequest(
                space_ids=[1, 2, 3], quality=CaptureQuality.OPTIMIZED, use_cache=True
            )
            _ = await engine.capture_all_spaces(request)  # noqa: F841
            await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Benchmark error: {e}")


async def benchmark_semantic_cache():
    """Benchmark semantic cache system"""
    try:
        from intelligence.semantic_cache_lsh import get_semantic_cache
        from macos_memory_manager import initialize_memory_manager

        # Initialize memory manager
        memory_manager = await initialize_memory_manager()

        # Get cache with memory manager
        cache = await get_semantic_cache(memory_manager=memory_manager)

        # Generate test load
        start_time = time.time()
        while time.time() - start_time < 60:
            for i in range(10):
                key = f"test_query_{i % 100}"
                await cache.put(key, {"result": f"data_{i}"}, context={"test": True})
                _ = await cache.get(key)  # noqa: F841
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Benchmark error: {e}")


async def run_full_benchmark():
    """Run complete benchmark suite"""
    profiler = MemoryProfiler()

    logger.info("🚀 Starting JARVIS Vision Memory Profiling")
    logger.info("=" * 80)

    # Set baseline
    profiler.set_baseline()

    # Benchmark multi-space vision
    await profiler.profile_component(
        "Multi-Space Vision", benchmark_multi_space_vision, duration_seconds=60
    )

    # Benchmark semantic cache
    await profiler.profile_component(
        "Semantic Cache", benchmark_semantic_cache, duration_seconds=60
    )

    # Generate report
    output_dir = os.path.expanduser("~/.jarvis/profiling")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    report = profiler.generate_report(output_file)
    profiler.print_summary()

    return report


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())
