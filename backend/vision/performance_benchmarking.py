#!/usr/bin/env python3
"""
Performance Benchmarking System for Vision System v2.0
Comprehensive performance testing and analysis for generated capabilities
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking"""
    # Test parameters
    warmup_iterations: int = 5
    test_iterations: int = 50
    cooldown_time: float = 1.0
    
    # Load testing
    concurrent_users: List[int] = field(default_factory=lambda: [1, 5, 10, 25])
    request_rates: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 20.0])
    
    # Resource monitoring
    monitor_interval: float = 0.1  # seconds
    memory_threshold_mb: float = 100.0
    cpu_threshold_percent: float = 50.0
    
    # Output configuration
    generate_plots: bool = True
    output_dir: Path = field(default_factory=lambda: Path("backend/data/benchmarks"))


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run"""
    test_id: str
    timestamp: datetime
    
    # Timing metrics
    latency_ms: float
    throughput_rps: float  # requests per second
    
    # Resource metrics
    cpu_percent: float
    memory_mb: float
    memory_delta_mb: float
    
    # Quality metrics
    success: bool
    error: Optional[str] = None
    
    # Statistical metrics
    percentiles: Dict[int, float] = field(default_factory=dict)  # p50, p95, p99
    

@dataclass 
class BenchmarkResult:
    """Complete benchmark result for a capability"""
    capability_id: str
    capability_name: str
    timestamp: datetime
    config: BenchmarkConfig
    
    # Aggregated metrics
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    
    # Percentiles
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput
    max_throughput_rps: float
    sustained_throughput_rps: float
    
    # Resource usage
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    memory_leak_detected: bool
    
    # Reliability
    success_rate: float
    error_types: Dict[str, int]
    
    # Load testing results
    load_test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Raw metrics
    raw_metrics: List[PerformanceMetrics] = field(default_factory=list)


class ResourceMonitor:
    """Monitors system resources during benchmarking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.monitoring = False
        self.metrics_history = []
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.monitoring = True
        self.metrics_history = []
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        
    def capture_metrics(self) -> Dict[str, float]:
        """Capture current resource metrics"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'memory_delta_mb': memory_mb - self.baseline_memory if self.baseline_memory else 0,
            'timestamp': time.time()
        }
        
        if self.monitoring:
            self.metrics_history.append(metrics)
            
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary"""
        if not self.metrics_history:
            return {}
            
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_mb'] for m in self.metrics_history]
        memory_deltas = [m['memory_delta_mb'] for m in self.metrics_history]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'memory_growth_mb': max(memory_deltas) - min(memory_deltas),
            'monitoring_duration': self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp']
        }


class LatencyProfiler:
    """Profiles latency characteristics"""
    
    def __init__(self):
        self.latencies = []
        
    def add_latency(self, latency_ms: float):
        """Add a latency measurement"""
        self.latencies.append(latency_ms)
        
    def get_statistics(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {}
            
        sorted_latencies = sorted(self.latencies)
        
        return {
            'count': len(self.latencies),
            'mean': statistics.mean(self.latencies),
            'median': statistics.median(self.latencies),
            'std': statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            'min': min(self.latencies),
            'max': max(self.latencies),
            'p50': sorted_latencies[int(len(sorted_latencies) * 0.50)],
            'p95': sorted_latencies[int(len(sorted_latencies) * 0.95)],
            'p99': sorted_latencies[int(len(sorted_latencies) * 0.99)]
        }
    
    def plot_distribution(self, output_path: Path):
        """Plot latency distribution"""
        if not self.latencies or len(self.latencies) < 10:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.latencies, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(self.latencies)
        plt.ylabel('Latency (ms)')
        plt.title('Latency Box Plot')
        
        plt.tight_layout()
        plt.savefig(output_path / 'latency_distribution.png')
        plt.close()


class LoadTester:
    """Performs load testing on capabilities"""
    
    def __init__(self):
        self.results = []
        
    async def test_concurrent_load(
        self,
        test_func: Callable,
        concurrent_users: int,
        duration_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """Test with concurrent users"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Metrics collection
        latencies = []
        errors = []
        completed = 0
        
        async def user_simulation():
            """Simulate a single user"""
            nonlocal completed, latencies, errors
            
            while time.time() < end_time:
                req_start = time.time()
                try:
                    result = await test_func()
                    latency = (time.time() - req_start) * 1000
                    latencies.append(latency)
                    completed += 1
                    
                    if not result.get('success', False):
                        errors.append(result.get('error', 'Unknown error'))
                        
                except Exception as e:
                    errors.append(str(e))
                    
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Launch concurrent users
        tasks = [user_simulation() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        total_duration = time.time() - start_time
        throughput = completed / total_duration if total_duration > 0 else 0
        
        return {
            'concurrent_users': concurrent_users,
            'duration': total_duration,
            'requests_completed': completed,
            'throughput_rps': throughput,
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'error_rate': len(errors) / completed if completed > 0 else 1.0,
            'errors': dict(zip(*np.unique(errors, return_counts=True))) if errors else {}
        }
    
    async def test_request_rate(
        self,
        test_func: Callable,
        target_rps: float,
        duration_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """Test at specific request rate"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        request_interval = 1.0 / target_rps
        
        # Metrics
        latencies = []
        errors = []
        completed = 0
        
        async def send_requests():
            """Send requests at target rate"""
            nonlocal completed, latencies, errors
            
            next_request_time = start_time
            
            while time.time() < end_time:
                # Wait until next request time
                wait_time = next_request_time - time.time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
                # Send request
                req_start = time.time()
                try:
                    result = await test_func()
                    latency = (time.time() - req_start) * 1000
                    latencies.append(latency)
                    completed += 1
                    
                    if not result.get('success', False):
                        errors.append(result.get('error', 'Unknown error'))
                        
                except Exception as e:
                    errors.append(str(e))
                    
                next_request_time += request_interval
        
        await send_requests()
        
        # Calculate results
        total_duration = time.time() - start_time
        actual_rps = completed / total_duration if total_duration > 0 else 0
        
        return {
            'target_rps': target_rps,
            'actual_rps': actual_rps,
            'duration': total_duration,
            'requests_completed': completed,
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0,
            'error_rate': len(errors) / completed if completed > 0 else 1.0
        }


class PerformanceBenchmark:
    """Main performance benchmarking system"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.resource_monitor = ResourceMonitor()
        self.latency_profiler = LatencyProfiler()
        self.load_tester = LoadTester()
        
        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        
    async def benchmark_capability(
        self,
        capability_code: str,
        capability_name: str,
        test_input: Dict[str, Any],
        capability_id: Optional[str] = None
    ) -> BenchmarkResult:
        """Perform comprehensive performance benchmark"""
        logger.info(f"Starting performance benchmark for {capability_name}")
        
        # Create test function
        async def test_func():
            # This is simplified - in reality, would execute in sandbox
            start = time.time()
            # Simulate execution
            await asyncio.sleep(0.01 + np.random.exponential(0.02))  # Mock latency
            success = np.random.random() > 0.05  # 95% success rate
            
            return {
                'success': success,
                'latency': (time.time() - start) * 1000,
                'error': None if success else 'Simulated error'
            }
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Warmup
        logger.info(f"Running {self.config.warmup_iterations} warmup iterations...")
        for _ in range(self.config.warmup_iterations):
            await test_func()
            
        # Force garbage collection
        gc.collect()
        await asyncio.sleep(self.config.cooldown_time)
        
        # Main benchmark
        logger.info(f"Running {self.config.test_iterations} test iterations...")
        metrics = []
        
        for i in range(self.config.test_iterations):
            # Capture pre-execution state
            pre_resources = self.resource_monitor.capture_metrics()
            
            # Execute
            start_time = time.time()
            result = await test_func()
            latency_ms = (time.time() - start_time) * 1000
            
            # Capture post-execution state
            post_resources = self.resource_monitor.capture_metrics()
            
            # Record metrics
            metric = PerformanceMetrics(
                test_id=f"iter_{i}",
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                throughput_rps=1000.0 / latency_ms,  # Single request throughput
                cpu_percent=post_resources['cpu_percent'],
                memory_mb=post_resources['memory_mb'],
                memory_delta_mb=post_resources['memory_delta_mb'],
                success=result['success'],
                error=result.get('error')
            )
            
            metrics.append(metric)
            self.latency_profiler.add_latency(latency_ms)
            
            # Brief pause between iterations
            await asyncio.sleep(0.01)
        
        # Load testing
        load_results = await self._perform_load_testing(test_func)
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        resource_summary = self.resource_monitor.get_summary()
        
        # Calculate aggregate metrics
        latency_stats = self.latency_profiler.get_statistics()
        
        # Check for memory leaks
        memory_leak_detected = self._detect_memory_leak(metrics)
        
        # Create benchmark result
        result = BenchmarkResult(
            capability_id=capability_id or capability_name,
            capability_name=capability_name,
            timestamp=datetime.now(),
            config=self.config,
            avg_latency_ms=latency_stats['mean'],
            min_latency_ms=latency_stats['min'],
            max_latency_ms=latency_stats['max'],
            std_latency_ms=latency_stats['std'],
            p50_latency_ms=latency_stats['p50'],
            p95_latency_ms=latency_stats['p95'],
            p99_latency_ms=latency_stats['p99'],
            max_throughput_rps=1000.0 / latency_stats['min'],
            sustained_throughput_rps=1000.0 / latency_stats['mean'],
            avg_cpu_percent=resource_summary.get('avg_cpu_percent', 0),
            max_cpu_percent=resource_summary.get('max_cpu_percent', 0),
            avg_memory_mb=resource_summary.get('avg_memory_mb', 0),
            max_memory_mb=resource_summary.get('max_memory_mb', 0),
            memory_leak_detected=memory_leak_detected,
            success_rate=sum(1 for m in metrics if m.success) / len(metrics),
            error_types=self._categorize_errors(metrics),
            load_test_results=load_results,
            raw_metrics=metrics
        )
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(result)
        
        # Save results
        self._save_results(result)
        
        logger.info(f"Benchmark complete: {capability_name}")
        logger.info(f"  Avg latency: {result.avg_latency_ms:.2f}ms")
        logger.info(f"  P99 latency: {result.p99_latency_ms:.2f}ms")
        logger.info(f"  Success rate: {result.success_rate:.2%}")
        
        return result
    
    async def _perform_load_testing(self, test_func: Callable) -> Dict[str, Any]:
        """Perform load testing scenarios"""
        load_results = {
            'concurrent_users': {},
            'request_rates': {}
        }
        
        # Test with different concurrent users
        logger.info("Testing concurrent load...")
        for users in self.config.concurrent_users:
            result = await self.load_tester.test_concurrent_load(
                test_func,
                users,
                duration_seconds=5.0
            )
            load_results['concurrent_users'][users] = result
            logger.info(f"  {users} users: {result['throughput_rps']:.1f} rps, "
                       f"{result['avg_latency_ms']:.1f}ms avg latency")
        
        # Test at different request rates
        logger.info("Testing request rates...")
        for rps in self.config.request_rates:
            result = await self.load_tester.test_request_rate(
                test_func,
                rps,
                duration_seconds=5.0
            )
            load_results['request_rates'][rps] = result
            logger.info(f"  {rps} rps: actual {result['actual_rps']:.1f} rps, "
                       f"p99 latency {result['p99_latency_ms']:.1f}ms")
        
        return load_results
    
    def _detect_memory_leak(self, metrics: List[PerformanceMetrics]) -> bool:
        """Detect potential memory leaks"""
        if len(metrics) < 10:
            return False
            
        # Get memory deltas
        memory_deltas = [m.memory_delta_mb for m in metrics]
        
        # Check if memory is consistently growing
        # Simple linear regression
        x = np.arange(len(memory_deltas))
        slope, _ = np.polyfit(x, memory_deltas, 1)
        
        # If slope is positive and significant, might be a leak
        return slope > 0.01  # 0.01 MB per iteration growth threshold
    
    def _categorize_errors(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Categorize errors from metrics"""
        error_counts = defaultdict(int)
        
        for metric in metrics:
            if not metric.success and metric.error:
                error_counts[metric.error] += 1
                
        return dict(error_counts)
    
    def _generate_plots(self, result: BenchmarkResult):
        """Generate performance plots"""
        output_dir = self.config.output_dir / result.capability_name
        output_dir.mkdir(exist_ok=True)
        
        # Latency distribution
        self.latency_profiler.plot_distribution(output_dir)
        
        # Resource usage over time
        self._plot_resource_usage(output_dir)
        
        # Load test results
        self._plot_load_test_results(result.load_test_results, output_dir)
    
    def _plot_resource_usage(self, output_dir: Path):
        """Plot resource usage over time"""
        if not self.resource_monitor.metrics_history:
            return
            
        plt.figure(figsize=(12, 6))
        
        timestamps = [m['timestamp'] for m in self.resource_monitor.metrics_history]
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        # CPU usage
        plt.subplot(1, 2, 1)
        cpu_values = [m['cpu_percent'] for m in self.resource_monitor.metrics_history]
        plt.plot(relative_times, cpu_values)
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU %')
        plt.title('CPU Usage Over Time')
        plt.axhline(y=self.config.cpu_threshold_percent, color='r', linestyle='--', alpha=0.5)
        
        # Memory usage
        plt.subplot(1, 2, 2)
        memory_values = [m['memory_mb'] for m in self.resource_monitor.metrics_history]
        plt.plot(relative_times, memory_values)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage Over Time')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_usage.png')
        plt.close()
    
    def _plot_load_test_results(self, load_results: Dict[str, Any], output_dir: Path):
        """Plot load test results"""
        if not load_results:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Concurrent users vs throughput
        if 'concurrent_users' in load_results:
            plt.subplot(1, 2, 1)
            users = sorted(load_results['concurrent_users'].keys())
            throughputs = [load_results['concurrent_users'][u]['throughput_rps'] for u in users]
            latencies = [load_results['concurrent_users'][u]['avg_latency_ms'] for u in users]
            
            ax1 = plt.gca()
            ax1.plot(users, throughputs, 'b-', label='Throughput')
            ax1.set_xlabel('Concurrent Users')
            ax1.set_ylabel('Throughput (rps)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(users, latencies, 'r--', label='Avg Latency')
            ax2.set_ylabel('Avg Latency (ms)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title('Concurrent Load Performance')
        
        # Request rate vs latency
        if 'request_rates' in load_results:
            plt.subplot(1, 2, 2)
            rates = sorted(load_results['request_rates'].keys())
            p95_latencies = [load_results['request_rates'][r]['p95_latency_ms'] for r in rates]
            p99_latencies = [load_results['request_rates'][r]['p99_latency_ms'] for r in rates]
            
            plt.plot(rates, p95_latencies, 'g-', label='P95 Latency')
            plt.plot(rates, p99_latencies, 'r-', label='P99 Latency')
            plt.xlabel('Request Rate (rps)')
            plt.ylabel('Latency (ms)')
            plt.title('Request Rate vs Latency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'load_test_results.png')
        plt.close()
    
    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results"""
        output_file = self.config.output_dir / f"{result.capability_name}_benchmark.json"
        
        # Convert to serializable format
        result_dict = {
            'capability_id': result.capability_id,
            'capability_name': result.capability_name,
            'timestamp': result.timestamp.isoformat(),
            'metrics': {
                'latency': {
                    'avg_ms': result.avg_latency_ms,
                    'min_ms': result.min_latency_ms,
                    'max_ms': result.max_latency_ms,
                    'std_ms': result.std_latency_ms,
                    'p50_ms': result.p50_latency_ms,
                    'p95_ms': result.p95_latency_ms,
                    'p99_ms': result.p99_latency_ms
                },
                'throughput': {
                    'max_rps': result.max_throughput_rps,
                    'sustained_rps': result.sustained_throughput_rps
                },
                'resources': {
                    'avg_cpu_percent': result.avg_cpu_percent,
                    'max_cpu_percent': result.max_cpu_percent,
                    'avg_memory_mb': result.avg_memory_mb,
                    'max_memory_mb': result.max_memory_mb,
                    'memory_leak_detected': result.memory_leak_detected
                },
                'reliability': {
                    'success_rate': result.success_rate,
                    'error_types': result.error_types
                }
            },
            'load_test_results': result.load_test_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def compare_benchmarks(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Compare multiple benchmark results"""
        comparison = {
            'capabilities': [r.capability_name for r in results],
            'metrics': {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            'avg_latency_ms',
            'p99_latency_ms',
            'sustained_throughput_rps',
            'avg_cpu_percent',
            'avg_memory_mb',
            'success_rate'
        ]
        
        for metric in metrics_to_compare:
            comparison['metrics'][metric] = {
                r.capability_name: getattr(r, metric)
                for r in results
            }
        
        # Find best performer for each metric
        comparison['best_performers'] = {}
        for metric in metrics_to_compare:
            if 'latency' in metric or 'cpu' in metric or 'memory' in metric:
                # Lower is better
                best = min(results, key=lambda r: getattr(r, metric))
            else:
                # Higher is better
                best = max(results, key=lambda r: getattr(r, metric))
                
            comparison['best_performers'][metric] = best.capability_name
        
        return comparison


# Singleton instance
_performance_benchmark: Optional[PerformanceBenchmark] = None


def get_performance_benchmark(config: Optional[BenchmarkConfig] = None) -> PerformanceBenchmark:
    """Get singleton instance of performance benchmark"""
    global _performance_benchmark
    if _performance_benchmark is None:
        _performance_benchmark = PerformanceBenchmark(config)
    return _performance_benchmark