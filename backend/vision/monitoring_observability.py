"""
Phase 3: Monitoring & Observability System
Real-time performance dashboards, bottleneck identification
Automatic performance regression detection, resource usage forecasting
"""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
import psutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric point"""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str] = None


@dataclass
class PerformanceAnomaly:
    """Detected performance anomaly"""
    timestamp: float
    metric_name: str
    severity: str  # 'warning', 'critical'
    description: str
    current_value: float
    expected_value: float
    deviation_percent: float


@dataclass
class Bottleneck:
    """Identified system bottleneck"""
    timestamp: float
    component: str  # 'cpu', 'memory', 'io', 'network'
    severity: float  # 0-1
    description: str
    metrics: Dict[str, float]
    recommendations: List[str]


class MetricsCollector:
    """Collect and aggregate system metrics"""
    
    def __init__(self):
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.collection_interval = 1.0  # seconds
        self._lock = threading.RLock()
        self._running = True
        self._collector_thread = None
        
    def start_collection(self):
        """Start metrics collection thread"""
        def collect_loop():
            while self._running:
                try:
                    # Collect system metrics
                    metrics = self._collect_system_metrics()
                    
                    # Store metrics
                    with self._lock:
                        for metric in metrics:
                            self.metrics_buffer[metric.metric_name].append(metric)
                    
                    time.sleep(self.collection_interval)
                    
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
        
        self._collector_thread = threading.Thread(target=collect_loop, daemon=True)
        self._collector_thread.start()
        logger.info("Metrics collector started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=5)
    
    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect current system metrics"""
        timestamp = time.time()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(PerformanceMetric(timestamp, 'cpu_percent', cpu_percent))
        
        # Per-core CPU
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
        for i, cpu in enumerate(cpu_per_core):
            metrics.append(PerformanceMetric(
                timestamp, f'cpu_core_{i}', cpu,
                tags={'core': str(i)}
            ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(timestamp, 'memory_percent', memory.percent))
        metrics.append(PerformanceMetric(timestamp, 'memory_available_mb', memory.available / 1024 / 1024))
        
        # Swap metrics
        swap = psutil.swap_memory()
        metrics.append(PerformanceMetric(timestamp, 'swap_percent', swap.percent))
        
        # Disk IO
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.append(PerformanceMetric(timestamp, 'disk_read_mb', disk_io.read_bytes / 1024 / 1024))
            metrics.append(PerformanceMetric(timestamp, 'disk_write_mb', disk_io.write_bytes / 1024 / 1024))
        
        # Network IO
        net_io = psutil.net_io_counters()
        metrics.append(PerformanceMetric(timestamp, 'network_recv_mb', net_io.bytes_recv / 1024 / 1024))
        metrics.append(PerformanceMetric(timestamp, 'network_sent_mb', net_io.bytes_sent / 1024 / 1024))
        
        # Process metrics
        process = psutil.Process()
        metrics.append(PerformanceMetric(timestamp, 'process_threads', process.num_threads()))
        metrics.append(PerformanceMetric(timestamp, 'process_cpu_percent', process.cpu_percent()))
        metrics.append(PerformanceMetric(timestamp, 'process_memory_mb', process.memory_info().rss / 1024 / 1024))
        
        return metrics
    
    def get_metrics(self, metric_name: str, duration_seconds: int = 60) -> List[PerformanceMetric]:
        """Get recent metrics for analysis"""
        with self._lock:
            if metric_name not in self.metrics_buffer:
                return []
            
            cutoff_time = time.time() - duration_seconds
            return [m for m in self.metrics_buffer[metric_name] if m.timestamp > cutoff_time]


class AnomalyDetector:
    """Detect performance anomalies and regressions"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.baseline_window = 300  # 5 minutes for baseline
        self.anomaly_threshold = 2.0  # standard deviations
        self.anomaly_history = deque(maxlen=100)
        
    def detect_anomalies(self) -> List[PerformanceAnomaly]:
        """Detect anomalies in recent metrics"""
        anomalies = []
        
        # Key metrics to monitor
        metrics_to_check = [
            ('cpu_percent', 80, 'high'),  # metric, threshold, direction
            ('memory_percent', 85, 'high'),
            ('process_cpu_percent', 50, 'high'),
            ('swap_percent', 20, 'high'),
        ]
        
        for metric_name, threshold, direction in metrics_to_check:
            recent_metrics = self.metrics_collector.get_metrics(metric_name, 60)
            if len(recent_metrics) < 10:
                continue
            
            # Calculate statistics
            values = [m.value for m in recent_metrics]
            current_value = values[-1]
            mean_value = np.mean(values[:-1])
            std_value = np.std(values[:-1])
            
            # Check for anomaly
            if direction == 'high' and current_value > threshold:
                deviation = (current_value - mean_value) / max(std_value, 0.1)
                
                if deviation > self.anomaly_threshold:
                    anomaly = PerformanceAnomaly(
                        timestamp=time.time(),
                        metric_name=metric_name,
                        severity='critical' if current_value > threshold * 1.2 else 'warning',
                        description=f"{metric_name} is abnormally high",
                        current_value=current_value,
                        expected_value=mean_value,
                        deviation_percent=(current_value - mean_value) / mean_value * 100
                    )
                    anomalies.append(anomaly)
                    self.anomaly_history.append(anomaly)
        
        return anomalies
    
    def detect_performance_regression(self) -> Optional[Dict[str, float]]:
        """Detect performance regression compared to baseline"""
        # Compare current performance to baseline
        baseline_metrics = self.metrics_collector.get_metrics('cpu_percent', self.baseline_window)
        recent_metrics = self.metrics_collector.get_metrics('cpu_percent', 60)
        
        if len(baseline_metrics) < 30 or len(recent_metrics) < 10:
            return None
        
        baseline_cpu = np.mean([m.value for m in baseline_metrics[:30]])
        recent_cpu = np.mean([m.value for m in recent_metrics])
        
        regression_percent = (recent_cpu - baseline_cpu) / baseline_cpu * 100
        
        if regression_percent > 20:  # 20% regression threshold
            return {
                'baseline_cpu': baseline_cpu,
                'current_cpu': recent_cpu,
                'regression_percent': regression_percent,
                'severity': 'critical' if regression_percent > 50 else 'warning'
            }
        
        return None


class BottleneckAnalyzer:
    """Identify system bottlenecks"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    def identify_bottlenecks(self) -> List[Bottleneck]:
        """Identify current system bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck analysis
        cpu_bottleneck = self._analyze_cpu_bottleneck()
        if cpu_bottleneck:
            bottlenecks.append(cpu_bottleneck)
        
        # Memory bottleneck analysis
        memory_bottleneck = self._analyze_memory_bottleneck()
        if memory_bottleneck:
            bottlenecks.append(memory_bottleneck)
        
        # IO bottleneck analysis
        io_bottleneck = self._analyze_io_bottleneck()
        if io_bottleneck:
            bottlenecks.append(io_bottleneck)
        
        return bottlenecks
    
    def _analyze_cpu_bottleneck(self) -> Optional[Bottleneck]:
        """Analyze CPU bottleneck"""
        cpu_metrics = self.metrics_collector.get_metrics('cpu_percent', 60)
        if not cpu_metrics:
            return None
        
        cpu_values = [m.value for m in cpu_metrics]
        avg_cpu = np.mean(cpu_values)
        max_cpu = np.max(cpu_values)
        
        if avg_cpu > 70:  # CPU bottleneck threshold
            # Analyze per-core distribution
            core_metrics = {}
            for i in range(psutil.cpu_count()):
                core_data = self.metrics_collector.get_metrics(f'cpu_core_{i}', 60)
                if core_data:
                    core_metrics[f'core_{i}'] = np.mean([m.value for m in core_data])
            
            # Calculate severity
            severity = min(1.0, avg_cpu / 100)
            
            return Bottleneck(
                timestamp=time.time(),
                component='cpu',
                severity=severity,
                description=f"High CPU usage detected: {avg_cpu:.1f}% average",
                metrics={
                    'avg_cpu': avg_cpu,
                    'max_cpu': max_cpu,
                    **core_metrics
                },
                recommendations=[
                    "Enable more aggressive CPU throttling",
                    "Reduce thread count for parallel operations",
                    "Increase quantization level to INT4",
                    "Enable dynamic batching to reduce computation"
                ]
            )
        
        return None
    
    def _analyze_memory_bottleneck(self) -> Optional[Bottleneck]:
        """Analyze memory bottleneck"""
        memory_metrics = self.metrics_collector.get_metrics('memory_percent', 60)
        swap_metrics = self.metrics_collector.get_metrics('swap_percent', 60)
        
        if not memory_metrics:
            return None
        
        avg_memory = np.mean([m.value for m in memory_metrics])
        avg_swap = np.mean([m.value for m in swap_metrics]) if swap_metrics else 0
        
        if avg_memory > 80 or avg_swap > 10:
            severity = min(1.0, (avg_memory / 100 + avg_swap / 50) / 2)
            
            return Bottleneck(
                timestamp=time.time(),
                component='memory',
                severity=severity,
                description=f"Memory pressure detected: {avg_memory:.1f}% RAM, {avg_swap:.1f}% swap",
                metrics={
                    'avg_memory': avg_memory,
                    'avg_swap': avg_swap
                },
                recommendations=[
                    "Reduce cache size limits",
                    "Enable more aggressive model pruning",
                    "Implement gradient checkpointing",
                    "Use memory-mapped files for large data"
                ]
            )
        
        return None
    
    def _analyze_io_bottleneck(self) -> Optional[Bottleneck]:
        """Analyze IO bottleneck"""
        disk_read = self.metrics_collector.get_metrics('disk_read_mb', 60)
        disk_write = self.metrics_collector.get_metrics('disk_write_mb', 60)
        
        if not disk_read or not disk_write:
            return None
        
        # Calculate IO rate (MB/s)
        read_rates = []
        write_rates = []
        
        for i in range(1, len(disk_read)):
            time_delta = disk_read[i].timestamp - disk_read[i-1].timestamp
            read_rate = (disk_read[i].value - disk_read[i-1].value) / time_delta
            write_rate = (disk_write[i].value - disk_write[i-1].value) / time_delta
            read_rates.append(read_rate)
            write_rates.append(write_rate)
        
        if not read_rates:
            return None
        
        avg_read_rate = np.mean(read_rates)
        avg_write_rate = np.mean(write_rates)
        total_io_rate = avg_read_rate + avg_write_rate
        
        if total_io_rate > 100:  # 100 MB/s threshold
            severity = min(1.0, total_io_rate / 200)
            
            return Bottleneck(
                timestamp=time.time(),
                component='io',
                severity=severity,
                description=f"High IO usage: {total_io_rate:.1f} MB/s",
                metrics={
                    'read_rate_mb_s': avg_read_rate,
                    'write_rate_mb_s': avg_write_rate,
                    'total_io_rate': total_io_rate
                },
                recommendations=[
                    "Enable async IO operations",
                    "Use in-memory caching for frequent reads",
                    "Batch write operations",
                    "Consider using faster storage (SSD)"
                ]
            )
        
        return None


class ResourceForecaster:
    """Forecast future resource usage"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    def forecast_resources(self, horizon_minutes: int = 5) -> Dict[str, Any]:
        """Forecast resource usage for next N minutes"""
        forecasts = {}
        
        # Forecast CPU usage
        cpu_forecast = self._forecast_metric('cpu_percent', horizon_minutes)
        if cpu_forecast:
            forecasts['cpu'] = cpu_forecast
        
        # Forecast memory usage
        memory_forecast = self._forecast_metric('memory_percent', horizon_minutes)
        if memory_forecast:
            forecasts['memory'] = memory_forecast
        
        return forecasts
    
    def _forecast_metric(self, metric_name: str, horizon_minutes: int) -> Optional[Dict[str, Any]]:
        """Forecast single metric using simple linear regression"""
        # Get historical data
        metrics = self.metrics_collector.get_metrics(metric_name, 600)  # 10 minutes
        
        if len(metrics) < 30:
            return None
        
        # Prepare data
        timestamps = np.array([m.timestamp for m in metrics])
        values = np.array([m.value for m in metrics])
        
        # Normalize timestamps
        t0 = timestamps[0]
        x = (timestamps - t0) / 60  # Convert to minutes
        
        # Simple linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, values, rcond=None)[0]
        
        # Forecast
        forecast_times = np.arange(x[-1], x[-1] + horizon_minutes, 0.5)
        forecast_values = m * forecast_times + c
        
        # Calculate confidence (based on fit quality)
        residuals = values - (m * x + c)
        confidence = 1 - (np.std(residuals) / np.mean(values))
        
        return {
            'current_value': values[-1],
            'forecast_values': forecast_values.tolist(),
            'forecast_times': forecast_times.tolist(),
            'trend': 'increasing' if m > 0 else 'decreasing',
            'rate_per_minute': m,
            'confidence': confidence
        }


class MonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector(self.metrics_collector)
        self.bottleneck_analyzer = BottleneckAnalyzer(self.metrics_collector)
        self.resource_forecaster = ResourceForecaster(self.metrics_collector)
        
        # Start collection
        self.metrics_collector.start_collection()
        
        logger.info("✅ Monitoring & Observability system initialized")
    
    def generate_report(self, output_file: str = "performance_report.pdf") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report_data = {
            'timestamp': time.time(),
            'summary': self._generate_summary(),
            'anomalies': [asdict(a) for a in self.anomaly_detector.detect_anomalies()],
            'bottlenecks': [asdict(b) for b in self.bottleneck_analyzer.identify_bottlenecks()],
            'forecasts': self.resource_forecaster.forecast_resources(),
            'regression': self.anomaly_detector.detect_performance_regression()
        }
        
        # Generate visualizations
        self._generate_visualizations(output_file)
        
        return report_data
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        # Get recent metrics
        cpu_metrics = self.metrics_collector.get_metrics('cpu_percent', 300)
        memory_metrics = self.metrics_collector.get_metrics('memory_percent', 300)
        
        if not cpu_metrics or not memory_metrics:
            return {}
        
        return {
            'avg_cpu': np.mean([m.value for m in cpu_metrics]),
            'max_cpu': np.max([m.value for m in cpu_metrics]),
            'avg_memory': np.mean([m.value for m in memory_metrics]),
            'max_memory': np.max([m.value for m in memory_metrics]),
            'total_anomalies': len(self.anomaly_detector.anomaly_history),
            'uptime_seconds': time.time() - cpu_metrics[0].timestamp
        }
    
    def _generate_visualizations(self, output_file: str):
        """Generate performance visualizations"""
        try:
            with PdfPages(output_file) as pdf:
                # CPU usage plot
                fig, ax = plt.subplots(figsize=(10, 6))
                self._plot_metric(ax, 'cpu_percent', 'CPU Usage (%)', 300)
                pdf.savefig(fig)
                plt.close(fig)
                
                # Memory usage plot
                fig, ax = plt.subplots(figsize=(10, 6))
                self._plot_metric(ax, 'memory_percent', 'Memory Usage (%)', 300)
                pdf.savefig(fig)
                plt.close(fig)
                
                logger.info(f"Generated performance report: {output_file}")
                
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def _plot_metric(self, ax, metric_name: str, title: str, duration: int):
        """Plot single metric"""
        metrics = self.metrics_collector.get_metrics(metric_name, duration)
        if not metrics:
            return
        
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        values = [m.value for m in metrics]
        
        ax.plot(timestamps, values, label=metric_name)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.grid(True, alpha=0.3)
        
        # Add threshold line
        if 'cpu' in metric_name:
            ax.axhline(y=25, color='r', linestyle='--', label='Target (25%)')
        elif 'memory' in metric_name:
            ax.axhline(y=80, color='r', linestyle='--', label='Warning (80%)')
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        return {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'active_anomalies': len(self.anomaly_detector.detect_anomalies()),
            'active_bottlenecks': len(self.bottleneck_analyzer.identify_bottlenecks()),
            'forecasts': self.resource_forecaster.forecast_resources(horizon_minutes=1)
        }
    
    def shutdown(self):
        """Shutdown monitoring system"""
        self.metrics_collector.stop_collection()
        logger.info("Monitoring system shutdown")


# Test function
def test_monitoring_system():
    """Test monitoring and observability system"""
    logger.info("=" * 60)
    logger.info("TESTING MONITORING & OBSERVABILITY SYSTEM")
    logger.info("=" * 60)
    
    dashboard = MonitoringDashboard()
    
    # Let it collect some data
    logger.info("\nCollecting metrics for 10 seconds...")
    time.sleep(10)
    
    # Get real-time status
    status = dashboard.get_real_time_status()
    logger.info(f"\nReal-time status:")
    logger.info(f"  CPU: {status['cpu_percent']:.1f}%")
    logger.info(f"  Memory: {status['memory_percent']:.1f}%")
    logger.info(f"  Active anomalies: {status['active_anomalies']}")
    logger.info(f"  Active bottlenecks: {status['active_bottlenecks']}")
    
    # Generate report
    logger.info("\nGenerating performance report...")
    report = dashboard.generate_report("test_performance_report.pdf")
    
    logger.info(f"\nReport summary:")
    for key, value in report['summary'].items():
        logger.info(f"  {key}: {value}")
    
    # Check for anomalies
    if report['anomalies']:
        logger.info(f"\nDetected {len(report['anomalies'])} anomalies")
    
    # Check for bottlenecks
    if report['bottlenecks']:
        logger.info(f"\nIdentified {len(report['bottlenecks'])} bottlenecks")
        for bottleneck in report['bottlenecks']:
            logger.info(f"  - {bottleneck['component']}: {bottleneck['description']}")
    
    dashboard.shutdown()
    logger.info("\n✅ Monitoring test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_monitoring_system()