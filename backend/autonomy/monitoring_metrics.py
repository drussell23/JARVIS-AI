#!/usr/bin/env python3
"""
Monitoring and Metrics System for JARVIS
Tracks performance, health, and usage metrics across all components
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Deque, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"      # Incremental count
    GAUGE = "gauge"          # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"           # Rate per time unit

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    name: str
    type: MetricType
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    std_dev: float
    percentile_95: float
    rate_per_minute: float
    labels: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.type.value,
            'count': self.count,
            'sum': round(self.sum, 3),
            'min': round(self.min, 3),
            'max': round(self.max, 3),
            'mean': round(self.mean, 3),
            'median': round(self.median, 3),
            'std_dev': round(self.std_dev, 3),
            'p95': round(self.percentile_95, 3),
            'rate_per_minute': round(self.rate_per_minute, 3),
            'labels': self.labels
        }

class MetricCollector:
    """Collects and manages metrics"""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque())
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_labels: Dict[str, Dict[str, str]] = {}
        
        # Performance tracking
        self.performance_thresholds = {
            'ocr_time': 2.0,           # seconds
            'analysis_time': 1.0,      # seconds
            'decision_time': 0.5,      # seconds
            'action_execution': 5.0,   # seconds
            'websocket_latency': 0.1  # seconds
        }
        
        # Health indicators
        self.health_metrics = {
            'system_uptime': 0,
            'error_rate': 0,
            'success_rate': 0,
            'queue_depth': 0,
            'memory_usage': 0,
            'cpu_usage': 0
        }
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'error_rate': 0.1,      # 10% error rate
            'queue_depth': 100,     # Queue size
            'memory_usage': 80,     # Percentage
            'response_time': 5.0    # seconds
        }
        
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        # Store metric type
        if name not in self.metric_types:
            self.metric_types[name] = metric_type
            
        # Create metric point
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        
        # Add to metrics
        self.metrics[name].append(point)
        
        # Store labels for reference
        if labels:
            self.metric_labels[name] = labels
            
        # Clean old data
        self._clean_old_data(name)
        
        # Check for alerts
        self._check_alerts(name, value)
        
    def increment_counter(self, name: str, increment: float = 1.0,
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        current = self._get_current_value(name) or 0
        self.record_metric(
            name, 
            current + increment,
            MetricType.COUNTER,
            labels
        )
        
    def set_gauge(self, name: str, value: float,
                 labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
        
    def record_duration(self, name: str, duration: float,
                       labels: Optional[Dict[str, str]] = None):
        """Record a duration/timing metric"""
        self.record_metric(name, duration, MetricType.HISTOGRAM, labels)
        
    def _clean_old_data(self, metric_name: str):
        """Remove data older than window size"""
        cutoff = datetime.now() - self.window_size
        
        while (self.metrics[metric_name] and 
               self.metrics[metric_name][0].timestamp < cutoff):
            self.metrics[metric_name].popleft()
            
    def _get_current_value(self, name: str) -> Optional[float]:
        """Get current value of a metric"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1].value
        return None
        
    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric triggers any alerts"""
        # Check performance thresholds
        if metric_name in self.performance_thresholds:
            threshold = self.performance_thresholds[metric_name]
            if value > threshold:
                self._create_alert(
                    'performance',
                    f"{metric_name} exceeded threshold: {value:.2f}s > {threshold}s"
                )
                
        # Check general thresholds
        if metric_name == 'error_rate' and value > self.alert_thresholds['error_rate']:
            self._create_alert(
                'error_rate',
                f"High error rate: {value:.1%}"
            )
        elif metric_name == 'queue_depth' and value > self.alert_thresholds['queue_depth']:
            self._create_alert(
                'queue_overflow',
                f"Queue depth critical: {int(value)} items"
            )
            
    def _create_alert(self, alert_type: str, message: str):
        """Create an alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {message}")
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = [p.value for p in self.metrics[name]]
        
        # Calculate rate per minute
        if len(self.metrics[name]) > 1:
            time_span = (
                self.metrics[name][-1].timestamp - 
                self.metrics[name][0].timestamp
            ).total_seconds() / 60.0
            
            if self.metric_types[name] == MetricType.COUNTER:
                rate = (values[-1] - values[0]) / max(time_span, 1.0)
            else:
                rate = len(values) / max(time_span, 1.0)
        else:
            rate = 0.0
            
        return MetricSummary(
            name=name,
            type=self.metric_types[name],
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            percentile_95=sorted(values)[int(len(values) * 0.95)] if values else 0.0,
            rate_per_minute=rate,
            labels=self.metric_labels.get(name, {})
        )
        
    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get summaries for all metrics"""
        summaries = {}
        
        for name in self.metrics:
            summary = self.get_metric_summary(name)
            if summary:
                summaries[name] = summary
                
        return summaries

class SystemMonitor:
    """High-level system monitoring"""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.start_time = datetime.now()
        self.monitoring_task = None
        self.is_monitoring = False
        
        # Component health scores
        self.component_health: Dict[str, float] = {}
        
        # System metrics
        self.system_metrics = {
            'total_captures': 0,
            'total_decisions': 0,
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'total_errors': 0
        }
        
    def record_capture(self, duration: float):
        """Record a screen capture"""
        self.metric_collector.record_duration('capture_duration', duration)
        self.metric_collector.increment_counter('total_captures')
        self.system_metrics['total_captures'] += 1
        
    def record_ocr(self, duration: float, text_regions: int):
        """Record OCR processing"""
        self.metric_collector.record_duration('ocr_duration', duration)
        self.metric_collector.set_gauge('ocr_regions', text_regions)
        
    def record_analysis(self, duration: float, windows_analyzed: int):
        """Record window analysis"""
        self.metric_collector.record_duration('analysis_duration', duration)
        self.metric_collector.set_gauge('windows_analyzed', windows_analyzed)
        
    def record_decision(self, duration: float, decisions_made: int):
        """Record decision making"""
        self.metric_collector.record_duration('decision_duration', duration)
        self.metric_collector.increment_counter('total_decisions', decisions_made)
        self.system_metrics['total_decisions'] += decisions_made
        
    def record_action_execution(self, duration: float, success: bool):
        """Record action execution"""
        self.metric_collector.record_duration('action_execution', duration)
        self.metric_collector.increment_counter('total_actions')
        self.system_metrics['total_actions'] += 1
        
        if success:
            self.metric_collector.increment_counter('successful_actions')
            self.system_metrics['successful_actions'] += 1
        else:
            self.metric_collector.increment_counter('failed_actions')
            self.system_metrics['failed_actions'] += 1
            
    def record_error(self, component: str, severity: str):
        """Record an error"""
        self.metric_collector.increment_counter('total_errors')
        self.metric_collector.increment_counter(
            f'errors_{component}',
            labels={'severity': severity}
        )
        self.system_metrics['total_errors'] += 1
        
    def update_component_health(self, component: str, health_score: float):
        """Update component health score"""
        self.component_health[component] = health_score
        self.metric_collector.set_gauge(
            f'health_{component}',
            health_score
        )
        
    def record_queue_depth(self, depth: int):
        """Record action queue depth"""
        self.metric_collector.set_gauge('queue_depth', depth)
        
    def record_websocket_latency(self, latency: float):
        """Record WebSocket latency"""
        self.metric_collector.record_duration('websocket_latency', latency)
        
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
        
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        logger.info("System monitoring stopped")
        
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                # Calculate system health
                await self._calculate_system_health()
                
                # Check for anomalies
                self._detect_anomalies()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _calculate_system_health(self):
        """Calculate overall system health"""
        # Calculate success rate
        total_actions = self.system_metrics['total_actions']
        if total_actions > 0:
            success_rate = self.system_metrics['successful_actions'] / total_actions
        else:
            success_rate = 1.0
            
        self.metric_collector.set_gauge('success_rate', success_rate)
        
        # Calculate error rate
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        error_rate = self.system_metrics['total_errors'] / max(uptime_minutes, 1)
        self.metric_collector.set_gauge('error_rate_per_minute', error_rate)
        
        # Overall system health
        avg_component_health = (
            statistics.mean(self.component_health.values())
            if self.component_health else 1.0
        )
        
        system_health = (
            avg_component_health * 0.4 +
            success_rate * 0.4 +
            (1.0 - min(error_rate / 10, 1.0)) * 0.2
        )
        
        self.metric_collector.set_gauge('system_health', system_health)
        
    def _detect_anomalies(self):
        """Detect anomalies in metrics"""
        summaries = self.metric_collector.get_all_summaries()
        
        for name, summary in summaries.items():
            # Check for high standard deviation
            if summary.std_dev > summary.mean * 0.5:  # 50% deviation
                logger.warning(
                    f"High variance in {name}: "
                    f"mean={summary.mean:.2f}, std={summary.std_dev:.2f}"
                )
                
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        uptime = datetime.now() - self.start_time
        summaries = self.metric_collector.get_all_summaries()
        
        return {
            'uptime': {
                'days': uptime.days,
                'hours': uptime.seconds // 3600,
                'minutes': (uptime.seconds % 3600) // 60
            },
            'system_metrics': self.system_metrics,
            'component_health': self.component_health,
            'performance_metrics': {
                name: summary.to_dict()
                for name, summary in summaries.items()
                if 'duration' in name or 'latency' in name
            },
            'throughput_metrics': {
                name: summary.to_dict()
                for name, summary in summaries.items()
                if 'total' in name or 'rate' in name
            },
            'current_values': {
                'queue_depth': self.metric_collector._get_current_value('queue_depth'),
                'system_health': self.metric_collector._get_current_value('system_health'),
                'success_rate': self.metric_collector._get_current_value('success_rate')
            },
            'alerts': self.metric_collector.alerts[-10:]  # Last 10 alerts
        }

# Global system monitor
system_monitor = SystemMonitor()

async def test_monitoring():
    """Test monitoring system"""
    print("📊 Testing Monitoring and Metrics System")
    print("=" * 50)
    
    monitor = SystemMonitor()
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Simulate various operations
    print("\n🔄 Simulating system operations...")
    
    # Captures
    for i in range(5):
        monitor.record_capture(0.1 + i * 0.05)
        await asyncio.sleep(0.1)
        
    # OCR
    monitor.record_ocr(0.5, 15)
    monitor.record_ocr(0.3, 8)
    
    # Analysis
    monitor.record_analysis(0.2, 5)
    monitor.record_analysis(0.4, 8)
    
    # Decisions
    monitor.record_decision(0.1, 3)
    monitor.record_decision(0.15, 2)
    
    # Actions
    monitor.record_action_execution(1.5, True)
    monitor.record_action_execution(2.0, True)
    monitor.record_action_execution(0.5, False)
    
    # Errors
    monitor.record_error('vision', 'medium')
    monitor.record_error('ocr', 'low')
    
    # Component health
    monitor.update_component_health('vision_pipeline', 0.95)
    monitor.update_component_health('decision_engine', 0.88)
    monitor.update_component_health('action_queue', 1.0)
    
    # Queue depth
    monitor.record_queue_depth(5)
    monitor.record_queue_depth(8)
    monitor.record_queue_depth(3)
    
    # Wait for metrics to accumulate
    await asyncio.sleep(2)
    
    # Get report
    report = monitor.get_monitoring_report()
    
    print(f"\n📈 Monitoring Report:")
    print(f"   Uptime: {report['uptime']['hours']}h {report['uptime']['minutes']}m")
    print(f"\n   System Metrics:")
    for metric, value in report['system_metrics'].items():
        print(f"     {metric}: {value}")
        
    print(f"\n   Component Health:")
    for component, health in report['component_health'].items():
        print(f"     {component}: {health:.2f}")
        
    print(f"\n   Current Values:")
    for metric, value in report['current_values'].items():
        if value is not None:
            print(f"     {metric}: {value:.2f}")
            
    # Stop monitoring
    await monitor.stop_monitoring()
    
    print("\n✅ Monitoring test complete!")

if __name__ == "__main__":
    asyncio.run(test_monitoring())