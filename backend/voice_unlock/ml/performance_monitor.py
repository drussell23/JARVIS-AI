"""
ML Performance Monitor for Voice Unlock
======================================

Real-time monitoring and visualization of ML model performance
and resource usage for the voice unlock system.
"""

import time
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import numpy as np

# For visualization (optional)
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = None


@dataclass
class ModelMetrics:
    """Metrics for a specific model"""
    model_id: str
    load_count: int = 0
    inference_count: int = 0
    total_inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_used: datetime = None
    error_count: int = 0


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for ML voice unlock system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Merge provided config with defaults
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Metrics storage (using deque for efficient time-series data)
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_history_size'])
        )
        
        # Model-specific metrics
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # System metrics
        self.system_metrics = {
            'start_time': datetime.now(),
            'total_authentications': 0,
            'successful_authentications': 0,
            'failed_authentications': 0,
            'total_enrollments': 0,
            'active_users': 0
        }
        
        # Performance thresholds for alerts
        self.thresholds = self.config['thresholds']
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Start monitoring if enabled
        if self.config['auto_start']:
            self.start_monitoring()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_history_size': 1000,  # Max metrics to keep per metric type
            'monitoring_interval': 5.0,  # seconds
            'auto_start': True,
            'enable_logging': True,
            'log_interval': 60.0,  # seconds
            'metrics_file': '~/.jarvis/voice_unlock/metrics.jsonl',
            'thresholds': {
                'memory_usage_mb': 400,
                'cpu_percent': 50,
                'inference_time_ms': 100,
                'error_rate': 0.1,
                'cache_hit_rate': 0.7  # Alert if below this
            }
        }
        
    def record_metric(self, metric_name: str, value: float, 
                     unit: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics_history[metric_name].append(metric)
        
        # Check thresholds
        self._check_threshold(metric_name, value)
        
    def record_inference(self, model_id: str, inference_time: float, 
                        success: bool = True, memory_used: float = 0):
        """Record model inference performance"""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = ModelMetrics(model_id=model_id)
            
        metrics = self.model_metrics[model_id]
        metrics.inference_count += 1
        metrics.total_inference_time += inference_time
        metrics.last_used = datetime.now()
        
        if not success:
            metrics.error_count += 1
            
        if memory_used > 0:
            metrics.memory_usage_mb = memory_used
            
        # Record to history
        self.record_metric(
            f"inference_time_{model_id}",
            inference_time * 1000,  # Convert to ms
            "ms",
            {"success": success}
        )
        
    def record_authentication(self, user_id: str, success: bool, 
                            confidence: float, processing_time: float):
        """Record authentication attempt"""
        self.system_metrics['total_authentications'] += 1
        
        if success:
            self.system_metrics['successful_authentications'] += 1
        else:
            self.system_metrics['failed_authentications'] += 1
            
        self.record_metric("auth_confidence", confidence, "%")
        self.record_metric("auth_processing_time", processing_time * 1000, "ms")
        
        # Calculate success rate
        success_rate = (
            self.system_metrics['successful_authentications'] / 
            self.system_metrics['total_authentications']
        )
        self.record_metric("auth_success_rate", success_rate, "%")
        
    def record_enrollment(self, user_id: str, success: bool, 
                         sample_count: int, processing_time: float):
        """Record enrollment attempt"""
        self.system_metrics['total_enrollments'] += 1
        
        if success:
            self.system_metrics['active_users'] += 1
            
        self.record_metric("enrollment_time", processing_time * 1000, "ms")
        self.record_metric("enrollment_samples", sample_count, "samples")
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # ML-specific stats
        total_model_memory = sum(
            m.memory_usage_mb for m in self.model_metrics.values()
        )
        
        # Calculate rates
        uptime = (datetime.now() - self.system_metrics['start_time']).total_seconds()
        auth_rate = self.system_metrics['total_authentications'] / max(uptime, 1)
        
        stats = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_mb': memory.available / 1024 / 1024,
                'process_memory_mb': process_memory.rss / 1024 / 1024,
                'ml_memory_mb': total_model_memory,
                'uptime_seconds': uptime
            },
            'authentication': {
                'total': self.system_metrics['total_authentications'],
                'successful': self.system_metrics['successful_authentications'],
                'failed': self.system_metrics['failed_authentications'],
                'success_rate': (
                    self.system_metrics['successful_authentications'] / 
                    max(self.system_metrics['total_authentications'], 1)
                ),
                'rate_per_minute': auth_rate * 60
            },
            'models': {
                'active': len([m for m in self.model_metrics.values() 
                             if m.last_used and 
                             (datetime.now() - m.last_used).seconds < 300]),
                'total': len(self.model_metrics),
                'total_inferences': sum(m.inference_count for m in self.model_metrics.values()),
                'avg_inference_time_ms': np.mean([
                    m.total_inference_time / max(m.inference_count, 1) * 1000
                    for m in self.model_metrics.values()
                ]) if self.model_metrics else 0
            },
            'users': {
                'active': self.system_metrics['active_users'],
                'enrollments': self.system_metrics['total_enrollments']
            }
        }
        
        # Record current stats
        self.record_metric("cpu_percent", cpu_percent, "%")
        self.record_metric("memory_percent", memory.percent, "%")
        self.record_metric("ml_memory_mb", total_model_memory, "MB")
        
        return stats
        
    def get_metric_history(self, metric_name: str, 
                          duration_minutes: int = 60) -> List[Tuple[datetime, float]]:
        """Get metric history for the specified duration"""
        if metric_name not in self.metrics_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        return [
            (m.timestamp, m.value)
            for m in self.metrics_history[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
    def get_model_report(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for a specific model"""
        if model_id not in self.model_metrics:
            return None
            
        metrics = self.model_metrics[model_id]
        
        return {
            'model_id': model_id,
            'load_count': metrics.load_count,
            'inference_count': metrics.inference_count,
            'avg_inference_time_ms': (
                metrics.total_inference_time / max(metrics.inference_count, 1) * 1000
            ),
            'memory_usage_mb': metrics.memory_usage_mb,
            'cache_hit_rate': (
                metrics.cache_hits / max(metrics.cache_hits + metrics.cache_misses, 1)
            ),
            'error_rate': metrics.error_count / max(metrics.inference_count, 1),
            'last_used': metrics.last_used.isoformat() if metrics.last_used else None
        }
        
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds threshold and trigger alerts"""
        threshold_map = {
            'cpu_percent': self.thresholds.get('cpu_percent'),
            'memory_percent': self.thresholds.get('memory_percent'),
            'ml_memory_mb': self.thresholds.get('memory_usage_mb'),
            'inference_time_ms': self.thresholds.get('inference_time_ms')
        }
        
        threshold = threshold_map.get(metric_name)
        if threshold is None:
            return
            
        # Check if threshold is exceeded
        exceeded = False
        if metric_name == 'cache_hit_rate':
            exceeded = value < threshold  # Below threshold is bad
        else:
            exceeded = value > threshold  # Above threshold is bad
            
        if exceeded:
            self._trigger_alert(metric_name, value, threshold)
            
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger performance alert"""
        alert = {
            'timestamp': datetime.now(),
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'severity': 'warning' if value < threshold * 1.5 else 'critical'
        }
        
        logger.warning(f"Performance alert: {metric_name}={value:.2f} (threshold={threshold})")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
    def add_alert_callback(self, callback):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_log_time = time.time()
        
        while self.monitoring:
            try:
                # Collect current stats
                self.get_current_stats()
                
                # Log periodically
                if self.config['enable_logging']:
                    current_time = time.time()
                    if current_time - last_log_time >= self.config['log_interval']:
                        self._log_metrics()
                        last_log_time = current_time
                        
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            time.sleep(self.config['monitoring_interval'])
            
    def _log_metrics(self):
        """Log metrics to file"""
        metrics_file = Path(self.config['metrics_file']).expanduser()
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metrics snapshot
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_current_stats(),
            'recent_metrics': {}
        }
        
        # Include recent values for key metrics
        for metric_name in ['cpu_percent', 'memory_percent', 'ml_memory_mb', 
                          'auth_success_rate', 'auth_processing_time']:
            history = self.get_metric_history(metric_name, 5)  # Last 5 minutes
            if history:
                recent_values = [v for _, v in history[-10:]]  # Last 10 values
                snapshot['recent_metrics'][metric_name] = {
                    'mean': np.mean(recent_values),
                    'max': np.max(recent_values),
                    'min': np.min(recent_values)
                }
                
        # Append to log file
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(snapshot) + '\n')
            
    def plot_metrics(self, metric_names: List[str], duration_minutes: int = 60):
        """Plot metrics over time (requires matplotlib)"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plotting")
            return
            
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
            
        for ax, metric_name in zip(axes, metric_names):
            history = self.get_metric_history(metric_name, duration_minutes)
            
            if not history:
                ax.text(0.5, 0.5, f'No data for {metric_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            timestamps, values = zip(*history)
            ax.plot(timestamps, values)
            ax.set_title(f'{metric_name} (last {duration_minutes} minutes)')
            ax.set_xlabel('Time')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def export_report(self, output_path: str):
        """Export comprehensive performance report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.system_metrics['start_time']).total_seconds(),
            'current_stats': self.get_current_stats(),
            'model_reports': {
                model_id: self.get_model_report(model_id)
                for model_id in self.model_metrics
            },
            'metric_summaries': {}
        }
        
        # Calculate metric summaries
        for metric_name in self.metrics_history:
            history = self.get_metric_history(metric_name, 60 * 24)  # Last 24 hours
            if history:
                values = [v for _, v in history]
                report['metric_summaries'][metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': {
                        '50': np.percentile(values, 50),
                        '90': np.percentile(values, 90),
                        '99': np.percentile(values, 99)
                    }
                }
                
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report exported to {output_path}")


# Global monitor instance
_monitor = None


def get_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor(config)
    return _monitor