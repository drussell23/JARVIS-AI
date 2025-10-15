"""
Performance Monitor for JARVIS Intelligent Vision System
Tracks performance metrics, generates reports, and provides insights.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .intelligent_query_classifier import get_query_classifier
from .adaptive_learning_system import get_learning_system
from .smart_query_router import get_smart_router
from .query_context_manager import get_context_manager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)

    # Classification metrics
    total_classifications: int = 0
    avg_classification_latency_ms: float = 0
    classification_cache_hit_rate: float = 0

    # Routing metrics
    total_queries: int = 0
    avg_query_latency_ms: float = 0
    intent_distribution: Dict[str, int] = field(default_factory=dict)

    # Learning metrics
    overall_accuracy: float = 0
    recent_accuracy: float = 0
    total_feedback_records: int = 0

    # Context metrics
    session_duration_minutes: float = 0
    detected_user_pattern: str = "unknown"
    pattern_confidence: float = 0

    # System health
    memory_usage_mb: float = 0
    errors_count: int = 0


class PerformanceMonitor:
    """
    Monitors performance of the intelligent vision system.
    Generates reports and provides real-time insights.
    """

    def __init__(self, report_interval_minutes: int = 60):
        """
        Initialize performance monitor

        Args:
            report_interval_minutes: How often to generate automatic reports
        """
        self.report_interval = timedelta(minutes=report_interval_minutes)
        self.last_report_time = datetime.now()

        # Metrics history
        self._metrics_history: deque = deque(maxlen=100)

        # Error tracking
        self._errors: List[Dict[str, Any]] = []
        self._error_count = 0

        # Performance warnings
        self._warnings: List[str] = []

        logger.info("[MONITOR] Performance monitor initialized")

    async def collect_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics from all system components

        Returns:
            PerformanceMetrics snapshot
        """
        metrics = PerformanceMetrics(timestamp=datetime.now())

        try:
            # Classifier metrics
            classifier = get_query_classifier()
            classifier_stats = classifier.get_performance_stats()
            metrics.total_classifications = classifier_stats.get('total_classifications', 0)
            metrics.avg_classification_latency_ms = classifier_stats.get('avg_latency_ms', 0)
            metrics.classification_cache_hit_rate = classifier_stats.get('cache_hit_rate', 0)

            # Router metrics
            try:
                router = get_smart_router()
                router_stats = router.get_routing_stats()
                metrics.total_queries = router_stats.get('total_queries', 0)
                metrics.avg_query_latency_ms = router_stats.get('avg_latency_ms', 0)
                metrics.intent_distribution = router_stats.get('distribution', {})
            except Exception as e:
                logger.warning(f"[MONITOR] Could not get router stats: {e}")

            # Learning system metrics
            learning_system = get_learning_system()
            learning_report = learning_system.get_accuracy_report()
            metrics.overall_accuracy = learning_report.get('overall_accuracy', 0)
            metrics.recent_accuracy = learning_report.get('recent_accuracy', 0)
            metrics.total_feedback_records = learning_report.get('total_queries', 0)

            # Context manager metrics
            context_manager = get_context_manager()
            session_stats = context_manager.get_session_stats()
            metrics.session_duration_minutes = session_stats.get('session_duration_minutes', 0)
            metrics.detected_user_pattern = session_stats.get('detected_pattern', 'unknown')
            metrics.pattern_confidence = session_stats.get('pattern_confidence', 0)

            # System health (memory usage)
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            except ImportError:
                metrics.memory_usage_mb = 0  # psutil not available

            metrics.errors_count = self._error_count

            # Store in history
            self._metrics_history.append(metrics)

            # Check for warnings
            self._check_performance_warnings(metrics)

        except Exception as e:
            logger.error(f"[MONITOR] Failed to collect metrics: {e}", exc_info=True)
            self._record_error("metrics_collection", str(e))

        return metrics

    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """Check for performance issues and generate warnings"""
        self._warnings.clear()

        # High latency warning
        if metrics.avg_query_latency_ms > 5000:  # 5 seconds
            self._warnings.append(
                f"High query latency: {metrics.avg_query_latency_ms:.0f}ms (target: <5000ms)"
            )

        # Low classification accuracy warning
        if metrics.total_feedback_records > 50 and metrics.recent_accuracy < 0.85:
            self._warnings.append(
                f"Low classification accuracy: {metrics.recent_accuracy:.1%} (target: >85%)"
            )

        # Low cache hit rate warning
        if metrics.total_classifications > 100 and metrics.classification_cache_hit_rate < 0.4:
            self._warnings.append(
                f"Low cache hit rate: {metrics.classification_cache_hit_rate:.1%} (target: >40%)"
            )

        # High memory usage warning (on 16GB system, warn at 12GB process usage)
        if metrics.memory_usage_mb > 12000:
            self._warnings.append(
                f"High memory usage: {metrics.memory_usage_mb:.0f}MB (warning threshold: 12GB)"
            )

        # High error rate warning
        if metrics.total_queries > 0:
            error_rate = metrics.errors_count / metrics.total_queries
            if error_rate > 0.05:  # 5% error rate
                self._warnings.append(
                    f"High error rate: {error_rate:.1%} ({metrics.errors_count} errors)"
                )

        if self._warnings:
            logger.warning(f"[MONITOR] Performance warnings: {', '.join(self._warnings)}")

    def _record_error(self, error_type: str, error_message: str):
        """Record an error for tracking"""
        self._errors.append({
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now()
        })
        self._error_count += 1

        # Keep only recent errors (last 100)
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Dictionary with performance report
        """
        if not self._metrics_history:
            return {'error': 'No metrics available yet'}

        latest_metrics = self._metrics_history[-1]

        # Calculate trends (compare recent vs older metrics)
        if len(self._metrics_history) >= 10:
            recent_avg_latency = sum(
                m.avg_query_latency_ms for m in list(self._metrics_history)[-5:]
            ) / 5
            older_avg_latency = sum(
                m.avg_query_latency_ms for m in list(self._metrics_history)[-10:-5]
            ) / 5
            latency_trend = "improving" if recent_avg_latency < older_avg_latency else "degrading"

            recent_avg_accuracy = sum(
                m.recent_accuracy for m in list(self._metrics_history)[-5:]
            ) / 5
            older_avg_accuracy = sum(
                m.recent_accuracy for m in list(self._metrics_history)[-10:-5]
            ) / 5
            accuracy_trend = "improving" if recent_avg_accuracy > older_avg_accuracy else "stable"
        else:
            latency_trend = "insufficient_data"
            accuracy_trend = "insufficient_data"

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_queries': latest_metrics.total_queries,
                'avg_latency_ms': latest_metrics.avg_query_latency_ms,
                'classification_accuracy': latest_metrics.recent_accuracy,
                'session_duration_minutes': latest_metrics.session_duration_minutes,
                'memory_usage_mb': latest_metrics.memory_usage_mb,
            },
            'classification': {
                'total_classifications': latest_metrics.total_classifications,
                'avg_latency_ms': latest_metrics.avg_classification_latency_ms,
                'cache_hit_rate': latest_metrics.classification_cache_hit_rate,
            },
            'routing': {
                'total_queries': latest_metrics.total_queries,
                'avg_latency_ms': latest_metrics.avg_query_latency_ms,
                'intent_distribution': latest_metrics.intent_distribution,
            },
            'learning': {
                'overall_accuracy': latest_metrics.overall_accuracy,
                'recent_accuracy': latest_metrics.recent_accuracy,
                'total_feedback': latest_metrics.total_feedback_records,
            },
            'user_patterns': {
                'detected_pattern': latest_metrics.detected_user_pattern,
                'pattern_confidence': latest_metrics.pattern_confidence,
            },
            'trends': {
                'latency': latency_trend,
                'accuracy': accuracy_trend,
            },
            'health': {
                'warnings': self._warnings.copy(),
                'errors_count': self._error_count,
                'recent_errors': self._errors[-10:] if self._errors else [],
            },
            'targets': {
                'metadata_latency': '<100ms',
                'visual_latency': '1-3s',
                'deep_analysis_latency': '3-10s',
                'classification_accuracy': '>95%',
                'cache_hit_rate': '>60%',
                'memory_usage': '<300MB for ML components',
            }
        }

        # Add detailed learning report
        learning_system = get_learning_system()
        report['detailed_learning'] = learning_system.get_accuracy_report()

        return report

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for dashboard display"""
        if not self._metrics_history:
            return {'status': 'initializing'}

        latest = self._metrics_history[-1]

        # Get component stats
        classifier = get_query_classifier()
        context_manager = get_context_manager()

        return {
            'status': 'operational',
            'timestamp': latest.timestamp.isoformat(),
            'queries': {
                'total': latest.total_queries,
                'avg_latency_ms': round(latest.avg_query_latency_ms, 1),
            },
            'classification': {
                'accuracy': round(latest.recent_accuracy * 100, 1),
                'cache_hit_rate': round(latest.classification_cache_hit_rate * 100, 1),
            },
            'distribution': latest.intent_distribution,
            'user_pattern': {
                'type': latest.detected_user_pattern,
                'confidence': round(latest.pattern_confidence * 100, 1),
            },
            'health': {
                'memory_mb': round(latest.memory_usage_mb, 1),
                'errors': self._error_count,
                'warnings': len(self._warnings),
            },
            'warnings': self._warnings.copy(),
        }

    def should_generate_report(self) -> bool:
        """Check if it's time for an automatic report"""
        return datetime.now() - self.last_report_time >= self.report_interval

    def mark_report_generated(self):
        """Mark that a report was just generated"""
        self.last_report_time = datetime.now()

    def get_performance_insights(self) -> List[str]:
        """
        Generate actionable performance insights

        Returns:
            List of insight strings
        """
        if not self._metrics_history:
            return ["Not enough data to generate insights"]

        insights = []
        latest = self._metrics_history[-1]

        # Latency insights
        if latest.avg_query_latency_ms < 1000:
            insights.append("✅ Excellent query response times (<1s average)")
        elif latest.avg_query_latency_ms < 3000:
            insights.append("✓ Good query response times (<3s average)")
        else:
            insights.append(f"⚠️ High query latency ({latest.avg_query_latency_ms:.0f}ms). Consider optimizing screenshot capture.")

        # Accuracy insights
        if latest.recent_accuracy >= 0.95:
            insights.append("✅ Classification accuracy is excellent (>95%)")
        elif latest.recent_accuracy >= 0.85:
            insights.append("✓ Classification accuracy is good (>85%)")
        elif latest.total_feedback_records > 50:
            insights.append(f"⚠️ Classification accuracy below target ({latest.recent_accuracy:.1%}). System is learning from feedback.")

        # Cache insights
        if latest.classification_cache_hit_rate >= 0.6:
            insights.append(f"✅ High cache efficiency ({latest.classification_cache_hit_rate:.1%} hit rate)")
        elif latest.total_classifications > 100:
            insights.append(f"ℹ️ Cache hit rate at {latest.classification_cache_hit_rate:.1%}. Users querying diverse topics.")

        # Pattern insights
        if latest.pattern_confidence >= 0.7:
            insights.append(f"✓ User pattern detected: {latest.detected_user_pattern} ({latest.pattern_confidence:.0%} confidence)")

        # Distribution insights
        if latest.intent_distribution:
            dominant_intent = max(latest.intent_distribution.items(), key=lambda x: x[1])
            insights.append(f"ℹ️ Most common query type: {dominant_intent[0]} ({dominant_intent[1]:.0f}% of queries)")

        # Memory insights
        if latest.memory_usage_mb > 0:
            if latest.memory_usage_mb < 300:
                insights.append(f"✅ Low memory footprint ({latest.memory_usage_mb:.0f}MB)")
            elif latest.memory_usage_mb < 1000:
                insights.append(f"✓ Reasonable memory usage ({latest.memory_usage_mb:.0f}MB)")
            else:
                insights.append(f"⚠️ High memory usage ({latest.memory_usage_mb:.0f}MB)")

        return insights


# Singleton instance
_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor(report_interval_minutes: int = 60) -> PerformanceMonitor:
    """Get or create the singleton performance monitor"""
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor(report_interval_minutes)

    return _monitor_instance
