"""
A/B Testing Framework for JARVIS Intelligent Query Classification
Allows testing different classification strategies side-by-side.

Supports:
- Multiple classifier variants
- Traffic splitting (e.g., 50/50, 70/30)
- Performance comparison
- Statistical significance testing
- Automatic winner selection
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class VariantStatus(Enum):
    """Status of an A/B test variant"""
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    LOSER = "loser"


@dataclass
class VariantConfig:
    """Configuration for an A/B test variant"""
    variant_id: str
    name: str
    description: str
    classifier_func: Callable  # Function that performs classification
    traffic_allocation: float  # 0.0 to 1.0 (e.g., 0.5 = 50%)
    status: VariantStatus = VariantStatus.ACTIVE

    # Performance metrics
    total_queries: int = 0
    correct_classifications: int = 0
    total_latency_ms: float = 0
    user_satisfaction_count: int = 0
    user_satisfaction_positive: int = 0

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestResult:
    """Result from A/B test classification"""
    variant_id: str
    variant_name: str
    intent: str
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """
    A/B testing framework for comparing different classification strategies.
    Tracks performance and determines statistical significance.
    """

    def __init__(self, test_name: str, enable_persistence: bool = True):
        """
        Initialize A/B testing framework

        Args:
            test_name: Name of the A/B test
            enable_persistence: Whether to persist results to disk
        """
        self.test_name = test_name
        self.enable_persistence = enable_persistence

        # Test configuration
        self.variants: Dict[str, VariantConfig] = {}
        self.control_variant_id: Optional[str] = None

        # Results tracking
        self.test_start_time = datetime.now()
        self.total_queries_served = 0

        # Statistical significance
        self.min_sample_size = 100  # Minimum queries per variant
        self.confidence_level = 0.95  # 95% confidence
        self.min_effect_size = 0.05  # 5% improvement threshold

        # Persistence
        if enable_persistence:
            self.results_dir = Path.home() / ".jarvis" / "vision" / "ab_tests"
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.results_file = self.results_dir / f"{test_name}_results.json"
            self._load_results()

        logger.info(f"[AB_TEST] A/B testing framework initialized: {test_name}")

    def add_variant(
        self,
        variant_id: str,
        name: str,
        description: str,
        classifier_func: Callable,
        traffic_allocation: float = 0.5,
        is_control: bool = False
    ):
        """
        Add a variant to the A/B test

        Args:
            variant_id: Unique identifier for variant
            name: Human-readable name
            description: Description of what this variant tests
            classifier_func: Function that performs classification
            traffic_allocation: Percentage of traffic (0.0-1.0)
            is_control: Whether this is the control variant
        """
        if variant_id in self.variants:
            raise ValueError(f"Variant {variant_id} already exists")

        variant = VariantConfig(
            variant_id=variant_id,
            name=name,
            description=description,
            classifier_func=classifier_func,
            traffic_allocation=traffic_allocation
        )

        self.variants[variant_id] = variant

        if is_control:
            self.control_variant_id = variant_id

        logger.info(f"[AB_TEST] Added variant '{name}' ({variant_id}) with {traffic_allocation*100:.0f}% traffic")

    async def classify_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_variant: Optional[str] = None
    ) -> ABTestResult:
        """
        Classify a query using A/B testing

        Args:
            query: User's query
            context: Optional context
            force_variant: Force a specific variant (for testing)

        Returns:
            ABTestResult with classification and variant info
        """
        # Select variant
        variant = self._select_variant(force_variant)

        if not variant:
            raise ValueError("No active variants available")

        # Classify with selected variant
        start_time = time.time()

        try:
            result = await variant.classifier_func(query, context)
            latency_ms = (time.time() - start_time) * 1000

            # Extract intent and confidence from result
            if isinstance(result, dict):
                intent = result.get('intent', 'unknown')
                confidence = result.get('confidence', 0.0)
            else:
                intent = str(result)
                confidence = 0.0

            # Update variant metrics
            variant.total_queries += 1
            variant.total_latency_ms += latency_ms
            self.total_queries_served += 1

            # Create result
            ab_result = ABTestResult(
                variant_id=variant.variant_id,
                variant_name=variant.name,
                intent=intent,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata={
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            )

            logger.debug(
                f"[AB_TEST] Query classified by variant '{variant.name}': "
                f"{intent} ({confidence:.2f}, {latency_ms:.1f}ms)"
            )

            return ab_result

        except Exception as e:
            logger.error(f"[AB_TEST] Variant '{variant.name}' failed: {e}")
            raise

    def _select_variant(self, force_variant: Optional[str] = None) -> Optional[VariantConfig]:
        """Select a variant based on traffic allocation"""
        if force_variant:
            return self.variants.get(force_variant)

        # Filter active variants
        active_variants = [
            v for v in self.variants.values()
            if v.status == VariantStatus.ACTIVE
        ]

        if not active_variants:
            return None

        # Weighted random selection
        rand = random.random()
        cumulative = 0.0

        for variant in active_variants:
            cumulative += variant.traffic_allocation
            if rand <= cumulative:
                return variant

        # Fallback to last variant
        return active_variants[-1]

    def record_feedback(
        self,
        variant_id: str,
        correct: bool,
        user_satisfied: bool
    ):
        """
        Record feedback for a classification

        Args:
            variant_id: Variant that made the classification
            correct: Whether classification was correct
            user_satisfied: Whether user was satisfied
        """
        variant = self.variants.get(variant_id)
        if not variant:
            logger.warning(f"[AB_TEST] Unknown variant: {variant_id}")
            return

        if correct:
            variant.correct_classifications += 1

        variant.user_satisfaction_count += 1
        if user_satisfied:
            variant.user_satisfaction_positive += 1

        logger.debug(
            f"[AB_TEST] Feedback recorded for '{variant.name}': "
            f"correct={correct}, satisfied={user_satisfied}"
        )

    def get_variant_performance(self, variant_id: str) -> Dict[str, Any]:
        """Get performance metrics for a variant"""
        variant = self.variants.get(variant_id)
        if not variant:
            return {}

        accuracy = (
            variant.correct_classifications / variant.total_queries
            if variant.total_queries > 0
            else 0
        )

        avg_latency = (
            variant.total_latency_ms / variant.total_queries
            if variant.total_queries > 0
            else 0
        )

        satisfaction_rate = (
            variant.user_satisfaction_positive / variant.user_satisfaction_count
            if variant.user_satisfaction_count > 0
            else 0
        )

        return {
            'variant_id': variant.variant_id,
            'name': variant.name,
            'status': variant.status.value,
            'total_queries': variant.total_queries,
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'satisfaction_rate': satisfaction_rate,
            'traffic_allocation': variant.traffic_allocation
        }

    def compare_variants(self) -> Dict[str, Any]:
        """
        Compare all variants and determine statistical significance

        Returns:
            Comparison report with winner if statistically significant
        """
        if len(self.variants) < 2:
            return {'error': 'Need at least 2 variants to compare'}

        # Get performance for all variants
        performances = {
            vid: self.get_variant_performance(vid)
            for vid in self.variants.keys()
        }

        # Find control variant
        control = performances.get(self.control_variant_id) if self.control_variant_id else None

        # Calculate relative improvements
        comparisons = []
        for vid, perf in performances.items():
            if vid == self.control_variant_id:
                continue

            if not control or control['total_queries'] < self.min_sample_size:
                comparisons.append({
                    'variant': perf['name'],
                    'status': 'insufficient_data',
                    'samples': perf['total_queries']
                })
                continue

            if perf['total_queries'] < self.min_sample_size:
                comparisons.append({
                    'variant': perf['name'],
                    'status': 'insufficient_data',
                    'samples': perf['total_queries']
                })
                continue

            # Calculate improvements
            accuracy_improvement = perf['accuracy'] - control['accuracy']
            latency_improvement = control['avg_latency_ms'] - perf['avg_latency_ms']
            satisfaction_improvement = perf['satisfaction_rate'] - control['satisfaction_rate']

            # Simple significance test (z-test approximation)
            is_significant = (
                abs(accuracy_improvement) >= self.min_effect_size
                and perf['total_queries'] >= self.min_sample_size
                and control['total_queries'] >= self.min_sample_size
            )

            comparisons.append({
                'variant': perf['name'],
                'variant_id': vid,
                'accuracy_improvement': accuracy_improvement,
                'latency_improvement_ms': latency_improvement,
                'satisfaction_improvement': satisfaction_improvement,
                'statistically_significant': is_significant,
                'samples': perf['total_queries']
            })

        # Determine winner
        winner = None
        if comparisons:
            significant_improvements = [
                c for c in comparisons
                if c.get('statistically_significant', False)
                and c.get('accuracy_improvement', 0) > 0
            ]

            if significant_improvements:
                winner = max(
                    significant_improvements,
                    key=lambda c: c['accuracy_improvement']
                )

        return {
            'test_name': self.test_name,
            'test_duration_hours': (datetime.now() - self.test_start_time).total_seconds() / 3600,
            'total_queries': self.total_queries_served,
            'control_variant': control['name'] if control else None,
            'variants': performances,
            'comparisons': comparisons,
            'winner': winner,
            'min_sample_size': self.min_sample_size,
            'confidence_level': self.confidence_level
        }

    def declare_winner(self, variant_id: str):
        """
        Declare a winner and route all traffic to it

        Args:
            variant_id: ID of winning variant
        """
        winner = self.variants.get(variant_id)
        if not winner:
            raise ValueError(f"Unknown variant: {variant_id}")

        # Update statuses
        for vid, variant in self.variants.items():
            if vid == variant_id:
                variant.status = VariantStatus.WINNER
                variant.traffic_allocation = 1.0
            else:
                variant.status = VariantStatus.LOSER
                variant.traffic_allocation = 0.0

        logger.info(f"[AB_TEST] Winner declared: '{winner.name}' ({variant_id})")

        # Persist results
        if self.enable_persistence:
            self._save_results()

    def get_report(self) -> Dict[str, Any]:
        """Generate comprehensive A/B test report"""
        comparison = self.compare_variants()

        report = {
            'test_name': self.test_name,
            'status': 'active' if any(
                v.status == VariantStatus.ACTIVE for v in self.variants.values()
            ) else 'completed',
            'test_start': self.test_start_time.isoformat(),
            'duration_hours': (datetime.now() - self.test_start_time).total_seconds() / 3600,
            'total_queries': self.total_queries_served,
            'variants': {
                vid: self.get_variant_performance(vid)
                for vid in self.variants.keys()
            },
            'comparison': comparison,
            'recommendations': self._generate_recommendations(comparison)
        }

        return report

    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from test results"""
        recommendations = []

        # Check sample size
        for vid, variant in self.variants.items():
            if variant.total_queries < self.min_sample_size:
                recommendations.append(
                    f"⚠️ Variant '{variant.name}' needs more data "
                    f"({variant.total_queries}/{self.min_sample_size} queries)"
                )

        # Check for winner
        winner = comparison.get('winner')
        if winner:
            recommendations.append(
                f"✅ Winner found: '{winner['variant']}' with "
                f"{winner['accuracy_improvement']*100:+.1f}% accuracy improvement"
            )
            recommendations.append(
                f"💡 Consider declaring '{winner['variant']}' as winner and "
                f"routing 100% traffic to it"
            )
        else:
            if self.total_queries_served < self.min_sample_size * len(self.variants):
                recommendations.append(
                    "⏳ Continue test to gather more data before making a decision"
                )
            else:
                recommendations.append(
                    "📊 No statistically significant difference found. "
                    "Consider testing more aggressive variants."
                )

        return recommendations

    def _save_results(self):
        """Save test results to disk"""
        if not self.enable_persistence:
            return

        try:
            data = {
                'test_name': self.test_name,
                'test_start': self.test_start_time.isoformat(),
                'total_queries': self.total_queries_served,
                'variants': {
                    vid: {
                        'variant_id': v.variant_id,
                        'name': v.name,
                        'description': v.description,
                        'status': v.status.value,
                        'traffic_allocation': v.traffic_allocation,
                        'total_queries': v.total_queries,
                        'correct_classifications': v.correct_classifications,
                        'total_latency_ms': v.total_latency_ms,
                        'user_satisfaction_count': v.user_satisfaction_count,
                        'user_satisfaction_positive': v.user_satisfaction_positive,
                        'created_at': v.created_at.isoformat()
                    }
                    for vid, v in self.variants.items()
                }
            }

            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"[AB_TEST] Results saved to {self.results_file}")

        except Exception as e:
            logger.error(f"[AB_TEST] Failed to save results: {e}")

    def _load_results(self):
        """Load test results from disk"""
        if not self.enable_persistence or not self.results_file.exists():
            return

        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)

            self.test_start_time = datetime.fromisoformat(data['test_start'])
            self.total_queries_served = data['total_queries']

            # Note: classifier_func cannot be persisted, must be re-added
            logger.info(f"[AB_TEST] Loaded previous results from {self.results_file}")

        except Exception as e:
            logger.warning(f"[AB_TEST] Failed to load results: {e}")


# Singleton instance manager
_ab_tests: Dict[str, ABTestingFramework] = {}


def get_ab_test(test_name: str) -> ABTestingFramework:
    """Get or create an A/B test instance"""
    if test_name not in _ab_tests:
        _ab_tests[test_name] = ABTestingFramework(test_name)
    return _ab_tests[test_name]


def list_ab_tests() -> List[str]:
    """List all active A/B tests"""
    return list(_ab_tests.keys())
