#!/usr/bin/env python3
"""
Gradual Rollout System for Vision System v2.0
Manages safe and controlled deployment of new capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import random
from enum import Enum
from collections import defaultdict
import numpy as np

from .capability_generator import GeneratedCapability
from .safety_verification_framework import VerificationReport, RiskLevel

logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Stages of capability rollout"""
    DEVELOPMENT = "development"      # Internal testing only
    CANARY = "canary"               # Small percentage of users
    BETA = "beta"                   # Beta users only
    GRADUAL = "gradual"             # Gradual percentage increase
    PRODUCTION = "production"        # Full production


class RolloutStrategy(Enum):
    """Rollout strategies"""
    PERCENTAGE_BASED = "percentage"  # Random percentage of requests
    USER_GROUP = "user_group"        # Specific user groups
    FEATURE_FLAG = "feature_flag"    # Feature flag controlled
    CANARY_DEPLOYMENT = "canary"     # Canary deployment
    BLUE_GREEN = "blue_green"        # Blue-green deployment


@dataclass
class RolloutConfig:
    """Configuration for capability rollout"""
    capability_id: str
    capability_name: str
    
    # Rollout parameters
    strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE_BASED
    initial_percentage: float = 1.0    # Start with 1% of traffic
    increment_percentage: float = 5.0   # Increase by 5% each step
    max_percentage: float = 100.0
    
    # Timing parameters
    monitoring_period: timedelta = timedelta(hours=24)  # Monitor for 24h before increase
    rollback_threshold: timedelta = timedelta(hours=1)  # Quick rollback window
    
    # Success criteria
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    max_latency_ms: float = 100.0
    min_requests_for_decision: int = 1000
    
    # User groups (for USER_GROUP strategy)
    beta_users: Set[str] = field(default_factory=set)
    excluded_users: Set[str] = field(default_factory=set)
    
    # Safety parameters
    risk_tolerance: Dict[RiskLevel, float] = field(default_factory=lambda: {
        RiskLevel.LOW: 100.0,
        RiskLevel.MEDIUM: 50.0,
        RiskLevel.HIGH: 10.0,
        RiskLevel.CRITICAL: 0.0
    })


@dataclass
class RolloutMetrics:
    """Metrics collected during rollout"""
    requests_total: int = 0
    requests_served: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    
    # Latency tracking
    latencies: List[float] = field(default_factory=list)
    
    # Error tracking
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # User feedback
    positive_feedback: int = 0
    negative_feedback: int = 0
    
    # Resource usage
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Time tracking
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class RolloutState:
    """Current state of a capability rollout"""
    capability_id: str
    stage: RolloutStage
    current_percentage: float
    
    # Metrics for current stage
    current_metrics: RolloutMetrics
    
    # Historical metrics
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Decision tracking
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    rollback_count: int = 0
    
    # State flags
    is_paused: bool = False
    is_rolled_back: bool = False
    rollback_reason: Optional[str] = None


class RolloutDecisionEngine:
    """Makes decisions about rollout progression"""
    
    def __init__(self):
        self.decision_history = []
        
    def should_advance_rollout(
        self,
        state: RolloutState,
        config: RolloutConfig,
        verification_report: Optional[VerificationReport] = None
    ) -> Tuple[bool, str]:
        """Determine if rollout should advance to next stage"""
        metrics = state.current_metrics
        
        # Check if enough data collected
        if metrics.requests_total < config.min_requests_for_decision:
            return False, f"Insufficient data: {metrics.requests_total} requests"
        
        # Check monitoring period
        monitoring_duration = datetime.now() - metrics.start_time
        if monitoring_duration < config.monitoring_period:
            remaining = config.monitoring_period - monitoring_duration
            return False, f"Monitoring period not complete: {remaining} remaining"
        
        # Calculate success rate
        success_rate = metrics.requests_success / metrics.requests_served if metrics.requests_served > 0 else 0
        if success_rate < config.min_success_rate:
            return False, f"Success rate too low: {success_rate:.2%}"
        
        # Calculate error rate
        error_rate = metrics.requests_failed / metrics.requests_served if metrics.requests_served > 0 else 0
        if error_rate > config.max_error_rate:
            return False, f"Error rate too high: {error_rate:.2%}"
        
        # Check latency
        if metrics.latencies:
            p99_latency = np.percentile(metrics.latencies, 99)
            if p99_latency > config.max_latency_ms:
                return False, f"P99 latency too high: {p99_latency:.1f}ms"
        
        # Check user feedback
        total_feedback = metrics.positive_feedback + metrics.negative_feedback
        if total_feedback > 10:  # Minimum feedback threshold
            positive_ratio = metrics.positive_feedback / total_feedback
            if positive_ratio < 0.8:  # 80% positive threshold
                return False, f"User satisfaction too low: {positive_ratio:.1%}"
        
        # Check risk level if verification report available
        if verification_report:
            max_percentage = config.risk_tolerance.get(
                verification_report.risk_level,
                0.0
            )
            if state.current_percentage >= max_percentage:
                return False, f"Risk level {verification_report.risk_level.value} limits rollout to {max_percentage}%"
        
        # All checks passed
        return True, "All criteria met for advancement"
    
    def should_rollback(
        self,
        state: RolloutState,
        config: RolloutConfig,
        real_time_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Determine if rollout should be rolled back"""
        metrics = state.current_metrics
        
        # Quick rollback window check
        time_since_stage_start = datetime.now() - metrics.start_time
        if time_since_stage_start > config.rollback_threshold:
            # Outside quick rollback window, require more evidence
            threshold_multiplier = 2.0
        else:
            threshold_multiplier = 1.0
        
        # Check for sudden failure spike
        if metrics.requests_served > 100:
            recent_failures = sum(
                1 for _ in range(min(100, len(metrics.latencies)))
                if metrics.latencies and metrics.latencies[-1] > config.max_latency_ms * 2
            )
            if recent_failures > 20:  # 20% recent failure rate
                return True, "Sudden failure spike detected"
        
        # Check error rate with threshold
        if metrics.requests_served > 0:
            error_rate = metrics.requests_failed / metrics.requests_served
            if error_rate > config.max_error_rate * threshold_multiplier:
                return True, f"Error rate {error_rate:.2%} exceeds threshold"
        
        # Check for critical errors
        critical_errors = ['crash', 'timeout', 'memory_error', 'security_violation']
        for error_type in critical_errors:
            if metrics.errors.get(error_type, 0) > 5:
                return True, f"Critical error detected: {error_type}"
        
        # Check real-time metrics if available
        if real_time_metrics:
            if real_time_metrics.get('cpu_overload', False):
                return True, "CPU overload detected"
            if real_time_metrics.get('memory_leak', False):
                return True, "Memory leak detected"
        
        return False, "No rollback conditions met"
    
    def calculate_next_percentage(
        self,
        state: RolloutState,
        config: RolloutConfig
    ) -> float:
        """Calculate next rollout percentage"""
        if state.stage == RolloutStage.CANARY:
            return min(config.initial_percentage, config.max_percentage)
        
        elif state.stage == RolloutStage.BETA:
            return min(10.0, config.max_percentage)  # 10% for beta
        
        elif state.stage == RolloutStage.GRADUAL:
            # Gradual increase
            next_percentage = state.current_percentage + config.increment_percentage
            
            # Adjust based on performance
            metrics = state.current_metrics
            if metrics.requests_success > 0:
                success_rate = metrics.requests_success / metrics.requests_served
                if success_rate > 0.99:  # Excellent performance
                    next_percentage += config.increment_percentage  # Double increment
                elif success_rate < 0.97:  # Marginal performance  
                    next_percentage = state.current_percentage + (config.increment_percentage / 2)
            
            return min(next_percentage, config.max_percentage)
        
        return state.current_percentage


class TrafficRouter:
    """Routes traffic based on rollout configuration"""
    
    def __init__(self):
        self.routing_cache = {}
        self.user_assignments = {}  # Cache user -> capability assignments
        
    def should_use_capability(
        self,
        capability_id: str,
        state: RolloutState,
        config: RolloutConfig,
        user_id: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if request should use the new capability"""
        # Check if rollout is active
        if state.is_paused or state.is_rolled_back:
            return False
        
        # Check excluded users
        if user_id and user_id in config.excluded_users:
            return False
        
        # Apply strategy
        if config.strategy == RolloutStrategy.PERCENTAGE_BASED:
            return self._percentage_based_routing(state.current_percentage, user_id)
        
        elif config.strategy == RolloutStrategy.USER_GROUP:
            return self._user_group_routing(config, user_id)
        
        elif config.strategy == RolloutStrategy.FEATURE_FLAG:
            return self._feature_flag_routing(capability_id, request_context)
        
        elif config.strategy == RolloutStrategy.CANARY_DEPLOYMENT:
            return self._canary_routing(state, user_id)
        
        return False
    
    def _percentage_based_routing(
        self,
        percentage: float,
        user_id: Optional[str]
    ) -> bool:
        """Route based on percentage"""
        if user_id:
            # Consistent routing for same user
            if user_id not in self.user_assignments:
                self.user_assignments[user_id] = random.random() * 100
            return self.user_assignments[user_id] < percentage
        else:
            # Random routing for anonymous
            return random.random() * 100 < percentage
    
    def _user_group_routing(
        self,
        config: RolloutConfig,
        user_id: Optional[str]
    ) -> bool:
        """Route based on user groups"""
        if not user_id:
            return False
            
        # Check beta users
        if user_id in config.beta_users:
            return True
            
        # Could extend with other group logic
        return False
    
    def _feature_flag_routing(
        self,
        capability_id: str,
        request_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Route based on feature flags"""
        if not request_context:
            return False
            
        # Check for feature flag in context
        feature_flags = request_context.get('feature_flags', {})
        return feature_flags.get(f"capability_{capability_id}", False)
    
    def _canary_routing(
        self,
        state: RolloutState,
        user_id: Optional[str]
    ) -> bool:
        """Route canary traffic"""
        # Simple hash-based canary
        if user_id:
            user_hash = hash(user_id) % 100
            return user_hash < state.current_percentage
        return False


class RolloutMonitor:
    """Monitors rollout metrics and health"""
    
    def __init__(self):
        self.monitoring_tasks = {}
        self.alert_handlers = []
        
    async def monitor_rollout(
        self,
        capability_id: str,
        state: RolloutState,
        config: RolloutConfig
    ):
        """Monitor a rollout continuously"""
        logger.info(f"Starting rollout monitoring for {capability_id}")
        
        while not state.is_rolled_back and not state.stage == RolloutStage.PRODUCTION:
            try:
                # Collect real-time metrics
                real_time_metrics = await self._collect_real_time_metrics(capability_id)
                
                # Update state metrics
                state.current_metrics.last_update = datetime.now()
                
                # Check for issues
                issues = self._check_for_issues(state, config, real_time_metrics)
                
                if issues:
                    await self._handle_issues(capability_id, issues)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error for {capability_id}: {e}")
                await asyncio.sleep(60)
    
    async def _collect_real_time_metrics(
        self,
        capability_id: str
    ) -> Dict[str, Any]:
        """Collect real-time metrics"""
        # This would integrate with actual monitoring systems
        # For now, return mock metrics
        return {
            'cpu_usage': random.uniform(10, 50),
            'memory_usage': random.uniform(100, 500),
            'active_requests': random.randint(0, 100),
            'error_rate': random.uniform(0, 0.1)
        }
    
    def _check_for_issues(
        self,
        state: RolloutState,
        config: RolloutConfig,
        real_time_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for issues requiring attention"""
        issues = []
        
        # Check error rate
        metrics = state.current_metrics
        if metrics.requests_served > 0:
            error_rate = metrics.requests_failed / metrics.requests_served
            if error_rate > config.max_error_rate * 0.8:  # 80% of threshold
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'value': error_rate,
                    'threshold': config.max_error_rate
                })
        
        # Check latency
        if metrics.latencies and len(metrics.latencies) > 10:
            recent_latencies = metrics.latencies[-100:]
            p95_latency = np.percentile(recent_latencies, 95)
            if p95_latency > config.max_latency_ms * 0.9:  # 90% of threshold
                issues.append({
                    'type': 'high_latency',
                    'severity': 'warning',
                    'value': p95_latency,
                    'threshold': config.max_latency_ms
                })
        
        # Check real-time metrics
        if real_time_metrics.get('cpu_usage', 0) > 80:
            issues.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'value': real_time_metrics['cpu_usage']
            })
        
        return issues
    
    async def _handle_issues(
        self,
        capability_id: str,
        issues: List[Dict[str, Any]]
    ):
        """Handle detected issues"""
        for issue in issues:
            logger.warning(f"Issue detected for {capability_id}: {issue}")
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                await handler(capability_id, issue)


class GradualRolloutSystem:
    """Main system for managing gradual capability rollouts"""
    
    def __init__(self):
        self.rollouts: Dict[str, RolloutState] = {}
        self.configs: Dict[str, RolloutConfig] = {}
        self.decision_engine = RolloutDecisionEngine()
        self.traffic_router = TrafficRouter()
        self.monitor = RolloutMonitor()
        
        # Storage
        self.storage_path = Path("backend/data/rollouts")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Load existing rollouts
        self._load_rollouts()
    
    async def create_rollout(
        self,
        capability: GeneratedCapability,
        verification_report: VerificationReport,
        config: Optional[RolloutConfig] = None
    ) -> str:
        """Create a new rollout for a capability"""
        # Create default config if not provided
        if not config:
            config = RolloutConfig(
                capability_id=capability.capability_id,
                capability_name=capability.name
            )
        
        # Adjust config based on risk
        if verification_report.risk_level == RiskLevel.HIGH:
            config.initial_percentage = 0.1  # Start very small
            config.increment_percentage = 1.0  # Increase slowly
            
        # Create initial state
        state = RolloutState(
            capability_id=capability.capability_id,
            stage=RolloutStage.DEVELOPMENT,
            current_percentage=0.0,
            current_metrics=RolloutMetrics()
        )
        
        # Store
        self.configs[capability.capability_id] = config
        self.rollouts[capability.capability_id] = state
        
        # Save
        self._save_rollout(capability.capability_id)
        
        logger.info(f"Created rollout for {capability.name} ({capability.capability_id})")
        
        return capability.capability_id
    
    async def advance_rollout(
        self,
        capability_id: str,
        force: bool = False
    ) -> Tuple[bool, str]:
        """Advance rollout to next stage"""
        if capability_id not in self.rollouts:
            return False, "Rollout not found"
        
        state = self.rollouts[capability_id]
        config = self.configs[capability_id]
        
        # Check if advancement is allowed
        if not force:
            can_advance, reason = self.decision_engine.should_advance_rollout(
                state, config
            )
            if not can_advance:
                return False, f"Cannot advance: {reason}"
        
        # Record current stage metrics
        state.stage_history.append({
            'stage': state.stage.value,
            'percentage': state.current_percentage,
            'metrics': {
                'requests': state.current_metrics.requests_total,
                'success_rate': state.current_metrics.requests_success / 
                               max(1, state.current_metrics.requests_served),
                'duration': (datetime.now() - state.current_metrics.start_time).total_seconds()
            },
            'timestamp': datetime.now().isoformat()
        })
        
        # Determine next stage
        if state.stage == RolloutStage.DEVELOPMENT:
            state.stage = RolloutStage.CANARY
            state.current_percentage = config.initial_percentage
            
        elif state.stage == RolloutStage.CANARY:
            state.stage = RolloutStage.BETA
            state.current_percentage = 10.0  # 10% for beta
            
        elif state.stage == RolloutStage.BETA:
            state.stage = RolloutStage.GRADUAL
            state.current_percentage = 25.0  # Start gradual at 25%
            
        elif state.stage == RolloutStage.GRADUAL:
            # Calculate next percentage
            next_percentage = self.decision_engine.calculate_next_percentage(state, config)
            
            if next_percentage >= config.max_percentage:
                state.stage = RolloutStage.PRODUCTION
                state.current_percentage = 100.0
            else:
                state.current_percentage = next_percentage
        
        # Reset metrics for new stage
        state.current_metrics = RolloutMetrics()
        
        # Record decision
        state.decisions.append({
            'action': 'advance',
            'from_stage': state.stage_history[-1]['stage'] if state.stage_history else 'none',
            'to_stage': state.stage.value,
            'percentage': state.current_percentage,
            'timestamp': datetime.now().isoformat(),
            'forced': force
        })
        
        # Start monitoring if not in production
        if state.stage != RolloutStage.PRODUCTION:
            asyncio.create_task(self.monitor.monitor_rollout(
                capability_id, state, config
            ))
        
        # Save state
        self._save_rollout(capability_id)
        
        logger.info(f"Advanced {capability_id} to {state.stage.value} at {state.current_percentage}%")
        
        return True, f"Advanced to {state.stage.value}"
    
    async def rollback_capability(
        self,
        capability_id: str,
        reason: str
    ) -> bool:
        """Rollback a capability"""
        if capability_id not in self.rollouts:
            return False
        
        state = self.rollouts[capability_id]
        
        # Mark as rolled back
        state.is_rolled_back = True
        state.rollback_reason = reason
        state.rollback_count += 1
        state.current_percentage = 0.0
        
        # Record decision
        state.decisions.append({
            'action': 'rollback',
            'stage': state.stage.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save state
        self._save_rollout(capability_id)
        
        logger.warning(f"Rolled back {capability_id}: {reason}")
        
        return True
    
    def should_use_capability(
        self,
        capability_id: str,
        user_id: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a request should use the capability"""
        if capability_id not in self.rollouts:
            return False
        
        state = self.rollouts[capability_id]
        config = self.configs[capability_id]
        
        # Record request
        state.current_metrics.requests_total += 1
        
        # Check routing decision
        should_use = self.traffic_router.should_use_capability(
            capability_id, state, config, user_id, request_context
        )
        
        if should_use:
            state.current_metrics.requests_served += 1
        
        return should_use
    
    def record_result(
        self,
        capability_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        user_feedback: Optional[str] = None
    ):
        """Record the result of using a capability"""
        if capability_id not in self.rollouts:
            return
        
        state = self.rollouts[capability_id]
        metrics = state.current_metrics
        
        # Update metrics
        if success:
            metrics.requests_success += 1
        else:
            metrics.requests_failed += 1
            if error:
                metrics.errors[error] += 1
        
        if latency_ms is not None:
            metrics.latencies.append(latency_ms)
        
        # Record feedback
        if user_feedback:
            if user_feedback.lower() in ['good', 'great', 'excellent', 'positive']:
                metrics.positive_feedback += 1
            elif user_feedback.lower() in ['bad', 'poor', 'negative', 'terrible']:
                metrics.negative_feedback += 1
        
        # Check for auto-rollback conditions
        should_rollback, rollback_reason = self.decision_engine.should_rollback(
            state, self.configs[capability_id]
        )
        
        if should_rollback:
            asyncio.create_task(self.rollback_capability(
                capability_id, rollback_reason
            ))
    
    def get_rollout_status(
        self,
        capability_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get rollout status"""
        if capability_id:
            if capability_id not in self.rollouts:
                return {'error': 'Rollout not found'}
            
            state = self.rollouts[capability_id]
            config = self.configs[capability_id]
            metrics = state.current_metrics
            
            return {
                'capability_id': capability_id,
                'stage': state.stage.value,
                'percentage': state.current_percentage,
                'is_active': not state.is_paused and not state.is_rolled_back,
                'metrics': {
                    'requests_total': metrics.requests_total,
                    'requests_served': metrics.requests_served,
                    'success_rate': metrics.requests_success / max(1, metrics.requests_served),
                    'avg_latency_ms': np.mean(metrics.latencies) if metrics.latencies else 0,
                    'error_types': dict(metrics.errors)
                },
                'rollback_count': state.rollback_count,
                'stage_history': state.stage_history
            }
        
        # Return summary of all rollouts
        summary = {
            'total_rollouts': len(self.rollouts),
            'by_stage': defaultdict(int),
            'active': 0,
            'rolled_back': 0
        }
        
        for state in self.rollouts.values():
            summary['by_stage'][state.stage.value] += 1
            if not state.is_paused and not state.is_rolled_back:
                summary['active'] += 1
            if state.is_rolled_back:
                summary['rolled_back'] += 1
        
        return summary
    
    def _save_rollout(self, capability_id: str):
        """Save rollout state to disk"""
        state = self.rollouts[capability_id]
        config = self.configs[capability_id]
        
        data = {
            'config': {
                'capability_id': config.capability_id,
                'capability_name': config.capability_name,
                'strategy': config.strategy.value,
                'initial_percentage': config.initial_percentage,
                'increment_percentage': config.increment_percentage,
                'max_percentage': config.max_percentage
            },
            'state': {
                'stage': state.stage.value,
                'current_percentage': state.current_percentage,
                'is_paused': state.is_paused,
                'is_rolled_back': state.is_rolled_back,
                'rollback_reason': state.rollback_reason,
                'rollback_count': state.rollback_count,
                'stage_history': state.stage_history,
                'decisions': state.decisions
            }
        }
        
        file_path = self.storage_path / f"{capability_id}_rollout.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_rollouts(self):
        """Load existing rollouts from disk"""
        for file_path in self.storage_path.glob("*_rollout.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Recreate config
                config = RolloutConfig(
                    capability_id=data['config']['capability_id'],
                    capability_name=data['config']['capability_name'],
                    strategy=RolloutStrategy(data['config']['strategy']),
                    initial_percentage=data['config']['initial_percentage'],
                    increment_percentage=data['config']['increment_percentage'],
                    max_percentage=data['config']['max_percentage']
                )
                
                # Recreate state
                state = RolloutState(
                    capability_id=config.capability_id,
                    stage=RolloutStage(data['state']['stage']),
                    current_percentage=data['state']['current_percentage'],
                    current_metrics=RolloutMetrics(),
                    stage_history=data['state'].get('stage_history', []),
                    decisions=data['state'].get('decisions', []),
                    is_paused=data['state'].get('is_paused', False),
                    is_rolled_back=data['state'].get('is_rolled_back', False),
                    rollback_reason=data['state'].get('rollback_reason'),
                    rollback_count=data['state'].get('rollback_count', 0)
                )
                
                self.configs[config.capability_id] = config
                self.rollouts[config.capability_id] = state
                
            except Exception as e:
                logger.error(f"Failed to load rollout from {file_path}: {e}")


# Singleton instance
_rollout_system: Optional[GradualRolloutSystem] = None


def get_gradual_rollout_system() -> GradualRolloutSystem:
    """Get singleton instance of gradual rollout system"""
    global _rollout_system
    if _rollout_system is None:
        _rollout_system = GradualRolloutSystem()
    return _rollout_system