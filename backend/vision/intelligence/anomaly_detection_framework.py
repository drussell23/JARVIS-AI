#!/usr/bin/env python3
"""
Anomaly Detection Framework - Proactive Intelligence System Component
Identifies unusual situations requiring intervention with zero hardcoding
Implements multi-layer detection: Visual, Behavioral, and System anomalies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import asyncio
from pathlib import Path
import statistics
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    # Visual Anomalies
    UNEXPECTED_POPUP = "unexpected_popup"
    ERROR_DIALOG = "error_dialog"
    UNUSUAL_LAYOUT = "unusual_layout"
    MISSING_ELEMENTS = "missing_elements"
    PERFORMANCE_VISUAL = "performance_visual"
    
    # Behavioral Anomalies
    REPEATED_FAILED_ATTEMPTS = "repeated_failed_attempts"
    UNUSUAL_NAVIGATION = "unusual_navigation"
    STUCK_STATE = "stuck_state"
    CIRCULAR_PATTERN = "circular_pattern"
    TIME_ANOMALY = "time_anomaly"
    
    # System Anomalies
    RESOURCE_WARNING = "resource_warning"
    NETWORK_ISSUE = "network_issue"
    PERMISSION_PROBLEM = "permission_problem"
    CRASH_INDICATOR = "crash_indicator"
    DATA_INCONSISTENCY = "data_inconsistency"


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AnomalyBaseline:
    """Baseline for normal behavior"""
    baseline_id: str
    category: str  # visual, behavioral, system
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    threshold_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'low': 2.0,
        'medium': 3.0,
        'high': 4.0,
        'critical': 5.0
    })
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


@dataclass
class Observation:
    """Observation data for anomaly detection"""
    timestamp: datetime
    observation_type: str  # screenshot_analysis, manual_screenshot, user_action, system_event
    data: Dict[str, Any]  # The actual observation data
    source: str  # Source of the observation (e.g., 'claude_vision_analyzer', 'manual_detection')
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def extract_features(self) -> Dict[str, float]:
        """Extract numerical features from observation for anomaly detection"""
        features = {}
        
        # Extract basic features
        features['has_error'] = 1.0 if self.metadata.get('has_error', False) else 0.0
        features['has_warning'] = 1.0 if self.metadata.get('has_warning', False) else 0.0
        features['confidence'] = float(self.metadata.get('confidence', 0.5))
        
        # Extract data-specific features
        if isinstance(self.data, dict):
            # Count entities
            entities = self.data.get('entities', {})
            features['entity_count'] = float(len(entities))
            
            # Count actions
            actions = self.data.get('actions', [])
            features['action_count'] = float(len(actions))
            
            # Text analysis features
            if 'analysis' in self.data:
                analysis_text = str(self.data['analysis'])
                features['text_length'] = float(len(analysis_text))
                features['error_mentions'] = float(analysis_text.lower().count('error'))
                features['warning_mentions'] = float(analysis_text.lower().count('warning'))
                features['failed_mentions'] = float(analysis_text.lower().count('fail'))
            
            # App-specific features
            features['is_known_app'] = 1.0 if self.data.get('app_id', 'unknown') != 'unknown' else 0.0
        
        return features


@dataclass
class DetectionRule:
    """Anomaly detection rule"""
    rule_id: str
    rule_type: str  # threshold, pattern, model
    condition: Callable[[Dict[str, Any]], bool]
    severity_calculator: Callable[[Dict[str, Any]], AnomalySeverity]
    anomaly_type: AnomalyType
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedAnomaly:
    """Detected anomaly instance"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    description: str
    evidence: Dict[str, Any]
    confidence: float
    suggested_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class AnomalyDetectionFramework:
    """Main anomaly detection framework"""
    
    def __init__(self, memory_allocation: Dict[str, int] = None):
        """Initialize with memory allocation"""
        # Default memory allocation (70MB total)
        self.memory_allocation = memory_allocation or {
            'baseline_models': 30 * 1024 * 1024,  # 30MB
            'detection_rules': 20 * 1024 * 1024,  # 20MB
            'anomaly_history': 20 * 1024 * 1024   # 20MB
        }
        
        # Baselines for different categories
        self.baselines: Dict[str, AnomalyBaseline] = {}
        
        # Detection rules
        self.detection_rules: List[DetectionRule] = []
        self._initialize_default_rules()
        
        # Anomaly history
        self.anomaly_history: deque = deque(maxlen=1000)
        self.active_anomalies: Dict[str, DetectedAnomaly] = {}
        
        # Feature extractors for different types
        self.feature_extractors = {
            'visual': self._extract_visual_features,
            'behavioral': self._extract_behavioral_features,
            'system': self._extract_system_features
        }
        
        # Statistical models for anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Recent observations for pattern detection
        self.recent_observations = {
            'visual': deque(maxlen=100),
            'behavioral': deque(maxlen=100),
            'system': deque(maxlen=100)
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_checks': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'processing_times': deque(maxlen=100)
        }
        
        logger.info("Initialized Anomaly Detection Framework")
    
    def _initialize_default_rules(self):
        """Initialize default detection rules"""
        # Visual anomaly rules
        self.detection_rules.extend([
            DetectionRule(
                rule_id="popup_detector",
                rule_type="pattern",
                condition=self._check_unexpected_popup,
                severity_calculator=self._calculate_popup_severity,
                anomaly_type=AnomalyType.UNEXPECTED_POPUP
            ),
            DetectionRule(
                rule_id="error_dialog_detector",
                rule_type="pattern",
                condition=self._check_error_dialog,
                severity_calculator=lambda x: AnomalySeverity.HIGH,
                anomaly_type=AnomalyType.ERROR_DIALOG
            ),
            DetectionRule(
                rule_id="layout_anomaly_detector",
                rule_type="threshold",
                condition=self._check_unusual_layout,
                severity_calculator=self._calculate_layout_severity,
                anomaly_type=AnomalyType.UNUSUAL_LAYOUT
            ),
        ])
        
        # Behavioral anomaly rules
        self.detection_rules.extend([
            DetectionRule(
                rule_id="repeated_failure_detector",
                rule_type="pattern",
                condition=self._check_repeated_failures,
                severity_calculator=self._calculate_failure_severity,
                anomaly_type=AnomalyType.REPEATED_FAILED_ATTEMPTS
            ),
            DetectionRule(
                rule_id="stuck_state_detector",
                rule_type="threshold",
                condition=self._check_stuck_state,
                severity_calculator=lambda x: AnomalySeverity.MEDIUM,
                anomaly_type=AnomalyType.STUCK_STATE
            ),
            DetectionRule(
                rule_id="circular_pattern_detector",
                rule_type="pattern",
                condition=self._check_circular_pattern,
                severity_calculator=lambda x: AnomalySeverity.LOW,
                anomaly_type=AnomalyType.CIRCULAR_PATTERN
            ),
        ])
        
        # System anomaly rules
        self.detection_rules.extend([
            DetectionRule(
                rule_id="resource_warning_detector",
                rule_type="threshold",
                condition=self._check_resource_warning,
                severity_calculator=self._calculate_resource_severity,
                anomaly_type=AnomalyType.RESOURCE_WARNING
            ),
            DetectionRule(
                rule_id="crash_indicator_detector",
                rule_type="pattern",
                condition=self._check_crash_indicators,
                severity_calculator=lambda x: AnomalySeverity.CRITICAL,
                anomaly_type=AnomalyType.CRASH_INDICATOR
            ),
        ])
    
    async def establish_baseline(self, observations: List[Dict[str, Any]], 
                               category: str) -> AnomalyBaseline:
        """Establish baseline from normal observations"""
        if not observations:
            raise ValueError("Need observations to establish baseline")
        
        # Extract features from observations
        features_list = []
        for obs in observations:
            features = self.feature_extractors[category](obs)
            features_list.append(features)
        
        # Calculate statistics
        feature_names = list(features_list[0].keys())
        feature_means = {}
        feature_stds = {}
        
        for feature_name in feature_names:
            values = [f[feature_name] for f in features_list]
            feature_means[feature_name] = statistics.mean(values)
            feature_stds[feature_name] = statistics.stdev(values) if len(values) > 1 else 0.1
        
        # Create baseline
        baseline = AnomalyBaseline(
            baseline_id=f"{category}_baseline_{datetime.now().timestamp()}",
            category=category,
            feature_means=feature_means,
            feature_stds=feature_stds,
            sample_count=len(observations),
            confidence=min(len(observations) / 100, 1.0)  # More samples = higher confidence
        )
        
        # Store baseline
        self.baselines[category] = baseline
        
        # Train isolation forest if enough data
        if len(features_list) >= 20:
            feature_matrix = np.array([[f[fn] for fn in feature_names] 
                                     for f in features_list])
            self.scaler.fit(feature_matrix)
            scaled_features = self.scaler.transform(feature_matrix)
            self.isolation_forest.fit(scaled_features)
        
        logger.info(f"Established baseline for {category} with {len(observations)} samples")
        return baseline
    
    async def monitor_realtime(self, observation: Dict[str, Any]) -> List[DetectedAnomaly]:
        """Real-time anomaly monitoring"""
        start_time = datetime.now()
        detected_anomalies = []
        
        # Determine observation category
        category = observation.get('category', 'visual')
        
        # Store in recent observations
        self.recent_observations[category].append(observation)
        
        # Extract features
        features = self.feature_extractors[category](observation)
        
        # Check against baseline if available
        if category in self.baselines:
            baseline_anomalies = self._check_baseline_deviation(
                features, self.baselines[category], observation
            )
            detected_anomalies.extend(baseline_anomalies)
        
        # Check detection rules
        for rule in self.detection_rules:
            if rule.enabled and rule.condition(observation):
                anomaly = DetectedAnomaly(
                    anomaly_id=f"anomaly_{datetime.now().timestamp()}",
                    anomaly_type=rule.anomaly_type,
                    severity=rule.severity_calculator(observation),
                    timestamp=datetime.now(),
                    description=self._generate_anomaly_description(
                        rule.anomaly_type, observation
                    ),
                    evidence=observation,
                    confidence=self._calculate_confidence(rule, observation),
                    suggested_actions=self._suggest_actions(rule.anomaly_type)
                )
                detected_anomalies.append(anomaly)
        
        # Use ML model if trained
        if hasattr(self.isolation_forest, 'n_estimators'):
            ml_anomalies = await self._detect_ml_anomalies(features, observation)
            detected_anomalies.extend(ml_anomalies)
        
        # Update statistics
        self.detection_stats['total_checks'] += 1
        self.detection_stats['anomalies_detected'] += len(detected_anomalies)
        self.detection_stats['processing_times'].append(
            (datetime.now() - start_time).total_seconds()
        )
        
        # Store anomalies
        for anomaly in detected_anomalies:
            self.anomaly_history.append(anomaly)
            self.active_anomalies[anomaly.anomaly_id] = anomaly
        
        return detected_anomalies
    
    def _check_baseline_deviation(self, features: Dict[str, float], 
                                baseline: AnomalyBaseline,
                                observation: Dict[str, Any]) -> List[DetectedAnomaly]:
        """Check deviation from baseline"""
        anomalies = []
        
        for feature_name, value in features.items():
            if feature_name not in baseline.feature_means:
                continue
            
            mean = baseline.feature_means[feature_name]
            std = baseline.feature_stds[feature_name]
            
            # Calculate z-score
            z_score = abs(value - mean) / (std + 0.001)  # Avoid division by zero
            
            # Check against thresholds
            for severity_name, threshold in baseline.threshold_multipliers.items():
                if z_score > threshold:
                    severity = AnomalySeverity[severity_name.upper()]
                    
                    anomaly = DetectedAnomaly(
                        anomaly_id=f"baseline_anomaly_{datetime.now().timestamp()}",
                        anomaly_type=self._infer_anomaly_type(feature_name, observation),
                        severity=severity,
                        timestamp=datetime.now(),
                        description=f"Feature '{feature_name}' deviates significantly from baseline "
                                  f"(z-score: {z_score:.2f})",
                        evidence={
                            'feature': feature_name,
                            'value': value,
                            'baseline_mean': mean,
                            'baseline_std': std,
                            'z_score': z_score
                        },
                        confidence=baseline.confidence * min(z_score / 10, 1.0),
                        suggested_actions=["Review baseline", "Check for environmental changes"]
                    )
                    anomalies.append(anomaly)
                    break
        
        return anomalies
    
    async def _detect_ml_anomalies(self, features: Dict[str, float], 
                                 observation: Dict[str, Any]) -> List[DetectedAnomaly]:
        """Detect anomalies using ML model"""
        anomalies = []
        
        try:
            # Prepare features
            feature_names = sorted(features.keys())
            feature_vector = np.array([[features[fn] for fn in feature_names]])
            
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.isolation_forest.predict(scaled_features)
            anomaly_score = self.isolation_forest.score_samples(scaled_features)[0]
            
            if prediction[0] == -1:  # Anomaly detected
                severity = self._score_to_severity(anomaly_score)
                
                anomaly = DetectedAnomaly(
                    anomaly_id=f"ml_anomaly_{datetime.now().timestamp()}",
                    anomaly_type=self._infer_anomaly_type_ml(features, observation),
                    severity=severity,
                    timestamp=datetime.now(),
                    description="Machine learning model detected unusual pattern",
                    evidence={
                        'anomaly_score': float(anomaly_score),
                        'features': features,
                        'observation': observation
                    },
                    confidence=abs(anomaly_score),
                    suggested_actions=["Investigate unusual pattern", "Compare with recent activity"]
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
        
        return anomalies
    
    def _extract_visual_features(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from visual observations"""
        features = {}
        
        # Layout features
        if 'layout' in observation:
            layout = observation['layout']
            features['element_count'] = len(layout.get('elements', []))
            features['layout_complexity'] = self._calculate_layout_complexity(layout)
            features['alignment_score'] = self._calculate_alignment_score(layout)
        
        # Color features
        if 'colors' in observation:
            features['color_variance'] = self._calculate_color_variance(observation['colors'])
            features['dominant_hue'] = observation['colors'].get('dominant_hue', 0)
        
        # Text features
        if 'text' in observation:
            features['text_density'] = len(observation['text'].get('content', '')) / 1000
            features['error_keywords'] = self._count_error_keywords(observation['text'])
        
        # Window features
        if 'window' in observation:
            features['window_count'] = len(observation['window'].get('windows', []))
            features['modal_present'] = 1.0 if observation['window'].get('has_modal') else 0.0
        
        return features
    
    def _extract_behavioral_features(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from behavioral observations"""
        features = {}
        
        # Action features
        if 'action_sequence' in observation:
            actions = observation['action_sequence']
            features['action_count'] = len(actions)
            features['unique_actions'] = len(set(actions))
            features['repetition_score'] = self._calculate_repetition_score(actions)
        
        # Timing features
        if 'timing' in observation:
            features['duration_seconds'] = observation['timing'].get('duration', 0)
            features['action_frequency'] = observation['timing'].get('actions_per_minute', 0)
            features['idle_ratio'] = observation['timing'].get('idle_ratio', 0)
        
        # Navigation features
        if 'navigation' in observation:
            features['back_count'] = observation['navigation'].get('back_count', 0)
            features['loop_score'] = observation['navigation'].get('loop_score', 0)
            features['dead_end_score'] = observation['navigation'].get('dead_end_score', 0)
        
        return features
    
    def _extract_system_features(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from system observations"""
        features = {}
        
        # Resource features
        if 'resources' in observation:
            features['cpu_usage'] = observation['resources'].get('cpu_percent', 0)
            features['memory_usage'] = observation['resources'].get('memory_percent', 0)
            features['disk_io'] = observation['resources'].get('disk_io_rate', 0)
        
        # Network features
        if 'network' in observation:
            features['latency_ms'] = observation['network'].get('latency', 0)
            features['packet_loss'] = observation['network'].get('packet_loss', 0)
            features['connection_errors'] = observation['network'].get('errors', 0)
        
        # Process features
        if 'processes' in observation:
            features['process_count'] = len(observation['processes'].get('active', []))
            features['crashed_count'] = observation['processes'].get('crashed', 0)
            features['unresponsive_count'] = observation['processes'].get('unresponsive', 0)
        
        return features
    
    # Detection condition methods
    def _check_unexpected_popup(self, observation: Dict[str, Any]) -> bool:
        """Check for unexpected popup"""
        if 'window' not in observation:
            return False
        
        window_data = observation['window']
        
        # Check for new modal/dialog
        if window_data.get('has_modal') and window_data.get('modal_unexpected'):
            return True
        
        # Check for popup characteristics
        if 'new_windows' in window_data:
            for window in window_data['new_windows']:
                if (window.get('type') in ['popup', 'dialog', 'alert'] and
                    not window.get('expected')):
                    return True
        
        return False
    
    def _check_error_dialog(self, observation: Dict[str, Any]) -> bool:
        """Check for error dialog"""
        if 'text' not in observation:
            return False
        
        error_keywords = ['error', 'failed', 'exception', 'crash', 'unable', 
                         'invalid', 'denied', 'refused', 'timeout']
        
        text_content = observation['text'].get('content', '').lower()
        
        # Check for error keywords in prominent text
        if any(keyword in text_content for keyword in error_keywords):
            # Verify it's in a dialog/modal context
            if observation.get('window', {}).get('has_modal'):
                return True
            
            # Check for error dialog visual patterns
            if observation.get('visual_patterns', {}).get('error_dialog_confidence', 0) > 0.7:
                return True
        
        return False
    
    def _check_unusual_layout(self, observation: Dict[str, Any]) -> bool:
        """Check for unusual layout"""
        if 'layout' not in observation:
            return False
        
        layout = observation['layout']
        
        # Check layout anomaly score
        if layout.get('anomaly_score', 0) > 0.8:
            return True
        
        # Check for missing expected elements
        if layout.get('missing_elements'):
            return True
        
        # Check for overlapping elements
        if layout.get('overlap_count', 0) > 3:
            return True
        
        return False
    
    def _check_repeated_failures(self, observation: Dict[str, Any]) -> bool:
        """Check for repeated failed attempts"""
        if 'action_sequence' not in observation:
            return False
        
        # Look for failure patterns in recent actions
        recent_actions = list(self.recent_observations['behavioral'])[-10:]
        
        failure_count = 0
        for obs in recent_actions:
            if obs.get('outcome') == 'failure' or obs.get('success') == False:
                failure_count += 1
        
        return failure_count >= 3
    
    def _check_stuck_state(self, observation: Dict[str, Any]) -> bool:
        """Check if stuck in same state"""
        if 'state' not in observation:
            return False
        
        current_state = observation['state'].get('state_id')
        if not current_state:
            return False
        
        # Check last N observations
        recent_states = []
        for obs in list(self.recent_observations['behavioral'])[-10:]:
            if 'state' in obs:
                recent_states.append(obs['state'].get('state_id'))
        
        # Stuck if in same state for 5+ observations
        if len(recent_states) >= 5 and all(s == current_state for s in recent_states[-5:]):
            return True
        
        # Also check duration in state
        if observation.get('state', {}).get('duration_seconds', 0) > 300:  # 5 minutes
            return True
        
        return False
    
    def _check_circular_pattern(self, observation: Dict[str, Any]) -> bool:
        """Check for circular navigation pattern"""
        if 'navigation' not in observation:
            return False
        
        # Check explicit loop score
        if observation['navigation'].get('loop_score', 0) > 0.8:
            return True
        
        # Check state sequence for cycles
        if 'state_sequence' in observation:
            sequence = observation['state_sequence']
            if len(sequence) >= 4:
                # Simple cycle detection
                for i in range(len(sequence) - 3):
                    if sequence[i] == sequence[i+2] and sequence[i+1] == sequence[i+3]:
                        return True
        
        return False
    
    def _check_resource_warning(self, observation: Dict[str, Any]) -> bool:
        """Check for resource warnings"""
        if 'resources' not in observation:
            return False
        
        resources = observation['resources']
        
        # High CPU usage
        if resources.get('cpu_percent', 0) > 90:
            return True
        
        # High memory usage
        if resources.get('memory_percent', 0) > 85:
            return True
        
        # Low disk space
        if resources.get('disk_free_gb', float('inf')) < 1:
            return True
        
        return False
    
    def _check_crash_indicators(self, observation: Dict[str, Any]) -> bool:
        """Check for crash indicators"""
        indicators = []
        
        # Check process crashes
        if observation.get('processes', {}).get('crashed', 0) > 0:
            indicators.append('process_crash')
        
        # Check for crash-related text
        if 'text' in observation:
            crash_keywords = ['crashed', 'quit unexpectedly', 'stopped working', 
                            'not responding', 'force quit']
            text = observation['text'].get('content', '').lower()
            if any(kw in text for kw in crash_keywords):
                indicators.append('crash_text')
        
        # Check for sudden state changes
        if observation.get('state_transition', {}).get('unexpected'):
            indicators.append('unexpected_transition')
        
        return len(indicators) >= 2
    
    # Severity calculation methods
    def _calculate_popup_severity(self, observation: Dict[str, Any]) -> AnomalySeverity:
        """Calculate severity for popup anomaly"""
        window_data = observation.get('window', {})
        
        # Modal dialogs are higher severity
        if window_data.get('has_modal'):
            return AnomalySeverity.MEDIUM
        
        # Multiple popups are higher severity
        if len(window_data.get('new_windows', [])) > 2:
            return AnomalySeverity.HIGH
        
        return AnomalySeverity.LOW
    
    def _calculate_layout_severity(self, observation: Dict[str, Any]) -> AnomalySeverity:
        """Calculate severity for layout anomaly"""
        layout = observation.get('layout', {})
        
        # Many missing elements is critical
        if len(layout.get('missing_elements', [])) > 5:
            return AnomalySeverity.HIGH
        
        # High overlap is medium
        if layout.get('overlap_count', 0) > 5:
            return AnomalySeverity.MEDIUM
        
        return AnomalySeverity.LOW
    
    def _calculate_failure_severity(self, observation: Dict[str, Any]) -> AnomalySeverity:
        """Calculate severity for repeated failures"""
        # Count consecutive failures
        consecutive = 0
        for obs in reversed(list(self.recent_observations['behavioral'])):
            if obs.get('outcome') == 'failure':
                consecutive += 1
            else:
                break
        
        if consecutive >= 5:
            return AnomalySeverity.HIGH
        elif consecutive >= 3:
            return AnomalySeverity.MEDIUM
        
        return AnomalySeverity.LOW
    
    def _calculate_resource_severity(self, observation: Dict[str, Any]) -> AnomalySeverity:
        """Calculate severity for resource warnings"""
        resources = observation.get('resources', {})
        
        critical_conditions = 0
        
        if resources.get('cpu_percent', 0) > 95:
            critical_conditions += 1
        if resources.get('memory_percent', 0) > 90:
            critical_conditions += 1
        if resources.get('disk_free_gb', float('inf')) < 0.5:
            critical_conditions += 1
        
        if critical_conditions >= 2:
            return AnomalySeverity.CRITICAL
        elif critical_conditions == 1:
            return AnomalySeverity.HIGH
        
        return AnomalySeverity.MEDIUM
    
    # Helper methods
    def _calculate_layout_complexity(self, layout: Dict[str, Any]) -> float:
        """Calculate layout complexity score"""
        elements = layout.get('elements', [])
        if not elements:
            return 0.0
        
        # Consider number of elements and nesting depth
        complexity = len(elements) / 10  # Normalize
        
        # Add nesting depth
        max_depth = max(e.get('depth', 0) for e in elements) if elements else 0
        complexity += max_depth / 5
        
        return min(complexity, 10.0)
    
    def _calculate_alignment_score(self, layout: Dict[str, Any]) -> float:
        """Calculate how well-aligned elements are"""
        elements = layout.get('elements', [])
        if len(elements) < 2:
            return 1.0
        
        # Simple alignment check - in production use proper geometric analysis
        x_positions = [e.get('x', 0) for e in elements]
        y_positions = [e.get('y', 0) for e in elements]
        
        # Check for common x/y values (indicates alignment)
        x_alignment = len(set(x_positions)) / len(x_positions)
        y_alignment = len(set(y_positions)) / len(y_positions)
        
        return 1.0 - (x_alignment + y_alignment) / 2
    
    def _calculate_color_variance(self, colors: Dict[str, Any]) -> float:
        """Calculate color variance in the observation"""
        if 'histogram' in colors:
            hist = np.array(colors['histogram'])
            return float(np.std(hist))
        
        if 'palette' in colors:
            return len(colors['palette']) / 10  # Normalize
        
        return 0.0
    
    def _count_error_keywords(self, text_data: Dict[str, Any]) -> float:
        """Count error-related keywords"""
        content = text_data.get('content', '').lower()
        
        error_keywords = ['error', 'warning', 'fail', 'exception', 'invalid', 
                         'unable', 'cannot', 'denied', 'refused']
        
        count = sum(1 for keyword in error_keywords if keyword in content)
        return float(count)
    
    def _calculate_repetition_score(self, actions: List[str]) -> float:
        """Calculate action repetition score"""
        if len(actions) < 2:
            return 0.0
        
        # Count consecutive repetitions
        repetitions = 0
        for i in range(1, len(actions)):
            if actions[i] == actions[i-1]:
                repetitions += 1
        
        return repetitions / (len(actions) - 1)
    
    def _infer_anomaly_type(self, feature_name: str, 
                           observation: Dict[str, Any]) -> AnomalyType:
        """Infer anomaly type from feature and observation"""
        # Visual features
        if feature_name in ['element_count', 'layout_complexity', 'alignment_score']:
            return AnomalyType.UNUSUAL_LAYOUT
        elif feature_name in ['color_variance', 'dominant_hue']:
            return AnomalyType.PERFORMANCE_VISUAL
        elif feature_name in ['modal_present', 'window_count']:
            return AnomalyType.UNEXPECTED_POPUP
        
        # Behavioral features
        elif feature_name in ['repetition_score', 'loop_score']:
            return AnomalyType.CIRCULAR_PATTERN
        elif feature_name in ['idle_ratio', 'duration_seconds']:
            return AnomalyType.STUCK_STATE
        elif feature_name in ['back_count', 'dead_end_score']:
            return AnomalyType.UNUSUAL_NAVIGATION
        
        # System features
        elif feature_name in ['cpu_usage', 'memory_usage']:
            return AnomalyType.RESOURCE_WARNING
        elif feature_name in ['latency_ms', 'packet_loss']:
            return AnomalyType.NETWORK_ISSUE
        elif feature_name in ['crashed_count', 'unresponsive_count']:
            return AnomalyType.CRASH_INDICATOR
        
        # Default
        return AnomalyType.DATA_INCONSISTENCY
    
    def _infer_anomaly_type_ml(self, features: Dict[str, float], 
                              observation: Dict[str, Any]) -> AnomalyType:
        """Infer anomaly type from ML detection"""
        # Find the most anomalous feature
        if 'baseline' in observation:
            baseline = observation['baseline']
            max_deviation = 0
            anomalous_feature = None
            
            for feature, value in features.items():
                if feature in baseline:
                    deviation = abs(value - baseline[feature])
                    if deviation > max_deviation:
                        max_deviation = deviation
                        anomalous_feature = feature
            
            if anomalous_feature:
                return self._infer_anomaly_type(anomalous_feature, observation)
        
        # Default based on category
        category = observation.get('category', 'visual')
        if category == 'visual':
            return AnomalyType.UNUSUAL_LAYOUT
        elif category == 'behavioral':
            return AnomalyType.UNUSUAL_NAVIGATION
        else:
            return AnomalyType.DATA_INCONSISTENCY
    
    def _score_to_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Convert anomaly score to severity"""
        # Isolation forest scores: closer to -1 is more anomalous
        abs_score = abs(anomaly_score)
        
        if abs_score > 0.8:
            return AnomalySeverity.CRITICAL
        elif abs_score > 0.6:
            return AnomalySeverity.HIGH
        elif abs_score > 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _calculate_confidence(self, rule: DetectionRule, 
                            observation: Dict[str, Any]) -> float:
        """Calculate detection confidence"""
        base_confidence = 0.8
        
        # Adjust based on evidence strength
        if 'confidence' in observation:
            base_confidence *= observation['confidence']
        
        # Adjust based on rule metadata
        if 'reliability' in rule.metadata:
            base_confidence *= rule.metadata['reliability']
        
        return min(base_confidence, 1.0)
    
    def _generate_anomaly_description(self, anomaly_type: AnomalyType, 
                                    observation: Dict[str, Any]) -> str:
        """Generate human-readable anomaly description"""
        descriptions = {
            AnomalyType.UNEXPECTED_POPUP: "Unexpected popup or dialog appeared",
            AnomalyType.ERROR_DIALOG: "Error dialog detected",
            AnomalyType.UNUSUAL_LAYOUT: "Screen layout is unusual or corrupted",
            AnomalyType.MISSING_ELEMENTS: "Expected UI elements are missing",
            AnomalyType.PERFORMANCE_VISUAL: "Visual performance issues detected",
            AnomalyType.REPEATED_FAILED_ATTEMPTS: "Multiple failed attempts detected",
            AnomalyType.UNUSUAL_NAVIGATION: "Unusual navigation pattern observed",
            AnomalyType.STUCK_STATE: "User appears stuck in current state",
            AnomalyType.CIRCULAR_PATTERN: "Circular navigation pattern detected",
            AnomalyType.TIME_ANOMALY: "Unusual time spent in current activity",
            AnomalyType.RESOURCE_WARNING: "System resource usage is critically high",
            AnomalyType.NETWORK_ISSUE: "Network connectivity problems detected",
            AnomalyType.PERMISSION_PROBLEM: "Permission or access issue encountered",
            AnomalyType.CRASH_INDICATOR: "Application crash indicators detected",
            AnomalyType.DATA_INCONSISTENCY: "Data inconsistency detected"
        }
        
        base_description = descriptions.get(anomaly_type, "Unknown anomaly detected")
        
        # Add context-specific details
        if anomaly_type == AnomalyType.RESOURCE_WARNING and 'resources' in observation:
            resources = observation['resources']
            details = []
            if resources.get('cpu_percent', 0) > 90:
                details.append(f"CPU: {resources['cpu_percent']}%")
            if resources.get('memory_percent', 0) > 85:
                details.append(f"Memory: {resources['memory_percent']}%")
            if details:
                base_description += f" ({', '.join(details)})"
        
        return base_description
    
    def _suggest_actions(self, anomaly_type: AnomalyType) -> List[str]:
        """Suggest actions for anomaly type"""
        suggestions = {
            AnomalyType.UNEXPECTED_POPUP: [
                "Close the popup if safe",
                "Check for malware",
                "Review recent software installations"
            ],
            AnomalyType.ERROR_DIALOG: [
                "Read error message carefully",
                "Take screenshot for reference",
                "Check application logs"
            ],
            AnomalyType.UNUSUAL_LAYOUT: [
                "Refresh the application",
                "Check display settings",
                "Restart the application"
            ],
            AnomalyType.MISSING_ELEMENTS: [
                "Wait for page to fully load",
                "Check internet connection",
                "Clear application cache"
            ],
            AnomalyType.REPEATED_FAILED_ATTEMPTS: [
                "Try alternative approach",
                "Check input validation",
                "Review error messages"
            ],
            AnomalyType.STUCK_STATE: [
                "Use navigation controls",
                "Restart the application",
                "Check for updates"
            ],
            AnomalyType.CIRCULAR_PATTERN: [
                "Break the loop with different action",
                "Return to home/main screen",
                "Review navigation flow"
            ],
            AnomalyType.RESOURCE_WARNING: [
                "Close unnecessary applications",
                "Check for runaway processes",
                "Monitor system resources"
            ],
            AnomalyType.CRASH_INDICATOR: [
                "Save any unsaved work immediately",
                "Restart the application",
                "Check crash logs"
            ]
        }
        
        return suggestions.get(anomaly_type, ["Investigate the anomaly", "Monitor for recurrence"])
    
    async def respond_to_anomaly(self, anomaly: DetectedAnomaly, 
                               action: str = "auto") -> Dict[str, Any]:
        """Respond to detected anomaly"""
        response = {
            'anomaly_id': anomaly.anomaly_id,
            'action_taken': action,
            'timestamp': datetime.now(),
            'success': False
        }
        
        if action == "auto":
            # Automatic response based on severity
            if anomaly.severity == AnomalySeverity.CRITICAL:
                response['action_taken'] = 'alert_user'
                response['details'] = "Critical anomaly requires user attention"
            elif anomaly.severity == AnomalySeverity.HIGH:
                response['action_taken'] = 'attempt_recovery'
                response['details'] = "Attempting automatic recovery"
            else:
                response['action_taken'] = 'monitor'
                response['details'] = "Monitoring anomaly"
        
        elif action == "dismiss":
            anomaly.resolved = True
            anomaly.resolution_time = datetime.now()
            response['success'] = True
            response['details'] = "Anomaly dismissed by user"
        
        elif action == "investigate":
            response['investigation'] = await self._investigate_anomaly(anomaly)
            response['success'] = True
        
        # Update anomaly
        if anomaly.anomaly_id in self.active_anomalies:
            self.active_anomalies[anomaly.anomaly_id] = anomaly
        
        return response
    
    async def _investigate_anomaly(self, anomaly: DetectedAnomaly) -> Dict[str, Any]:
        """Investigate anomaly for more details"""
        investigation = {
            'related_anomalies': [],
            'pattern_analysis': {},
            'root_cause_hypothesis': []
        }
        
        # Find related anomalies
        for historical in self.anomaly_history:
            if (historical.anomaly_type == anomaly.anomaly_type and 
                historical.anomaly_id != anomaly.anomaly_id):
                time_diff = abs((historical.timestamp - anomaly.timestamp).total_seconds())
                if time_diff < 300:  # Within 5 minutes
                    investigation['related_anomalies'].append({
                        'anomaly_id': historical.anomaly_id,
                        'time_diff_seconds': time_diff,
                        'severity': historical.severity.name
                    })
        
        # Analyze patterns
        similar_count = sum(1 for h in self.anomaly_history 
                          if h.anomaly_type == anomaly.anomaly_type)
        investigation['pattern_analysis'] = {
            'frequency': similar_count,
            'first_occurrence': min((h.timestamp for h in self.anomaly_history 
                                   if h.anomaly_type == anomaly.anomaly_type), 
                                  default=anomaly.timestamp),
            'trend': 'increasing' if similar_count > 5 else 'stable'
        }
        
        # Generate hypotheses
        if anomaly.anomaly_type == AnomalyType.REPEATED_FAILED_ATTEMPTS:
            investigation['root_cause_hypothesis'].extend([
                "Input validation issues",
                "Authentication problems",
                "Backend service unavailable"
            ])
        elif anomaly.anomaly_type == AnomalyType.RESOURCE_WARNING:
            investigation['root_cause_hypothesis'].extend([
                "Memory leak in application",
                "Too many background processes",
                "Insufficient system resources"
            ])
        
        return investigation
    
    async def detect_anomaly(self, observation: Observation) -> Optional[DetectedAnomaly]:
        """Detect anomalies in an observation using the Observation class"""
        # Convert Observation to dict format for compatibility
        obs_dict = {
            'timestamp': observation.timestamp.isoformat(),
            'type': observation.observation_type,
            'source': observation.source,
            'confidence': observation.metadata.get('confidence', 0.7),
            **observation.data
        }
        
        # Add metadata fields to observation dict
        for key, value in observation.metadata.items():
            if key not in obs_dict:
                obs_dict[key] = value
        
        # Use existing monitor_realtime method
        anomalies = await self.monitor_realtime(obs_dict)
        
        # Return the highest severity anomaly if multiple detected
        if anomalies:
            return max(anomalies, key=lambda a: a.severity.value)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        stats = {
            'total_checks': self.detection_stats['total_checks'],
            'total_anomalies': self.detection_stats['anomalies_detected'],
            'active_anomalies': len(self.active_anomalies),
            'detection_rate': (self.detection_stats['anomalies_detected'] / 
                             max(self.detection_stats['total_checks'], 1)),
            'average_processing_time': (statistics.mean(self.detection_stats['processing_times'])
                                      if self.detection_stats['processing_times'] else 0),
            'anomaly_types_distribution': {},
            'severity_distribution': {},
            'baselines_established': len(self.baselines)
        }
        
        # Count by type and severity
        for anomaly in self.anomaly_history:
            # Type distribution
            type_name = anomaly.anomaly_type.value
            stats['anomaly_types_distribution'][type_name] = \
                stats['anomaly_types_distribution'].get(type_name, 0) + 1
            
            # Severity distribution
            severity_name = anomaly.severity.name
            stats['severity_distribution'][severity_name] = \
                stats['severity_distribution'].get(severity_name, 0) + 1
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage"""
        return {
            'baseline_models': len(str(self.baselines).encode()),
            'detection_rules': len(str([r.rule_id for r in self.detection_rules]).encode()),
            'anomaly_history': len(str(list(self.anomaly_history)).encode()),
            'total': sum([
                len(str(self.baselines).encode()),
                len(str([r.rule_id for r in self.detection_rules]).encode()),
                len(str(list(self.anomaly_history)).encode())
            ])
        }


# Global instance
_anomaly_framework_instance = None

def get_anomaly_detection_framework() -> AnomalyDetectionFramework:
    """Get or create anomaly detection framework instance"""
    global _anomaly_framework_instance
    if _anomaly_framework_instance is None:
        _anomaly_framework_instance = AnomalyDetectionFramework()
    return _anomaly_framework_instance