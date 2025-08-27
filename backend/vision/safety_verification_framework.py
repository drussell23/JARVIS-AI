#!/usr/bin/env python3
"""
Safety Verification Framework for Vision System v2.0
Comprehensive validation system for generated capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from enum import Enum
from collections import defaultdict
import numpy as np

from .safe_capability_synthesis import SafetyValidator, SafetyViolation
from .sandbox_testing_environment import (
    SandboxTestRunner, TestCase, TestResult, SandboxConfig
)
from .capability_generator import GeneratedCapability

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Levels of safety verification"""
    BASIC = "basic"          # Syntax and static analysis
    STANDARD = "standard"    # Basic + sandbox testing
    COMPREHENSIVE = "comprehensive"  # Standard + behavior analysis
    PRODUCTION = "production"  # All checks + performance validation


class RiskLevel(Enum):
    """Risk levels for capabilities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehaviorTest:
    """Test for capability behavior verification"""
    test_id: str
    name: str
    description: str
    test_type: str  # isolation, side_effects, resource_usage, output_validation
    test_func: Callable
    severity: str = "medium"


@dataclass
class VerificationReport:
    """Comprehensive verification report"""
    capability_id: str
    verification_level: VerificationLevel
    timestamp: datetime
    
    # Safety analysis
    safety_score: float
    risk_level: RiskLevel
    safety_violations: List[SafetyViolation] = field(default_factory=list)
    
    # Testing results
    test_results: List[TestResult] = field(default_factory=list)
    behavior_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Overall assessment
    approved: bool = False
    approval_conditions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BehaviorAnalyzer:
    """Analyzes capability behavior for safety"""
    
    def __init__(self):
        self.behavior_tests = self._create_behavior_tests()
        
    def _create_behavior_tests(self) -> List[BehaviorTest]:
        """Create standard behavior tests"""
        return [
            BehaviorTest(
                test_id="isolation_test",
                name="Process Isolation",
                description="Verifies capability doesn't access external processes",
                test_type="isolation",
                test_func=self._test_process_isolation,
                severity="high"
            ),
            BehaviorTest(
                test_id="side_effects_test",
                name="Side Effects",
                description="Checks for unintended side effects",
                test_type="side_effects",
                test_func=self._test_side_effects,
                severity="high"
            ),
            BehaviorTest(
                test_id="resource_usage_test",
                name="Resource Usage",
                description="Monitors resource consumption",
                test_type="resource_usage",
                test_func=self._test_resource_usage,
                severity="medium"
            ),
            BehaviorTest(
                test_id="output_validation_test",
                name="Output Validation",
                description="Validates output format and content",
                test_type="output_validation",
                test_func=self._test_output_validation,
                severity="medium"
            ),
            BehaviorTest(
                test_id="determinism_test",
                name="Determinism",
                description="Checks if behavior is deterministic",
                test_type="determinism",
                test_func=self._test_determinism,
                severity="low"
            )
        ]
    
    async def analyze_behavior(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Analyze capability behavior from test results"""
        behavior_analysis = {
            'tests_performed': [],
            'issues_found': [],
            'risk_indicators': [],
            'behavior_score': 1.0
        }
        
        # Run behavior tests
        for behavior_test in self.behavior_tests:
            try:
                result = await behavior_test.test_func(capability, test_results)
                
                behavior_analysis['tests_performed'].append({
                    'test': behavior_test.name,
                    'passed': result['passed'],
                    'details': result.get('details', '')
                })
                
                if not result['passed']:
                    behavior_analysis['issues_found'].append({
                        'test': behavior_test.name,
                        'severity': behavior_test.severity,
                        'description': result.get('issue', 'Test failed')
                    })
                    
                    # Adjust behavior score
                    severity_penalty = {
                        'low': 0.1,
                        'medium': 0.2,
                        'high': 0.3,
                        'critical': 0.5
                    }
                    behavior_analysis['behavior_score'] -= severity_penalty.get(
                        behavior_test.severity, 0.2
                    )
                    
            except Exception as e:
                logger.error(f"Behavior test {behavior_test.name} failed: {e}")
                behavior_analysis['issues_found'].append({
                    'test': behavior_test.name,
                    'severity': 'high',
                    'description': f'Test error: {str(e)}'
                })
        
        # Ensure score is non-negative
        behavior_analysis['behavior_score'] = max(0.0, behavior_analysis['behavior_score'])
        
        # Identify risk indicators
        behavior_analysis['risk_indicators'] = self._identify_risk_indicators(
            behavior_analysis['issues_found']
        )
        
        return behavior_analysis
    
    async def _test_process_isolation(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Test that capability doesn't access external processes"""
        # Check for process/subprocess usage in code
        forbidden_patterns = ['subprocess', 'os.system', 'os.exec', 'Popen']
        
        for pattern in forbidden_patterns:
            if pattern in capability.handler_code:
                return {
                    'passed': False,
                    'issue': f'Found forbidden pattern: {pattern}'
                }
        
        # Check test logs for process spawning
        for result in test_results:
            for log in result.logs:
                if 'fork' in log.lower() or 'exec' in log.lower():
                    return {
                        'passed': False,
                        'issue': 'Process spawning detected in logs'
                    }
        
        return {'passed': True}
    
    async def _test_side_effects(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Test for unintended side effects"""
        # Check for global state modification
        if 'global' in capability.handler_code:
            return {
                'passed': False,
                'issue': 'Global state modification detected'
            }
        
        # Check for file system operations
        file_ops = ['open(', 'write(', 'mkdir', 'remove', 'delete']
        for op in file_ops:
            if op in capability.handler_code:
                return {
                    'passed': False,
                    'issue': f'File system operation detected: {op}'
                }
        
        return {'passed': True}
    
    async def _test_resource_usage(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Test resource usage patterns"""
        # Analyze test durations
        durations = [r.duration for r in test_results if r.success]
        
        if not durations:
            return {
                'passed': False,
                'issue': 'No successful test results to analyze'
            }
        
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        # Check for excessive execution time
        if max_duration > 10.0:  # 10 seconds
            return {
                'passed': False,
                'issue': f'Excessive execution time: {max_duration:.2f}s'
            }
        
        # Check for resource patterns in code
        if 'while True' in capability.handler_code and 'break' not in capability.handler_code:
            return {
                'passed': False,
                'issue': 'Potential infinite loop detected'
            }
        
        return {
            'passed': True,
            'details': f'Avg duration: {avg_duration:.2f}s, Max: {max_duration:.2f}s'
        }
    
    async def _test_output_validation(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Test output format and content"""
        valid_outputs = 0
        
        for result in test_results:
            if result.output and isinstance(result.output, dict):
                # Check for required fields
                if 'success' in result.output:
                    valid_outputs += 1
        
        if valid_outputs < len(test_results) * 0.8:  # 80% threshold
            return {
                'passed': False,
                'issue': 'Output format validation failed'
            }
        
        return {'passed': True}
    
    async def _test_determinism(
        self,
        capability: GeneratedCapability,
        test_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Test if capability behavior is deterministic"""
        # Check for randomness in code
        random_indicators = ['random', 'uuid', 'time.time()', 'datetime.now()']
        
        for indicator in random_indicators:
            if indicator in capability.handler_code:
                return {
                    'passed': True,  # Not a failure, just a note
                    'details': f'Non-deterministic element found: {indicator}'
                }
        
        return {'passed': True}
    
    def _identify_risk_indicators(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Identify risk indicators from issues"""
        risk_indicators = []
        
        high_severity_count = sum(1 for issue in issues if issue['severity'] == 'high')
        if high_severity_count > 0:
            risk_indicators.append(f"{high_severity_count} high severity issues")
        
        # Check specific issue patterns
        for issue in issues:
            if 'process' in issue['description'].lower():
                risk_indicators.append("Process isolation concerns")
            if 'resource' in issue['description'].lower():
                risk_indicators.append("Resource usage concerns")
            if 'side effect' in issue['description'].lower():
                risk_indicators.append("Side effect concerns")
        
        return list(set(risk_indicators))  # Remove duplicates


class PerformanceValidator:
    """Validates capability performance"""
    
    def __init__(self):
        self.performance_thresholds = {
            'latency_ms': 100,      # Max acceptable latency
            'memory_mb': 50,        # Max memory usage
            'cpu_percent': 25,      # Max CPU usage
            'success_rate': 0.95    # Min success rate
        }
    
    async def validate_performance(
        self,
        capability: GeneratedCapability,
        benchmark_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate capability performance against thresholds"""
        validation = {
            'meets_requirements': True,
            'metrics': {},
            'failures': []
        }
        
        # Check latency
        avg_duration_ms = benchmark_results.get('avg_duration', 0) * 1000
        validation['metrics']['latency_ms'] = avg_duration_ms
        
        if avg_duration_ms > self.performance_thresholds['latency_ms']:
            validation['meets_requirements'] = False
            validation['failures'].append(
                f"Latency {avg_duration_ms:.1f}ms exceeds threshold "
                f"{self.performance_thresholds['latency_ms']}ms"
            )
        
        # Check memory usage
        avg_memory = benchmark_results.get('avg_memory_mb', 0)
        validation['metrics']['memory_mb'] = avg_memory
        
        if avg_memory > self.performance_thresholds['memory_mb']:
            validation['meets_requirements'] = False
            validation['failures'].append(
                f"Memory usage {avg_memory:.1f}MB exceeds threshold "
                f"{self.performance_thresholds['memory_mb']}MB"
            )
        
        # Check success rate
        success_rate = benchmark_results.get('success_rate', 0)
        validation['metrics']['success_rate'] = success_rate
        
        if success_rate < self.performance_thresholds['success_rate']:
            validation['meets_requirements'] = False
            validation['failures'].append(
                f"Success rate {success_rate:.2%} below threshold "
                f"{self.performance_thresholds['success_rate']:.2%}"
            )
        
        return validation


class SafetyVerificationFramework:
    """Main framework for comprehensive safety verification"""
    
    def __init__(self):
        self.safety_validator = SafetyValidator()
        self.sandbox_runner = SandboxTestRunner()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.performance_validator = PerformanceValidator()
        
        # Verification history
        self.verification_history: Dict[str, VerificationReport] = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
    
    async def verify_capability(
        self,
        capability: GeneratedCapability,
        verification_level: VerificationLevel = VerificationLevel.STANDARD
    ) -> VerificationReport:
        """Perform comprehensive capability verification"""
        logger.info(f"Starting {verification_level.value} verification for {capability.name}")
        
        # Initialize report
        report = VerificationReport(
            capability_id=capability.capability_id,
            verification_level=verification_level,
            timestamp=datetime.now(),
            safety_score=0.0,
            risk_level=RiskLevel.HIGH
        )
        
        # Step 1: Safety validation
        safety_result = await self.safety_validator.validate_capability(
            capability.handler_code,
            {'capability_name': capability.name}
        )
        
        report.safety_violations = safety_result['violations']
        report.safety_score = 1.0 - safety_result['risk_score']
        
        # Early exit if critical safety violations
        if not safety_result['safe']:
            report.risk_level = RiskLevel.CRITICAL
            report.approved = False
            report.recommendations = safety_result['recommendations']
            return report
        
        # Step 2: Sandbox testing (if STANDARD or higher)
        if verification_level.value in ['standard', 'comprehensive', 'production']:
            test_cases = self.sandbox_runner.create_standard_test_cases(capability.name)
            
            # Add capability-specific test cases
            if capability.training_examples:
                for i, example in enumerate(capability.training_examples[:3]):
                    test_cases.append(TestCase(
                        test_id=f"{capability.name}_example_{i}",
                        name=f"Training Example {i+1}",
                        input_data=example,
                        timeout=10.0
                    ))
            
            # Run tests
            test_summary = await self.sandbox_runner.test_capability(
                capability.handler_code,
                test_cases
            )
            
            report.test_results = test_summary['results']
            
            # Analyze test results
            test_success_rate = test_summary['passed'] / test_summary['total_tests']
            if test_success_rate < 0.8:
                report.risk_level = RiskLevel.HIGH
                report.recommendations.append(
                    f"Low test success rate: {test_success_rate:.1%}"
                )
        
        # Step 3: Behavior analysis (if COMPREHENSIVE or higher)
        if verification_level.value in ['comprehensive', 'production']:
            behavior_analysis = await self.behavior_analyzer.analyze_behavior(
                capability,
                report.test_results
            )
            
            report.behavior_results = behavior_analysis
            
            # Adjust safety score based on behavior
            report.safety_score *= behavior_analysis['behavior_score']
        
        # Step 4: Performance validation (if PRODUCTION)
        if verification_level == VerificationLevel.PRODUCTION:
            benchmark_results = await self.sandbox_runner.benchmark_capability(
                capability.handler_code,
                {'command': 'benchmark test', 'context': {}},
                iterations=10
            )
            
            performance_validation = await self.performance_validator.validate_performance(
                capability,
                benchmark_results
            )
            
            report.performance_metrics = performance_validation['metrics']
            
            if not performance_validation['meets_requirements']:
                report.recommendations.extend(performance_validation['failures'])
        
        # Step 5: Calculate final risk level
        report.risk_level = self._calculate_risk_level(report)
        
        # Step 6: Determine approval
        report.approved = self._determine_approval(report)
        
        # Step 7: Generate approval conditions
        if report.approved:
            report.approval_conditions = self._generate_approval_conditions(report)
        
        # Store in history
        self.verification_history[capability.capability_id] = report
        
        logger.info(
            f"Verification complete: {capability.name} - "
            f"Risk: {report.risk_level.value}, Approved: {report.approved}"
        )
        
        return report
    
    def _calculate_risk_level(self, report: VerificationReport) -> RiskLevel:
        """Calculate overall risk level"""
        # Start with safety score
        risk_score = 1.0 - report.safety_score
        
        # Add penalties for violations
        violation_penalty = len(report.safety_violations) * 0.1
        risk_score += violation_penalty
        
        # Add penalties for test failures
        if report.test_results:
            test_failure_rate = sum(
                1 for r in report.test_results if not r.success
            ) / len(report.test_results)
            risk_score += test_failure_rate * 0.3
        
        # Add penalties for behavior issues
        if report.behavior_results:
            behavior_penalty = len(report.behavior_results.get('issues_found', [])) * 0.1
            risk_score += behavior_penalty
        
        # Determine risk level
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if risk_score >= self.risk_thresholds[level] - 0.01:
                return level
        
        return RiskLevel.LOW
    
    def _determine_approval(self, report: VerificationReport) -> bool:
        """Determine if capability should be approved"""
        # Never approve critical risk
        if report.risk_level == RiskLevel.CRITICAL:
            return False
        
        # Check safety score
        if report.safety_score < 0.6:
            return False
        
        # Check test results
        if report.test_results:
            success_rate = sum(
                1 for r in report.test_results if r.success
            ) / len(report.test_results)
            if success_rate < 0.7:
                return False
        
        # Check behavior issues
        if report.behavior_results:
            high_severity_issues = [
                issue for issue in report.behavior_results.get('issues_found', [])
                if issue['severity'] == 'high'
            ]
            if high_severity_issues:
                return False
        
        # Check performance (for production level)
        if report.verification_level == VerificationLevel.PRODUCTION:
            if report.performance_metrics.get('success_rate', 0) < 0.9:
                return False
        
        return True
    
    def _generate_approval_conditions(self, report: VerificationReport) -> List[str]:
        """Generate conditions for capability approval"""
        conditions = []
        
        # Add risk-based conditions
        if report.risk_level == RiskLevel.HIGH:
            conditions.append("Requires manual review before production use")
            conditions.append("Limited to sandbox environment initially")
        elif report.risk_level == RiskLevel.MEDIUM:
            conditions.append("Monitor closely during initial deployment")
            conditions.append("Gradual rollout recommended")
        
        # Add performance conditions
        if report.performance_metrics:
            if report.performance_metrics.get('latency_ms', 0) > 50:
                conditions.append("Performance optimization recommended")
        
        # Add behavior conditions
        if report.behavior_results:
            if report.behavior_results.get('risk_indicators'):
                conditions.append("Address identified risk indicators")
        
        # Default condition
        if not conditions:
            conditions.append("Standard monitoring and logging required")
        
        return conditions
    
    def get_verification_summary(
        self,
        capability_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get verification summary"""
        if capability_id:
            report = self.verification_history.get(capability_id)
            if report:
                return {
                    'capability_id': capability_id,
                    'verified': True,
                    'approved': report.approved,
                    'risk_level': report.risk_level.value,
                    'safety_score': report.safety_score,
                    'conditions': report.approval_conditions,
                    'timestamp': report.timestamp.isoformat()
                }
            else:
                return {'capability_id': capability_id, 'verified': False}
        
        # Return summary of all verifications
        summary = {
            'total_verified': len(self.verification_history),
            'approved': sum(
                1 for r in self.verification_history.values() if r.approved
            ),
            'risk_distribution': defaultdict(int)
        }
        
        for report in self.verification_history.values():
            summary['risk_distribution'][report.risk_level.value] += 1
        
        return summary


# Singleton instance
_verification_framework: Optional[SafetyVerificationFramework] = None


def get_safety_verification_framework() -> SafetyVerificationFramework:
    """Get singleton instance of safety verification framework"""
    global _verification_framework
    if _verification_framework is None:
        _verification_framework = SafetyVerificationFramework()
    return _verification_framework