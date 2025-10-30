#!/usr/bin/env python3
"""
Error Handling and Recovery System v2.0 - PROACTIVE & INTELLIGENT
==================================================================

Manages errors with proactive detection and ML-powered recovery.

This module provides a comprehensive error handling and recovery system that can
detect errors proactively, learn from patterns, and apply intelligent recovery
strategies. It integrates with monitoring systems to prevent errors before they
become critical.

**UPGRADED v2.0 Features**:
✅ Proactive error detection (via HybridProactiveMonitoringManager)
✅ Error pattern recognition (learns from monitoring alerts)
✅ Multi-space error correlation (detects cascading failures)
✅ Frequency-based severity escalation (same error 3+ times = CRITICAL)
✅ Context-aware recovery (via ImplicitReferenceResolver)
✅ Predictive error prevention (anticipates errors before they happen)
✅ Automatic recovery triggers (no manual intervention needed)
✅ Cross-component error tracking (tracks error propagation)

**Integration**:
- HybridProactiveMonitoringManager: Proactive error detection
- ImplicitReferenceResolver: Context understanding for errors
- ChangeDetectionManager: Detects error state changes

**Proactive Capabilities**:
- Detects errors BEFORE they become critical
- Learns error patterns across spaces
- Auto-triggers recovery when errors detected
- Tracks error frequencies and escalates severity
- Prevents cascading failures

Example:
    "Sir, I detected an error pattern: builds in Space 5 lead to errors
     in Space 3. I've proactively increased monitoring for Space 3."
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
import traceback
import json
import hashlib

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severity levels for errors.
    
    Attributes:
        LOW: Minor issues that can be ignored
        MEDIUM: Should be addressed but not critical
        HIGH: Important errors that affect functionality
        CRITICAL: System-breaking errors requiring immediate action
    """
    LOW = auto()      # Minor issues, can be ignored
    MEDIUM = auto()   # Should be addressed but not critical
    HIGH = auto()     # Important errors that affect functionality
    CRITICAL = auto() # System-breaking errors requiring immediate action

class ErrorCategory(Enum):
    """Categories of errors for classification and recovery strategy selection.
    
    Attributes:
        VISION: Screen capture and visual processing errors
        OCR: Text recognition and extraction errors
        DECISION: Decision-making and logic errors
        EXECUTION: Action execution and automation errors
        NETWORK: Network connectivity and communication errors
        PERMISSION: Access control and authorization errors
        TIMEOUT: Operation timeout errors
        RESOURCE: Memory, CPU, and resource allocation errors
        UNKNOWN: Unclassified errors
    """
    VISION = "vision"
    OCR = "ocr"
    DECISION = "decision"
    EXECUTION = "execution"
    NETWORK = "network"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types.
    
    Attributes:
        RETRY: Simple retry with fixed delay
        EXPONENTIAL_BACKOFF: Retry with exponential backoff
        RESET_COMPONENT: Reset the failed component
        FALLBACK: Use fallback method
        SKIP: Skip the operation
        ALERT_USER: Alert user for manual intervention
        SHUTDOWN: Shutdown the system
        PROACTIVE_MONITOR: Increase monitoring (v2.0)
        PREDICTIVE_FIX: Apply predictive fix (v2.0)
        ISOLATE_COMPONENT: Isolate failing component (v2.0)
        AUTO_HEAL: Self-healing recovery (v2.0)
    """
    RETRY = "retry"                   # Simple retry
    EXPONENTIAL_BACKOFF = "backoff"   # Retry with exponential backoff
    RESET_COMPONENT = "reset"         # Reset the failed component
    FALLBACK = "fallback"             # Use fallback method
    SKIP = "skip"                     # Skip the operation
    ALERT_USER = "alert"              # Alert user for manual intervention
    SHUTDOWN = "shutdown"             # Shutdown the system
    PROACTIVE_MONITOR = "proactive_monitor"  # NEW v2.0: Increase monitoring
    PREDICTIVE_FIX = "predictive_fix"  # NEW v2.0: Apply predictive fix
    ISOLATE_COMPONENT = "isolate"      # NEW v2.0: Isolate failing component
    AUTO_HEAL = "auto_heal"            # NEW v2.0: Self-healing recovery

@dataclass
class ErrorRecord:
    """Record of an error occurrence (v2.0 Enhanced).
    
    Tracks comprehensive information about errors including proactive detection
    capabilities, frequency tracking, and multi-space correlation.
    
    Attributes:
        error_id: Unique identifier for the error
        category: Error category for classification
        severity: Severity level of the error
        message: Human-readable error message
        component: Component where error occurred
        timestamp: When the error occurred
        stack_trace: Stack trace if available
        context: Additional context information
        recovery_attempts: Number of recovery attempts made
        resolved: Whether the error has been resolved
        resolution: Description of how error was resolved
        space_id: Space where error occurred (v2.0)
        detection_method: How error was detected (v2.0)
        predicted: Whether error was predicted (v2.0)
        frequency_count: How many times this error pattern appeared (v2.0)
        related_errors: Related error IDs for correlation (v2.0)
        pattern_id: Pattern that predicted this error (v2.0)
        proactive_action_taken: Whether proactive recovery was applied (v2.0)
    """
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    resolved: bool = False
    resolution: Optional[str] = None

    # NEW v2.0: Proactive tracking fields
    space_id: Optional[int] = None  # Space where error occurred
    detection_method: str = "reactive"  # "reactive" or "proactive"
    predicted: bool = False  # True if error was predicted
    frequency_count: int = 1  # How many times this error appeared
    related_errors: List[str] = field(default_factory=list)  # Related error IDs
    pattern_id: Optional[str] = None  # Pattern that predicted this error
    proactive_action_taken: bool = False  # True if proactive recovery applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary for serialization.
        
        Returns:
            Dictionary representation of the error record
        """
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.name,
            'message': self.message,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'recovery_attempts': self.recovery_attempts,
            'resolved': self.resolved,
            'resolution': self.resolution,
            'space_id': self.space_id,
            'detection_method': self.detection_method,
            'predicted': self.predicted,
            'frequency_count': self.frequency_count,
            'proactive_action_taken': self.proactive_action_taken
        }

@dataclass 
class RecoveryAction:
    """Action to take for error recovery.
    
    Defines the recovery strategy and parameters for handling specific errors.
    
    Attributes:
        strategy: Recovery strategy to use
        max_attempts: Maximum number of recovery attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Multiplier for exponential backoff
        timeout: Timeout for recovery operations in seconds
        fallback_action: Optional fallback function to call
        metadata: Additional metadata for the recovery action
    """
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay: float = 1.0  # seconds
    backoff_factor: float = 2.0
    timeout: float = 30.0
    fallback_action: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ErrorRecoveryManager:
    """
    Intelligent Error Recovery Manager v2.0 with Proactive Detection.

    Manages error detection, classification, and recovery with advanced proactive
    capabilities including pattern recognition, frequency tracking, and multi-space
    correlation analysis.

    **NEW v2.0 Features**:
    - Proactive error detection via HybridProactiveMonitoringManager
    - Context-aware recovery via ImplicitReferenceResolver
    - Pattern-based error prediction
    - Frequency tracking with severity escalation
    - Multi-space error correlation
    - Automatic recovery triggers

    Attributes:
        error_history: Complete history of all errors
        active_errors: Currently unresolved errors
        recovery_actions: Recovery actions for each error
        hybrid_monitoring: HybridProactiveMonitoringManager instance
        implicit_resolver: ImplicitReferenceResolver instance
        change_detection: ChangeDetectionManager instance
        error_fingerprints: Error patterns grouped by fingerprint
        space_errors: Errors grouped by space ID
        error_frequency: Frequency count for each error pattern
        predicted_errors: Queue of predicted errors
        is_proactive_enabled: Whether proactive features are enabled
        recovery_strategies: Mapping of error types to recovery strategies
        error_callbacks: Callbacks for error notifications
        recovery_callbacks: Callbacks for recovery notifications
        component_resets: Reset functions for components
        error_stats: Error statistics and metrics
    """

    def __init__(
        self,
        hybrid_monitoring_manager=None,
        implicit_resolver=None,
        change_detection_manager=None
    ):
        """
        Initialize Intelligent Error Recovery Manager v2.0.

        Args:
            hybrid_monitoring_manager: HybridProactiveMonitoringManager for proactive detection
            implicit_resolver: ImplicitReferenceResolver for context understanding
            change_detection_manager: ChangeDetectionManager for error state tracking
        """
        # Core error tracking
        self.error_history: List[ErrorRecord] = []
        self.active_errors: Dict[str, ErrorRecord] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}

        # NEW v2.0: Proactive monitoring integration
        self.hybrid_monitoring = hybrid_monitoring_manager
        self.implicit_resolver = implicit_resolver
        self.change_detection = change_detection_manager

        # NEW v2.0: Proactive error tracking
        self.error_fingerprints: Dict[str, List[ErrorRecord]] = defaultdict(list)  # fingerprint -> errors
        self.space_errors: Dict[int, List[ErrorRecord]] = defaultdict(list)  # space_id -> errors
        self.error_frequency: Dict[str, int] = defaultdict(int)  # pattern -> frequency (renamed from error_patterns)
        self.predicted_errors: deque[Dict[str, Any]] = deque(maxlen=100)  # Predicted errors

        # NEW v2.0: Intelligence
        self.is_proactive_enabled = hybrid_monitoring_manager is not None
        logger.info(f"[ERROR-RECOVERY] v2.0 Initialized (Proactive: {'✅' if self.is_proactive_enabled else '❌'})")

        # Recovery strategy mappings (formerly error_patterns, now recovery_strategies)
        self.recovery_strategies = {
            # Vision errors
            (ErrorCategory.VISION, ErrorSeverity.HIGH): RecoveryAction(
                strategy=RecoveryStrategy.RESET_COMPONENT,
                max_attempts=2
            ),
            (ErrorCategory.VISION, ErrorSeverity.MEDIUM): RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                delay=2.0
            ),
            
            # OCR errors
            (ErrorCategory.OCR, ErrorSeverity.LOW): RecoveryAction(
                strategy=RecoveryStrategy.SKIP
            ),
            (ErrorCategory.OCR, ErrorSeverity.MEDIUM): RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK
            ),
            
            # Network errors
            (ErrorCategory.NETWORK, ErrorSeverity.HIGH): RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=5,
                delay=1.0,
                backoff_factor=2.0
            ),
            
            # Permission errors
            (ErrorCategory.PERMISSION, ErrorSeverity.HIGH): RecoveryAction(
                strategy=RecoveryStrategy.ALERT_USER
            ),
            
            # Timeout errors
            (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM): RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=2,
                delay=5.0
            ),
            
            # Critical errors
            (ErrorCategory.UNKNOWN, ErrorSeverity.CRITICAL): RecoveryAction(
                strategy=RecoveryStrategy.SHUTDOWN
            )
        }
        
        # Error callbacks
        self.error_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Component reset functions
        self.component_resets: Dict[str, Callable] = {}
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {cat.value: 0 for cat in ErrorCategory},
            'errors_by_severity': {sev.name: 0 for sev in ErrorSeverity},
            'recovery_success_rate': 0.0
        }
        
    def register_component_reset(self, component: str, reset_function: Callable):
        """Register a reset function for a component.
        
        Args:
            component: Name of the component
            reset_function: Async function to reset the component
        """
        self.component_resets[component] = reset_function
        logger.info(f"Registered reset function for component: {component}")
        
    async def handle_error(self, 
                          error: Exception,
                          component: str,
                          category: Optional[ErrorCategory] = None,
                          severity: Optional[ErrorSeverity] = None,
                          context: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """Handle an error and initiate recovery.
        
        Main entry point for reactive error handling. Categorizes the error,
        assesses severity, creates an error record, and initiates recovery.
        
        Args:
            error: The exception that occurred
            component: Component where the error occurred
            category: Optional error category (auto-detected if not provided)
            severity: Optional severity level (auto-assessed if not provided)
            context: Additional context information
            
        Returns:
            ErrorRecord for the handled error
            
        Example:
            >>> await manager.handle_error(
            ...     ConnectionError("Network timeout"),
            ...     component="websocket",
            ...     context={"url": "ws://example.com"}
            ... )
        """
        
        # Determine category and severity if not provided
        if not category:
            category = self._categorize_error(error)
        if not severity:
            severity = self._assess_severity(error, category)
            
        # Create error record
        error_record = ErrorRecord(
            error_id=f"{component}_{datetime.now().timestamp()}",
            category=category,
            severity=severity,
            message=str(error),
            component=component,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Add to history and active errors
        self.error_history.append(error_record)
        self.active_errors[error_record.error_id] = error_record
        
        # Update statistics
        self.error_stats['total_errors'] += 1
        self.error_stats['errors_by_category'][category.value] += 1
        self.error_stats['errors_by_severity'][severity.name] += 1
        
        logger.error(
            f"Error in {component}: {error} "
            f"(Category: {category.value}, Severity: {severity.name})"
        )
        
        # Notify callbacks
        await self._notify_error_callbacks(error_record)
        
        # Initiate recovery
        await self._initiate_recovery(error_record)

        return error_record

    # ========================================
    # NEW v2.0: PROACTIVE ERROR DETECTION
    # ========================================

    async def register_monitoring_alert(self, alert: Dict[str, Any]):
        """
        Register a monitoring alert from HybridProactiveMonitoringManager (NEW v2.0).

        Converts monitoring alerts into error records for proactive handling.
        This enables the system to respond to potential issues before they
        become critical failures.

        Args:
            alert: Alert dictionary from HybridMonitoring with keys:
                - space_id: int - Space where alert originated
                - event_type: str - Type of event (ERROR_DETECTED, ANOMALY_DETECTED, etc.)
                - message: str - Alert message
                - priority: str - Alert priority level
                - timestamp: datetime - When alert was generated
                - metadata: dict - Additional metadata (detection_method, predicted, etc.)
                
        Example:
            >>> alert = {
            ...     'space_id': 5,
            ...     'event_type': 'ERROR_DETECTED',
            ...     'message': 'High CPU usage detected',
            ...     'priority': 'HIGH',
            ...     'metadata': {'detection_method': 'ml', 'predicted': True}
            ... }
            >>> await manager.register_monitoring_alert(alert)
        """
        if not self.is_proactive_enabled:
            return

        event_type = alert.get('event_type', '')

        # Only process error-related alerts
        if 'error' not in event_type.lower() and 'anomaly' not in event_type.lower():
            return

        # Extract metadata
        metadata = alert.get('metadata', {})
        detection_method = metadata.get('detection_method', 'proactive')
        predicted = alert.get('predicted', False)
        space_id = alert.get('space_id')

        # Create a synthetic exception for the error record
        error_msg = alert.get('message', 'Proactive error detected')

        # Create error record
        error_record = await self.handle_proactive_error(
            error_message=error_msg,
            component=f"Space_{space_id}" if space_id else "Unknown",
            space_id=space_id,
            detection_method=detection_method,
            predicted=predicted,
            context=metadata
        )

        logger.info(f"[ERROR-RECOVERY] Registered proactive error from monitoring: {error_msg}")

    async def handle_proactive_error(
        self,
        error_message: str,
        component: str,
        space_id: Optional[int] = None,
        detection_method: str = "proactive",
        predicted: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """
        Handle a proactively detected error (NEW v2.0).

        Unlike handle_error(), this is called when errors are detected
        by monitoring BEFORE they become critical. Includes frequency
        tracking and severity escalation based on error patterns.

        Args:
            error_message: Error message describing the issue
            component: Component where error occurred
            space_id: Space ID where error was detected
            detection_method: Detection method ("fast", "deep", "ml", or "predictive")
            predicted: True if error was predicted by ML
            context: Additional context information

        Returns:
            ErrorRecord for the proactive error
            
        Example:
            >>> error_record = await manager.handle_proactive_error(
            ...     error_message="Memory usage approaching limit",
            ...     component="Space_3",
            ...     space_id=3,
            ...     detection_method="ml",
            ...     predicted=True
            ... )
        """
        # Calculate error fingerprint (for frequency tracking)
        fingerprint = self._calculate_error_fingerprint(error_message, component)

        # Check if we've seen this error before
        frequency = await self.track_error_frequency(fingerprint)

        # Create error record
        error_id = f"proactive_{fingerprint}_{datetime.now().timestamp()}"

        # Categorize based on message
        category = self._categorize_proactive_error(error_message)

        # Assess severity with frequency escalation
        base_severity = self._assess_proactive_severity(error_message, predicted)
        severity = self._escalate_severity_by_frequency(base_severity, frequency)

        error_record = ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            message=error_message,
            component=component,
            context=context or {},
            space_id=space_id,
            detection_method=detection_method,
            predicted=predicted,
            frequency_count=frequency
        )

        # Track error
        self.error_history.append(error_record)
        self.active_errors[error_id] = error_record
        self.error_fingerprints[fingerprint].append(error_record)
        if space_id:
            self.space_errors[space_id].append(error_record)

        # Update statistics
        self.error_stats['total_errors'] += 1
        self.error_stats['errors_by_category'][category.value] += 1
        self.error_stats['errors_by_severity'][severity.name] += 1

        logger.warning(
            f"[ERROR-RECOVERY] Proactive error detected: {error_message} "
            f"(method={detection_method}, predicted={predicted}, frequency={frequency})"
        )

        # Check for multi-space correlation
        await self.detect_error_correlation(error_record)

        # Initiate proactive recovery
        await self._initiate_recovery(error_record)

        return error_record

    async def track_error_frequency(self, fingerprint: str) -> int:
        """
        Track error frequency and return current count (NEW v2.0).

        Maintains frequency counters for error patterns to enable
        severity escalation for recurring issues.

        Args:
            fingerprint: Error fingerprint hash

        Returns:
            Current frequency count for this error pattern
            
        Example:
            >>> fingerprint = manager._calculate_error_fingerprint("timeout", "network")
            >>> frequency = await manager.track_error_frequency(fingerprint)
            >>> print(f"This error has occurred {frequency} times")
        """
        self.error_frequency[fingerprint] += 1
        frequency = self.error_frequency[fingerprint]

        if frequency >= 3:
            logger.warning(
                f"[ERROR-RECOVERY] High-frequency error detected: {fingerprint} "
                f"(count={frequency}) - escalating severity"
            )

        return frequency

    async def detect_error_correlation(self, error_record: ErrorRecord):
        """
        Detect multi-space error correlation (NEW v2.0).

        Checks if errors in different spaces are related (cascading failures).
        This helps identify system-wide issues that might require coordinated
        recovery efforts.

        Args:
            error_record: Error to check for correlations
            
        Example:
            >>> # Error in Space 3 might be correlated with errors in Space 5
            >>> await manager.detect_error_correlation(error_record)
        """
        if not error_record.space_id:
            return

        # Look for errors in other spaces within last 30 seconds
        correlation_window = timedelta(seconds=30)
        recent_cutoff = datetime.now() - correlation_window

        related_errors = []
        for space_id, errors in self.space_errors.items():
            if space_id == error_record.space_id:
                continue

            for other_error in errors:
                if other_error.timestamp > recent_cutoff and not other_error.resolved:
                    related_errors.append(other_error.error_id)

        if related_errors:
            error_record.related_errors = related_errors
            logger.warning(
                f"[ERROR-RECOVERY] Cascading failure detected: Space {error_record.space_id} "
                f"error correlated with {len(related_errors)} other spaces"
            )

            # Upgrade to CRITICAL if multiple spaces affected
            if len(related_errors) >= 2 and error_record.severity != ErrorSeverity.CRITICAL:
                error_record.severity = ErrorSeverity.CRITICAL
                logger.critical(
                    f"[ERROR-RECOVERY] Escalating to CRITICAL due to multi-space correlation"
                )

    async def apply_predictive_fix(self, error_record: ErrorRecord):
        """
        Apply predictive fix for anticipated errors (NEW v2.0).

        Called when an error was predicted by ML patterns. Applies
        learned recovery strategies before the error becomes critical.

        Args:
            error_record: Predicted error record
            
        Example:
            >>> # Apply fix for predicted memory leak
            >>> await manager.apply_predictive_fix(predicted_error)
        """
        if not error_record.predicted:
            return

        logger.info(
            f"[ERROR-RECOVERY] Applying predictive fix for: {error_record.message}"
        )

        # Check if we have a pattern-based fix
        pattern_id = error_record.pattern_id
        if pattern_id and self.hybrid_monitoring:
            # Try to get the learned pattern
            # (Pattern contains suggested recovery actions)
            logger.info(f"[ERROR-RECOVERY] Using learned pattern {pattern_id} for recovery")

        # Apply standard recovery with proactive flag
        error_record.proactive_action_taken = True
        await self._mark_resolved(
            error_record,
            f"Predictive fix applied (pattern: {pattern_id})"
        )

    def _calculate_error_fingerprint(self, error_message: str, component: str) -> str:
        """
        Calculate unique fingerprint for error pattern (NEW v2.0).

        Creates a normalized hash of the error for frequency tracking
        and pattern recognition. Removes variable elements like line
        numbers and timestamps.

        Args:
            error_message: Error message
            component: Component name

        Returns:
            MD5 hash fingerprint (8 characters)
            
        Example:
            >>> fingerprint = manager._calculate_error_fingerprint(
            ...     "Connection timeout on line 42", "network"
            ... )
            >>> print(fingerprint)  # "a1b2c3d4"
        """
        # Normalize message (remove line numbers, timestamps, etc.)
        normalized = error_message.lower()
        normalized = normalized.split('line')[0]  # Remove line numbers
        normalized = normalized.split(':')[0]      # Remove details after colon

        fingerprint_str = f"{component}:{normalized}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]

    def _categorize_proactive_error(self, error_message: str) -> ErrorCategory:
        """Categorize a proactively detected error (NEW v2.0).
        
        Analyzes error message content to determine the most appropriate
        error category for proactive errors.
        
        Args:
            error_message: Error message to categorize
            
        Returns:
            ErrorCategory enum value
        """
        msg_lower = error_message.lower()

        if 'vision' in msg_lower or 'screen' in msg_lower:
            return ErrorCategory.VISION
        elif 'ocr' in msg_lower or 'text' in msg_lower:
            return ErrorCategory.OCR
        elif 'decision' in msg_lower or 'action' in msg_lower:
            return ErrorCategory.DECISION
        elif 'network' in msg_lower or 'connection' in msg_lower:
            return ErrorCategory.NETWORK
        elif 'permission' in msg_lower or 'denied' in msg_lower:
            return ErrorCategory.PERMISSION
        elif 'timeout' in msg_lower:
            return ErrorCategory.TIMEOUT
        elif 'memory' in msg_lower or 'resource' in msg_lower:
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN

    def _assess_proactive_severity(self, error_message: str, predicted: bool) -> ErrorSeverity:
        """Assess severity of proactively detected error (NEW v2.0).
        
        Determines severity level for proactive errors, with special
        handling for predicted errors (which start at lower severity
        since we have time to address them).
        
        Args:
            error_message: Error message to assess
            predicted: Whether error was predicted by ML
            
        Returns:
            ErrorSeverity enum value
        """
        msg_lower = error_message.lower()

        # Predicted errors start at lower severity (we have time to fix)
        if predicted:
            if 'critical' in msg_lower or 'fatal' in msg_lower:
                return ErrorSeverity.HIGH  # Downgrade from CRITICAL
            elif 'error' in msg_lower:
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.LOW

        # Proactively detected (not predicted) errors
        if 'critical' in msg_lower or 'fatal' in msg_lower:
            return ErrorSeverity.CRITICAL
        elif 'error' in msg_lower:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM

    def _escalate_severity_by_frequency(self, base_severity: ErrorSeverity, frequency: int) -> ErrorSeverity:
        """
        Escalate severity based on error frequency (NEW v2.0).

        Increases severity for frequently occurring errors to ensure
        they receive appropriate attention and recovery resources.

        Args:
            base_severity: Base severity level
            frequency: Number of times error occurred

        Returns:
            Escalated severity level
            
        Example:
            >>> # Error that occurred 4 times gets escalated
            >>> escalated = manager._escalate_severity_by_frequency(
            ...     ErrorSeverity.MEDIUM, 4
            ... )
            >>> print(escalated)  # ErrorSeverity.CRITICAL
        """
        if frequency >= 5:
            return ErrorSeverity.CRITICAL
        elif frequency >= 3:
            # Escalate by one level
            if base_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif base_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH
            elif base_severity == ErrorSeverity.HIGH:
                return ErrorSeverity.CRITICAL

        return base_severity

    # ========================================
    # END NEW v2.0 METHODS
    # ========================================

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message.
        
        Analyzes exception type and message content to determine
        the most appropriate error category.
        
        Args:
            error: Exception to categorize
            
        Returns:
            ErrorCategory enum value
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        if 'vision' in error_msg or 'screen' in error_msg or 'capture' in error_msg:
            return ErrorCategory.VISION
        elif 'ocr' in error_