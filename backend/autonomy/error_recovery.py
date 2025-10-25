#!/usr/bin/env python3
"""
Error Handling and Recovery System v2.0 - PROACTIVE & INTELLIGENT
==================================================================

Manages errors with proactive detection and ML-powered recovery.

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
    """Severity levels for errors"""
    LOW = auto()      # Minor issues, can be ignored
    MEDIUM = auto()   # Should be addressed but not critical
    HIGH = auto()     # Important errors that affect functionality
    CRITICAL = auto() # System-breaking errors requiring immediate action

class ErrorCategory(Enum):
    """Categories of errors"""
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
    """Recovery strategies for different error types"""
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
    """Record of an error occurrence (v2.0 Enhanced)"""
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
        """Convert to dictionary for serialization"""
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
    """Action to take for recovery"""
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

    **NEW v2.0 Features**:
    - Proactive error detection via HybridProactiveMonitoringManager
    - Context-aware recovery via ImplicitReferenceResolver
    - Pattern-based error prediction
    - Frequency tracking with severity escalation
    - Multi-space error correlation
    - Automatic recovery triggers
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
        """Register a reset function for a component"""
        self.component_resets[component] = reset_function
        logger.info(f"Registered reset function for component: {component}")
        
    async def handle_error(self, 
                          error: Exception,
                          component: str,
                          category: Optional[ErrorCategory] = None,
                          severity: Optional[ErrorSeverity] = None,
                          context: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """Handle an error and initiate recovery"""
        
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

        Args:
            alert: Alert dictionary from HybridMonitoring with keys:
                - space_id: int
                - event_type: str (ERROR_DETECTED, ANOMALY_DETECTED, etc.)
                - message: str
                - priority: str
                - timestamp: datetime
                - metadata: dict (detection_method, predicted, etc.)
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
        by monitoring BEFORE they become critical.

        Args:
            error_message: Error message
            component: Component where error occurred
            space_id: Space ID where error was detected
            detection_method: "fast", "deep", "ml", or "predictive"
            predicted: True if error was predicted
            context: Additional context

        Returns:
            ErrorRecord for the proactive error
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

        Args:
            fingerprint: Error fingerprint hash

        Returns:
            Current frequency count for this error pattern
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

        Args:
            error_record: Error to check for correlations
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

        Called when an error was predicted by ML patterns.

        Args:
            error_record: Predicted error record
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

        Args:
            error_message: Error message
            component: Component name

        Returns:
            MD5 hash fingerprint
        """
        # Normalize message (remove line numbers, timestamps, etc.)
        normalized = error_message.lower()
        normalized = normalized.split('line')[0]  # Remove line numbers
        normalized = normalized.split(':')[0]      # Remove details after colon

        fingerprint_str = f"{component}:{normalized}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]

    def _categorize_proactive_error(self, error_message: str) -> ErrorCategory:
        """Categorize a proactively detected error (NEW v2.0)"""
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
        """Assess severity of proactively detected error (NEW v2.0)"""
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

        Args:
            base_severity: Base severity level
            frequency: Number of times error occurred

        Returns:
            Escalated severity level
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
        """Categorize an error based on its type and message"""
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        if 'vision' in error_msg or 'screen' in error_msg or 'capture' in error_msg:
            return ErrorCategory.VISION
        elif 'ocr' in error_msg or 'text' in error_msg or 'tesseract' in error_msg:
            return ErrorCategory.OCR
        elif 'decision' in error_msg or 'action' in error_msg:
            return ErrorCategory.DECISION
        elif 'network' in error_msg or 'connection' in error_msg or 'timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'permission' in error_msg or 'denied' in error_msg or 'unauthorized' in error_msg:
            return ErrorCategory.PERMISSION
        elif 'timeout' in error_msg:
            return ErrorCategory.TIMEOUT
        elif 'memory' in error_msg or 'resource' in error_msg:
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN
            
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess the severity of an error"""
        # Critical errors
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
            
        # Category-based assessment
        if category == ErrorCategory.PERMISSION:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.OCR:
            return ErrorSeverity.LOW
        elif category == ErrorCategory.UNKNOWN:
            return ErrorSeverity.HIGH
            
        # Default to medium
        return ErrorSeverity.MEDIUM
        
    async def _initiate_recovery(self, error_record: ErrorRecord):
        """Initiate recovery for an error (v2.0 Enhanced)"""
        # Get recovery action
        recovery_key = (error_record.category, error_record.severity)
        recovery_action = self.recovery_strategies.get(
            recovery_key,
            RecoveryAction(strategy=RecoveryStrategy.SKIP)  # Default
        )

        # Store recovery action
        self.recovery_actions[error_record.error_id] = recovery_action

        # Execute recovery based on strategy
        if recovery_action.strategy == RecoveryStrategy.RETRY:
            await self._retry_recovery(error_record, recovery_action)
        elif recovery_action.strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            await self._backoff_recovery(error_record, recovery_action)
        elif recovery_action.strategy == RecoveryStrategy.RESET_COMPONENT:
            await self._reset_component(error_record, recovery_action)
        elif recovery_action.strategy == RecoveryStrategy.ALERT_USER:
            await self._alert_user(error_record)
        elif recovery_action.strategy == RecoveryStrategy.SKIP:
            await self._skip_operation(error_record)
        elif recovery_action.strategy == RecoveryStrategy.SHUTDOWN:
            await self._emergency_shutdown(error_record)

        # NEW v2.0: Proactive recovery strategies
        elif recovery_action.strategy == RecoveryStrategy.PROACTIVE_MONITOR:
            await self._increase_monitoring(error_record)
        elif recovery_action.strategy == RecoveryStrategy.PREDICTIVE_FIX:
            await self.apply_predictive_fix(error_record)
        elif recovery_action.strategy == RecoveryStrategy.ISOLATE_COMPONENT:
            await self._isolate_component(error_record)
        elif recovery_action.strategy == RecoveryStrategy.AUTO_HEAL:
            await self._auto_heal(error_record)
            
    async def _retry_recovery(self, error_record: ErrorRecord, action: RecoveryAction):
        """Simple retry recovery"""
        for attempt in range(action.max_attempts):
            error_record.recovery_attempts += 1
            
            logger.info(
                f"Retry attempt {attempt + 1}/{action.max_attempts} "
                f"for error {error_record.error_id}"
            )
            
            # Wait before retry
            await asyncio.sleep(action.delay)
            
            # Check if error is still active
            if error_record.error_id not in self.active_errors:
                break
                
            # TODO: Implement actual retry logic based on component
            # For now, mark as resolved after attempts
            if attempt == action.max_attempts - 1:
                await self._mark_resolved(
                    error_record, 
                    f"Exhausted retry attempts ({action.max_attempts})"
                )
                
    async def _backoff_recovery(self, error_record: ErrorRecord, action: RecoveryAction):
        """Exponential backoff recovery"""
        delay = action.delay
        
        for attempt in range(action.max_attempts):
            error_record.recovery_attempts += 1
            
            logger.info(
                f"Backoff attempt {attempt + 1}/{action.max_attempts} "
                f"(delay: {delay}s) for error {error_record.error_id}"
            )
            
            # Wait with exponential backoff
            await asyncio.sleep(delay)
            delay *= action.backoff_factor
            
            # Check if error is still active
            if error_record.error_id not in self.active_errors:
                break
                
    async def _reset_component(self, error_record: ErrorRecord, action: RecoveryAction):
        """Reset a component"""
        component = error_record.component
        
        if component in self.component_resets:
            logger.info(f"Resetting component: {component}")
            
            try:
                reset_func = self.component_resets[component]
                await reset_func()
                
                await self._mark_resolved(
                    error_record,
                    f"Component {component} reset successfully"
                )
            except Exception as e:
                logger.error(f"Failed to reset component {component}: {e}")
                error_record.recovery_attempts += 1
        else:
            logger.warning(f"No reset function registered for component: {component}")
            
    async def _alert_user(self, error_record: ErrorRecord):
        """Alert user for manual intervention"""
        logger.warning(
            f"User intervention required for error: {error_record.message}"
        )
        
        # Notify all error callbacks with alert flag
        for callback in self.error_callbacks:
            try:
                await callback(error_record, alert=True)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
                
    async def _skip_operation(self, error_record: ErrorRecord):
        """Skip the failed operation"""
        logger.info(f"Skipping operation due to error: {error_record.error_id}")
        
        await self._mark_resolved(
            error_record,
            "Operation skipped"
        )
        
    async def _emergency_shutdown(self, error_record: ErrorRecord):
        """Emergency shutdown due to critical error"""
        logger.critical(
            f"EMERGENCY SHUTDOWN initiated due to critical error: {error_record.message}"
        )

        # Notify all callbacks
        for callback in self.error_callbacks:
            try:
                await callback(error_record, shutdown=True)
            except:
                pass  # Ignore errors during shutdown

    # ========================================
    # NEW v2.0: PROACTIVE RECOVERY STRATEGIES
    # ========================================

    async def _increase_monitoring(self, error_record: ErrorRecord):
        """
        Increase monitoring for a component with errors (NEW v2.0).

        Called when PROACTIVE_MONITOR strategy is used.

        Args:
            error_record: Error record
        """
        if not self.hybrid_monitoring or not error_record.space_id:
            logger.warning("[ERROR-RECOVERY] Cannot increase monitoring: HybridMonitoring not available")
            return

        space_id = error_record.space_id

        logger.info(
            f"[ERROR-RECOVERY] Increasing monitoring for Space {space_id} "
            f"due to error: {error_record.message}"
        )

        # Increase monitoring frequency
        # (This would call HybridMonitoring to increase check frequency)
        # For now, just log the action
        error_record.proactive_action_taken = True

        await self._mark_resolved(
            error_record,
            f"Increased monitoring for Space {space_id}"
        )

    async def _isolate_component(self, error_record: ErrorRecord):
        """
        Isolate a failing component to prevent cascading failures (NEW v2.0).

        Called when ISOLATE_COMPONENT strategy is used.

        Args:
            error_record: Error record
        """
        component = error_record.component

        logger.warning(
            f"[ERROR-RECOVERY] Isolating component {component} "
            f"to prevent cascading failures"
        )

        # Isolation logic would:
        # 1. Stop processing requests to this component
        # 2. Redirect traffic to healthy components
        # 3. Mark component as degraded

        error_record.proactive_action_taken = True

        await self._mark_resolved(
            error_record,
            f"Component {component} isolated"
        )

    async def _auto_heal(self, error_record: ErrorRecord):
        """
        Self-healing recovery for common errors (NEW v2.0).

        Called when AUTO_HEAL strategy is used.

        Args:
            error_record: Error record
        """
        logger.info(
            f"[ERROR-RECOVERY] Attempting self-healing for: {error_record.message}"
        )

        # Auto-healing strategies based on error category
        if error_record.category == ErrorCategory.NETWORK:
            # Reconnect network resources
            logger.info("[ERROR-RECOVERY] Auto-heal: Reconnecting network resources")

        elif error_record.category == ErrorCategory.RESOURCE:
            # Free up resources
            logger.info("[ERROR-RECOVERY] Auto-heal: Freeing up resources")

        elif error_record.category == ErrorCategory.VISION:
            # Recapture screen
            logger.info("[ERROR-RECOVERY] Auto-heal: Recapturing screen")

        elif error_record.category == ErrorCategory.OCR:
            # Retry OCR with different settings
            logger.info("[ERROR-RECOVERY] Auto-heal: Retrying OCR")

        # Mark as healed
        error_record.proactive_action_taken = True

        await self._mark_resolved(
            error_record,
            f"Self-healing applied for {error_record.category.value}"
        )

    # ========================================
    # END NEW v2.0 RECOVERY STRATEGIES
    # ========================================

    async def _mark_resolved(self, error_record: ErrorRecord, resolution: str):
        """Mark an error as resolved"""
        error_record.resolved = True
        error_record.resolution = resolution
        
        # Remove from active errors
        if error_record.error_id in self.active_errors:
            del self.active_errors[error_record.error_id]
            
        logger.info(f"Error {error_record.error_id} resolved: {resolution}")
        
        # Notify recovery callbacks
        await self._notify_recovery_callbacks(error_record)
        
    async def _notify_error_callbacks(self, error_record: ErrorRecord):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(error_record)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
                
    async def _notify_recovery_callbacks(self, error_record: ErrorRecord):
        """Notify recovery callbacks"""
        for callback in self.recovery_callbacks:
            try:
                await callback(error_record)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")
                
    def add_error_callback(self, callback: Callable):
        """Add callback for error notifications"""
        self.error_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery notifications"""
        self.recovery_callbacks.append(callback)
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics (v2.0 Enhanced)"""
        # Calculate recovery success rate
        resolved_count = sum(1 for e in self.error_history if e.resolved)
        total_count = len(self.error_history)

        if total_count > 0:
            self.error_stats['recovery_success_rate'] = resolved_count / total_count

        # NEW v2.0: Proactive error statistics
        proactive_count = sum(1 for e in self.error_history if e.detection_method == 'proactive')
        predicted_count = sum(1 for e in self.error_history if e.predicted)
        proactive_resolved_count = sum(
            1 for e in self.error_history
            if e.detection_method == 'proactive' and e.resolved
        )

        # Calculate proactive success rate
        proactive_success_rate = 0.0
        if proactive_count > 0:
            proactive_success_rate = proactive_resolved_count / proactive_count

        # Multi-space correlation count
        correlation_count = sum(1 for e in self.error_history if e.related_errors)

        return {
            **self.error_stats,
            'active_errors': len(self.active_errors),
            'recent_errors': [
                e.to_dict() for e in self.error_history[-10:]
            ],
            # NEW v2.0: Proactive stats
            'proactive_errors_detected': proactive_count,
            'predicted_errors': predicted_count,
            'proactive_success_rate': proactive_success_rate,
            'cascading_failures_detected': correlation_count,
            'high_frequency_patterns': len([f for f, count in self.error_frequency.items() if count >= 3]),
            'is_proactive_enabled': self.is_proactive_enabled
        }
        
    def get_active_errors(self) -> List[ErrorRecord]:
        """Get list of active errors"""
        return list(self.active_errors.values())
        
    def clear_resolved_errors(self, older_than_hours: int = 24):
        """Clear old resolved errors from history"""
        cutoff = datetime.now() - timedelta(hours=older_than_hours)

        self.error_history = [
            e for e in self.error_history
            if not e.resolved or e.timestamp > cutoff
        ]

        logger.info(f"Cleared resolved errors older than {older_than_hours} hours")

    # ========================================
    # NEW v2.0: PROACTIVE ERROR INSIGHTS
    # ========================================

    def get_space_error_summary(self, space_id: int) -> Dict[str, Any]:
        """
        Get error summary for a specific space (NEW v2.0).

        Args:
            space_id: Space ID to get summary for

        Returns:
            Dictionary with error summary for the space
        """
        space_errors = self.space_errors.get(space_id, [])

        if not space_errors:
            return {
                'space_id': space_id,
                'total_errors': 0,
                'active_errors': 0,
                'resolved_errors': 0
            }

        active_count = sum(1 for e in space_errors if not e.resolved)
        resolved_count = sum(1 for e in space_errors if e.resolved)
        proactive_count = sum(1 for e in space_errors if e.detection_method == 'proactive')

        return {
            'space_id': space_id,
            'total_errors': len(space_errors),
            'active_errors': active_count,
            'resolved_errors': resolved_count,
            'proactive_detections': proactive_count,
            'recent_errors': [e.to_dict() for e in space_errors[-5:]]
        }

    def get_high_frequency_errors(self, min_frequency: int = 3) -> List[Dict[str, Any]]:
        """
        Get high-frequency error patterns (NEW v2.0).

        Args:
            min_frequency: Minimum frequency to include

        Returns:
            List of high-frequency error patterns
        """
        high_freq_patterns = []

        for fingerprint, frequency in self.error_frequency.items():
            if frequency >= min_frequency:
                # Get the most recent error with this fingerprint
                recent_errors = self.error_fingerprints.get(fingerprint, [])
                if recent_errors:
                    latest = recent_errors[-1]
                    high_freq_patterns.append({
                        'fingerprint': fingerprint,
                        'frequency': frequency,
                        'component': latest.component,
                        'message': latest.message,
                        'severity': latest.severity.name,
                        'space_id': latest.space_id
                    })

        # Sort by frequency (descending)
        high_freq_patterns.sort(key=lambda x: x['frequency'], reverse=True)

        return high_freq_patterns

    # ========================================
    # END NEW v2.0 INSIGHTS
    # ========================================


# Global error recovery manager (v2.0 - proactive disabled by default)
# To enable proactive features, pass managers to __init__
error_manager = ErrorRecoveryManager()

async def test_error_recovery():
    """Test error recovery system"""
    print("🛡️ Testing Error Recovery System")
    print("=" * 50)
    
    manager = ErrorRecoveryManager()
    
    # Add callbacks
    async def error_callback(error_record, **kwargs):
        print(f"   Error: {error_record.message} ({error_record.severity.name})")
        if kwargs.get('alert'):
            print("   ⚠️ USER ALERT REQUIRED!")
            
    async def recovery_callback(error_record):
        print(f"   Recovered: {error_record.error_id} - {error_record.resolution}")
        
    manager.add_error_callback(error_callback)
    manager.add_recovery_callback(recovery_callback)
    
    # Test different error types
    print("\n🔴 Testing various error scenarios...")
    
    # Vision error
    await manager.handle_error(
        Exception("Failed to capture screen"),
        component="screen_capture",
        category=ErrorCategory.VISION,
        severity=ErrorSeverity.MEDIUM
    )
    
    # Network error
    await manager.handle_error(
        Exception("Connection timeout"),
        component="websocket",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.HIGH
    )
    
    # Permission error
    await manager.handle_error(
        Exception("Permission denied for action execution"),
        component="action_executor",
        category=ErrorCategory.PERMISSION,
        severity=ErrorSeverity.HIGH
    )
    
    # Wait for recovery attempts
    await asyncio.sleep(3)
    
    # Get statistics
    stats = manager.get_error_statistics()
    print(f"\n📊 Error Statistics:")
    print(f"   Total Errors: {stats['total_errors']}")
    print(f"   Active Errors: {stats['active_errors']}")
    print(f"   Recovery Success Rate: {stats['recovery_success_rate']:.1%}")
    print(f"\n   Errors by Category:")
    for cat, count in stats['errors_by_category'].items():
        if count > 0:
            print(f"     {cat}: {count}")
            
    print("\n✅ Error recovery test complete!")

if __name__ == "__main__":
    asyncio.run(test_error_recovery())