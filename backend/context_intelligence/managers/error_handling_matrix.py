"""
Error Handling Matrix for JARVIS
=================================

Provides comprehensive error handling with graceful degradation:

Priority 1: Try primary method
   ↓ (fails)
Priority 2: Try fallback method
   ↓ (fails)
Priority 3: Return partial results + warning
   ↓ (fails)
Priority 4: Return user-friendly error message

Features:
- Dynamic fallback chains
- Partial result aggregation
- Automatic error recovery
- User-friendly error messages
- Full async support

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION STATES & RESULTS
# ============================================================================

class ExecutionPriority(Enum):
    """Execution priority levels"""
    PRIMARY = 1      # Main method
    FALLBACK = 2     # First fallback
    SECONDARY = 3    # Second fallback
    TERTIARY = 4     # Third fallback
    LAST_RESORT = 5  # Final attempt


class ExecutionStatus(Enum):
    """Execution status for each method"""
    SUCCESS = "success"              # Method succeeded
    FAILED = "failed"                # Method failed
    SKIPPED = "skipped"              # Method was skipped
    PARTIAL = "partial"              # Method returned partial results
    TIMEOUT = "timeout"              # Method timed out
    NOT_AVAILABLE = "not_available"  # Method not available


class ResultQuality(Enum):
    """Quality of final result"""
    FULL = "full"              # All methods succeeded or primary succeeded
    PARTIAL = "partial"        # Some methods succeeded
    DEGRADED = "degraded"      # Fallback methods succeeded
    MINIMAL = "minimal"        # Only last resort succeeded
    FAILED = "failed"          # All methods failed


@dataclass
class MethodResult:
    """Result of a single method execution"""
    method_name: str
    priority: ExecutionPriority
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Complete execution report with all attempts"""
    operation_name: str
    final_status: ResultQuality
    final_result: Any
    methods_attempted: List[MethodResult]
    total_duration: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful"""
        return self.final_status in [ResultQuality.FULL, ResultQuality.PARTIAL, ResultQuality.DEGRADED]

    @property
    def has_result(self) -> bool:
        """Check if we have any result"""
        return self.final_result is not None


# ============================================================================
# FALLBACK CHAIN DEFINITION
# ============================================================================

@dataclass
class MethodDefinition:
    """Definition of a method in the fallback chain"""
    name: str
    func: Callable
    priority: ExecutionPriority
    timeout: Optional[float] = None
    required: bool = False  # If True, failure stops the chain
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FallbackChain:
    """
    Defines a chain of methods to try in priority order

    Example:
        chain = FallbackChain("screenshot")
        chain.add_primary(capture_with_cg, timeout=5.0)
        chain.add_fallback(capture_with_screencapture, timeout=10.0)
        chain.add_last_resort(capture_with_pyautogui)
    """

    def __init__(self, operation_name: str):
        """
        Initialize fallback chain

        Args:
            operation_name: Name of the operation (for logging)
        """
        self.operation_name = operation_name
        self.methods: List[MethodDefinition] = []

    def add_method(
        self,
        func: Callable,
        priority: ExecutionPriority,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        required: bool = False,
        *args,
        **kwargs
    ) -> 'FallbackChain':
        """
        Add a method to the chain

        Args:
            func: Async function to execute
            priority: Priority level
            name: Method name (uses func.__name__ if not provided)
            timeout: Timeout for this method
            required: If True, failure stops the chain
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Self for chaining
        """
        method = MethodDefinition(
            name=name or func.__name__,
            func=func,
            priority=priority,
            timeout=timeout,
            required=required,
            args=args,
            kwargs=kwargs
        )
        self.methods.append(method)
        return self

    def add_primary(
        self,
        func: Callable,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> 'FallbackChain':
        """Add primary method"""
        return self.add_method(func, ExecutionPriority.PRIMARY, name, timeout, False, *args, **kwargs)

    def add_fallback(
        self,
        func: Callable,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> 'FallbackChain':
        """Add first fallback method"""
        return self.add_method(func, ExecutionPriority.FALLBACK, name, timeout, False, *args, **kwargs)

    def add_secondary(
        self,
        func: Callable,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> 'FallbackChain':
        """Add secondary fallback method"""
        return self.add_method(func, ExecutionPriority.SECONDARY, name, timeout, False, *args, **kwargs)

    def add_last_resort(
        self,
        func: Callable,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        *args,
        **kwargs
    ) -> 'FallbackChain':
        """Add last resort method"""
        return self.add_method(func, ExecutionPriority.LAST_RESORT, name, timeout, False, *args, **kwargs)

    def get_sorted_methods(self) -> List[MethodDefinition]:
        """Get methods sorted by priority"""
        return sorted(self.methods, key=lambda m: m.priority.value)


# ============================================================================
# PARTIAL RESULT AGGREGATOR
# ============================================================================

class PartialResultAggregator:
    """
    Aggregates partial results from multiple methods

    Strategies:
    - FIRST_SUCCESS: Use first successful result
    - BEST_RESULT: Use best result based on quality metric
    - MERGE: Merge all results
    - UNION: Union of all results (for lists/sets)
    """

    def __init__(self, strategy: str = "first_success"):
        """
        Initialize aggregator

        Args:
            strategy: Aggregation strategy (first_success, best_result, merge, union)
        """
        self.strategy = strategy
        logger.info(f"[PARTIAL-AGGREGATOR] Initialized with strategy: {strategy}")

    def aggregate(
        self,
        results: List[MethodResult],
        quality_func: Optional[Callable[[Any], float]] = None
    ) -> Tuple[Any, List[str]]:
        """
        Aggregate partial results

        Args:
            results: List of method results
            quality_func: Optional function to assess result quality

        Returns:
            (aggregated_result, warnings)
        """
        logger.debug(f"[PARTIAL-AGGREGATOR] Aggregating {len(results)} results")

        # Filter to successful results
        successful = [r for r in results if r.status == ExecutionStatus.SUCCESS and r.result is not None]

        if not successful:
            logger.warning("[PARTIAL-AGGREGATOR] No successful results to aggregate")
            return None, ["No successful results available"]

        warnings = []

        if self.strategy == "first_success":
            # Use first successful result
            result = successful[0].result
            if len(successful) < len(results):
                warnings.append(f"Using result from {successful[0].method_name} (primary method failed)")

        elif self.strategy == "best_result":
            # Use best result based on quality function
            if quality_func:
                best = max(successful, key=lambda r: quality_func(r.result))
                result = best.result
                if best != successful[0]:
                    warnings.append(f"Using best result from {best.method_name}")
            else:
                result = successful[0].result
                warnings.append("No quality function provided, using first result")

        elif self.strategy == "merge":
            # Merge results (assumes dict results)
            result = {}
            for r in successful:
                if isinstance(r.result, dict):
                    result.update(r.result)
                else:
                    logger.warning(f"[PARTIAL-AGGREGATOR] Cannot merge non-dict result from {r.method_name}")
            if len(successful) < len(results):
                warnings.append(f"Merged {len(successful)}/{len(results)} results")

        elif self.strategy == "union":
            # Union of results (assumes list/set results)
            result = []
            for r in successful:
                if isinstance(r.result, (list, set, tuple)):
                    result.extend(r.result if isinstance(r.result, (list, tuple)) else list(r.result))
                else:
                    logger.warning(f"[PARTIAL-AGGREGATOR] Cannot union non-list result from {r.method_name}")
            result = list(set(result))  # Remove duplicates
            if len(successful) < len(results):
                warnings.append(f"Unioned {len(successful)}/{len(results)} results")

        else:
            logger.error(f"[PARTIAL-AGGREGATOR] Unknown strategy: {self.strategy}")
            result = successful[0].result
            warnings.append(f"Unknown strategy {self.strategy}, using first result")

        logger.info(f"[PARTIAL-AGGREGATOR] Aggregation complete: {len(warnings)} warnings")
        return result, warnings


# ============================================================================
# ERROR RECOVERY STRATEGY
# ============================================================================

class ErrorRecoveryStrategy:
    """
    Defines how to recover from errors

    Strategies:
    - CONTINUE: Continue to next method
    - RETRY: Retry the same method
    - SKIP_TO_PRIORITY: Skip to specific priority level
    - ABORT: Stop execution
    """

    def __init__(self, strategy: str = "continue", max_retries: int = 0):
        """
        Initialize recovery strategy

        Args:
            strategy: Recovery strategy (continue, retry, skip_to_priority, abort)
            max_retries: Maximum retries for retry strategy
        """
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = defaultdict(int)

        logger.info(f"[RECOVERY-STRATEGY] Initialized: {strategy} (max_retries={max_retries})")

    async def handle_error(
        self,
        method: MethodDefinition,
        error: Exception
    ) -> Tuple[str, Optional[MethodDefinition]]:
        """
        Handle error and determine next action

        Args:
            method: Method that failed
            error: Exception that occurred

        Returns:
            (action, method_to_retry) - action is "continue", "retry", "skip", or "abort"
        """
        logger.warning(f"[RECOVERY-STRATEGY] Handling error from {method.name}: {error}")

        if self.strategy == "continue":
            return "continue", None

        elif self.strategy == "retry":
            retry_count = self.retry_counts[method.name]
            if retry_count < self.max_retries:
                self.retry_counts[method.name] += 1
                logger.info(f"[RECOVERY-STRATEGY] Retrying {method.name} (attempt {retry_count + 1}/{self.max_retries})")
                return "retry", method
            else:
                logger.warning(f"[RECOVERY-STRATEGY] Max retries exceeded for {method.name}")
                return "continue", None

        elif self.strategy == "abort":
            logger.error(f"[RECOVERY-STRATEGY] Aborting execution due to error in {method.name}")
            return "abort", None

        else:
            logger.warning(f"[RECOVERY-STRATEGY] Unknown strategy: {self.strategy}, continuing")
            return "continue", None


# ============================================================================
# ERROR HANDLING MATRIX (Main Coordinator)
# ============================================================================

class ErrorHandlingMatrix:
    """
    Main coordinator for error handling with graceful degradation

    Executes fallback chains with:
    - Priority-based method execution
    - Automatic fallback on failure
    - Partial result aggregation
    - Error recovery strategies
    - User-friendly error messages
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        aggregation_strategy: str = "first_success",
        recovery_strategy: str = "continue"
    ):
        """
        Initialize error handling matrix

        Args:
            default_timeout: Default timeout for methods
            aggregation_strategy: Strategy for aggregating partial results
            recovery_strategy: Strategy for error recovery
        """
        self.default_timeout = default_timeout
        self.aggregator = PartialResultAggregator(strategy=aggregation_strategy)
        self.recovery = ErrorRecoveryStrategy(strategy=recovery_strategy)

        logger.info(f"[ERROR-MATRIX] Initialized (timeout={default_timeout}s)")

    async def execute_chain(
        self,
        chain: FallbackChain,
        stop_on_success: bool = True,
        collect_partial: bool = True
    ) -> ExecutionReport:
        """
        Execute a fallback chain with graceful degradation

        Args:
            chain: FallbackChain to execute
            stop_on_success: Stop after first success (Priority 1-4)
            collect_partial: Collect partial results from all methods

        Returns:
            ExecutionReport with results and metadata
        """
        logger.info(f"[ERROR-MATRIX] Executing chain: {chain.operation_name}")

        start_time = time.time()
        methods_attempted: List[MethodResult] = []
        successful_results: List[MethodResult] = []
        warnings: List[str] = []
        errors: List[str] = []

        # Get methods in priority order
        sorted_methods = chain.get_sorted_methods()

        for method in sorted_methods:
            # Execute method
            result = await self._execute_method(method)
            methods_attempted.append(result)

            # Track result
            if result.status == ExecutionStatus.SUCCESS:
                logger.info(f"[ERROR-MATRIX] ✅ {method.name} succeeded")
                successful_results.append(result)

                # Stop on success if configured
                if stop_on_success and method.priority in [ExecutionPriority.PRIMARY, ExecutionPriority.FALLBACK]:
                    logger.info(f"[ERROR-MATRIX] Stopping after success in {method.name}")
                    break

            else:
                logger.warning(f"[ERROR-MATRIX] ❌ {method.name} failed: {result.error}")
                if result.error:
                    errors.append(f"{method.name}: {result.error}")

                # Handle error with recovery strategy
                action, retry_method = await self.recovery.handle_error(method, Exception(result.error or "Unknown error"))

                if action == "retry" and retry_method:
                    # Retry the method
                    retry_result = await self._execute_method(retry_method)
                    methods_attempted.append(retry_result)
                    if retry_result.status == ExecutionStatus.SUCCESS:
                        successful_results.append(retry_result)

                elif action == "abort":
                    errors.append("Execution aborted due to error recovery strategy")
                    break

                # Check if required method failed
                if method.required:
                    logger.error(f"[ERROR-MATRIX] Required method {method.name} failed, stopping chain")
                    errors.append(f"Required method {method.name} failed")
                    break

        # Calculate total duration
        total_duration = time.time() - start_time

        # Determine final result and quality
        final_result, final_status, message = self._finalize_result(
            chain.operation_name,
            methods_attempted,
            successful_results,
            collect_partial
        )

        # Aggregate warnings from partial results
        if collect_partial and len(successful_results) > 1:
            _, agg_warnings = self.aggregator.aggregate(successful_results)
            warnings.extend(agg_warnings)

        # Add warnings for failed methods
        failed_count = len([r for r in methods_attempted if r.status == ExecutionStatus.FAILED])
        if failed_count > 0 and len(successful_results) > 0:
            warnings.append(f"{failed_count} method(s) failed, using fallback results")

        logger.info(f"[ERROR-MATRIX] Chain complete: {final_status.value} ({total_duration:.2f}s)")

        return ExecutionReport(
            operation_name=chain.operation_name,
            final_status=final_status,
            final_result=final_result,
            methods_attempted=methods_attempted,
            total_duration=total_duration,
            warnings=warnings,
            errors=errors,
            message=message
        )

    async def _execute_method(self, method: MethodDefinition) -> MethodResult:
        """Execute a single method with timeout"""
        logger.debug(f"[ERROR-MATRIX] Executing {method.name} (priority={method.priority.value})")

        start_time = time.time()

        try:
            # Apply timeout
            timeout = method.timeout or self.default_timeout

            # Execute with timeout
            result = await asyncio.wait_for(
                method.func(*method.args, **method.kwargs),
                timeout=timeout
            )

            duration = time.time() - start_time

            logger.info(f"[ERROR-MATRIX] {method.name} completed in {duration:.2f}s")

            return MethodResult(
                method_name=method.name,
                priority=method.priority,
                status=ExecutionStatus.SUCCESS,
                result=result,
                duration=duration,
                metadata=method.metadata
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"[ERROR-MATRIX] {method.name} timed out after {duration:.2f}s")

            return MethodResult(
                method_name=method.name,
                priority=method.priority,
                status=ExecutionStatus.TIMEOUT,
                error=f"Timeout after {duration:.2f}s",
                duration=duration,
                metadata=method.metadata
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[ERROR-MATRIX] {method.name} failed: {e}", exc_info=True)

            return MethodResult(
                method_name=method.name,
                priority=method.priority,
                status=ExecutionStatus.FAILED,
                error=str(e),
                duration=duration,
                metadata=method.metadata
            )

    def _finalize_result(
        self,
        operation_name: str,
        all_attempts: List[MethodResult],
        successful: List[MethodResult],
        collect_partial: bool
    ) -> Tuple[Any, ResultQuality, str]:
        """
        Finalize result and determine quality

        Returns:
            (final_result, quality, message)
        """
        if not successful:
            # All methods failed
            logger.error(f"[ERROR-MATRIX] All methods failed for {operation_name}")
            return None, ResultQuality.FAILED, f"All methods failed for {operation_name}"

        # Determine result quality based on which methods succeeded
        primary_succeeded = any(r.priority == ExecutionPriority.PRIMARY for r in successful)
        fallback_succeeded = any(r.priority == ExecutionPriority.FALLBACK for r in successful)

        if primary_succeeded:
            quality = ResultQuality.FULL
            message = f"{operation_name} completed successfully"
        elif fallback_succeeded:
            quality = ResultQuality.DEGRADED
            message = f"{operation_name} completed using fallback method"
        elif len(successful) > 1:
            quality = ResultQuality.PARTIAL
            message = f"{operation_name} completed with partial results"
        else:
            quality = ResultQuality.MINIMAL
            message = f"{operation_name} completed with minimal results"

        # Aggregate results if needed
        if collect_partial and len(successful) > 1:
            final_result, _ = self.aggregator.aggregate(successful)
        else:
            final_result = successful[0].result

        return final_result, quality, message

    async def execute_with_fallbacks(
        self,
        operation_name: str,
        primary: Callable,
        fallbacks: List[Callable],
        *args,
        **kwargs
    ) -> ExecutionReport:
        """
        Convenience method to execute with simple fallback list

        Args:
            operation_name: Name of operation
            primary: Primary method
            fallbacks: List of fallback methods
            *args: Arguments to pass to all methods
            **kwargs: Keyword arguments to pass to all methods

        Returns:
            ExecutionReport
        """
        chain = FallbackChain(operation_name)
        chain.add_primary(primary, *args, **kwargs)

        for i, fallback in enumerate(fallbacks):
            if i == 0:
                chain.add_fallback(fallback, *args, **kwargs)
            elif i == len(fallbacks) - 1:
                chain.add_last_resort(fallback, *args, **kwargs)
            else:
                chain.add_secondary(fallback, *args, **kwargs)

        return await self.execute_chain(chain)


# ============================================================================
# ERROR MESSAGE GENERATOR
# ============================================================================

class ErrorMessageGenerator:
    """
    Generates user-friendly error messages

    Features:
    - Context-aware messages
    - Actionable suggestions
    - Technical details (optional)
    """

    @staticmethod
    def generate_message(
        report: ExecutionReport,
        include_technical: bool = False,
        include_suggestions: bool = True
    ) -> str:
        """
        Generate user-friendly error message

        Args:
            report: ExecutionReport
            include_technical: Include technical details
            include_suggestions: Include actionable suggestions

        Returns:
            Formatted error message
        """
        if report.success:
            # Success message
            message = f"✅ {report.message}"

            if report.warnings:
                message += f"\n\n⚠️  Warnings:\n"
                for warning in report.warnings:
                    message += f"  • {warning}\n"

            return message

        # Failure message
        message = f"❌ {report.operation_name} failed\n\n"

        # Add error details
        if report.errors:
            message += "Errors encountered:\n"
            for error in report.errors[:3]:  # Limit to 3 errors
                message += f"  • {error}\n"

            if len(report.errors) > 3:
                message += f"  ... and {len(report.errors) - 3} more\n"

        # Add suggestions if enabled
        if include_suggestions:
            suggestions = ErrorMessageGenerator._generate_suggestions(report)
            if suggestions:
                message += f"\n💡 Suggestions:\n"
                for suggestion in suggestions:
                    message += f"  • {suggestion}\n"

        # Add technical details if enabled
        if include_technical:
            message += f"\n🔧 Technical Details:\n"
            message += f"  • Attempted {len(report.methods_attempted)} method(s)\n"
            message += f"  • Duration: {report.total_duration:.2f}s\n"

            for attempt in report.methods_attempted:
                status_icon = "✅" if attempt.status == ExecutionStatus.SUCCESS else "❌"
                message += f"  {status_icon} {attempt.method_name}: {attempt.status.value}\n"

        return message

    @staticmethod
    def _generate_suggestions(report: ExecutionReport) -> List[str]:
        """Generate actionable suggestions based on errors"""
        suggestions = []

        # Check for common error patterns
        error_text = " ".join(report.errors).lower()

        if "timeout" in error_text:
            suggestions.append("Try increasing the timeout value")
            suggestions.append("Check if the operation is running on a slow system")

        if "permission" in error_text:
            suggestions.append("Check system permissions in Settings > Privacy & Security")

        if "network" in error_text or "offline" in error_text:
            suggestions.append("Check your internet connection")

        if "api" in error_text or "rate limit" in error_text:
            suggestions.append("Check your API key and rate limits")

        if not suggestions:
            suggestions.append("Review the error details above")
            suggestions.append("Try the operation again")

        return suggestions


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_matrix: Optional[ErrorHandlingMatrix] = None


def get_error_handling_matrix() -> Optional[ErrorHandlingMatrix]:
    """Get the global error handling matrix instance"""
    return _global_matrix


def initialize_error_handling_matrix(
    default_timeout: float = 30.0,
    aggregation_strategy: str = "first_success",
    recovery_strategy: str = "continue"
) -> ErrorHandlingMatrix:
    """Initialize the global error handling matrix"""
    global _global_matrix
    _global_matrix = ErrorHandlingMatrix(
        default_timeout=default_timeout,
        aggregation_strategy=aggregation_strategy,
        recovery_strategy=recovery_strategy
    )
    logger.info("[ERROR-MATRIX] Global instance initialized")
    return _global_matrix
