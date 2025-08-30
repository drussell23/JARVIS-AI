"""
Circuit Breaker Pattern for API Resilience
Prevents cascading failures and provides graceful degradation
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    state_changes: List[tuple] = None
    
    def __post_init__(self):
        if self.state_changes is None:
            self.state_changes = []

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker implementation with exponential backoff"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 half_open_max_calls: int = 3):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            expected_exception: Exception type to catch
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._stats = CircuitStats()
        
        # Callbacks for monitoring
        self._on_state_change = None
        self._on_failure = None
        self._on_success = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics"""
        return self._stats
    
    def _record_success(self):
        """Record successful call"""
        self._stats.successful_calls += 1
        self._stats.last_success_time = datetime.now()
        self._stats.consecutive_failures = 0
        
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._transition_to_closed()
        
        if self._on_success:
            self._on_success()
    
    def _record_failure(self):
        """Record failed call"""
        self._stats.failed_calls += 1
        self._stats.last_failure_time = datetime.now()
        self._stats.consecutive_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._transition_to_open()
        
        if self._on_failure:
            self._on_failure()
    
    def _transition_to_open(self):
        """Transition to open state"""
        if self._state != CircuitState.OPEN:
            old_state = self._state
            self._state = CircuitState.OPEN
            self._stats.state_changes.append((old_state, CircuitState.OPEN, datetime.now()))
            
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
            
            if self._on_state_change:
                self._on_state_change(old_state, CircuitState.OPEN)
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        if self._state != CircuitState.CLOSED:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._stats.state_changes.append((old_state, CircuitState.CLOSED, datetime.now()))
            
            logger.info("Circuit breaker closed - service recovered")
            
            if self._on_state_change:
                self._on_state_change(old_state, CircuitState.CLOSED)
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        if self._state == CircuitState.OPEN:
            old_state = self._state
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._stats.state_changes.append((old_state, CircuitState.HALF_OPEN, datetime.now()))
            
            logger.info("Circuit breaker half-open - testing service")
            
            if self._on_state_change:
                self._on_state_change(old_state, CircuitState.HALF_OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try reset"""
        return (self._last_failure_time and 
                time.time() - self._last_failure_time >= self.recovery_timeout)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function fails
        """
        self._stats.total_calls += 1
        
        # Check if circuit should transition from open to half-open
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._transition_to_half_open()
        
        # Reject call if circuit is open
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker is open. Service unavailable for "
                f"{self.recovery_timeout - (time.time() - self._last_failure_time):.0f} more seconds"
            )
        
        # Reject call if half-open and already testing
        if self._state == CircuitState.HALF_OPEN and self._half_open_calls >= self.half_open_max_calls:
            raise CircuitBreakerError("Circuit breaker is half-open and testing capacity reached")
        
        try:
            # Make the actual call
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            
            # If half-open, immediately go back to open
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            
            raise e
    
    def set_callbacks(self, 
                     on_state_change: Optional[Callable] = None,
                     on_failure: Optional[Callable] = None,
                     on_success: Optional[Callable] = None):
        """Set monitoring callbacks"""
        self._on_state_change = on_state_change
        self._on_failure = on_failure
        self._on_success = on_success
    
    def manual_reset(self):
        """Manually reset circuit breaker to closed state"""
        self._transition_to_closed()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status"""
        success_rate = 0
        if self._stats.total_calls > 0:
            success_rate = self._stats.successful_calls / self._stats.total_calls
        
        return {
            "state": self._state.value,
            "is_healthy": self._state == CircuitState.CLOSED,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "success_rate": success_rate,
                "consecutive_failures": self._stats.consecutive_failures,
                "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
                "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls
            }
        }

class MultiServiceCircuitBreaker:
    """Manage multiple circuit breakers for different services"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = {
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "expected_exception": Exception,
            "half_open_max_calls": 3
        }
    
    def add_service(self, service_name: str, **config):
        """Add a new service with circuit breaker"""
        breaker_config = {**self._default_config, **config}
        self._breakers[service_name] = CircuitBreaker(**breaker_config)
        logger.info(f"Added circuit breaker for service: {service_name}")
    
    def get_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for a service"""
        if service_name not in self._breakers:
            self.add_service(service_name)
        return self._breakers[service_name]
    
    async def call(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """Call function through service's circuit breaker"""
        breaker = self.get_breaker(service_name)
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status of all services"""
        return {
            service: breaker.get_health_status()
            for service, breaker in self._breakers.items()
        }
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services"""
        return [
            service for service, breaker in self._breakers.items()
            if breaker.state != CircuitState.CLOSED
        ]