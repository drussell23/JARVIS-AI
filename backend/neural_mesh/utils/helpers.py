"""
JARVIS Neural Mesh - Helper Utilities

Common utility functions used across the Neural Mesh system.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize an object to JSON, handling special types."""
    def default_handler(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, set):
            return list(o)
        elif hasattr(o, "to_dict"):
            return o.to_dict()
        elif hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)

    return json.dumps(obj, default=default_handler)


def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for async functions with retry logic.

    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Exceptions to catch and retry on

    Example:
        @async_retry(retries=3, delay=1.0)
        async def flaky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < retries:
                        logger.warning(
                            "Retry %d/%d for %s: %s",
                            attempt + 1,
                            retries,
                            func.__name__,
                            e,
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "All %d retries failed for %s: %s",
                            retries,
                            func.__name__,
                            e,
                        )

            raise last_exception  # type: ignore

        return wrapper
    return decorator


class measure_time:
    """
    Context manager to measure execution time.

    Usage:
        async with measure_time("operation"):
            await some_operation()

        # Or as decorator:
        @measure_time("function_name")
        async def some_function():
            ...
    """

    def __init__(self, name: str, log_level: int = logging.DEBUG) -> None:
        self.name = name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    async def __aenter__(self) -> "measure_time":
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            logger.log(
                self.log_level,
                "%s completed in %.2fms",
                self.name,
                self.elapsed_ms,
            )

    def __enter__(self) -> "measure_time":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            logger.log(
                self.log_level,
                "%s completed in %.2fms",
                self.name,
                self.elapsed_ms,
            )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with measure_time(self.name or func.__name__, self.log_level):
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with measure_time(self.name or func.__name__, self.log_level):
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore
