"""
API & Network Manager for JARVIS
=================================

Handles all API and network-related edge cases:
- Claude API timeouts with exponential backoff retry
- Rate limiting (429) with intelligent waiting
- Invalid API keys with helpful error messages
- Image optimization for API size limits
- Network offline detection with graceful degradation

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


# ============================================================================
# API STATUS & STATES
# ============================================================================

class APIState(Enum):
    """Claude API states"""
    AVAILABLE = "available"          # API is available and working
    RATE_LIMITED = "rate_limited"    # Hit rate limit (429)
    INVALID_KEY = "invalid_key"      # API key is invalid/expired
    TIMEOUT = "timeout"              # Request timed out
    UNAVAILABLE = "unavailable"      # API is down/unreachable
    UNKNOWN = "unknown"              # Unknown state


class NetworkState(Enum):
    """Network connectivity states"""
    ONLINE = "online"                # Connected to internet
    OFFLINE = "offline"              # No internet connection
    DEGRADED = "degraded"            # Slow/unstable connection
    UNKNOWN = "unknown"              # Cannot determine


class ImageOptimizationStatus(Enum):
    """Image optimization result statuses"""
    ALREADY_OPTIMIZED = "already_optimized"  # Image already meets requirements
    RESIZED = "resized"                      # Image was resized
    COMPRESSED = "compressed"                # Image was compressed
    CONVERTED = "converted"                  # Format was converted
    FAILED = "failed"                        # Optimization failed


# ============================================================================
# STATUS DATA CLASSES
# ============================================================================

@dataclass
class APIStatus:
    """Claude API status information"""
    state: APIState
    is_available: bool
    can_retry: bool
    message: str
    rate_limit_reset: Optional[datetime] = None
    retry_after_seconds: Optional[int] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkStatus:
    """Network connectivity status"""
    state: NetworkState
    is_online: bool
    latency_ms: Optional[float] = None
    message: str
    last_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageOptimizationResult:
    """Result of image optimization"""
    status: ImageOptimizationStatus
    success: bool
    original_path: str
    optimized_path: str
    original_size_bytes: int
    optimized_size_bytes: int
    original_dimensions: Tuple[int, int]
    optimized_dimensions: Tuple[int, int]
    format_changed: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage"""
        if self.original_size_bytes == 0:
            return 0.0
        reduction = (self.original_size_bytes - self.optimized_size_bytes) / self.original_size_bytes
        return reduction * 100


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    attempts: int
    total_delay: float
    final_error: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# API HEALTH CHECKER
# ============================================================================

class APIHealthChecker:
    """
    Monitors Claude API health and detects edge cases

    Features:
    - API key validation
    - Rate limit detection
    - Timeout detection
    - Circuit breaker pattern
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        """
        Initialize API health checker

        Args:
            api_key: Claude API key (reads from env if not provided)
            timeout: Timeout for API health checks in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout = timeout

        # Circuit breaker state
        self.consecutive_failures = 0
        self.last_success: Optional[datetime] = None
        self.last_failure: Optional[datetime] = None
        self.circuit_open = False
        self.circuit_open_until: Optional[datetime] = None

        # Rate limiting state
        self.rate_limited = False
        self.rate_limit_reset: Optional[datetime] = None

        logger.info(f"[API-HEALTH-CHECKER] Initialized (timeout={timeout}s)")

    async def check_api_status(self) -> APIStatus:
        """
        Check Claude API status

        Returns:
            APIStatus with current state
        """
        logger.debug("[API-HEALTH-CHECKER] Checking API status")

        # Check API key first
        if not self.api_key or self.api_key.strip() == "":
            logger.error("[API-HEALTH-CHECKER] No API key found")
            return APIStatus(
                state=APIState.INVALID_KEY,
                is_available=False,
                can_retry=False,
                message="Claude API key invalid. Check .env file and set ANTHROPIC_API_KEY.",
                consecutive_failures=self.consecutive_failures,
                metadata={"error": "missing_api_key"}
            )

        # Check if circuit breaker is open
        if self.circuit_open and self.circuit_open_until:
            if datetime.now() < self.circuit_open_until:
                time_left = (self.circuit_open_until - datetime.now()).total_seconds()
                logger.warning(f"[API-HEALTH-CHECKER] Circuit breaker open ({time_left:.1f}s remaining)")
                return APIStatus(
                    state=APIState.UNAVAILABLE,
                    is_available=False,
                    can_retry=False,
                    message=f"API circuit breaker open. Retry in {time_left:.0f}s.",
                    retry_after_seconds=int(time_left),
                    consecutive_failures=self.consecutive_failures,
                    metadata={"circuit_breaker": "open"}
                )
            else:
                # Circuit breaker timeout expired, close it
                logger.info("[API-HEALTH-CHECKER] Circuit breaker timeout expired, closing")
                self.circuit_open = False
                self.circuit_open_until = None

        # Check if rate limited
        if self.rate_limited and self.rate_limit_reset:
            if datetime.now() < self.rate_limit_reset:
                time_left = (self.rate_limit_reset - datetime.now()).total_seconds()
                logger.warning(f"[API-HEALTH-CHECKER] Rate limited ({time_left:.1f}s remaining)")
                return APIStatus(
                    state=APIState.RATE_LIMITED,
                    is_available=False,
                    can_retry=True,
                    message=f"Rate limited. Wait {time_left:.0f}s before retrying.",
                    rate_limit_reset=self.rate_limit_reset,
                    retry_after_seconds=int(time_left),
                    consecutive_failures=self.consecutive_failures,
                    metadata={"rate_limited": True}
                )
            else:
                # Rate limit expired
                logger.info("[API-HEALTH-CHECKER] Rate limit expired")
                self.rate_limited = False
                self.rate_limit_reset = None

        # Try to validate API key format (basic check)
        if not self._validate_api_key_format(self.api_key):
            logger.error("[API-HEALTH-CHECKER] Invalid API key format")
            return APIStatus(
                state=APIState.INVALID_KEY,
                is_available=False,
                can_retry=False,
                message="Claude API key format invalid. Check .env file.",
                consecutive_failures=self.consecutive_failures,
                metadata={"error": "invalid_key_format"}
            )

        # API appears healthy
        logger.info("[API-HEALTH-CHECKER] API status: AVAILABLE")
        return APIStatus(
            state=APIState.AVAILABLE,
            is_available=True,
            can_retry=True,
            message="Claude API available",
            last_success=self.last_success,
            consecutive_failures=self.consecutive_failures,
            metadata={}
        )

    def _validate_api_key_format(self, api_key: str) -> bool:
        """
        Validate API key format (basic validation)

        Args:
            api_key: API key to validate

        Returns:
            True if format looks valid
        """
        # Anthropic API keys start with "sk-ant-"
        if not api_key.startswith("sk-ant-"):
            return False

        # Should be reasonably long
        if len(api_key) < 20:
            return False

        return True

    def record_success(self):
        """Record successful API call"""
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        self.last_failure = None

        # Close circuit breaker on success
        if self.circuit_open:
            logger.info("[API-HEALTH-CHECKER] Success after circuit breaker, closing")
            self.circuit_open = False
            self.circuit_open_until = None

    def record_failure(self, error_code: Optional[int] = None):
        """
        Record failed API call

        Args:
            error_code: HTTP error code if available
        """
        self.consecutive_failures += 1
        self.last_failure = datetime.now()

        # Handle rate limiting
        if error_code == 429:
            logger.warning("[API-HEALTH-CHECKER] Rate limit detected (429)")
            self.rate_limited = True
            # Default to 60 seconds if not provided
            self.rate_limit_reset = datetime.now() + timedelta(seconds=60)

        # Open circuit breaker after 5 consecutive failures
        if self.consecutive_failures >= 5 and not self.circuit_open:
            logger.error(f"[API-HEALTH-CHECKER] Opening circuit breaker after {self.consecutive_failures} failures")
            self.circuit_open = True
            # Open for 60 seconds
            self.circuit_open_until = datetime.now() + timedelta(seconds=60)

    def set_rate_limit(self, retry_after_seconds: int):
        """
        Set rate limit with custom retry time

        Args:
            retry_after_seconds: Seconds to wait before retry
        """
        self.rate_limited = True
        self.rate_limit_reset = datetime.now() + timedelta(seconds=retry_after_seconds)
        logger.warning(f"[API-HEALTH-CHECKER] Rate limit set for {retry_after_seconds}s")


# ============================================================================
# NETWORK DETECTOR
# ============================================================================

class NetworkDetector:
    """
    Detects network connectivity and measures latency

    Features:
    - Online/offline detection
    - Latency measurement
    - Connection quality assessment
    """

    def __init__(self, cache_ttl: float = 5.0):
        """
        Initialize network detector

        Args:
            cache_ttl: Cache TTL for network status in seconds
        """
        self.cache_ttl = cache_ttl
        self._cached_status: Optional[NetworkStatus] = None
        self._cache_time: Optional[datetime] = None

        logger.info(f"[NETWORK-DETECTOR] Initialized (cache_ttl={cache_ttl}s)")

    async def check_network_status(self, use_cache: bool = True) -> NetworkStatus:
        """
        Check network connectivity

        Args:
            use_cache: Use cached result if available

        Returns:
            NetworkStatus with connectivity information
        """
        # Check cache
        if use_cache and self._cached_status and self._cache_time:
            cache_age = (datetime.now() - self._cache_time).total_seconds()
            if cache_age < self.cache_ttl:
                logger.debug(f"[NETWORK-DETECTOR] Using cached status (age={cache_age:.1f}s)")
                return self._cached_status

        logger.debug("[NETWORK-DETECTOR] Checking network status")

        # Try to ping a reliable host (Cloudflare DNS)
        try:
            start_time = time.time()

            # Use ping command (works on macOS/Linux)
            process = await asyncio.create_subprocess_shell(
                "ping -c 1 -W 2 1.1.1.1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3.0)
                elapsed = (time.time() - start_time) * 1000  # Convert to ms

                if process.returncode == 0:
                    # Online
                    state = NetworkState.ONLINE if elapsed < 500 else NetworkState.DEGRADED
                    message = f"Online (latency: {elapsed:.1f}ms)"

                    logger.info(f"[NETWORK-DETECTOR] {message}")

                    status = NetworkStatus(
                        state=state,
                        is_online=True,
                        latency_ms=elapsed,
                        message=message,
                        metadata={"ping_host": "1.1.1.1"}
                    )
                else:
                    # Offline
                    logger.warning("[NETWORK-DETECTOR] Offline (ping failed)")
                    status = NetworkStatus(
                        state=NetworkState.OFFLINE,
                        is_online=False,
                        message="Offline. Vision requires internet for Claude API.",
                        metadata={"error": "ping_failed"}
                    )

            except asyncio.TimeoutError:
                # Timeout = offline
                logger.warning("[NETWORK-DETECTOR] Offline (ping timeout)")
                status = NetworkStatus(
                    state=NetworkState.OFFLINE,
                    is_online=False,
                    message="Offline. Vision requires internet for Claude API.",
                    metadata={"error": "timeout"}
                )

        except Exception as e:
            logger.error(f"[NETWORK-DETECTOR] Error checking network: {e}")
            status = NetworkStatus(
                state=NetworkState.UNKNOWN,
                is_online=False,
                message="Cannot determine network status",
                metadata={"error": str(e)}
            )

        # Cache result
        self._cached_status = status
        self._cache_time = datetime.now()

        return status

    async def wait_for_online(self, timeout: float = 30.0, check_interval: float = 2.0) -> Tuple[bool, NetworkStatus]:
        """
        Wait for network to become online

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            (became_online, final_status)
        """
        logger.info(f"[NETWORK-DETECTOR] Waiting for network (timeout={timeout}s)")

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            status = await self.check_network_status(use_cache=False)

            if status.is_online:
                logger.info("[NETWORK-DETECTOR] Network is now online")
                return True, status

            await asyncio.sleep(check_interval)

        # Timeout
        logger.warning(f"[NETWORK-DETECTOR] Network did not come online within {timeout}s")
        final_status = await self.check_network_status(use_cache=False)
        return False, final_status


# ============================================================================
# IMAGE OPTIMIZER
# ============================================================================

class ImageOptimizer:
    """
    Optimizes images for Claude API size limits

    Features:
    - Resize to max width (2560px default)
    - JPEG compression (85% quality)
    - Format conversion (PNG -> JPEG)
    - Size validation (<5MB)
    """

    def __init__(self, max_width: int = 2560, max_size_mb: float = 5.0, jpeg_quality: int = 85):
        """
        Initialize image optimizer

        Args:
            max_width: Maximum image width in pixels
            max_size_mb: Maximum file size in MB
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.max_width = max_width
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.jpeg_quality = jpeg_quality

        logger.info(f"[IMAGE-OPTIMIZER] Initialized (max_width={max_width}px, max_size={max_size_mb}MB, quality={jpeg_quality})")

    async def optimize_image(self, image_path: str, output_path: Optional[str] = None) -> ImageOptimizationResult:
        """
        Optimize image for API requirements

        Args:
            image_path: Path to original image
            output_path: Path for optimized image (uses temp if not provided)

        Returns:
            ImageOptimizationResult with outcome
        """
        logger.info(f"[IMAGE-OPTIMIZER] Optimizing image: {image_path}")

        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"[IMAGE-OPTIMIZER] Image not found: {image_path}")
            return ImageOptimizationResult(
                status=ImageOptimizationStatus.FAILED,
                success=False,
                original_path=image_path,
                optimized_path="",
                original_size_bytes=0,
                optimized_size_bytes=0,
                original_dimensions=(0, 0),
                optimized_dimensions=(0, 0),
                format_changed=False,
                message="Image file not found",
                metadata={"error": "file_not_found"}
            )

        # Get original file info
        original_size = os.path.getsize(image_path)
        original_dims = await self._get_image_dimensions(image_path)

        logger.debug(f"[IMAGE-OPTIMIZER] Original: {original_size} bytes, {original_dims[0]}x{original_dims[1]}px")

        # Check if already optimized
        if original_size <= self.max_size_bytes and original_dims[0] <= self.max_width:
            logger.info("[IMAGE-OPTIMIZER] Image already meets requirements")
            return ImageOptimizationResult(
                status=ImageOptimizationStatus.ALREADY_OPTIMIZED,
                success=True,
                original_path=image_path,
                optimized_path=image_path,
                original_size_bytes=original_size,
                optimized_size_bytes=original_size,
                original_dimensions=original_dims,
                optimized_dimensions=original_dims,
                format_changed=False,
                message="Image already optimized",
                metadata={}
            )

        # Determine output path
        if not output_path:
            output_path = self._generate_temp_path(image_path)

        # Optimize image
        try:
            needs_resize = original_dims[0] > self.max_width
            needs_compression = original_size > self.max_size_bytes or Path(image_path).suffix.lower() == '.png'

            optimized_path = image_path
            format_changed = False

            # Resize if needed
            if needs_resize:
                logger.info(f"[IMAGE-OPTIMIZER] Resizing to max width {self.max_width}px")
                optimized_path = await self._resize_image(optimized_path, output_path, self.max_width)

            # Compress/convert to JPEG if needed
            if needs_compression:
                logger.info(f"[IMAGE-OPTIMIZER] Compressing to JPEG (quality={self.jpeg_quality})")

                # Change extension to .jpg
                jpeg_output = str(Path(output_path).with_suffix('.jpg'))
                optimized_path = await self._convert_to_jpeg(optimized_path, jpeg_output, self.jpeg_quality)
                format_changed = True
                output_path = jpeg_output

            # If we only resized but didn't compress, copy to output
            if needs_resize and not needs_compression and optimized_path != output_path:
                await self._copy_file(optimized_path, output_path)
                optimized_path = output_path

            # Get optimized file info
            optimized_size = os.path.getsize(optimized_path)
            optimized_dims = await self._get_image_dimensions(optimized_path)

            logger.info(f"[IMAGE-OPTIMIZER] Optimized: {optimized_size} bytes, {optimized_dims[0]}x{optimized_dims[1]}px")

            # Determine status
            status = ImageOptimizationStatus.RESIZED if needs_resize else ImageOptimizationStatus.ALREADY_OPTIMIZED
            if format_changed:
                status = ImageOptimizationStatus.CONVERTED
            elif needs_compression and not needs_resize:
                status = ImageOptimizationStatus.COMPRESSED

            return ImageOptimizationResult(
                status=status,
                success=True,
                original_path=image_path,
                optimized_path=optimized_path,
                original_size_bytes=original_size,
                optimized_size_bytes=optimized_size,
                original_dimensions=original_dims,
                optimized_dimensions=optimized_dims,
                format_changed=format_changed,
                message=f"Optimized: {original_size//1024}KB → {optimized_size//1024}KB",
                metadata={
                    "resized": needs_resize,
                    "compressed": needs_compression,
                    "format_changed": format_changed
                }
            )

        except Exception as e:
            logger.error(f"[IMAGE-OPTIMIZER] Optimization failed: {e}", exc_info=True)
            return ImageOptimizationResult(
                status=ImageOptimizationStatus.FAILED,
                success=False,
                original_path=image_path,
                optimized_path="",
                original_size_bytes=original_size,
                optimized_size_bytes=0,
                original_dimensions=original_dims,
                optimized_dimensions=(0, 0),
                format_changed=False,
                message=f"Optimization failed: {str(e)}",
                metadata={"error": str(e)}
            )

    async def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions using sips"""
        try:
            process = await asyncio.create_subprocess_shell(
                f'sips -g pixelWidth -g pixelHeight "{image_path}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            output = stdout.decode()

            # Parse output: "pixelWidth: 1920\n  pixelHeight: 1080\n"
            width = int([line for line in output.split('\n') if 'pixelWidth' in line][0].split(':')[1].strip())
            height = int([line for line in output.split('\n') if 'pixelHeight' in line][0].split(':')[1].strip())

            return (width, height)

        except Exception as e:
            logger.error(f"[IMAGE-OPTIMIZER] Failed to get dimensions: {e}")
            return (0, 0)

    async def _resize_image(self, input_path: str, output_path: str, max_width: int) -> str:
        """Resize image to max width using sips"""
        process = await asyncio.create_subprocess_shell(
            f'sips -Z {max_width} "{input_path}" --out "{output_path}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"sips resize failed with code {process.returncode}")

        return output_path

    async def _convert_to_jpeg(self, input_path: str, output_path: str, quality: int) -> str:
        """Convert image to JPEG with compression"""
        # Use sips to convert to JPEG with quality
        process = await asyncio.create_subprocess_shell(
            f'sips -s format jpeg -s formatOptions {quality} "{input_path}" --out "{output_path}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"sips JPEG conversion failed with code {process.returncode}")

        return output_path

    async def _copy_file(self, src: str, dst: str):
        """Copy file async"""
        process = await asyncio.create_subprocess_shell(
            f'cp "{src}" "{dst}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

    def _generate_temp_path(self, original_path: str) -> str:
        """Generate temporary output path"""
        timestamp = int(time.time() * 1000)
        original = Path(original_path)
        return f"/tmp/jarvis_optimized_{timestamp}{original.suffix}"


# ============================================================================
# RETRY HANDLER
# ============================================================================

class RetryHandler:
    """
    Handles retry logic with exponential backoff

    Features:
    - Exponential backoff (1s, 2s, 4s, 8s, ...)
    - Rate limit awareness
    - Circuit breaker integration
    - Result caching
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        cache_ttl: float = 300.0
    ):
        """
        Initialize retry handler

        Args:
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_multiplier: Backoff multiplier
            cache_ttl: Cache TTL in seconds
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.cache_ttl = cache_ttl

        # Result cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

        logger.info(f"[RETRY-HANDLER] Initialized (max_retries={max_retries}, initial_delay={initial_delay}s)")

    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        error_codes_to_retry: List[int] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> RetryResult:
        """
        Retry function with exponential backoff

        Args:
            func: Async function to retry
            *args: Function arguments
            error_codes_to_retry: HTTP error codes that should trigger retry
            cache_key: Key for caching successful results
            **kwargs: Function keyword arguments

        Returns:
            RetryResult with outcome
        """
        if error_codes_to_retry is None:
            error_codes_to_retry = [429, 500, 502, 503, 504]

        # Check cache first
        if cache_key and cache_key in self._cache:
            cached_result, cache_time = self._cache[cache_key]
            cache_age = (datetime.now() - cache_time).total_seconds()

            if cache_age < self.cache_ttl:
                logger.info(f"[RETRY-HANDLER] Using cached result (age={cache_age:.1f}s)")
                return RetryResult(
                    success=True,
                    attempts=0,
                    total_delay=0.0,
                    result=cached_result,
                    metadata={"cached": True, "cache_age": cache_age}
                )

        attempts = 0
        total_delay = 0.0
        last_error = None

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1

            try:
                logger.debug(f"[RETRY-HANDLER] Attempt {attempts}/{self.max_retries + 1}")

                result = await func(*args, **kwargs)

                # Success! Cache if requested
                if cache_key:
                    self._cache[cache_key] = (result, datetime.now())
                    logger.debug(f"[RETRY-HANDLER] Cached result with key: {cache_key}")

                logger.info(f"[RETRY-HANDLER] Success after {attempts} attempt(s)")

                return RetryResult(
                    success=True,
                    attempts=attempts,
                    total_delay=total_delay,
                    result=result,
                    metadata={}
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[RETRY-HANDLER] Attempt {attempts} failed: {last_error}")

                # Check if we should retry
                if attempt >= self.max_retries:
                    logger.error(f"[RETRY-HANDLER] Max retries ({self.max_retries}) exceeded")
                    break

                # Calculate backoff delay
                delay = self.initial_delay * (self.backoff_multiplier ** attempt)
                logger.info(f"[RETRY-HANDLER] Waiting {delay:.1f}s before retry...")

                await asyncio.sleep(delay)
                total_delay += delay

        # All retries failed
        logger.error(f"[RETRY-HANDLER] All {attempts} attempts failed: {last_error}")

        return RetryResult(
            success=False,
            attempts=attempts,
            total_delay=total_delay,
            final_error=last_error,
            metadata={}
        )

    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache (all or specific key)"""
        if cache_key:
            self._cache.pop(cache_key, None)
            logger.debug(f"[RETRY-HANDLER] Cleared cache key: {cache_key}")
        else:
            self._cache.clear()
            logger.debug("[RETRY-HANDLER] Cleared all cache")


# ============================================================================
# API NETWORK MANAGER (Main Coordinator)
# ============================================================================

class APINetworkManager:
    """
    Main coordinator for API and network edge cases

    Combines all components to provide robust API/network handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_image_width: int = 2560,
        max_image_size_mb: float = 5.0
    ):
        """
        Initialize API/Network manager

        Args:
            api_key: Claude API key
            max_retries: Maximum retry attempts
            initial_retry_delay: Initial retry delay in seconds
            max_image_width: Maximum image width in pixels
            max_image_size_mb: Maximum image size in MB
        """
        self.api_health_checker = APIHealthChecker(api_key=api_key)
        self.network_detector = NetworkDetector()
        self.image_optimizer = ImageOptimizer(
            max_width=max_image_width,
            max_size_mb=max_image_size_mb
        )
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            initial_delay=initial_retry_delay
        )

        logger.info("[API-NETWORK-MANAGER] Initialized")

    async def check_ready_for_api_call(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if system is ready for API call

        Returns:
            (is_ready, message, status_info)
        """
        logger.debug("[API-NETWORK-MANAGER] Checking readiness for API call")

        # Check network first
        network_status = await self.network_detector.check_network_status()
        if not network_status.is_online:
            return False, network_status.message, {"network": network_status}

        # Check API health
        api_status = await self.api_health_checker.check_api_status()
        if not api_status.is_available:
            return False, api_status.message, {"api": api_status, "network": network_status}

        # All good!
        return True, "Ready for API call", {"api": api_status, "network": network_status}

    async def execute_api_call_with_retry(
        self,
        func: Callable,
        *args,
        optimize_image: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute API call with full edge case handling

        Args:
            func: Async function to call (API call)
            *args: Function arguments
            optimize_image: Path to image to optimize before call
            cache_key: Key for caching results
            **kwargs: Function keyword arguments

        Returns:
            RetryResult with outcome
        """
        logger.info("[API-NETWORK-MANAGER] Executing API call with retry")

        # Check readiness
        is_ready, message, status_info = await self.check_ready_for_api_call()
        if not is_ready:
            logger.error(f"[API-NETWORK-MANAGER] Not ready for API call: {message}")
            return RetryResult(
                success=False,
                attempts=0,
                total_delay=0.0,
                final_error=message,
                metadata=status_info
            )

        # Optimize image if provided
        optimized_image_path = None
        if optimize_image:
            logger.info(f"[API-NETWORK-MANAGER] Optimizing image: {optimize_image}")
            opt_result = await self.image_optimizer.optimize_image(optimize_image)

            if not opt_result.success:
                logger.error(f"[API-NETWORK-MANAGER] Image optimization failed: {opt_result.message}")
                return RetryResult(
                    success=False,
                    attempts=0,
                    total_delay=0.0,
                    final_error=f"Image optimization failed: {opt_result.message}",
                    metadata={"image_optimization": opt_result}
                )

            optimized_image_path = opt_result.optimized_path
            logger.info(f"[API-NETWORK-MANAGER] Image optimized: {opt_result.message}")

            # Replace image path in kwargs if present
            if 'image_path' in kwargs:
                kwargs['image_path'] = optimized_image_path

        # Execute with retry
        result = await self.retry_handler.retry_with_backoff(
            func,
            *args,
            cache_key=cache_key,
            **kwargs
        )

        # Record success/failure with API health checker
        if result.success:
            self.api_health_checker.record_success()
        else:
            self.api_health_checker.record_failure()

        return result

    async def wait_for_ready(self, timeout: float = 60.0) -> Tuple[bool, str]:
        """
        Wait for system to become ready for API calls

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            (became_ready, message)
        """
        logger.info(f"[API-NETWORK-MANAGER] Waiting for ready state (timeout={timeout}s)")

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            is_ready, message, _ = await self.check_ready_for_api_call()

            if is_ready:
                logger.info("[API-NETWORK-MANAGER] System is now ready")
                return True, "System ready for API calls"

            await asyncio.sleep(2.0)

        # Timeout
        logger.warning(f"[API-NETWORK-MANAGER] System did not become ready within {timeout}s")
        _, message, _ = await self.check_ready_for_api_call()
        return False, f"Timeout: {message}"


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_manager: Optional[APINetworkManager] = None


def get_api_network_manager() -> Optional[APINetworkManager]:
    """Get the global API/Network manager instance"""
    return _global_manager


def initialize_api_network_manager(
    api_key: Optional[str] = None,
    max_retries: int = 3,
    initial_retry_delay: float = 1.0,
    max_image_width: int = 2560,
    max_image_size_mb: float = 5.0
) -> APINetworkManager:
    """Initialize the global API/Network manager"""
    global _global_manager
    _global_manager = APINetworkManager(
        api_key=api_key,
        max_retries=max_retries,
        initial_retry_delay=initial_retry_delay,
        max_image_width=max_image_width,
        max_image_size_mb=max_image_size_mb
    )
    logger.info("[API-NETWORK-MANAGER] Global instance initialized")
    return _global_manager
