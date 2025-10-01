"""
Claude API Content Streamer
===========================

Advanced, robust streaming content generation from Claude API with:
- Intelligent retry logic with exponential backoff
- Multiple model support with automatic fallback
- Real-time streaming statistics and monitoring
- Token estimation and rate limiting
- Caching for outline generation
- Comprehensive error handling
"""

import asyncio
import logging
from typing import AsyncIterator, Optional, Callable, Dict, Any, List
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available - install with: pip install anthropic")


@dataclass
class StreamingStats:
    """Statistics for streaming session"""
    start_time: float = field(default_factory=time.time)
    total_tokens: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    errors_encountered: int = 0
    retries_attempted: int = 0
    model_used: str = ""

    def get_duration(self) -> float:
        """Get streaming duration in seconds"""
        return time.time() - self.start_time

    def get_tokens_per_second(self) -> float:
        """Calculate tokens per second"""
        duration = self.get_duration()
        return self.total_tokens / duration if duration > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "duration_seconds": self.get_duration(),
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "tokens_per_second": self.get_tokens_per_second(),
            "errors": self.errors_encountered,
            "retries": self.retries_attempted,
            "model": self.model_used
        }


class ClaudeContentStreamer:
    """Advanced streaming content generation from Claude API with full async support"""

    # Model configurations with fallback chain
    MODELS = {
        "primary": "claude-3-5-sonnet-20241022",
        "fallback": "claude-3-5-sonnet-20240620",
        "budget": "claude-3-haiku-20240307"
    }

    # Rate limiting configuration
    MAX_TOKENS_PER_MINUTE = 40000
    MAX_REQUESTS_PER_MINUTE = 50

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize advanced Claude streamer

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self._client = None
        self._outline_cache: Dict[str, Dict[str, Any]] = {}
        self._request_times: List[float] = []
        self._token_counts: List[int] = []
        self._session_stats: List[StreamingStats] = []

        if not self.api_key:
            logger.warning("No Anthropic API key provided - streaming will use mock data")
        else:
            logger.info("Claude streamer initialized with API key")

    async def __aenter__(self):
        """Async context manager entry"""
        self._ensure_client()
        logger.info("Claude streamer context entered")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        logger.info("Claude streamer context exiting")
        # Log session statistics
        if self._session_stats:
            total_tokens = sum(s.total_tokens for s in self._session_stats)
            total_duration = sum(s.get_duration() for s in self._session_stats)
            logger.info(f"Session complete: {total_tokens} tokens in {total_duration:.1f}s")
        return False

    def _ensure_client(self):
        """Ensure we have a Claude API client"""
        if not ANTHROPIC_AVAILABLE:
            return False

        if self._client is None and self.api_key:
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✓ Claude API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude API client: {e}")
                return False

        return self._client is not None

    def _check_rate_limits(self, estimated_tokens: int = 0) -> bool:
        """
        Check if we're within rate limits

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            True if within limits, False otherwise
        """
        current_time = time.time()
        one_minute_ago = current_time - 60

        # Clean up old entries
        self._request_times = [t for t in self._request_times if t > one_minute_ago]
        self._token_counts = self._token_counts[-len(self._request_times):]

        # Check request limit
        if len(self._request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit: Too many requests ({len(self._request_times)}/min)")
            return False

        # Check token limit
        total_tokens_this_minute = sum(self._token_counts)
        if total_tokens_this_minute + estimated_tokens > self.MAX_TOKENS_PER_MINUTE:
            logger.warning(f"Rate limit: Too many tokens ({total_tokens_this_minute}/min)")
            return False

        return True

    def _record_request(self, tokens: int = 0):
        """Record a request for rate limiting"""
        self._request_times.append(time.time())
        self._token_counts.append(tokens)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()

    async def _retry_with_backoff(self,
                                  func: Callable,
                                  max_retries: int = 3,
                                  base_delay: float = 1.0,
                                  *args,
                                  **kwargs) -> Any:
        """
        Execute function with exponential backoff retry

        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except anthropic.RateLimitError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached for rate limit")
            except anthropic.APIError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached for API error")
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error: {e}")
                break

        raise last_exception if last_exception else Exception("Unknown error in retry logic")

    async def stream_content(self,
                           prompt: str,
                           max_tokens: int = 4096,
                           model: Optional[str] = None,
                           chunk_callback: Optional[Callable[[str], None]] = None,
                           stats_callback: Optional[Callable[[StreamingStats], None]] = None) -> AsyncIterator[str]:
        """
        Advanced async streaming with automatic retry, fallback, and statistics

        Args:
            prompt: The content generation prompt
            max_tokens: Maximum tokens to generate
            model: Claude model to use (defaults to primary model)
            chunk_callback: Optional callback for each chunk
            stats_callback: Optional callback for streaming statistics

        Yields:
            Content chunks as they're generated
        """
        # Initialize statistics
        stats = StreamingStats()
        model = model or self.MODELS["primary"]
        stats.model_used = model

        if not self._ensure_client():
            logger.info("Using mock content streaming (no API client)")
            async for chunk in self._mock_stream(prompt):
                stats.total_chunks += 1
                stats.total_chars += len(chunk)
                if chunk_callback:
                    chunk_callback(chunk)
                if stats_callback:
                    stats_callback(stats)
                yield chunk
            return

        # Check rate limits
        estimated_tokens = self._estimate_tokens(prompt) + max_tokens
        if not self._check_rate_limits(estimated_tokens):
            logger.warning("Rate limit reached, waiting before streaming...")
            await asyncio.sleep(2)

        # Try streaming with model fallback chain
        models_to_try = [model, self.MODELS["fallback"], self.MODELS["budget"]]
        last_error = None

        for attempt, current_model in enumerate(models_to_try):
            try:
                logger.info(f"Attempting to stream with {current_model}")
                stats.model_used = current_model

                # Record request for rate limiting
                self._record_request(estimated_tokens)

                # Stream with current model
                async for chunk in self._stream_with_model(
                    current_model,
                    prompt,
                    max_tokens,
                    stats,
                    chunk_callback,
                    stats_callback
                ):
                    yield chunk

                # Success - exit loop
                logger.info(f"✓ Streaming completed successfully with {current_model}")
                if stats_callback:
                    stats_callback(stats)
                return

            except anthropic.RateLimitError as e:
                last_error = e
                stats.errors_encountered += 1
                logger.warning(f"Rate limit with {current_model}, trying next model...")
                await asyncio.sleep(2)
                continue

            except anthropic.APIError as e:
                last_error = e
                stats.errors_encountered += 1
                if attempt < len(models_to_try) - 1:
                    logger.warning(f"API error with {current_model}: {e}, trying fallback...")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"All models failed, falling back to mock")
                    break

            except Exception as e:
                last_error = e
                stats.errors_encountered += 1
                logger.error(f"Unexpected error with {current_model}: {e}")
                break

        # All models failed - use mock
        logger.warning("All Claude models failed, using mock streaming")
        async for chunk in self._mock_stream(prompt):
            stats.total_chunks += 1
            stats.total_chars += len(chunk)
            if chunk_callback:
                chunk_callback(chunk)
            if stats_callback:
                stats_callback(stats)
            yield chunk

    async def _stream_with_model(self,
                                 model: str,
                                 prompt: str,
                                 max_tokens: int,
                                 stats: StreamingStats,
                                 chunk_callback: Optional[Callable],
                                 stats_callback: Optional[Callable]) -> AsyncIterator[str]:
        """
        Internal method to stream with a specific model

        Args:
            model: Model to use
            prompt: Content prompt
            max_tokens: Maximum tokens
            stats: Statistics object to update
            chunk_callback: Optional chunk callback
            stats_callback: Optional stats callback

        Yields:
            Content chunks
        """
        # Run streaming in executor to avoid blocking
        def sync_stream():
            with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text

        # Convert sync generator to async
        sync_gen = sync_stream()

        try:
            while True:
                try:
                    # Get next chunk asynchronously
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: next(sync_gen)
                    )

                    # Update statistics
                    stats.total_chunks += 1
                    stats.total_chars += len(chunk)
                    stats.total_tokens += self._estimate_tokens(chunk)

                    # Callbacks
                    if chunk_callback:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            chunk_callback,
                            chunk
                        )

                    if stats_callback and stats.total_chunks % 10 == 0:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            stats_callback,
                            stats
                        )

                    yield chunk

                    # Small async delay to prevent overwhelming
                    await asyncio.sleep(0.01)

                except StopIteration:
                    break

        finally:
            # Cleanup
            try:
                sync_gen.close()
            except:
                pass

    async def generate_outline(self,
                              prompt: str,
                              use_cache: bool = True,
                              max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a document outline with caching and retry logic

        Args:
            prompt: Outline generation prompt
            use_cache: Whether to use cached outlines
            max_retries: Maximum retry attempts

        Returns:
            Structured outline dictionary
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt)
            if cache_key in self._outline_cache:
                logger.info("✓ Using cached outline")
                return self._outline_cache[cache_key]

        if not self._ensure_client():
            return self._mock_outline()

        # Check rate limits
        estimated_tokens = self._estimate_tokens(prompt) + 2048
        if not self._check_rate_limits(estimated_tokens):
            logger.warning("Rate limit reached for outline, waiting...")
            await asyncio.sleep(2)

        # Try with model fallback
        models_to_try = [self.MODELS["primary"], self.MODELS["fallback"]]

        for model in models_to_try:
            try:
                logger.info(f"Generating outline with {model}")

                # Record request
                self._record_request(estimated_tokens)

                # Create outline with retry
                async def _create_outline():
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self._client.messages.create(
                            model=model,
                            max_tokens=2048,
                            messages=[{
                                "role": "user",
                                "content": prompt + "\n\nProvide the outline in a clear, structured format with sections and key points."
                            }]
                        )
                    )
                    return response

                response = await self._retry_with_backoff(
                    _create_outline,
                    max_retries=max_retries
                )

                outline_text = response.content[0].text
                logger.info(f"✓ Outline generated successfully ({len(outline_text)} chars)")

                # Parse the outline
                outline = self._parse_outline(outline_text)

                # Cache it
                if use_cache:
                    cache_key = self._get_cache_key(prompt)
                    self._outline_cache[cache_key] = outline
                    logger.info("✓ Outline cached for future use")

                return outline

            except Exception as e:
                logger.error(f"Error generating outline with {model}: {e}")
                continue

        # All models failed - return mock
        logger.warning("All outline generation attempts failed, using mock")
        return self._mock_outline()

    def _parse_outline(self, outline_text: str) -> Dict[str, Any]:
        """Parse outline text into structured format"""
        lines = outline_text.strip().split('\n')

        title = ""
        sections = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Title (usually first line or starts with #)
            if line.startswith('# ') or (not title and not sections):
                title = line.lstrip('#').strip()
                continue

            # Section heading (## or numbered)
            if line.startswith('## ') or (line[0].isdigit() and '.' in line[:3]):
                if current_section:
                    sections.append(current_section)

                section_name = line.lstrip('#').lstrip('0123456789. ').strip()
                current_section = {
                    "name": section_name,
                    "points": []
                }
                continue

            # Section point (- or * or numbered sub-item)
            if current_section and (line.startswith('- ') or line.startswith('* ') or
                                   (line[0:2].strip() and line[0:2].strip()[0].isdigit())):
                point = line.lstrip('-*0123456789. ').strip()
                current_section["points"].append(point)

        # Add final section
        if current_section:
            sections.append(current_section)

        return {
            "title": title or "Untitled Document",
            "sections": sections
        }

    def _mock_outline(self) -> Dict[str, Any]:
        """Generate a mock outline for testing"""
        return {
            "title": "Sample Document",
            "sections": [
                {
                    "name": "Introduction",
                    "points": ["Context and background", "Thesis statement"]
                },
                {
                    "name": "Main Content",
                    "points": ["Key point 1", "Key point 2", "Key point 3"]
                },
                {
                    "name": "Conclusion",
                    "points": ["Summary", "Final thoughts"]
                }
            ]
        }

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics

        Returns:
            Dictionary of session stats
        """
        if not self._session_stats:
            return {"sessions": 0, "total_tokens": 0}

        return {
            "sessions": len(self._session_stats),
            "total_tokens": sum(s.total_tokens for s in self._session_stats),
            "total_duration": sum(s.get_duration() for s in self._session_stats),
            "average_tokens_per_second": sum(s.get_tokens_per_second() for s in self._session_stats) / len(self._session_stats),
            "total_errors": sum(s.errors_encountered for s in self._session_stats),
            "total_retries": sum(s.retries_attempted for s in self._session_stats),
            "models_used": list(set(s.model_used for s in self._session_stats))
        }

    def clear_cache(self):
        """Clear the outline cache"""
        cleared = len(self._outline_cache)
        self._outline_cache.clear()
        logger.info(f"Cleared {cleared} cached outlines")

    def get_cache_size(self) -> int:
        """Get number of cached outlines"""
        return len(self._outline_cache)

    async def _mock_stream(self, prompt: str) -> AsyncIterator[str]:
        """Enhanced mock content streaming for testing"""
        # Detect document type from prompt
        prompt_lower = prompt.lower()

        if "mla" in prompt_lower:
            mock_content = """Student Name
Instructor Name
Course Name
1 October 2025

Dogs: Loyal Companions

## Introduction

Dogs have been humanity's faithful companions for thousands of years. From ancient hunting partners to modern family pets, dogs have played crucial roles in human society (Smith 45).

## The History of Dog Domestication

Archaeological evidence suggests that dogs were first domesticated from wolves approximately 15,000 years ago (Johnson and Brown 112). This partnership between humans and canines represents one of the most successful interspecies relationships in history.

## Characteristics and Breeds

Modern dogs exhibit remarkable diversity, with over 300 recognized breeds worldwide. Each breed possesses unique characteristics suited to specific roles, from herding sheep to providing emotional support.

## The Human-Canine Bond

Research demonstrates that dogs provide numerous physical and psychological benefits to their owners. Studies show that dog ownership can reduce stress, lower blood pressure, and increase physical activity levels (Williams 78).

## Conclusion

Dogs remain invaluable companions in contemporary society. Their loyalty, intelligence, and adaptability ensure their continued importance in human lives for generations to come.

Works Cited

Johnson, Mary, and Robert Brown. "Canine Domestication History." Journal of Archaeological Science, vol. 45, 2020, pp. 110-125.

Smith, John. The Evolution of Dogs. Academic Press, 2019.

Williams, Sarah. "Health Benefits of Pet Ownership." Medical Journal, vol. 12, no. 3, 2021, pp. 75-82."""

        elif "apa" in prompt_lower:
            mock_content = """Running head: DOGS AS COMPANIONS

Dogs as Loyal Companions

Student Name
Institution Name

## Abstract

This paper examines the historical and contemporary role of dogs in human society, their domestication, breed diversity, and the psychological benefits of dog ownership.

## Dogs as Loyal Companions

Dogs have been humanity's faithful companions for millennia (Smith, 2019). Their domestication represents a pivotal moment in human history.

## History of Domestication

Archaeological evidence indicates that dogs were first domesticated approximately 15,000 years ago (Johnson & Brown, 2020). This partnership has proven remarkably successful.

## Modern Dog Breeds

Contemporary dogs exhibit tremendous diversity, with over 300 recognized breeds globally. Each breed possesses characteristics suited to specific functions.

## Psychological Benefits

Research demonstrates that dogs provide significant health benefits to owners, including stress reduction and increased physical activity (Williams, 2021).

## Conclusion

Dogs continue to serve as invaluable companions in modern society, providing both practical assistance and emotional support.

References

Johnson, M., & Brown, R. (2020). Canine domestication history. Journal of Archaeological Science, 45, 110-125.

Smith, J. (2019). The evolution of dogs. Academic Press.

Williams, S. (2021). Health benefits of pet ownership. Medical Journal, 12(3), 75-82."""

        else:
            mock_content = """# Dogs: Loyal Companions

## Introduction

Dogs have been humanity's faithful companions for thousands of years, serving roles from hunting partners to beloved family pets.

## History

Archaeological evidence suggests dogs were first domesticated from wolves approximately 15,000 years ago, making them one of the first domesticated animals.

## Breeds and Characteristics

Modern dogs display remarkable diversity, with over 300 recognized breeds worldwide. Each breed has unique characteristics suited to specific roles.

## Human-Canine Bond

Research shows that dogs provide numerous benefits to their owners, including reduced stress, increased physical activity, and emotional support.

## Conclusion

Dogs remain invaluable companions in contemporary society, their loyalty and intelligence ensuring their continued importance in human lives."""

        # Stream the mock content with natural pacing
        words = mock_content.split()
        for i, word in enumerate(words):
            # Add space except for first word and after newlines
            chunk = word if i == 0 or words[i-1].endswith('\n') else " " + word
            yield chunk
            # Variable delay for more natural feel
            await asyncio.sleep(0.03 if len(word) < 5 else 0.05)


# Global instance
_claude_streamer: Optional[ClaudeContentStreamer] = None


def get_claude_streamer(api_key: Optional[str] = None) -> ClaudeContentStreamer:
    """Get or create global Claude streamer instance"""
    global _claude_streamer
    if _claude_streamer is None:
        _claude_streamer = ClaudeContentStreamer(api_key)
    return _claude_streamer
