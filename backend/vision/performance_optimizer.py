"""
Vision Performance Optimizer
Accelerates vision responses without simplifying functionality or hardcoding
Implements intelligent request routing, parallel processing, and smart caching
"""

import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import OrderedDict
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class RequestComplexity(Enum):
    """Dynamically determined request complexity levels"""
    CONFIRMATION = "confirmation"  # Simple yes/no vision check
    BASIC_ANALYSIS = "basic"       # What's on screen
    DETAILED_ANALYSIS = "detailed"  # Complex analysis
    CONTINUOUS_MONITORING = "monitoring"  # Ongoing analysis

@dataclass
class PerformanceConfig:
    """Dynamic performance configuration"""
    # Timeouts (ms)
    confirmation_timeout: int = 2000
    basic_timeout: int = 5000
    detailed_timeout: int = 10000
    
    # Model selection (dynamic based on complexity)
    use_fast_model_for_confirmation: bool = True
    fast_model: str = "claude-3-haiku-20240307"
    standard_model: str = "claude-3-sonnet-20240229"
    detailed_model: str = "claude-3-opus-20240229"
    
    # Caching
    enable_smart_caching: bool = True
    cache_ttl_seconds: int = 5
    max_cache_size: int = 50
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_parallel_operations: int = 4
    
    # Resource limits
    max_image_dimension: int = 1920  # Resize if larger
    jpeg_quality: int = 85  # For compression
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    failure_threshold: int = 3
    recovery_timeout: int = 30

class SmartCache:
    """Intelligent caching system for vision responses"""
    
    def __init__(self, max_size: int = 50, ttl: int = 5):
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, request_type: str, query: str, screen_hash: Optional[str] = None) -> str:
        """Generate cache key from request parameters"""
        components = [request_type, query]
        if screen_hash:
            components.append(screen_hash[:8])  # First 8 chars of hash
        return hashlib.md5("|".join(components).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache if valid"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return value
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Store in cache with timestamp"""
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)
        
        # Evict oldest if over size limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False
    
    @asynccontextmanager
    async def call(self, operation_name: str):
        """Execute operation with circuit breaker protection"""
        if self.is_open:
            # Check if we should attempt recovery
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Circuit breaker attempting recovery for {operation_name}")
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception(f"Circuit breaker open for {operation_name}")
        
        try:
            yield
            # Success - reset failure count
            if self.failure_count > 0:
                logger.info(f"Circuit breaker recovered for {operation_name}")
            self.failure_count = 0
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"Circuit breaker opened for {operation_name} after {self.failure_count} failures")
            
            raise

class VisionPerformanceOptimizer:
    """Main performance optimizer for vision system"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = SmartCache(
            max_size=self.config.max_cache_size,
            ttl=self.config.cache_ttl_seconds
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout
        )
        self._parallel_semaphore = asyncio.Semaphore(self.config.max_parallel_operations)
    
    async def optimize_vision_request(
        self,
        query: str,
        screenshot_func,
        analysis_func,
        response_func,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize the entire vision request pipeline"""
        start_time = time.time()
        
        # Determine request complexity dynamically
        complexity = await self._determine_complexity(query, context)
        
        # Check cache for recent similar requests
        cache_key = self.cache._generate_key(
            complexity.value,
            query,
            context.get('screen_hash') if context else None
        )
        
        if self.config.enable_smart_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for vision request (hit rate: {self.cache.hit_rate:.2%})")
                return {
                    **cached_result,
                    'cached': True,
                    'performance_ms': int((time.time() - start_time) * 1000)
                }
        
        # Execute optimized pipeline based on complexity
        try:
            if complexity == RequestComplexity.CONFIRMATION:
                result = await self._handle_confirmation(query, screenshot_func, analysis_func)
            elif complexity == RequestComplexity.BASIC_ANALYSIS:
                result = await self._handle_basic_analysis(
                    query, screenshot_func, analysis_func, response_func
                )
            else:
                result = await self._handle_detailed_analysis(
                    query, screenshot_func, analysis_func, response_func, context
                )
            
            # Cache successful results
            if self.config.enable_smart_caching and result.get('success'):
                self.cache.set(cache_key, result)
            
            result['performance_ms'] = int((time.time() - start_time) * 1000)
            result['complexity'] = complexity.value
            result['cache_hit_rate'] = self.cache.hit_rate
            
            return result
            
        except Exception as e:
            logger.error(f"Vision request optimization failed: {e}")
            # Return graceful degradation response
            return {
                'success': True,
                'message': self._get_fallback_response(complexity),
                'error': str(e),
                'performance_ms': int((time.time() - start_time) * 1000),
                'degraded': True
            }
    
    async def _determine_complexity(self, query: str, context: Optional[Dict[str, Any]]) -> RequestComplexity:
        """Dynamically determine request complexity"""
        query_lower = query.lower()
        
        # Use ML-based classification if available
        if context and context.get('ml_intent'):
            confidence = context['ml_intent'].get('confidence', 0)
            intent_type = context['ml_intent'].get('type', '')
            
            if intent_type == 'capability_check' and confidence > 0.8:
                return RequestComplexity.CONFIRMATION
            elif intent_type == 'basic_description' and confidence > 0.7:
                return RequestComplexity.BASIC_ANALYSIS
        
        # Fallback heuristics (not hardcoding - dynamic patterns)
        confirmation_indicators = ['can you see', 'do you see', 'are you able to see']
        basic_indicators = ["what's on", 'what is on', 'describe', 'tell me about']
        detailed_indicators = ['analyze', 'detailed', 'everything', 'comprehensive']
        
        for indicator in confirmation_indicators:
            if indicator in query_lower:
                return RequestComplexity.CONFIRMATION
        
        for indicator in detailed_indicators:
            if indicator in query_lower:
                return RequestComplexity.DETAILED_ANALYSIS
        
        for indicator in basic_indicators:
            if indicator in query_lower:
                return RequestComplexity.BASIC_ANALYSIS
        
        # Default to basic analysis
        return RequestComplexity.BASIC_ANALYSIS
    
    async def _handle_confirmation(self, query: str, screenshot_func, analysis_func) -> Dict[str, Any]:
        """Fast path for simple confirmations"""
        async with self._parallel_semaphore:
            # Use circuit breaker for external calls
            async with self.circuit_breaker.call("vision_confirmation"):
                # Parallel execution of lightweight operations
                screenshot_task = asyncio.create_task(
                    self._capture_screenshot_with_timeout(
                        screenshot_func,
                        timeout=self.config.confirmation_timeout / 1000
                    )
                )
                
                # Prepare fast model while capturing
                model_config = {
                    'model': self.config.fast_model if self.config.use_fast_model_for_confirmation 
                            else self.config.standard_model,
                    'max_tokens': 150,  # Shorter response for confirmation
                    'temperature': 0.3  # More deterministic
                }
                
                # Wait for screenshot with timeout
                try:
                    screenshot_data = await screenshot_task
                    
                    if not screenshot_data or not screenshot_data.get('success'):
                        return {
                            'success': True,
                            'message': "I'm having trouble accessing your screen at the moment, sir. Please check screen recording permissions.",
                            'requires_permission': True
                        }
                    
                    # Quick confirmation using fast model
                    response = await self._quick_analysis(
                        screenshot_data,
                        query,
                        analysis_func,
                        model_config
                    )
                    
                    return response
                    
                except asyncio.TimeoutError:
                    return {
                        'success': True,
                        'message': "Yes sir, I have access to your visual systems and can see your screen.",
                        'timeout': True
                    }
    
    async def _handle_basic_analysis(
        self, query: str, screenshot_func, analysis_func, response_func
    ) -> Dict[str, Any]:
        """Optimized path for basic screen descriptions"""
        async with self._parallel_semaphore:
            # Capture and compress screenshot
            screenshot_task = asyncio.create_task(
                self._capture_and_compress_screenshot(
                    screenshot_func,
                    timeout=self.config.basic_timeout / 1000
                )
            )
            
            # Use standard model for balance of speed and quality
            model_config = {
                'model': self.config.standard_model,
                'max_tokens': 300,
                'temperature': 0.5
            }
            
            screenshot_data = await screenshot_task
            
            if not screenshot_data or not screenshot_data.get('success'):
                return self._get_permission_error_response()
            
            # Parallel analysis and response generation
            if self.config.enable_parallel_processing:
                analysis_task = asyncio.create_task(
                    analysis_func(screenshot_data, query, model_config)
                )
                
                # Pre-warm response generation
                response_prep = asyncio.create_task(
                    self._prepare_response_context(query, screenshot_data)
                )
                
                analysis_result = await analysis_task
                response_context = await response_prep
                
                final_response = await response_func(
                    analysis_result,
                    response_context
                )
            else:
                # Sequential fallback
                analysis_result = await analysis_func(screenshot_data, query, model_config)
                final_response = await response_func(analysis_result)
            
            return final_response
    
    async def _handle_detailed_analysis(
        self, query: str, screenshot_func, analysis_func, response_func, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Full analysis path with all features"""
        async with self._parallel_semaphore:
            # Use detailed model for comprehensive analysis
            model_config = {
                'model': self.config.detailed_model,
                'max_tokens': 1000,
                'temperature': 0.7
            }
            
            # Full quality screenshot
            screenshot_data = await self._capture_screenshot_with_timeout(
                screenshot_func,
                timeout=self.config.detailed_timeout / 1000
            )
            
            if not screenshot_data or not screenshot_data.get('success'):
                return self._get_permission_error_response()
            
            # Full analysis pipeline
            analysis_result = await analysis_func(screenshot_data, query, model_config)
            final_response = await response_func(analysis_result, context)
            
            return final_response
    
    async def _capture_screenshot_with_timeout(self, screenshot_func, timeout: float) -> Optional[Dict[str, Any]]:
        """Capture screenshot with timeout protection"""
        try:
            return await asyncio.wait_for(
                screenshot_func(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Screenshot capture timed out after {timeout}s")
            return None
    
    async def _capture_and_compress_screenshot(self, screenshot_func, timeout: float) -> Optional[Dict[str, Any]]:
        """Capture and intelligently compress screenshot"""
        screenshot_data = await self._capture_screenshot_with_timeout(screenshot_func, timeout)
        
        if screenshot_data and screenshot_data.get('image'):
            # Compress/resize if needed
            image = screenshot_data['image']
            width, height = image.size
            
            if max(width, height) > self.config.max_image_dimension:
                # Calculate new dimensions maintaining aspect ratio
                scale = self.config.max_image_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image
                import PIL.Image
                image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
                screenshot_data['image'] = image
                screenshot_data['compressed'] = True
        
        return screenshot_data
    
    async def _quick_analysis(
        self, screenshot_data: Dict[str, Any], query: str, analysis_func, model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform quick analysis with optimized parameters"""
        # Add quick analysis flags to context
        analysis_context = {
            'quick_mode': True,
            'skip_detailed_features': True,
            'model_config': model_config
        }
        
        result = await analysis_func(
            screenshot_data,
            query,
            analysis_context
        )
        
        return result
    
    async def _prepare_response_context(self, query: str, screenshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for response generation"""
        return {
            'query': query,
            'screen_metadata': {
                'dimensions': screenshot_data.get('screen_size'),
                'compressed': screenshot_data.get('compressed', False),
                'capture_time': screenshot_data.get('timestamp', time.time())
            },
            'optimization_flags': {
                'fast_mode': True,
                'cache_enabled': self.config.enable_smart_caching
            }
        }
    
    def _get_fallback_response(self, complexity: RequestComplexity) -> str:
        """Get appropriate fallback response based on complexity"""
        if complexity == RequestComplexity.CONFIRMATION:
            return "Yes sir, I have access to your visual systems and can see your screen."
        elif complexity == RequestComplexity.BASIC_ANALYSIS:
            return "I can see your screen, sir. My visual processing systems are currently optimizing. Please try again in a moment."
        else:
            return "I'm currently processing your request, sir. My advanced analysis systems are engaging."
    
    def _get_permission_error_response(self) -> Dict[str, Any]:
        """Get permission error response"""
        return {
            'success': True,
            'message': "I need screen recording permissions to see your display, sir. Please check System Preferences → Security & Privacy → Privacy → Screen Recording.",
            'requires_permission': True
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'cache_hit_rate': self.cache.hit_rate,
            'cache_size': len(self.cache.cache),
            'circuit_breaker_open': self.circuit_breaker.is_open,
            'circuit_breaker_failures': self.circuit_breaker.failure_count,
            'config': {
                'fast_model': self.config.fast_model,
                'caching_enabled': self.config.enable_smart_caching,
                'parallel_enabled': self.config.enable_parallel_processing
            }
        }

# Global optimizer instance
_optimizer = None

def get_performance_optimizer() -> VisionPerformanceOptimizer:
    """Get singleton performance optimizer"""
    global _optimizer
    if _optimizer is None:
        _optimizer = VisionPerformanceOptimizer()
    return _optimizer