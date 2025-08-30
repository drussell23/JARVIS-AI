"""
Memory-Efficient Claude Vision Analyzer - Optimized for 16GB RAM systems
Features: Intelligent compression, caching, batch processing, and resource management
"""

import base64
import io
import hashlib
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import numpy as np
from anthropic import Anthropic
import json
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

@dataclass
class CachedResult:
    """Cached vision analysis result with metadata"""
    result: Dict[str, Any]
    timestamp: datetime
    image_hash: str
    prompt_hash: str
    size_bytes: int
    access_count: int = 0
    last_accessed: datetime = None

class CompressionStrategy:
    """Intelligent image compression based on analysis needs"""
    
    @staticmethod
    def compress_for_text_reading(image: Image.Image) -> Tuple[Image.Image, int]:
        """High quality for text extraction"""
        buffer = io.BytesIO()
        # Keep high quality for text
        image.save(buffer, format="PNG", optimize=True)
        return image, buffer.tell()
    
    @staticmethod
    def compress_for_ui_detection(image: Image.Image) -> Tuple[Image.Image, int]:
        """Medium quality for UI element detection"""
        # Reduce size while maintaining UI clarity
        max_dimension = 1920
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return image, buffer.tell()
    
    @staticmethod
    def compress_for_activity_monitoring(image: Image.Image) -> Tuple[Image.Image, int]:
        """Lower quality for general activity detection"""
        # More aggressive compression for monitoring
        max_dimension = 1280
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, optimize=True)
        return image, buffer.tell()

class MemoryEfficientVisionAnalyzer:
    """Memory-efficient Claude Vision Analyzer with caching and optimization"""
    
    def __init__(self, api_key: str, cache_dir: str = "./vision_cache", 
                 max_cache_size_mb: int = 500, max_memory_usage_mb: int = 2048):
        """Initialize with memory constraints"""
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.max_memory_usage = max_memory_usage_mb * 1024 * 1024
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache with size limit
        self._memory_cache: Dict[str, CachedResult] = {}
        self._cache_size = 0
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Compression strategies
        self.compression_strategies = {
            "text": CompressionStrategy.compress_for_text_reading,
            "ui": CompressionStrategy.compress_for_ui_detection,
            "activity": CompressionStrategy.compress_for_activity_monitoring
        }
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "total_bytes_processed": 0,
            "compression_savings": 0
        }
        
        # Load persistent cache
        self._load_persistent_cache()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage of the process"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure"""
        current_usage = self._get_memory_usage()
        return current_usage > self.max_memory_usage * 0.8  # 80% threshold
    
    def _compress_image(self, image: Any, analysis_type: str = "ui") -> Tuple[Image.Image, int, int]:
        """Compress image based on analysis type"""
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype == object:
                raise ValueError("Invalid numpy array dtype. Expected uint8 array.")
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Original size
        original_buffer = io.BytesIO()
        pil_image.save(original_buffer, format="PNG")
        original_size = original_buffer.tell()
        
        # Apply compression strategy
        compression_func = self.compression_strategies.get(analysis_type, 
                                                          self.compression_strategies["ui"])
        compressed_image, compressed_size = compression_func(pil_image)
        
        # Track compression savings
        self.metrics["compression_savings"] += original_size - compressed_size
        
        return compressed_image, original_size, compressed_size
    
    def _generate_cache_key(self, image_data: bytes, prompt: str) -> str:
        """Generate unique cache key for image+prompt combination"""
        image_hash = hashlib.md5(image_data).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{image_hash}_{prompt_hash}"
    
    def _evict_cache_if_needed(self, required_space: int):
        """Evict cache entries if needed to make space"""
        if self._cache_size + required_space <= self.max_cache_size:
            return
        
        # Sort by last accessed time (LRU)
        sorted_cache = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1].last_accessed or x[1].timestamp
        )
        
        while self._cache_size + required_space > self.max_cache_size and sorted_cache:
            key, entry = sorted_cache.pop(0)
            self._cache_size -= entry.size_bytes
            del self._memory_cache[key]
            
            # Also remove from persistent cache
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
    
    def _load_persistent_cache(self):
        """Load cache from disk on startup"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                        entry = pickle.load(f)
                        # Only load recent entries (last 24 hours)
                        if datetime.now() - entry.timestamp < timedelta(hours=24):
                            key = filename[:-4]  # Remove .pkl
                            self._memory_cache[key] = entry
                            self._cache_size += entry.size_bytes
                except Exception:
                    pass  # Skip corrupted cache files
    
    def _save_to_persistent_cache(self, key: str, entry: CachedResult):
        """Save cache entry to disk"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # Non-critical if cache save fails
    
    async def analyze_screenshot(self, image: Any, prompt: str, 
                               analysis_type: str = "ui", 
                               use_cache: bool = True) -> Dict[str, Any]:
        """Analyze screenshot with memory-efficient processing"""
        
        # Check memory pressure
        if self._check_memory_pressure():
            gc.collect()  # Force garbage collection
            
            # If still under pressure, clear some cache
            if self._check_memory_pressure():
                self._evict_cache_if_needed(self.max_cache_size // 4)
        
        # Compress image based on analysis type
        compressed_image, original_size, compressed_size = self._compress_image(image, analysis_type)
        
        # Convert to base64
        buffer = io.BytesIO()
        if analysis_type == "text":
            compressed_image.save(buffer, format="PNG", optimize=True)
        else:
            compressed_image.save(buffer, format="JPEG", quality=85)
        
        image_data = buffer.getvalue()
        self.metrics["total_bytes_processed"] += len(image_data)
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(image_data, prompt)
            
            if cache_key in self._memory_cache:
                # Cache hit
                self.metrics["cache_hits"] += 1
                cached_entry = self._memory_cache[cache_key]
                cached_entry.access_count += 1
                cached_entry.last_accessed = datetime.now()
                return cached_entry.result
            
            self.metrics["cache_misses"] += 1
        
        # Make API call
        image_base64 = base64.b64encode(image_data).decode()
        
        try:
            self.metrics["api_calls"] += 1
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg" if analysis_type != "text" else "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            result = self._parse_claude_response(message.content[0].text)
            
            # Cache the result
            if use_cache:
                cached_entry = CachedResult(
                    result=result,
                    timestamp=datetime.now(),
                    image_hash=hashlib.md5(image_data).hexdigest(),
                    prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                    size_bytes=len(image_data) + len(str(result)),
                    last_accessed=datetime.now()
                )
                
                self._evict_cache_if_needed(cached_entry.size_bytes)
                self._memory_cache[cache_key] = cached_entry
                self._cache_size += cached_entry.size_bytes
                
                # Save to persistent cache
                self._save_to_persistent_cache(cache_key, cached_entry)
            
            return result
            
        except Exception as e:
            # Return error with context
            return {
                "error": str(e),
                "description": "Failed to analyze image",
                "compression_info": {
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": compressed_size / original_size
                }
            }
    
    def _parse_claude_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's response into structured data"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "description": response,
            "has_updates": "update" in response.lower(),
            "applications_mentioned": self._extract_app_names(response),
            "actions_suggested": self._extract_actions(response)
        }
    
    def _extract_app_names(self, text: str) -> List[str]:
        """Extract application names from Claude's response"""
        common_apps = [
            "Chrome", "Safari", "Firefox", "Mail", "Messages", "Slack",
            "VS Code", "Xcode", "Terminal", "Finder", "System Preferences",
            "App Store", "Activity Monitor", "Spotify", "Discord"
        ]
        
        found_apps = []
        text_lower = text.lower()
        
        for app in common_apps:
            if app.lower() in text_lower:
                found_apps.append(app)
        
        return found_apps
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract suggested actions from Claude's response"""
        action_keywords = [
            "should update", "recommend updating", "needs to be updated",
            "click on", "open", "close", "restart", "install"
        ]
        
        actions = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in action_keywords:
                if keyword in sentence_lower:
                    actions.append(sentence.strip())
                    break
        
        return actions
    
    async def batch_analyze_regions(self, image: Any, regions: List[Dict[str, Any]], 
                                  analysis_type: str = "ui") -> List[Dict[str, Any]]:
        """Batch process multiple regions of an image efficiently"""
        # Convert to PIL Image once
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
        
        # Process regions in parallel
        tasks = []
        for region in regions:
            # Extract region
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('width', 100), region.get('height', 100)
            region_image = pil_image.crop((x, y, x + w, y + h))
            
            # Create task
            prompt = region.get('prompt', 'Analyze this region')
            task = self.analyze_screenshot(region_image, prompt, analysis_type)
            tasks.append(task)
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks)
        
        # Add region info to results
        for i, result in enumerate(results):
            result['region'] = regions[i]
        
        return results
    
    async def analyze_with_change_detection(self, current_image: Any, previous_image: Optional[Any], 
                                          prompt: str, threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze only if significant changes detected"""
        if previous_image is None:
            return await self.analyze_screenshot(current_image, prompt)
        
        # Convert to numpy arrays for comparison
        if isinstance(current_image, Image.Image):
            current_array = np.array(current_image)
        else:
            current_array = current_image
            
        if isinstance(previous_image, Image.Image):
            previous_array = np.array(previous_image)
        else:
            previous_array = previous_image
        
        # Calculate difference
        diff = np.mean(np.abs(current_array.astype(float) - previous_array.astype(float))) / 255.0
        
        if diff < threshold:
            return {
                "description": "No significant changes detected",
                "changed": False,
                "difference_score": diff
            }
        
        # Significant change detected, analyze
        result = await self.analyze_screenshot(current_image, prompt)
        result["changed"] = True
        result["difference_score"] = diff
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_hit_rate = 0
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_api_calls": self.metrics["api_calls"],
            "cache_size_mb": self._cache_size / (1024 * 1024),
            "max_cache_size_mb": self.max_cache_size / (1024 * 1024),
            "compression_savings_mb": self.metrics["compression_savings"] / (1024 * 1024),
            "total_processed_mb": self.metrics["total_bytes_processed"] / (1024 * 1024),
            "memory_usage_mb": self._get_memory_usage() / (1024 * 1024)
        }
    
    def cleanup_old_cache(self, days: int = 1):
        """Clean up cache entries older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Clean memory cache
        keys_to_remove = []
        for key, entry in self._memory_cache.items():
            if entry.timestamp < cutoff:
                keys_to_remove.append(key)
                self._cache_size -= entry.size_bytes
        
        for key in keys_to_remove:
            del self._memory_cache[key]
            
            # Remove from disk
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        
        # Force garbage collection
        gc.collect()
        
        return len(keys_to_remove)