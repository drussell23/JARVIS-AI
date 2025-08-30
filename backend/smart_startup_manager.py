#!/usr/bin/env python3
"""
Smart Startup Manager for JARVIS Backend
Handles intelligent resource-aware model loading to prevent crashes
"""

import asyncio
import psutil
import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import sys

logger = logging.getLogger(__name__)

class LoadPhase(Enum):
    """Loading phases for progressive startup"""
    CRITICAL = "critical"     # Bare minimum for API to respond
    ESSENTIAL = "essential"   # Core functionality 
    ENHANCED = "enhanced"     # Advanced features
    OPTIONAL = "optional"     # Nice to have features

@dataclass
class SystemResources:
    """Current system resource state"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: int
    memory_total_mb: int
    cpu_count: int
    load_average: Tuple[float, float, float]
    
    @property
    def is_healthy(self) -> bool:
        """Check if system has enough resources"""
        return (
            self.cpu_percent < 80 and 
            self.memory_percent < 85 and
            self.memory_available_mb > 500
        )
    
    @property
    def can_load_heavy_model(self) -> bool:
        """Check if we can load memory-intensive models"""
        return (
            self.cpu_percent < 60 and
            self.memory_percent < 70 and
            self.memory_available_mb > 1000
        )

class SmartStartupManager:
    """Manages intelligent startup with resource monitoring"""
    
    def __init__(self, 
                 max_memory_percent: float = 80,
                 max_cpu_percent: float = 75,
                 check_interval: float = 5.0):  # Changed from 0.5 to 5.0 seconds
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.check_interval = check_interval
        self.last_check_time = 0  # Track last check to prevent rapid polling
        
        # Resource monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        self.shutdown_requested = False
        
        # Model loading state
        self.loaded_models: Dict[str, Any] = {}
        self.failed_models: Dict[str, str] = {}
        self.loading_queue: asyncio.Queue = asyncio.Queue()
        
        # Executors
        self.cpu_count = multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=min(4, self.cpu_count))
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info("🛑 Shutdown requested, cleaning up...")
        self.shutdown_requested = True
        
    def get_system_resources(self) -> SystemResources:
        """Get current system resource usage"""
        # Rate limit resource checks to prevent CPU spinning
        current_time = time.time()
        if current_time - self.last_check_time < 1.0:  # Minimum 1 second between checks
            # Return cached values if checking too frequently
            if hasattr(self, '_cached_resources'):
                return self._cached_resources
        
        self.last_check_time = current_time
        
        memory = psutil.virtual_memory()
        # Use interval=None to avoid blocking, get instantaneous value
        cpu_percent = psutil.cpu_percent(interval=None)
        
        resources = SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=int(memory.available / 1024 / 1024),
            memory_total_mb=int(memory.total / 1024 / 1024),
            cpu_count=self.cpu_count,
            load_average=os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        )
        
        self._cached_resources = resources
        return resources
    
    async def wait_for_resources(self, 
                                required_memory_mb: int = 500,
                                required_cpu_headroom: float = 20) -> bool:
        """Wait until system has enough resources"""
        max_wait = 30  # Maximum 30 seconds
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait:
            resources = self.get_system_resources()
            
            if (resources.memory_available_mb >= required_memory_mb and
                resources.cpu_percent < (100 - required_cpu_headroom)):
                return True
                
            logger.info(
                f"⏳ Waiting for resources (need {required_memory_mb}MB RAM, "
                f"have {resources.memory_available_mb}MB, CPU at {resources.cpu_percent:.1f}%)"
            )
            await asyncio.sleep(2)
            
        return False
    
    async def load_with_resource_check(self, 
                                     load_func: callable,
                                     model_name: str,
                                     required_memory_mb: int = 200) -> Optional[Any]:
        """Load a model with resource checking"""
        # Check resources before loading
        if not await self.wait_for_resources(required_memory_mb):
            logger.warning(f"⚠️  Insufficient resources to load {model_name}")
            return None
            
        resources_before = self.get_system_resources()
        
        try:
            # Load the model
            logger.info(f"📦 Loading {model_name} (requires ~{required_memory_mb}MB)...")
            start_time = time.time()
            
            # Run in executor to prevent blocking
            loop = asyncio.get_event_loop()
            if required_memory_mb > 500:  # Heavy models in process pool
                result = await loop.run_in_executor(self.process_executor, load_func)
            else:  # Light models in thread pool
                result = await loop.run_in_executor(self.thread_executor, load_func)
                
            load_time = time.time() - start_time
            
            # Check resource usage after loading
            resources_after = self.get_system_resources()
            memory_used = resources_before.memory_available_mb - resources_after.memory_available_mb
            
            logger.info(
                f"✅ {model_name} loaded in {load_time:.1f}s "
                f"(used {memory_used}MB memory)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            self.failed_models[model_name] = str(e)
            return None
    
    async def progressive_model_loading(self):
        """Progressive model loading with resource management"""
        from utils.progressive_model_loader import model_loader
        
        # Phase 1: Critical models only
        logger.info("🚀 Phase 1: Loading critical models...")
        resources = self.get_system_resources()
        logger.info(
            f"📊 System resources: {resources.memory_available_mb}MB free, "
            f"CPU at {resources.cpu_percent:.1f}%"
        )
        
        # Load only the absolute minimum models
        critical_models = {
            "chatbot": ("chatbots.claude_chatbot", "ClaudeChatbot"),
            "vision_status": ("api.vision_status_endpoint", "get_vision_status"),
        }
        
        for name, (module_path, class_name) in critical_models.items():
            if self.shutdown_requested:
                break
                
            try:
                import importlib
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    self.loaded_models[name] = getattr(module, class_name)
                    logger.info(f"✅ Critical model {name} loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load critical model {name}: {e}")
        
        # Server can now start accepting requests
        logger.info("✅ Critical models loaded - server ready for requests!")
        
        # Phase 2: Essential models (background)
        if not self.shutdown_requested and resources.is_healthy:
            asyncio.create_task(self._load_essential_models())
        
        # Phase 3: Enhanced models (lazy)
        if not self.shutdown_requested:
            asyncio.create_task(self._setup_lazy_loading())
    
    async def _load_essential_models(self):
        """Load essential models in background"""
        await asyncio.sleep(2)  # Give server time to start
        
        logger.info("⚡ Phase 2: Loading essential models in background...")
        
        essential_models = [
            ("vision_system", 300),  # model_name, required_memory_mb
            ("voice_core", 200),
            ("ml_audio", 150),
        ]
        
        for model_name, required_memory in essential_models:
            if self.shutdown_requested:
                break
                
            resources = self.get_system_resources()
            if not resources.is_healthy:
                logger.warning(f"⚠️  Delaying {model_name} - system under load")
                await asyncio.sleep(5)
                continue
            
            # Queue model for loading
            await self.loading_queue.put((model_name, required_memory))
    
    async def _setup_lazy_loading(self):
        """Setup lazy loading for enhancement models"""
        await asyncio.sleep(10)  # Wait for essential models
        
        logger.info("🔮 Phase 3: Enhancement models ready for lazy loading")
        
        # Models will be loaded on-demand when first accessed
    
    async def resource_monitor(self):
        """Continuous resource monitoring with efficient polling"""
        consecutive_high_cpu = 0
        consecutive_high_memory = 0
        
        while not self.shutdown_requested:
            try:
                # Get resources (rate-limited internally)
                resources = self.get_system_resources()
                
                # Only log if consistently high (prevents log spam)
                if resources.memory_percent > 85:
                    consecutive_high_memory += 1
                    if consecutive_high_memory == 3:  # Log only when it reaches 3
                        logger.warning(
                            f"⚠️  Sustained high memory usage: {resources.memory_percent:.1f}% "
                            f"({resources.memory_available_mb}MB free)"
                        )
                else:
                    consecutive_high_memory = 0
                
                if resources.cpu_percent > 80:
                    consecutive_high_cpu += 1
                    if consecutive_high_cpu == 3:  # Log only when it reaches 3
                        logger.warning(f"⚠️  Sustained high CPU usage: {resources.cpu_percent:.1f}%")
                else:
                    consecutive_high_cpu = 0
                
                # Emergency measures if critically low on memory
                if resources.memory_percent > 95:
                    logger.error("🚨 CRITICAL: Memory exhausted, triggering emergency cleanup!")
                    await self._emergency_cleanup()
                    # Longer sleep after emergency cleanup
                    await asyncio.sleep(30)
                    continue
                
                # Adaptive sleep interval based on system health
                if resources.is_healthy:
                    sleep_time = self.check_interval * 2  # Double interval when healthy
                else:
                    sleep_time = self.check_interval
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)  # Longer sleep on error
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when memory is critical"""
        import gc
        from core.memory_quantizer import memory_quantizer
        
        logger.info("🚨 Starting emergency memory cleanup...")
        
        # Use memory quantizer for aggressive cleanup
        await memory_quantizer._reduce_memory_usage()
        
        # Additional emergency measures
        # Clear executor queues
        if hasattr(self.thread_executor, '_threads'):
            self.thread_executor._threads.clear()
        
        # Force multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Log memory usage after cleanup
        resources = self.get_system_resources()
        logger.info(f"🧹 After emergency cleanup: {resources.memory_available_mb}MB free ({resources.memory_percent:.1f}% used)")
    
    def get_startup_status(self) -> Dict[str, Any]:
        """Get current startup status"""
        resources = self.get_system_resources()
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "phase": self._get_current_phase(),
            "models_loaded": len(self.loaded_models),
            "models_failed": len(self.failed_models),
            "resources": {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "memory_free_mb": resources.memory_available_mb,
            },
            "health": "healthy" if resources.is_healthy else "degraded",
            "loaded_models": list(self.loaded_models.keys()),
            "failed_models": self.failed_models,
        }
    
    def _get_current_phase(self) -> str:
        """Determine current loading phase"""
        loaded_count = len(self.loaded_models)
        if loaded_count < 3:
            return LoadPhase.CRITICAL.value
        elif loaded_count < 10:
            return LoadPhase.ESSENTIAL.value
        elif loaded_count < 20:
            return LoadPhase.ENHANCED.value
        else:
            return LoadPhase.OPTIONAL.value
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down startup manager...")
        self.shutdown_requested = True
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)
        
        logger.info("✅ Startup manager shutdown complete")

# Global instance
startup_manager = SmartStartupManager()

async def smart_startup():
    """Main startup orchestrator"""
    logger.info("🎯 JARVIS Smart Startup Manager v2.0")
    logger.info("=" * 60)
    
    # Import and start memory quantizer
    from core.memory_quantizer import memory_quantizer, start_memory_optimization
    
    # Start memory optimization service
    memory_task = asyncio.create_task(start_memory_optimization())
    
    # Start resource monitor
    monitor_task = asyncio.create_task(startup_manager.resource_monitor())
    
    # Get initial memory status
    memory_status = memory_quantizer.get_memory_status()
    logger.info(f"📊 Memory quantizer started - Level: {memory_status['current_level']}, Usage: {memory_status['memory_usage_gb']:.1f}GB")
    
    # Start progressive loading
    await startup_manager.progressive_model_loading()
    
    # Keep monitoring
    try:
        await asyncio.gather(monitor_task, memory_task)
    except asyncio.CancelledError:
        pass
    
    await startup_manager.shutdown()

# Startup status endpoint
async def get_startup_status():
    """Get current startup status for monitoring"""
    return startup_manager.get_startup_status()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(smart_startup())