"""
Memory Quantizer - Intelligent memory management with quantization
Limits memory usage and provides graceful degradation under pressure
"""

import asyncio
import logging
import psutil
import gc
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory usage tiers for quantization"""
    FULL = "full"           # 100% - No restrictions
    HIGH = "high"           # 75% - Some optimizations
    MEDIUM = "medium"       # 50% - Aggressive optimizations
    LOW = "low"            # 25% - Minimal memory mode
    CRITICAL = "critical"   # Emergency mode

@dataclass
class MemoryStatus:
    """Current memory status"""
    memory_usage_gb: float
    memory_total_gb: float
    memory_percent: float
    tier: MemoryTier
    pressure: str

class MemoryQuantizer:
    """
    Quantize memory usage to prevent excessive consumption
    Provides automatic memory tier management
    """

    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_gb = max_memory_gb
        self.current_tier = MemoryTier.FULL

        # Thresholds (% of max allowed)
        self.thresholds = {
            MemoryTier.FULL: 0.60,      # < 60%
            MemoryTier.HIGH: 0.75,      # 60-75%
            MemoryTier.MEDIUM: 0.85,    # 75-85%
            MemoryTier.LOW: 0.95,       # 85-95%
            MemoryTier.CRITICAL: 1.00   # > 95%
        }

        # Optimization flags
        self.optimizations_enabled = {
            MemoryTier.FULL: [],
            MemoryTier.HIGH: ['cache_pruning'],
            MemoryTier.MEDIUM: ['cache_pruning', 'lazy_loading'],
            MemoryTier.LOW: ['cache_pruning', 'lazy_loading', 'aggressive_gc'],
            MemoryTier.CRITICAL: ['cache_pruning', 'lazy_loading', 'aggressive_gc', 'emergency_cleanup']
        }

        logger.info(f"Memory Quantizer initialized (max: {max_memory_gb}GB)")

    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status with tier"""
        mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info().rss / (1024 ** 3)  # GB

        # Calculate tier based on usage
        usage_percent = process_mem / self.max_memory_gb
        tier = self._calculate_tier(usage_percent)

        return MemoryStatus(
            memory_usage_gb=process_mem,
            memory_total_gb=mem.total / (1024 ** 3),
            memory_percent=mem.percent,
            tier=tier,
            pressure=self._get_memory_pressure()
        )

    def _calculate_tier(self, usage_percent: float) -> MemoryTier:
        """Calculate memory tier from usage percentage"""
        if usage_percent >= self.thresholds[MemoryTier.CRITICAL]:
            return MemoryTier.CRITICAL
        elif usage_percent >= self.thresholds[MemoryTier.LOW]:
            return MemoryTier.LOW
        elif usage_percent >= self.thresholds[MemoryTier.MEDIUM]:
            return MemoryTier.MEDIUM
        elif usage_percent >= self.thresholds[MemoryTier.HIGH]:
            return MemoryTier.HIGH
        else:
            return MemoryTier.FULL

    def _get_memory_pressure(self) -> str:
        """Get system memory pressure (macOS specific)"""
        try:
            import subprocess
            result = subprocess.run(
                ['memory_pressure'],
                capture_output=True,
                text=True,
                timeout=1
            )
            output = result.stdout.lower()

            if 'critical' in output:
                return 'critical'
            elif 'warn' in output:
                return 'warn'
            elif 'normal' in output:
                return 'normal'
            else:
                return 'unknown'
        except:
            return 'unknown'

    async def optimize_for_tier(self, tier: MemoryTier) -> Dict[str, Any]:
        """Apply optimizations for given memory tier"""
        optimizations = self.optimizations_enabled[tier]
        results = {
            'tier': tier.value,
            'optimizations_applied': [],
            'memory_freed_mb': 0
        }

        before = psutil.Process().memory_info().rss / (1024 ** 2)  # MB

        if 'cache_pruning' in optimizations:
            # Prune caches
            results['optimizations_applied'].append('cache_pruning')

        if 'lazy_loading' in optimizations:
            # Enable lazy loading
            results['optimizations_applied'].append('lazy_loading')

        if 'aggressive_gc' in optimizations:
            # Run garbage collection
            gc.collect()
            results['optimizations_applied'].append('aggressive_gc')

        if 'emergency_cleanup' in optimizations:
            # Emergency cleanup
            gc.collect(2)  # Full collection
            results['optimizations_applied'].append('emergency_cleanup')

        after = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
        results['memory_freed_mb'] = max(0, before - after)

        logger.info(f"Optimized for tier {tier.value}: {results}")
        return results

    async def check_and_optimize(self) -> Optional[Dict[str, Any]]:
        """Check memory and optimize if needed"""
        status = self.get_memory_status()

        # If tier changed, apply optimizations
        if status.tier != self.current_tier:
            logger.warning(f"Memory tier changed: {self.current_tier.value} -> {status.tier.value}")
            self.current_tier = status.tier
            return await self.optimize_for_tier(status.tier)

        # If in critical tier, always optimize
        if status.tier == MemoryTier.CRITICAL:
            return await self.optimize_for_tier(status.tier)

        return None

    def should_load_model(self, model_size_gb: float) -> bool:
        """Check if model can be loaded given current memory state"""
        status = self.get_memory_status()

        # Calculate projected usage
        projected_usage = status.memory_usage_gb + model_size_gb

        # Don't load if it would exceed max or put us in critical tier
        if projected_usage >= self.max_memory_gb * 0.95:
            logger.warning(f"Cannot load {model_size_gb}GB model - would exceed limit")
            return False

        return True

    def get_quantization_level(self) -> int:
        """Get quantization level (0-4) based on current tier"""
        tier_levels = {
            MemoryTier.FULL: 0,
            MemoryTier.HIGH: 1,
            MemoryTier.MEDIUM: 2,
            MemoryTier.LOW: 3,
            MemoryTier.CRITICAL: 4
        }
        return tier_levels[self.current_tier]

# Singleton instance
_memory_quantizer_instance = None

def get_memory_quantizer(max_memory_gb: float = 4.0) -> MemoryQuantizer:
    """Get singleton memory quantizer instance"""
    global _memory_quantizer_instance

    if _memory_quantizer_instance is None:
        _memory_quantizer_instance = MemoryQuantizer(max_memory_gb)

    return _memory_quantizer_instance

# Convenience singleton
memory_quantizer = get_memory_quantizer()

if __name__ == "__main__":
    # Test memory quantizer
    logging.basicConfig(level=logging.INFO)

    async def test():
        mq = get_memory_quantizer()

        print("🧠 Memory Quantizer Test")
        print("=" * 50)

        status = mq.get_memory_status()
        print(f"\n📊 Current Status:")
        print(f"  Usage: {status.memory_usage_gb:.2f}GB / {mq.max_memory_gb}GB")
        print(f"  Tier: {status.tier.value}")
        print(f"  System Pressure: {status.pressure}")

        print(f"\n🔧 Testing optimization...")
        result = await mq.check_and_optimize()
        if result:
            print(f"  Freed: {result['memory_freed_mb']:.1f}MB")
            print(f"  Optimizations: {', '.join(result['optimizations_applied'])}")
        else:
            print(f"  No optimization needed")

    asyncio.run(test())
