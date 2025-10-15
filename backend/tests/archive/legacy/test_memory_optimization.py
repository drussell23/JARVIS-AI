#!/usr/bin/env python3
"""
Memory Optimization Test Suite
==============================

Tests the ML memory optimization to ensure we meet the 35% target on 16GB systems.
"""

import asyncio
import psutil
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Import optimized components
from ml_memory_manager import get_ml_memory_manager
from context_aware_loader import (
    get_context_loader, 
    SystemContext, 
    ProximityLevel,
    initialize_context_aware_loading
)
from voice_unlock.ml.ml_manager import get_ml_manager
from voice_unlock.ml.quantized_models import quantize_voice_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizationTester:
    """Test suite for memory optimization"""
    
    def __init__(self):
        self.ml_memory_manager = get_ml_memory_manager()
        self.context_loader = get_context_loader()
        self.voice_ml_manager = get_ml_manager()
        self.results = {
            "start_time": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "tests": []
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        memory = psutil.virtual_memory()
        return {
            "total_ram_gb": memory.total / 1024**3,
            "available_ram_gb": memory.available / 1024**3,
            "cpu_count": psutil.cpu_count(),
            "platform": psutil.MACOS if hasattr(psutil, 'MACOS') else "unknown"
        }
        
    def _measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system_percent": memory.percent,
            "system_used_gb": memory.used / 1024**3,
            "system_available_gb": memory.available / 1024**3,
            "process_mb": process.memory_info().rss / 1024**2,
            "ml_models_mb": self.ml_memory_manager.get_memory_usage()["ml_models_mb"]
        }
        
    async def test_baseline_memory(self):
        """Test 1: Baseline memory usage with no models"""
        logger.info("\n🧪 Test 1: Baseline Memory Usage")
        
        # Ensure no models loaded
        await self.context_loader.update_context(SystemContext.IDLE)
        await asyncio.sleep(2)
        
        baseline = self._measure_memory()
        logger.info(f"Baseline memory: {baseline['system_percent']:.1f}% "
                   f"(Process: {baseline['process_mb']:.1f}MB)")
        
        self.results["tests"].append({
            "name": "baseline",
            "memory": baseline,
            "passed": baseline['system_percent'] < 35
        })
        
        return baseline['system_percent'] < 35
        
    async def test_voice_command_memory(self):
        """Test 2: Memory usage during voice commands"""
        logger.info("\n🧪 Test 2: Voice Command Memory Usage")
        
        # Simulate voice command context
        await self.context_loader.update_context(
            SystemContext.VOICE_COMMAND,
            proximity=ProximityLevel.NEAR
        )
        await asyncio.sleep(3)
        
        voice_memory = self._measure_memory()
        logger.info(f"Voice command memory: {voice_memory['system_percent']:.1f}% "
                   f"(ML models: {voice_memory['ml_models_mb']:.1f}MB)")
        
        self.results["tests"].append({
            "name": "voice_command",
            "memory": voice_memory,
            "passed": voice_memory['system_percent'] < 35
        })
        
        return voice_memory['system_percent'] < 35
        
    async def test_authentication_memory(self):
        """Test 3: Memory usage during authentication"""
        logger.info("\n🧪 Test 3: Authentication Memory Usage")
        
        # First quantize voice models
        models_dir = Path.home() / '.jarvis' / 'models'
        quantized_dir = models_dir / 'quantized'
        
        if models_dir.exists() and any(models_dir.glob("*.pkl")):
            logger.info("Quantizing voice models...")
            quantize_voice_models(models_dir, quantized_dir, target="int8")
        
        # Simulate authentication context
        await self.context_loader.update_context(
            SystemContext.AUTHENTICATION,
            proximity=ProximityLevel.NEAR
        )
        await asyncio.sleep(3)
        
        auth_memory = self._measure_memory()
        logger.info(f"Authentication memory: {auth_memory['system_percent']:.1f}% "
                   f"(ML models: {auth_memory['ml_models_mb']:.1f}MB)")
        
        self.results["tests"].append({
            "name": "authentication",
            "memory": auth_memory,
            "passed": auth_memory['system_percent'] < 35
        })
        
        return auth_memory['system_percent'] < 35
        
    async def test_multi_context_memory(self):
        """Test 4: Memory usage with multiple contexts"""
        logger.info("\n🧪 Test 4: Multi-Context Memory Usage")
        
        # Simulate multiple active contexts
        await self.context_loader.update_context(
            SystemContext.CONVERSATION,
            secondary={SystemContext.SCREEN_ANALYSIS},
            proximity=ProximityLevel.MEDIUM
        )
        await asyncio.sleep(3)
        
        multi_memory = self._measure_memory()
        logger.info(f"Multi-context memory: {multi_memory['system_percent']:.1f}% "
                   f"(ML models: {multi_memory['ml_models_mb']:.1f}MB)")
        
        self.results["tests"].append({
            "name": "multi_context",
            "memory": multi_memory,
            "passed": multi_memory['system_percent'] < 35
        })
        
        return multi_memory['system_percent'] < 35
        
    async def test_memory_pressure_handling(self):
        """Test 5: Memory pressure handling"""
        logger.info("\n🧪 Test 5: Memory Pressure Handling")
        
        # Simulate high memory pressure
        initial_memory = self._measure_memory()
        
        # Force load multiple models
        try:
            await self.ml_memory_manager.load_model("whisper_base")
            await self.ml_memory_manager.load_model("vision_encoder") 
            await self.ml_memory_manager.load_model("embeddings")
            await self.ml_memory_manager.load_model("sentiment")
        except:
            pass  # Some may fail due to memory limits
            
        loaded_memory = self._measure_memory()
        
        # Trigger memory critical context
        await self.context_loader.update_context(SystemContext.MEMORY_CRITICAL)
        await asyncio.sleep(5)
        
        recovered_memory = self._measure_memory()
        
        logger.info(f"Memory pressure test:")
        logger.info(f"  Initial: {initial_memory['system_percent']:.1f}%")
        logger.info(f"  Loaded: {loaded_memory['system_percent']:.1f}%")
        logger.info(f"  Recovered: {recovered_memory['system_percent']:.1f}%")
        
        self.results["tests"].append({
            "name": "memory_pressure",
            "initial": initial_memory,
            "loaded": loaded_memory,
            "recovered": recovered_memory,
            "passed": recovered_memory['system_percent'] < 35
        })
        
        return recovered_memory['system_percent'] < 35
        
    async def test_proximity_based_loading(self):
        """Test 6: Proximity-based model loading"""
        logger.info("\n🧪 Test 6: Proximity-Based Loading")
        
        proximity_results = {}
        
        for proximity in [ProximityLevel.FAR, ProximityLevel.MEDIUM, ProximityLevel.NEAR]:
            await self.context_loader.handle_proximity_change(proximity)
            await asyncio.sleep(2)
            
            memory = self._measure_memory()
            proximity_results[proximity.value] = memory
            
            logger.info(f"Proximity {proximity.value}: {memory['system_percent']:.1f}% "
                       f"(Models: {memory['ml_models_mb']:.1f}MB)")
        
        self.results["tests"].append({
            "name": "proximity_loading",
            "results": proximity_results,
            "passed": all(m['system_percent'] < 35 for m in proximity_results.values())
        })
        
        return all(m['system_percent'] < 35 for m in proximity_results.values())
        
    async def test_context_transitions(self):
        """Test 7: Context transition memory management"""
        logger.info("\n🧪 Test 7: Context Transitions")
        
        transitions = [
            (SystemContext.IDLE, ProximityLevel.FAR),
            (SystemContext.VOICE_COMMAND, ProximityLevel.NEAR),
            (SystemContext.AUTHENTICATION, ProximityLevel.NEAR),
            (SystemContext.CONVERSATION, ProximityLevel.MEDIUM),
            (SystemContext.SCREEN_ANALYSIS, ProximityLevel.MEDIUM),
            (SystemContext.IDLE, ProximityLevel.AWAY)
        ]
        
        transition_results = []
        
        for context, proximity in transitions:
            await self.context_loader.update_context(context, proximity=proximity)
            await asyncio.sleep(2)
            
            memory = self._measure_memory()
            transition_results.append({
                "context": context.value,
                "proximity": proximity.value,
                "memory_percent": memory['system_percent'],
                "ml_models_mb": memory['ml_models_mb']
            })
            
            logger.info(f"{context.value} @ {proximity.value}: "
                       f"{memory['system_percent']:.1f}%")
        
        max_memory = max(r['memory_percent'] for r in transition_results)
        
        self.results["tests"].append({
            "name": "context_transitions",
            "transitions": transition_results,
            "max_memory": max_memory,
            "passed": max_memory < 35
        })
        
        return max_memory < 35
        
    async def test_sustained_usage(self):
        """Test 8: Sustained usage memory stability"""
        logger.info("\n🧪 Test 8: Sustained Usage (60 seconds)")
        
        measurements = []
        contexts = [SystemContext.VOICE_COMMAND, SystemContext.CONVERSATION, SystemContext.IDLE]
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 60:
            # Cycle through contexts
            context = contexts[iteration % len(contexts)]
            await self.context_loader.update_context(context)
            
            # Measure every 5 seconds
            if iteration % 5 == 0:
                memory = self._measure_memory()
                measurements.append({
                    "time": time.time() - start_time,
                    "memory_percent": memory['system_percent'],
                    "context": context.value
                })
                
                logger.info(f"t={measurements[-1]['time']:.0f}s: "
                           f"{memory['system_percent']:.1f}% ({context.value})")
            
            await asyncio.sleep(1)
            iteration += 1
        
        max_sustained = max(m['memory_percent'] for m in measurements)
        avg_sustained = sum(m['memory_percent'] for m in measurements) / len(measurements)
        
        self.results["tests"].append({
            "name": "sustained_usage",
            "duration_seconds": 60,
            "measurements": measurements,
            "max_memory": max_sustained,
            "avg_memory": avg_sustained,
            "passed": max_sustained < 35
        })
        
        return max_sustained < 35
        
    async def run_all_tests(self):
        """Run all memory optimization tests"""
        logger.info("=" * 60)
        logger.info("🚀 JARVIS Memory Optimization Test Suite")
        logger.info(f"Target: <35% memory usage on 16GB system")
        logger.info(f"System RAM: {self.results['system_info']['total_ram_gb']:.1f}GB")
        logger.info("=" * 60)
        
        # Initialize context-aware loading
        await initialize_context_aware_loading()
        
        # Run tests
        test_results = []
        test_results.append(await self.test_baseline_memory())
        test_results.append(await self.test_voice_command_memory())
        test_results.append(await self.test_authentication_memory())
        test_results.append(await self.test_multi_context_memory())
        test_results.append(await self.test_memory_pressure_handling())
        test_results.append(await self.test_proximity_based_loading())
        test_results.append(await self.test_context_transitions())
        test_results.append(await self.test_sustained_usage())
        
        # Summary
        passed = sum(test_results)
        total = len(test_results)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ Tests Passed: {passed}/{total}")
        
        if passed == total:
            logger.info("🎉 All tests passed! Memory target achieved!")
        else:
            logger.info("❌ Some tests failed. Further optimization needed.")
            
        # Save results
        self.results["summary"] = {
            "passed": passed,
            "total": total,
            "success": passed == total
        }
        
        results_path = Path("memory_optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"\nDetailed results saved to: {results_path}")
        
        return passed == total
        

async def main():
    """Run the memory optimization tests"""
    tester = MemoryOptimizationTester()
    success = await tester.run_all_tests()
    
    # Cleanup
    ml_manager = get_ml_memory_manager()
    ml_manager.shutdown()
    
    await get_context_loader().stop()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)