"""
Phase 2: Integrated Architecture Optimization System
Combines parallel processing, advanced quantization, and smart caching
Target: 35% CPU usage with 5x performance improvement
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import Phase 2 components
try:
    from parallel_processing_pipeline import ParallelProcessingPipeline, create_parallel_pipeline
    from advanced_quantization import (
        QuantizationConfig, QuantizationType, create_quantized_model,
        ModelPruner, DynamicQuantizer
    )
    from smart_caching_system import SmartCache, CachedModel
except ImportError:
    from .parallel_processing_pipeline import ParallelProcessingPipeline, create_parallel_pipeline
    from .advanced_quantization import (
        QuantizationConfig, QuantizationType, create_quantized_model,
        ModelPruner, DynamicQuantizer
    )
    from .smart_caching_system import SmartCache, CachedModel

logger = logging.getLogger(__name__)

@dataclass
class Phase2Config:
    """Configuration for Phase 2 optimizations"""
    # Parallel processing
    num_vision_threads: int = 2
    num_inference_threads: int = 2
    num_update_threads: int = 1
    
    # Quantization
    quantization_type: QuantizationType = QuantizationType.INT8
    pruning_sparsity: float = 0.5
    dynamic_quantization: bool = True
    
    # Caching
    cache_size_mb: float = 500
    cache_ttl_seconds: float = 60
    enable_compression: bool = True
    
    # Performance targets
    target_cpu: float = 35.0
    target_latency_ms: float = 100.0

class Phase2OptimizedSystem:
    """
    Integrated system combining all Phase 2 optimizations:
    - Parallel processing pipeline
    - Advanced quantization (INT4/INT8/FP16)
    - Smart caching with temporal coherence
    - 50% model pruning
    """
    
    def __init__(self, base_model: nn.Module, config: Phase2Config = None):
        self.config = config or Phase2Config()
        self.base_model = base_model
        
        logger.info("🚀 Initializing Phase 2 Optimized System")
        
        # 1. Create parallel processing pipeline
        self.pipeline = create_parallel_pipeline(
            num_vision_threads=self.config.num_vision_threads,
            num_inference_threads=self.config.num_inference_threads,
            num_update_threads=self.config.num_update_threads,
            target_cpu=self.config.target_cpu
        )
        
        # 2. Create smart cache
        self.cache = SmartCache(
            max_size_mb=self.config.cache_size_mb,
            ttl_seconds=self.config.cache_ttl_seconds,
            enable_compression=self.config.enable_compression
        )
        
        # 3. Quantize and prune model
        self._optimize_model()
        
        # Performance tracking
        self.metrics = {
            'cpu_usage': [],
            'latency': [],
            'cache_hit_rate': 0,
            'memory_saved': 0
        }
        
        logger.info("✅ Phase 2 Optimized System ready")
        logger.info(f"   Target CPU: {self.config.target_cpu}%")
        logger.info(f"   Target latency: {self.config.target_latency_ms}ms")
    
    def _optimize_model(self):
        """Apply quantization and pruning to model"""
        logger.info("Optimizing model with quantization and pruning...")
        
        # Generate calibration data
        calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(10)]
        
        # Create quantization config
        quant_config = QuantizationConfig(
            default_type=self.config.quantization_type,
            pruning_sparsity=self.config.pruning_sparsity,
            dynamic_threshold=0.95
        )
        
        # Apply quantization and pruning
        self.optimized_model = create_quantized_model(
            self.base_model,
            quant_config,
            calibration_data
        )
        
        # Wrap with caching
        self.cached_model = CachedModel(self.optimized_model, self.cache)
        
        # Calculate memory savings
        original_params = sum(p.numel() for p in self.base_model.parameters())
        optimized_params = sum(
            p.numel() for p in self.optimized_model.parameters() 
            if p.requires_grad
        )
        
        self.metrics['memory_saved'] = 1 - (optimized_params / original_params)
        logger.info(f"Model optimized: {self.metrics['memory_saved']:.1%} memory saved")
    
    async def process_vision_task(self, image: np.ndarray, 
                                 task_type: str = "inference") -> Dict[str, Any]:
        """
        Process vision task through optimized pipeline
        Uses parallel processing, caching, and quantized inference
        """
        start_time = time.time()
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float() / 255.0
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        else:
            image_tensor = image
        
        # Check temporal coherence cache first
        temporal_result = self.cache.exploit_temporal_coherence(image_tensor)
        if temporal_result is not None:
            logger.debug("Using temporal coherence cache")
            return temporal_result
        
        # Process through parallel pipeline
        try:
            # Submit to pipeline
            result = await self.pipeline.process_async(image, task_type)
            
            # Run quantized inference with caching
            with torch.no_grad():
                output = self.cached_model(image_tensor)
            
            # Prepare result
            processed_result = {
                'output': output,
                'predictions': output.argmax(dim=-1).tolist() if output.dim() > 1 else output.item(),
                'confidence': output.max().item() if output.dim() > 1 else 1.0,
                'latency_ms': (time.time() - start_time) * 1000,
                'cache_stats': self.cache.get_stats()
            }
            
            # Update temporal cache
            self.cache.update_temporal_result(processed_result)
            
            # Track metrics
            self._update_metrics(processed_result['latency_ms'])
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    def _update_metrics(self, latency_ms: float):
        """Update performance metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        self.metrics['cpu_usage'].append(cpu_usage)
        if len(self.metrics['cpu_usage']) > 100:
            self.metrics['cpu_usage'].pop(0)
        
        # Latency
        self.metrics['latency'].append(latency_ms)
        if len(self.metrics['latency']) > 100:
            self.metrics['latency'].pop(0)
        
        # Cache hit rate
        cache_stats = self.cache.get_stats()
        self.metrics['cache_hit_rate'] = cache_stats['hit_rate']
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'avg_latency_ms': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'cache_hit_rate': self.metrics['cache_hit_rate'],
            'memory_saved': self.metrics['memory_saved'],
            'pipeline_metrics': self.pipeline.get_metrics(),
            'cache_stats': self.cache.get_stats(),
            'target_cpu': self.config.target_cpu,
            'target_latency_ms': self.config.target_latency_ms
        }
    
    def verify_performance_targets(self) -> Dict[str, bool]:
        """Verify if performance targets are met"""
        metrics = self.get_performance_metrics()
        
        return {
            'cpu_target_met': metrics['avg_cpu_usage'] <= self.config.target_cpu,
            'latency_target_met': metrics['avg_latency_ms'] <= self.config.target_latency_ms,
            'cache_effective': metrics['cache_hit_rate'] > 0.5,
            'memory_optimized': metrics['memory_saved'] >= 0.5
        }
    
    def shutdown(self):
        """Shutdown system"""
        self.pipeline.shutdown()
        self.cache.clear()
        logger.info("Phase 2 system shutdown complete")

async def test_phase2_system():
    """Test Phase 2 integrated system"""
    logger.info("=" * 60)
    logger.info("PHASE 2 ARCHITECTURE OPTIMIZATION TEST")
    logger.info("=" * 60)
    
    # Create test model
    test_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    # Create Phase 2 system
    config = Phase2Config(
        target_cpu=35.0,
        target_latency_ms=100.0,
        quantization_type=QuantizationType.INT8,
        pruning_sparsity=0.5
    )
    
    system = Phase2OptimizedSystem(test_model, config)
    
    # Test with multiple images
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
    
    logger.info("\n🔄 Processing test images...")
    
    for i, image in enumerate(test_images):
        result = await system.process_vision_task(image)
        
        if i % 5 == 0:
            logger.info(f"\nImage {i}:")
            logger.info(f"  Latency: {result.get('latency_ms', 0):.1f}ms")
            logger.info(f"  Cache hit rate: {result.get('cache_stats', {}).get('hit_rate', 0):.1%}")
        
        # Add some temporal coherence
        if i == 10:
            # Submit very similar image
            similar_image = test_images[9] + np.random.randint(-5, 5, test_images[9].shape)
            result = await system.process_vision_task(similar_image)
            logger.info(f"\nSimilar image (temporal coherence test):")
            logger.info(f"  Latency: {result.get('latency_ms', 0):.1f}ms")
        
        await asyncio.sleep(0.1)  # Simulate real-time processing
    
    # Final metrics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL PERFORMANCE METRICS:")
    logger.info("=" * 60)
    
    metrics = system.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    # Verify targets
    logger.info("\n" + "=" * 60)
    logger.info("TARGET VERIFICATION:")
    logger.info("=" * 60)
    
    targets_met = system.verify_performance_targets()
    for target, met in targets_met.items():
        status = "✅ PASS" if met else "❌ FAIL"
        logger.info(f"{target}: {status}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"Average CPU usage: {metrics['avg_cpu_usage']:.1f}% (target: 35%)")
    logger.info(f"Average latency: {metrics['avg_latency_ms']:.1f}ms (target: 100ms)")
    logger.info(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    logger.info(f"Memory saved: {metrics['memory_saved']:.1%}")
    
    reduction = (75 - metrics['avg_cpu_usage']) / 75 * 100  # From ~75% baseline
    logger.info(f"\n🎯 CPU reduction achieved: {reduction:.0f}%")
    
    system.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_phase2_system())