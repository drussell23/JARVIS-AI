"""
Optimized ML Voice System for 16GB MacBook Pro
Integrates all optimization components
"""

import os
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Import all optimization components
from .ml_enhanced_voice_system import MLEnhancedVoiceSystem
from .optimization_config import OPTIMIZATION_CONFIG, get_preset
from .model_manager import get_model_manager, ModelPriority
from .streaming_processor import StreamingVADProcessor, ChunkedFeatureExtractor
from .coreml_acceleration import UnifiedAccelerator, AcceleratedFeatureExtractor
from .resource_monitor import ResourceMonitor, AdaptiveResourceManager
from .config import VOICE_CONFIG

logger = logging.getLogger(__name__)

class OptimizedVoiceSystem(MLEnhancedVoiceSystem):
    """
    Fully optimized voice system with all enhancements
    """
    
    def __init__(self, anthropic_api_key: str, preset: str = "16gb_macbook_pro"):
        # Use optimized configuration
        self.optimization_config = get_preset(preset)
        
        # Initialize base system
        super().__init__(anthropic_api_key)
        
        # Model manager for efficient memory usage
        self.model_manager = get_model_manager()
        self._register_model_factories()
        
        # Streaming processor
        self.streaming_processor = None
        self._init_streaming_processor()
        
        # Hardware acceleration
        self.accelerator = UnifiedAccelerator(self.optimization_config.macos)
        self.accel_features = AcceleratedFeatureExtractor(self.accelerator)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(self.optimization_config)
        self.resource_manager = AdaptiveResourceManager(self.resource_monitor)
        
        # Register components for adaptive management
        self.resource_manager.register_component("model_manager", self.model_manager)
        self.resource_manager.register_component("streaming_processor", self.streaming_processor)
        self.resource_manager.register_component("voice_system", self)
        
        logger.info(f"Optimized Voice System initialized with preset: {preset}")
    
    def _register_model_factories(self):
        """Register model creation functions with priorities"""
        
        # Essential models (always loaded)
        self.model_manager.register_model_factory(
            "vad",
            lambda: self.vad,
            ModelPriority.ESSENTIAL
        )
        
        # High priority models
        self.model_manager.register_model_factory(
            "feature_scaler",
            lambda: self.feature_scaler,
            ModelPriority.HIGH
        )
        
        # Medium priority models (load on demand)
        self.model_manager.register_model_factory(
            "wake_word_nn",
            lambda: self._create_wake_word_nn(),
            ModelPriority.MEDIUM
        )
        
        self.model_manager.register_model_factory(
            "personalized_svm",
            lambda: self._create_svm(),
            ModelPriority.MEDIUM
        )
        
        # Low priority models (unload quickly)
        self.model_manager.register_model_factory(
            "anomaly_detector",
            lambda: self._create_anomaly_detector(),
            ModelPriority.LOW
        )
    
    def _init_streaming_processor(self):
        """Initialize streaming audio processor"""
        def process_callback(audio_chunk: np.ndarray) -> Dict[str, Any]:
            # Extract features using accelerated methods
            features = self.accel_features.extract_spectral_features(
                audio_chunk, self.config.audio_sample_rate
            )
            return features
        
        def vad_callback(audio_chunk: np.ndarray) -> bool:
            # Use VAD if available
            if self.vad:
                # Convert to 16-bit for VAD
                audio_16bit = (audio_chunk * 32768).astype(np.int16).tobytes()
                return self.vad.is_speech(audio_16bit, self.config.audio_sample_rate)
            return True
        
        self.streaming_processor = StreamingVADProcessor(
            process_callback=process_callback,
            vad_callback=vad_callback,
            config=self.optimization_config.streaming,
            sample_rate=self.config.audio_sample_rate
        )
    
    async def detect_wake_word(self, audio_data: np.ndarray, 
                             user_id: str = "default") -> Tuple[bool, float, Optional[str]]:
        """
        Optimized wake word detection using streaming and lazy loading
        """
        # Check resource usage before processing (macOS-aware)
        snapshot = self.resource_monitor.get_current_snapshot()
        if snapshot:
            available_gb = snapshot.memory_available_mb / 1024.0
            if available_gb < 0.5:  # Less than 500MB available
                logger.warning(f"Memory critical ({available_gb:.1f}GB available), skipping detection")
                return False, 0.0, "System resources exhausted"
        
        # Use streaming processor if enabled
        if self.streaming_processor and self.optimization_config.streaming.low_latency_mode:
            # Feed to streaming processor
            self.streaming_processor.feed_audio(audio_data)
            
            # Get results
            results = self.streaming_processor.get_results(timeout=0.05)
            
            # Process results
            for result in results:
                # Run wake word detection on processed chunks
                if "audio_chunk" in result:
                    detection = await super().detect_wake_word(
                        result["audio_chunk"], user_id
                    )
                    if detection[0]:  # Wake word detected
                        return detection
            
            return False, 0.0, "No wake word in stream"
        
        # Fall back to regular detection
        return await super().detect_wake_word(audio_data, user_id)
    
    def _get_deep_embedding(self, audio_data: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Get deep embedding with lazy loading and acceleration"""
        try:
            # Lazy load neural network
            if not self.wake_word_nn:
                self.wake_word_nn = self.model_manager.get_model("wake_word_nn")
                
                # Accelerate if possible
                if self.wake_word_nn and self.accelerator:
                    self.wake_word_nn = self.accelerator.accelerate_model(
                        self.wake_word_nn,
                        "wake_word_nn",
                        input_shape=(128, 128)  # Mel spectrogram shape
                    )
            
            # Use parent implementation
            return super()._get_deep_embedding(audio_data, sr)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def extract_wake_word_features(self, audio_data: np.ndarray, 
                                 sr: int = 16000) -> Dict[str, Any]:
        """Extract features using hardware acceleration"""
        # Use accelerated feature extraction
        features = self.accel_features.extract_spectral_features(audio_data, sr)
        
        # Add additional features from parent
        parent_features = super().extract_wake_word_features(audio_data, sr)
        features.update(parent_features)
        
        return features
    
    async def start(self):
        """Start optimized voice system"""
        logger.info("Starting Optimized Voice System...")
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Start streaming processor
        if self.streaming_processor:
            self.streaming_processor.start()
        
        # Preload essential models
        if self.optimization_config.model_swap.preload_essential:
            logger.info("Preloading essential models...")
            self.model_manager.get_model("vad")
            self.model_manager.get_model("feature_scaler")
        
        # Start parent system
        await super().start()
        
        # Log resource usage
        stats = self.resource_monitor.get_stats()
        logger.info(f"System started - CPU: {stats.get('average_cpu_1min', 0):.1f}%, "
                   f"Memory: {stats.get('average_memory_1min', 0):.1f}%")
    
    async def stop(self):
        """Stop optimized voice system"""
        logger.info("Stopping Optimized Voice System...")
        
        # Stop streaming
        if self.streaming_processor:
            self.streaming_processor.stop()
        
        # Stop monitoring
        self.resource_monitor.stop()
        
        # Export performance data
        self.resource_monitor.export_history(
            os.path.join(self.model_dir, "resource_history.json")
        )
        
        # Shutdown model manager
        self.model_manager.shutdown()
        
        # Stop parent
        await super().stop()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            "voice_metrics": self.get_performance_metrics(),
            "resource_stats": self.resource_monitor.get_stats(),
            "model_stats": self.model_manager.get_stats(),
            "streaming_stats": self.streaming_processor.get_stats() if self.streaming_processor else {},
            "acceleration": {
                "coreml_available": self.accelerator.coreml is not None,
                "metal_available": self.accelerator.metal is not None
            }
        }
        
        return stats

# Convenience function to create optimized system
async def create_optimized_jarvis(api_key: str, 
                                preset: str = "16gb_macbook_pro") -> OptimizedVoiceSystem:
    """
    Create an optimized JARVIS voice system
    
    Presets:
    - "16gb_macbook_pro": Balanced performance for 16GB RAM
    - "8gb_mac": Memory-optimized for 8GB systems  
    - "32gb_mac_studio": Maximum performance for high-end systems
    """
    system = OptimizedVoiceSystem(api_key, preset)
    await system.start()
    return system

# Example usage
async def demo_optimized_system():
    """Demo the optimized voice system"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
    
    # Create optimized system
    system = await create_optimized_jarvis(api_key, "16gb_macbook_pro")
    
    try:
        print("\n=== Optimized Voice System Demo ===")
        
        # Show initial stats
        stats = system.get_optimization_stats()
        print(f"\nSystem Stats:")
        print(f"- Models loaded: {stats['model_stats']['num_loaded']}")
        print(f"- Memory usage: {stats['model_stats']['memory_usage_mb']:.1f}MB")
        print(f"- CPU average: {stats['resource_stats'].get('average_cpu_1min', 0):.1f}%")
        print(f"- Acceleration: CoreML={stats['acceleration']['coreml_available']}, "
              f"Metal={stats['acceleration']['metal_available']}")
        
        # Test streaming audio processing
        print("\nTesting streaming audio processing...")
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks
        
        for i in range(5):
            # Generate test audio chunk
            audio_chunk = np.random.randn(int(sample_rate * chunk_duration)) * 0.1
            
            # Process through streaming system
            result = await system.detect_wake_word(audio_chunk)
            print(f"Chunk {i}: detected={result[0]}, confidence={result[1]:.3f}")
            
            await asyncio.sleep(0.1)
        
        # Final stats
        print("\nFinal Statistics:")
        final_stats = system.get_optimization_stats()
        print(f"- Chunks processed: {final_stats['streaming_stats'].get('chunks_processed', 0)}")
        print(f"- Average latency: {final_stats['streaming_stats'].get('average_latency_ms', 0):.1f}ms")
        print(f"- Models in memory: {final_stats['model_stats']['num_loaded']}")
        print(f"- Total adaptations: {final_stats['resource_stats'].get('total_adaptations', 0)}")
        
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(demo_optimized_system())