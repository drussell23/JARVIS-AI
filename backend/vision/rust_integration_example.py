#!/usr/bin/env python3
"""
Rust Integration Example for JARVIS Vision System
Demonstrates how to use Rust processor to reduce CPU usage
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any

# Import the Rust processor
try:
    from .rust_vision_processor import RustVisionProcessor

    RUST_AVAILABLE = True
except ImportError:
    # Fallback if import fails
    RUST_AVAILABLE = False
    print("⚠️  Rust processor not available, using Python fallback")

logger = logging.getLogger(__name__)


class OptimizedVisionSystem:
    """Vision system with Rust optimization for reduced CPU usage"""

    def __init__(self):
        self.rust_processor = RustVisionProcessor() if RUST_AVAILABLE else None
        self.processing_stats = {
            "total_operations": 0,
            "rust_operations": 0,
            "python_fallbacks": 0,
            "total_processing_time": 0.0,
            "cpu_usage_samples": [],
        }

        if self.rust_processor and self.rust_processor.rust_available:
            logger.info("🚀 Rust-optimized vision system initialized")
        else:
            logger.warning("⚠️  Using Python-only vision system")

    async def process_vision_command(
        self, command: str, image_data: np.ndarray
    ) -> Dict[str, Any]:
        """Process vision commands with Rust optimization"""
        start_time = time.time()

        try:
            # Use Rust processor if available
            if self.rust_processor and self.rust_processor.rust_available:
                processed_data = self.rust_processor.process_vision_data(image_data)
                self.processing_stats["rust_operations"] += 1
                logger.debug("✅ Vision processing completed with Rust")
            else:
                # Fallback to Python processing
                processed_data = self._python_vision_processing(image_data)
                self.processing_stats["python_fallbacks"] += 1
                logger.debug("⚠️  Vision processing completed with Python fallback")

            # Analyze the processed data
            analysis_result = await self._analyze_vision_data(processed_data, command)

            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["total_operations"] += 1
            self.processing_stats["total_processing_time"] += processing_time

            return {
                "success": True,
                "command": command,
                "processed_data_shape": processed_data.shape,
                "analysis": analysis_result,
                "processing_time": processing_time,
                "processor_used": (
                    "rust"
                    if self.rust_processor and self.rust_processor.rust_available
                    else "python"
                ),
            }

        except Exception as e:
            logger.error(f"Error processing vision command: {e}")
            return {"success": False, "error": str(e), "command": command}

    async def process_audio_command(
        self, command: str, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Process audio commands with Rust optimization"""
        start_time = time.time()

        try:
            # Use Rust processor if available
            if self.rust_processor and self.rust_processor.rust_available:
                processed_audio = self.rust_processor.process_audio_data(
                    audio_data, sample_rate
                )
                self.processing_stats["rust_operations"] += 1
                logger.debug("✅ Audio processing completed with Rust")
            else:
                # Fallback to Python processing
                processed_audio = self._python_audio_processing(audio_data, sample_rate)
                self.processing_stats["python_fallbacks"] += 1
                logger.debug("⚠️  Audio processing completed with Python fallback")

            # Analyze the processed audio
            analysis_result = await self._analyze_audio_data(processed_audio, command)

            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["total_operations"] += 1
            self.processing_stats["total_processing_time"] += processing_time

            return {
                "success": True,
                "command": command,
                "processed_audio_shape": processed_audio.shape,
                "analysis": analysis_result,
                "processing_time": processing_time,
                "processor_used": (
                    "rust"
                    if self.rust_processor and self.rust_processor.rust_available
                    else "python"
                ),
            }

        except Exception as e:
            logger.error(f"Error processing audio command: {e}")
            return {"success": False, "error": str(e), "command": command}

    async def _analyze_vision_data(
        self, processed_data: np.ndarray, command: str
    ) -> Dict[str, Any]:
        """Analyze processed vision data"""
        # Simulate vision analysis
        await asyncio.sleep(0.01)  # Small delay to simulate processing

        return {
            "data_type": "vision",
            "data_size": processed_data.size,
            "mean_value": float(np.mean(processed_data)),
            "std_value": float(np.std(processed_data)),
            "command_understood": True,
        }

    async def _analyze_audio_data(
        self, processed_audio: np.ndarray, command: str
    ) -> Dict[str, Any]:
        """Analyze processed audio data"""
        # Simulate audio analysis
        await asyncio.sleep(0.01)  # Small delay to simulate processing

        return {
            "data_type": "audio",
            "data_size": processed_audio.size,
            "mean_value": float(np.mean(processed_audio)),
            "std_value": float(np.std(processed_audio)),
            "command_understood": True,
        }

    def _python_vision_processing(self, image_data: np.ndarray) -> np.ndarray:
        """Python fallback for vision processing"""
        # Simple normalization
        normalized = (image_data - 0.5) * 2.0
        return np.clip(normalized, -1.0, 1.0)

    def _python_audio_processing(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Python fallback for audio processing"""
        # Simple audio processing
        windowed = audio_data * 0.54  # Hamming window
        normalized = windowed * 2.0
        return np.clip(normalized, -1.0, 1.0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        total_ops = self.processing_stats["total_operations"]
        avg_time = (
            (self.processing_stats["total_processing_time"] / total_ops)
            if total_ops > 0
            else 0
        )

        rust_percentage = (
            (self.processing_stats["rust_operations"] / total_ops * 100)
            if total_ops > 0
            else 0
        )
        python_percentage = (
            (self.processing_stats["python_fallbacks"] / total_ops * 100)
            if total_ops > 0
            else 0
        )

        return {
            "total_operations": total_ops,
            "rust_operations": self.processing_stats["rust_operations"],
            "python_fallbacks": self.processing_stats["python_fallbacks"],
            "rust_usage_percentage": rust_percentage,
            "python_usage_percentage": python_percentage,
            "average_processing_time": avg_time,
            "total_processing_time": self.processing_stats["total_processing_time"],
        }

    def benchmark_system(self, num_operations: int = 100) -> Dict[str, Any]:
        """Benchmark the system performance"""
        logger.info(f"🔍 Benchmarking system with {num_operations} operations...")

        # Generate test data
        vision_data = np.random.random((224, 224, 3)).astype(np.float32)
        audio_data = np.random.random(1000).astype(np.float32)

        # Benchmark vision processing
        vision_start = time.time()
        for i in range(num_operations):
            if self.rust_processor and self.rust_processor.rust_available:
                self.rust_processor.process_vision_data(vision_data)
            else:
                self._python_vision_processing(vision_data)
        vision_time = time.time() - vision_start

        # Benchmark audio processing
        audio_start = time.time()
        for i in range(num_operations):
            if self.rust_processor and self.rust_processor.rust_available:
                self.rust_processor.process_audio_data(audio_data, 44100)
            else:
                self._python_audio_processing(audio_data, 44100)
        audio_time = time.time() - audio_start

        return {
            "num_operations": num_operations,
            "vision_processing_time": vision_time,
            "audio_processing_time": audio_time,
            "total_benchmark_time": vision_time + audio_time,
            "operations_per_second": num_operations / ((vision_time + audio_time) / 2),
            "rust_available": self.rust_processor
            and self.rust_processor.rust_available,
        }


# Example usage
async def main():
    """Example of using the optimized vision system"""
    logging.basicConfig(level=logging.INFO)

    # Initialize the system
    vision_system = OptimizedVisionSystem()

    # Generate test data
    test_image = np.random.random((224, 224, 3)).astype(np.float32)
    test_audio = np.random.random(1000).astype(np.float32)

    print("🧪 Testing Rust-optimized vision system...")

    # Test vision processing
    vision_result = await vision_system.process_vision_command(
        "can you see my screen?", test_image
    )
    print(f"✅ Vision processing: {vision_result}")

    # Test audio processing
    audio_result = await vision_system.process_audio_command(
        "what did I just say?", test_audio, 44100
    )
    print(f"✅ Audio processing: {audio_result}")

    # Get performance stats
    stats = vision_system.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(
        f"  Rust operations: {stats['rust_operations']} ({stats['rust_usage_percentage']:.1f}%)"
    )
    print(
        f"  Python fallbacks: {stats['python_fallbacks']} ({stats['python_usage_percentage']:.1f}%)"
    )
    print(f"  Average processing time: {stats['average_processing_time']:.4f}s")

    # Run benchmark
    benchmark = vision_system.benchmark_system(50)
    print(f"\n🚀 Benchmark Results:")
    print(f"  Operations per second: {benchmark['operations_per_second']:.1f}")
    print(f"  Vision processing time: {benchmark['vision_processing_time']:.3f}s")
    print(f"  Audio processing time: {benchmark['audio_processing_time']:.3f}s")
    print(f"  Rust available: {benchmark['rust_available']}")

    print(f"\n🎉 Rust integration test completed!")


if __name__ == "__main__":
    asyncio.run(main())
