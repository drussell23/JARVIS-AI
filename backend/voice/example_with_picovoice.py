#!/usr/bin/env python3
"""
Example: Using JARVIS with Picovoice for ultra-fast wake word detection
"""

import os
import asyncio
import numpy as np
import time

# Set your Picovoice key (normally this would be in .env)
os.environ["PICOVOICE_ACCESS_KEY"] = "e9AVn4el49rhJxxUDILvK2vYOTbFZx1ZSlSnsZqEu3kLY3Ix8Ghckg=="
os.environ["USE_PICOVOICE"] = "true"

from voice.optimized_voice_system import create_optimized_jarvis

async def main():
    """Example of using optimized JARVIS with Picovoice"""
    
    # Your Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")
    
    print("🚀 Starting JARVIS with Picovoice wake word detection...")
    print("=" * 60)
    
    # Create optimized system for 16GB MacBook Pro
    system = await create_optimized_jarvis(api_key, "16gb_macbook_pro")
    
    # Show initial stats
    stats = system.get_optimization_stats()
    print(f"\n📊 System initialized:")
    print(f"   - Picovoice enabled: {system.config.use_picovoice}")
    print(f"   - Wake word threshold: {system.config.wake_word_threshold_default}")
    print(f"   - Memory usage: {stats['model_stats']['memory_usage_mb']:.1f}MB")
    print(f"   - Acceleration: {stats['acceleration']}")
    
    print("\n🎤 Simulating audio input (in real use, this would be from microphone)...")
    print("   Picovoice provides ~10ms wake word detection latency!")
    
    # Simulate continuous audio streaming
    sample_rate = 16000
    chunk_duration = 0.1  # 100ms chunks
    
    for i in range(10):
        # In real use, this would be audio from microphone
        audio_chunk = np.random.randn(int(sample_rate * chunk_duration)) * 0.1
        
        # Time the detection
        start_time = time.time()
        
        # Process through optimized system
        # Picovoice does initial detection, ML verifies
        result = await system.detect_wake_word(audio_chunk)
        
        detection_time = (time.time() - start_time) * 1000  # ms
        
        if result[0]:
            print(f"✅ Chunk {i}: WAKE WORD DETECTED! "
                  f"(confidence: {result[1]:.3f}, time: {detection_time:.1f}ms)")
        else:
            print(f"   Chunk {i}: No wake word (time: {detection_time:.1f}ms)")
        
        # Small delay to simulate real-time streaming
        await asyncio.sleep(0.05)
    
    # Show final performance metrics
    print("\n📈 Performance Summary:")
    final_stats = system.get_optimization_stats()
    
    print(f"   - Chunks processed: {final_stats['streaming_stats'].get('chunks_processed', 0)}")
    print(f"   - Average latency: {final_stats['streaming_stats'].get('average_latency_ms', 0):.1f}ms")
    print(f"   - Memory usage: {final_stats['model_stats']['memory_usage_mb']:.1f}MB")
    print(f"   - CPU usage: {final_stats['resource_stats'].get('average_cpu_1min', 0):.1f}%")
    
    # Picovoice-specific stats if available
    if hasattr(system, 'hybrid_detector') and system.hybrid_detector:
        if hasattr(system.hybrid_detector, 'picovoice_detector'):
            pv_metrics = system.hybrid_detector.picovoice_detector.get_metrics()
            print(f"\n🎯 Picovoice Stats:")
            print(f"   - Frames processed: {pv_metrics['total_frames_processed']}")
            print(f"   - Wake words detected: {pv_metrics['total_detections']}")
    
    print("\n✨ Benefits of Picovoice integration:")
    print("   - 10ms detection latency (vs 50-250ms traditional)")
    print("   - 1-2% CPU usage (vs 15-25% traditional)")
    print("   - Works offline, no network latency")
    print("   - Handles 'Hey Jarvis' and 'Jarvis' variations")
    
    # Cleanup
    await system.stop()
    print("\n👋 System stopped successfully!")

async def test_sensitivity_levels():
    """Test different sensitivity levels"""
    print("\n🔧 Testing different sensitivity levels...")
    
    # Test with different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        os.environ["WAKE_WORD_THRESHOLD"] = str(threshold)
        print(f"\nTesting with threshold: {threshold}")
        # ... run detection tests ...

if __name__ == "__main__":
    print("JARVIS + Picovoice Integration Example")
    print("=====================================")
    print(f"Picovoice Key: {'✅ Set' if os.getenv('PICOVOICE_ACCESS_KEY') else '❌ Not set'}")
    print()
    
    # Run main example
    asyncio.run(main())
    
    # Optional: test different sensitivities
    # asyncio.run(test_sensitivity_levels())