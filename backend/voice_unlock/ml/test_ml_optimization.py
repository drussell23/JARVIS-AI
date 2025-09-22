#!/usr/bin/env python3
"""
Test Script for ML Optimization in Voice Unlock
===============================================

Demonstrates memory-efficient ML operations for 16GB RAM systems.
"""

import os
import sys
import time
import psutil
import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.voice_unlock.ml import VoiceUnlockMLSystem
from backend.voice_unlock.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLOptimizationTester:
    """Test harness for ML optimization features"""
    
    def __init__(self):
        self.ml_system = VoiceUnlockMLSystem()
        self.results = {
            'memory_usage': [],
            'inference_times': [],
            'cache_performance': [],
            'model_loads': []
        }
        
    def generate_test_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic test audio"""
        # Generate speech-like audio with formants
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Fundamental frequency (pitch)
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Varying pitch
        
        # Formants (characteristic of speech)
        f1 = 700 + 100 * np.sin(2 * np.pi * 0.3 * t)
        f2 = 1700 + 200 * np.sin(2 * np.pi * 0.2 * t)
        
        # Generate audio with formants
        audio = (0.3 * np.sin(2 * np.pi * f0 * t) +
                0.2 * np.sin(2 * np.pi * f1 * t) +
                0.1 * np.sin(2 * np.pi * f2 * t))
        
        # Add some noise
        noise = np.random.normal(0, 0.01, audio.shape)
        audio += noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)
        
    def test_memory_efficiency(self, num_users: int = 10):
        """Test memory usage with multiple users"""
        logger.info(f"Testing memory efficiency with {num_users} users...")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Enroll multiple users
        for i in range(num_users):
            user_id = f"test_user_{i}"
            
            # Generate enrollment samples
            samples = []
            for j in range(3):
                audio = self.generate_test_audio()
                samples.append(audio)
                
            # Enroll user
            result = self.ml_system.enroll_user(user_id, samples)
            
            # Record memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            self.results['memory_usage'].append({
                'user_count': i + 1,
                'memory_mb': current_memory,
                'increase_mb': memory_increase,
                'success': result['success']
            })
            
            logger.info(f"User {i+1}: Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
        return self.results['memory_usage']
        
    def test_dynamic_loading(self, num_authentications: int = 50):
        """Test dynamic model loading/unloading"""
        logger.info(f"Testing dynamic loading with {num_authentications} authentications...")
        
        # Create 5 test users
        user_ids = []
        for i in range(5):
            user_id = f"dynamic_user_{i}"
            samples = [self.generate_test_audio() for _ in range(3)]
            self.ml_system.enroll_user(user_id, samples)
            user_ids.append(user_id)
            
        # Perform authentications with different users
        for i in range(num_authentications):
            # Pick random user
            user_id = user_ids[i % len(user_ids)]
            
            # Generate test audio
            audio = self.generate_test_audio()
            
            # Time the authentication
            start_time = time.time()
            result = self.ml_system.authenticate_user(user_id, audio)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            self.results['inference_times'].append({
                'attempt': i + 1,
                'user_id': user_id,
                'inference_ms': inference_time,
                'authenticated': result['authenticated'],
                'cache_hit': i > 0 and (i % len(user_ids)) < len(user_ids)
            })
            
            if i % 10 == 0:
                logger.info(f"Auth {i+1}: {inference_time:.1f}ms")
                
        return self.results['inference_times']
        
    def test_cache_performance(self):
        """Test cache hit rates and performance impact"""
        logger.info("Testing cache performance...")
        
        # Create test user
        user_id = "cache_test_user"
        samples = [self.generate_test_audio() for _ in range(3)]
        self.ml_system.enroll_user(user_id, samples)
        
        # Perform repeated authentications
        for round_num in range(3):
            logger.info(f"Cache test round {round_num + 1}")
            
            for i in range(10):
                audio = self.generate_test_audio()
                
                start_time = time.time()
                result = self.ml_system.authenticate_user(user_id, audio)
                inference_time = (time.time() - start_time) * 1000
                
                # Get cache stats
                ml_stats = self.ml_system.ml_manager.get_performance_report()
                
                self.results['cache_performance'].append({
                    'round': round_num + 1,
                    'attempt': i + 1,
                    'inference_ms': inference_time,
                    'cache_hit_rate': ml_stats['cache']['hit_rate'],
                    'cache_size_mb': ml_stats['cache']['size_mb']
                })
                
            # Clear cache between rounds to test cold start
            if round_num < 2:
                self.ml_system.ml_manager.cache.clear()
                logger.info("Cache cleared")
                
        return self.results['cache_performance']
        
    def test_quantization_impact(self):
        """Test impact of model quantization"""
        logger.info("Testing quantization impact...")
        
        # Test with and without quantization
        for use_quantization in [False, True]:
            # Update configuration
            self.ml_system.ml_manager.config['enable_quantization'] = use_quantization
            
            user_id = f"quant_test_{use_quantization}"
            samples = [self.generate_test_audio() for _ in range(3)]
            
            # Enroll
            enroll_result = self.ml_system.enroll_user(user_id, samples)
            
            # Test authentication accuracy and speed
            correct = 0
            total_time = 0
            
            for i in range(20):
                audio = self.generate_test_audio()
                
                start_time = time.time()
                result = self.ml_system.authenticate_user(user_id, audio)
                inference_time = (time.time() - start_time) * 1000
                
                if result['authenticated']:
                    correct += 1
                total_time += inference_time
                
            accuracy = correct / 20
            avg_time = total_time / 20
            
            # Get memory usage
            ml_stats = self.ml_system.ml_manager.get_performance_report()
            
            logger.info(f"Quantization={use_quantization}: "
                       f"Accuracy={accuracy:.2f}, "
                       f"Avg time={avg_time:.1f}ms, "
                       f"Memory={ml_stats['memory']['ml_memory_mb']:.1f}MB")
                       
        # Restore default
        self.ml_system.ml_manager.config['enable_quantization'] = True
        
    def test_stress_scenario(self):
        """Stress test with high load"""
        logger.info("Running stress test...")
        
        # Monitor memory during stress test
        memory_timeline = []
        start_time = time.time()
        
        # Create many users quickly
        for i in range(20):
            user_id = f"stress_user_{i}"
            samples = [self.generate_test_audio() for _ in range(3)]
            
            self.ml_system.enroll_user(user_id, samples)
            
            # Rapid authentications
            for j in range(5):
                audio = self.generate_test_audio()
                self.ml_system.authenticate_user(user_id, audio)
                
            # Record memory
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            elapsed = time.time() - start_time
            memory_timeline.append((elapsed, memory_mb))
            
            logger.info(f"Stress test user {i+1}/20: Memory={memory_mb:.1f}MB")
            
        return memory_timeline
        
    def generate_report(self, output_dir: str = "ml_optimization_report"):
        """Generate comprehensive test report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Memory usage plot
        if self.results['memory_usage']:
            ax = axes[0, 0]
            users = [r['user_count'] for r in self.results['memory_usage']]
            memory = [r['memory_mb'] for r in self.results['memory_usage']]
            ax.plot(users, memory, 'b-o')
            ax.set_xlabel('Number of Users')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage vs Number of Users')
            ax.grid(True)
            
        # 2. Inference time plot
        if self.results['inference_times']:
            ax = axes[0, 1]
            attempts = [r['attempt'] for r in self.results['inference_times']]
            times = [r['inference_ms'] for r in self.results['inference_times']]
            ax.plot(attempts, times, 'g-', alpha=0.7)
            ax.set_xlabel('Authentication Attempt')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Inference Time Over Multiple Attempts')
            ax.grid(True)
            
        # 3. Cache performance
        if self.results['cache_performance']:
            ax = axes[1, 0]
            cache_data = self.results['cache_performance']
            rounds = {}
            for item in cache_data:
                round_num = item['round']
                if round_num not in rounds:
                    rounds[round_num] = {'attempts': [], 'times': []}
                rounds[round_num]['attempts'].append(item['attempt'])
                rounds[round_num]['times'].append(item['inference_ms'])
                
            for round_num, data in rounds.items():
                ax.plot(data['attempts'], data['times'], 
                       label=f'Round {round_num}', alpha=0.7)
            ax.set_xlabel('Attempt within Round')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Cache Performance Across Rounds')
            ax.legend()
            ax.grid(True)
            
        # 4. Performance summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate summary statistics
        summary_text = "ML Optimization Summary\n" + "="*30 + "\n\n"
        
        if self.results['memory_usage']:
            final_memory = self.results['memory_usage'][-1]['memory_mb']
            memory_per_user = self.results['memory_usage'][-1]['increase_mb'] / len(self.results['memory_usage'])
            summary_text += f"Final Memory Usage: {final_memory:.1f} MB\n"
            summary_text += f"Avg Memory per User: {memory_per_user:.1f} MB\n\n"
            
        if self.results['inference_times']:
            times = [r['inference_ms'] for r in self.results['inference_times']]
            summary_text += f"Avg Inference Time: {np.mean(times):.1f} ms\n"
            summary_text += f"Min Inference Time: {np.min(times):.1f} ms\n"
            summary_text += f"Max Inference Time: {np.max(times):.1f} ms\n\n"
            
        # Get current performance report
        perf_report = self.ml_system.get_performance_report()
        summary_text += f"Cache Hit Rate: {perf_report['ml_performance']['cache']['hit_rate']:.1f}%\n"
        summary_text += f"Active Models: {perf_report['system_stats']['models']['active']}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_optimization_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump({
                'test_results': self.results,
                'final_performance_report': perf_report,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        logger.info(f"Report generated in {output_dir}/")


def main():
    """Run ML optimization tests"""
    logger.info("Starting ML optimization tests...")
    
    tester = MLOptimizationTester()
    
    # Run tests
    try:
        # Test 1: Memory efficiency
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Memory Efficiency")
        logger.info("="*50)
        tester.test_memory_efficiency(num_users=10)
        
        # Test 2: Dynamic loading
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Dynamic Model Loading")
        logger.info("="*50)
        tester.test_dynamic_loading(num_authentications=50)
        
        # Test 3: Cache performance
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Cache Performance")
        logger.info("="*50)
        tester.test_cache_performance()
        
        # Test 4: Quantization impact
        logger.info("\n" + "="*50)
        logger.info("TEST 4: Quantization Impact")
        logger.info("="*50)
        tester.test_quantization_impact()
        
        # Test 5: Stress test
        logger.info("\n" + "="*50)
        logger.info("TEST 5: Stress Test")
        logger.info("="*50)
        stress_results = tester.test_stress_scenario()
        
        # Generate report
        logger.info("\n" + "="*50)
        logger.info("Generating Report")
        logger.info("="*50)
        tester.generate_report()
        
        # Show final performance
        logger.info("\n" + "="*50)
        logger.info("Final Performance Report")
        logger.info("="*50)
        final_report = tester.ml_system.get_performance_report()
        logger.info(json.dumps(final_report, indent=2))
        
        # Export diagnostics
        tester.ml_system.export_diagnostics("ml_diagnostics.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        tester.ml_system.cleanup()
        
    logger.info("\nML optimization tests complete!")


if __name__ == "__main__":
    main()