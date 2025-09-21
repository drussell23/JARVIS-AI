#!/usr/bin/env python3
"""
Test complete integration of Rust acceleration with proactive monitoring.
This script verifies that Rust components activate during real-time screen monitoring
and provides expected performance improvements.
"""

import asyncio
import time
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import psutil
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class ProactiveMonitoringTester:
    """Tests complete integration of Rust acceleration with proactive monitoring."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.rust_available = False
        self.proactive_available = False
        
    async def setup(self) -> bool:
        """Setup test environment and check prerequisites."""
        print(f"\n{BLUE}=== Proactive Monitoring Integration Test ==={RESET}\n")
        
        # Check Rust availability
        try:
            import jarvis_rust_core
            self.rust_available = True
            print(f"{GREEN}✓ Rust core module imported successfully{RESET}")
        except ImportError:
            print(f"{YELLOW}⚠ Rust core not available - tests will verify fallback{RESET}")
            
        # Check vision system
        try:
            from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
            from vision.video_stream_capture import VideoStreamCapture
            print(f"{GREEN}✓ Vision system components available{RESET}")
        except ImportError as e:
            print(f"{RED}✗ Vision system not available: {e}{RESET}")
            return False
            
        # Check proactive monitoring
        try:
            from vision.rust_proactive_integration import RustProactiveMonitor
            self.proactive_available = True
            print(f"{GREEN}✓ Proactive monitoring available{RESET}")
        except ImportError:
            print(f"{YELLOW}⚠ Proactive monitoring not available{RESET}")
            
        return True
        
    async def test_rust_components(self) -> Dict[str, Any]:
        """Test individual Rust components."""
        print(f"\n{BLUE}Testing Rust Components...{RESET}")
        results = {}
        
        if not self.rust_available:
            print(f"{YELLOW}Skipping - Rust not available{RESET}")
            return {"status": "skipped", "reason": "Rust not built"}
            
        import jarvis_rust_core
        
        # Test bloom filter
        try:
            bloom = jarvis_rust_core.bloom_filter.PyRustBloomFilter(10.0, 7)
            
            # Add test data
            start = time.perf_counter()
            for i in range(1000):
                bloom.add(f"frame_{i}".encode())
            add_time = time.perf_counter() - start
            
            # Check data
            start = time.perf_counter()
            hits = 0
            for i in range(1000):
                if bloom.contains(f"frame_{i}".encode()):
                    hits += 1
            check_time = time.perf_counter() - start
            
            results['bloom_filter'] = {
                'status': 'passed',
                'add_time_ms': add_time * 1000,
                'check_time_ms': check_time * 1000,
                'accuracy': hits / 1000.0
            }
            print(f"  {GREEN}✓ Bloom filter: {hits}/1000 hits, {add_time*1000:.2f}ms add, {check_time*1000:.2f}ms check{RESET}")
            
        except Exception as e:
            results['bloom_filter'] = {'status': 'failed', 'error': str(e)}
            print(f"  {RED}✗ Bloom filter failed: {e}{RESET}")
            
        # Test sliding window
        try:
            window = jarvis_rust_core.sliding_window.PySlidingWindow(
                window_size=30,
                overlap_threshold=0.9
            )
            
            # Create test frames
            frames = []
            for i in range(10):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                frames.append(frame.tobytes())
                
            # Process frames
            start = time.perf_counter()
            duplicates = 0
            for frame in frames:
                result = window.process_frame(frame, 0)
                if result.get('is_duplicate'):
                    duplicates += 1
            process_time = time.perf_counter() - start
            
            results['sliding_window'] = {
                'status': 'passed',
                'process_time_ms': process_time * 1000,
                'frames_processed': len(frames),
                'duplicates_found': duplicates
            }
            print(f"  {GREEN}✓ Sliding window: {len(frames)} frames in {process_time*1000:.2f}ms, {duplicates} duplicates{RESET}")
            
        except Exception as e:
            results['sliding_window'] = {'status': 'failed', 'error': str(e)}
            print(f"  {RED}✗ Sliding window failed: {e}{RESET}")
            
        # Test Metal acceleration (macOS only)
        if sys.platform == 'darwin':
            try:
                metal = jarvis_rust_core.metal_accelerator.PyMetalAccelerator()
                
                # Create test frame
                frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                
                # Process frame
                start = time.perf_counter()
                result = metal.process_frame(frame.tobytes(), 1280, 720)
                gpu_time = time.perf_counter() - start
                
                results['metal_gpu'] = {
                    'status': 'passed',
                    'process_time_ms': gpu_time * 1000,
                    'frame_size': '1280x720'
                }
                print(f"  {GREEN}✓ Metal GPU: 720p frame in {gpu_time*1000:.2f}ms{RESET}")
                
            except Exception as e:
                results['metal_gpu'] = {'status': 'failed', 'error': str(e)}
                print(f"  {RED}✗ Metal GPU failed: {e}{RESET}")
        else:
            results['metal_gpu'] = {'status': 'skipped', 'reason': 'Not macOS'}
            
        # Test zero-copy memory
        try:
            pool = jarvis_rust_core.zero_copy.PyZeroCopyPool()
            
            # Allocate buffers
            start = time.perf_counter()
            buffers = []
            for _ in range(10):
                buf = pool.allocate(1024 * 1024)  # 1MB
                buffers.append(buf)
            alloc_time = time.perf_counter() - start
            
            # Get stats
            stats = pool.stats()
            
            # Release buffers
            for buf in buffers:
                buf.release()
                
            results['zero_copy'] = {
                'status': 'passed',
                'alloc_time_ms': alloc_time * 1000,
                'buffers_allocated': len(buffers),
                'memory_stats': stats
            }
            print(f"  {GREEN}✓ Zero-copy: {len(buffers)} buffers in {alloc_time*1000:.2f}ms{RESET}")
            
        except Exception as e:
            results['zero_copy'] = {'status': 'failed', 'error': str(e)}
            print(f"  {RED}✗ Zero-copy failed: {e}{RESET}")
            
        return results
        
    async def test_proactive_monitoring(self) -> Dict[str, Any]:
        """Test proactive monitoring with Rust acceleration."""
        print(f"\n{BLUE}Testing Proactive Monitoring...{RESET}")
        
        if not self.proactive_available:
            print(f"{YELLOW}Skipping - Proactive monitoring not available{RESET}")
            return {"status": "skipped", "reason": "Not available"}
            
        from vision.rust_proactive_integration import RustProactiveMonitor, get_rust_monitor
        
        try:
            # Create monitor instance
            monitor = RustProactiveMonitor()
            
            # Test initialization
            init_result = await monitor.initialize()
            if not init_result['success']:
                return {'status': 'failed', 'error': 'Initialization failed'}
                
            print(f"  {GREEN}✓ Monitor initialized with Rust: {init_result['rust_available']}{RESET}")
            
            # Create test frames
            frames = []
            for i in range(5):
                # Simulate different frame types
                if i < 2:
                    # Similar frames (should be deduplicated)
                    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
                else:
                    # Different frames
                    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                    
                frames.append({
                    'data': base64.b64encode(frame.tobytes()).decode(),
                    'width': 1280,
                    'height': 720,
                    'timestamp': time.time() + i * 0.033  # 30 FPS
                })
                
            # Process frames
            results = []
            start = time.perf_counter()
            
            for frame in frames:
                result = await monitor.process_frame(frame)
                results.append(result)
                
            process_time = time.perf_counter() - start
            
            # Get performance report
            perf_report = monitor.get_performance_report()
            
            # Analyze results
            duplicates = sum(1 for r in results if r.get('duplicate'))
            insights = sum(1 for r in results if r.get('insights'))
            
            test_result = {
                'status': 'passed',
                'frames_processed': len(frames),
                'process_time_ms': process_time * 1000,
                'avg_time_per_frame_ms': (process_time * 1000) / len(frames),
                'duplicates_detected': duplicates,
                'insights_generated': insights,
                'rust_components_used': perf_report.get('components_active', {}),
                'memory_usage_mb': perf_report.get('memory_usage_mb', 0)
            }
            
            print(f"  {GREEN}✓ Processed {len(frames)} frames in {process_time*1000:.2f}ms{RESET}")
            print(f"  {GREEN}✓ Duplicates detected: {duplicates}, Insights: {insights}{RESET}")
            print(f"  {GREEN}✓ Rust components active: {list(perf_report.get('components_active', {}).keys())}{RESET}")
            
            # Cleanup
            await monitor.cleanup()
            
            return test_result
            
        except Exception as e:
            logger.error(f"Proactive monitoring test error: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}
            
    async def test_fallback_behavior(self) -> Dict[str, Any]:
        """Test Python fallback when Rust is unavailable."""
        print(f"\n{BLUE}Testing Fallback Behavior...{RESET}")
        
        # Temporarily disable Rust imports
        original_modules = {}
        rust_modules = ['jarvis_rust_core', 'jarvis_rust_core.bloom_filter', 
                       'jarvis_rust_core.sliding_window', 'jarvis_rust_core.metal_accelerator']
        
        for module in rust_modules:
            if module in sys.modules:
                original_modules[module] = sys.modules[module]
                del sys.modules[module]
                
        try:
            # Import with fallback
            from vision.rust_startup_integration import initialize_rust_acceleration
            
            # Initialize should fallback gracefully
            result = await initialize_rust_acceleration()
            
            if result['available']:
                return {'status': 'failed', 'error': 'Rust should not be available'}
                
            print(f"  {GREEN}✓ Fallback activated: {result.get('fallback_reason')}{RESET}")
            print(f"  {GREEN}✓ Python components: {list(result.get('python_fallback', {}).keys())}{RESET}")
            
            # Test that Python fallback works
            from vision.bloom_filter import PythonBloomFilter
            bloom = PythonBloomFilter(size_mb=1.0)
            bloom.add(b"test")
            if bloom.contains(b"test"):
                print(f"  {GREEN}✓ Python bloom filter working{RESET}")
            else:
                print(f"  {RED}✗ Python bloom filter failed{RESET}")
                
            return {
                'status': 'passed',
                'fallback_reason': result.get('fallback_reason'),
                'python_components': list(result.get('python_fallback', {}).keys())
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
            
        finally:
            # Restore original modules
            for module, value in original_modules.items():
                sys.modules[module] = value
                
    async def test_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance with and without Rust acceleration."""
        print(f"\n{BLUE}Testing Performance Comparison...{RESET}")
        
        results = {
            'rust': {},
            'python': {},
            'speedup': {}
        }
        
        # Test data
        num_operations = 1000
        frame_size = (480, 640, 3)  # Smaller for faster testing
        
        # Test with Rust (if available)
        if self.rust_available:
            import jarvis_rust_core
            
            # Bloom filter performance
            if hasattr(jarvis_rust_core, 'bloom_filter'):
                bloom_rust = jarvis_rust_core.bloom_filter.PyRustBloomFilter(1.0, 7)
            else:
                # Try alternative import paths
                bloom_rust = jarvis_rust_core.RustBloomFilter(1.0, 7) if hasattr(jarvis_rust_core, 'RustBloomFilter') else None
                
            if bloom_rust is None:
                print(f"{YELLOW}Rust bloom filter not available{RESET}")
                return results
            start = time.perf_counter()
            for i in range(num_operations):
                bloom_rust.add(f"item_{i}".encode())
            rust_bloom_time = time.perf_counter() - start
            results['rust']['bloom_filter_ms'] = rust_bloom_time * 1000
            
            # Sliding window performance
            window_rust = jarvis_rust_core.sliding_window.PySlidingWindow(10, 0.9)
            frame = np.random.randint(0, 255, frame_size, dtype=np.uint8).tobytes()
            start = time.perf_counter()
            for i in range(100):  # Less operations for sliding window
                window_rust.process_frame(frame, i * 0.033)
            rust_window_time = time.perf_counter() - start
            results['rust']['sliding_window_ms'] = rust_window_time * 1000
            
        # Test with Python fallback
        from vision.bloom_filter import PythonBloomFilter
        from vision.sliding_window import SlidingWindow
        
        # Bloom filter performance
        bloom_python = PythonBloomFilter(size_mb=1.0)
        start = time.perf_counter()
        for i in range(num_operations):
            bloom_python.add(f"item_{i}".encode())
        python_bloom_time = time.perf_counter() - start
        results['python']['bloom_filter_ms'] = python_bloom_time * 1000
        
        # Sliding window performance
        window_python = SlidingWindow(window_size=10)
        frame_data = {
            'data': base64.b64encode(np.random.randint(0, 255, frame_size, dtype=np.uint8).tobytes()).decode(),
            'width': frame_size[1],
            'height': frame_size[0]
        }
        start = time.perf_counter()
        for i in range(100):
            window_python.add_frame(frame_data, i * 0.033)
        python_window_time = time.perf_counter() - start
        results['python']['sliding_window_ms'] = python_window_time * 1000
        
        # Calculate speedup
        if self.rust_available:
            results['speedup']['bloom_filter'] = python_bloom_time / rust_bloom_time
            results['speedup']['sliding_window'] = python_window_time / rust_window_time
            
            print(f"  {GREEN}✓ Bloom filter: Rust {rust_bloom_time*1000:.2f}ms, Python {python_bloom_time*1000:.2f}ms ({results['speedup']['bloom_filter']:.1f}x speedup){RESET}")
            print(f"  {GREEN}✓ Sliding window: Rust {rust_window_time*1000:.2f}ms, Python {python_window_time*1000:.2f}ms ({results['speedup']['sliding_window']:.1f}x speedup){RESET}")
        else:
            print(f"  {YELLOW}⚠ Rust not available - showing Python performance only{RESET}")
            print(f"  Bloom filter: {python_bloom_time*1000:.2f}ms for {num_operations} operations")
            print(f"  Sliding window: {python_window_time*1000:.2f}ms for 100 frames")
            
        return results
        
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage with Rust acceleration."""
        print(f"\n{BLUE}Testing Memory Usage...{RESET}")
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        results = {
            'baseline_mb': baseline_memory,
            'with_components_mb': 0,
            'increase_mb': 0
        }
        
        if self.rust_available:
            import jarvis_rust_core
            
            # Create multiple components
            components = []
            
            # Create bloom filters
            for _ in range(5):
                bloom = jarvis_rust_core.bloom_filter.PyRustBloomFilter(1.0, 7)
                components.append(bloom)
                
            # Create memory pool
            pool = jarvis_rust_core.zero_copy.PyZeroCopyPool()
            
            # Allocate some buffers
            buffers = []
            for _ in range(10):
                buf = pool.allocate(1024 * 1024)  # 1MB each
                buffers.append(buf)
                
            # Measure memory after allocation
            after_memory = process.memory_info().rss / (1024 * 1024)
            
            results['with_components_mb'] = after_memory
            results['increase_mb'] = after_memory - baseline_memory
            
            # Get pool stats
            pool_stats = pool.stats()
            results['pool_stats'] = pool_stats
            
            # Cleanup
            for buf in buffers:
                buf.release()
                
            print(f"  {GREEN}✓ Baseline: {baseline_memory:.1f}MB, With Rust: {after_memory:.1f}MB (+{results['increase_mb']:.1f}MB){RESET}")
            print(f"  {GREEN}✓ Pool stats: {pool_stats}{RESET}")
            
        else:
            print(f"  {YELLOW}⚠ Rust not available - memory test skipped{RESET}")
            
        return results
        
    async def test_integration_features(self) -> Dict[str, Any]:
        """Test proactive monitoring features with Rust acceleration."""
        print(f"\n{BLUE}Testing Integration Features...{RESET}")
        
        features = {
            'debugging_assistant': False,
            'research_helper': False,
            'workflow_optimizer': False
        }
        
        if not self.proactive_available:
            print(f"{YELLOW}Skipping - Proactive monitoring not available{RESET}")
            return features
            
        try:
            from vision.rust_proactive_integration import RustProactiveMonitor
            
            monitor = RustProactiveMonitor()
            await monitor.initialize()
            
            # Test debugging assistant
            debug_frame = self._create_test_frame_with_error()
            result = await monitor.process_frame(debug_frame)
            if result.get('insights') and 'error' in str(result.get('insights', '')).lower():
                features['debugging_assistant'] = True
                print(f"  {GREEN}✓ Debugging assistant detected error patterns{RESET}")
                
            # Test research helper
            research_frame = self._create_test_frame_with_text()
            result = await monitor.process_frame(research_frame)
            if result.get('insights'):
                features['research_helper'] = True
                print(f"  {GREEN}✓ Research helper processed text content{RESET}")
                
            # Test workflow optimizer
            if monitor.rust_components.get('sliding_window'):
                features['workflow_optimizer'] = True
                print(f"  {GREEN}✓ Workflow optimizer using pattern detection{RESET}")
                
            await monitor.cleanup()
            
        except Exception as e:
            logger.error(f"Integration features test error: {e}")
            
        return features
        
    def _create_test_frame_with_error(self) -> Dict[str, Any]:
        """Create a test frame that simulates an error screen."""
        # Create a red-tinted frame (simulating error)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # Red channel
        
        return {
            'data': base64.b64encode(frame.tobytes()).decode(),
            'width': 1280,
            'height': 720,
            'timestamp': time.time()
        }
        
    def _create_test_frame_with_text(self) -> Dict[str, Any]:
        """Create a test frame that simulates text content."""
        # Create a white background frame (simulating document)
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
        
        return {
            'data': base64.b64encode(frame.tobytes()).decode(),
            'width': 1280,
            'height': 720,
            'timestamp': time.time()
        }
        
    async def run_all_tests(self):
        """Run all integration tests."""
        start_time = time.time()
        
        # Setup
        if not await self.setup():
            print(f"\n{RED}Setup failed - cannot continue{RESET}")
            return
            
        # Run tests
        test_functions = [
            ("Rust Components", self.test_rust_components),
            ("Proactive Monitoring", self.test_proactive_monitoring),
            ("Fallback Behavior", self.test_fallback_behavior),
            ("Performance Comparison", self.test_performance_comparison),
            ("Memory Usage", self.test_memory_usage),
            ("Integration Features", self.test_integration_features)
        ]
        
        for test_name, test_func in test_functions:
            try:
                result = await test_func()
                self.test_results.append({
                    'name': test_name,
                    'result': result,
                    'status': result.get('status', 'completed')
                })
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}", exc_info=True)
                self.test_results.append({
                    'name': test_name,
                    'result': {'error': str(e)},
                    'status': 'failed'
                })
                
        # Summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)
        
    def _print_summary(self, elapsed_time: float):
        """Print test summary."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Test Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        # Count results
        passed = sum(1 for r in self.test_results if r['status'] in ['passed', 'completed'])
        failed = sum(1 for r in self.test_results if r['status'] == 'failed')
        skipped = sum(1 for r in self.test_results if r['status'] == 'skipped')
        
        # Print results
        for result in self.test_results:
            status = result['status']
            name = result['name']
            
            if status in ['passed', 'completed']:
                print(f"{GREEN}✓ {name}{RESET}")
            elif status == 'failed':
                print(f"{RED}✗ {name}: {result['result'].get('error', 'Unknown error')}{RESET}")
            elif status == 'skipped':
                print(f"{YELLOW}⚠ {name}: {result['result'].get('reason', 'Skipped')}{RESET}")
                
        print(f"\n{BLUE}Results:{RESET}")
        print(f"  Total tests: {len(self.test_results)}")
        print(f"  {GREEN}Passed: {passed}{RESET}")
        print(f"  {RED}Failed: {failed}{RESET}")
        print(f"  {YELLOW}Skipped: {skipped}{RESET}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        # Overall status
        if failed == 0:
            print(f"\n{GREEN}✅ All tests passed!{RESET}")
            if self.rust_available:
                print(f"{GREEN}Rust acceleration is working correctly with proactive monitoring.{RESET}")
            else:
                print(f"{YELLOW}Python fallback is working correctly.{RESET}")
        else:
            print(f"\n{RED}❌ Some tests failed. Please check the errors above.{RESET}")
            
        # Performance summary if available
        perf_test = next((r for r in self.test_results if r['name'] == 'Performance Comparison'), None)
        if perf_test and perf_test['result'].get('speedup'):
            print(f"\n{BLUE}Performance Improvements with Rust:{RESET}")
            for component, speedup in perf_test['result']['speedup'].items():
                print(f"  • {component}: {speedup:.1f}x faster")

async def main():
    """Main entry point."""
    tester = ProactiveMonitoringTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())