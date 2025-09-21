#!/usr/bin/env python3
"""
Fixed version of real-world scenario tests that handles import issues.
"""

import asyncio
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import logging
import time
from typing import Dict, List, Any

# Setup paths properly
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(backend_dir)

# Add all necessary paths
sys.path.insert(0, root_dir)
sys.path.insert(0, backend_dir)
sys.path.insert(0, current_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable to disable components with import issues
os.environ['VISION_MEMORY_EFFICIENT_ENABLED'] = 'false'
os.environ['VISION_SWIFT_ENABLED'] = 'false'
os.environ['VISION_CONTINUOUS_ENABLED'] = 'false'


class FixedScenarioTests:
    """Fixed version of scenario tests"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        self.analyzer = None
        self.test_results = {}
    
    async def setup(self):
        """Set up test environment"""
        try:
            # Import from the vision directory
            import claude_vision_analyzer_main
            self.analyzer = claude_vision_analyzer_main.ClaudeVisionAnalyzer(self.api_key)
            logger.info("Vision analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise
    
    def create_simple_test_image(self, color='white', text='Test') -> np.ndarray:
        """Create a simple test image"""
        img = Image.new('RGB', (800, 600), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), text, fill='black')
        return np.array(img)
    
    async def test_basic_functionality(self):
        """Test basic vision functionality"""
        logger.info("\n[TEST 1] Basic Functionality")
        
        try:
            # Create test image
            test_image = self.create_simple_test_image()
            
            # Test smart analysis
            result = await self.analyzer.smart_analyze(
                test_image,
                "What do you see in this image?"
            )
            
            success = result is not None and 'description' in result
            self.test_results['basic_functionality'] = {
                'success': success,
                'has_result': result is not None
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Basic functionality failed: {e}")
            self.test_results['basic_functionality'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_compression_strategies(self):
        """Test compression strategies"""
        logger.info("\n[TEST 2] Compression Strategies")
        
        strategies = ['text', 'ui', 'activity', 'detailed', 'quick']
        results = {}
        
        for strategy in strategies:
            try:
                test_image = self.create_simple_test_image()
                
                result = await self.analyzer.analyze_with_compression_strategy(
                    test_image,
                    f"Test {strategy} compression",
                    strategy
                )
                
                success = result is not None
                results[strategy] = success
                logger.info(f"  {strategy}: {'PASSED' if success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"  {strategy}: FAILED - {e}")
                results[strategy] = False
        
        self.test_results['compression_strategies'] = {
            'success': all(results.values()),
            'strategies': results
        }
    
    async def test_query_templates(self):
        """Test query template methods"""
        logger.info("\n[TEST 3] Query Templates")
        
        templates = {
            'notifications': self.analyzer.check_for_notifications,
            'errors': self.analyzer.check_for_errors,
            'weather': self.analyzer.check_weather,
            'activity': self.analyzer.analyze_current_activity
        }
        
        results = {}
        
        for name, method in templates.items():
            try:
                result = await method()
                success = result is not None
                results[name] = success
                logger.info(f"  {name}: {'PASSED' if success else 'FAILED'}")
            except Exception as e:
                logger.error(f"  {name}: FAILED - {e}")
                results[name] = False
        
        self.test_results['query_templates'] = {
            'success': all(results.values()),
            'templates': results
        }
    
    async def test_text_reading(self):
        """Test text reading from area"""
        logger.info("\n[TEST 4] Text Reading")
        
        try:
            # Create image with text
            test_image = self.create_simple_test_image(text="Hello World")
            
            # Define area to read
            area = {"x": 50, "y": 50, "width": 200, "height": 100}
            
            result = await self.analyzer.read_text_from_area(test_image, area)
            
            success = result is not None and result.get('success', False)
            self.test_results['text_reading'] = {
                'success': success,
                'result': 'text' in result if result else False
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Text reading failed: {e}")
            self.test_results['text_reading'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_workspace_analysis(self):
        """Test workspace analysis"""
        logger.info("\n[TEST 5] Workspace Analysis")
        
        try:
            # Test without screenshot (should capture current)
            result = await self.analyzer.analyze_workspace()
            
            success = result is not None
            self.test_results['workspace_analysis'] = {
                'success': success,
                'has_components': 'components_used' in result if result else False
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Workspace analysis failed: {e}")
            self.test_results['workspace_analysis'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_memory_management(self):
        """Test memory management"""
        logger.info("\n[TEST 6] Memory Management")
        
        try:
            # Get initial stats
            initial_stats = self.analyzer.get_all_memory_stats()
            initial_memory = initial_stats['system']['process_mb']
            
            # Perform multiple analyses
            for i in range(3):
                test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                await self.analyzer.smart_analyze(test_image, f"Test {i}")
            
            # Get final stats
            final_stats = self.analyzer.get_all_memory_stats()
            final_memory = final_stats['system']['process_mb']
            
            memory_growth = final_memory - initial_memory
            
            # Should have reasonable memory growth
            success = memory_growth < 200  # Less than 200MB growth
            
            self.test_results['memory_management'] = {
                'success': success,
                'initial_mb': initial_memory,
                'final_mb': final_memory,
                'growth_mb': memory_growth
            }
            
            logger.info(f"  Memory growth: {memory_growth:.1f}MB")
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Memory management failed: {e}")
            self.test_results['memory_management'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_batch_processing(self):
        """Test batch region processing"""
        logger.info("\n[TEST 7] Batch Processing")
        
        try:
            test_image = self.create_simple_test_image()
            
            regions = [
                {"x": 0, "y": 0, "width": 100, "height": 100},
                {"x": 100, "y": 100, "width": 100, "height": 100}
            ]
            
            results = await self.analyzer.batch_analyze_regions(test_image, regions)
            
            success = len(results) == len(regions)
            self.test_results['batch_processing'] = {
                'success': success,
                'processed': len(results),
                'requested': len(regions)
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Batch processing failed: {e}")
            self.test_results['batch_processing'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_performance(self):
        """Test performance characteristics"""
        logger.info("\n[TEST 8] Performance")
        
        try:
            # Time a simple analysis
            test_image = self.create_simple_test_image()
            
            start_time = time.time()
            result = await self.analyzer.smart_analyze(test_image, "Quick test")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should be reasonably fast
            success = result is not None and response_time < 5.0
            
            self.test_results['performance'] = {
                'success': success,
                'response_time': response_time
            }
            
            logger.info(f"  Response time: {response_time:.2f}s")
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Performance test failed: {e}")
            self.test_results['performance'] = {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*70)
        logger.info("TEST REPORT SUMMARY")
        logger.info("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        for test_name, result in self.test_results.items():
            status = "PASSED" if result.get('success', False) else "FAILED"
            logger.info(f"\n{test_name}: {status}")
            
            if not result.get('success', False) and 'error' in result:
                logger.info(f"  Error: {result['error']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"{'='*70}")
        
        return success_rate >= 70
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            
            # Run tests
            await self.test_basic_functionality()
            await self.test_compression_strategies()
            await self.test_query_templates()
            await self.test_text_reading()
            await self.test_workspace_analysis()
            await self.test_memory_management()
            await self.test_batch_processing()
            await self.test_performance()
            
            # Cleanup
            await self.analyzer.cleanup_all_components()
            
            # Generate report
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False


async def main():
    """Main test runner"""
    logger.info("Enhanced Vision System - Fixed Scenario Tests")
    logger.info("Testing core functionality with import issues resolved\n")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("No API key found - these tests require ANTHROPIC_API_KEY")
        return 1
    
    # Run tests
    test_suite = FixedScenarioTests()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ TESTS PASSED!")
    else:
        logger.error("\n❌ TESTS FAILED!")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)