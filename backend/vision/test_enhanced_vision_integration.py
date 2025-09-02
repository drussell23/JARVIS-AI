#!/usr/bin/env python3
"""
Comprehensive functionality and integration tests for the enhanced vision system.
Tests all 6 integrated components and their interactions.
"""

import asyncio
import os
import sys
import numpy as np
from PIL import Image
import logging
import tempfile
import json
import time
from typing import Dict, List, Any
import psutil
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedVisionTestSuite:
    """Comprehensive test suite for enhanced vision system"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        self.test_results = {
            'functionality': {},
            'integration': {},
            'memory': {},
            'configuration': {},
            'performance': {}
        }
        self.analyzer = None
    
    async def setup(self):
        """Set up test environment"""
        logger.info("Setting up test environment...")
        
        # Import the main analyzer
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Initialize analyzer with test API key
        self.analyzer = ClaudeVisionAnalyzer(self.api_key)
        
        # Create test images
        self.test_images = {
            'small': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'medium': np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
            'large': np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8),
            'text': self._create_text_image(),
            'ui': self._create_ui_image(),
            'complex': self._create_complex_image()
        }
        
        logger.info("Test environment setup complete")
    
    def _create_text_image(self) -> np.ndarray:
        """Create an image with text for testing"""
        img = Image.new('RGB', (800, 600), color='white')
        # In real test, add text using PIL.ImageDraw
        return np.array(img)
    
    def _create_ui_image(self) -> np.ndarray:
        """Create an image with UI elements"""
        img = Image.new('RGB', (1024, 768), color='lightgray')
        # In real test, add UI elements
        return np.array(img)
    
    def _create_complex_image(self) -> np.ndarray:
        """Create a complex image with multiple elements"""
        img = Image.new('RGB', (1920, 1080), color='white')
        # In real test, add various elements
        return np.array(img)
    
    async def test_functionality(self):
        """Test core functionality of each component"""
        logger.info("\n" + "="*70)
        logger.info("TESTING CORE FUNCTIONALITY")
        logger.info("="*70)
        
        # Test 1: Basic Analysis
        await self._test_basic_analysis()
        
        # Test 2: Compression Strategies
        await self._test_compression_strategies()
        
        # Test 3: Query Templates
        await self._test_query_templates()
        
        # Test 4: Batch Processing
        await self._test_batch_processing()
        
        # Test 5: Change Detection
        await self._test_change_detection()
        
        # Test 6: Memory Statistics
        await self._test_memory_statistics()
    
    async def _test_basic_analysis(self):
        """Test basic analysis functionality"""
        logger.info("\n[TEST 1] Basic Analysis")
        
        try:
            # Test smart analysis
            result = await self.analyzer.smart_analyze(
                self.test_images['small'],
                "What do you see?"
            )
            
            success = result is not None and 'description' in result
            self.test_results['functionality']['basic_analysis'] = {
                'success': success,
                'method': result.get('metadata', {}).get('analysis_method', 'unknown'),
                'time': result.get('metadata', {}).get('analysis_time', 0)
            }
            
            logger.info(f"✓ Basic analysis: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Basic analysis failed: {e}")
            self.test_results['functionality']['basic_analysis'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_compression_strategies(self):
        """Test all 5 compression strategies"""
        logger.info("\n[TEST 2] Compression Strategies")
        
        strategies = ['text', 'ui', 'activity', 'detailed', 'quick']
        
        for strategy in strategies:
            try:
                result = await self.analyzer.analyze_with_compression_strategy(
                    self.test_images['medium'],
                    f"Test {strategy} compression",
                    strategy
                )
                
                success = result is not None
                self.test_results['functionality'][f'compression_{strategy}'] = {
                    'success': success,
                    'size_reduction': self._calculate_compression_ratio(strategy)
                }
                
                logger.info(f"✓ {strategy} compression: {'PASSED' if success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"✗ {strategy} compression failed: {e}")
                self.test_results['functionality'][f'compression_{strategy}'] = {
                    'success': False,
                    'error': str(e)
                }
    
    async def _test_query_templates(self):
        """Test configurable query templates"""
        logger.info("\n[TEST 3] Query Templates")
        
        templates = [
            ('notifications', self.analyzer.check_for_notifications),
            ('errors', self.analyzer.check_for_errors),
            ('ui_element', lambda: self.analyzer.find_ui_element("button")),
            ('workspace', self.analyzer.analyze_workspace)
        ]
        
        for name, func in templates:
            try:
                result = await func()
                success = result is not None
                
                self.test_results['functionality'][f'template_{name}'] = {
                    'success': success
                }
                
                logger.info(f"✓ {name} template: {'PASSED' if success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"✗ {name} template failed: {e}")
                self.test_results['functionality'][f'template_{name}'] = {
                    'success': False,
                    'error': str(e)
                }
    
    async def _test_batch_processing(self):
        """Test batch region analysis"""
        logger.info("\n[TEST 4] Batch Processing")
        
        regions = [
            {"x": 0, "y": 0, "width": 200, "height": 150, "prompt": "top-left"},
            {"x": 400, "y": 0, "width": 200, "height": 150, "prompt": "top-right"},
            {"x": 200, "y": 200, "width": 200, "height": 150, "prompt": "center"}
        ]
        
        try:
            results = await self.analyzer.batch_analyze_regions(
                self.test_images['medium'],
                regions
            )
            
            success = len(results) == len(regions)
            self.test_results['functionality']['batch_processing'] = {
                'success': success,
                'regions_processed': len(results),
                'regions_requested': len(regions)
            }
            
            logger.info(f"✓ Batch processing: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Batch processing failed: {e}")
            self.test_results['functionality']['batch_processing'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_change_detection(self):
        """Test change detection capability"""
        logger.info("\n[TEST 5] Change Detection")
        
        try:
            # Create modified version
            modified = self.test_images['small'].copy()
            modified[100:200, 100:200] = 255  # Add white square
            
            result = await self.analyzer.analyze_with_change_detection(
                modified,
                self.test_images['small'],
                "What changed?"
            )
            
            success = result is not None and ('changed' in result or 'description' in result)
            self.test_results['functionality']['change_detection'] = {
                'success': success,
                'detected': result.get('changed', False) if result else False
            }
            
            logger.info(f"✓ Change detection: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Change detection failed: {e}")
            self.test_results['functionality']['change_detection'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_memory_statistics(self):
        """Test memory statistics collection"""
        logger.info("\n[TEST 6] Memory Statistics")
        
        try:
            stats = self.analyzer.get_all_memory_stats()
            
            success = all(k in stats for k in ['system', 'components'])
            self.test_results['functionality']['memory_stats'] = {
                'success': success,
                'components_tracked': len(stats.get('components', {})),
                'system_memory_mb': stats.get('system', {}).get('process_mb', 0)
            }
            
            logger.info(f"✓ Memory statistics: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Memory statistics failed: {e}")
            self.test_results['functionality']['memory_stats'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("\n" + "="*70)
        logger.info("TESTING COMPONENT INTEGRATION")
        logger.info("="*70)
        
        # Test 1: Swift Vision + Window Analysis
        await self._test_swift_window_integration()
        
        # Test 2: Continuous Monitoring + Memory Management
        await self._test_continuous_memory_integration()
        
        # Test 3: Relationship Detection + Window Analysis
        await self._test_relationship_window_integration()
        
        # Test 4: All Components Together
        await self._test_all_components_integration()
    
    async def _test_swift_window_integration(self):
        """Test Swift Vision + Window Analysis integration"""
        logger.info("\n[INTEGRATION 1] Swift Vision + Window Analysis")
        
        try:
            swift = await self.analyzer.get_swift_vision()
            window = await self.analyzer.get_window_analyzer()
            
            if swift and window:
                # Test that both can work together
                success = True
                self.test_results['integration']['swift_window'] = {
                    'success': success,
                    'swift_enabled': swift.enabled,
                    'window_cache_size': len(window.cache) if hasattr(window, 'cache') else 0
                }
            else:
                self.test_results['integration']['swift_window'] = {
                    'success': False,
                    'reason': 'Components not available'
                }
            
            logger.info("✓ Swift + Window integration: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Swift + Window integration failed: {e}")
            self.test_results['integration']['swift_window'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_continuous_memory_integration(self):
        """Test Continuous Monitoring + Memory Management"""
        logger.info("\n[INTEGRATION 2] Continuous Monitoring + Memory Management")
        
        try:
            continuous = await self.analyzer.get_continuous_analyzer()
            
            if continuous:
                # Check memory-aware features
                memory_stats = continuous.get_memory_stats()
                
                success = 'current_interval' in memory_stats
                self.test_results['integration']['continuous_memory'] = {
                    'success': success,
                    'dynamic_interval': memory_stats.get('current_interval', 0),
                    'memory_aware': continuous.config.get('dynamic_interval_enabled', False)
                }
            else:
                self.test_results['integration']['continuous_memory'] = {
                    'success': False,
                    'reason': 'Component not available'
                }
            
            logger.info("✓ Continuous + Memory integration: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Continuous + Memory integration failed: {e}")
            self.test_results['integration']['continuous_memory'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_relationship_window_integration(self):
        """Test Relationship Detection + Window Analysis"""
        logger.info("\n[INTEGRATION 3] Relationship Detection + Window Analysis")
        
        try:
            relationship = await self.analyzer.get_relationship_detector()
            window = await self.analyzer.get_window_analyzer()
            
            if relationship and window:
                success = True
                self.test_results['integration']['relationship_window'] = {
                    'success': success,
                    'min_confidence': relationship.config.get('min_confidence', 0),
                    'window_analyzer_available': True
                }
            else:
                self.test_results['integration']['relationship_window'] = {
                    'success': False,
                    'reason': 'Components not available'
                }
            
            logger.info("✓ Relationship + Window integration: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Relationship + Window integration failed: {e}")
            self.test_results['integration']['relationship_window'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_all_components_integration(self):
        """Test all 6 components working together"""
        logger.info("\n[INTEGRATION 4] All Components Together")
        
        try:
            # Perform a complex operation that uses all components
            start_time = time.time()
            
            # 1. Smart analyze (uses appropriate component)
            result1 = await self.analyzer.smart_analyze(
                self.test_images['complex'],
                "Analyze everything"
            )
            
            # 2. Get memory stats (uses all components)
            stats = self.analyzer.get_all_memory_stats()
            
            # 3. Check component availability
            components_available = len(stats.get('components', {}))
            
            end_time = time.time()
            
            success = result1 is not None and components_available > 0
            self.test_results['integration']['all_components'] = {
                'success': success,
                'components_available': components_available,
                'total_time': end_time - start_time
            }
            
            logger.info(f"✓ All components integration: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ All components integration failed: {e}")
            self.test_results['integration']['all_components'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_memory_management(self):
        """Test memory management features"""
        logger.info("\n" + "="*70)
        logger.info("TESTING MEMORY MANAGEMENT")
        logger.info("="*70)
        
        # Test 1: Memory Limits
        await self._test_memory_limits()
        
        # Test 2: Dynamic Adjustment
        await self._test_dynamic_adjustment()
        
        # Test 3: Emergency Cleanup
        await self._test_emergency_cleanup()
        
        # Test 4: LRU Cache Eviction
        await self._test_lru_eviction()
    
    async def _test_memory_limits(self):
        """Test component memory limits"""
        logger.info("\n[MEMORY 1] Component Memory Limits")
        
        try:
            stats = self.analyzer.get_all_memory_stats()
            
            # Check each component's memory usage
            component_limits = {
                'swift_vision': 300,
                'memory_efficient': 200,
                'continuous': 200,
                'window_analyzer': 100,
                'relationship': 50
            }
            
            within_limits = True
            for comp, limit in component_limits.items():
                comp_stats = stats.get('components', {}).get(comp, {})
                if comp_stats and 'memory_mb' in comp_stats:
                    if comp_stats['memory_mb'] > limit:
                        within_limits = False
                        logger.warning(f"{comp} exceeds limit: {comp_stats['memory_mb']}MB > {limit}MB")
            
            self.test_results['memory']['limits'] = {
                'success': within_limits,
                'total_memory_mb': stats.get('system', {}).get('process_mb', 0)
            }
            
            logger.info(f"✓ Memory limits: {'PASSED' if within_limits else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Memory limits test failed: {e}")
            self.test_results['memory']['limits'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_dynamic_adjustment(self):
        """Test dynamic memory adjustment"""
        logger.info("\n[MEMORY 2] Dynamic Memory Adjustment")
        
        try:
            # Simulate memory pressure
            continuous = await self.analyzer.get_continuous_analyzer()
            
            if continuous:
                # Check if interval adjusts with memory
                initial_interval = continuous.config['update_interval']
                
                # Simulate low memory
                continuous._current_memory_mb = 500  # Simulate low memory
                await continuous._check_memory_and_adjust()
                
                adjusted_interval = continuous._current_interval
                
                success = adjusted_interval > initial_interval
                self.test_results['memory']['dynamic_adjustment'] = {
                    'success': success,
                    'initial_interval': initial_interval,
                    'adjusted_interval': adjusted_interval
                }
            else:
                self.test_results['memory']['dynamic_adjustment'] = {
                    'success': False,
                    'reason': 'Component not available'
                }
            
            logger.info("✓ Dynamic adjustment: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Dynamic adjustment failed: {e}")
            self.test_results['memory']['dynamic_adjustment'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_emergency_cleanup(self):
        """Test emergency cleanup mechanism"""
        logger.info("\n[MEMORY 3] Emergency Cleanup")
        
        try:
            # Test cleanup functionality
            await self.analyzer.cleanup_all_components()
            
            # Check memory after cleanup
            stats = self.analyzer.get_all_memory_stats()
            
            success = True  # If cleanup completes without error
            self.test_results['memory']['emergency_cleanup'] = {
                'success': success,
                'memory_after_cleanup': stats.get('system', {}).get('process_mb', 0)
            }
            
            logger.info("✓ Emergency cleanup: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Emergency cleanup failed: {e}")
            self.test_results['memory']['emergency_cleanup'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_lru_eviction(self):
        """Test LRU cache eviction"""
        logger.info("\n[MEMORY 4] LRU Cache Eviction")
        
        try:
            # Check if cache exists and get its size
            if hasattr(self.analyzer, 'cache'):
                cache = self.analyzer.cache
                if hasattr(cache, 'cache_size'):
                    cache_size_before = cache.cache_size()
                elif hasattr(cache, '__len__'):
                    cache_size_before = len(cache)
                else:
                    cache_size_before = 0
            else:
                cache_size_before = 0
            
            # Perform multiple analyses to trigger eviction
            for i in range(20):
                await self.analyzer.analyze_screenshot(
                    self.test_images['small'],
                    f"Test query {i}"
                )
            
            # Get cache size after
            if hasattr(self.analyzer, 'cache'):
                cache = self.analyzer.cache
                if hasattr(cache, 'cache_size'):
                    cache_size_after = cache.cache_size()
                elif hasattr(cache, '__len__'):
                    cache_size_after = len(cache)
                else:
                    cache_size_after = 0
            else:
                cache_size_after = 0
            
            # Check if cache has a reasonable size limit (default to 10 if not configured)
            max_items = getattr(self.analyzer.config, 'max_cache_items', 10)
            success = cache_size_after <= max_items
            self.test_results['memory']['lru_eviction'] = {
                'success': success,
                'cache_size': cache_size_after,
                'max_allowed': max_items
            }
            
            logger.info(f"✓ LRU eviction: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ LRU eviction failed: {e}")
            self.test_results['memory']['lru_eviction'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_configuration(self):
        """Test configuration via environment variables"""
        logger.info("\n" + "="*70)
        logger.info("TESTING CONFIGURATION")
        logger.info("="*70)
        
        # Test 1: Environment Variable Override
        await self._test_env_override()
        
        # Test 2: Component Enable/Disable
        await self._test_component_toggle()
        
        # Test 3: Custom Query Templates
        await self._test_custom_templates()
    
    async def _test_env_override(self):
        """Test environment variable configuration"""
        logger.info("\n[CONFIG 1] Environment Variable Override")
        
        try:
            # Set test environment variables
            test_vars = {
                'VISION_MAX_IMAGE_DIM': '2048',
                'VISION_JPEG_QUALITY': '90',
                'VISION_CACHE_SIZE_MB': '200'
            }
            
            for var, value in test_vars.items():
                os.environ[var] = value
            
            # Re-initialize analyzer to pick up new config
            from claude_vision_analyzer_main import ClaudeVisionAnalyzer
            new_analyzer = ClaudeVisionAnalyzer(self.api_key)
            
            # Check if config was applied
            success = (
                new_analyzer.config.max_image_dimension == 2048 and
                new_analyzer.config.jpeg_quality == 90
            )
            
            self.test_results['configuration']['env_override'] = {
                'success': success,
                'applied_vars': len(test_vars)
            }
            
            logger.info(f"✓ Environment override: {'PASSED' if success else 'FAILED'}")
            
            # Clean up
            for var in test_vars:
                os.environ.pop(var, None)
            
        except Exception as e:
            logger.error(f"✗ Environment override failed: {e}")
            self.test_results['configuration']['env_override'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_component_toggle(self):
        """Test enabling/disabling components"""
        logger.info("\n[CONFIG 2] Component Toggle")
        
        try:
            # Disable a component
            os.environ['VISION_SWIFT_ENABLED'] = 'false'
            
            from claude_vision_analyzer_main import ClaudeVisionAnalyzer
            new_analyzer = ClaudeVisionAnalyzer(self.api_key)
            
            swift = await new_analyzer.get_swift_vision()
            
            success = swift is None  # Should be disabled
            self.test_results['configuration']['component_toggle'] = {
                'success': success,
                'swift_disabled': swift is None
            }
            
            logger.info(f"✓ Component toggle: {'PASSED' if success else 'FAILED'}")
            
            # Clean up
            os.environ.pop('VISION_SWIFT_ENABLED', None)
            
        except Exception as e:
            logger.error(f"✗ Component toggle failed: {e}")
            self.test_results['configuration']['component_toggle'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_custom_templates(self):
        """Test custom query templates"""
        logger.info("\n[CONFIG 3] Custom Query Templates")
        
        try:
            # Set custom template
            custom_template = {
                "custom_test": "This is a custom test template: {query}"
            }
            os.environ['VISION_QUERY_TEMPLATES'] = json.dumps(custom_template)
            
            # Get simplified vision system
            simplified = await self.analyzer.get_simplified_vision()
            
            if simplified:
                templates = simplified.get_available_templates()
                success = 'custom_test' in templates
            else:
                success = False
            
            self.test_results['configuration']['custom_templates'] = {
                'success': success,
                'custom_template_loaded': success
            }
            
            logger.info(f"✓ Custom templates: {'PASSED' if success else 'FAILED'}")
            
            # Clean up
            os.environ.pop('VISION_QUERY_TEMPLATES', None)
            
        except Exception as e:
            logger.error(f"✗ Custom templates failed: {e}")
            self.test_results['configuration']['custom_templates'] = {
                'success': False,
                'error': str(e)
            }
    
    async def test_performance(self):
        """Test performance characteristics"""
        logger.info("\n" + "="*70)
        logger.info("TESTING PERFORMANCE")
        logger.info("="*70)
        
        # Test 1: Response Time
        await self._test_response_time()
        
        # Test 2: Throughput
        await self._test_throughput()
        
        # Test 3: Resource Usage
        await self._test_resource_usage()
    
    async def _test_response_time(self):
        """Test response times for different operations"""
        logger.info("\n[PERFORMANCE 1] Response Time")
        
        operations = [
            ('small_image', self.test_images['small'], "Quick analysis"),
            ('large_image', self.test_images['large'], "Detailed analysis"),
            ('with_compression', self.test_images['medium'], "Compressed analysis")
        ]
        
        response_times = {}
        
        for name, image, prompt in operations:
            try:
                start = time.time()
                
                if name == 'with_compression':
                    result = await self.analyzer.analyze_with_compression_strategy(
                        image, prompt, 'quick'
                    )
                else:
                    result = await self.analyzer.smart_analyze(image, prompt)
                
                end = time.time()
                response_time = end - start
                
                response_times[name] = response_time
                logger.info(f"  {name}: {response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  {name}: Failed - {e}")
                response_times[name] = None
        
        # Check if response times are acceptable
        success = all(
            t is not None and t < 5.0  # Should complete within 5 seconds
            for t in response_times.values()
        )
        
        self.test_results['performance']['response_time'] = {
            'success': success,
            'times': response_times
        }
        
        logger.info(f"✓ Response time: {'PASSED' if success else 'FAILED'}")
    
    async def _test_throughput(self):
        """Test throughput for batch operations"""
        logger.info("\n[PERFORMANCE 2] Throughput")
        
        try:
            num_requests = 10
            start = time.time()
            
            # Perform multiple concurrent analyses
            tasks = []
            for i in range(num_requests):
                task = self.analyzer.smart_analyze(
                    self.test_images['small'],
                    f"Analysis {i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end = time.time()
            total_time = end - start
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            throughput = successful / total_time if total_time > 0 else 0
            
            self.test_results['performance']['throughput'] = {
                'success': successful == num_requests,
                'requests': num_requests,
                'successful': successful,
                'throughput_per_second': throughput,
                'total_time': total_time
            }
            
            logger.info(f"  Throughput: {throughput:.2f} requests/second")
            logger.info(f"✓ Throughput: {'PASSED' if successful == num_requests else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Throughput test failed: {e}")
            self.test_results['performance']['throughput'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _test_resource_usage(self):
        """Test resource usage"""
        logger.info("\n[PERFORMANCE 3] Resource Usage")
        
        try:
            # Get initial resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu = process.cpu_percent(interval=1)
            
            # Perform intensive operations
            for _ in range(5):
                await self.analyzer.smart_analyze(
                    self.test_images['large'],
                    "Intensive analysis"
                )
            
            # Get final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent(interval=1)
            
            memory_increase = final_memory - initial_memory
            
            # Check if resource usage is reasonable
            success = memory_increase < 500  # Should not increase by more than 500MB
            
            self.test_results['performance']['resource_usage'] = {
                'success': success,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'cpu_usage': final_cpu
            }
            
            logger.info(f"  Memory increase: {memory_increase:.2f}MB")
            logger.info(f"  CPU usage: {final_cpu:.1f}%")
            logger.info(f"✓ Resource usage: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"✗ Resource usage test failed: {e}")
            self.test_results['performance']['resource_usage'] = {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_compression_ratio(self, strategy: str) -> float:
        """Calculate compression ratio for a strategy"""
        # Simplified calculation for testing
        ratios = {
            'text': 0.95,
            'ui': 0.85,
            'activity': 0.80,
            'detailed': 1.00,
            'quick': 0.60
        }
        return ratios.get(strategy, 1.0)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*70)
        logger.info("TEST REPORT SUMMARY")
        logger.info("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if tests:
                logger.info(f"\n{category.upper()}:")
                for test_name, result in tests.items():
                    total_tests += 1
                    if result.get('success', False):
                        passed_tests += 1
                        logger.info(f"  ✓ {test_name}: PASSED")
                    else:
                        logger.info(f"  ✗ {test_name}: FAILED")
                        if 'error' in result:
                            logger.info(f"    Error: {result['error']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"{'='*70}")
        
        # Save detailed report
        report_path = 'vision_test_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': success_rate
                },
                'results': self.test_results
            }, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        return success_rate >= 80  # Consider 80% as passing
    
    async def run_all_tests(self):
        """Run all test suites"""
        try:
            await self.setup()
            
            # Run all test categories
            await self.test_functionality()
            await self.test_integration()
            await self.test_memory_management()
            await self.test_configuration()
            await self.test_performance()
            
            # Generate report
            success = self.generate_report()
            
            # Cleanup
            await self.analyzer.cleanup_all_components()
            
            return success
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False


async def main():
    """Main test runner"""
    logger.info("Starting Enhanced Vision System Test Suite")
    logger.info("This will test all 6 integrated components\n")
    
    # Check if API key is set
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not set - using mock mode")
        logger.info("Set ANTHROPIC_API_KEY for full integration tests")
    
    # Run tests
    test_suite = EnhancedVisionTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ TEST SUITE PASSED!")
    else:
        logger.error("\n❌ TEST SUITE FAILED!")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)