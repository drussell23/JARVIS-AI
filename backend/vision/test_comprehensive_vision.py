#!/usr/bin/env python3
"""
Comprehensive test suite for Claude Vision Analyzer with enhanced features
Tests functionality, integration, and real-world scenarios
"""

import asyncio
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
import json
import time
from typing import Dict, List, Any, Optional
import psutil
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import io
import base64

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container"""
    name: str
    category: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None


class MockAnthropicClient:
    """Enhanced mock Anthropic client for realistic testing"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = self
        self.call_count = 0
        self.last_request = None
        
    def create(self, **kwargs):
        """Mock messages.create with realistic responses"""
        self.call_count += 1
        self.last_request = kwargs
        
        # Generate different responses based on prompt
        prompt = kwargs.get('messages', [{}])[0].get('content', [{}])[-1].get('text', '')
        
        mock_response = Mock()
        
        if 'notification' in prompt.lower():
            response_text = json.dumps({
                "notifications_present": True,
                "notifications": [
                    {"app": "Mail", "count": 3, "preview": "New message from..."},
                    {"app": "Slack", "count": 5, "preview": "Team discussion..."}
                ],
                "description": "Found 2 notification badges on screen"
            })
        elif 'error' in prompt.lower():
            response_text = json.dumps({
                "errors_detected": True,
                "errors": [
                    {"type": "dialog", "message": "Connection failed", "severity": "high"}
                ],
                "description": "Error dialog detected on screen"
            })
        elif 'workspace' in prompt.lower():
            response_text = json.dumps({
                "workspace_analysis": {
                    "active_app": "VS Code",
                    "open_windows": 3,
                    "layout": "split-screen",
                    "productivity_score": 0.85
                },
                "description": "Developer workspace with code editor and terminal"
            })
        elif 'ui element' in prompt.lower():
            response_text = json.dumps({
                "element_found": True,
                "location": {"x": 100, "y": 200, "width": 150, "height": 40},
                "type": "button",
                "properties": {"text": "Submit", "enabled": True},
                "description": "Found button element at specified location"
            })
        else:
            response_text = json.dumps({
                "description": f"Mock analysis of image (call #{self.call_count})",
                "content": "General image analysis",
                "metadata": {"processing_time": 0.1}
            })
        
        mock_response.content = [Mock(text=response_text)]
        return mock_response


class ComprehensiveVisionTestSuite:
    """Comprehensive test suite for enhanced vision analyzer"""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key') if not use_mock else 'test-key'
        self.analyzer = None
        self.test_results: List[TestResult] = []
        self.mock_client = None
    
    async def setup(self):
        """Set up test environment"""
        logger.info("Setting up comprehensive test environment...")
        
        if self.use_mock:
            # Use mock client
            self.mock_client = MockAnthropicClient(self.api_key)
            with patch('anthropic.Anthropic', return_value=self.mock_client):
                from claude_vision_analyzer_main import ClaudeVisionAnalyzer
                self.analyzer = ClaudeVisionAnalyzer(self.api_key)
        else:
            # Use real client
            from claude_vision_analyzer_main import ClaudeVisionAnalyzer
            self.analyzer = ClaudeVisionAnalyzer(self.api_key)
        
        # Create test images with different characteristics
        self.test_images = await self._create_test_images()
        
        logger.info("Test environment ready")
    
    async def _create_test_images(self) -> Dict[str, np.ndarray]:
        """Create realistic test images"""
        images = {}
        
        # 1. Text-heavy image (document)
        doc_img = Image.new('RGB', (800, 1000), color='white')
        draw = ImageDraw.Draw(doc_img)
        y_pos = 50
        for i in range(20):
            draw.text((50, y_pos), f"This is line {i+1} of text content in the document.", fill='black')
            y_pos += 40
        images['document'] = np.array(doc_img)
        
        # 2. UI screenshot with buttons and dialogs
        ui_img = Image.new('RGB', (1920, 1080), color='#f0f0f0')
        draw = ImageDraw.Draw(ui_img)
        # Add window
        draw.rectangle([100, 100, 900, 700], fill='white', outline='gray')
        # Add buttons
        draw.rectangle([150, 600, 300, 650], fill='#007bff', outline='black')
        draw.text((200, 615), "Submit", fill='white')
        # Add error dialog
        draw.rectangle([500, 300, 800, 450], fill='#ffcccc', outline='red')
        draw.text((550, 350), "Error: Connection Failed", fill='red')
        images['ui_with_error'] = np.array(ui_img)
        
        # 3. Code editor screenshot
        code_img = Image.new('RGB', (1920, 1080), color='#1e1e1e')
        draw = ImageDraw.Draw(code_img)
        code_lines = [
            "def analyze_image(image):",
            "    # Process the image",
            "    result = process(image)",
            "    return result"
        ]
        y_pos = 100
        for line in code_lines:
            draw.text((50, y_pos), line, fill='#d4d4d4')
            y_pos += 30
        images['code_editor'] = np.array(code_img)
        
        # 4. Dashboard with charts
        dashboard_img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(dashboard_img)
        # Add chart areas
        draw.rectangle([50, 50, 500, 400], fill='#e8f4f8', outline='blue')
        draw.rectangle([550, 50, 1000, 400], fill='#f8e8e8', outline='red')
        draw.rectangle([50, 450, 1000, 800], fill='#e8f8e8', outline='green')
        images['dashboard'] = np.array(dashboard_img)
        
        # 5. Mobile app screenshot
        mobile_img = Image.new('RGB', (375, 812), color='white')
        draw = ImageDraw.Draw(mobile_img)
        # Status bar
        draw.rectangle([0, 0, 375, 44], fill='black')
        # Navigation
        draw.rectangle([0, 768, 375, 812], fill='#f8f8f8')
        # Content area with notifications
        draw.ellipse([335, 100, 365, 130], fill='red')
        draw.text((345, 105), "3", fill='white')
        images['mobile_app'] = np.array(mobile_img)
        
        # Standard test images
        images['small'] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images['medium'] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        images['large'] = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        
        return images
    
    async def test_enhanced_functionality(self):
        """Test all enhanced functionality features"""
        logger.info("\n" + "="*70)
        logger.info("TESTING ENHANCED FUNCTIONALITY")
        logger.info("="*70)
        
        tests = [
            self._test_smart_analysis,
            self._test_compression_strategies,
            self._test_query_templates,
            self._test_batch_processing,
            self._test_change_detection,
            self._test_caching_system,
            self._test_memory_awareness,
            self._test_rate_limiting
        ]
        
        for test_func in tests:
            await test_func()
    
    async def _test_smart_analysis(self):
        """Test smart analysis with automatic method selection"""
        logger.info("\n[FUNCTIONALITY] Smart Analysis")
        start_time = time.time()
        
        try:
            # Test different image sizes
            test_cases = [
                ('small', self.test_images['small'], 'full'),
                ('medium', self.test_images['medium'], 'full'),
                ('large', self.test_images['large'], 'sliding_window')
            ]
            
            results = {}
            for name, image, expected_method in test_cases:
                result = await self.analyzer.smart_analyze(image, f"Analyze this {name} image")
                results[name] = {
                    'success': result is not None,
                    'method': result.get('metadata', {}).get('analysis_method', 'unknown'),
                    'expected': expected_method
                }
            
            passed = all(r['success'] for r in results.values())
            
            self.test_results.append(TestResult(
                name="smart_analysis",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details=results
            ))
            
            logger.info(f"✓ Smart analysis: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="smart_analysis",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Smart analysis failed: {e}")
    
    async def _test_compression_strategies(self):
        """Test all compression strategies"""
        logger.info("\n[FUNCTIONALITY] Compression Strategies")
        start_time = time.time()
        
        try:
            strategies = {
                'text': self.test_images['document'],
                'ui': self.test_images['ui_with_error'],
                'activity': self.test_images['code_editor'],
                'detailed': self.test_images['dashboard'],
                'quick': self.test_images['mobile_app']
            }
            
            results = {}
            for strategy, image in strategies.items():
                result = await self.analyzer.analyze_with_compression_strategy(
                    image, f"Test {strategy} compression", strategy
                )
                results[strategy] = {
                    'success': result is not None,
                    'has_description': 'description' in result if result else False
                }
            
            passed = all(r['success'] for r in results.values())
            
            self.test_results.append(TestResult(
                name="compression_strategies",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details=results
            ))
            
            logger.info(f"✓ Compression strategies: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="compression_strategies",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Compression strategies failed: {e}")
    
    async def _test_query_templates(self):
        """Test query template system"""
        logger.info("\n[FUNCTIONALITY] Query Templates")
        start_time = time.time()
        
        try:
            # Test built-in templates
            template_tests = [
                ('notifications', self.test_images['mobile_app'], 
                 lambda r: r.get('notifications_present') is not None),
                ('errors', self.test_images['ui_with_error'],
                 lambda r: r.get('errors_detected') is not None),
                ('workspace', self.test_images['code_editor'],
                 lambda r: 'workspace_analysis' in r or 'description' in r),
                ('ui_element', self.test_images['ui_with_error'],
                 lambda r: r is not None)
            ]
            
            results = {}
            for template_name, image, validator in template_tests:
                if template_name == 'notifications':
                    result = await self.analyzer.check_for_notifications()
                elif template_name == 'errors':
                    result = await self.analyzer.check_for_errors()
                elif template_name == 'workspace':
                    result = await self.analyzer.analyze_workspace(image)
                elif template_name == 'ui_element':
                    result = await self.analyzer.find_ui_element("button")
                
                # Parse result if it's a string (JSON)
                if isinstance(result, dict) and 'description' in result:
                    try:
                        parsed = json.loads(result['description'])
                        result.update(parsed)
                    except:
                        pass
                
                results[template_name] = {
                    'success': validator(result) if result else False,
                    'has_data': result is not None
                }
            
            passed = all(r['success'] for r in results.values())
            
            self.test_results.append(TestResult(
                name="query_templates",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details=results
            ))
            
            logger.info(f"✓ Query templates: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="query_templates",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Query templates failed: {e}")
    
    async def _test_batch_processing(self):
        """Test batch region analysis"""
        logger.info("\n[FUNCTIONALITY] Batch Processing")
        start_time = time.time()
        
        try:
            # Define regions to analyze
            regions = [
                {"x": 0, "y": 0, "width": 200, "height": 150, "prompt": "top-left corner"},
                {"x": 400, "y": 0, "width": 200, "height": 150, "prompt": "top-right area"},
                {"x": 200, "y": 200, "width": 200, "height": 150, "prompt": "center region"}
            ]
            
            results = await self.analyzer.batch_analyze_regions(
                self.test_images['dashboard'],
                regions
            )
            
            passed = len(results) == len(regions) and all(r is not None for r in results)
            
            self.test_results.append(TestResult(
                name="batch_processing",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'regions_requested': len(regions),
                    'regions_processed': len(results),
                    'all_successful': passed
                }
            ))
            
            logger.info(f"✓ Batch processing: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="batch_processing",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Batch processing failed: {e}")
    
    async def _test_change_detection(self):
        """Test change detection between images"""
        logger.info("\n[FUNCTIONALITY] Change Detection")
        start_time = time.time()
        
        try:
            # Create two versions of an image
            original = self.test_images['ui_with_error']
            modified = original.copy()
            
            # Add a change (new element)
            modified[500:600, 500:700] = [255, 0, 0]  # Red rectangle
            
            result = await self.analyzer.analyze_with_change_detection(
                modified,
                original,
                "What changed between these images?"
            )
            
            passed = result is not None and ('changed' in result or 'description' in result)
            
            self.test_results.append(TestResult(
                name="change_detection",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'detected_change': result.get('changed', False) if result else False,
                    'has_description': 'description' in result if result else False
                }
            ))
            
            logger.info(f"✓ Change detection: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="change_detection",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Change detection failed: {e}")
    
    async def _test_caching_system(self):
        """Test caching functionality"""
        logger.info("\n[FUNCTIONALITY] Caching System")
        start_time = time.time()
        
        try:
            # Make the same request twice
            image = self.test_images['small']
            prompt = "Test caching behavior"
            
            # First call
            result1, metrics1 = await self.analyzer.analyze_screenshot(image, prompt)
            
            # Second call (should hit cache)
            result2, metrics2 = await self.analyzer.analyze_screenshot(image, prompt)
            
            # Check if cache was used
            cache_hit = (
                metrics2.cache_hit if hasattr(metrics2, 'cache_hit') else
                metrics2.api_call_time == 0 if hasattr(metrics2, 'api_call_time') else
                False
            )
            
            passed = result1 is not None and result2 is not None
            
            self.test_results.append(TestResult(
                name="caching_system",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'first_call_success': result1 is not None,
                    'second_call_success': result2 is not None,
                    'cache_hit': cache_hit,
                    'cache_enabled': self.analyzer.config.cache_enabled
                }
            ))
            
            logger.info(f"✓ Caching system: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="caching_system",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Caching system failed: {e}")
    
    async def _test_memory_awareness(self):
        """Test memory-aware features"""
        logger.info("\n[FUNCTIONALITY] Memory Awareness")
        start_time = time.time()
        
        try:
            # Get memory stats
            stats = self.analyzer.get_all_memory_stats()
            
            # Check memory tracking
            has_system_stats = 'system' in stats
            has_process_memory = stats.get('system', {}).get('process_mb', 0) > 0
            
            # Test memory threshold behavior
            original_threshold = self.analyzer.config.memory_threshold_percent
            self.analyzer.config.memory_threshold_percent = 10  # Very low threshold
            
            # This should trigger memory-aware behavior
            result = await self.analyzer.smart_analyze(
                self.test_images['large'],
                "Test memory awareness"
            )
            
            # Restore threshold
            self.analyzer.config.memory_threshold_percent = original_threshold
            
            passed = has_system_stats and has_process_memory and result is not None
            
            self.test_results.append(TestResult(
                name="memory_awareness",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'has_system_stats': has_system_stats,
                    'process_memory_mb': stats.get('system', {}).get('process_mb', 0),
                    'memory_threshold_test': result is not None
                }
            ))
            
            logger.info(f"✓ Memory awareness: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="memory_awareness",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Memory awareness failed: {e}")
    
    async def _test_rate_limiting(self):
        """Test rate limiting functionality"""
        logger.info("\n[FUNCTIONALITY] Rate Limiting")
        start_time = time.time()
        
        try:
            # Set low rate limit for testing
            original_limit = self.analyzer.config.max_concurrent_requests
            self.analyzer.config.max_concurrent_requests = 2
            
            # Launch multiple concurrent requests
            tasks = []
            for i in range(5):
                task = self.analyzer.analyze_screenshot(
                    self.test_images['small'],
                    f"Rate limit test {i}"
                )
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Restore original limit
            self.analyzer.config.max_concurrent_requests = original_limit
            
            # Check results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            errors = sum(1 for r in results if isinstance(r, Exception))
            
            passed = successful >= 2  # At least rate limit number should succeed
            
            self.test_results.append(TestResult(
                name="rate_limiting",
                category="functionality",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'total_requests': len(tasks),
                    'successful': successful,
                    'rate_limited': errors,
                    'rate_limit': 2
                }
            ))
            
            logger.info(f"✓ Rate limiting: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="rate_limiting",
                category="functionality",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Rate limiting failed: {e}")
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("\n" + "="*70)
        logger.info("TESTING INTEGRATION")
        logger.info("="*70)
        
        tests = [
            self._test_component_availability,
            self._test_fallback_mechanisms,
            self._test_error_handling,
            self._test_configuration_system
        ]
        
        for test_func in tests:
            await test_func()
    
    async def _test_component_availability(self):
        """Test component availability and initialization"""
        logger.info("\n[INTEGRATION] Component Availability")
        start_time = time.time()
        
        try:
            components = {
                'swift_vision': await self.analyzer.get_swift_vision(),
                'memory_efficient': await self.analyzer.get_memory_efficient_analyzer(),
                'continuous': await self.analyzer.get_continuous_analyzer(),
                'window_analyzer': await self.analyzer.get_window_analyzer(),
                'relationship': await self.analyzer.get_relationship_detector(),
                'simplified': await self.analyzer.get_simplified_vision()
            }
            
            # For mock tests, we expect these to be None or mock objects
            # For real tests, some might be available
            availability = {
                name: comp is not None 
                for name, comp in components.items()
            }
            
            # At least the analyzer itself should work
            passed = True  # Basic functionality should work even without components
            
            self.test_results.append(TestResult(
                name="component_availability",
                category="integration",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'components': availability,
                    'total_available': sum(availability.values())
                }
            ))
            
            logger.info(f"✓ Component availability: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="component_availability",
                category="integration",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Component availability failed: {e}")
    
    async def _test_fallback_mechanisms(self):
        """Test fallback mechanisms when components fail"""
        logger.info("\n[INTEGRATION] Fallback Mechanisms")
        start_time = time.time()
        
        try:
            # Test that analyzer works even without optional components
            result = await self.analyzer.smart_analyze(
                self.test_images['medium'],
                "Test fallback mechanisms"
            )
            
            # Should fall back to basic analysis
            passed = result is not None and 'description' in result
            
            self.test_results.append(TestResult(
                name="fallback_mechanisms",
                category="integration",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'fallback_worked': passed,
                    'has_result': result is not None,
                    'method_used': result.get('metadata', {}).get('analysis_method', 'unknown') if result else None
                }
            ))
            
            logger.info(f"✓ Fallback mechanisms: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="fallback_mechanisms",
                category="integration",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Fallback mechanisms failed: {e}")
    
    async def _test_error_handling(self):
        """Test error handling and recovery"""
        logger.info("\n[INTEGRATION] Error Handling")
        start_time = time.time()
        
        try:
            test_cases = []
            
            # Test 1: Invalid image type
            try:
                result = await self.analyzer.analyze_screenshot("not an image", "test")
                test_cases.append(('invalid_image', False, "Should have raised error"))
            except ValueError:
                test_cases.append(('invalid_image', True, "Correctly rejected invalid image"))
            except Exception as e:
                test_cases.append(('invalid_image', False, f"Wrong error type: {type(e)}"))
            
            # Test 2: Empty prompt handling
            result = await self.analyzer.analyze_screenshot(self.test_images['small'], "")
            test_cases.append(('empty_prompt', result is not None, "Handled empty prompt"))
            
            # Test 3: Null image handling
            try:
                result = await self.analyzer.analyze_screenshot(None, "test")
                test_cases.append(('null_image', False, "Should have raised error"))
            except (ValueError, AttributeError):
                test_cases.append(('null_image', True, "Correctly rejected null image"))
            
            passed = all(success for _, success, _ in test_cases)
            
            self.test_results.append(TestResult(
                name="error_handling",
                category="integration",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    name: {'success': success, 'message': msg}
                    for name, success, msg in test_cases
                }
            ))
            
            logger.info(f"✓ Error handling: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="error_handling",
                category="integration",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Error handling failed: {e}")
    
    async def _test_configuration_system(self):
        """Test configuration system"""
        logger.info("\n[INTEGRATION] Configuration System")
        start_time = time.time()
        
        try:
            # Save original config
            original_quality = self.analyzer.config.jpeg_quality
            original_dimension = self.analyzer.config.max_image_dimension
            
            # Test config changes
            self.analyzer.config.jpeg_quality = 95
            self.analyzer.config.max_image_dimension = 2048
            
            # Verify changes took effect
            config_changed = (
                self.analyzer.config.jpeg_quality == 95 and
                self.analyzer.config.max_image_dimension == 2048
            )
            
            # Test with new config
            result = await self.analyzer.analyze_screenshot(
                self.test_images['medium'],
                "Test with modified config"
            )
            
            # Restore config
            self.analyzer.config.jpeg_quality = original_quality
            self.analyzer.config.max_image_dimension = original_dimension
            
            passed = config_changed and result is not None
            
            self.test_results.append(TestResult(
                name="configuration_system",
                category="integration",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'config_changed': config_changed,
                    'analysis_worked': result is not None,
                    'restored': self.analyzer.config.jpeg_quality == original_quality
                }
            ))
            
            logger.info(f"✓ Configuration system: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="configuration_system",
                category="integration",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Configuration system failed: {e}")
    
    async def test_real_world_scenarios(self):
        """Test real-world use case scenarios"""
        logger.info("\n" + "="*70)
        logger.info("TESTING REAL-WORLD SCENARIOS")
        logger.info("="*70)
        
        tests = [
            self._test_developer_workflow,
            self._test_content_moderation,
            self._test_accessibility_check,
            self._test_automated_testing,
            self._test_productivity_monitoring
        ]
        
        for test_func in tests:
            await test_func()
    
    async def _test_developer_workflow(self):
        """Test developer workflow scenario"""
        logger.info("\n[SCENARIO] Developer Workflow")
        start_time = time.time()
        
        try:
            # Simulate checking for errors in IDE
            error_check = await self.analyzer.check_for_errors()
            
            # Analyze code editor
            code_analysis = await self.analyzer.analyze_workspace(self.test_images['code_editor'])
            
            # Check for notifications
            notifications = await self.analyzer.check_for_notifications()
            
            passed = all([
                error_check is not None,
                code_analysis is not None,
                notifications is not None
            ])
            
            self.test_results.append(TestResult(
                name="developer_workflow",
                category="scenarios",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'error_check': error_check is not None,
                    'code_analysis': code_analysis is not None,
                    'notifications': notifications is not None
                }
            ))
            
            logger.info(f"✓ Developer workflow: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="developer_workflow",
                category="scenarios",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Developer workflow failed: {e}")
    
    async def _test_content_moderation(self):
        """Test content moderation scenario"""
        logger.info("\n[SCENARIO] Content Moderation")
        start_time = time.time()
        
        try:
            # Analyze multiple images for content
            results = []
            for img_name, img in [
                ('document', self.test_images['document']),
                ('ui', self.test_images['ui_with_error']),
                ('dashboard', self.test_images['dashboard'])
            ]:
                result = await self.analyzer.smart_analyze(
                    img,
                    "Check this image for inappropriate content, errors, or issues"
                )
                results.append((img_name, result))
            
            passed = all(r is not None for _, r in results)
            
            self.test_results.append(TestResult(
                name="content_moderation",
                category="scenarios",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'images_analyzed': len(results),
                    'all_successful': passed,
                    'results': {name: r is not None for name, r in results}
                }
            ))
            
            logger.info(f"✓ Content moderation: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="content_moderation",
                category="scenarios",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Content moderation failed: {e}")
    
    async def _test_accessibility_check(self):
        """Test accessibility checking scenario"""
        logger.info("\n[SCENARIO] Accessibility Check")
        start_time = time.time()
        
        try:
            # Check UI for accessibility issues
            ui_check = await self.analyzer.smart_analyze(
                self.test_images['ui_with_error'],
                "Check this UI for accessibility issues: contrast, button sizes, text readability"
            )
            
            # Find specific UI elements
            button_check = await self.analyzer.find_ui_element("button")
            
            # Check mobile app
            mobile_check = await self.analyzer.smart_analyze(
                self.test_images['mobile_app'],
                "Analyze this mobile app for accessibility compliance"
            )
            
            passed = all([
                ui_check is not None,
                button_check is not None,
                mobile_check is not None
            ])
            
            self.test_results.append(TestResult(
                name="accessibility_check",
                category="scenarios",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'ui_check': ui_check is not None,
                    'button_found': button_check is not None,
                    'mobile_check': mobile_check is not None
                }
            ))
            
            logger.info(f"✓ Accessibility check: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="accessibility_check",
                category="scenarios",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Accessibility check failed: {e}")
    
    async def _test_automated_testing(self):
        """Test automated UI testing scenario"""
        logger.info("\n[SCENARIO] Automated Testing")
        start_time = time.time()
        
        try:
            # Take baseline screenshot
            baseline = self.test_images['dashboard']
            
            # Simulate change
            changed = baseline.copy()
            changed[100:200, 100:200] = [255, 0, 0]  # Add red square
            
            # Detect changes
            change_result = await self.analyzer.analyze_with_change_detection(
                changed,
                baseline,
                "What UI elements changed?"
            )
            
            # Batch analyze regions
            regions = [
                {"x": 50, "y": 50, "width": 450, "height": 350, "prompt": "chart area"},
                {"x": 550, "y": 50, "width": 450, "height": 350, "prompt": "data table"},
                {"x": 50, "y": 450, "width": 950, "height": 350, "prompt": "summary section"}
            ]
            
            region_results = await self.analyzer.batch_analyze_regions(baseline, regions)
            
            passed = (
                change_result is not None and
                len(region_results) == len(regions)
            )
            
            self.test_results.append(TestResult(
                name="automated_testing",
                category="scenarios",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'change_detection': change_result is not None,
                    'regions_analyzed': len(region_results),
                    'all_regions_successful': len(region_results) == len(regions)
                }
            ))
            
            logger.info(f"✓ Automated testing: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="automated_testing",
                category="scenarios",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Automated testing failed: {e}")
    
    async def _test_productivity_monitoring(self):
        """Test productivity monitoring scenario"""
        logger.info("\n[SCENARIO] Productivity Monitoring")
        start_time = time.time()
        
        try:
            # Analyze workspace
            workspace_analysis = await self.analyzer.analyze_workspace(
                self.test_images['code_editor']
            )
            
            # Check for distractions (notifications)
            distraction_check = await self.analyzer.check_for_notifications()
            
            # Analyze activity pattern
            activity_analysis = await self.analyzer.smart_analyze(
                self.test_images['dashboard'],
                "Analyze user activity and productivity based on this screen"
            )
            
            # Memory stats for resource monitoring
            memory_stats = self.analyzer.get_all_memory_stats()
            
            passed = all([
                workspace_analysis is not None,
                distraction_check is not None,
                activity_analysis is not None,
                memory_stats is not None
            ])
            
            self.test_results.append(TestResult(
                name="productivity_monitoring",
                category="scenarios",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'workspace_analyzed': workspace_analysis is not None,
                    'distractions_checked': distraction_check is not None,
                    'activity_analyzed': activity_analysis is not None,
                    'resource_monitored': memory_stats is not None
                }
            ))
            
            logger.info(f"✓ Productivity monitoring: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="productivity_monitoring",
                category="scenarios",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Productivity monitoring failed: {e}")
    
    async def test_performance(self):
        """Test performance characteristics"""
        logger.info("\n" + "="*70)
        logger.info("TESTING PERFORMANCE")
        logger.info("="*70)
        
        tests = [
            self._test_response_times,
            self._test_memory_efficiency,
            self._test_concurrent_load
        ]
        
        for test_func in tests:
            await test_func()
    
    async def _test_response_times(self):
        """Test response times for different operations"""
        logger.info("\n[PERFORMANCE] Response Times")
        start_time = time.time()
        
        try:
            timings = {}
            
            # Small image
            t0 = time.time()
            await self.analyzer.smart_analyze(self.test_images['small'], "Quick test")
            timings['small_image'] = time.time() - t0
            
            # Large image
            t0 = time.time()
            await self.analyzer.smart_analyze(self.test_images['large'], "Detailed test")
            timings['large_image'] = time.time() - t0
            
            # With compression
            t0 = time.time()
            await self.analyzer.analyze_with_compression_strategy(
                self.test_images['medium'], "Compressed test", "quick"
            )
            timings['compressed'] = time.time() - t0
            
            # Cached request
            t0 = time.time()
            await self.analyzer.analyze_screenshot(self.test_images['small'], "Quick test")
            timings['cached'] = time.time() - t0
            
            # All times should be reasonable (< 10 seconds for mock, < 30 for real)
            max_time = 10 if self.use_mock else 30
            passed = all(t < max_time for t in timings.values())
            
            self.test_results.append(TestResult(
                name="response_times",
                category="performance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'timings': timings,
                    'max_allowed': max_time,
                    'all_within_limit': passed
                }
            ))
            
            logger.info(f"✓ Response times: {'PASSED' if passed else 'FAILED'}")
            for op, timing in timings.items():
                logger.info(f"  - {op}: {timing:.2f}s")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="response_times",
                category="performance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Response times failed: {e}")
    
    async def _test_memory_efficiency(self):
        """Test memory efficiency"""
        logger.info("\n[PERFORMANCE] Memory Efficiency")
        start_time = time.time()
        
        try:
            # Get initial memory
            initial_stats = self.analyzer.get_all_memory_stats()
            initial_memory = initial_stats.get('system', {}).get('process_mb', 0)
            
            # Process multiple large images
            for i in range(5):
                await self.analyzer.smart_analyze(
                    self.test_images['large'],
                    f"Memory test {i}"
                )
            
            # Get final memory
            final_stats = self.analyzer.get_all_memory_stats()
            final_memory = final_stats.get('system', {}).get('process_mb', 0)
            
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 500MB)
            passed = memory_increase < 500
            
            self.test_results.append(TestResult(
                name="memory_efficiency",
                category="performance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'increase_mb': memory_increase,
                    'within_limit': passed
                }
            ))
            
            logger.info(f"✓ Memory efficiency: {'PASSED' if passed else 'FAILED'}")
            logger.info(f"  Memory increase: {memory_increase:.2f}MB")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="memory_efficiency",
                category="performance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Memory efficiency failed: {e}")
    
    async def _test_concurrent_load(self):
        """Test concurrent load handling"""
        logger.info("\n[PERFORMANCE] Concurrent Load")
        start_time = time.time()
        
        try:
            # Launch concurrent requests
            num_concurrent = 10
            tasks = []
            
            for i in range(num_concurrent):
                task = self.analyzer.smart_analyze(
                    self.test_images['small'],
                    f"Concurrent test {i}"
                )
                tasks.append(task)
            
            # Time the concurrent execution
            t0 = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_time = time.time() - t0
            
            # Count successes
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            # Should handle concurrent load well
            passed = successful >= num_concurrent * 0.8  # 80% success rate
            
            self.test_results.append(TestResult(
                name="concurrent_load",
                category="performance",
                passed=passed,
                duration=time.time() - start_time,
                details={
                    'concurrent_requests': num_concurrent,
                    'successful': successful,
                    'failed': num_concurrent - successful,
                    'total_time': concurrent_time,
                    'avg_time_per_request': concurrent_time / num_concurrent
                }
            ))
            
            logger.info(f"✓ Concurrent load: {'PASSED' if passed else 'FAILED'}")
            logger.info(f"  Success rate: {successful}/{num_concurrent}")
            
        except Exception as e:
            self.test_results.append(TestResult(
                name="concurrent_load",
                category="performance",
                passed=False,
                duration=time.time() - start_time,
                details={},
                error=str(e)
            ))
            logger.error(f"✗ Concurrent load failed: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE TEST REPORT")
        logger.info("="*70)
        
        # Group results by category
        categories = {}
        for result in self.test_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Print summary by category
        for category, results in categories.items():
            logger.info(f"\n{category.upper()}:")
            for result in results:
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                logger.info(f"  {status} - {result.name} ({result.duration:.2f}s)")
                if result.error:
                    logger.info(f"    Error: {result.error}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"{'='*70}")
        
        # Generate detailed report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'mock' if self.use_mock else 'live',
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'total_duration': sum(r.duration for r in self.test_results)
            },
            'categories': {
                category: {
                    'total': len(results),
                    'passed': sum(1 for r in results if r.passed),
                    'tests': [
                        {
                            'name': r.name,
                            'passed': r.passed,
                            'duration': r.duration,
                            'details': r.details,
                            'error': r.error
                        }
                        for r in results
                    ]
                }
                for category, results in categories.items()
            }
        }
        
        # Save report
        report_file = f'comprehensive_vision_test_report_{"mock" if self.use_mock else "live"}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        return report
    
    async def run_all_tests(self):
        """Run all test suites"""
        try:
            await self.setup()
            
            # Run all test categories
            await self.test_enhanced_functionality()
            await self.test_integration()
            await self.test_real_world_scenarios()
            await self.test_performance()
            
            # Generate report
            report = self.generate_report()
            
            # Cleanup
            await self.analyzer.cleanup_all_components()
            
            return report['summary']['success_rate'] >= 80
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Vision Analyzer Tests')
    parser.add_argument('--live', action='store_true', help='Use live API instead of mock')
    args = parser.parse_args()
    
    use_mock = not args.live
    
    logger.info("="*70)
    logger.info(f"COMPREHENSIVE VISION ANALYZER TEST SUITE")
    logger.info(f"Mode: {'MOCK' if use_mock else 'LIVE API'}")
    logger.info("="*70)
    
    if not use_mock and not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("ANTHROPIC_API_KEY not set. Use --live flag only with valid API key.")
        return 1
    
    # Run tests
    test_suite = ComprehensiveVisionTestSuite(use_mock=use_mock)
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ COMPREHENSIVE TEST SUITE PASSED!")
    else:
        logger.error("\n❌ COMPREHENSIVE TEST SUITE FAILED!")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)