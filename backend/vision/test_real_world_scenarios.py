#!/usr/bin/env python3
"""
Real-world scenario tests for the enhanced vision system.
Tests practical use cases that users would encounter.
"""

import asyncio
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
import time
from typing import Dict, List, Any

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

# Add vision directory to path as well
vision_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vision_path)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealWorldScenarioTests:
    """Test real-world use cases for the vision system"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        self.analyzer = None
        self.test_results = {}
    
    async def setup(self):
        """Set up test environment"""
        try:
            from claude_vision_analyzer_main import ClaudeVisionAnalyzer
            self.analyzer = ClaudeVisionAnalyzer(self.api_key)
            logger.info("Vision analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise
    
    def create_code_editor_image(self) -> np.ndarray:
        """Create an image that looks like a code editor"""
        img = Image.new('RGB', (1920, 1080), color='#1e1e1e')
        draw = ImageDraw.Draw(img)
        
        # Add code-like text
        y_pos = 50
        code_lines = [
            "def analyze_screen(image):",
            "    # Process the image",
            "    result = vision_api.analyze(image)",
            "    return result",
            "",
            "class VisionSystem:",
            "    def __init__(self):",
            "        self.components = []"
        ]
        
        for i, line in enumerate(code_lines):
            # Line numbers
            draw.text((10, y_pos), str(i+1), fill='#858585')
            # Code
            draw.text((50, y_pos), line, fill='#d4d4d4')
            y_pos += 25
        
        return np.array(img)
    
    def create_browser_image(self) -> np.ndarray:
        """Create an image that looks like a web browser"""
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        # Browser chrome
        draw.rectangle([0, 0, 1920, 80], fill='#f0f0f0')
        draw.text((100, 25), "https://example.com", fill='#333333')
        
        # Web content
        draw.text((100, 150), "Welcome to Example Website", fill='#000000')
        draw.rectangle([100, 200, 500, 300], fill='#3498db')
        draw.text((250, 240), "Click Here", fill='white')
        
        return np.array(img)
    
    def create_notification_image(self) -> np.ndarray:
        """Create an image with notifications"""
        img = Image.new('RGB', (1920, 1080), color='#f5f5f5')
        draw = ImageDraw.Draw(img)
        
        # Notification banner
        draw.rectangle([1520, 50, 1870, 150], fill='#ff4444')
        draw.text((1550, 80), "3 New Messages", fill='white')
        
        # Error dialog
        draw.rectangle([600, 400, 1200, 600], fill='white', outline='#ff0000')
        draw.text((700, 450), "Error: File not found", fill='#ff0000')
        draw.rectangle([900, 540, 1000, 580], fill='#0066cc')
        draw.text((925, 550), "OK", fill='white')
        
        return np.array(img)
    
    def create_multi_window_image(self) -> np.ndarray:
        """Create an image with multiple application windows"""
        img = Image.new('RGB', (1920, 1080), color='#cccccc')
        draw = ImageDraw.Draw(img)
        
        # Window 1 - Code editor
        draw.rectangle([50, 50, 900, 600], fill='#1e1e1e', outline='#333333')
        draw.text((60, 60), "main.py - Visual Studio Code", fill='white')
        
        # Window 2 - Terminal
        draw.rectangle([950, 50, 1850, 400], fill='#000000', outline='#333333')
        draw.text((960, 60), "Terminal", fill='#00ff00')
        draw.text((960, 100), "$ python test.py", fill='#00ff00')
        
        # Window 3 - Browser
        draw.rectangle([200, 650, 1600, 1000], fill='white', outline='#333333')
        draw.text((210, 660), "Documentation - Chrome", fill='#333333')
        
        return np.array(img)
    
    async def test_code_analysis_scenario(self):
        """Test: Analyzing code in an editor"""
        logger.info("\n[SCENARIO 1] Code Editor Analysis")
        
        try:
            code_image = self.create_code_editor_image()
            
            # Test 1: Read code content
            result = await self.analyzer.analyze_with_compression_strategy(
                code_image,
                "What code is visible on the screen?",
                "text"  # Use text strategy for code
            )
            
            success = result is not None
            self.test_results['code_analysis'] = {
                'success': success,
                'description': 'Code content analysis'
            }
            
            # Test 2: Find specific function
            element_result = await self.analyzer.find_ui_element("analyze_screen function")
            
            if element_result and element_result.get('success'):
                logger.info("  ✓ Found code elements successfully")
            else:
                logger.info("  ⚠ Code element search needs improvement")
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Code analysis failed: {e}")
            self.test_results['code_analysis'] = {'success': False, 'error': str(e)}
    
    async def test_web_interaction_scenario(self):
        """Test: Analyzing web browser content"""
        logger.info("\n[SCENARIO 2] Web Browser Interaction")
        
        try:
            browser_image = self.create_browser_image()
            
            # Test 1: Find clickable elements
            result = await self.analyzer.analyze_with_compression_strategy(
                browser_image,
                "Find all clickable buttons and links",
                "ui"  # Use UI strategy for web elements
            )
            
            success = result is not None
            
            # Test 2: Read web content
            text_result = await self.analyzer.read_text_from_area(
                browser_image,
                {"x": 100, "y": 150, "width": 800, "height": 50}
            )
            
            self.test_results['web_interaction'] = {
                'success': success,
                'found_elements': bool(result),
                'read_text': bool(text_result)
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Web interaction failed: {e}")
            self.test_results['web_interaction'] = {'success': False, 'error': str(e)}
    
    async def test_notification_detection_scenario(self):
        """Test: Detecting notifications and errors"""
        logger.info("\n[SCENARIO 3] Notification Detection")
        
        try:
            notification_image = self.create_notification_image()
            
            # Test 1: Check for notifications
            notifications = await self.analyzer.check_for_notifications()
            
            # Test 2: Check for errors
            errors = await self.analyzer.check_for_errors()
            
            # Test 3: Analyze with activity strategy
            result = await self.analyzer.analyze_with_compression_strategy(
                notification_image,
                "Are there any notifications or alerts on screen?",
                "activity"  # Use activity strategy for monitoring
            )
            
            success = all([
                notifications is not None,
                errors is not None,
                result is not None
            ])
            
            self.test_results['notification_detection'] = {
                'success': success,
                'notifications_checked': notifications is not None,
                'errors_checked': errors is not None
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Notification detection failed: {e}")
            self.test_results['notification_detection'] = {'success': False, 'error': str(e)}
    
    async def test_multi_window_scenario(self):
        """Test: Analyzing multiple windows and relationships"""
        logger.info("\n[SCENARIO 4] Multi-Window Analysis")
        
        try:
            multi_window_image = self.create_multi_window_image()
            
            # Test 1: Analyze workspace
            workspace = await self.analyzer.analyze_workspace()
            
            # Test 2: Detect window relationships
            window_analyzer = await self.analyzer.get_window_analyzer()
            relationship_detector = await self.analyzer.get_relationship_detector()
            
            # Test 3: Smart analysis for overview
            result = await self.analyzer.smart_analyze(
                multi_window_image,
                "What applications are open and how are they related?"
            )
            
            success = result is not None
            
            self.test_results['multi_window'] = {
                'success': success,
                'workspace_analyzed': workspace is not None,
                'components_available': all([window_analyzer, relationship_detector])
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Multi-window analysis failed: {e}")
            self.test_results['multi_window'] = {'success': False, 'error': str(e)}
    
    async def test_memory_pressure_scenario(self):
        """Test: System behavior under memory pressure"""
        logger.info("\n[SCENARIO 5] Memory Pressure Handling")
        
        try:
            # Create a large image to simulate memory pressure
            large_image = np.random.randint(0, 255, (4000, 6000, 3), dtype=np.uint8)
            
            # Get initial memory stats
            initial_stats = self.analyzer.get_all_memory_stats()
            initial_memory = initial_stats['system']['process_mb']
            
            # Test 1: Process large image with quick strategy
            result1 = await self.analyzer.analyze_with_compression_strategy(
                large_image,
                "Quick analysis of large image",
                "quick"  # Should use minimal resources
            )
            
            # Test 2: Check if memory management kicked in
            mid_stats = self.analyzer.get_all_memory_stats()
            mid_memory = mid_stats['system']['process_mb']
            
            # Test 3: Process another large image
            result2 = await self.analyzer.analyze_with_compression_strategy(
                large_image,
                "Another large image analysis",
                "quick"
            )
            
            # Test 4: Cleanup and check memory
            await self.analyzer.cleanup_all_components()
            
            final_stats = self.analyzer.get_all_memory_stats()
            final_memory = final_stats['system']['process_mb']
            
            # Memory should not grow excessively
            memory_growth = final_memory - initial_memory
            success = all([result1, result2]) and memory_growth < 500  # Less than 500MB growth
            
            self.test_results['memory_pressure'] = {
                'success': success,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': mid_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth
            }
            
            logger.info(f"  Memory growth: {memory_growth:.1f}MB")
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Memory pressure test failed: {e}")
            self.test_results['memory_pressure'] = {'success': False, 'error': str(e)}
    
    async def test_continuous_monitoring_scenario(self):
        """Test: Continuous screen monitoring"""
        logger.info("\n[SCENARIO 6] Continuous Monitoring")
        
        try:
            continuous = await self.analyzer.get_continuous_analyzer()
            
            if not continuous:
                logger.warning("  ⚠ Continuous analyzer not available")
                self.test_results['continuous_monitoring'] = {
                    'success': False,
                    'reason': 'Component not available'
                }
                return
            
            # Test monitoring for changes
            test_images = [
                self.create_code_editor_image(),
                self.create_browser_image(),
                self.create_notification_image()
            ]
            
            results = []
            for i, img in enumerate(test_images):
                # Simulate continuous monitoring
                result = await self.analyzer.smart_analyze(
                    img,
                    f"Monitor screen state {i+1}"
                )
                results.append(result is not None)
                
                # Check memory stats
                stats = continuous.get_memory_stats()
                logger.info(f"  Monitor {i+1}: Interval={stats.get('current_interval', 'N/A')}s")
                
                # Small delay to simulate real monitoring
                await asyncio.sleep(0.5)
            
            success = all(results)
            
            self.test_results['continuous_monitoring'] = {
                'success': success,
                'screens_monitored': len(results),
                'all_successful': all(results)
            }
            
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Continuous monitoring failed: {e}")
            self.test_results['continuous_monitoring'] = {'success': False, 'error': str(e)}
    
    async def test_performance_scenario(self):
        """Test: Performance under load"""
        logger.info("\n[SCENARIO 7] Performance Under Load")
        
        try:
            # Test batch processing performance
            num_requests = 5
            start_time = time.time()
            
            # Create batch of different scenarios
            tasks = [
                self.analyzer.smart_analyze(self.create_code_editor_image(), "Analyze code"),
                self.analyzer.find_ui_element("button"),
                self.analyzer.check_for_notifications(),
                self.analyzer.analyze_workspace(),
                self.analyzer.check_for_errors()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            avg_time = total_time / num_requests
            
            # Should complete reasonably fast
            success = successful == num_requests and avg_time < 2.0  # Less than 2s per request
            
            self.test_results['performance'] = {
                'success': success,
                'total_requests': num_requests,
                'successful_requests': successful,
                'total_time': total_time,
                'avg_time_per_request': avg_time
            }
            
            logger.info(f"  Processed {successful}/{num_requests} in {total_time:.2f}s")
            logger.info(f"  Average: {avg_time:.2f}s per request")
            logger.info(f"  Result: {'PASSED' if success else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ✗ Performance test failed: {e}")
            self.test_results['performance'] = {'success': False, 'error': str(e)}
    
    def generate_report(self):
        """Generate scenario test report"""
        logger.info("\n" + "="*70)
        logger.info("REAL-WORLD SCENARIO TEST REPORT")
        logger.info("="*70)
        
        total_scenarios = len(self.test_results)
        passed_scenarios = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        for scenario, result in self.test_results.items():
            status = "PASSED" if result.get('success', False) else "FAILED"
            logger.info(f"\n{scenario}: {status}")
            
            if not result.get('success', False):
                if 'error' in result:
                    logger.info(f"  Error: {result['error']}")
                elif 'reason' in result:
                    logger.info(f"  Reason: {result['reason']}")
        
        success_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL: {passed_scenarios}/{total_scenarios} scenarios passed ({success_rate:.1f}%)")
        logger.info(f"{'='*70}")
        
        return success_rate >= 70  # 70% pass rate for scenarios
    
    async def run_all_scenarios(self):
        """Run all scenario tests"""
        try:
            await self.setup()
            
            # Run scenarios
            await self.test_code_analysis_scenario()
            await self.test_web_interaction_scenario()
            await self.test_notification_detection_scenario()
            await self.test_multi_window_scenario()
            await self.test_memory_pressure_scenario()
            await self.test_continuous_monitoring_scenario()
            await self.test_performance_scenario()
            
            # Generate report
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"Scenario tests failed: {e}")
            return False


async def main():
    """Main test runner"""
    logger.info("Enhanced Vision System - Real-World Scenario Tests")
    logger.info("Testing practical use cases\n")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("No API key found - these tests require ANTHROPIC_API_KEY")
        logger.info("These scenarios test real vision analysis capabilities")
        return 1
    
    # Run tests
    test_suite = RealWorldScenarioTests()
    success = await test_suite.run_all_scenarios()
    
    if success:
        logger.info("\n✅ SCENARIO TESTS PASSED!")
    else:
        logger.error("\n❌ SCENARIO TESTS FAILED!")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)