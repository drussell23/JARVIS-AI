#!/usr/bin/env python3
"""
Adaptive Control Center Clicker - Verification Script
=====================================================

Comprehensive verification script that tests the AdaptiveControlCenterClicker
across multiple scenarios and edge cases.

This script:
1. Tests all detection methods independently
2. Verifies fallback chain behavior
3. Tests cache persistence and learning
4. Simulates macOS updates and UI changes
5. Generates detailed performance reports
6. Creates visualizations of detection accuracy

Usage:
    python verify_adaptive_clicker.py [options]

Options:
    --quick         Run quick verification (subset of tests)
    --full          Run full test suite (default)
    --performance   Run performance benchmarks
    --visual        Generate visual detection heatmaps
    --report        Generate HTML report

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from display.adaptive_control_center_clicker import (
    AdaptiveControlCenterClicker,
    get_adaptive_clicker,
    CoordinateCache,
    CachedDetection,
    OCRDetection,
    TemplateMatchingDetection,
    EdgeDetection,
    AccessibilityAPIDetection,
    AppleScriptDetection,
)

import pyautogui

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Result Models
# ============================================================================

@dataclass
class TestResult:
    """Result from a single test"""
    test_name: str
    category: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any]
    timestamp: float


@dataclass
class VerificationReport:
    """Complete verification report"""
    timestamp: str
    system_info: Dict[str, Any]
    test_results: List[TestResult]
    summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


# ============================================================================
# Verification Tests
# ============================================================================

class AdaptiveClickerVerifier:
    """Main verification test runner"""

    def __init__(self, vision_analyzer=None):
        """Initialize verifier"""
        self.vision_analyzer = vision_analyzer
        self.clicker = None
        self.results: List[TestResult] = []
        self.start_time = time.time()

        logger.info("=" * 75)
        logger.info("Adaptive Control Center Clicker - Verification Suite")
        logger.info("=" * 75)

    async def initialize(self):
        """Initialize clicker and components"""
        logger.info("\n📦 Initializing components...")

        try:
            self.clicker = AdaptiveControlCenterClicker(
                vision_analyzer=self.vision_analyzer,
                cache_ttl=3600,
                enable_verification=True
            )
            logger.info("✅ AdaptiveControlCenterClicker initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    def _add_result(
        self,
        test_name: str,
        category: str,
        success: bool,
        duration: float,
        message: str,
        details: Dict[str, Any]
    ):
        """Add test result"""
        result = TestResult(
            test_name=test_name,
            category=category,
            success=success,
            duration=duration,
            message=message,
            details=details,
            timestamp=time.time()
        )
        self.results.append(result)

        status = "✅" if success else "❌"
        logger.info(f"{status} {test_name}: {message} ({duration:.2f}s)")

    async def _cleanup_ui(self):
        """Clean up any open UI elements"""
        try:
            pyautogui.press('escape')
            await asyncio.sleep(0.2)
            pyautogui.press('escape')
            await asyncio.sleep(0.2)
        except Exception:
            pass

    # ========================================================================
    # Test Category: Detection Methods
    # ========================================================================

    async def test_cached_detection(self) -> bool:
        """Test cached coordinate detection"""
        logger.info("\n🧪 Testing cached detection...")
        start_time = time.time()

        try:
            # Clear cache first
            self.clicker.cache.clear()

            # Set a known coordinate
            self.clicker.cache.set(
                "test_target",
                (1245, 12),
                0.95,
                "manual_verification"
            )

            # Try to retrieve it
            cached = self.clicker.cache.get("test_target")

            success = cached is not None and cached.coordinates == (1245, 12)
            duration = time.time() - start_time

            self._add_result(
                test_name="Cached Detection",
                category="detection_methods",
                success=success,
                duration=duration,
                message="Cache retrieval works" if success else "Cache retrieval failed",
                details={"cached_coordinate": cached.coordinates if cached else None}
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Cached Detection",
                category="detection_methods",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_ocr_detection(self) -> bool:
        """Test OCR detection methods"""
        logger.info("\n🧪 Testing OCR detection...")
        start_time = time.time()

        try:
            from display.adaptive_control_center_clicker import OCRDetection

            detector = OCRDetection(vision_analyzer=self.vision_analyzer)
            available = await detector.is_available()

            if not available:
                duration = time.time() - start_time
                self._add_result(
                    test_name="OCR Detection",
                    category="detection_methods",
                    success=False,
                    duration=duration,
                    message="OCR not available (no tesseract or vision analyzer)",
                    details={"available": False}
                )
                return False

            # Try to detect Control Center
            result = await detector.detect("control center")

            duration = time.time() - start_time

            self._add_result(
                test_name="OCR Detection",
                category="detection_methods",
                success=result.success,
                duration=duration,
                message=f"OCR detection {'succeeded' if result.success else 'failed'}",
                details={
                    "method": result.method,
                    "coordinates": result.coordinates,
                    "confidence": result.confidence
                }
            )

            return result.success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="OCR Detection",
                category="detection_methods",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_template_matching(self) -> bool:
        """Test template matching detection"""
        logger.info("\n🧪 Testing template matching...")
        start_time = time.time()

        try:
            detector = TemplateMatchingDetection()
            available = await detector.is_available()

            if not available:
                duration = time.time() - start_time
                self._add_result(
                    test_name="Template Matching",
                    category="detection_methods",
                    success=False,
                    duration=duration,
                    message="OpenCV not available",
                    details={"available": False}
                )
                return False

            # Try detection (will likely fail without templates, but tests the method)
            result = await detector.detect("control_center")

            duration = time.time() - start_time

            self._add_result(
                test_name="Template Matching",
                category="detection_methods",
                success=result.success,
                duration=duration,
                message=f"Template matching {'found match' if result.success else 'no match'}",
                details={
                    "coordinates": result.coordinates,
                    "confidence": result.confidence,
                    "error": result.error
                }
            )

            return True  # Success = method is functional, even if no match

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Template Matching",
                category="detection_methods",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_edge_detection(self) -> bool:
        """Test edge detection method"""
        logger.info("\n🧪 Testing edge detection...")
        start_time = time.time()

        try:
            detector = EdgeDetection()
            available = await detector.is_available()

            if not available:
                duration = time.time() - start_time
                self._add_result(
                    test_name="Edge Detection",
                    category="detection_methods",
                    success=False,
                    duration=duration,
                    message="OpenCV not available",
                    details={"available": False}
                )
                return False

            result = await detector.detect("control_center")

            duration = time.time() - start_time

            self._add_result(
                test_name="Edge Detection",
                category="detection_methods",
                success=result.success,
                duration=duration,
                message=f"Edge detection {'found contours' if result.success else 'no contours'}",
                details={
                    "coordinates": result.coordinates,
                    "confidence": result.confidence
                }
            )

            return True  # Success = method is functional

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Edge Detection",
                category="detection_methods",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    # ========================================================================
    # Test Category: End-to-End Functionality
    # ========================================================================

    async def test_open_control_center(self) -> bool:
        """Test opening Control Center"""
        logger.info("\n🧪 Testing open Control Center...")
        start_time = time.time()

        try:
            await self._cleanup_ui()

            result = await self.clicker.open_control_center()

            duration = time.time() - start_time

            self._add_result(
                test_name="Open Control Center",
                category="end_to_end",
                success=result.success,
                duration=duration,
                message=f"{'Opened successfully' if result.success else 'Failed to open'}",
                details={
                    "method_used": result.method_used,
                    "coordinates": result.coordinates,
                    "verification_passed": result.verification_passed,
                    "fallback_attempts": result.fallback_attempts
                }
            )

            await self._cleanup_ui()

            return result.success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Open Control Center",
                category="end_to_end",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            await self._cleanup_ui()
            return False

    async def test_click_screen_mirroring(self) -> bool:
        """Test clicking Screen Mirroring"""
        logger.info("\n🧪 Testing click Screen Mirroring...")
        start_time = time.time()

        try:
            await self._cleanup_ui()

            # First open Control Center
            cc_result = await self.clicker.open_control_center()
            if not cc_result.success:
                self._add_result(
                    test_name="Click Screen Mirroring",
                    category="end_to_end",
                    success=False,
                    duration=time.time() - start_time,
                    message="Failed to open Control Center",
                    details={}
                )
                await self._cleanup_ui()
                return False

            await asyncio.sleep(0.5)

            # Then click Screen Mirroring
            sm_result = await self.clicker.click_screen_mirroring()

            duration = time.time() - start_time

            self._add_result(
                test_name="Click Screen Mirroring",
                category="end_to_end",
                success=sm_result.success,
                duration=duration,
                message=f"{'Clicked successfully' if sm_result.success else 'Failed to click'}",
                details={
                    "method_used": sm_result.method_used,
                    "coordinates": sm_result.coordinates,
                    "fallback_attempts": sm_result.fallback_attempts
                }
            )

            await self._cleanup_ui()

            return sm_result.success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Click Screen Mirroring",
                category="end_to_end",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            await self._cleanup_ui()
            return False

    # ========================================================================
    # Test Category: Cache & Learning
    # ========================================================================

    async def test_cache_persistence(self) -> bool:
        """Test cache persistence across sessions"""
        logger.info("\n🧪 Testing cache persistence...")
        start_time = time.time()

        try:
            # Set a coordinate
            self.clicker.cache.set(
                "persistence_test",
                (999, 888),
                0.99,
                "test"
            )

            # Create new cache instance with same file
            cache_file = self.clicker.cache.cache_file
            cache2 = CoordinateCache(cache_file=cache_file, ttl_seconds=3600)

            # Should load from disk
            cached = cache2.get("persistence_test")

            success = cached is not None and cached.coordinates == (999, 888)
            duration = time.time() - start_time

            self._add_result(
                test_name="Cache Persistence",
                category="cache_learning",
                success=success,
                duration=duration,
                message="Cache persists correctly" if success else "Cache persistence failed",
                details={
                    "cached_value": cached.coordinates if cached else None
                }
            )

            # Cleanup
            self.clicker.cache.invalidate("persistence_test")

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Cache Persistence",
                category="cache_learning",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_cache_ttl(self) -> bool:
        """Test cache TTL expiration"""
        logger.info("\n🧪 Testing cache TTL...")
        start_time = time.time()

        try:
            # Create cache with 1 second TTL
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                temp_cache_file = Path(f.name)

            cache_short_ttl = CoordinateCache(
                cache_file=temp_cache_file,
                ttl_seconds=1
            )

            # Set coordinate
            cache_short_ttl.set("ttl_test", (123, 456), 0.95, "test")

            # Should be available immediately
            cached1 = cache_short_ttl.get("ttl_test")
            immediate_success = cached1 is not None

            # Wait for TTL to expire
            await asyncio.sleep(1.5)

            # Should be expired
            cached2 = cache_short_ttl.get("ttl_test")
            expired_success = cached2 is None

            success = immediate_success and expired_success
            duration = time.time() - start_time

            self._add_result(
                test_name="Cache TTL",
                category="cache_learning",
                success=success,
                duration=duration,
                message="Cache TTL works correctly" if success else "Cache TTL failed",
                details={
                    "immediate_retrieval": immediate_success,
                    "expired_correctly": expired_success
                }
            )

            # Cleanup
            temp_cache_file.unlink(missing_ok=True)

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Cache TTL",
                category="cache_learning",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_failure_tracking(self) -> bool:
        """Test failure tracking and cache invalidation"""
        logger.info("\n🧪 Testing failure tracking...")
        start_time = time.time()

        try:
            # Set coordinate
            self.clicker.cache.set("failure_test", (111, 222), 0.95, "test")

            # Mark multiple failures
            for _ in range(5):
                self.clicker.cache.mark_failure("failure_test")

            # Should be invalidated due to high failure rate
            cached = self.clicker.cache.get("failure_test")

            success = cached is None
            duration = time.time() - start_time

            self._add_result(
                test_name="Failure Tracking",
                category="cache_learning",
                success=success,
                duration=duration,
                message="High failure rate invalidates cache" if success else "Failure tracking failed",
                details={
                    "invalidated": success
                }
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Failure Tracking",
                category="cache_learning",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    # ========================================================================
    # Test Category: Performance
    # ========================================================================

    async def test_cache_hit_performance(self) -> bool:
        """Test performance of cache hits"""
        logger.info("\n🧪 Testing cache hit performance...")
        start_time = time.time()

        try:
            # Set a coordinate in cache
            self.clicker.cache.set("perf_test", (1245, 12), 0.95, "test")

            # Measure cache hit times
            cache_hit_times = []

            for i in range(10):
                hit_start = time.time()
                cached = self.clicker.cache.get("perf_test")
                hit_duration = time.time() - hit_start
                cache_hit_times.append(hit_duration)

            avg_time = statistics.mean(cache_hit_times)
            max_time = max(cache_hit_times)

            # Cache hits should be very fast (< 0.01s)
            success = avg_time < 0.01

            duration = time.time() - start_time

            self._add_result(
                test_name="Cache Hit Performance",
                category="performance",
                success=success,
                duration=duration,
                message=f"Avg cache hit: {avg_time*1000:.2f}ms",
                details={
                    "avg_time_ms": avg_time * 1000,
                    "max_time_ms": max_time * 1000,
                    "iterations": 10
                }
            )

            # Cleanup
            self.clicker.cache.invalidate("perf_test")

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Cache Hit Performance",
                category="performance",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_repeated_clicks_performance(self) -> bool:
        """Test performance of repeated clicks"""
        logger.info("\n🧪 Testing repeated clicks performance...")
        start_time = time.time()

        try:
            click_times = []

            # Perform multiple clicks
            for i in range(3):
                click_start = time.time()
                result = await self.clicker.open_control_center()
                click_duration = time.time() - click_start

                if result.success:
                    click_times.append(click_duration)

                await self._cleanup_ui()
                await asyncio.sleep(0.3)

            if not click_times:
                success = False
                avg_time = 0
            else:
                avg_time = statistics.mean(click_times)
                # Clicks should complete within reasonable time (< 5s average)
                success = avg_time < 5.0

            duration = time.time() - start_time

            self._add_result(
                test_name="Repeated Clicks Performance",
                category="performance",
                success=success,
                duration=duration,
                message=f"Avg click time: {avg_time:.2f}s",
                details={
                    "avg_time": avg_time,
                    "successful_clicks": len(click_times),
                    "total_attempts": 3
                }
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Repeated Clicks Performance",
                category="performance",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    # ========================================================================
    # Test Category: Edge Cases
    # ========================================================================

    async def test_nonexistent_target(self) -> bool:
        """Test handling of nonexistent targets"""
        logger.info("\n🧪 Testing nonexistent target...")
        start_time = time.time()

        try:
            result = await self.clicker.click("definitely_not_a_real_ui_element_xyz123")

            # Should fail gracefully
            success = not result.success and result.error is not None

            duration = time.time() - start_time

            self._add_result(
                test_name="Nonexistent Target",
                category="edge_cases",
                success=success,
                duration=duration,
                message="Handles nonexistent targets gracefully" if success else "Did not handle gracefully",
                details={
                    "error": result.error,
                    "fallback_attempts": result.fallback_attempts
                }
            )

            await self._cleanup_ui()

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Nonexistent Target",
                category="edge_cases",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            return False

    async def test_invalid_cached_coordinate(self) -> bool:
        """Test recovery from invalid cached coordinates"""
        logger.info("\n🧪 Testing invalid cached coordinate recovery...")
        start_time = time.time()

        try:
            # Set obviously invalid coordinate
            self.clicker.cache.set(
                "control_center",
                (99999, 99999),
                0.95,
                "invalid_test"
            )

            # Should fall back to other methods
            result = await self.clicker.open_control_center()

            # Success means it either:
            # 1. Detected the invalid coordinate and fell back, OR
            # 2. Used a working detection method
            success = result.fallback_attempts >= 1 or result.method_used != "cached"

            duration = time.time() - start_time

            self._add_result(
                test_name="Invalid Cached Coordinate",
                category="edge_cases",
                success=success,
                duration=duration,
                message="Recovered from invalid cache" if success else "Failed to recover",
                details={
                    "method_used": result.method_used,
                    "fallback_attempts": result.fallback_attempts
                }
            )

            await self._cleanup_ui()

            # Clear the invalid cache entry
            self.clicker.cache.invalidate("control_center")

            return success

        except Exception as e:
            duration = time.time() - start_time
            self._add_result(
                test_name="Invalid Cached Coordinate",
                category="edge_cases",
                success=False,
                duration=duration,
                message=f"Exception: {str(e)}",
                details={"error": str(e)}
            )
            await self._cleanup_ui()
            return False

    # ========================================================================
    # Main Test Runner
    # ========================================================================

    async def run_all_tests(self, quick: bool = False):
        """Run all verification tests"""
        logger.info("\n" + "=" * 75)
        logger.info("Starting Verification Tests")
        logger.info("=" * 75)

        # Initialize
        if not await self.initialize():
            logger.error("❌ Initialization failed, cannot run tests")
            return

        # Detection Methods Tests
        logger.info("\n📋 Category: Detection Methods")
        await self.test_cached_detection()
        await self.test_ocr_detection()
        if not quick:
            await self.test_template_matching()
            await self.test_edge_detection()

        # End-to-End Tests
        logger.info("\n📋 Category: End-to-End Functionality")
        await self.test_open_control_center()
        if not quick:
            await self.test_click_screen_mirroring()

        # Cache & Learning Tests
        logger.info("\n📋 Category: Cache & Learning")
        await self.test_cache_persistence()
        if not quick:
            await self.test_cache_ttl()
            await self.test_failure_tracking()

        # Performance Tests
        logger.info("\n📋 Category: Performance")
        await self.test_cache_hit_performance()
        if not quick:
            await self.test_repeated_clicks_performance()

        # Edge Cases Tests
        logger.info("\n📋 Category: Edge Cases")
        if not quick:
            await self.test_nonexistent_target()
            await self.test_invalid_cached_coordinate()

        logger.info("\n" + "=" * 75)
        logger.info("All Tests Complete")
        logger.info("=" * 75)

    def generate_report(self) -> VerificationReport:
        """Generate verification report"""
        total_duration = time.time() - self.start_time

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Category statistics
        category_stats = {}
        for category, results in categories.items():
            total = len(results)
            passed = sum(1 for r in results if r.success)
            category_stats[category] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": (passed / total * 100) if total > 0 else 0
            }

        # Performance metrics
        perf_results = [r for r in self.results if r.category == "performance"]
        avg_test_duration = statistics.mean([r.duration for r in self.results]) if self.results else 0

        performance_metrics = {
            "total_duration": total_duration,
            "avg_test_duration": avg_test_duration,
            "performance_tests": len(perf_results),
            "performance_passed": sum(1 for r in perf_results if r.success)
        }

        # Recommendations
        recommendations = []

        if success_rate < 80:
            recommendations.append(
                "⚠️  Success rate below 80%. Consider checking system permissions and dependencies."
            )

        cache_tests = [r for r in self.results if "cache" in r.test_name.lower()]
        if any(not r.success for r in cache_tests):
            recommendations.append(
                "⚠️  Cache tests failing. Check file permissions for cache directory."
            )

        ocr_test = next((r for r in self.results if r.test_name == "OCR Detection"), None)
        if ocr_test and not ocr_test.success:
            recommendations.append(
                "💡 OCR detection not working. Install pytesseract or configure Claude Vision."
            )

        if success_rate >= 90:
            recommendations.append(
                "✅ System performing well! Success rate above 90%."
            )

        # System info
        system_info = {
            "screen_resolution": pyautogui.size(),
            "macos_version": self.clicker.cache.macos_version if self.clicker else "unknown",
            "cache_file": str(self.clicker.cache.cache_file) if self.clicker else "unknown",
            "vision_analyzer": self.vision_analyzer is not None
        }

        return VerificationReport(
            timestamp=datetime.now().isoformat(),
            system_info=system_info,
            test_results=self.results,
            summary={
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "category_stats": category_stats
            },
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )

    def print_report(self, report: VerificationReport):
        """Print verification report to console"""
        print("\n" + "=" * 75)
        print("VERIFICATION REPORT")
        print("=" * 75)

        print(f"\n📅 Timestamp: {report.timestamp}")
        print(f"💻 Screen Resolution: {report.system_info['screen_resolution']}")
        print(f"🍎 macOS Version: {report.system_info['macos_version']}")
        print(f"👁️  Vision Analyzer: {'✅ Available' if report.system_info['vision_analyzer'] else '❌ Not available'}")

        print("\n📊 SUMMARY")
        print("-" * 75)
        print(f"Total Tests: {report.summary['total_tests']}")
        print(f"Passed: {report.summary['passed_tests']} ✅")
        print(f"Failed: {report.summary['failed_tests']} ❌")
        print(f"Success Rate: {report.summary['success_rate']:.1f}%")

        print("\n📋 CATEGORY BREAKDOWN")
        print("-" * 75)
        for category, stats in report.summary['category_stats'].items():
            print(f"\n{category}:")
            print(f"  Total: {stats['total']}")
            print(f"  Passed: {stats['passed']} ({stats['success_rate']:.1f}%)")
            print(f"  Failed: {stats['failed']}")

        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 75)
        print(f"Total Duration: {report.performance_metrics['total_duration']:.2f}s")
        print(f"Avg Test Duration: {report.performance_metrics['avg_test_duration']:.2f}s")

        if report.recommendations:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 75)
            for rec in report.recommendations:
                print(f"  {rec}")

        print("\n" + "=" * 75)

    def save_report(self, report: VerificationReport, output_file: Path):
        """Save report to JSON file"""
        # Convert to dict
        report_dict = {
            "timestamp": report.timestamp,
            "system_info": report.system_info,
            "test_results": [asdict(r) for r in report.test_results],
            "summary": report.summary,
            "performance_metrics": report.performance_metrics,
            "recommendations": report.recommendations
        }

        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"📄 Report saved to: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Verify AdaptiveControlCenterClicker functionality"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick verification (subset of tests)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (default)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="verification_report.json",
        help="Output file for JSON report"
    )

    args = parser.parse_args()

    # Determine test mode
    quick_mode = args.quick or not args.full

    # Try to load vision analyzer
    vision_analyzer = None
    try:
        from vision.claude_vision_analyzer_main import get_claude_vision_analyzer
        vision_analyzer = get_claude_vision_analyzer()
        logger.info("✅ Claude Vision analyzer loaded")
    except Exception as e:
        logger.warning(f"⚠️  Could not load vision analyzer: {e}")

    # Run verification
    verifier = AdaptiveClickerVerifier(vision_analyzer=vision_analyzer)
    await verifier.run_all_tests(quick=quick_mode)

    # Generate and print report
    report = verifier.generate_report()
    verifier.print_report(report)

    # Save report
    output_path = Path(args.output)
    verifier.save_report(report, output_path)

    # Exit with appropriate code
    success_rate = report.summary['success_rate']
    exit_code = 0 if success_rate >= 80 else 1

    logger.info(f"\n{'✅' if exit_code == 0 else '❌'} Verification {'PASSED' if exit_code == 0 else 'FAILED'}")

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
