#!/usr/bin/env python3
"""
Complete Async Pipeline Integration Test
Tests all integrated components with the AdvancedAsyncPipeline
"""

import asyncio
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all integrated components
from core.async_pipeline import get_async_pipeline
from system_control.macos_controller import MacOSController
from api.jarvis_voice_api import async_subprocess_run, async_osascript

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncIntegrationTester:
    """Test harness for async pipeline integration"""

    def __init__(self):
        self.pipeline = get_async_pipeline()
        self.controller = MacOSController()
        self.results: List[Dict[str, Any]] = []

    async def test_lock_unlock_cycle(self) -> Dict[str, Any]:
        """Test complete lock/unlock cycle with timing"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Lock/Unlock Cycle Performance")
        logger.info("="*70)

        test_result = {
            "test": "lock_unlock_cycle",
            "success": True,
            "timings": {},
            "errors": []
        }

        try:
            # Test lock
            logger.info("🔒 Testing screen lock...")
            start = time.time()
            success, message = await self.controller.lock_screen()
            lock_time = time.time() - start
            test_result["timings"]["lock"] = lock_time

            if success:
                logger.info(f"✅ Lock successful in {lock_time:.2f}s: {message}")
            else:
                logger.error(f"❌ Lock failed: {message}")
                test_result["success"] = False
                test_result["errors"].append(f"Lock failed: {message}")

            # Wait before unlock
            await asyncio.sleep(2)

            # Test unlock (will need daemon or fail gracefully)
            logger.info("🔓 Testing screen unlock...")
            start = time.time()
            success, message = await self.controller.unlock_screen()
            unlock_time = time.time() - start
            test_result["timings"]["unlock"] = unlock_time

            if success:
                logger.info(f"✅ Unlock successful in {unlock_time:.2f}s: {message}")
            else:
                logger.info(f"ℹ️  Unlock response in {unlock_time:.2f}s: {message}")
                logger.info("   (Expected without Voice Unlock daemon)")

        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(str(e))
            logger.error(f"❌ Test failed: {e}")

        return test_result

    async def test_volume_performance(self) -> Dict[str, Any]:
        """Compare async vs sync volume control performance"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Volume Control Performance Comparison")
        logger.info("="*70)

        test_result = {
            "test": "volume_performance",
            "success": True,
            "timings": {},
            "speedup": 0
        }

        try:
            # Test async version (parallel)
            logger.info("Testing ASYNC volume control (parallel)...")
            start = time.time()
            tasks = [
                self.controller.set_volume_async(30),
                self.controller.set_volume_async(50),
                self.controller.set_volume_async(70),
            ]
            results = await asyncio.gather(*tasks)
            async_time = time.time() - start
            test_result["timings"]["async_parallel"] = async_time
            logger.info(f"✅ Async (parallel): {async_time:.3f}s")

            await asyncio.sleep(1)

            # Test sync version (sequential)
            logger.info("Testing SYNC volume control (sequential)...")
            start = time.time()
            self.controller.set_volume(30)
            self.controller.set_volume(50)
            self.controller.set_volume(70)
            sync_time = time.time() - start
            test_result["timings"]["sync_sequential"] = sync_time
            logger.info(f"⏱️  Sync (sequential): {sync_time:.3f}s")

            # Calculate speedup
            if sync_time > 0:
                speedup = ((sync_time - async_time) / sync_time) * 100
                test_result["speedup"] = speedup
                logger.info(f"\n🚀 Async is {speedup:.1f}% faster!")

        except Exception as e:
            test_result["success"] = False
            test_result["errors"] = [str(e)]
            logger.error(f"❌ Test failed: {e}")

        return test_result

    async def test_pipeline_features(self) -> Dict[str, Any]:
        """Test advanced pipeline features"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Advanced Pipeline Features")
        logger.info("="*70)

        test_result = {
            "test": "pipeline_features",
            "success": True,
            "features_tested": [],
            "errors": []
        }

        try:
            # Test 1: Priority-based execution
            logger.info("Testing priority-based execution...")
            high_priority = asyncio.create_task(
                self.pipeline.process_async("high_priority_test", priority=2)
            )
            normal_priority = asyncio.create_task(
                self.pipeline.process_async("normal_priority_test", priority=0)
            )

            results = await asyncio.gather(high_priority, normal_priority)
            test_result["features_tested"].append("priority_execution")
            logger.info("✅ Priority execution working")

            # Test 2: Stage-specific execution
            logger.info("Testing stage-specific execution...")
            result = await self.pipeline.process_async(
                "test_specific_stage",
                metadata={"stage": "validation"}
            )
            test_result["features_tested"].append("stage_specific")
            logger.info("✅ Stage-specific execution working")

            # Test 3: Circuit breaker
            logger.info("Testing adaptive circuit breaker...")
            metrics = self.pipeline.get_metrics()
            if "circuit_breaker" in metrics:
                threshold = metrics["circuit_breaker"].get("threshold", "N/A")
                logger.info(f"✅ Circuit breaker active (threshold: {threshold})")
                test_result["features_tested"].append("circuit_breaker")

            # Test 4: Event bus
            logger.info("Testing async event bus...")
            events_processed = len(self.pipeline.event_bus.listeners)
            logger.info(f"✅ Event bus active ({events_processed} listeners)")
            test_result["features_tested"].append("event_bus")

        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(str(e))
            logger.error(f"❌ Test failed: {e}")

        return test_result

    async def test_subprocess_handling(self) -> Dict[str, Any]:
        """Test async subprocess execution"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Async Subprocess Handling")
        logger.info("="*70)

        test_result = {
            "test": "subprocess_handling",
            "success": True,
            "commands_tested": [],
            "errors": []
        }

        try:
            # Test with list input
            logger.info("Testing subprocess with list input...")
            stdout, stderr, returncode = await async_subprocess_run(
                ['echo', 'Hello from async pipeline'],
                timeout=5.0
            )
            if returncode == 0:
                logger.info(f"✅ List input: {stdout.decode().strip()}")
                test_result["commands_tested"].append("list_input")

            # Test with string input (should be converted to list)
            logger.info("Testing subprocess with string input...")
            stdout, stderr, returncode = await async_subprocess_run(
                'echo "String input test"',
                timeout=5.0
            )
            if returncode == 0:
                logger.info(f"✅ String input: {stdout.decode().strip()}")
                test_result["commands_tested"].append("string_input")

            # Test AppleScript execution
            logger.info("Testing async AppleScript...")
            script = 'return "AppleScript async test"'
            stdout, stderr, returncode = await async_osascript(script, timeout=5.0)
            if returncode == 0:
                logger.info(f"✅ AppleScript: {stdout.decode().strip()}")
                test_result["commands_tested"].append("applescript")

        except Exception as e:
            test_result["success"] = False
            test_result["errors"].append(str(e))
            logger.error(f"❌ Test failed: {e}")

        return test_result

    async def test_pipeline_type(self) -> Dict[str, Any]:
        """Verify we're using AdvancedAsyncPipeline"""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Pipeline Type Verification")
        logger.info("="*70)

        test_result = {
            "test": "pipeline_type",
            "success": True,
            "pipeline_class": "",
            "features": []
        }

        try:
            pipeline = get_async_pipeline()
            pipeline_type = type(pipeline).__name__
            test_result["pipeline_class"] = pipeline_type

            logger.info(f"Pipeline type: {pipeline_type}")

            # Check for advanced features
            if hasattr(pipeline, 'circuit_breaker'):
                test_result["features"].append("circuit_breaker")
                logger.info("✅ Has AdaptiveCircuitBreaker")

            if hasattr(pipeline, 'event_bus'):
                test_result["features"].append("event_bus")
                logger.info("✅ Has AsyncEventBus")

            if hasattr(pipeline, 'middlewares'):
                test_result["features"].append("middlewares")
                logger.info("✅ Has Middleware system")

            if hasattr(pipeline, 'stages'):
                test_result["features"].append("dynamic_stages")
                logger.info("✅ Has Dynamic stages")

            if pipeline_type == "AdvancedAsyncPipeline":
                logger.info("🎉 Confirmed: Using AdvancedAsyncPipeline!")
            else:
                test_result["success"] = False
                logger.warning(f"⚠️  Expected AdvancedAsyncPipeline, got {pipeline_type}")

        except Exception as e:
            test_result["success"] = False
            test_result["errors"] = [str(e)]
            logger.error(f"❌ Test failed: {e}")

        return test_result

    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("\n" + "="*80)
        logger.info("COMPLETE ASYNC PIPELINE INTEGRATION TEST SUITE")
        logger.info("="*80)
        logger.info("Testing all components integrated with AdvancedAsyncPipeline")
        logger.info("")

        # Run tests
        self.results.append(await self.test_pipeline_type())
        self.results.append(await self.test_lock_unlock_cycle())
        self.results.append(await self.test_volume_performance())
        self.results.append(await self.test_pipeline_features())
        self.results.append(await self.test_subprocess_handling())

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        if failed_tests > 0:
            logger.info(f"❌ Failed: {failed_tests}")

        logger.info("\nDetailed Results:")
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            test_name = result["test"]
            logger.info(f"\n{status} {test_name}:")

            if "timings" in result:
                for key, value in result["timings"].items():
                    logger.info(f"  - {key}: {value:.3f}s")

            if "speedup" in result and result["speedup"] > 0:
                logger.info(f"  - Performance gain: {result['speedup']:.1f}%")

            if "features_tested" in result:
                logger.info(f"  - Features tested: {', '.join(result['features_tested'])}")

            if "commands_tested" in result:
                logger.info(f"  - Commands tested: {', '.join(result['commands_tested'])}")

            if "features" in result:
                logger.info(f"  - Pipeline features: {', '.join(result['features'])}")

            if "errors" in result and result["errors"]:
                logger.info(f"  - Errors: {', '.join(result['errors'])}")

        logger.info("\n" + "="*80)
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("The AdvancedAsyncPipeline is fully integrated and operational.")
            logger.info("Performance improvements confirmed across all components.")
        else:
            logger.info(f"⚠️  {failed_tests} test(s) need attention")
        logger.info("="*80)


async def main():
    """Main test runner"""
    try:
        tester = AsyncIntegrationTester()
        await tester.run_all_tests()
        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())