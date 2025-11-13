#!/usr/bin/env python3
"""
Voice Pipeline Testing & Diagnostics
====================================

Comprehensive testing utility for the enhanced voice pipeline.

Tests:
- VAD filtering (WebRTC + Silero)
- Audio windowing (5s global, 2s unlock, 3s command)
- Streaming safeguard (command detection)
- End-to-end unlock flow
- Performance metrics

Usage:
    # Run all tests
    python -m voice.test_voice_pipeline

    # Run specific test
    python -m voice.test_voice_pipeline --test vad

    # Run with verbose logging
    python -m voice.test_voice_pipeline --verbose

    # Test with custom audio file
    python -m voice.test_voice_pipeline --audio path/to/audio.wav
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoicePipelineTester:
    """Comprehensive voice pipeline testing utility"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }

    async def run_all_tests(self):
        """Run all voice pipeline tests"""
        logger.info("🧪 Starting Voice Pipeline Test Suite")
        logger.info("=" * 60)

        tests = [
            ("VAD Availability", self.test_vad_availability),
            ("VAD Filtering", self.test_vad_filtering),
            ("Audio Windowing", self.test_audio_windowing),
            ("Streaming Safeguard", self.test_streaming_safeguard),
            ("Unified API", self.test_unified_api),
            ("Performance Benchmarks", self.test_performance),
        ]

        for test_name, test_func in tests:
            await self._run_test(test_name, test_func)

        self._print_summary()

    async def _run_test(self, name: str, test_func):
        """Run a single test and track results"""
        self.results["tests_run"] += 1

        logger.info(f"\n📝 Test: {name}")
        logger.info("-" * 60)

        try:
            start = time.time()
            await test_func()
            duration = (time.time() - start) * 1000

            self.results["tests_passed"] += 1
            logger.info(f"✅ PASSED ({duration:.1f}ms)")

        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append({"test": name, "error": str(e)})
            logger.error(f"❌ FAILED: {e}")
            if self.verbose:
                logger.exception("Full traceback:")

    async def test_vad_availability(self):
        """Test VAD components are available and loadable"""
        from voice.vad.pipeline import get_vad_pipeline, VADPipelineConfig

        # Test configuration
        config = VADPipelineConfig()
        logger.info(f"VAD Config: primary={config.primary_vad}, secondary={config.secondary_vad}")

        # Test pipeline creation
        vad_pipeline = get_vad_pipeline()
        assert vad_pipeline is not None, "VAD pipeline creation failed"

        logger.info(f"✓ VAD Pipeline created successfully")
        logger.info(f"  - Primary: {vad_pipeline.config.primary_vad}")
        logger.info(f"  - Secondary: {vad_pipeline.config.secondary_vad}")
        logger.info(f"  - Strategy: {vad_pipeline.config.combination_strategy}")

    async def test_vad_filtering(self):
        """Test VAD filtering on synthetic audio"""
        from voice.vad.pipeline import get_vad_pipeline

        # Create test audio: 10 seconds with speech + silence pattern
        sample_rate = 16000
        duration_seconds = 10

        # Simulate speech (1-2s) + silence (3s) + speech (4-5s) + silence (rest)
        audio = np.zeros(sample_rate * duration_seconds, dtype=np.float32)

        # Add "speech" (noise) to specific segments
        audio[sample_rate * 1:sample_rate * 2] = np.random.randn(sample_rate).astype(np.float32) * 0.3  # 1-2s
        audio[sample_rate * 4:sample_rate * 5] = np.random.randn(sample_rate).astype(np.float32) * 0.3  # 4-5s

        original_duration = len(audio) / sample_rate
        logger.info(f"Input audio: {original_duration}s")

        # Run VAD filtering
        vad_pipeline = get_vad_pipeline()
        filtered_audio = await vad_pipeline.filter_audio_async(audio)

        filtered_duration = len(filtered_audio) / sample_rate
        reduction = ((original_duration - filtered_duration) / original_duration) * 100

        logger.info(f"✓ VAD filtering complete")
        logger.info(f"  - Original: {original_duration:.1f}s")
        logger.info(f"  - Filtered: {filtered_duration:.1f}s")
        logger.info(f"  - Reduction: {reduction:.1f}%")

        # Expect significant reduction (should filter ~60% silence)
        assert reduction > 40, f"Expected >40% reduction, got {reduction:.1f}%"

    async def test_audio_windowing(self):
        """Test audio windowing limits"""
        from voice.audio_windowing import get_window_manager

        window_manager = get_window_manager()

        # Test 1: Global truncation (5s)
        audio_60s = np.random.randn(16000 * 60).astype(np.float32)
        truncated = window_manager.truncate_to_max(audio_60s)
        duration = len(truncated) / 16000

        logger.info(f"✓ Global truncation: 60s → {duration:.1f}s")
        assert duration <= 5.1, f"Expected ≤5s, got {duration:.1f}s"

        # Test 2: Unlock truncation (2s)
        truncated = window_manager.truncate_for_unlock(audio_60s)
        duration = len(truncated) / 16000

        logger.info(f"✓ Unlock truncation: 60s → {duration:.1f}s")
        assert duration <= 2.1, f"Expected ≤2s, got {duration:.1f}s"

        # Test 3: Command truncation (3s)
        truncated = window_manager.truncate_for_command(audio_60s)
        duration = len(truncated) / 16000

        logger.info(f"✓ Command truncation: 60s → {duration:.1f}s")
        assert duration <= 3.1, f"Expected ≤3s, got {duration:.1f}s"

        # Test 4: Mode-aware preparation
        prepared = window_manager.prepare_for_transcription(audio_60s, mode='unlock')
        duration = len(prepared) / 16000

        logger.info(f"✓ Mode-aware preparation (unlock): {duration:.1f}s")
        assert duration <= 2.1, f"Expected ≤2s, got {duration:.1f}s"

    async def test_streaming_safeguard(self):
        """Test command detection in streaming safeguard"""
        from voice.streaming_safeguard import (
            StreamingSafeguard,
            CommandDetectionConfig,
            CommandMatchStrategy
        )

        # Test different matching strategies
        strategies = [
            (CommandMatchStrategy.EXACT, ["unlock", "UNLOCK"], ["unlocking", "unlock screen"]),
            (CommandMatchStrategy.CONTAINS, ["unlock screen", "UNLOCK MY SCREEN"], ["screen"]),
            (CommandMatchStrategy.WORD_BOUNDARY, ["unlock", "lock screen"], ["unlocking", "locks"]),
        ]

        for strategy, should_match, should_not_match in strategies:
            config = CommandDetectionConfig(
                target_commands=["unlock", "lock"],
                match_strategy=strategy,
                detection_cooldown=0.0  # Disable cooldown for testing
            )
            safeguard = StreamingSafeguard(config)

            logger.info(f"Testing strategy: {strategy.value}")

            # Test positive matches
            for text in should_match:
                event = await safeguard.check_transcription(text)
                assert event is not None, f"Expected match for '{text}' with {strategy.value}"
                logger.info(f"  ✓ Matched: '{text}' → '{event.command}'")
                # Reset cooldown for next test
                safeguard.reset()

            # Test negative matches
            for text in should_not_match:
                event = await safeguard.check_transcription(text)
                assert event is None, f"Expected NO match for '{text}' with {strategy.value}"
                logger.info(f"  ✓ Ignored: '{text}'")

    async def test_unified_api(self):
        """Test unified VAD API function"""
        from voice.unified_vad_api import run_vad_pipeline_numpy

        # Create test audio (10 seconds)
        audio = np.random.randn(16000 * 10).astype(np.float32)

        # Test unlock mode
        clean_audio = await run_vad_pipeline_numpy(
            audio,
            max_seconds=2.0,
            mode='unlock',
            enable_vad=True,
            enable_windowing=True
        )

        duration = len(clean_audio) / 16000
        logger.info(f"✓ Unlock mode: 10s → {duration:.1f}s")
        assert duration <= 2.1, f"Expected ≤2s, got {duration:.1f}s"

        # Test command mode
        clean_audio = await run_vad_pipeline_numpy(
            audio,
            max_seconds=3.0,
            mode='command'
        )

        duration = len(clean_audio) / 16000
        logger.info(f"✓ Command mode: 10s → {duration:.1f}s")
        assert duration <= 3.1, f"Expected ≤3s, got {duration:.1f}s"

        # Test dictation mode
        clean_audio = await run_vad_pipeline_numpy(
            audio,
            max_seconds=5.0,
            mode='dictation'
        )

        duration = len(clean_audio) / 16000
        logger.info(f"✓ Dictation mode: 10s → {duration:.1f}s")
        assert duration <= 5.1, f"Expected ≤5s, got {duration:.1f}s"

    async def test_performance(self):
        """Benchmark pipeline performance"""
        from voice.unified_vad_api import run_vad_pipeline_numpy

        # Test different audio durations
        durations = [5, 10, 30, 60]

        logger.info("Performance benchmarks:")

        for duration in durations:
            audio = np.random.randn(16000 * duration).astype(np.float32)

            start = time.time()
            clean_audio = await run_vad_pipeline_numpy(
                audio,
                max_seconds=2.0,
                mode='unlock',
                enable_vad=True,
                enable_windowing=True
            )
            processing_time = (time.time() - start) * 1000

            output_duration = len(clean_audio) / 16000
            real_time_factor = (processing_time / 1000) / duration

            logger.info(f"  {duration}s audio:")
            logger.info(f"    - Processing time: {processing_time:.1f}ms")
            logger.info(f"    - Output: {output_duration:.1f}s")
            logger.info(f"    - RT factor: {real_time_factor:.3f}x")

            # Ensure real-time performance (processing should be faster than input)
            assert real_time_factor < 1.0, f"Expected RT factor <1.0, got {real_time_factor:.3f}x"

    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)

        total = self.results["tests_run"]
        passed = self.results["tests_passed"]
        failed = self.results["tests_failed"]
        pass_rate = (passed / total * 100) if total > 0 else 0

        logger.info(f"Total Tests:  {total}")
        logger.info(f"Passed:       {passed} ✅")
        logger.info(f"Failed:       {failed} ❌")
        logger.info(f"Pass Rate:    {pass_rate:.1f}%")

        if failed > 0:
            logger.info("\n❌ FAILED TESTS:")
            for error in self.results["errors"]:
                logger.error(f"  - {error['test']}: {error['error']}")

        logger.info("=" * 60)

        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error(f"⚠️  {failed} TEST(S) FAILED")

        return failed == 0


async def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Pipeline Testing Utility")
    parser.add_argument('--test', help="Specific test to run")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    parser.add_argument('--audio', help="Path to audio file for testing")

    args = parser.parse_args()

    tester = VoicePipelineTester(verbose=args.verbose)

    if args.test:
        # Run specific test
        test_method = f"test_{args.test}"
        if hasattr(tester, test_method):
            await tester._run_test(args.test.title(), getattr(tester, test_method))
        else:
            logger.error(f"Unknown test: {args.test}")
            logger.info("Available tests: vad_availability, vad_filtering, audio_windowing, streaming_safeguard, unified_api, performance")
            sys.exit(1)
    else:
        # Run all tests
        await tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if tester.results["tests_failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
