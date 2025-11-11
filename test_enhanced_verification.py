#!/usr/bin/env python3
"""
Test the enhanced speaker verification system with adaptive learning
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from voice.speaker_verification_service import get_speaker_verification_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_with_calibration():
    """Test the enhanced verification system with calibration mode"""

    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING ENHANCED SPEAKER VERIFICATION WITH ADAPTIVE LEARNING")
    logger.info("="*80)

    # Initialize the service
    service = await get_speaker_verification_service()

    # Get the primary user
    primary_user = "Derek J. Russell"

    logger.info(f"\n📋 Testing with speaker: {primary_user}")

    # Enable calibration mode manually
    logger.info("\n🎯 Enabling calibration mode...")
    result = await service.enable_calibration_mode(primary_user)
    logger.info(f"   Result: {result}")

    # Load test audio samples
    audio_dir = Path("/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/voice/audio_samples")

    if not audio_dir.exists():
        logger.error(f"❌ Audio samples directory not found: {audio_dir}")
        return

    # Get audio files
    audio_files = list(audio_dir.glob("derek_*.wav"))[:5]  # Test with first 5 files

    if not audio_files:
        logger.error("❌ No audio files found")
        return

    logger.info(f"\n📂 Found {len(audio_files)} test audio files")

    # Test each audio file
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n🎤 Test #{i}: {audio_file.name}")

        try:
            # Read audio data
            with open(audio_file, 'rb') as f:
                audio_data = f.read()

            # Skip WAV header (44 bytes) and get raw PCM
            audio_data = audio_data[44:]

            # Verify speaker
            result = await service.verify_speaker(
                audio_data=audio_data,
                speaker_name=primary_user
            )

            # Display results
            logger.info(f"   ✅ Verified: {result['verified']}")
            logger.info(f"   📊 Confidence: {result['confidence']:.2%}")
            logger.info(f"   🎯 Threshold: {result.get('adaptive_threshold', 'N/A')}")

            if 'suggestion' in result:
                logger.info(f"   💡 Suggestion: {result['suggestion']}")

            # Small delay between tests
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"   ❌ Error: {e}")

    # Show calibration status
    if service.calibration_mode:
        logger.info("\n🔄 Calibration still in progress...")
        logger.info(f"   Samples collected: {len(service.calibration_samples)}")
    else:
        logger.info("\n✅ Calibration completed!")

    # Show verification history
    if primary_user in service.verification_history:
        history = service.verification_history[primary_user]
        recent = history[-5:]
        avg_confidence = np.mean([h['confidence'] for h in recent])
        success_rate = sum(1 for h in recent if h['verified']) / len(recent) * 100

        logger.info(f"\n📊 Verification Statistics:")
        logger.info(f"   Recent attempts: {len(recent)}")
        logger.info(f"   Average confidence: {avg_confidence:.2%}")
        logger.info(f"   Success rate: {success_rate:.1f}%")

    # Test with live simulation (simulating JARVIS sending data)
    logger.info("\n🎮 Simulating JARVIS live audio (int16 PCM)...")

    # Create simulated int16 PCM audio
    if audio_files:
        with open(audio_files[0], 'rb') as f:
            wav_data = f.read()[44:]  # Skip header

        # Convert to int16 array and back to bytes (simulating JARVIS format)
        audio_array = np.frombuffer(wav_data, dtype=np.int16)
        simulated_jarvis_audio = audio_array.tobytes()

        result = await service.verify_speaker(
            audio_data=simulated_jarvis_audio,
            speaker_name=primary_user
        )

        logger.info(f"   JARVIS simulation result:")
        logger.info(f"   ✅ Verified: {result['verified']}")
        logger.info(f"   📊 Confidence: {result['confidence']:.2%}")
        logger.info(f"   🎯 Adaptive threshold: {result.get('adaptive_threshold', 'N/A')}")

    logger.info("\n" + "="*80)
    logger.info("✅ ENHANCED VERIFICATION TEST COMPLETE")
    logger.info("="*80)

    # Cleanup
    await service.cleanup()


if __name__ == "__main__":
    asyncio.run(test_with_calibration())