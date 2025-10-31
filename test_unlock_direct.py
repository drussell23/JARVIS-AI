#!/usr/bin/env python3
"""
Direct test of voice unlock - simulates the actual command flow
"""

import asyncio
import sys
import os
import base64
import numpy as np
sys.path.append('backend')

from voice_unlock.intelligent_voice_unlock_service import IntelligentVoiceUnlockService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_unlock():
    """Test the voice unlock with simulated audio saying 'unlock my screen'"""

    logger.info("\n" + "="*60)
    logger.info("TESTING VOICE UNLOCK: 'unlock my screen'")
    logger.info("="*60 + "\n")

    try:
        # Initialize service
        logger.info("🔧 Initializing Voice Unlock Service...")
        service = IntelligentVoiceUnlockService()
        await service.initialize()
        logger.info("✅ Service initialized\n")

        # Create test audio that will trigger Whisper to return "unlock my screen"
        # We'll use a simple approach: create valid audio format that Whisper will process
        sample_rate = 16000
        duration = 2  # 2 seconds

        # Generate a simple audio signal
        t = np.linspace(0, duration, sample_rate * duration)

        # Create a more speech-like pattern
        # Mix multiple frequencies to simulate voice
        audio = np.zeros_like(t)

        # Add fundamental frequency and harmonics (typical for male voice)
        fundamental = 120  # Hz - typical male voice fundamental
        for harmonic in range(1, 8):
            freq = fundamental * harmonic
            # Decrease amplitude for higher harmonics
            amplitude = 0.1 / harmonic
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Add amplitude envelope to simulate speech rhythm
        envelope = np.ones_like(t)
        # Create gaps to simulate word boundaries
        envelope[8000:8500] *= 0.1   # Gap between words
        envelope[16000:16500] *= 0.1  # Gap between words
        audio *= envelope

        # Convert to int16 PCM
        audio = np.clip(audio, -1, 1)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        logger.info(f"🎤 Created test audio: {len(audio_bytes)} bytes")
        logger.info("   (Note: This is synthetic audio - transcription may not be perfect)\n")

        # Process the command
        logger.info("🔓 Processing 'unlock my screen' command...")
        result = await service.process_voice_unlock_command(audio_bytes)

        # Display results
        logger.info("\n📊 RESULTS:")
        logger.info("-" * 40)

        if result.get('transcription'):
            logger.info(f"Transcription: '{result['transcription']}'")
        else:
            logger.info("Transcription: [No transcription]")

        if result.get('command'):
            logger.info(f"Command: {result['command']}")

        if result.get('action'):
            logger.info(f"Action: {result['action']}")

        logger.info(f"Success: {result.get('success', False)}")

        if result.get('message'):
            logger.info(f"Message: {result['message']}")

        # Check verification results
        if 'verification' in result:
            ver = result['verification']
            logger.info("\n🔐 VOICE VERIFICATION:")
            logger.info("-" * 40)
            logger.info(f"Verified: {ver.get('verified', False)}")
            logger.info(f"Speaker: {ver.get('speaker_name', 'Unknown')}")
            logger.info(f"Confidence: {ver.get('confidence', 0.0):.2%}")
            logger.info(f"Is Owner: {ver.get('is_owner', False)}")

        # Analysis
        logger.info("\n📈 ANALYSIS:")
        logger.info("-" * 40)

        if result.get('transcription') == '[transcription failed]':
            logger.warning("⚠️ Transcription is still failing")
            logger.info("   The audio format fixes may need more work")
        elif result.get('transcription'):
            logger.info("✅ Transcription is working!")
            logger.info(f"   Got: '{result['transcription']}'")

        if result.get('verification', {}).get('confidence', 0) > 0:
            logger.info("✅ Speaker verification is working!")
        else:
            logger.info("⚠️ Speaker verification confidence is 0")
            logger.info("   This is expected with synthetic test audio")
            logger.info("   Real voice audio should work better")

        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETE")
        logger.info("="*60)

        return result

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_unlock())

    if result and result.get('success'):
        logger.info("\n🎉 SUCCESS: Voice unlock is working!")
    else:
        logger.info("\n💡 Next steps:")
        logger.info("   1. Test with real voice audio")
        logger.info("   2. Check that Whisper models are loaded")
        logger.info("   3. Verify speaker profiles are in database")