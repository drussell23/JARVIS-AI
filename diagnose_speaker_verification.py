#!/usr/bin/env python3
"""
Diagnose Speaker Verification Issues
This script checks the entire speaker verification pipeline to find where it's failing
"""

import asyncio
import numpy as np
import sys
import os
sys.path.append('backend')

from voice.speaker_verification_service import SpeakerVerificationService
from voice.engines.speechbrain_engine import SpeechBrainEngine
from voice.stt_config import STTConfig
from intelligence.learning_database import JARVISLearningDatabase
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def diagnose():
    """Comprehensive diagnostic of speaker verification"""

    logger.info("="*60)
    logger.info("SPEAKER VERIFICATION DIAGNOSTIC")
    logger.info("="*60)

    try:
        # Step 1: Check database and profiles
        logger.info("\n🔍 STEP 1: Checking Database and Profiles...")
        db = JARVISLearningDatabase()
        await db.initialize()

        profiles = await db.get_all_speaker_profiles()
        logger.info(f"Found {len(profiles)} speaker profile(s) in database")

        for profile in profiles:
            logger.info(f"\n📋 Profile: {profile['speaker_name']}")
            logger.info(f"   Speaker ID: {profile['speaker_id']}")
            logger.info(f"   Primary User: {profile.get('is_primary_user', False)}")
            logger.info(f"   Security Level: {profile.get('security_level', 'standard')}")
            logger.info(f"   Total Samples: {profile.get('total_samples', 0)}")

            # Check embedding
            embedding_bytes = profile.get('voiceprint_embedding')
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                logger.info(f"   Embedding shape: {embedding.shape}")
                logger.info(f"   Embedding dtype: {embedding.dtype}")
                logger.info(f"   Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
                logger.info(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
                logger.info(f"   Is normalized: {np.abs(np.linalg.norm(embedding) - 1.0) < 0.01}")
            else:
                logger.error(f"   ❌ No embedding found!")

        if len(profiles) == 0:
            logger.error("\n❌ PROBLEM: No speaker profiles found in database!")
            logger.info("   You need to enroll your voice first")
            logger.info("   Run: python3 backend/voice/enroll_voice.py")
            return

        # Step 2: Initialize SpeechBrain engine
        logger.info("\n🔍 STEP 2: Initializing SpeechBrain Engine...")

        config = STTConfig()
        model_config = config.models.get('speechbrain-wav2vec2')

        if not model_config:
            logger.error("❌ SpeechBrain model config not found!")
            return

        engine = SpeechBrainEngine(model_config)
        await engine.initialize()
        logger.info("✅ SpeechBrain engine initialized")

        # Step 3: Test embedding extraction
        logger.info("\n🔍 STEP 3: Testing Speaker Embedding Extraction...")

        # Create test audio (synthetic)
        sample_rate = 16000
        duration = 3
        t = np.linspace(0, duration, sample_rate * duration)

        # Generate realistic voice-like audio
        audio = np.zeros_like(t)
        fundamental = 120  # Hz
        for harmonic in range(1, 8):
            freq = fundamental * harmonic
            amplitude = 0.1 / harmonic
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Add envelope
        envelope = np.ones_like(t)
        envelope[8000:8500] *= 0.1
        envelope[16000:16500] *= 0.1
        audio *= envelope

        # Convert to bytes
        audio = np.clip(audio, -1, 1)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        logger.info(f"Created test audio: {len(audio_bytes)} bytes")

        try:
            test_embedding = await engine.extract_speaker_embedding(audio_bytes)
            logger.info(f"✅ Embedding extracted successfully!")
            logger.info(f"   Shape: {test_embedding.shape}")
            logger.info(f"   Dtype: {test_embedding.dtype}")
            logger.info(f"   Range: [{test_embedding.min():.4f}, {test_embedding.max():.4f}]")
            logger.info(f"   Norm: {np.linalg.norm(test_embedding):.4f}")
        except Exception as e:
            logger.error(f"❌ Embedding extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Step 4: Test similarity computation
        logger.info("\n🔍 STEP 4: Testing Similarity Computation...")

        # Get first profile's embedding
        first_profile = profiles[0]
        stored_embedding = np.frombuffer(
            first_profile['voiceprint_embedding'],
            dtype=np.float64
        )

        logger.info(f"Comparing test embedding with stored profile: {first_profile['speaker_name']}")

        # Test cosine similarity
        similarity = engine._compute_cosine_similarity(test_embedding, stored_embedding)
        logger.info(f"Cosine similarity: {similarity:.4f} ({similarity*100:.2f}%)")

        # Also compute raw cosine similarity (without normalization)
        test_flat = test_embedding.flatten()
        stored_flat = stored_embedding.flatten()

        logger.info(f"\nDetailed similarity analysis:")
        logger.info(f"   Test embedding shape: {test_flat.shape}")
        logger.info(f"   Stored embedding shape: {stored_flat.shape}")

        if test_flat.shape == stored_flat.shape:
            dot_product = np.dot(test_flat, stored_flat)
            norm1 = np.linalg.norm(test_flat)
            norm2 = np.linalg.norm(stored_flat)
            raw_similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

            logger.info(f"   Dot product: {dot_product:.4f}")
            logger.info(f"   Test norm: {norm1:.4f}")
            logger.info(f"   Stored norm: {norm2:.4f}")
            logger.info(f"   Raw cosine similarity: {raw_similarity:.4f}")
            logger.info(f"   Mapped similarity [0,1]: {(raw_similarity + 1) / 2:.4f}")
        else:
            logger.error(f"   ❌ Shape mismatch! Cannot compare embeddings")
            logger.info(f"      This is likely the problem!")
            logger.info(f"      Expected: {stored_flat.shape}")
            logger.info(f"      Got: {test_flat.shape}")

        # Step 5: Test full verification
        logger.info("\n🔍 STEP 5: Testing Full Speaker Verification...")

        try:
            is_verified, confidence = await engine.verify_speaker(
                audio_bytes,
                stored_embedding,
                threshold=0.25
            )

            logger.info(f"Verification result:")
            logger.info(f"   Verified: {is_verified}")
            logger.info(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            logger.info(f"   Threshold: 0.25 (25%)")

            if confidence == 0.0:
                logger.error(f"\n❌ PROBLEM FOUND: Confidence is 0!")
                logger.info("   This indicates embedding extraction or comparison is failing")
            elif confidence < 0.25:
                logger.warning(f"\n⚠️ Confidence below threshold")
                logger.info("   This is expected with synthetic test audio")
                logger.info("   Real voice audio should produce higher confidence")
            else:
                logger.info(f"\n✅ Confidence above threshold - verification would succeed!")

        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("\n" + "="*60)
        logger.info("DIAGNOSTIC COMPLETE")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose())