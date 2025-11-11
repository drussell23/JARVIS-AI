#!/usr/bin/env python3
"""Test voice verification with actual JARVIS setup."""

import asyncio
import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_verification():
    """Test the actual verification as JARVIS would do it."""

    print("\n" + "="*80)
    print("TESTING LIVE VOICE VERIFICATION")
    print("="*80)

    # Initialize the speaker verification service exactly as JARVIS does
    from backend.voice.speaker_verification_service import SpeakerVerificationService

    print("\n1️⃣  INITIALIZING SERVICE:")
    print("-" * 40)

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n✅ Service initialized:")
    print(f"   Model dimension: {service.current_model_dimension}")
    print(f"   Verification threshold: {service.verification_threshold}")
    print(f"   Loaded profiles: {len(service.speaker_profiles)}")

    # Check what profiles are loaded
    print("\n2️⃣  LOADED PROFILES:")
    print("-" * 40)

    if service.speaker_profiles:
        for name, profile in service.speaker_profiles.items():
            print(f"\n📌 Profile: {name}")
            print(f"   ID: {profile.get('speaker_id')}")
            print(f"   Is Primary: {profile.get('is_primary_user')}")

            # Check embedding
            if 'embedding' in profile:
                emb = profile['embedding']
                if isinstance(emb, np.ndarray):
                    print(f"   Embedding: {emb.shape} {emb.dtype}")
                    print(f"   Embedding norm: {np.linalg.norm(emb):.4f}")
                else:
                    print(f"   Embedding type: {type(emb)}")
            else:
                print("   ❌ No embedding!")

            # Check acoustic features
            if 'acoustic_features' in profile:
                af = profile['acoustic_features']
                if af and isinstance(af, dict):
                    print(f"   Acoustic features: {len(af)} parameters")
                    if 'pitch_mean' in af:
                        print(f"   Pitch: {af['pitch_mean']:.1f} Hz")
                else:
                    print(f"   Acoustic features: {af}")
            else:
                print("   ❌ No acoustic features")

            print(f"   Threshold: {profile.get('threshold', 'default')}")
            print(f"   Total samples: {profile.get('total_samples', 0)}")
    else:
        print("❌ No profiles loaded!")

    # Test with random audio to see what happens
    print("\n3️⃣  TESTING WITH RANDOM AUDIO:")
    print("-" * 40)

    # Create test audio (16kHz, 1 second)
    test_audio = np.random.randn(16000).astype(np.float32)
    # Convert to bytes as JARVIS would
    test_audio_bytes = test_audio.tobytes()

    print("\n Testing verification...")
    result = await service.verify_speaker(test_audio_bytes)

    print(f"\n Result:")
    print(f"   Verified: {result.get('verified', False)}")
    print(f"   Confidence: {result.get('confidence', 0):.4f}")
    print(f"   Speaker: {result.get('speaker_name', 'None')}")
    print(f"   Is Owner: {result.get('is_owner', False)}")

    # Check why confidence might be 0
    print("\n4️⃣  DIAGNOSTICS:")
    print("-" * 40)

    if result.get('confidence', 0) == 0:
        print("\n⚠️  Confidence is 0%, possible reasons:")

        if len(service.speaker_profiles) == 0:
            print("   ❌ No profiles loaded")

        if service.current_model_dimension == 192:
            print("   ✅ Model dimension correct (192D)")
        else:
            print(f"   ❌ Model dimension mismatch: {service.current_model_dimension}D")

        # Check if embeddings have correct shape
        for name, profile in service.speaker_profiles.items():
            if 'embedding' in profile:
                emb = profile['embedding']
                if isinstance(emb, np.ndarray):
                    if emb.shape[0] != service.current_model_dimension:
                        print(f"   ❌ {name} embedding dimension mismatch: {emb.shape[0]}D vs {service.current_model_dimension}D")
                    else:
                        print(f"   ✅ {name} embedding dimension matches")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_verification())