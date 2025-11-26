#!/usr/bin/env python3
"""
Voice Profile Diagnostics Script

This script checks the health of voice profiles in the database and identifies
any issues that could cause 0% confidence during voice verification.

Run with: python backend/scripts/check_voice_profiles.py
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def check_profiles():
    """Check all voice profiles for issues"""
    print("\n" + "=" * 80)
    print("🔍 JARVIS Voice Profile Diagnostics")
    print("=" * 80)

    try:
        from intelligence.learning_database import get_learning_database

        # Get database instance
        db = await get_learning_database()

        if not db or not db._initialized:
            print("❌ ERROR: Learning database not initialized")
            return

        print(f"✅ Database connected: {type(db).__name__}")

        # Get all speaker profiles
        profiles = await db.get_all_speaker_profiles()
        print(f"\n📊 Found {len(profiles)} profile(s) in database\n")

        if not profiles:
            print("⚠️  WARNING: No voice profiles found!")
            print("   You need to enroll a voice profile first.")
            print("   Use the voice enrollment feature or run create_derek_profile_final.py")
            return

        # Analyze each profile
        issues_found = 0

        for i, profile in enumerate(profiles, 1):
            speaker_name = profile.get("speaker_name", "Unknown")
            speaker_id = profile.get("speaker_id", "N/A")
            total_samples = profile.get("total_samples", 0)
            embedding_dim = profile.get("embedding_dimension", "unknown")
            created_at = profile.get("created_at", "unknown")

            print(f"\n{'─' * 60}")
            print(f"Profile #{i}: {speaker_name} (ID: {speaker_id})")
            print(f"{'─' * 60}")
            print(f"  Created: {created_at}")
            print(f"  Total samples: {total_samples}")
            print(f"  Embedding dimension (stored): {embedding_dim}")

            # Check voiceprint embedding
            embedding_bytes = profile.get("voiceprint_embedding")

            if not embedding_bytes:
                print(f"  ❌ CRITICAL: No voiceprint embedding!")
                print(f"     This profile has NO embedding data - verification will ALWAYS fail")
                print(f"     Solution: Re-enroll this speaker's voice profile")
                issues_found += 1
                continue

            print(f"  Embedding bytes: {len(embedding_bytes)} bytes")

            # Try to deserialize
            try:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                actual_dim = embedding.shape[0]
                norm = np.linalg.norm(embedding)
                mean_val = embedding.mean()
                min_val = embedding.min()
                max_val = embedding.max()
                std_val = embedding.std()

                print(f"  Actual dimension: {actual_dim}")
                print(f"  Embedding norm: {norm:.6f}")
                print(f"  Stats: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

                # Check for issues
                if actual_dim == 0:
                    print(f"  ❌ CRITICAL: Embedding is EMPTY (0 dimensions)")
                    print(f"     This profile will ALWAYS fail verification")
                    issues_found += 1
                elif norm == 0 or norm < 1e-6:
                    print(f"  ❌ CRITICAL: Embedding has ZERO NORM ({norm:.10f})")
                    print(f"     This profile will ALWAYS fail with 0% confidence")
                    print(f"     All values are approximately zero - data is corrupted")
                    issues_found += 1
                elif norm < 0.1:
                    print(f"  ⚠️  WARNING: Embedding norm is very low ({norm:.6f})")
                    print(f"     This may cause unreliable verification")
                    issues_found += 1
                elif actual_dim not in [192, 256, 512, 768]:
                    print(f"  ⚠️  WARNING: Unusual embedding dimension ({actual_dim})")
                    print(f"     Expected: 192 (ECAPA-TDNN), 256, 512, or 768")
                    issues_found += 1
                else:
                    print(f"  ✅ Embedding appears VALID")

                # Check for NaN or Inf
                if np.any(np.isnan(embedding)):
                    print(f"  ❌ CRITICAL: Embedding contains NaN values!")
                    issues_found += 1
                if np.any(np.isinf(embedding)):
                    print(f"  ❌ CRITICAL: Embedding contains Inf values!")
                    issues_found += 1

            except Exception as e:
                print(f"  ❌ ERROR: Failed to deserialize embedding: {e}")
                issues_found += 1

            # Check acoustic features
            acoustic_features = {
                "pitch_mean_hz": profile.get("pitch_mean_hz"),
                "formant_f1_hz": profile.get("formant_f1_hz"),
                "formant_f2_hz": profile.get("formant_f2_hz"),
            }

            has_acoustic = any(v is not None for v in acoustic_features.values())
            if has_acoustic:
                print(f"  ✅ Has acoustic features:")
                for k, v in acoustic_features.items():
                    if v is not None:
                        print(f"     {k}: {v}")
            else:
                print(f"  ⚠️  No acoustic features stored (legacy profile)")

        # Summary
        print(f"\n{'=' * 80}")
        print("📋 SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total profiles: {len(profiles)}")
        print(f"Issues found: {issues_found}")

        if issues_found > 0:
            print(f"\n❌ {issues_found} issue(s) detected that may cause 0% confidence!")
            print("\nRecommended actions:")
            print("  1. Re-enroll affected voice profiles")
            print("  2. Check audio recording equipment")
            print("  3. Ensure microphone permissions are granted")
        else:
            print(f"\n✅ All profiles appear healthy!")

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_profiles())
