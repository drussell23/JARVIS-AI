#!/usr/bin/env python3
"""
Fix Corrupted Speaker Profile Script

This script fixes profiles with corrupted embeddings (NaN values) by:
1. Finding all voice samples associated with the profile
2. Re-extracting embeddings from those samples
3. Averaging valid embeddings to create a new voiceprint
4. Updating the profile in the database

Usage:
    python backend/scripts/fix_corrupted_profile.py [speaker_name]

If no speaker_name is provided, it will fix all corrupted profiles.
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


async def fix_profile(db, speaker_name: str = None):
    """Fix corrupted speaker profile by re-generating embedding from voice samples."""
    print(f"\n{'=' * 80}")
    print(f"🔧 JARVIS Voice Profile Repair Tool")
    print(f"{'=' * 80}\n")

    # Get profiles
    profiles = await db.get_all_speaker_profiles()

    # Filter to specified speaker or find corrupted ones
    profiles_to_fix = []
    for p in profiles:
        name = p.get("speaker_name")
        embedding_bytes = p.get("voiceprint_embedding")

        if speaker_name and name != speaker_name:
            continue

        if not embedding_bytes:
            print(f"⚠️  {name}: No embedding, skipping (needs full re-enrollment)")
            continue

        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        if np.any(np.isnan(embedding)):
            print(f"❌ {name}: Embedding contains NaN values - NEEDS REPAIR")
            profiles_to_fix.append(p)
        elif np.any(np.isinf(embedding)):
            print(f"❌ {name}: Embedding contains Inf values - NEEDS REPAIR")
            profiles_to_fix.append(p)
        elif np.linalg.norm(embedding) < 1e-6:
            print(f"❌ {name}: Embedding has zero norm - NEEDS REPAIR")
            profiles_to_fix.append(p)
        else:
            print(f"✅ {name}: Embedding is valid (norm={np.linalg.norm(embedding):.4f})")

    if not profiles_to_fix:
        if speaker_name:
            print(f"\n✅ Profile '{speaker_name}' does not need repair (or doesn't exist)")
        else:
            print(f"\n✅ All profiles are healthy! No repair needed.")
        return

    print(f"\n📋 Found {len(profiles_to_fix)} profile(s) needing repair\n")

    # Try to load SpeechBrain engine for re-extraction
    try:
        from voice.engines.speechbrain_engine import SpeechBrainEngine
        from voice.stt_config import ModelConfig, STTEngine

        model_config = ModelConfig(
            name="speechbrain-wav2vec2",
            engine=STTEngine.SPEECHBRAIN,
            disk_size_mb=380,
            ram_required_gb=2.0,
            vram_required_gb=1.8,
            expected_accuracy=0.96,
            avg_latency_ms=150,
            supports_fine_tuning=True,
            model_path="speechbrain/asr-wav2vec2-commonvoice-en",
        )

        engine = SpeechBrainEngine(model_config)
        await engine.initialize()
        await engine._load_speaker_encoder()
        print("✅ SpeechBrain engine loaded for embedding extraction\n")

    except Exception as e:
        print(f"❌ Could not load SpeechBrain engine: {e}")
        print("   Cannot repair profiles without embedding extraction capability")
        return

    # Repair each profile
    for profile in profiles_to_fix:
        name = profile.get("speaker_name")
        speaker_id = profile.get("speaker_id")

        print(f"\n{'─' * 60}")
        print(f"Repairing: {name} (ID: {speaker_id})")
        print(f"{'─' * 60}")

        # Get voice samples for this speaker
        try:
            async with db.db.cursor() as cursor:
                # Check if using Cloud SQL (PostgreSQL) or SQLite
                is_cloud = hasattr(db, '_is_cloud') and db._is_cloud

                if is_cloud:
                    await cursor.execute(
                        "SELECT sample_id, audio_data, quality_score FROM voice_samples WHERE speaker_id = %s AND audio_data IS NOT NULL",
                        (speaker_id,)
                    )
                else:
                    await cursor.execute(
                        "SELECT sample_id, audio_data, quality_score FROM voice_samples WHERE speaker_id = ? AND audio_data IS NOT NULL",
                        (speaker_id,)
                    )

                samples = await cursor.fetchall()

            print(f"Found {len(samples)} voice sample(s) with audio data")

            if not samples:
                print(f"❌ No audio samples available for re-extraction")
                print(f"   Profile {name} needs complete re-enrollment via voice capture")
                continue

            # Extract embeddings from samples
            valid_embeddings = []

            for sample in samples:
                sample_id = sample.get("sample_id") or sample[0]
                audio_data = sample.get("audio_data") or sample[1]

                if not audio_data or len(audio_data) < 1000:
                    print(f"   Sample {sample_id}: Too short, skipping")
                    continue

                try:
                    embedding = await engine.extract_speaker_embedding(audio_data)

                    # Validate the extracted embedding
                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                        print(f"   Sample {sample_id}: Extraction produced NaN/Inf, skipping")
                        continue

                    norm = np.linalg.norm(embedding)
                    if norm < 1e-6:
                        print(f"   Sample {sample_id}: Zero-norm embedding, skipping")
                        continue

                    valid_embeddings.append(embedding)
                    print(f"   Sample {sample_id}: Valid embedding (norm={norm:.4f})")

                except Exception as e:
                    print(f"   Sample {sample_id}: Extraction failed - {e}")
                    continue

            if not valid_embeddings:
                print(f"\n❌ No valid embeddings could be extracted from samples")
                print(f"   Profile {name} needs complete re-enrollment")
                continue

            # Average the valid embeddings
            print(f"\n✅ Extracted {len(valid_embeddings)} valid embedding(s)")
            new_embedding = np.mean(valid_embeddings, axis=0).astype(np.float32)
            new_norm = np.linalg.norm(new_embedding)

            print(f"   New averaged embedding: shape={new_embedding.shape}, norm={new_norm:.4f}")

            # Validate final embedding
            if np.any(np.isnan(new_embedding)) or np.any(np.isinf(new_embedding)):
                print(f"❌ Averaged embedding is still invalid!")
                continue

            # Update the profile in database
            embedding_bytes = new_embedding.tobytes()

            try:
                async with db.db.cursor() as cursor:
                    if is_cloud:
                        await cursor.execute(
                            """UPDATE speaker_profiles
                               SET voiceprint_embedding = %s,
                                   embedding_dimension = %s,
                                   last_updated = CURRENT_TIMESTAMP
                               WHERE speaker_id = %s""",
                            (embedding_bytes, new_embedding.shape[0], speaker_id)
                        )
                    else:
                        await cursor.execute(
                            """UPDATE speaker_profiles
                               SET voiceprint_embedding = ?,
                                   embedding_dimension = ?,
                                   last_updated = CURRENT_TIMESTAMP
                               WHERE speaker_id = ?""",
                            (embedding_bytes, new_embedding.shape[0], speaker_id)
                        )

                    await db.db.commit()

                print(f"\n✅ Successfully repaired profile '{name}'!")
                print(f"   New embedding: {new_embedding.shape[0]}D, norm={new_norm:.4f}")

            except Exception as e:
                print(f"\n❌ Failed to update database: {e}")

        except Exception as e:
            print(f"❌ Error processing profile: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("🔧 Repair complete!")
    print(f"{'=' * 80}\n")


async def main():
    speaker_name = sys.argv[1] if len(sys.argv) > 1 else None

    if speaker_name:
        print(f"Fixing profile: {speaker_name}")
    else:
        print("Checking all profiles for corruption...")

    try:
        from intelligence.learning_database import get_learning_database

        db = await get_learning_database()

        if not db or not db._initialized:
            print("❌ ERROR: Learning database not initialized")
            return

        await fix_profile(db, speaker_name)

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
