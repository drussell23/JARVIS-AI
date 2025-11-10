#!/usr/bin/env python3
"""Test voice sample retrieval directly from Cloud SQL."""

import asyncio
import asyncpg
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_retrieval():
    """Test retrieving voice samples from Cloud SQL."""

    # Get password from GCP Secret Manager
    try:
        from core.secret_manager import get_secret
        db_password = get_secret("jarvis-db-password")
        print("✅ Retrieved database password from Secret Manager")
    except Exception as e:
        print(f"❌ Failed to get password from Secret Manager: {e}")
        print("💡 Run: gcloud secrets versions access latest --secret='jarvis-db-password'")
        return

    # Connect to Cloud SQL
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # First, check what speakers we have
        print("\n=== CHECKING SPEAKERS ===")
        speakers = await conn.fetch("""
            SELECT speaker_id, speaker_name, created_at
            FROM speaker_profiles
            ORDER BY created_at DESC
        """)

        for speaker in speakers:
            print(f"Speaker {speaker['speaker_id']}: {speaker['speaker_name']} (created {speaker['created_at']})")

        # Now check voice samples for each speaker
        print("\n=== CHECKING VOICE SAMPLES ===")
        for speaker in speakers:
            samples = await conn.fetch("""
                SELECT sample_id, speaker_id,
                       LENGTH(audio_data) as audio_size,
                       audio_hash, quality_score,
                       recording_timestamp
                FROM voice_samples
                WHERE speaker_id = $1
                ORDER BY sample_id DESC
                LIMIT 5
            """, speaker['speaker_id'])

            print(f"\nSpeaker '{speaker['speaker_name']}' (ID: {speaker['speaker_id']}):")
            if samples:
                for sample in samples:
                    audio_info = f"audio: {sample['audio_size']} bytes" if sample['audio_size'] else "audio: NULL"
                    print(f"  Sample {sample['sample_id']}: {audio_info}, "
                          f"hash: {sample['audio_hash'][:8] if sample['audio_hash'] else 'NULL'}..., "
                          f"quality: {sample['quality_score']}")
            else:
                print(f"  No samples found")

        # Check the latest sample specifically
        print("\n=== LATEST SAMPLE (ID: 115) ===")
        latest = await conn.fetchrow("""
            SELECT sample_id, speaker_id, LENGTH(audio_data) as audio_size,
                   audio_hash, transcription
            FROM voice_samples
            WHERE sample_id = 115
        """)

        if latest:
            print(f"Sample 115: Speaker ID {latest['speaker_id']}, "
                  f"Audio: {latest['audio_size']} bytes, "
                  f"Hash: {latest['audio_hash'][:16] if latest['audio_hash'] else 'NULL'}...")
        else:
            print("Sample 115 not found")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(test_retrieval())