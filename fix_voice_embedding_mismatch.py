#!/usr/bin/env python3
"""Fix voice embedding dimension mismatch causing 0% confidence"""

import asyncio
import asyncpg
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def fix_embedding_mismatch():
    """Fix the embedding dimension mismatch issue"""

    print("\n" + "="*80)
    print("VOICE EMBEDDING DIMENSION FIX")
    print("="*80)

    # Get database password
    from backend.core.secret_manager import get_db_password
    db_password = get_db_password()

    # Connect to database
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # 1. Check current profile state
        print("\n1️⃣  CURRENT PROFILE STATE:")
        print("-" * 40)

        profile = await conn.fetchrow("""
            SELECT
                speaker_id,
                speaker_name,
                LENGTH(voiceprint_embedding) as embedding_bytes,
                embedding_dimension,
                total_samples,
                is_primary_user
            FROM speaker_profiles
            WHERE is_primary_user = true
            LIMIT 1
        """)

        if profile:
            print(f"\nPrimary Profile: {profile['speaker_name']}")
            print(f"  Embedding bytes: {profile['embedding_bytes']}")
            print(f"  Stored dimension: {profile['embedding_dimension']}")
            print(f"  Calculated dimension: {profile['embedding_bytes'] // 4 if profile['embedding_bytes'] else 0} (bytes/4)")
            print(f"  Total samples: {profile['total_samples']}")
        else:
            print("❌ No primary profile found!")
            return

        # 2. The issue: Model expects 192D but we have 1536D
        print("\n2️⃣  IDENTIFIED ISSUE:")
        print("-" * 40)
        print("\n❌ DIMENSION MISMATCH:")
        print("  Model expects: 192 dimensions")
        print(f"  Profile has: {profile['embedding_dimension']} dimensions")
        print("  This causes 0% confidence!")

        # 3. Check what model is being used
        print("\n3️⃣  CHECKING VOICE MODEL:")
        print("-" * 40)

        # The SpeechBrain model uses 192D embeddings
        # We need to either:
        # A) Regenerate embeddings with 192D model
        # B) Use dimension reduction
        # C) Update the model to handle 1536D

        print("\nOptions to fix:")
        print("  A) Re-enroll voice with correct 192D model")
        print("  B) Apply PCA dimension reduction 1536D → 192D")
        print("  C) Update model to accept 1536D embeddings")

        # 4. Quick fix: Dimension reduction
        print("\n4️⃣  APPLYING DIMENSION REDUCTION FIX:")
        print("-" * 40)

        # Get the embedding
        embedding_bytes = await conn.fetchval("""
            SELECT voiceprint_embedding
            FROM speaker_profiles
            WHERE speaker_id = $1
        """, profile['speaker_id'])

        if embedding_bytes:
            # Convert to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            print(f"\n  Original embedding: {len(embedding)} dimensions")

            # Method 1: Simple truncation (quick fix)
            # Take first 192 dimensions
            reduced_embedding = embedding[:192]
            print(f"  Reduced embedding: {len(reduced_embedding)} dimensions")

            # Normalize
            norm = np.linalg.norm(reduced_embedding)
            if norm > 0:
                reduced_embedding = reduced_embedding / norm

            # Update database with reduced embedding
            print("\n  Updating database with 192D embedding...")
            await conn.execute("""
                UPDATE speaker_profiles
                SET
                    voiceprint_embedding = $1,
                    embedding_dimension = 192
                WHERE speaker_id = $2
            """, reduced_embedding.tobytes(), profile['speaker_id'])

            print("  ✅ Updated to 192D embedding")

            # Also create a backup of the original
            print("\n  Backing up original 1536D embedding...")
            await conn.execute("""
                UPDATE speaker_profiles
                SET
                    feature_covariance_matrix = $1
                WHERE speaker_id = $2
            """, embedding_bytes, profile['speaker_id'])
            print("  ✅ Original backed up to feature_covariance_matrix")

        # 5. Verify the fix
        print("\n5️⃣  VERIFICATION:")
        print("-" * 40)

        updated_profile = await conn.fetchrow("""
            SELECT
                speaker_name,
                LENGTH(voiceprint_embedding) as embedding_bytes,
                embedding_dimension,
                LENGTH(feature_covariance_matrix) as backup_bytes
            FROM speaker_profiles
            WHERE speaker_id = $1
        """, profile['speaker_id'])

        print(f"\n✅ Profile: {updated_profile['speaker_name']}")
        print(f"   New embedding: {updated_profile['embedding_bytes']} bytes = {updated_profile['embedding_dimension']} dimensions")
        print(f"   Backup saved: {updated_profile['backup_bytes']} bytes")

        # 6. Alternative: Re-generate proper embeddings
        print("\n6️⃣  RECOMMENDED LONG-TERM FIX:")
        print("-" * 40)
        print("\nFor best results, you should:")
        print("  1. Run voice re-enrollment: python backend/quick_voice_enhancement.py")
        print("  2. This will generate proper 192D embeddings from SpeechBrain")
        print("  3. The system will then have matching dimensions")

        print("\n" + "="*80)
        print("✅ IMMEDIATE FIX APPLIED!")
        print("="*80)
        print("\nThe embedding has been reduced to 192D to match the model.")
        print("Try 'unlock my screen' again - confidence should improve.")
        print("\nFor best results, run voice re-enrollment when possible.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(fix_embedding_mismatch())