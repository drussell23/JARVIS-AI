#!/usr/bin/env python3
"""Diagnose voice verification confidence issues."""

import asyncio
import asyncpg
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def diagnose_voice_issues():
    """Diagnose why voice confidence is low (19.06% instead of 85%+)."""

    print("\n" + "="*80)
    print("VOICE VERIFICATION DIAGNOSTIC REPORT")
    print("="*80)
    print("\nCurrent Issue: Voice confidence at 19.06% (needs 85%+ for unlock)")

    # Get database password
    try:
        from core.secret_manager import get_secret
        db_password = get_secret("jarvis-db-password")
    except Exception as e:
        print(f"❌ Failed to get password from Secret Manager: {e}")
        print("💡 Run: gcloud secrets versions access latest --secret='jarvis-db-password'")
        return

    # Connect to database
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # 1. Check speaker profiles and their data completeness
        print("\n1️⃣  SPEAKER PROFILES CHECK:")
        print("-" * 40)

        profiles = await conn.fetch("""
            SELECT
                speaker_id,
                speaker_name,
                total_samples,
                LENGTH(voiceprint_embedding) as embedding_size,
                enrollment_quality_score,
                verification_count,
                successful_verifications,
                failed_verifications,
                last_verified,
                pitch_mean_hz,
                pitch_std_hz,
                formant_f1_hz,
                formant_f2_hz,
                spectral_centroid_hz,
                energy_mean,
                speaking_rate_wpm
            FROM speaker_profiles
            ORDER BY speaker_id
        """)

        for profile in profiles:
            print(f"\n👤 Speaker: {profile['speaker_name']} (ID: {profile['speaker_id']})")
            print(f"   Samples: {profile['total_samples'] or 0}")
            print(f"   Embedding: {profile['embedding_size'] or 0} bytes")
            print(f"   Enrollment Quality: {profile['enrollment_quality_score'] or 'N/A'}")
            print(f"   Verifications: {profile['successful_verifications'] or 0} success / {profile['failed_verifications'] or 0} failed")

            # Check if acoustic features are present
            features_present = []
            features_missing = []

            feature_checks = {
                'pitch_mean_hz': 'Pitch',
                'formant_f1_hz': 'Formant F1',
                'formant_f2_hz': 'Formant F2',
                'spectral_centroid_hz': 'Spectral Centroid',
                'energy_mean': 'Energy',
                'speaking_rate_wpm': 'Speaking Rate'
            }

            for field, name in feature_checks.items():
                if profile[field] is not None:
                    features_present.append(name)
                else:
                    features_missing.append(name)

            if features_present:
                print(f"   ✅ Features present: {', '.join(features_present)}")
            if features_missing:
                print(f"   ❌ Features MISSING: {', '.join(features_missing)}")

        # 2. Check voice samples quality
        print("\n2️⃣  VOICE SAMPLES ANALYSIS:")
        print("-" * 40)

        sample_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_samples,
                COUNT(audio_data) as samples_with_audio,
                COUNT(mfcc_features) as samples_with_mfcc,
                COUNT(audio_fingerprint) as samples_with_fingerprint,
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                AVG(duration_ms) as avg_duration_ms
            FROM voice_samples
            WHERE speaker_id = 1
        """)

        print(f"\nSamples for Derek J. Russell (ID: 1):")
        print(f"   Total: {sample_stats['total_samples']}")
        print(f"   With Audio: {sample_stats['samples_with_audio']}")
        print(f"   With MFCC: {sample_stats['samples_with_mfcc']}")
        print(f"   With Fingerprint: {sample_stats['samples_with_fingerprint']}")
        print(f"   Quality: avg={sample_stats['avg_quality']:.2f}, min={sample_stats['min_quality']:.2f}, max={sample_stats['max_quality']:.2f}")
        print(f"   Duration: avg={sample_stats['avg_duration_ms']:.1f}ms")

        # 3. Check recent verification attempts
        print("\n3️⃣  RECENT VERIFICATION ATTEMPTS:")
        print("-" * 40)

        # Check misheard queries (might show verification issues)
        recent_failures = await conn.fetch("""
            SELECT
                query_id,
                original_query,
                timestamp
            FROM misheard_queries
            WHERE original_query LIKE '%unlock%'
               OR original_query LIKE '%verification%'
            ORDER BY timestamp DESC
            LIMIT 5
        """)

        if recent_failures:
            print("\nRecent unlock/verification attempts in misheard_queries:")
            for failure in recent_failures:
                print(f"   {failure['timestamp']}: '{failure['original_query']}'")
        else:
            print("\nNo recent unlock attempts found in misheard_queries")

        # 4. Check BEAST MODE configuration
        print("\n4️⃣  BEAST MODE STATUS:")
        print("-" * 40)

        # Check for BEAST MODE in patterns
        beast_patterns = await conn.fetch("""
            SELECT pattern_id, pattern_type, confidence_score
            FROM patterns
            WHERE pattern_data::text LIKE '%BEAST%'
               OR pattern_type LIKE '%beast%'
            LIMIT 5
        """)

        if beast_patterns:
            print("\n✅ BEAST MODE patterns found:")
            for pattern in beast_patterns:
                print(f"   Pattern {pattern['pattern_id']}: {pattern['pattern_type']} (confidence: {pattern['confidence_score']})")
        else:
            print("\n⚠️  No BEAST MODE patterns found in database")

        # 5. Diagnostic Summary
        print("\n" + "="*80)
        print("DIAGNOSIS SUMMARY:")
        print("="*80)

        issues = []

        # Check for missing embeddings
        if not profiles or profiles[0]['embedding_size'] is None or profiles[0]['embedding_size'] == 0:
            issues.append("❌ No voiceprint embedding stored - need to re-enroll")

        # Check for missing acoustic features
        derek_profile = [p for p in profiles if p['speaker_id'] == 1]
        if derek_profile:
            profile = derek_profile[0]
            if not profile['pitch_mean_hz']:
                issues.append("❌ Missing acoustic features (pitch, formants, etc.) - need feature extraction")
            if profile['total_samples'] and profile['total_samples'] < 10:
                issues.append("⚠️  Low sample count - need more enrollment samples")

        # Check sample quality
        if sample_stats['samples_with_mfcc'] == 0:
            issues.append("❌ No MFCC features in samples - need feature extraction")
        if sample_stats['avg_quality'] and sample_stats['avg_quality'] < 0.7:
            issues.append("⚠️  Low average sample quality - may need better recordings")

        if issues:
            print("\n🔴 ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ No major issues found - may be a runtime matching problem")

        print("\n📋 RECOMMENDED ACTIONS:")
        print("   1. Re-run voice enrollment with feature extraction")
        print("   2. Ensure BEAST MODE is properly initialized")
        print("   3. Check voice authentication matching algorithm")
        print("   4. Verify audio capture quality during unlock attempts")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(diagnose_voice_issues())