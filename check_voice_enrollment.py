#!/usr/bin/env python3
"""
Check Voice Biometric Enrollment Status
"""
import asyncio
import sys
import os

sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def check_enrollment():
    print("\n" + "="*70)
    print("VOICE BIOMETRIC ENROLLMENT STATUS")
    print("="*70 + "\n")
    
    # Check 1: Database connection
    print("1. Checking learning database...")
    try:
        from intelligence.learning_database import get_learning_database
        db = await get_learning_database()
        print("   ✅ Learning database connected")
        
        # Check for speaker profiles
        print("\n2. Checking speaker profiles in database...")
        
        try:
            # Get all speaker profiles using the proper method
            profiles = await db.get_all_speaker_profiles()
            
            if profiles:
                print(f"   ✅ Found {len(profiles)} speaker profile(s):\n")
                for profile in profiles:
                    speaker_name = profile['speaker_name']
                    speaker_id = profile['speaker_id']
                    total_samples = profile['total_samples']
                    is_owner = profile['is_primary_user']
                    
                    owner_badge = "👑 OWNER" if is_owner else "Guest"
                    print(f"   📊 {speaker_name}")
                    print(f"      - Samples: {total_samples}")
                    print(f"      - Status: {owner_badge}")
                    print(f"      - Speaker ID: {speaker_id}\n")
            else:
                print("   ⚠️  No speaker profiles found in database")
                print("\n   You need to enroll your voice first!")
                
        except Exception as e:
            print(f"   ⚠️  Could not query database: {e}")
            import traceback
            traceback.print_exc()
            print("   Database may need to be initialized")
            
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check 2: Voice enrollment file
    print("\n3. Checking voice enrollment files...")
    enrollment_dir = os.path.expanduser("~/.jarvis")
    enrollment_file = os.path.join(enrollment_dir, "voice_unlock_enrollment.json")
    
    if os.path.exists(enrollment_file):
        import json
        with open(enrollment_file, 'r') as f:
            data = json.load(f)
            print(f"   ✅ Enrollment file found")
            print(f"      - Status: {data.get('status', 'unknown')}")
            print(f"      - Speaker: {data.get('speaker_name', 'unknown')}")
    else:
        print(f"   ⚠️  No enrollment file at: {enrollment_file}")
    
    print("\n" + "="*70)
    print("ENROLLMENT CHECK COMPLETE")
    print("="*70 + "\n")
    
    print("📋 Next Steps:")
    print("\nIf no voice samples found:")
    print("1. Run enrollment script:")
    print("   python3 backend/voice_unlock/setup_voice_unlock.py")
    print("\n2. Or enroll via JARVIS:")
    print('   Say: "Jarvis, enroll my voice"')
    print("\n3. Record 25+ voice samples for best accuracy")
    print("\nOnce enrolled, voice biometric unlock will work automatically!")

if __name__ == "__main__":
    asyncio.run(check_enrollment())
