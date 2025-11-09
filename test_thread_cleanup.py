#!/usr/bin/env python3
"""
Test script to verify background thread cleanup works properly
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_speaker_service_cleanup():
    """Test that speaker verification service cleans up threads properly"""
    print("=" * 70)
    print("Testing Speaker Verification Service Thread Cleanup")
    print("=" * 70)

    # Import speaker service
    from voice.speaker_verification_service import SpeakerVerificationService

    print("\n1. Creating SpeakerVerificationService...")
    service = SpeakerVerificationService()

    # Check initial thread count
    initial_thread_count = threading.active_count()
    initial_threads = [t.name for t in threading.enumerate()]
    print(f"   Initial threads: {initial_thread_count}")
    print(f"   Thread names: {initial_threads}")

    # Initialize the service (this will start background thread)
    print("\n2. Initializing service (fast mode with background preload)...")
    try:
        await service.initialize_fast()
        print("   ✅ Service initialized")
    except Exception as e:
        print(f"   ⚠️  Initialization warning: {e}")
        print("   (Expected if database not available)")

    # Check thread count after initialization
    await asyncio.sleep(0.5)  # Let thread start
    after_init_count = threading.active_count()
    after_init_threads = [t.name for t in threading.enumerate()]
    print(f"\n3. After initialization:")
    print(f"   Thread count: {after_init_count}")
    print(f"   Thread names: {after_init_threads}")
    print(f"   New threads: {after_init_count - initial_thread_count}")

    # Cleanup
    print("\n4. Running cleanup...")
    await service.cleanup()

    # Wait a moment for threads to finish
    await asyncio.sleep(1.0)

    # Check final thread count
    final_thread_count = threading.active_count()
    final_threads = [t.name for t in threading.enumerate()]
    print(f"\n5. After cleanup:")
    print(f"   Thread count: {final_thread_count}")
    print(f"   Thread names: {final_threads}")

    # Verify cleanup worked
    lingering_threads = final_thread_count - initial_thread_count
    print(f"\n6. Results:")
    print(f"   Lingering threads: {lingering_threads}")

    if lingering_threads == 0:
        print("   ✅ SUCCESS: All threads cleaned up properly!")
        return True
    else:
        print(f"   ❌ FAILURE: {lingering_threads} thread(s) still running")
        # Show which threads are lingering
        new_threads = [t for t in final_threads if t not in initial_threads]
        if new_threads:
            print(f"   Lingering thread names: {new_threads}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_speaker_service_cleanup())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
