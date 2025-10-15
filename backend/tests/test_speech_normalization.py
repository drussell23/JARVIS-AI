"""
Test Speech-to-Text Normalization for Context Intelligence
==========================================================

Tests that common speech-to-text errors are properly corrected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.context.context_integration_bridge import ContextIntegrationBridge
from backend.core.context.multi_space_context_graph import MultiSpaceContextGraph


def test_speech_normalization():
    """Test that speech-to-text errors are normalized"""
    print("\n" + "="*80)
    print(" Speech-to-Text Normalization Tests")
    print("="*80 + "\n")

    # Create a minimal bridge just to access the normalization method
    graph = MultiSpaceContextGraph()
    bridge = ContextIntegrationBridge(graph)

    test_cases = [
        # "and" → "in" corrections
        ("can you see my terminal and the other window", "can you see my terminal in the other window"),
        ("can you see my terminal and the other space", "can you see my terminal in the other space"),
        ("see terminal and another window", "see my terminal in another window"),

        # "on" → "in" corrections
        ("can you see terminal on the other window", "can you see my terminal in the other window"),
        ("do you see the error on another space", "do you see the error in another space"),

        # "of" → "in" corrections
        ("can you see terminal of the other window", "can you see my terminal in the other window"),

        # Missing possessive corrections
        ("can you see terminal", "can you see my terminal"),
        ("do you see browser", "do you see my browser"),
        ("see the terminal", "see my terminal"),

        # Filler word removal
        ("can you um see my terminal", "can you see my terminal"),
        ("do you uh see the error", "do you see the error"),
        ("can you like see my terminal", "can you see my terminal"),

        # Combined corrections
        ("um can you see terminal and the other window", "can you see my terminal in the other window"),
        ("uh do you like see browser on another space", "do you see my browser in another space"),
    ]

    passed = 0
    failed = 0

    for original, expected in test_cases:
        normalized = bridge._normalize_speech_query(original)

        if normalized == expected:
            print(f"✓ PASS")
            print(f"  Input:    '{original}'")
            print(f"  Output:   '{normalized}'")
            print()
            passed += 1
        else:
            print(f"✗ FAIL")
            print(f"  Input:    '{original}'")
            print(f"  Expected: '{expected}'")
            print(f"  Got:      '{normalized}'")
            print()
            failed += 1

    print("="*80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} total")
    print("="*80 + "\n")

    if failed == 0:
        print("🎉 All speech normalization tests passed!\n")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed\n")
        return False


if __name__ == "__main__":
    success = test_speech_normalization()
    sys.exit(0 if success else 1)
