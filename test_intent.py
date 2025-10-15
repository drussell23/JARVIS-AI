#!/usr/bin/env python3
"""Test intent detection for desktop space queries"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the intent detection logic
text = "What's happening across my desktop spaces?"
text_lower = text.lower()

# Vision patterns from async_pipeline.py
vision_patterns = [
    "across my desktop spaces",
    "happening across my desktop",
    "desktop spaces",
    "desktop space",
    "across my desktop",
    "happening across",
    "what's happening across",
    "what is happening across",
    "my workspace",
    "what am i working on",
    "what's happening",
    "what is happening"
]

print(f"Testing: {text}")
print(f"Lower: {text_lower}\n")

print("Pattern matches:")
for pattern in vision_patterns:
    if pattern in text_lower:
        print(f"  ✓ '{pattern}' MATCHES")
    else:
        print(f"  ✗ '{pattern}' no match")

# Test if it would be detected as vision
detected_intents = []
if any(pattern in text_lower for pattern in vision_patterns):
    detected_intents.append("vision")
    print(f"\n✅ Would be detected as VISION intent")
else:
    print(f"\n❌ Would NOT be detected as vision intent")