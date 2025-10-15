#!/usr/bin/env python3
"""Test if multi-space detection is working"""

import sys
sys.path.insert(0, 'backend')

from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension

# Test the detection
ext = MultiSpaceIntelligenceExtension()
test_query = "What's happening across my desktop spaces?"

# Check detection
should_use = ext.should_use_multi_space(test_query)
intent = ext.query_detector.detect_intent(test_query)

print(f"Query: '{test_query}'")
print(f"Should use multi-space: {should_use}")
print(f"Detected intent type: {intent.query_type}")
print(f"Intent target app: {intent.target_app}")
print(f"Intent confidence: {intent.confidence}")
print(f"Space references: {intent.space_references}")

# Test pattern directly
import re
pattern = r"\b(?:across|between|among)\s+(?:\w+\s+)?(?:spaces?|desktops?)"
match = re.search(pattern, test_query.lower())
print(f"\nDirect pattern match: {match is not None}")
if match:
    print(f"Match found: '{match.group()}'")