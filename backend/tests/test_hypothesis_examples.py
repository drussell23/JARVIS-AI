"""
Property-based testing examples with Hypothesis for JARVIS

These tests use Hypothesis to automatically generate test cases,
finding edge cases that manual tests might miss.
"""

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule


# Example 1: Testing string processing utilities
@given(st.text())
def test_string_round_trip(text):
    """Test that encoding and decoding strings is reversible"""
    # Example property: encode -> decode should be identity
    encoded = text.encode("utf-8")
    decoded = encoded.decode("utf-8")
    assert decoded == text


@given(st.text(min_size=1))
def test_string_length_positive(text):
    """Test that non-empty strings have positive length"""
    assert len(text) > 0


# Example 2: Testing numeric operations
@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    """Test that addition is commutative: a + b == b + a"""
    assert a + b == b + a


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_absolute_value_non_negative(x):
    """Test that absolute value is always non-negative"""
    assert abs(x) >= 0


# Example 3: Testing list operations
@given(st.lists(st.integers()))
def test_list_reverse_twice(lst):
    """Test that reversing a list twice gives the original list"""
    assert list(reversed(list(reversed(lst)))) == lst


@given(st.lists(st.integers(), min_size=1))
def test_list_max_is_member(lst):
    """Test that max of a list is actually in the list"""
    maximum = max(lst)
    assert maximum in lst


# Example 4: Testing dictionary operations
@given(st.dictionaries(st.text(), st.integers()))
def test_dict_keys_after_update(dict1):
    """Test that dictionary keys are preserved after operations"""
    original_keys = set(dict1.keys())
    dict2 = dict1.copy()
    dict2.update(dict1)
    assert set(dict2.keys()) == original_keys


# Example 5: Testing confidence scores (0.0 to 1.0)
@given(st.floats(min_value=0.0, max_value=1.0))
def test_confidence_score_valid_range(confidence):
    """Test that confidence scores are in valid range"""
    assert 0.0 <= confidence <= 1.0

    # Test confidence thresholds
    if confidence >= 0.8:
        assert confidence >= 0.8  # High confidence
    elif confidence <= 0.3:
        assert confidence <= 0.3  # Low confidence


# Example 6: Testing goal inference patterns
@given(
    st.text(min_size=1, max_size=500),
    st.floats(min_value=0.0, max_value=1.0),
    st.lists(st.text(), min_size=0, max_size=10),
)
def test_goal_pattern_structure(goal_text, confidence, context):
    """Test that goal patterns maintain valid structure"""
    goal_pattern = {"goal_text": goal_text, "confidence": confidence, "context": context}

    assert "goal_text" in goal_pattern
    assert "confidence" in goal_pattern
    assert "context" in goal_pattern
    assert isinstance(goal_pattern["context"], list)
    assert 0.0 <= goal_pattern["confidence"] <= 1.0


# Example 7: Stateful testing with state machine
class ContextStoreStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for context store operations
    Tests that context store maintains invariants across operations
    """

    def __init__(self):
        super().__init__()
        self.store = {}
        self.total_items = 0

    @rule(key=st.text(), value=st.integers())
    def add_item(self, key, value):
        """Add an item to the store"""
        self.store[key] = value
        self.total_items = len(self.store)

    @rule(key=st.text())
    def remove_item(self, key):
        """Remove an item from the store"""
        if key in self.store:
            del self.store[key]
            self.total_items = len(self.store)

    @invariant()
    def total_matches_length(self):
        """Invariant: total_items always matches actual length"""
        assert self.total_items == len(self.store)

    @invariant()
    def no_duplicate_keys(self):
        """Invariant: no duplicate keys should exist"""
        assert len(self.store.keys()) == len(set(self.store.keys()))


# Run the state machine test
TestContextStore = ContextStoreStateMachine.TestCase


# Example 8: Testing with custom strategies
@st.composite
def goal_pattern_strategy(draw):
    """Custom strategy for generating realistic goal patterns"""
    goal_types = ["open_app", "close_app", "switch_space", "query", "action"]
    return {
        "goal_type": draw(st.sampled_from(goal_types)),
        "confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
        "timestamp": draw(st.integers(min_value=0, max_value=2**32)),
        "context": draw(st.lists(st.text(), min_size=0, max_size=5)),
    }


@given(goal_pattern_strategy())
def test_goal_pattern_realistic(pattern):
    """Test with realistic goal patterns"""
    assert pattern["goal_type"] in ["open_app", "close_app", "switch_space", "query", "action"]
    assert 0.5 <= pattern["confidence"] <= 1.0
    assert 0 <= pattern["timestamp"] <= 2**32
    assert len(pattern["context"]) <= 5


# Example 9: Testing with examples decorator
@given(st.integers())
@example(0)  # Edge case: zero
@example(-1)  # Edge case: negative
@example(2**31 - 1)  # Edge case: max int
def test_integer_operations_with_examples(n):
    """Test integer operations with specific edge cases"""
    # Property: n * 2 / 2 should equal n (for integers)
    if n != 0:
        assert (n * 2) // 2 == n


# Example 10: Testing with settings
@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(allow_nan=False), min_size=1))
def test_confidence_aggregation(confidences):
    """Test confidence score aggregation (with custom settings)"""
    # Normalize confidences to 0-1 range
    normalized = [max(0.0, min(1.0, abs(c))) for c in confidences]

    # Test average is within bounds
    avg = sum(normalized) / len(normalized)
    assert 0.0 <= avg <= 1.0

    # Test max/min are within bounds
    assert 0.0 <= max(normalized) <= 1.0
    assert 0.0 <= min(normalized) <= 1.0


if __name__ == "__main__":
    # Run hypothesis tests
    pytest.main([__file__, "-v"])
