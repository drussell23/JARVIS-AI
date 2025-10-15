"""
Test Suite for Implicit Reference Resolver
==========================================

Tests the advanced "what does it say?" understanding system.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.nlp.implicit_reference_resolver import (
    ImplicitReferenceResolver,
    QueryAnalyzer,
    QueryIntent,
    ConversationalContext,
    VisualAttentionTracker
)
from backend.core.context.multi_space_context_graph import MultiSpaceContextGraph


async def test_query_analyzer():
    """Test query intent classification"""
    print("\n" + "="*80)
    print(" TEST 1: Query Intent Classification")
    print("="*80 + "\n")

    analyzer = QueryAnalyzer()

    test_queries = [
        ("what does it say?", QueryIntent.DESCRIBE),
        ("explain that error", QueryIntent.EXPLAIN),
        ("what's wrong?", QueryIntent.DIAGNOSE),
        ("how do I fix it?", QueryIntent.FIX),
        ("what happened earlier?", QueryIntent.RECALL),
    ]

    passed = 0
    for query, expected_intent in test_queries:
        parsed = analyzer.analyze(query)
        actual_intent = parsed.intent

        if actual_intent == expected_intent:
            print(f"✓ '{query}' → {actual_intent.value}")
            passed += 1
        else:
            print(f"✗ '{query}' → {actual_intent.value} (expected {expected_intent.value})")

    print(f"\n{'✅' if passed == len(test_queries) else '❌'} Passed: {passed}/{len(test_queries)}\n")
    return passed == len(test_queries)


async def test_pronoun_extraction():
    """Test pronoun and reference extraction"""
    print("\n" + "="*80)
    print(" TEST 2: Pronoun & Reference Extraction")
    print("="*80 + "\n")

    analyzer = QueryAnalyzer()

    test_queries = [
        ("what does it say?", ["it"]),
        ("explain that error", ["that"]),
        ("what's this about?", ["this"]),
        ("how do I fix the error?", ["the error"]),
    ]

    passed = 0
    for query, expected_refs in test_queries:
        parsed = analyzer.analyze(query)
        actual_refs = [ref.text for ref in parsed.references]

        match = all(exp in actual_refs for exp in expected_refs)
        if match:
            print(f"✓ '{query}' → {actual_refs}")
            passed += 1
        else:
            print(f"✗ '{query}' → {actual_refs} (expected {expected_refs})")

    print(f"\n{'✅' if passed == len(test_queries) else '❌'} Passed: {passed}/{len(test_queries)}\n")
    return passed == len(test_queries)


async def test_conversational_context():
    """Test conversational context tracking"""
    print("\n" + "="*80)
    print(" TEST 3: Conversational Context Tracking")
    print("="*80 + "\n")

    context = ConversationalContext()

    # Simulate conversation
    context.add_turn(
        user_query="what's the error?",
        jarvis_response="The error is: ModuleNotFoundError: No module named 'requests'",
        context_used={"type": "error", "details": {"error": "ModuleNotFoundError"}}
    )

    context.add_turn(
        user_query="how do I fix it?",
        jarvis_response="You can fix it by running: pip install requests",
        context_used={"type": "fix", "command": "pip install requests"}
    )

    # Test retrieval
    recent_turns = context.get_recent_turns(count=2)
    last_entity = context.get_last_mentioned_entity()

    print(f"✓ Recorded {len(context.turns)} conversation turns")
    print(f"✓ Recent turns: {len(recent_turns)}")
    print(f"✓ Last entity: {last_entity[0] if last_entity else 'None'}")

    success = len(context.turns) == 2 and last_entity is not None
    print(f"\n{'✅' if success else '❌'} Conversational context tracking works\n")
    return success


async def test_visual_attention():
    """Test visual attention tracking"""
    print("\n" + "="*80)
    print(" TEST 4: Visual Attention Tracking")
    print("="*80 + "\n")

    tracker = VisualAttentionTracker()

    # Record attention events
    tracker.record_attention(
        space_id=1,
        app_name="Terminal",
        content_summary="ModuleNotFoundError: No module named 'requests'",
        content_type="error",
        significance="critical"
    )

    tracker.record_attention(
        space_id=2,
        app_name="Chrome",
        content_summary="Python documentation for requests module",
        content_type="documentation",
        significance="normal"
    )

    # Test retrieval
    recent_critical = tracker.get_recent_critical()
    error_events = tracker.get_most_recent_by_type("error")

    print(f"✓ Recorded {len(tracker.attention_events)} attention events")
    print(f"✓ Recent critical: {recent_critical.content_summary[:50] if recent_critical else 'None'}...")
    print(f"✓ Most recent error: {error_events.content_summary[:50] if error_events else 'None'}...")

    success = recent_critical is not None and error_events is not None
    print(f"\n{'✅' if success else '❌'} Visual attention tracking works\n")
    return success


async def test_implicit_reference_resolution():
    """Test full implicit reference resolution"""
    print("\n" + "="*80)
    print(" TEST 5: Implicit Reference Resolution - 'what does it say?'")
    print("="*80 + "\n")

    # Setup
    graph = MultiSpaceContextGraph()
    resolver = ImplicitReferenceResolver(graph)

    # Simulate terminal error
    print("1. Simulating terminal error in Space 1...")
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="python app.py",
        errors=["ModuleNotFoundError: No module named 'requests'"],
        exit_code=1
    )

    # Record visual attention
    resolver.record_visual_attention(
        space_id=1,
        app_name="Terminal",
        ocr_text="ModuleNotFoundError: No module named 'requests'",
        content_type="error",
        significance="critical"
    )

    print("2. User asks: 'what does it say?'\n")

    # Test resolution
    result = await resolver.resolve_query("what does it say?")

    print(f"✓ Intent: {result['intent']}")
    print(f"✓ Confidence: {result['confidence']:.2f}")
    print(f"✓ Referent source: {result['referent'].get('source')}")
    print(f"✓ Referent type: {result['referent'].get('type')}")
    print(f"\n🤖 JARVIS Response:\n{result['response']}\n")

    success = (
        result['intent'] in ['explain', 'describe'] and
        result['referent'].get('type') in ['error', 'workspace_error'] and
        'ModuleNotFoundError' in result['response']
    )

    print(f"{'✅' if success else '❌'} Implicit reference resolution works\n")
    return success


async def test_multi_turn_conversation():
    """Test multi-turn conversation with pronoun resolution"""
    print("\n" + "="*80)
    print(" TEST 6: Multi-Turn Conversation")
    print("="*80 + "\n")

    graph = MultiSpaceContextGraph()
    resolver = ImplicitReferenceResolver(graph)

    # Setup error
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="npm test",
        errors=["TypeError: Cannot read property 'verify' of undefined"],
        exit_code=1
    )

    resolver.record_visual_attention(
        space_id=1,
        app_name="Terminal",
        ocr_text="TypeError: Cannot read property 'verify' of undefined",
        content_type="error",
        significance="critical"
    )

    # Turn 1
    print("Turn 1:")
    print("User: 'what's the error?'")
    result1 = await resolver.resolve_query("what's the error?")
    print(f"JARVIS: {result1['response'][:100]}...\n")

    # Turn 2 - pronoun reference
    print("Turn 2:")
    print("User: 'how do I fix it?'")  # "it" should refer to the error from Turn 1
    result2 = await resolver.resolve_query("how do I fix it?")
    print(f"JARVIS: {result2['response'][:100]}...\n")

    # Verify conversation was tracked
    turns = resolver.conversational_context.get_recent_turns()
    print(f"✓ Tracked {len(turns)} conversation turns")

    success = len(turns) == 2 and result2['intent'] == 'fix'
    print(f"\n{'✅' if success else '❌'} Multi-turn conversation works\n")
    return success


async def test_cross_space_reference():
    """Test references across multiple spaces"""
    print("\n" + "="*80)
    print(" TEST 7: Cross-Space Reference Resolution")
    print("="*80 + "\n")

    graph = MultiSpaceContextGraph()
    resolver = ImplicitReferenceResolver(graph)

    # Error in Space 1
    print("1. Error appears in Terminal (Space 1)")
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="python script.py",
        errors=["FileNotFoundError: data.csv"],
        exit_code=1
    )

    resolver.record_visual_attention(
        space_id=1,
        app_name="Terminal",
        ocr_text="FileNotFoundError: data.csv",
        content_type="error",
        significance="critical"
    )

    # User switches to Space 2 (browser)
    print("2. User switches to Chrome (Space 2) to research")
    graph.set_active_space(2)
    resolver.record_visual_attention(
        space_id=2,
        app_name="Chrome",
        ocr_text="Python FileNotFoundError documentation",
        content_type="documentation",
        significance="normal"
    )

    # User asks about error while in different space
    print("3. User (still in Space 2) asks: 'what was that error?'\n")
    result = await resolver.resolve_query("what was that error?")

    print(f"✓ Resolved to Space {result['referent'].get('space_id', 'unknown')}")
    print(f"✓ Response: {result['response'][:100]}...\n")

    success = result['referent'].get('space_id') == 1 and 'FileNotFoundError' in result['response']
    print(f"{'✅' if success else '❌'} Cross-space reference resolution works\n")
    return success


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print(" Implicit Reference Resolver - Comprehensive Test Suite")
    print("="*80)

    tests = [
        ("Query Intent Classification", test_query_analyzer),
        ("Pronoun Extraction", test_pronoun_extraction),
        ("Conversational Context", test_conversational_context),
        ("Visual Attention", test_visual_attention),
        ("Implicit Reference", test_implicit_reference_resolution),
        ("Multi-Turn Conversation", test_multi_turn_conversation),
        ("Cross-Space Reference", test_cross_space_reference),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!\n")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
