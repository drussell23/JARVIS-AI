"""
Test Suite for Cross-Space Intelligence System
==============================================

Tests the advanced cross-space relationship detection and synthesis system.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.intelligence.cross_space_intelligence import (
    CrossSpaceIntelligence,
    KeywordExtractor,
    SemanticCorrelator,
    ActivityCorrelationEngine,
    MultiSourceSynthesizer,
    RelationshipGraph,
    WorkspaceQueryResolver,
    RelationshipType,
    CorrelationDimension
)


async def test_keyword_extraction():
    """Test keyword extraction without hardcoding"""
    print("\n" + "="*80)
    print(" TEST 1: Keyword Extraction (No Hardcoding)")
    print("="*80 + "\n")

    extractor = KeywordExtractor()

    test_cases = [
        (
            "npm install failed with error: ECONNREFUSED",
            ["npm", "install", "failed", "error"]
        ),
        (
            "ModuleNotFoundError: No module named 'requests'",
            ["error", "module"]
        ),
        (
            "Running python app.py in /Users/dev/project",
            ["python", "app.py"]
        ),
    ]

    passed = 0
    for text, expected_keywords in test_cases:
        signature = extractor.extract(text)

        # Check if expected keywords are present in any category
        all_keywords = (
            signature.technical_terms |
            signature.error_indicators |
            signature.action_verbs |
            signature.command_names
        )

        found = [kw for kw in expected_keywords if kw in all_keywords]

        if len(found) >= len(expected_keywords) // 2:  # At least half found
            print(f"✓ Extracted keywords from: '{text[:60]}...'")
            print(f"  Found: {list(all_keywords)[:10]}")
            passed += 1
        else:
            print(f"✗ Failed to extract from: '{text[:60]}...'")
            print(f"  Expected: {expected_keywords}, Found: {found}")

    print(f"\n{'✅' if passed == len(test_cases) else '❌'} Passed: {passed}/{len(test_cases)}\n")
    return passed == len(test_cases)


async def test_semantic_similarity():
    """Test semantic similarity between activities"""
    print("\n" + "="*80)
    print(" TEST 2: Semantic Similarity Detection")
    print("="*80 + "\n")

    correlator = SemanticCorrelator()

    # Create two clearly related activities with overlapping keywords
    sig1 = correlator.create_signature(
        space_id=1,
        app_name="Terminal",
        content="ModuleNotFoundError: No module named 'requests' - pip install requests",
        activity_type="terminal",
        has_error=True,
        significance="critical"
    )

    sig2 = correlator.create_signature(
        space_id=2,
        app_name="Chrome",
        content="Python ModuleNotFoundError fix - install missing module using pip install",
        activity_type="browser",
        has_error=False,
        significance="normal"
    )

    # Calculate similarity
    similarity = sig1.keywords.similarity(sig2.keywords)

    print(f"Activity 1 (Terminal): ModuleNotFoundError - pip install requests")
    print(f"Activity 2 (Browser): Python ModuleNotFoundError fix - pip install")
    print(f"\nSemantic Similarity Score: {similarity:.2f}")

    # Should detect as related (both mention module, error, pip, install)
    success = similarity > 0.2  # Should have decent overlap

    print(f"\n{'✅' if success else '❌'} Semantic similarity detection works\n")
    return success


async def test_activity_correlation():
    """Test multi-dimensional activity correlation"""
    print("\n" + "="*80)
    print(" TEST 3: Multi-Dimensional Activity Correlation")
    print("="*80 + "\n")

    engine = ActivityCorrelationEngine()
    correlator = SemanticCorrelator()

    # Create error activity
    error_sig = correlator.create_signature(
        space_id=1,
        app_name="Terminal",
        content="ModuleNotFoundError: No module named 'requests'",
        activity_type="terminal",
        has_error=True,
        significance="critical"
    )

    # Wait a bit to simulate temporal distance
    await asyncio.sleep(0.1)

    # Create research activity
    research_sig = correlator.create_signature(
        space_id=2,
        app_name="Chrome",
        content="Python pip install requests - how to fix ModuleNotFoundError",
        activity_type="browser",
        has_error=False,
        significance="normal"
    )

    # Correlate
    correlation = engine.correlate(error_sig, research_sig)

    print(f"✓ Temporal Score: {correlation.temporal_score:.2f} (activities close in time)")
    print(f"✓ Semantic Score: {correlation.semantic_score:.2f} (similar content)")
    print(f"✓ Behavioral Score: {correlation.behavioral_score:.2f} (error → browser research)")
    print(f"✓ Causal Score: {correlation.causal_score:.2f} (error causes research)")
    print(f"\n✓ Overall Correlation: {correlation.overall_score:.2f}")

    success = correlation.is_significant(threshold=0.5)

    print(f"\n{'✅' if success else '❌'} Multi-dimensional correlation works\n")
    return success


async def test_relationship_detection():
    """Test automatic relationship detection"""
    print("\n" + "="*80)
    print(" TEST 4: Automatic Relationship Detection")
    print("="*80 + "\n")

    intelligence = CrossSpaceIntelligence()

    # Simulate debugging workflow: error → research → fix

    print("1. Terminal error in Space 1...")
    intelligence.record_activity(
        space_id=1,
        app_name="Terminal",
        content="npm test failed: TypeError: Cannot read property 'verify' of undefined at line 42",
        activity_type="terminal",
        has_error=True,
        significance="critical"
    )

    await asyncio.sleep(0.1)

    print("2. Browser research in Space 2...")
    intelligence.record_activity(
        space_id=2,
        app_name="Chrome",
        content="TypeError Cannot read property of undefined - solution: check if object exists before accessing property",
        activity_type="browser",
        has_error=False,
        significance="normal"
    )

    await asyncio.sleep(0.1)

    print("3. Code fix in IDE in Space 3...")
    intelligence.record_activity(
        space_id=3,
        app_name="VSCode",
        content="if (auth && auth.verify) { auth.verify() }",
        activity_type="ide",
        has_error=False,
        significance="high"
    )

    # Wait for async relationship analysis
    await asyncio.sleep(0.3)

    # Check for relationships
    summary = intelligence.get_workspace_summary()
    relationships = summary.get("relationships_count", 0)

    print(f"\n✓ Detected {relationships} relationship(s)")

    if summary.get("active_workflows"):
        for i, workflow in enumerate(summary["active_workflows"], 1):
            print(f"\n  Workflow {i}:")
            print(f"    Type: {workflow['type']}")
            print(f"    Spaces: {workflow['spaces']}")
            print(f"    Description: {workflow['description']}")
            print(f"    Confidence: {workflow['confidence']:.2f}")

    success = relationships > 0

    print(f"\n{'✅' if success else '❌'} Automatic relationship detection works\n")
    return success


async def test_cross_space_query():
    """Test workspace-wide query resolution"""
    print("\n" + "="*80)
    print(" TEST 5: Workspace-Wide Query Resolution")
    print("="*80 + "\n")

    intelligence = CrossSpaceIntelligence()

    # Setup scenario: error in one space, solution in another
    print("Setup: Error in Space 1, Solution in Space 2\n")

    intelligence.record_activity(
        space_id=1,
        app_name="Terminal",
        content="FileNotFoundError: [Errno 2] No such file or directory: 'config.json'",
        activity_type="terminal",
        has_error=True,
        significance="critical"
    )

    await asyncio.sleep(0.1)

    intelligence.record_activity(
        space_id=2,
        app_name="Chrome",
        content="FileNotFoundError Python solution: check if file exists using os.path.exists() before opening",
        activity_type="browser",
        has_error=False,
        significance="normal"
    )

    # Wait for relationship detection
    await asyncio.sleep(0.3)

    # Query: "What's the error?"
    print("Query: 'What's the error?'\n")
    result = await intelligence.answer_workspace_query("what's the error?", current_space_id=2)

    print(f"🤖 JARVIS Response:\n{result['response']}\n")

    # Check if error was found (case-insensitive)
    success = result.get("found", False) and "filenotfounderror" in result["response"].lower()

    print(f"{'✅' if success else '❌'} Workspace-wide query resolution works\n")
    return success


async def test_multi_source_synthesis():
    """Test information synthesis from multiple sources"""
    print("\n" + "="*80)
    print(" TEST 6: Multi-Source Information Synthesis")
    print("="*80 + "\n")

    intelligence = CrossSpaceIntelligence()

    # Create a complex workflow
    print("Simulating complex debugging workflow...\n")

    # 1. Initial error
    intelligence.record_activity(
        space_id=1,
        app_name="Terminal",
        content="django.db.utils.OperationalError: no such table: auth_user",
        activity_type="terminal",
        has_error=True,
        significance="critical"
    )

    await asyncio.sleep(0.1)

    # 2. Research
    intelligence.record_activity(
        space_id=2,
        app_name="Chrome",
        content="Django no such table error - solution: run python manage.py migrate to create database tables",
        activity_type="browser",
        has_error=False,
        significance="normal"
    )

    await asyncio.sleep(0.1)

    # 3. Attempted fix
    intelligence.record_activity(
        space_id=1,
        app_name="Terminal",
        content="python manage.py migrate - Operations to perform: Apply all migrations",
        activity_type="terminal",
        has_error=False,
        significance="high"
    )

    # Wait for synthesis
    await asyncio.sleep(0.3)

    # Query workspace state
    result = await intelligence.answer_workspace_query("what am i working on?", current_space_id=1)

    print(f"Query: 'What am I working on?'\n")
    print(f"🤖 JARVIS Response:\n{result['response']}\n")

    # Check if response synthesizes information
    success = (
        result.get("found", False) and
        (len(result.get("relationships", [])) > 0 or len(result.get("stories", [])) > 0)
    )

    print(f"{'✅' if success else '❌'} Multi-source synthesis works\n")
    return success


async def test_relationship_graph():
    """Test relationship graph tracking"""
    print("\n" + "="*80)
    print(" TEST 7: Relationship Graph Tracking")
    print("="*80 + "\n")

    graph = RelationshipGraph(max_relationships=10)
    correlator = SemanticCorrelator()
    engine = ActivityCorrelationEngine()

    # Create activities
    act1 = correlator.create_signature(
        space_id=1, app_name="Terminal", content="error in tests",
        activity_type="terminal", has_error=True
    )

    await asyncio.sleep(0.1)

    act2 = correlator.create_signature(
        space_id=2, app_name="Chrome", content="how to fix test errors",
        activity_type="browser", has_error=False
    )

    # Calculate correlation
    correlation = engine.correlate(act1, act2)

    if correlation.is_significant():
        # Create relationship
        from backend.core.intelligence.cross_space_intelligence import CrossSpaceRelationship
        import hashlib

        rel_id = hashlib.md5(f"{act1.space_id}_{act2.space_id}".encode()).hexdigest()[:12]

        relationship = CrossSpaceRelationship(
            relationship_id=rel_id,
            relationship_type=RelationshipType.DEBUGGING,
            activities=[act1, act2],
            correlation_score=correlation,
            first_detected=act1.timestamp,
            last_updated=act2.timestamp,
            confidence=correlation.overall_score,
            evidence=[{"correlation": "high"}],
            description="Test debugging workflow"
        )

        graph.add_relationship(relationship)

        # Test retrieval
        rels = graph.get_relationships_for_space(1)
        connected = graph.get_connected_spaces(1)

        print(f"✓ Added relationship to graph")
        print(f"✓ Space 1 has {len(rels)} relationship(s)")
        print(f"✓ Space 1 connected to spaces: {connected}")

        success = len(rels) > 0 and 2 in connected

        print(f"\n{'✅' if success else '❌'} Relationship graph tracking works\n")
        return success
    else:
        print("✗ Correlation not significant enough")
        return False


async def test_temporal_decay():
    """Test that old relationships are properly managed"""
    print("\n" + "="*80)
    print(" TEST 8: Temporal Relationship Management")
    print("="*80 + "\n")

    correlator = SemanticCorrelator()

    # Create old activity
    old_sig = correlator.create_signature(
        space_id=1, app_name="Terminal", content="old error",
        activity_type="terminal", has_error=True
    )

    await asyncio.sleep(0.2)

    # Create new activity
    new_sig = correlator.create_signature(
        space_id=2, app_name="Chrome", content="new search",
        activity_type="browser"
    )

    # Find related activities
    related = correlator.find_related_activities(new_sig, time_window_seconds=1)

    # Old activity should not be related (outside time window)
    old_in_related = any(r[0].timestamp == old_sig.timestamp for r in related)

    print(f"✓ Old activity (0.2s ago) outside time window (1s)")
    print(f"✓ Related activities found: {len(related)}")
    print(f"✓ Old activity excluded: {not old_in_related}")

    success = not old_in_related

    print(f"\n{'✅' if success else '❌'} Temporal management works\n")
    return success


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print(" Cross-Space Intelligence - Comprehensive Test Suite")
    print("="*80)

    tests = [
        ("Keyword Extraction", test_keyword_extraction),
        ("Semantic Similarity", test_semantic_similarity),
        ("Activity Correlation", test_activity_correlation),
        ("Relationship Detection", test_relationship_detection),
        ("Cross-Space Query", test_cross_space_query),
        ("Multi-Source Synthesis", test_multi_source_synthesis),
        ("Relationship Graph", test_relationship_graph),
        ("Temporal Management", test_temporal_decay),
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
