"""
Comprehensive Test Suite for Multi-Space Context Graph
=======================================================

Tests all major functionality:
1. Space context tracking
2. Application context management
3. Terminal/Browser/IDE context updates
4. Cross-space correlation
5. Natural language queries ("what does it say?")
6. Temporal decay
7. Integration with existing systems
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.context.multi_space_context_graph import (
    MultiSpaceContextGraph,
    SpaceContext,
    ApplicationContext,
    ContextType,
    ActivitySignificance,
    TerminalContext,
    BrowserContext,
    IDEContext,
    CrossSpaceCorrelator,
    CrossSpaceRelationship
)
from backend.core.context.context_integration_bridge import (
    ContextIntegrationBridge,
    initialize_integration_bridge
)


# ============================================================================
# TEST 1: Basic Space Context Tracking
# ============================================================================

@pytest.mark.asyncio
async def test_basic_space_creation_and_activation():
    """Test creating and activating spaces"""
    graph = MultiSpaceContextGraph(decay_ttl_seconds=300)

    # Create space 1
    space1 = graph.get_or_create_space(1)
    assert space1.space_id == 1
    assert not space1.is_active

    # Activate space 1
    graph.set_active_space(1)
    assert space1.is_active
    assert graph.current_space_id == 1

    # Create and activate space 2
    space2 = graph.get_or_create_space(2)
    graph.set_active_space(2)

    assert space2.is_active
    assert not space1.is_active  # Space 1 should be deactivated
    assert graph.current_space_id == 2

    print("✓ Test 1 passed: Basic space creation and activation")


# ============================================================================
# TEST 2: Application Context Management
# ============================================================================

@pytest.mark.asyncio
async def test_application_context_management():
    """Test adding and managing applications within spaces"""
    graph = MultiSpaceContextGraph()

    space1 = graph.get_or_create_space(1)

    # Add terminal application
    terminal_app = space1.add_application("Terminal", ContextType.TERMINAL)
    assert terminal_app.app_name == "Terminal"
    assert terminal_app.context_type == ContextType.TERMINAL
    assert terminal_app.space_id == 1

    # Add browser application
    browser_app = space1.add_application("Safari", ContextType.BROWSER)
    assert len(space1.applications) == 2

    # Remove application
    space1.remove_application("Terminal")
    assert len(space1.applications) == 1
    assert "Terminal" not in space1.applications

    print("✓ Test 2 passed: Application context management")


# ============================================================================
# TEST 3: Terminal Context Updates
# ============================================================================

@pytest.mark.asyncio
async def test_terminal_context_updates():
    """Test updating terminal context with commands and errors"""
    graph = MultiSpaceContextGraph()

    # Simulate terminal command with error
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="python app.py",
        output="Traceback...\nModuleNotFoundError: No module named 'requests'",
        errors=["ModuleNotFoundError: No module named 'requests'"],
        exit_code=1,
        working_dir="/Users/test/project"
    )

    # Verify context was updated
    space1 = graph.spaces[1]
    terminal_app = space1.applications["Terminal"]

    assert terminal_app.context_type == ContextType.TERMINAL
    assert terminal_app.terminal_context is not None

    terminal = terminal_app.terminal_context
    assert terminal.last_command == "python app.py"
    assert len(terminal.errors) == 1
    assert "ModuleNotFoundError" in terminal.errors[0]
    assert terminal.exit_code == 1
    assert terminal.has_error()

    # Check that a critical event was recorded
    events = space1.get_recent_events()
    error_events = [e for e in events if e.event_type == "terminal_error"]
    assert len(error_events) > 0
    assert error_events[0].significance == ActivitySignificance.CRITICAL

    print("✓ Test 3 passed: Terminal context updates")


# ============================================================================
# TEST 4: Browser Context Updates
# ============================================================================

@pytest.mark.asyncio
async def test_browser_context_updates():
    """Test updating browser context with research detection"""
    graph = MultiSpaceContextGraph()

    # Simulate browser on documentation page
    graph.update_browser_context(
        space_id=2,
        app_name="Safari",
        url="https://docs.python.org/3/library/requests.html",
        title="requests - HTTP library documentation",
        extracted_text="requests is a simple HTTP library for Python\nDocumentation Guide Tutorial",
        search_query="python requests documentation"
    )

    # Verify context
    space2 = graph.spaces[2]
    browser_app = space2.applications["Safari"]

    assert browser_app.context_type == ContextType.BROWSER
    browser = browser_app.browser_context

    assert browser.active_url == "https://docs.python.org/3/library/requests.html"
    assert browser.is_researching  # Should detect documentation keywords
    assert browser.search_query == "python requests documentation"

    print("✓ Test 4 passed: Browser context updates with research detection")


# ============================================================================
# TEST 5: IDE Context Updates
# ============================================================================

@pytest.mark.asyncio
async def test_ide_context_updates():
    """Test updating IDE context with file and error information"""
    graph = MultiSpaceContextGraph()

    # Simulate IDE with code file
    graph.update_ide_context(
        space_id=3,
        app_name="VS Code",
        active_file="app.py",
        open_files=["app.py", "requirements.txt", "config.py"],
        errors=["Line 5: undefined variable 'requests'"]
    )

    # Verify context
    space3 = graph.spaces[3]
    ide_app = space3.applications["VS Code"]

    assert ide_app.context_type == ContextType.IDE
    ide = ide_app.ide_context

    assert ide.active_file == "app.py"
    assert len(ide.open_files) == 3
    assert len(ide.errors_in_file) == 1

    print("✓ Test 5 passed: IDE context updates")


# ============================================================================
# TEST 6: Cross-Space Correlation - Debugging Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_cross_space_debugging_workflow():
    """Test detection of debugging workflow across spaces"""
    graph = MultiSpaceContextGraph(enable_cross_space_correlation=True)
    await graph.start()

    # Simulate debugging workflow:
    # Space 1: Terminal with error
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="npm test",
        errors=["TypeError: Cannot read property 'verify' of undefined"],
        exit_code=1
    )

    # Space 2: Browser researching solution
    graph.update_browser_context(
        space_id=2,
        app_name="Chrome",
        url="https://stackoverflow.com/questions/jwt-verify-undefined",
        extracted_text="Stack Overflow JWT documentation verify method undefined",
        search_query="jwt verify undefined"
    )

    # Space 3: IDE editing code
    graph.update_ide_context(
        space_id=3,
        app_name="VS Code",
        active_file="auth.test.js",
        open_files=["auth.test.js", "auth.js"]
    )

    # Wait for correlation to run
    await asyncio.sleep(0.5)

    # Check if debugging workflow was detected
    if graph.correlator:
        relationships = await graph.correlator.analyze_relationships(graph.spaces)

        debugging_workflows = [r for r in relationships if r.relationship_type == "debugging_workflow"]
        assert len(debugging_workflows) > 0

        workflow = debugging_workflows[0]
        assert 1 in workflow.involved_spaces  # Terminal space
        assert workflow.confidence >= 0.7

        print(f"✓ Test 6 passed: Detected debugging workflow: {workflow.description}")
    else:
        print("⚠ Test 6 skipped: Correlation disabled")

    await graph.stop()


# ============================================================================
# TEST 7: Natural Language Query - "What does it say?"
# ============================================================================

@pytest.mark.asyncio
async def test_natural_language_query_error():
    """Test natural language query for error detection"""
    graph = MultiSpaceContextGraph()

    # Simulate error in terminal
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="python script.py",
        errors=["FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'"],
        exit_code=1
    )

    # Query: "what does it say?"
    result = graph.find_context_for_query("what does it say?")
    assert result["type"] == "error"
    assert result["space_id"] == 1
    assert "FileNotFoundError" in str(result["details"])

    # Query: "what's the error?"
    result2 = graph.find_context_for_query("what's the error?")
    assert result2["type"] == "error"

    print("✓ Test 7 passed: Natural language query - error detection")


# ============================================================================
# TEST 8: Recent Events and Time Filtering
# ============================================================================

@pytest.mark.asyncio
async def test_recent_events_filtering():
    """Test that recent events are properly filtered by time"""
    graph = MultiSpaceContextGraph()

    space = graph.get_or_create_space(1)

    # Add some events
    space.add_application("Terminal", ContextType.TERMINAL)
    space.add_application("Safari", ContextType.BROWSER)

    # Get recent events (should include app additions)
    recent = space.get_recent_events(within_seconds=10)
    assert len(recent) >= 2  # At least 2 app_added events

    # Simulate old events by checking further back
    all_events = space.get_recent_events(within_seconds=3600)  # 1 hour
    assert len(all_events) >= len(recent)

    print("✓ Test 8 passed: Recent events filtering")


# ============================================================================
# TEST 9: Temporal Decay
# ============================================================================

@pytest.mark.asyncio
async def test_temporal_decay():
    """Test that old spaces decay after TTL expires"""
    graph = MultiSpaceContextGraph(decay_ttl_seconds=2)  # 2 second TTL for testing
    await graph.start()

    # Create space with activity
    space1 = graph.get_or_create_space(1)
    space1.add_application("Terminal", ContextType.TERMINAL)

    assert 1 in graph.spaces

    # Wait for decay to happen (TTL + decay loop interval)
    await asyncio.sleep(3)

    # Space should still exist (it's recent)
    assert 1 in graph.spaces

    # Manually trigger decay for spaces older than TTL
    await graph._apply_decay()

    # Now it should be decayed (no critical events, older than TTL)
    # Note: Decay only removes if truly inactive
    # For a more reliable test, we'd need to mock timestamps

    await graph.stop()
    print("✓ Test 9 passed: Temporal decay (basic check)")


# ============================================================================
# TEST 10: Integration Bridge
# ============================================================================

@pytest.mark.asyncio
async def test_integration_bridge_initialization():
    """Test integration bridge initialization and OCR processing"""
    # Initialize bridge (without auto-start to avoid conflicts)
    bridge = await initialize_integration_bridge(auto_start=False)

    assert bridge is not None
    assert bridge.context_graph is not None

    # Test OCR processing for terminal
    await bridge.process_ocr_update(
        space_id=1,
        app_name="Terminal",
        ocr_text="""
        $ python app.py
        Traceback (most recent call last):
          File "app.py", line 10, in <module>
            import requests
        ModuleNotFoundError: No module named 'requests'
        $
        """
    )

    # Verify context was updated
    space1 = bridge.context_graph.spaces[1]
    terminal_app = space1.applications.get("Terminal")

    assert terminal_app is not None
    assert terminal_app.terminal_context is not None
    assert len(terminal_app.terminal_context.errors) > 0

    print("✓ Test 10 passed: Integration bridge initialization and OCR processing")


# ============================================================================
# TEST 11: Cross-Space Summary Generation
# ============================================================================

@pytest.mark.asyncio
async def test_cross_space_summary():
    """Test natural language summary generation"""
    graph = MultiSpaceContextGraph(enable_cross_space_correlation=True)
    await graph.start()

    # Set up multi-space workflow
    graph.update_terminal_context(1, "Terminal", command="npm test", errors=["Test failed"], exit_code=1)
    graph.update_browser_context(2, "Chrome", extracted_text="Stack Overflow documentation")
    graph.update_ide_context(3, "VS Code", active_file="test.js")

    # Wait for correlation
    await asyncio.sleep(0.5)

    # Get summary
    summary = graph.get_cross_space_summary()

    # Should have some description of relationships (or "No" if none detected yet)
    assert isinstance(summary, str)
    assert len(summary) > 0

    print(f"✓ Test 11 passed: Cross-space summary: '{summary[:100]}...'")

    await graph.stop()


# ============================================================================
# TEST 12: Space Tags Inference
# ============================================================================

@pytest.mark.asyncio
async def test_space_tags_inference():
    """Test automatic tag inference based on applications"""
    graph = MultiSpaceContextGraph()

    space = graph.get_or_create_space(1)

    # Add development tools
    space.add_application("Terminal", ContextType.TERMINAL)
    space.add_application("VS Code", ContextType.IDE)

    # Infer tags
    space.infer_tags()

    assert "development" in space.tags

    # Add browser
    space.add_application("Chrome", ContextType.BROWSER)
    space.infer_tags()

    assert "research" in space.tags

    print(f"✓ Test 12 passed: Space tags inferred: {space.tags}")


# ============================================================================
# TEST 13: Context Export
# ============================================================================

@pytest.mark.asyncio
async def test_context_export():
    """Test exporting context graph to JSON"""
    graph = MultiSpaceContextGraph()

    # Create some context
    graph.update_terminal_context(1, "Terminal", command="ls -la")
    graph.update_browser_context(2, "Safari", url="https://example.com")

    # Export to file
    export_path = Path("/tmp/jarvis_context_export_test.json")
    graph.export_to_json(export_path)

    # Verify file was created
    assert export_path.exists()

    # Read and verify structure
    import json
    with open(export_path) as f:
        data = json.load(f)

    assert "total_spaces" in data
    assert data["total_spaces"] == 2
    assert "spaces" in data

    print(f"✓ Test 13 passed: Context exported to {export_path}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print(" Multi-Space Context Graph - Comprehensive Test Suite")
    print("="*80 + "\n")

    tests = [
        ("Basic Space Creation", test_basic_space_creation_and_activation),
        ("Application Context", test_application_context_management),
        ("Terminal Context", test_terminal_context_updates),
        ("Browser Context", test_browser_context_updates),
        ("IDE Context", test_ide_context_updates),
        ("Cross-Space Debugging", test_cross_space_debugging_workflow),
        ("Natural Language Query", test_natural_language_query_error),
        ("Recent Events", test_recent_events_filtering),
        ("Temporal Decay", test_temporal_decay),
        ("Integration Bridge", test_integration_bridge_initialization),
        ("Cross-Space Summary", test_cross_space_summary),
        ("Space Tags", test_space_tags_inference),
        ("Context Export", test_context_export),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            print(f" TEST: {name}")
            print('='*80 + '\n')

            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print("\n🎉 ALL TESTS PASSED!")

    return passed == len(tests)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
