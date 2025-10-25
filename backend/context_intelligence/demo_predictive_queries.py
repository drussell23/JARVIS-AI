#!/usr/bin/env python3
"""
Demo Script for JARVIS Predictive Analytics Engine
==================================================

This demonstrates the predictive/analytical query system with various examples.

Run with:
    python backend/context_intelligence/demo_predictive_queries.py

Or run specific scenarios:
    python backend/context_intelligence/demo_predictive_queries.py --scenario progress
    python backend/context_intelligence/demo_predictive_queries.py --scenario bugs
    python backend/context_intelligence/demo_predictive_queries.py --scenario next_steps
    python backend/context_intelligence/demo_predictive_queries.py --scenario all

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.context_intelligence.analyzers.predictive_analyzer import (
    PredictiveAnalyzer,
    initialize_predictive_analyzer
)
from backend.context_intelligence.handlers.predictive_query_handler import (
    PredictiveQueryHandler,
    PredictiveQueryRequest,
    initialize_predictive_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

async def demo_progress_check():
    """Demo: Progress check query"""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Progress Check")
    print("=" * 80)
    print("\nQuery: 'Am I making progress?'\n")

    handler = get_predictive_handler()

    response = await handler.check_progress(repo_path=".")

    print_response(response)


async def demo_next_steps():
    """Demo: Next steps recommendation"""
    print("\n" + "=" * 80)
    print("SCENARIO 2: Next Steps")
    print("=" * 80)
    print("\nQuery: 'What should I work on next?'\n")

    handler = get_predictive_handler()

    response = await handler.get_next_steps(repo_path=".")

    print_response(response)


async def demo_bug_detection():
    """Demo: Bug detection"""
    print("\n" + "=" * 80)
    print("SCENARIO 3: Bug Detection")
    print("=" * 80)
    print("\nQuery: 'Are there any potential bugs?'\n")

    handler = get_predictive_handler()

    # First, simulate some errors
    analyzer = handler.analyzer
    analyzer.record_error(
        error_type="ImportError",
        location="/Users/test/project/main.py:15",
        details={"message": "No module named 'foo'", "severity": "high"}
    )
    analyzer.record_error(
        error_type="ImportError",
        location="/Users/test/project/utils.py:8",
        details={"message": "No module named 'foo'", "severity": "high"}
    )
    analyzer.record_error(
        error_type="TypeError",
        location="/Users/test/project/handler.py:42",
        details={"message": "unsupported operand type(s)", "severity": "medium"}
    )

    print("ℹ️  (Simulated 3 errors for demo purposes)")

    response = await handler.detect_bugs()

    print_response(response)


async def demo_workflow_optimization():
    """Demo: Workflow optimization"""
    print("\n" + "=" * 80)
    print("SCENARIO 4: Workflow Optimization")
    print("=" * 80)
    print("\nQuery: 'How can I improve my workflow?'\n")

    handler = get_predictive_handler()

    # Simulate some workflow activities
    analyzer = handler.analyzer
    for i in range(50):
        analyzer.record_activity(
            activity_type="space_switch" if i % 10 == 0 else "code_edit",
            details={"space_id": (i % 3) + 1, "app_name": "VSCode"}
        )

    print("ℹ️  (Simulated 50 workflow activities for demo purposes)")

    request = PredictiveQueryRequest(
        query="How can I improve my workflow?"
    )

    response = await handler.handle_query(request)

    print_response(response)


async def demo_custom_query():
    """Demo: Custom query"""
    print("\n" + "=" * 80)
    print("SCENARIO 5: Custom Query")
    print("=" * 80)

    handler = get_predictive_handler()

    queries = [
        "How's my code quality?",
        "What patterns do you see?",
        "Am I being productive?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'\n")

        request = PredictiveQueryRequest(query=query)
        response = await handler.handle_query(request)

        print_response(response, compact=True)
        print()


async def demo_with_vision():
    """Demo: Query with Claude Vision (if available)"""
    print("\n" + "=" * 80)
    print("SCENARIO 6: Query with Claude Vision")
    print("=" * 80)
    print("\nQuery: 'Explain what this code does' (with vision)\n")

    handler = get_predictive_handler()

    # Check if vision is available
    if not handler.vision_analyzer or not handler.vision_analyzer._claude_available:
        print("⚠️  Claude Vision not available (anthropic library not installed)")
        print("Install with: pip install anthropic")
        return

    request = PredictiveQueryRequest(
        query="Explain what this code does",
        use_vision=True,
        capture_screen=True
    )

    response = await handler.handle_query(request)

    print_response(response)


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

async def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "=" * 80)
    print("JARVIS Predictive Analytics - Interactive Mode")
    print("=" * 80)
    print("\nType your queries below (or 'quit' to exit)")
    print("Examples:")
    print("  - Am I making progress?")
    print("  - What should I work on next?")
    print("  - Are there any bugs?")
    print("  - How can I improve my workflow?")
    print()

    handler = get_predictive_handler()

    while True:
        try:
            query = input("\n📊 Query: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            request = PredictiveQueryRequest(query=query)
            response = await handler.handle_query(request)

            print()
            print_response(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================================
# HELPERS
# ============================================================================

def print_response(response, compact: bool = False):
    """Pretty print a response"""
    if not response.success:
        print(f"❌ Error: {response.response_text}")
        return

    if compact:
        # Compact mode - just show key info
        if response.analytics:
            print(f"📊 {response.analytics.query_type.value}")
            if response.analytics.insights:
                print(f"💡 {response.analytics.insights[0]}")
    else:
        # Full mode
        print(response.response_text)

        # Show metadata
        print(f"\n{'─' * 80}")
        print(f"✓ Success | Confidence: {response.confidence:.0%} | Query Type: {response.analytics.query_type.value if response.analytics else 'unknown'}")

        if response.vision_analysis:
            print(f"👁️  Vision Analysis: {'✓' if response.vision_analysis.get('success') else '✗'}")


def get_predictive_handler() -> PredictiveQueryHandler:
    """Get or create the predictive handler"""
    from backend.context_intelligence.handlers.predictive_query_handler import get_predictive_handler, initialize_predictive_handler

    handler = get_predictive_handler()
    if not handler:
        handler = initialize_predictive_handler()

    return handler


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Demo JARVIS Predictive Analytics")
    parser.add_argument(
        "--scenario",
        choices=["progress", "next_steps", "bugs", "workflow", "custom", "vision", "interactive", "all"],
        default="all",
        help="Which scenario to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 80)
    print("JARVIS Predictive Analytics Engine - Demo")
    print("=" * 80)
    print("\nInitializing system...")

    # Initialize the handler
    handler = get_predictive_handler()

    print("✓ System initialized\n")

    # Run scenarios
    if args.scenario == "interactive":
        await interactive_mode()
    elif args.scenario == "all":
        await demo_progress_check()
        await asyncio.sleep(0.5)

        await demo_next_steps()
        await asyncio.sleep(0.5)

        await demo_bug_detection()
        await asyncio.sleep(0.5)

        await demo_workflow_optimization()
        await asyncio.sleep(0.5)

        await demo_custom_query()
        await asyncio.sleep(0.5)

        await demo_with_vision()
    elif args.scenario == "progress":
        await demo_progress_check()
    elif args.scenario == "next_steps":
        await demo_next_steps()
    elif args.scenario == "bugs":
        await demo_bug_detection()
    elif args.scenario == "workflow":
        await demo_workflow_optimization()
    elif args.scenario == "custom":
        await demo_custom_query()
    elif args.scenario == "vision":
        await demo_with_vision()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
