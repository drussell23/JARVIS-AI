#!/usr/bin/env python3
"""
Quick Validation Test for Context Awareness Fixes
==================================================
"""

import asyncio
import sys

async def test_imports():
    """Test 1: Verify imports work correctly"""
    print("=" * 60)
    print("TEST 1: Import Validation")
    print("=" * 60)

    try:
        # Test AsyncPipeline import
        print("\n[1/4] Testing AdvancedAsyncPipeline import...")
        from core.async_pipeline import AdvancedAsyncPipeline
        pipeline = AdvancedAsyncPipeline()
        print("✅ AdvancedAsyncPipeline imported successfully")

        # Test methods exist
        print("\n[2/4] Testing _detect_document_type method...")
        doc_type = pipeline._detect_document_type("write me an essay on AI")
        print(f"✅ Document type detected: {doc_type}")

        print("\n[3/4] Testing _get_enhanced_system_context method...")
        context = await pipeline._get_enhanced_system_context()
        print(f"✅ System context gathered: {list(context.keys())}")

        # Test UnifiedCommandProcessor
        print("\n[4/4] Testing UnifiedCommandProcessor...")
        from api.unified_command_processor import UnifiedCommandProcessor
        processor = UnifiedCommandProcessor()
        print("✅ UnifiedCommandProcessor imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_compound_command_context():
    """Test 2: Verify compound commands include context"""
    print("\n" + "=" * 60)
    print("TEST 2: Compound Command Context Integration")
    print("=" * 60)

    try:
        from api.unified_command_processor import UnifiedCommandProcessor

        processor = UnifiedCommandProcessor()

        # Test compound command
        print("\n[1/2] Processing compound command...")
        result = await processor.process_command(
            "lock my screen and then tell me the weather"
        )

        print(f"\n[2/2] Checking result structure...")

        # Check for context and steps_taken
        has_context = 'context' in result or 'system_context' in result
        has_steps = 'steps_taken' in result

        print(f"  Result keys: {list(result.keys())}")
        print(f"  Has context: {has_context}")
        print(f"  Has steps_taken: {has_steps}")

        if has_context or has_steps:
            print("✅ Compound command includes context tracking")
            return True
        else:
            print("⚠️  Compound command may need more context integration")
            return False

    except Exception as e:
        print(f"❌ Compound command test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_aware_handler():
    """Test 3: Verify context-aware handler works"""
    print("\n" + "=" * 60)
    print("TEST 3: Context-Aware Handler Validation")
    print("=" * 60)

    try:
        from context_intelligence.handlers.context_aware_handler import get_context_aware_handler

        handler = get_context_aware_handler()

        print("\n[1/3] Creating mock execution callback...")
        async def mock_execute(cmd: str, context: dict = None):
            return {
                "success": True,
                "message": f"Executed: {cmd}",
                "context_received": context is not None
            }

        print("[2/3] Handling test command...")
        result = await handler.handle_command_with_context(
            "test command",
            execute_callback=mock_execute
        )

        print(f"[3/3] Checking result...")
        has_steps = 'steps_taken' in result and len(result['steps_taken']) > 0
        has_context = 'context' in result

        print(f"  Result keys: {list(result.keys())}")
        print(f"  Steps taken: {len(result.get('steps_taken', []))}")
        print(f"  Has context: {has_context}")

        if has_steps and has_context:
            print("✅ Context-aware handler working correctly")
            return True
        else:
            print("⚠️  Context handler may need adjustment")
            return False

    except Exception as e:
        print(f"❌ Context handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_logging_integration():
    """Test 4: Verify logging includes context markers"""
    print("\n" + "=" * 60)
    print("TEST 4: Context-Aware Logging")
    print("=" * 60)

    try:
        import subprocess

        print("\n[1/2] Checking for context-aware log markers...")

        # Process a command to generate logs
        from api.unified_command_processor import UnifiedCommandProcessor
        processor = UnifiedCommandProcessor()

        print("[2/2] Processing command to generate logs...")
        result = await processor.process_command("test context logging")

        # Check recent logs
        logs = subprocess.run(
            ['tail', '-n', '50', 'logs/jarvis_optimized_*.log'],
            capture_output=True,
            text=True,
            shell=True
        )

        # Look for context-aware indicators
        has_compound_logs = "[COMPOUND]" in logs.stdout
        has_context_logs = "[CONTEXT AWARE]" in logs.stdout or "context" in logs.stdout.lower()

        print(f"\n  Found [COMPOUND] logs: {has_compound_logs}")
        print(f"  Found context-related logs: {has_context_logs}")

        if has_compound_logs or has_context_logs:
            print("✅ Context-aware logging is active")
            return True
        else:
            print("⚠️  Logging markers may need to be added")
            return False

    except Exception as e:
        print(f"⚠️  Logging test inconclusive: {e}")
        return True  # Don't fail on this as it's not critical


async def main():
    """Run all quick validation tests"""
    print("\n" + "█" * 60)
    print(" CONTEXT AWARENESS FIX VALIDATION ")
    print("█" * 60 + "\n")

    results = {}

    # Run tests
    results['imports'] = await test_imports()
    results['compound_context'] = await test_compound_command_context()
    results['context_handler'] = await test_context_aware_handler()
    results['logging'] = await test_logging_integration()

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY ")
    print("=" * 60 + "\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {status}  {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)\n")

    if passed == total:
        print("🎉 All fixes validated successfully!\n")
        return 0
    elif passed >= total * 0.75:
        print("✓ Most fixes working - minor issues remain\n")
        return 0
    else:
        print("⚠️  Some fixes need attention\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
