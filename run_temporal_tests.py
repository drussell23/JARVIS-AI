#!/usr/bin/env python3
"""
Simple test runner for TemporalQueryHandler v3.0 tests

This script runs the tests without complex imports.
"""

import subprocess
import sys

def run_tests():
    """Run TemporalQueryHandler tests"""

    print("=" * 60)
    print("JARVIS v3.0 - TemporalQueryHandler Test Suite")
    print("=" * 60)
    print()

    tests = [
        # Unit tests
        ("Unit Tests - Enums", "tests/unit/backend/test_temporal_query_handler_v3.py::TestTemporalQueryHelpers::test_change_type_string_conversion"),
        ("Unit Tests - Query Types", "tests/unit/backend/test_temporal_query_handler_v3.py::TestTemporalQueryHelpers::test_query_type_string_conversion"),

        # Integration tests
        ("E2E - Import Verification", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_temporal_handler_import"),
        ("E2E - V3 Enums", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_v3_enums_exist"),
        ("E2E - main.py Integration", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_main_py_imports_temporal_handler"),
        ("E2E - Alert Queues", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_alert_queues_have_correct_sizes"),
        ("E2E - Methods Exist", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_handler_has_v3_methods"),
        ("E2E - Attributes Exist", "tests/e2e/test_temporal_handler_usage_verification.py::TestTemporalHandlerUsageVerification::test_handler_has_v3_attributes"),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_path in tests:
        print(f"\n[TEST] {name}")
        print("-" * 60)

        result = subprocess.run(
            ['python', '-m', 'pytest', test_path, '-xvs', '--tb=short'],
            cwd='/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent',
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ PASSED: {name}")
            passed += 1
        elif 'SKIPPED' in result.stdout or 'skipped' in result.stdout:
            print(f"⏭️  SKIPPED: {name}")
            skipped += 1
        else:
            print(f"❌ FAILED: {name}")
            print("\nError output:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed:  {passed}")
    print(f"❌ Failed:  {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"📊 Total:   {passed + failed + skipped}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(run_tests())
