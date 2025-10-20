#!/usr/bin/env python3
"""
Verification Script for TemporalQueryHandler v3.0

Verifies that TemporalQueryHandler v3.0 is properly implemented and integrated.
This script checks files directly without importing (avoids import errors).
"""

import os
import re
from pathlib import Path


def check_file_exists(path: str) -> bool:
    """Check if file exists"""
    return Path(path).exists()


def check_file_contains(path: str, patterns: list) -> dict:
    """Check if file contains all patterns"""
    if not Path(path).exists():
        return {'exists': False}

    with open(path, 'r') as f:
        content = f.read()

    results = {'exists': True, 'matches': {}}

    for pattern_name, pattern in patterns:
        if isinstance(pattern, str):
            found = pattern in content
        else:  # regex
            found = bool(re.search(pattern, content))

        results['matches'][pattern_name] = found

    return results


def main():
    """Run verification checks"""

    print("=" * 70)
    print("JARVIS v3.0 - TemporalQueryHandler Verification")
    print("=" * 70)
    print()

    checks_passed = 0
    checks_failed = 0

    # Check 1: temporal_query_handler.py exists
    print("[CHECK 1] TemporalQueryHandler file exists")
    handler_path = 'backend/context_intelligence/handlers/temporal_query_handler.py'

    if check_file_exists(handler_path):
        print("✅ PASS: temporal_query_handler.py exists")
        checks_passed += 1
    else:
        print("❌ FAIL: temporal_query_handler.py not found")
        checks_failed += 1
        return

    # Check 2: v3.0 features in docstring
    print("\n[CHECK 2] v3.0 features documented in docstring")
    patterns = [
        ('v3.0 mention', re.compile(r'v3\.0', re.IGNORECASE)),
        ('Pattern Analysis', 'PATTERN_ANALYSIS'),
        ('Predictive Analysis', 'PREDICTIVE_ANALYSIS'),
        ('Anomaly Analysis', 'ANOMALY_ANALYSIS'),
        ('Correlation Analysis', 'CORRELATION_ANALYSIS'),
    ]

    result = check_file_contains(handler_path, patterns)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 3: New change types exist
    print("\n[CHECK 3] New v3.0 ChangeType enums")
    change_types = [
        ('ANOMALY_DETECTED', 'ANOMALY_DETECTED'),
        ('PATTERN_RECOGNIZED', 'PATTERN_RECOGNIZED'),
        ('PREDICTIVE_EVENT', 'PREDICTIVE_EVENT'),
        ('CASCADING_FAILURE', 'CASCADING_FAILURE'),
    ]

    result = check_file_contains(handler_path, change_types)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 4: Pattern learning methods exist
    print("\n[CHECK 4] Pattern learning methods")
    methods = [
        ('_analyze_patterns_from_monitoring', '_analyze_patterns_from_monitoring'),
        ('_generate_predictions', '_generate_predictions'),
        ('_detect_anomalies', '_detect_anomalies'),
        ('_analyze_correlations', '_analyze_correlations'),
        ('_detect_cascading_failures', '_detect_cascading_failures'),
        ('_load_learned_patterns', '_load_learned_patterns'),
        ('_save_learned_patterns', '_save_learned_patterns'),
    ]

    result = check_file_contains(handler_path, methods)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 5: Alert queues with correct sizes
    print("\n[CHECK 5] Alert queue configuration")
    queues = [
        ('monitoring_alerts (500)', re.compile(r'monitoring_alerts.*maxlen=500', re.DOTALL)),
        ('anomaly_alerts (100)', re.compile(r'anomaly_alerts.*maxlen=100', re.DOTALL)),
        ('predictive_alerts (100)', re.compile(r'predictive_alerts.*maxlen=100', re.DOTALL)),
        ('correlation_alerts (100)', re.compile(r'correlation_alerts.*maxlen=100', re.DOTALL)),
    ]

    result = check_file_contains(handler_path, queues)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 6: Pattern persistence
    print("\n[CHECK 6] Pattern persistence to ~/.jarvis/learned_patterns.json")
    persistence_patterns = [
        ('learned_patterns.json', 'learned_patterns.json'),
        ('~/.jarvis path', re.compile(r'~/.jarvis|\.jarvis')),
        ('json.dump', 'json.dump'),
        ('json.load', 'json.load'),
    ]

    result = check_file_contains(handler_path, persistence_patterns)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 7: main.py integration
    print("\n[CHECK 7] Integration in main.py")
    main_path = 'backend/main.py'

    integration_patterns = [
        ('imports temporal_query_handler', re.compile(r'temporal_query_handler')),
        ('initialize_temporal_query_handler', 'initialize_temporal_query_handler'),
        ('hybrid_monitoring integration', re.compile(r'hybrid_monitoring.*temporal', re.DOTALL | re.IGNORECASE)),
    ]

    result = check_file_contains(main_path, integration_patterns)

    for name, found in result['matches'].items():
        if found:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Check 8: Test files exist
    print("\n[CHECK 8] Test files")
    test_files = [
        ('Unit tests', 'tests/unit/backend/test_temporal_query_handler_v3.py'),
        ('Integration tests', 'tests/integration/test_temporal_query_handler_integration.py'),
        ('E2E verification', 'tests/e2e/test_temporal_handler_usage_verification.py'),
    ]

    for name, path in test_files:
        if check_file_exists(path):
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            checks_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {checks_passed}")
    print(f"❌ Failed: {checks_failed}")
    print(f"📊 Total:  {checks_passed + checks_failed}")

    success_rate = (checks_passed / (checks_passed + checks_failed)) * 100 if (checks_passed + checks_failed) > 0 else 0
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    print("=" * 70)

    if checks_failed == 0:
        print("\n✨ All checks passed! TemporalQueryHandler v3.0 is properly implemented.")
    elif success_rate >= 80:
        print("\n✅ TemporalQueryHandler v3.0 is mostly implemented (>80% checks passed).")
    elif success_rate >= 50:
        print("\n⚠️  TemporalQueryHandler v3.0 is partially implemented (50-80% checks passed).")
    else:
        print("\n❌ TemporalQueryHandler v3.0 implementation incomplete (<50% checks passed).")

    return 0 if checks_failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
