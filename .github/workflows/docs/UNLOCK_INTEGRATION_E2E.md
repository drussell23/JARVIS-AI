# Unlock Integration E2E Testing - Advanced Async Edition

## Overview

Comprehensive, asynchronous end-to-end testing for screen unlock integration with **zero hardcoding**, **concurrent execution**, and **advanced async patterns**.

## 🚀 Features

### ✨ Advanced Async Capabilities

- **Concurrent Test Execution:**
  - Multiple tests run in parallel using asyncio
  - Configurable concurrency limits (default: 5 concurrent tests)
  - Semaphore-based resource management
  - ThreadPoolExecutor for CPU-bound operations

- **Advanced Timeout Handling:**
  - Per-test timeouts with detailed error reporting
  - asyncio.wait_for() for precise timeout control
  - Graceful timeout recovery and cleanup

- **Async File I/O:**
  - aiofiles for non-blocking file operations
  - Async report writing
  - Fallback to sync I/O if aiofiles unavailable

- **Resource Pooling:**
  - Async locks for thread-safe result recording
  - Executor cleanup and resource management
  - Proper async context manager usage

### 🎯 Test Modes

| Mode | Concurrency | Safety | Real Locks | Use Case |
|------|-------------|--------|------------|----------|
| **Mock** | 5 tests | 🟢 Safe | No | CI/CD, PRs, fast validation |
| **Integration** | 3 tests | 🟡 Safe | No | macOS logic testing |
| **Real** | 1 test | 🔴 Careful | Yes | Final validation (self-hosted) |

### 📊 Async Metrics Tracking

Each test tracks:
- **Wall clock duration**: Total time including I/O
- **Async duration**: Pure async operation time
- **Async overhead**: Difference between wall clock and async time
- **Started/completed timestamps**: Full test lifecycle
- **Concurrent execution stats**: Tests running in parallel

## 🧪 Test Suites

### 1. Keychain Retrieval
**What:** Tests password retrieval from macOS Keychain
**Timeout:** 30s
**Async:** ✅ Full async with asyncio.wait_for()

```python
await asyncio.wait_for(
    unlock.get_password_from_keychain(),
    timeout=10.0
)
```

### 2. Unlock Logic
**What:** Tests unlock flow and logic
**Timeout:** 45s
**Async:** ✅ Concurrent execution enabled

### 3. Secure Password Typer
**What:** Tests Core Graphics secure password typing
**Timeout:** 60s
**Async:** ✅ Full async with metrics tracking
**Tests:**
- Core Graphics availability
- Typing configuration
- Memory security
- Fallback mechanisms

### 4. Intelligent Voice Service
**What:** Tests intelligent voice unlock service integration
**Timeout:** 90s
**Async:** ✅ Service initialization and availability
**Tests:**
- Service availability
- Initialization status
- Component integration

### 5. Screen Detector Integration
**What:** Tests screen lock detector from voice_unlock
**Timeout:** 30s
**Async:** ✅ Multi-method detection
**Tests:**
- CGSession detection
- Screensaver detection
- LoginWindow detection
- Detailed state retrieval

### 6. Adaptive Timing
**What:** Tests adaptive timing system
**Timeout:** 45s
**Async:** ✅ System load detection
**Tests:**
- System load detection (psutil/uptime)
- Timing adjustment logic
- Performance under load

### 7. Memory Security
**What:** Tests secure memory handling
**Timeout:** 30s
**Async:** ✅ Memory operations
**Tests:**
- Password obfuscation
- Secure memory clearing
- 3-pass overwrite
- Garbage collection

### 8. Fallback Mechanisms
**What:** Tests 3-tier fallback chain
**Timeout:** 60s
**Async:** ✅ Chain validation
**Tests:**
- intelligent_voice_unlock_service availability
- macos_keychain_unlock availability
- macos_controller availability
- Fallback priority order

### 9. Error Handling
**What:** Tests error scenarios and recovery
**Timeout:** 40s
**Async:** ✅ Concurrent error simulation

### 10. Performance
**What:** Tests unlock performance across multiple cycles
**Timeout:** 120s
**Async:** ✅ Parallel performance testing
**Metrics:**
- Average unlock time
- Min/max unlock time
- Performance vs baseline
- System load impact

### 11. Security Checks
**What:** Comprehensive security validation
**Timeout:** 60s
**Async:** ⚠️  Sequential (must run after others)
**Checks:**
- No hardcoded passwords
- Keychain usage verified
- Core Graphics usage
- Secure memory handling
- Log sanitization
- Secure communication

### 12. Full E2E
**What:** Complete unlock cycle (real mode only)
**Timeout:** 180s
**Async:** ⚠️  Sequential (safety)

## 🚀 Usage

### Quick Start

**Run Mock Tests (Fast, Concurrent):**
```bash
gh workflow run unlock-integration-e2e.yml
```

**Run Integration Tests (macOS, Moderate Concurrency):**
```bash
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration
```

**Run Real E2E Tests (Self-Hosted, Sequential):**
```bash
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=real
```

### Advanced Usage

**With Custom Concurrency:**
```bash
# Mock mode with 10 concurrent tests
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=mock

# Set MAX_CONCURRENT env var in workflow
```

**Stress Testing:**
```bash
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration \
  -f stress_test=true \
  -f unlock_cycles=20
```

**Performance Baseline Testing:**
```bash
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration \
  -f performance_baseline=1500
```

**Single Test Suite:**
```bash
# Only test keychain retrieval
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration

# Matrix will run each suite separately
```

## 📈 Async Performance

### Execution Flow

```
Mock Mode (5 concurrent):
┌─────────────────────────────────────────────┐
│ Keychain │ Unlock │ AppleScript │ Screen │ Error │
│ (30s)    │ (45s)  │ (20s)       │ (30s)  │ (40s) │
└─────────────────────────────────────────────┘
         ↓ (all run concurrently)
     Performance (120s) ← Sequential
         ↓
     Security (60s) ← Sequential

Total: ~210s (vs ~245s sequential) = 14% faster
```

### Integration Mode (3 concurrent):
```
┌───────────────────────────────────┐
│ Test 1 │ Test 2 │ Test 3 │
│ (30s)  │ (45s)  │ (20s)  │
└───────────────────────────────────┘
      ↓
  Performance (120s)
      ↓
  Security (60s)

Total: ~225s (vs ~285s sequential) = 21% faster
```

### Real Mode (1 concurrent):
```
Test 1 → Test 2 → Full E2E (180s)

Total: Variable (safety first)
```

## 📊 Report Format

### JSON Report
```json
{
  "mode": "mock",
  "timestamp": "2025-10-30T12:34:56",
  "duration_ms": 12450.5,
  "async_metrics": {
    "concurrent_execution": 5,
    "avg_async_overhead_ms": 2.3,
    "tests_with_metrics": 7
  },
  "summary": {
    "total": 7,
    "passed": 7,
    "failed": 0,
    "success_rate": 100.0
  },
  "tests": [
    {
      "name": "keychain_retrieval",
      "success": true,
      "duration_ms": 52.3,
      "message": "Retrieved password (length: 15)",
      "started_at": "2025-10-30T12:34:56",
      "completed_at": "2025-10-30T12:34:56",
      "async_metrics": {
        "async_duration_ms": 50.1,
        "wall_clock_ms": 52.3
      }
    }
  ]
}
```

### Text Summary
```
Unlock Integration E2E Test Summary
===================================
Mode: MOCK
Timestamp: 2025-10-30T12:34:56
Duration: 12450.5ms
Max Concurrent: 5

Results:
  Total:   7
  Passed:  7 ✅
  Failed:  0 ❌
  Success: 100.0%

Async Metrics:
  Concurrent Tests: 5
  Avg Overhead:     2.3ms
  Tests w/Metrics:  7
```

## 🔐 Security

### Keychain Requirements

**For Integration/Real Testing:**
```bash
# Password must be stored in keychain
security add-generic-password \
  -s "com.jarvis.voiceunlock" \
  -a "unlock_token" \
  -w "${PASSWORD}"
```

### CI/CD Safety

**GitHub Actions (Hosted):**
- Automatically uses mock mode for PRs
- No credentials needed
- 5 concurrent tests for fast feedback

**Self-Hosted Runners:**
- Can run real tests
- Requires keychain setup
- Sequential execution for safety (MAX_CONCURRENT=1)

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODE` | `mock` | Test execution mode |
| `TEST_DURATION` | `600` | Max test duration (seconds) |
| `UNLOCK_CYCLES` | `5` | Number of unlock cycles |
| `STRESS_TEST` | `false` | Enable stress testing |
| `KEYCHAIN_TEST` | `true` | Include keychain tests |
| `PERFORMANCE_BASELINE` | `2000` | Performance baseline (ms) |
| `MAX_CONCURRENT` | `5` | Max concurrent tests |
| `RESULTS_DIR` | `test-results/unlock-e2e` | Results directory |

### Workflow Inputs

**test_mode:**
- Options: `mock`, `integration`, `real`
- Default: `mock`

**test_duration:**
- Range: 60-3600 seconds
- Default: 600

**unlock_cycles:**
- Range: 1-50
- Default: 5

**stress_test:**
- Type: boolean
- Default: false

**keychain_test:**
- Type: boolean
- Default: true

**performance_baseline:**
- Range: 500-10000 ms
- Default: 2000

## 🛠️ Troubleshooting

### Issue: Tests Timeout

**Cause:** Network latency, slow I/O, or actual hangs
**Solution:** Check individual test timeouts and async metrics

```bash
# Check async overhead in report
cat test-results/unlock-e2e/report-*.json | jq '.async_metrics'
```

### Issue: Concurrent Tests Fail

**Cause:** Resource contention or race conditions
**Solution:** Reduce MAX_CONCURRENT

```yaml
env:
  MAX_CONCURRENT: 3  # Reduce from 5
```

### Issue: Async Overhead High

**Cause:** Event loop congestion or blocking operations
**Solution:** Review test implementation for blocking calls

```python
# Bad: Blocking I/O
with open(file) as f:
    data = f.read()

# Good: Async I/O
async with aiofiles.open(file) as f:
    data = await f.read()
```

### Issue: Tests Hang in Integration Mode

**Cause:** Missing keychain entry on macOS runner
**Solution:** Verify keychain on runner

```bash
security find-generic-password -s "com.jarvis.voiceunlock"
```

## 📚 Best Practices

### 1. Start with Mock Mode
```bash
# Fast feedback, highly concurrent
gh workflow run unlock-integration-e2e.yml -f test_mode=mock
```

### 2. Use Integration for Logic Testing
```bash
# Test on macOS without real unlocks
gh workflow run unlock-integration-e2e.yml -f test_mode=integration
```

### 3. Reserve Real Testing for Final Validation
```bash
# Manual trigger, self-hosted only
gh workflow run unlock-integration-e2e.yml -f test_mode=real
```

### 4. Monitor Async Metrics
- Check `async_metrics` in reports
- Look for high overhead (>10ms indicates issues)
- Compare concurrent vs sequential timing

### 5. Adjust Concurrency Based on Mode
- Mock: High concurrency (5+) - no I/O
- Integration: Moderate (3-5) - some I/O
- Real: Sequential (1) - safety first

### 6. Run Performance Tests Periodically
```bash
# Weekly performance regression testing
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration \
  -f unlock_cycles=20 \
  -f performance_baseline=1500
```

## 🔄 Integration with CI/CD

### Pull Request Workflow
```yaml
on:
  pull_request:
    paths:
      - 'backend/core/async_pipeline.py'
      - 'backend/macos_keychain_unlock.py'
```

**What happens:**
1. Automatically runs mock tests
2. 5 concurrent tests for speed
3. Reports results on PR
4. Blocks merge if tests fail

### Daily Testing
```yaml
schedule:
  - cron: '0 4 * * *'  # 4 AM daily
```

**What happens:**
1. Runs integration tests on macOS
2. 3 concurrent tests
3. Creates issue if failures
4. Tracks async performance trends

### Manual Validation
```bash
# Before release
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration \
  -f stress_test=true \
  -f unlock_cycles=20
```

## 📝 Example Outputs

### Successful Run (Mock Mode, Concurrent)
```
🚀 Unlock Integration E2E Test Runner (Async Mode)
🚀 Starting Unlock Integration E2E Tests (Async Mode)
Mode: MOCK
Suite: all
Max Concurrent: 5

📊 Running 6 tests in parallel...
✅ keychain_retrieval: Mock keychain retrieval successful (52.3ms) [async: 50.1ms]
✅ unlock_logic: Mock unlock logic validated (105.7ms) [async: 103.2ms]
✅ applescript_typing: Mock AppleScript typing successful (82.1ms) [async: 80.5ms]
✅ screen_detection: Mock screen detection successful (31.4ms) [async: 30.1ms]
✅ error_handling: Error handling validated (54.8ms) [async: 52.3ms]
✅ performance: Avg: 52.3ms, Min: 48.1ms, Max: 58.7ms (1234.5ms) [async: 1230.1ms]

🔒 Running security checks...
✅ security_checks: Security checks: 4/4 passed (523.2ms)

================================================================================
📊 UNLOCK INTEGRATION E2E TEST REPORT (ASYNC)
================================================================================
Mode: MOCK
Duration: 1950.3ms
Concurrent: 5 tests
✅ Passed: 7
❌ Failed: 0
📈 Success Rate: 100.0%
⚡ Avg Async Overhead: 2.3ms
================================================================================
```

### Failed Run (Integration Mode)
```
🚀 Unlock Integration E2E Test Runner (Async Mode)
Mode: INTEGRATION
Max Concurrent: 3

📊 Running 6 tests in parallel...
✅ keychain_retrieval: Retrieved password (length: 15) (245.3ms)
❌ unlock_logic: Timeout after 10s (10000.0ms)
✅ screen_detection: Screen detection working (locked=false) (123.4ms)

================================================================================
📊 UNLOCK INTEGRATION E2E TEST REPORT (ASYNC)
================================================================================
Mode: INTEGRATION
Duration: 11523.7ms
Concurrent: 3 tests
✅ Passed: 2
❌ Failed: 1
📈 Success Rate: 66.7%
================================================================================

❌ 1 test(s) failed
```

## 🆘 Support

### Viewing Logs
```bash
# View workflow logs
gh run view --log

# View specific job
gh run view <run-id> --job test-mock-mode --log

# Download results
gh run download <run-id>
```

### Common Commands
```bash
# List recent runs
gh run list --workflow unlock-integration-e2e.yml --limit 10

# Re-run failed tests
gh run rerun <run-id> --failed

# Cancel running test
gh run cancel <run-id>
```

### Creating Issues

If tests consistently fail:
1. Check async metrics in report
2. Review concurrent execution logs
3. Verify keychain configuration
4. Create issue with:
   - Run ID
   - Async metrics (overhead, concurrency)
   - Error messages
   - Test mode used

## 📚 Resources

**Related Documentation:**
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [aiofiles](https://github.com/Tinche/aiofiles)
- [GitHub Actions](https://docs.github.com/en/actions)

**Project Files:**
- Workflow: `.github/workflows/unlock-integration-e2e.yml`
- Documentation: `.github/workflows/docs/UNLOCK_INTEGRATION_E2E.md`
- Backend: `backend/macos_keychain_unlock.py`
- Pipeline: `backend/core/async_pipeline.py`

---

**Last Updated:** 2025-10-30
**Version:** 2.0.0 (Async Edition)
**Status:** ✅ Production Ready with Advanced Async
