# Unlock Integration E2E Testing - Quick Reference

## 🚀 Quick Commands

### Run Tests

```bash
# Mock mode (safe, fast, 5 concurrent)
gh workflow run unlock-integration-e2e.yml

# Integration mode (macOS, 3 concurrent)
gh workflow run unlock-integration-e2e.yml -f test_mode=integration

# Real mode (self-hosted, sequential)
gh workflow run unlock-integration-e2e.yml -f test_mode=real

# With stress testing
gh workflow run unlock-integration-e2e.yml -f stress_test=true -f unlock_cycles=20

# Performance baseline check
gh workflow run unlock-integration-e2e.yml \
  -f test_mode=integration \
  -f performance_baseline=1500
```

### View Results

```bash
# List recent runs
gh run list --workflow unlock-integration-e2e.yml --limit 5

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log

# Download results
gh run download <run-id>

# Re-run failed
gh run rerun <run-id> --failed
```

## 🎯 Test Modes

| Mode | Concurrency | Safety | Use Case |
|------|-------------|--------|----------|
| Mock | 5 | 🟢 | CI/CD, fast |
| Integration | 3 | 🟡 | macOS testing |
| Real | 1 | 🔴 | Final validation |

## 📋 Test Suites

- ✅ Keychain Retrieval (30s timeout)
- ✅ Unlock Logic (45s timeout)
- ✅ AppleScript Typing (20s timeout)
- ✅ Screen Detection (30s timeout)
- ✅ Error Handling (40s timeout)
- ✅ Performance (120s timeout)
- ✅ Security Checks (60s timeout)
- ✅ Full E2E (180s timeout, real mode only)

## 🔒 Security Checklist

- [ ] Keychain configured (for integration/real)
- [ ] Password stored: `com.jarvis.voiceunlock`
- [ ] No hardcoded credentials
- [ ] Mock mode for CI/CD
- [ ] Self-hosted for real tests

## ⚙️ Configuration

```bash
export TEST_MODE=mock           # mock|integration|real
export TEST_DURATION=600        # seconds
export UNLOCK_CYCLES=5          # number of cycles
export STRESS_TEST=false        # true|false
export PERFORMANCE_BASELINE=2000 # milliseconds
export MAX_CONCURRENT=5         # concurrent tests
```

## 📊 Async Metrics

Each test tracks:
- Wall clock duration
- Async duration
- Async overhead
- Concurrent execution stats

## 🛠️ Troubleshooting

```bash
# Check keychain
security find-generic-password -s "com.jarvis.voiceunlock" -a "unlock_token" -w

# Reduce concurrency if tests timeout
# Set MAX_CONCURRENT=3 in workflow

# Check async metrics in report
cat test-results/unlock-e2e/report-*.json | jq '.async_metrics'
```

## 📈 Performance Targets

**Mock Mode:**
- Total: < 2 seconds
- Per test: < 100ms
- Async overhead: < 5ms

**Integration Mode:**
- Total: < 5 seconds
- Unlock: < 2000ms (baseline)
- Async overhead: < 10ms

**Real Mode:**
- Unlock: < 2500ms
- Success rate: > 95%

## 🎓 Best Practices

1. **Always start with mock mode** (fast feedback)
2. **Test integration before real** (safety)
3. **Monitor async metrics** (performance)
4. **Adjust concurrency based on mode** (reliability)
5. **Run performance tests weekly** (regression detection)
6. **Review security checks** (no credential leaks)

## 🆘 Quick Fixes

**Tests timeout:**
→ Reduce MAX_CONCURRENT or increase TEST_DURATION

**High async overhead:**
→ Check for blocking I/O in tests

**Integration fails on macOS:**
→ Verify keychain entry exists

**Real tests won't run:**
→ Need self-hosted runner with keychain

**Concurrent tests fail:**
→ Reduce concurrency or check for race conditions

---

**Full Docs:** [UNLOCK_INTEGRATION_E2E.md](UNLOCK_INTEGRATION_E2E.md)
