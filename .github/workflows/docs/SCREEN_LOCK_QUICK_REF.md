# Screen Lock/Unlock E2E Testing - Quick Reference

## 🚀 Quick Commands

### Run Tests

```bash
# Mock mode (safe, default)
gh workflow run screen-lock-e2e-test.yml

# Integration mode
gh workflow run screen-lock-e2e-test.yml -f test_mode=integration

# Real mode (self-hosted only)
gh workflow run screen-lock-e2e-test.yml -f test_mode=real

# With voice commands
gh workflow run screen-lock-e2e-test.yml -f voice_test=true

# Stress testing
gh workflow run screen-lock-e2e-test.yml -f stress_test=true

# Full suite
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=integration \
  -f voice_test=true \
  -f stress_test=true
```

### View Results

```bash
# List recent runs
gh run list --workflow screen-lock-e2e-test.yml --limit 5

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log

# Re-run failed
gh run rerun <run-id> --failed
```

## 🎯 Test Modes

| Mode | Safety | Real Locks | Use Case |
|------|--------|------------|----------|
| **Mock** | 🟢 Safe | No | CI/CD, PRs |
| **Integration** | 🟡 Safe | No | Logic testing |
| **Real** | 🔴 Careful | Yes | Final validation |

## 📋 Test Coverage

- ✅ Lock detection
- ✅ Lock screen
- ✅ Unlock screen
- ✅ Voice lock command
- ✅ Voice unlock command
- ✅ Security validation
- ✅ Stress testing

## 🔒 Security Checklist

- [ ] No credentials in code
- [ ] Keychain configured (self-hosted)
- [ ] Mock mode for CI
- [ ] Credentials not in logs
- [ ] Secure communication

## ⚙️ Configuration

```bash
export TEST_MODE=mock          # mock|integration|real
export TEST_DURATION=300       # seconds
export VOICE_TEST=true         # true|false
export STRESS_TEST=false       # true|false
```

## 🐛 Troubleshooting

```bash
# Run locally
cd .github/workflows/scripts
python3 screen_lock_e2e_test.py

# Check dependencies
pip install pytest pytest-asyncio aiohttp

# Verify keychain (self-hosted)
security find-generic-password -s "jarvis_unlock"

# Check permissions (macOS)
tccutil reset All
```

## 📊 Success Criteria

**Mock/Integration:**
- All tests pass
- No errors
- Security checks pass

**Real Testing:**
- Lock works
- Detection accurate
- Security validated
- Unlock verified (manual)

## 🎓 Best Practices

1. **Always start with mock mode**
2. **Test integration before real**
3. **Enable voice tests**
4. **Run stress tests weekly**
5. **Monitor security checks**
6. **Review logs regularly**

## 📈 Metrics

**Performance Targets:**
- Lock time: < 1500ms
- Unlock time: < 1500ms
- Detection: < 100ms
- Voice command: < 2000ms

**Reliability Targets:**
- Success rate: > 95%
- Error rate: < 5%
- Recovery: 100%

## 🆘 Quick Fixes

**Tests failing in mock:**
→ Check code logic errors

**Integration fails on macOS:**
→ Verify dependencies

**Real tests won't run:**
→ Need self-hosted runner

**Credential errors:**
→ Configure keychain

**Voice tests fail:**
→ Check voice_unlock code exists

---

**Full Docs:** [SCREEN_LOCK_E2E_TESTING.md](SCREEN_LOCK_E2E_TESTING.md)
