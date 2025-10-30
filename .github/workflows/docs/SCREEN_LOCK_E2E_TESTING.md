# Screen Lock/Unlock End-to-End Testing

## Overview

Comprehensive, secure, and advanced E2E testing for screen lock/unlock functionality with **zero hardcoding** and **zero credential exposure**.

## 🎯 Features

### ✨ Advanced Capabilities

- **3 Test Modes:**
  - 🟢 **Mock:** Safe CI testing with simulated locks (no real system interaction)
  - 🟡 **Integration:** Tests logic without real screen locks
  - 🔴 **Real:** Full E2E with actual screen locks (self-hosted only)

- **Security First:**
  - ✅ No credentials in code
  - ✅ Keychain-based authentication
  - ✅ Secure by default
  - ✅ CI-safe mock mode

- **Comprehensive Testing:**
  - Screen lock detection
  - Lock functionality
  - Unlock functionality
  - Voice command integration
  - Security validation
  - Stress testing

### 🔒 Security Features

1. **No Credential Exposure:**
   - All sensitive data in keychain
   - No passwords in code or logs
   - Mock mode for CI/CD

2. **Automated Security Checks:**
   - Scans for hardcoded credentials
   - Verifies keychain usage
   - Checks secure communication

3. **Safe Defaults:**
   - Mock mode by default
   - Real tests only on self-hosted
   - Unlock requires explicit approval

## 🚀 Usage

### Quick Start

**Run Mock Tests (Safe):**
```bash
gh workflow run screen-lock-e2e-test.yml
```

**Run Integration Tests:**
```bash
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=integration
```

**Run Real E2E Tests (Self-Hosted Only):**
```bash
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=real
```

### Advanced Usage

**With Voice Commands:**
```bash
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=integration \
  -f voice_test=true
```

**Stress Testing:**
```bash
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=mock \
  -f stress_test=true
```

**Full Suite:**
```bash
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=integration \
  -f voice_test=true \
  -f stress_test=true \
  -f test_duration=600
```

## 📋 Test Modes Explained

### Mode 1: Mock Testing (Default)

**What it does:**
- Simulates all system calls
- No real screen interaction
- Perfect for CI/CD

**When to use:**
- Pull requests
- Automated testing
- Quick validation
- CI pipelines

**Example:**
```yaml
TEST_MODE: mock
```

**Output:**
```
🔒 [MOCK] Locking screen...
✅ Lock Screen (50.25ms)
🔓 [MOCK] Unlocking screen...
✅ Unlock Screen (48.73ms)
```

### Mode 2: Integration Testing

**What it does:**
- Tests actual code logic
- No real screen locks
- Validates integrations

**When to use:**
- Testing on macOS runners
- Logic validation
- Integration checks
- Before real testing

**Example:**
```yaml
TEST_MODE: integration
```

**Output:**
```
🔒 [INTEGRATION] Testing lock screen logic...
✅ Lock Screen (105.47ms)
🔓 [INTEGRATION] Testing unlock screen logic...
✅ Unlock Screen (98.23ms)
```

### Mode 3: Real Testing

**What it does:**
- Actual screen locking
- Real unlock attempts
- Full E2E validation

**When to use:**
- Self-hosted runners only
- Final validation
- Production testing
- Manual verification

**Requirements:**
- Self-hosted macOS runner
- Keychain configured
- Screen access

**Example:**
```yaml
TEST_MODE: real
```

**Output:**
```
🔒 [REAL] Locking screen...
✅ Lock Screen (1250.47ms)
⚠️  Real unlock requires keychain access - using mock for security
✅ Unlock Screen (98.23ms)
```

## 🧪 Test Suite

### Test 1: Screen Lock Detection

**What:** Detects current screen lock state
**Mode Support:** Mock, Integration, Real
**Duration:** ~10-50ms

**Tests:**
- Lock state query
- State accuracy
- Response time

### Test 2: Lock Screen

**What:** Locks the screen
**Mode Support:** Mock, Integration, Real
**Duration:** ~50-1500ms

**Tests:**
- Lock command execution
- State transition
- Verification

### Test 3: Unlock Screen

**What:** Unlocks the screen
**Mode Support:** Mock, Integration (Real limited)
**Duration:** ~50-1500ms

**Tests:**
- Unlock command execution
- Credential handling
- State transition

**Security Note:** Real unlock requires keychain and is limited in automated tests for security.

### Test 4: Voice Command Lock

**What:** Lock via voice command
**Mode Support:** Mock, Integration, Real
**Duration:** ~100-2000ms

**Tests:**
- Voice recognition
- Command parsing
- Lock execution

**Commands Tested:**
- "lock my screen"
- "lock screen"
- "lock it"

### Test 5: Voice Command Unlock

**What:** Unlock via voice command
**Mode Support:** Mock, Integration
**Duration:** ~100-2000ms

**Tests:**
- Voice recognition
- Command parsing
- Security validation

**Commands Tested:**
- "unlock my screen"
- "unlock screen"
- "unlock it"

### Test 6: Security Validation

**What:** Comprehensive security checks
**Mode Support:** All
**Duration:** ~500-2000ms

**Checks:**
- No hardcoded credentials
- Keychain usage
- Secure communication
- Log sanitization

### Test 7: Stress Testing

**What:** Multiple rapid lock/unlock cycles
**Mode Support:** Mock, Integration
**Duration:** Variable (5-10 cycles)

**Tests:**
- Rapid operations
- State consistency
- Error handling
- Performance degradation

## 📊 Metrics & Reporting

### Tracked Metrics

**Performance:**
- Lock time (avg, min, max)
- Unlock time (avg, min, max)
- Voice command latency
- Test duration

**Reliability:**
- Success rate
- Failure count
- Error types
- Recovery success

**Security:**
- Credential scans passed
- Keychain usage verified
- Secure practices validated

### Report Format

```
📊 Screen Lock/Unlock E2E Test Report
Mode: INTEGRATION
Duration: 15.42s
✅ Passed: 6
❌ Failed: 0
🔒 Lock Cycles: 3
🔓 Unlock Cycles: 3
⏱️  Avg Lock Time: 105.47ms
⏱️  Avg Unlock Time: 98.23ms
🎤 Voice Tests: 2
🔒 Security Checks: 1
```

## 🔐 Security

### Credential Management

**Never in Code:**
```python
# ❌ NEVER DO THIS
password = "my_password"
```

**Always in Keychain:**
```python
# ✅ CORRECT
from keychain import get_credential
password = get_credential("jarvis_unlock")
```

### Keychain Setup (Self-Hosted)

**One-Time Setup:**
```bash
# Store credential securely
security add-generic-password \
  -s "jarvis_unlock" \
  -a "${USER}" \
  -w "${PASSWORD}"
```

**Retrieve in Code:**
```python
import subprocess

def get_unlock_password():
    result = subprocess.run(
        ["security", "find-generic-password",
         "-s", "jarvis_unlock",
         "-w"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()
```

### CI/CD Safety

**GitHub Actions (Hosted):**
- Automatically uses mock mode
- No credentials needed
- Safe for pull requests

**Self-Hosted Runners:**
- Can run real tests
- Requires keychain setup
- Manual trigger recommended

## 🎮 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_MODE` | `mock` | Test execution mode |
| `TEST_DURATION` | `300` | Max test duration (seconds) |
| `VOICE_TEST` | `true` | Include voice command tests |
| `STRESS_TEST` | `false` | Run stress tests |

### Workflow Inputs

**test_mode:**
- Options: `mock`, `integration`, `real`
- Default: `mock`

**test_duration:**
- Range: 60-3600 seconds
- Default: 300

**voice_test:**
- Type: boolean
- Default: true

**stress_test:**
- Type: boolean
- Default: false

## 🛠️ Troubleshooting

### Issue: Tests Fail in Mock Mode

**Cause:** Code logic errors
**Solution:** Check test output for specific failures

```bash
# Run locally
python3 .github/workflows/scripts/screen_lock_e2e_test.py
```

### Issue: Integration Tests Fail on macOS

**Cause:** Missing dependencies or permissions
**Solution:** Check dependencies and permissions

```bash
# Check dependencies
pip install -r backend/requirements.txt

# Verify Python version
python3 --version
```

### Issue: Real Tests Won't Run

**Cause:** Not on self-hosted runner
**Solution:** Ensure runner is self-hosted macOS

```yaml
runs-on: self-hosted
if: needs.setup-environment.outputs.can-test-real == 'true'
```

### Issue: Credential Errors

**Cause:** Keychain not configured
**Solution:** Set up keychain on self-hosted runner

```bash
# Verify keychain
security find-generic-password -s "jarvis_unlock"
```

### Issue: Voice Tests Failing

**Cause:** Voice unlock code not found
**Solution:** Verify code exists

```bash
# Check for voice unlock code
ls -la backend/voice_unlock/
```

## 📈 Best Practices

### 1. Always Start with Mock Mode

Test changes with mock mode first:
```bash
gh workflow run screen-lock-e2e-test.yml -f test_mode=mock
```

### 2. Use Integration for Logic Testing

Validate logic without real locks:
```bash
gh workflow run screen-lock-e2e-test.yml -f test_mode=integration
```

### 3. Real Testing Only When Necessary

Reserve real testing for final validation:
```bash
# Manual trigger recommended
gh workflow run screen-lock-e2e-test.yml -f test_mode=real
```

### 4. Enable Voice Tests

Always test voice commands:
```bash
gh workflow run screen-lock-e2e-test.yml -f voice_test=true
```

### 5. Run Stress Tests Periodically

Weekly stress testing recommended:
```bash
# Schedule or manual trigger
gh workflow run screen-lock-e2e-test.yml -f stress_test=true
```

### 6. Monitor Security Checks

Review security check results regularly:
- Check for credential leaks
- Verify keychain usage
- Validate secure practices

## 🔄 Integration with CI/CD

### Pull Request Workflow

```yaml
on:
  pull_request:
    paths:
      - 'backend/voice_unlock/**'
```

**What happens:**
1. Automatically runs mock tests
2. Reports results on PR
3. Blocks merge if tests fail

### Daily Testing

```yaml
schedule:
  - cron: '0 3 * * *'
```

**What happens:**
1. Runs full test suite daily
2. Creates issue if failures
3. Tracks trends over time

### Manual Validation

```bash
# Before release
gh workflow run screen-lock-e2e-test.yml \
  -f test_mode=integration \
  -f voice_test=true \
  -f stress_test=true
```

## 📝 Example Outputs

### Successful Run

```
🚀 Starting Screen Lock/Unlock E2E Tests
Mode: INTEGRATION

▶️  Running: Lock Detection
✅ Lock Detection (15.23ms)

▶️  Running: Lock Screen
✅ Lock Screen (105.47ms)

▶️  Running: Unlock Screen
✅ Unlock Screen (98.23ms)

▶️  Running: Voice Lock
✅ Voice Lock (187.54ms)

▶️  Running: Voice Unlock
✅ Voice Unlock (165.32ms)

▶️  Running: Security Check
✅ Security Check (523.18ms)

================================================================================
📊 Screen Lock/Unlock E2E Test Report
Mode: INTEGRATION
Duration: 15.42s
✅ Passed: 6
❌ Failed: 0
🔒 Lock Cycles: 2
🔓 Unlock Cycles: 2
⏱️  Avg Lock Time: 105.47ms
⏱️  Avg Unlock Time: 98.23ms
🎤 Voice Tests: 2
🔒 Security Checks: 1
================================================================================
```

### Failed Run

```
🚀 Starting Screen Lock/Unlock E2E Tests
Mode: MOCK

▶️  Running: Lock Detection
✅ Lock Detection (12.45ms)

▶️  Running: Lock Screen
❌ Lock Screen - Timeout waiting for lock confirmation

▶️  Running: Security Check
✅ Security Check (487.32ms)

================================================================================
📊 Screen Lock/Unlock E2E Test Report
Mode: MOCK
Duration: 8.23s
✅ Passed: 2
❌ Failed: 1
🔒 Lock Cycles: 0
🔓 Unlock Cycles: 0
🔒 Security Checks: 1
================================================================================
```

## 🆘 Support

### Getting Help

**Check Logs:**
```bash
# View workflow logs
gh run view --log

# View specific job
gh run view <run-id> --job <job-id> --log
```

**Common Commands:**
```bash
# List runs
gh run list --workflow screen-lock-e2e-test.yml

# Re-run failed tests
gh run rerun <run-id> --failed

# Cancel running test
gh run cancel <run-id>
```

### Creating Issues

If tests consistently fail:
1. Check workflow logs
2. Verify configuration
3. Create issue with:
   - Run ID
   - Error messages
   - Test mode used
   - Environment details

## 📚 Resources

**Related Documentation:**
- [GitHub Actions](https://docs.github.com/en/actions)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [macOS Security](https://support.apple.com/guide/security)

**Project Files:**
- Workflow: `.github/workflows/screen-lock-e2e-test.yml`
- Test Script: `.github/workflows/scripts/screen_lock_e2e_test.py`
- Voice Unlock: `backend/voice_unlock/`

---

**Last Updated:** 2025-10-30
**Version:** 1.0.0
**Status:** ✅ Production Ready
