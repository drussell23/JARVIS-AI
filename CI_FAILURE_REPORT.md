# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #2
- **Branch**: `main`
- **Commit**: `bec0c2f2120868d31fdf158fa5e81a2b2ac3f0bd`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-10-31T03:04:28Z
- **Triggered By**: @drussell23
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 33s |
| 2 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 32s |
| 3 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 27s |
| 4 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 30s |
| 5 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 29s |
| 6 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 35s |

## Detailed Analysis

### 1. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:44Z
**Completed**: 2025-10-31T03:05:17Z
**Duration**: 33 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235811)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:13.9181823Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:13.9196313Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:14.2742849Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:14.2809304Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:49Z
**Completed**: 2025-10-31T03:05:21Z
**Duration**: 32 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235812)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:19.5198495Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:19.5213236Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:20.0019432Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:20.0088957Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:44Z
**Completed**: 2025-10-31T03:05:11Z
**Duration**: 27 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235820)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:08.4121088Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:08.4137741Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:08.8356682Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:08.8425616Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:42Z
**Completed**: 2025-10-31T03:05:12Z
**Duration**: 30 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235825)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:09.8421219Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:09.8436017Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:10.2380147Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:10.2446605Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:44Z
**Completed**: 2025-10-31T03:05:13Z
**Duration**: 29 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235827)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:09.9547992Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:09.9563701Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:10.4209559Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:10.4277793Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T03:04:42Z
**Completed**: 2025-10-31T03:05:17Z
**Duration**: 35 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18961392619/job/54149235836)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-10-31T03:05:13.4982316Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-10-31T03:05:13.5000989Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-10-31T03:05:14.0173511Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-10-31T03:05:14.0240501Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

## Action Items

- [ ] Review detailed logs for each failed job
- [ ] Implement suggested fixes
- [ ] Add or update tests to prevent regression
- [ ] Verify fixes locally before pushing
- [ ] Update CI/CD configuration if needed

## Additional Resources

- [Workflow File](.github/workflows/)
- [CI/CD Documentation](../../docs/ci-cd/)
- [Troubleshooting Guide](../../docs/troubleshooting/)

---

📊 *Report generated on 2025-10-31T03:06:18.390567*
🤖 *JARVIS CI/CD Auto-PR Manager*
