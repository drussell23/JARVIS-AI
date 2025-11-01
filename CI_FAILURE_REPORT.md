# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: WebSocket Self-Healing Validation
- **Run Number**: #3
- **Branch**: `main`
- **Commit**: `2f4cdae03550ec71a901e79240b0837d16c668c7`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-11-01T03:04:35Z
- **Triggered By**: @drussell23
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147)

## Failure Overview

Total Failed Jobs: **6**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨) | test_failure | high | 26s |
| 2 | WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄) | test_failure | high | 27s |
| 3 | WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓) | test_failure | high | 32s |
| 4 | WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗) | test_failure | high | 29s |
| 5 | WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌) | test_failure | high | 32s |
| 6 | WebSocket Health Tests (latency-performance, Latency & Performance, ⚡) | test_failure | high | 31s |

## Detailed Analysis

### 1. WebSocket Health Tests (message-delivery, Message Delivery & Reliability, 📨)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:45Z
**Completed**: 2025-11-01T03:05:11Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242069992)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:08.9810849Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:08.9825713Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:09.4084529Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:09.4148771Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 2. WebSocket Health Tests (self-healing, Self-Healing & Recovery, 🔄)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:45Z
**Completed**: 2025-11-01T03:05:12Z
**Duration**: 27 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242069995)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:10.1482393Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:10.1499577Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:10.7123145Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:10.7191408Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 3. WebSocket Health Tests (heartbeat-monitoring, Heartbeat & Health Monitoring, 💓)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:45Z
**Completed**: 2025-11-01T03:05:17Z
**Duration**: 32 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242069997)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:15.2213263Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:15.2228222Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:15.6511342Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:15.6584369Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 4. WebSocket Health Tests (concurrent-connections, Concurrent Connections, 🔗)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:44Z
**Completed**: 2025-11-01T03:05:13Z
**Duration**: 29 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242070002)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:11.4390036Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:11.4407158Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:11.9574288Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:11.9640335Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 5. WebSocket Health Tests (connection-lifecycle, Connection Lifecycle, 🔌)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:46Z
**Completed**: 2025-11-01T03:05:18Z
**Duration**: 32 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242070008)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:15.5085657Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:15.5104399Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:16.0957537Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:16.1027794Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 6. WebSocket Health Tests (latency-performance, Latency & Performance, ⚡)

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-01T03:04:45Z
**Completed**: 2025-11-01T03:05:16Z
**Duration**: 31 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18990387147/job/54242070039)

#### Failed Steps

- **Step 5**: Install Python Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 43: `2025-11-01T03:05:13.2735464Z ERROR: Cannot install psutil==5.9.6 and psutil==5.9.8 because these pac`
    - Line 56: `2025-11-01T03:05:13.2751973Z ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/late`
    - Line 57: `2025-11-01T03:05:13.7205307Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 65: `2025-11-01T03:05:13.7274040Z [36;1m  echo "❌ Some tests failed - review logs" >> $GITHUB_STEP_SUMMA`

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

📊 *Report generated on 2025-11-01T03:06:02.796463*
🤖 *JARVIS CI/CD Auto-PR Manager*
