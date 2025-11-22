# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #138
- **Branch**: `cursor/investigate-n8n-integration-for-jarvis-and-mas-claude-4.5-sonnet-thinking-c21a`
- **Commit**: `5b93f343985eee86e7f90012e2759737b0e56bee`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-11-22T04:30:37Z
- **Triggered By**: @drussell23
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS/actions/runs/19590328481)

## Failure Overview

Total Failed Jobs: **22**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 49s |
| 2 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 37s |
| 3 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 54s |
| 4 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 40s |
| 5 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 44s |
| 6 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 38s |
| 7 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 41s |
| 8 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 47s |
| 9 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 40s |
| 10 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 56s |
| 11 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 41s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 46s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 46s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 45s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 39s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 39s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 62s |
| 18 | Run Unlock Integration E2E / Mock Tests - security-checks | test_failure | high | 37s |
| 19 | Run Unlock Integration E2E / Generate Test Summary | test_failure | high | 7s |
| 20 | Run Biometric Voice E2E / Generate Biometric Test Summary | test_failure | high | 5s |
| 21 | Generate Combined Test Summary | test_failure | high | 2s |
| 22 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:17Z
**Completed**: 2025-11-22T04:32:06Z
**Duration**: 49 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365854)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:04.0446135Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:04.0459416Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:04.0958556Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:04.4497829Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:04.1056916Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:04.3111342Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:04.4497829Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:31:55.2863553Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:31:56.8397925Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:31:57.0060696Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:31:55.5531050Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:31:55.5635531Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:31:56.3752789Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:09Z
**Completed**: 2025-11-22T04:32:46Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365855)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:45.1468198Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:45.1477330Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:45.1832908Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:45.5543959Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:45.1934786Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:45.4062817Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:45.5543959Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:36.1575776Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:37.6577223Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:37.8176702Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:36.3932969Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:36.4005156Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:37.3005764Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:45Z
**Completed**: 2025-11-22T04:32:39Z
**Duration**: 54 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365856)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:36.3697075Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:36.3707769Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:36.4122904Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:36.7882779Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:36.4267774Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:36.6397687Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:36.7882779Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:27.5718523Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:28.9171682Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:29.0814723Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:27.7756145Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:27.7780689Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:28.6623572Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:07Z
**Completed**: 2025-11-22T04:31:47Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365857)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:31:45.6069454Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:31:45.6079018Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:31:45.6443655Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:31:46.0127076Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:31:45.6546300Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:31:45.8646196Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:31:46.0127076Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:31:36.0818533Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:31:37.8370623Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:31:38.0051020Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:31:36.3174112Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:31:36.3248548Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:31:37.2304476Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:56Z
**Completed**: 2025-11-22T04:32:40Z
**Duration**: 44 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365860)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:38.6820409Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:38.6829907Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:38.7206292Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:39.0990590Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:38.7313444Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:38.9453748Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:39.0990590Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:29.4811851Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:30.8502659Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:31.0107565Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:29.6754831Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:29.6790059Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:30.5653371Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:17Z
**Completed**: 2025-11-22T04:31:55Z
**Duration**: 38 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365862)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:31:54.2778834Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:31:54.2788883Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:31:54.3215318Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:31:54.6960225Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:31:54.3315839Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:31:54.5465663Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:31:54.6960225Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:31:44.7860729Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:31:46.1777669Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:31:46.3405969Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:31:44.9881620Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:31:44.9917688Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:31:45.8909705Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:52Z
**Completed**: 2025-11-22T04:32:33Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365863)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:31.4771910Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:31.4781125Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:31.5150026Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:31.8889914Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:31.5257289Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:31.7395129Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:31.8889914Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:21.8181343Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:24.0545378Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:24.2160549Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:22.0656744Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:22.0785954Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:22.9720095Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:12Z
**Completed**: 2025-11-22T04:32:59Z
**Duration**: 47 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365865)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:56.3427719Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:56.3436799Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:56.3873867Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:56.7383583Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:56.3978181Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:56.6035385Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:56.7383583Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:48.0087202Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:49.5676863Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:49.7311344Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:48.2813763Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:48.2917657Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:49.0890705Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:33:03Z
**Completed**: 2025-11-22T04:33:43Z
**Duration**: 40 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365867)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:33:41.1827093Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:33:41.1837644Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:33:41.2197117Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:33:41.5961782Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:33:41.2304458Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:33:41.4474663Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:33:41.5961782Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:33:31.8233214Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:33:33.5718249Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:33:33.7446117Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:33:32.0717779Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:33:32.0795086Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:33:32.9633854Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:20Z
**Completed**: 2025-11-22T04:33:16Z
**Duration**: 56 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365868)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:33:13.8452390Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:33:13.8461585Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:33:13.8843102Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:33:14.2642717Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:33:13.8947804Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:33:14.1149554Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:33:14.2642717Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:33:04.8410590Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:33:06.2466901Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:33:06.4069305Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:33:05.0366510Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:33:05.0398218Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:33:05.9080380Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:06Z
**Completed**: 2025-11-22T04:32:47Z
**Duration**: 41 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365869)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:46.0749866Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:46.0759542Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:46.1117588Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:46.4869079Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:46.1220638Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:46.3351384Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:46.4869079Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:36.4793856Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:38.3614806Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:38.5227136Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:36.7182637Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:36.7253393Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:37.6347882Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:20Z
**Completed**: 2025-11-22T04:32:06Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365870)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:04.0384655Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:04.0396777Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:04.1196730Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:04.5424917Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:04.1316114Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:04.3675260Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:04.5424917Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:31:53.1564804Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:31:54.8092380Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:31:54.9849172Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:31:53.4048110Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:31:53.4195707Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:31:54.3624928Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:31:17Z
**Completed**: 2025-11-22T04:32:03Z
**Duration**: 46 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365873)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:00.7324388Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:00.7334202Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:00.7721714Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:01.1252345Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:00.7817071Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:00.9903548Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:01.1252345Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:31:51.9344934Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:31:53.8448304Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:31:54.0063440Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:31:52.2043839Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:31:52.2151167Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:31:52.9989578Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:58Z
**Completed**: 2025-11-22T04:33:43Z
**Duration**: 45 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365875)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:33:41.8344569Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:33:41.8354556Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:33:41.8758260Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:33:42.2565585Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:33:41.8865742Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:33:42.1044194Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:33:42.2565585Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:33:32.5124692Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:33:33.9444310Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:33:34.1055378Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:33:32.7094670Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:33:32.7131524Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:33:33.6053212Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:08Z
**Completed**: 2025-11-22T04:32:47Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365876)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:32:45.9307219Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:32:45.9316971Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:32:45.9684854Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:32:46.3370881Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:32:45.9784748Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:32:46.1897032Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:32:46.3370881Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:36.4691509Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:38.2135103Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:38.3734425Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:36.7003544Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:36.7075305Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:37.5865380Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-22T04:32:22Z
**Completed**: 2025-11-22T04:33:01Z
**Duration**: 39 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365877)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:33:00.2441370Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:33:00.2451232Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:33:00.2823765Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:33:00.6575752Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:33:00.2929204Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:33:00.5057170Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:33:00.6575752Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:32:50.9879041Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:32:52.7754686Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:32:52.9352780Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:32:51.2358925Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:32:51.2430603Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:32:52.1383072Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:32:18Z
**Completed**: 2025-11-22T04:33:20Z
**Duration**: 62 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107365885)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 64: `2025-11-22T04:33:17.6059571Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 65: `2025-11-22T04:33:17.6068908Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 66: `2025-11-22T04:33:17.6467157Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 2
  - Sample matches:
    - Line 69: `2025-11-22T04:33:17.6574173Z   name: test-results-biometric-mock-edge-case-database-failure`
    - Line 97: `2025-11-22T04:33:18.0331483Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 3
  - Sample matches:
    - Line 72: `2025-11-22T04:33:17.6575024Z   if-no-files-found: warn`
    - Line 87: `2025-11-22T04:33:17.8784272Z ##[warning]No files were found with the provided path: test-results/bio`
    - Line 97: `2025-11-22T04:33:18.0331483Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `AssertionError|Exception`
  - Occurrences: 4
  - Sample matches:
    - Line 0: `2025-11-22T04:33:08.4988030Z   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)`
    - Line 52: `2025-11-22T04:33:09.9647887Z Downloading exceptiongroup-1.3.1-py3-none-any.whl (16 kB)`
    - Line 61: `2025-11-22T04:33:10.1273426Z Installing collected packages: typing-extensions, tomli, pygments, prop`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 19: `2025-11-22T04:33:08.7199404Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 20: `2025-11-22T04:33:08.7232449Z   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 37: `2025-11-22T04:33:09.6153863Z Using cached pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:32:49Z
**Completed**: 2025-11-22T04:33:26Z
**Duration**: 37 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107367750)

#### Failed Steps

- **Step 6**: Run Mock Tests

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 39: `2025-11-22T04:33:24.3270017Z 2025-11-22 04:33:24,326 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 56: `2025-11-22T04:33:24.3397066Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line 39: `2025-11-22T04:33:24.3270017Z 2025-11-22 04:33:24,326 - __main__ - ERROR - ❌ 1 test(s) failed`
    - Line 48: `2025-11-22T04:33:24.3277918Z ❌ Failed: 1`
    - Line 97: `2025-11-22T04:33:24.9492910Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 62: `2025-11-22T04:33:24.3468261Z   if-no-files-found: warn`
    - Line 97: `2025-11-22T04:33:24.9492910Z ##[warning]The process '/usr/bin/git' failed with exit code 128`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Review test cases and ensure code changes haven't broken existing functionality

---

### 19. Run Unlock Integration E2E / Generate Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:33:29Z
**Completed**: 2025-11-22T04:33:36Z
**Duration**: 7 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107429800)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 1: `2025-11-22T04:33:32.7481741Z Starting download of artifact to: /home/runner/work/JARVIS/JARVIS/all-r`
    - Line 97: `2025-11-22T04:33:33.2337755Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 13
  - Sample matches:
    - Line 39: `2025-11-22T04:33:33.0686361Z [36;1mTOTAL_FAILED=0[0m`
    - Line 44: `2025-11-22T04:33:33.0689227Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 46: `2025-11-22T04:33:33.0690696Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 4: `2025-11-22T04:33:32.8766514Z (node:1989) [DEP0005] DeprecationWarning: Buffer() is deprecated due to`
    - Line 5: `2025-11-22T04:33:32.8824140Z (Use `node --trace-deprecation ...` to show where the warning was creat`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 20. Run Biometric Voice E2E / Generate Biometric Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:33:46Z
**Completed**: 2025-11-22T04:33:51Z
**Duration**: 5 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107435504)

#### Failed Steps

- **Step 4**: Check Test Status

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 97: `2025-11-22T04:33:49.3744200Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 11
  - Sample matches:
    - Line 32: `2025-11-22T04:33:49.2151631Z [36;1mTOTAL_FAILED=0[0m`
    - Line 37: `2025-11-22T04:33:49.2157493Z [36;1m    FAILED=$(jq -r '.summary.failed' "$report")[0m`
    - Line 39: `2025-11-22T04:33:49.2160200Z [36;1m    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 21. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:33:54Z
**Completed**: 2025-11-22T04:33:56Z
**Duration**: 2 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107438149)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 83: `2025-11-22T04:33:55.6935705Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 47: `2025-11-22T04:33:55.6731553Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 50: `2025-11-22T04:33:55.6733546Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 54: `2025-11-22T04:33:55.6735539Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 22. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-11-22T04:34:00Z
**Completed**: 2025-11-22T04:34:04Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19590328481/job/56107440130)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -1: `2025-11-22T04:34:01.9378872Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -9: `2025-11-22T04:34:01.8141644Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -8: `2025-11-22T04:34:01.8143402Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -3: `2025-11-22T04:34:01.9345169Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

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

📊 *Report generated on 2025-11-22T04:35:00.108892*
🤖 *JARVIS CI/CD Auto-PR Manager*
