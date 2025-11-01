# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Priority 2 - Biometric Voice Unlock E2E Testing
- **Run Number**: #49
- **Branch**: `main`
- **Commit**: `51f5ee1129cb0d28080b542dcb809d7beecaf799`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-10-31T22:31:42Z
- **Triggered By**: @drussell23
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608)

## Failure Overview

Total Failed Jobs: **17**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Mock Biometric Tests - wake-word-detection | timeout | high | 25s |
| 2 | Mock Biometric Tests - stt-transcription | timeout | high | 21s |
| 3 | Mock Biometric Tests - voice-verification | timeout | high | 23s |
| 4 | Mock Biometric Tests - adaptive-thresholds | timeout | high | 23s |
| 5 | Mock Biometric Tests - embedding-validation | timeout | high | 23s |
| 6 | Mock Biometric Tests - edge-case-cold-start | timeout | high | 21s |
| 7 | Mock Biometric Tests - anti-spoofing | timeout | high | 23s |
| 8 | Mock Biometric Tests - dimension-adaptation | timeout | high | 25s |
| 9 | Mock Biometric Tests - edge-case-noise | timeout | high | 19s |
| 10 | Mock Biometric Tests - edge-case-voice-drift | timeout | high | 18s |
| 11 | Mock Biometric Tests - voice-synthesis-detection | timeout | high | 24s |
| 12 | Mock Biometric Tests - profile-quality-assessment | timeout | high | 22s |
| 13 | Mock Biometric Tests - performance-baseline | timeout | high | 21s |
| 14 | Mock Biometric Tests - end-to-end-flow | timeout | high | 18s |
| 15 | Mock Biometric Tests - replay-attack-detection | timeout | high | 18s |
| 16 | Mock Biometric Tests - edge-case-database-failure | test_failure | high | 20s |
| 17 | Mock Biometric Tests - security-validation | timeout | high | 21s |

## Detailed Analysis

### 1. Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:05Z
**Completed**: 2025-10-31T22:32:30Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004973)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:28.0834585Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:28.0844835Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:28.1224597Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:28.1327354Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:28.3519754Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:20.3138643Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:20.5371217Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:27.3213027Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:18.9271923Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:18.9285250Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:19.9422248Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:21Z
**Completed**: 2025-10-31T22:32:42Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004978)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:40.1918276Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:40.1928394Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:40.2376480Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:40.2482634Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:40.4637532Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:32.2578741Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:32.5854996Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:39.0413450Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:30.9824332Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:30.9837808Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:31.9648790Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:18Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004980)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:17.3320244Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:17.3329890Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:17.3732341Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:17.3831003Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:17.5938538Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:09.2085067Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:09.4053449Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:16.1912169Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:07.9478596Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:07.9492457Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:08.8541371Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:18Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004989)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:15.4797977Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:15.4807035Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:15.5160142Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:15.5250892Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:15.7197884Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:08.8310823Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:09.0491538Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:14.4981234Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:07.6220689Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:07.6232875Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:08.4683272Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:17Z
**Completed**: 2025-10-31T22:32:40Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004990)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:38.7457731Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:38.7467788Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:38.7865381Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:38.7971402Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:39.0127072Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:31.2327374Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:31.4390361Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:38.0505615Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:30.0561982Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:30.0575302Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:30.9760281Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:16Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004995)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:14.8246755Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:14.8255676Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:14.8591643Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:14.8700222Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:15.0831914Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:07.0931763Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:07.2944980Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:13.8562339Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:05.9487144Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:05.9500398Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:06.8309630Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:01Z
**Completed**: 2025-10-31T22:32:24Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004996)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:22.6843982Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:22.6853548Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:22.7289562Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:22.7394819Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:22.9604017Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:14.5494633Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:14.7775809Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:21.4055147Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:13.2471928Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:13.2487838Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:14.2407187Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:00Z
**Completed**: 2025-10-31T22:32:25Z
**Duration**: 25 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004998)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:22.6025115Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:22.6034299Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:22.6384732Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:22.6489554Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:22.8591375Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:14.5782583Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:14.7710384Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:21.4439529Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:13.3202415Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:13.3215701Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:14.2171081Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:14Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232004999)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:12.4095243Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:12.4105499Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:12.4471565Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:12.4573911Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:12.6686269Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:04.8926507Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:05.1244368Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:11.7362232Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:03.4703304Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:03.4716101Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:04.4875213Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:42Z
**Completed**: 2025-10-31T22:33:00Z
**Duration**: 18 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005000)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:58.9527413Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:58.9537285Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:58.9906368Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:59.0006763Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:59.2101892Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:51.0635190Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:51.2639915Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:57.9412147Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:49.7813413Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:49.7826310Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:50.7157296Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:33Z
**Completed**: 2025-10-31T22:32:57Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005002)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:54.6122499Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:54.6131624Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:54.6482686Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:54.6583701Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:54.8706565Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:46.9183474Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:47.1100998Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:53.8734677Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:45.6669814Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:45.6683871Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:46.5787517Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:00Z
**Completed**: 2025-10-31T22:32:22Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005003)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:20.3490705Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:20.3499865Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:20.3904178Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:20.4002423Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:20.6137511Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:12.6443319Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:12.8481461Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:19.5896771Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:11.3818684Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:11.3832761Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:12.3438568Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:57Z
**Completed**: 2025-10-31T22:32:18Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005006)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:17.1120028Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:17.1130683Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:17.1521480Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:17.1624378Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:17.3780044Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:09.1991493Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:09.4040528Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:16.0018763Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:08.0352543Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:08.0365265Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:08.9301588Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:13Z
**Duration**: 18 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005013)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:11.7954470Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:11.7963540Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:11.8391928Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:11.8496259Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:12.0668499Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:04.1971887Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:04.3834773Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:11.0983029Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:03.0116438Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:03.0131045Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:03.9499421Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:31:55Z
**Completed**: 2025-10-31T22:32:13Z
**Duration**: 18 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005017)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:11.7136063Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:11.7145290Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:11.7506830Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:11.7605747Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:11.9701252Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:04.2713158Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:04.4556821Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:10.9703505Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:03.1224397Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:03.1238069Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:04.0031100Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T22:32:22Z
**Completed**: 2025-10-31T22:32:42Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005022)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:40.1582654Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:40.1591564Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:40.1943954Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 67: `2025-10-31T22:32:40.2043235Z   name: test-results-biometric-mock-edge-case-database-failure`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:40.2044011Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:40.4021074Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:33.7510629Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:33.9817618Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:39.5779553Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:32.5676352Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:32.5688305Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:33.4191490Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:21Z
**Completed**: 2025-10-31T22:32:42Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757608/job/54232005024)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:32:41.2462768Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:32:41.2471653Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:32:41.2816024Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:41.2915413Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:41.5043556Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:32:33.4241728Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:32:33.6270477Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:32:40.1657666Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:32:32.2890134Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:32:32.2902997Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:32:33.1696455Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

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

📊 *Report generated on 2025-10-31T22:34:45.348657*
🤖 *JARVIS CI/CD Auto-PR Manager*
