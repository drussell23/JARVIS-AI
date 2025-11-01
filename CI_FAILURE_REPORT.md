# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: Complete Unlock Test Suite (Master)
- **Run Number**: #25
- **Branch**: `main`
- **Commit**: `51f5ee1129cb0d28080b542dcb809d7beecaf799`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-10-31T22:31:42Z
- **Triggered By**: @drussell23
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601)

## Failure Overview

Total Failed Jobs: **30**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Run Unlock Integration E2E / Mock Tests - intelligent-voice-service | timeout | high | 16s |
| 2 | Run Unlock Integration E2E / Mock Tests - unlock-logic | timeout | high | 21s |
| 3 | Run Unlock Integration E2E / Mock Tests - keychain-retrieval | timeout | high | 20s |
| 4 | Run Unlock Integration E2E / Mock Tests - secure-password-typer | timeout | high | 20s |
| 5 | Run Unlock Integration E2E / Mock Tests - memory-security | timeout | high | 13s |
| 6 | Run Unlock Integration E2E / Mock Tests - adaptive-timing | timeout | high | 15s |
| 7 | Run Unlock Integration E2E / Mock Tests - error-handling | timeout | high | 19s |
| 8 | Run Unlock Integration E2E / Mock Tests - performance | timeout | high | 18s |
| 9 | Run Unlock Integration E2E / Mock Tests - screen-detector-integration | timeout | high | 14s |
| 10 | Run Unlock Integration E2E / Mock Tests - security-checks | timeout | high | 16s |
| 11 | Run Unlock Integration E2E / Mock Tests - fallback-mechanisms | timeout | high | 18s |
| 12 | Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription | timeout | high | 27s |
| 13 | Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection | timeout | high | 21s |
| 14 | Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment | timeout | high | 21s |
| 15 | Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation | timeout | high | 20s |
| 16 | Run Biometric Voice E2E / Mock Biometric Tests - voice-verification | timeout | high | 20s |
| 17 | Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation | timeout | high | 24s |
| 18 | Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds | timeout | high | 20s |
| 19 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start | timeout | high | 21s |
| 20 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise | timeout | high | 24s |
| 21 | Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection | timeout | high | 26s |
| 22 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift | timeout | high | 26s |
| 23 | Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure | test_failure | high | 23s |
| 24 | Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection | timeout | high | 23s |
| 25 | Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing | timeout | high | 26s |
| 26 | Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline | timeout | high | 22s |
| 27 | Run Biometric Voice E2E / Mock Biometric Tests - security-validation | timeout | high | 35s |
| 28 | Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow | timeout | high | 24s |
| 29 | Generate Combined Test Summary | test_failure | high | 4s |
| 30 | Notify Test Status | test_failure | high | 4s |

## Detailed Analysis

### 1. Run Unlock Integration E2E / Mock Tests - intelligent-voice-service

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:16Z
**Completed**: 2025-10-31T22:32:32Z
**Duration**: 16 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008398)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:32:31.0401056Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:31.0476606Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:31.2728190Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:32:28.9957282Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:32:29.2571062Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:32:30.7969768Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:32:28.8952276Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:32:28.9231284Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:32:29.2571062Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 2. Run Unlock Integration E2E / Mock Tests - unlock-logic

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:17Z
**Completed**: 2025-10-31T22:33:38Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008401)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:35.2807003Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:35.2880468Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:35.5040221Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:33.3923653Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:33.5410876Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:35.0569222Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:33.3589748Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:33.3703326Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:33.5410876Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 3. Run Unlock Integration E2E / Mock Tests - keychain-retrieval

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:35Z
**Completed**: 2025-10-31T22:32:55Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008402)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:32:52.8061528Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:52.8134990Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:53.0218414Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:32:50.8222028Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:32:51.0293216Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:32:52.5888953Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:32:50.7535307Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:32:50.7723369Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:32:51.0293216Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 4. Run Unlock Integration E2E / Mock Tests - secure-password-typer

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:27Z
**Completed**: 2025-10-31T22:32:47Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008404)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:32:45.5599890Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:32:45.5673875Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:32:45.7805530Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:32:43.6340747Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:32:43.8347341Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:32:45.3507435Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:32:43.5970972Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:32:43.6088332Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:32:43.8347341Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 5. Run Unlock Integration E2E / Mock Tests - memory-security

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:02Z
**Completed**: 2025-10-31T22:33:15Z
**Duration**: 13 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008412)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:13.7989986Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:13.8064570Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:14.0133680Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:11.9448462Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:12.1277160Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:13.6079355Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:11.8936025Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:11.9094453Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:12.1277160Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 6. Run Unlock Integration E2E / Mock Tests - adaptive-timing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:57Z
**Completed**: 2025-10-31T22:33:12Z
**Duration**: 15 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008417)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:10.6222016Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:10.6297193Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:10.8409378Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:08.6624745Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:08.8200528Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:10.3820245Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:08.6087315Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:08.6256157Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:08.8200528Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 7. Run Unlock Integration E2E / Mock Tests - error-handling

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:03Z
**Completed**: 2025-10-31T22:33:22Z
**Duration**: 19 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008419)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 64: `2025-10-31T22:33:19.4602096Z ##[error]Process completed with exit code 1.`
    - Line 67: `2025-10-31T22:33:19.4675855Z   name: test-results-mock-error-handling`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:19.4676535Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:19.6783833Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:17.5622456Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:17.6948253Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:19.2475714Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:17.5224662Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:17.5338112Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:17.6948253Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 8. Run Unlock Integration E2E / Mock Tests - performance

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:33Z
**Completed**: 2025-10-31T22:33:51Z
**Duration**: 18 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008421)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:49.1259175Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:49.1328734Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:49.3419392Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:47.3150604Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:47.5952687Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:48.9025074Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:47.2155544Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:47.2428341Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:47.5952687Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 9. Run Unlock Integration E2E / Mock Tests - screen-detector-integration

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:50Z
**Completed**: 2025-10-31T22:33:04Z
**Duration**: 14 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008426)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:02.4322151Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:02.4402094Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:02.6610121Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:00.1888071Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:00.3766521Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:02.0427653Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:00.1157613Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:00.1391248Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:00.3766521Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 10. Run Unlock Integration E2E / Mock Tests - security-checks

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:31Z
**Completed**: 2025-10-31T22:33:47Z
**Duration**: 16 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008428)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:45.0301568Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:45.0376315Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:45.2517336Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:43.0430265Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:43.2269016Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:44.7641875Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:42.9761616Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:42.9959160Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:43.2269016Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 11. Run Unlock Integration E2E / Mock Tests - fallback-mechanisms

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:32:57Z
**Completed**: 2025-10-31T22:33:15Z
**Duration**: 18 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232008432)

#### Failed Steps

- **Step 5**: Copy Test Script

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 64: `2025-10-31T22:33:12.6030434Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:12.6104906Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:12.8208493Z ##[warning]No files were found with the provided path: test-results/unl`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 28: `2025-10-31T22:33:10.6730652Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 42: `2025-10-31T22:33:10.8496237Z Installing collected packages: typing-extensions, tomli, pygments, pycp`
    - Line 44: `2025-10-31T22:33:12.3843887Z Successfully installed aiodns-3.5.0 aiofiles-25.1.0 aiohappyeyeballs-2.`

- Pattern: `timeout|timed out`
  - Occurrences: 4
  - Sample matches:
    - Line 16: `2025-10-31T22:33:10.6085901Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`
    - Line 19: `2025-10-31T22:33:10.6273860Z Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)`
    - Line 42: `2025-10-31T22:33:10.8496237Z Installing collected packages: typing-extensions, tomli, pygments, pycp`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 12. Run Biometric Voice E2E / Mock Biometric Tests - stt-transcription

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:00Z
**Completed**: 2025-10-31T22:33:27Z
**Duration**: 27 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010003)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:23.7290244Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:23.7301066Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:23.7680279Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:23.7780145Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:23.9927025Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:15.6928722Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:15.8913463Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:22.5269584Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:14.4425276Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:14.4437980Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:15.4117842Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 13. Run Biometric Voice E2E / Mock Biometric Tests - wake-word-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:15Z
**Completed**: 2025-10-31T22:33:36Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010006)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:33.8004816Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:33.8017640Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:33.8403848Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:33.8506932Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:34.0654975Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:26.3197956Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:26.5078109Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:33.1082983Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:25.1362605Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:25.1375883Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:26.0264623Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 14. Run Biometric Voice E2E / Mock Biometric Tests - profile-quality-assessment

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:07Z
**Completed**: 2025-10-31T22:33:28Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010008)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:27.0417971Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:27.0428302Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:27.0847942Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:27.0953488Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:27.3096264Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:19.4017237Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:19.6099550Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:26.2873750Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:18.1405005Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:18.1417821Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:19.1077519Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 15. Run Biometric Voice E2E / Mock Biometric Tests - embedding-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:15Z
**Completed**: 2025-10-31T22:33:35Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010010)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:33.3677150Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:33.3686126Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:33.4036116Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:33.4138857Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:33.6325615Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:25.8159916Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:26.0308394Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:32.6295749Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:24.5109621Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:24.5122765Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:25.4843477Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 16. Run Biometric Voice E2E / Mock Biometric Tests - voice-verification

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:17Z
**Completed**: 2025-10-31T22:33:37Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010016)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:35.3368011Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:35.3377379Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:35.3798788Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:35.3901242Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:35.6101480Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:27.5031639Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:27.7117047Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:34.5070024Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:25.9544155Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:25.9558464Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:26.9375424Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 17. Run Biometric Voice E2E / Mock Biometric Tests - dimension-adaptation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:39Z
**Completed**: 2025-10-31T22:34:03Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010022)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:34:01.3248630Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:34:01.3258827Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:34:01.3651765Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:34:01.3754112Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:34:01.5916332Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:53.6433157Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:53.8979125Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:34:00.3022269Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:52.1118473Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:52.1132943Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:53.1540633Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 18. Run Biometric Voice E2E / Mock Biometric Tests - adaptive-thresholds

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:18Z
**Completed**: 2025-10-31T22:33:38Z
**Duration**: 20 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010025)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:37.0926881Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:37.0937328Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:37.1358138Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:37.1465598Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:37.3670488Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:28.7642848Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:28.9713235Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:35.8085138Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:27.4367261Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:27.4381522Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:28.4046607Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 19. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-cold-start

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:12Z
**Completed**: 2025-10-31T22:33:33Z
**Duration**: 21 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010028)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:31.6404514Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:31.6413908Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:31.6805713Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:31.6911440Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:31.9024523Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:24.2147127Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:24.4273925Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:30.9871528Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:22.9616972Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:22.9630165Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:23.9082168Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 20. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-noise

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:44Z
**Completed**: 2025-10-31T22:34:08Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010029)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:34:06.2554066Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:34:06.2563861Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:34:06.2961188Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:34:06.3063147Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:34:06.5281530Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:58.1309898Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:58.3950187Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:34:05.0547791Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:56.6230329Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:56.6245521Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:57.6515120Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 21. Run Biometric Voice E2E / Mock Biometric Tests - replay-attack-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:07Z
**Completed**: 2025-10-31T22:33:33Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010035)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:31.8920409Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:31.8929385Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:31.9285459Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:31.9387871Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:32.1518038Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:23.9886116Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:24.1822750Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:31.1361784Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:22.7933755Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:22.7946658Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:23.7069649Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 22. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-voice-drift

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:50Z
**Completed**: 2025-10-31T22:34:16Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010042)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:34:13.8912948Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:34:13.8921584Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:34:13.9263110Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:34:13.9363443Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:34:14.1440046Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:34:06.1112294Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:34:06.3044574Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:34:12.8103713Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:34:04.7853381Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:34:04.7866814Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:34:05.6833980Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 23. Run Biometric Voice E2E / Mock Biometric Tests - edge-case-database-failure

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T22:33:14Z
**Completed**: 2025-10-31T22:33:37Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010045)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:35.9053703Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:35.9062615Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:35.9465043Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 1
  - Sample matches:
    - Line 67: `2025-10-31T22:33:35.9568678Z   name: test-results-biometric-mock-edge-case-database-failure`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:35.9569502Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:36.1745741Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:28.0791821Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:28.3409477Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:35.1280386Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:26.5592384Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:26.5606060Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:27.5800623Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 24. Run Biometric Voice E2E / Mock Biometric Tests - voice-synthesis-detection

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:40Z
**Completed**: 2025-10-31T22:34:03Z
**Duration**: 23 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010046)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:34:01.0835056Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:34:01.0844755Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:34:01.1202767Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:34:01.1304907Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:34:01.3397479Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:53.3642267Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:53.5913532Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:59.9730228Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:52.1473046Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:52.1486444Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:53.0882131Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 25. Run Biometric Voice E2E / Mock Biometric Tests - anti-spoofing

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:21Z
**Completed**: 2025-10-31T22:33:47Z
**Duration**: 26 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010050)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:45.5917626Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:45.5926392Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:45.6284739Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:45.6384754Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:45.8479665Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:37.5119048Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:37.7014376Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:44.5204986Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:36.1265373Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:36.1278510Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:37.0892132Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 26. Run Biometric Voice E2E / Mock Biometric Tests - performance-baseline

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:36Z
**Completed**: 2025-10-31T22:33:58Z
**Duration**: 22 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010055)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:57.0761341Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:57.0770674Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:57.1199355Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:57.1310112Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:57.3486748Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:49.4704936Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:49.6625187Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:56.3204255Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:48.2841343Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:48.2854995Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:49.2148444Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 27. Run Biometric Voice E2E / Mock Biometric Tests - security-validation

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:38Z
**Completed**: 2025-10-31T22:34:13Z
**Duration**: 35 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010058)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:34:11.8071975Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:34:11.8083806Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:34:11.8516438Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:34:11.8617940Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:34:12.0870175Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:34:03.1133091Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:34:03.4045824Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:34:10.5576280Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:34:01.6024088Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:34:01.6037853Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:34:02.6108246Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 28. Run Biometric Voice E2E / Mock Biometric Tests - end-to-end-flow

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-10-31T22:33:36Z
**Completed**: 2025-10-31T22:34:00Z
**Duration**: 24 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232010060)

#### Failed Steps

- **Step 4**: Install Dependencies

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 3
  - Sample matches:
    - Line 62: `2025-10-31T22:33:58.0524296Z ERROR: Could not find a version that satisfies the requirement google-c`
    - Line 63: `2025-10-31T22:33:58.0533935Z ERROR: No matching distribution found for google-cloud-sql-python-conne`
    - Line 64: `2025-10-31T22:33:58.0930811Z ##[error]Process completed with exit code 1.`

- Pattern: `WARN|Warning|warning`
  - Occurrences: 2
  - Sample matches:
    - Line 70: `2025-10-31T22:33:58.1038419Z   if-no-files-found: warn`
    - Line 85: `2025-10-31T22:33:58.3169623Z ##[warning]No files were found with the provided path: test-results/bio`

- Pattern: `AssertionError|Exception`
  - Occurrences: 3
  - Sample matches:
    - Line 49: `2025-10-31T22:33:49.8720588Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)`
    - Line 59: `2025-10-31T22:33:50.0901176Z Installing collected packages: typing-extensions, tomli, pygments, prop`
    - Line 61: `2025-10-31T22:33:57.2252725Z Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.2 aiosignal-`

- Pattern: `timeout|timed out`
  - Occurrences: 6
  - Sample matches:
    - Line 17: `2025-10-31T22:33:48.6342580Z Collecting async-timeout<6.0,>=4.0 (from aiohttp)`
    - Line 18: `2025-10-31T22:33:48.6355829Z   Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)`
    - Line 35: `2025-10-31T22:33:49.5532092Z Downloading pytest_timeout-2.4.0-py3-none-any.whl (14 kB)`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations

---

### 29. Generate Combined Test Summary

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T22:34:47Z
**Completed**: 2025-10-31T22:34:51Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232137653)

#### Failed Steps

- **Step 2**: Generate Combined Summary

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line 81: `2025-10-31T22:34:49.1893380Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 9
  - Sample matches:
    - Line 45: `2025-10-31T22:34:49.1626214Z [36;1mif [ "failure" = "success" ]; then[0m`
    - Line 48: `2025-10-31T22:34:49.1628225Z [36;1m  echo "- ❌ **Unlock Integration E2E:** failure" >> $GITHUB_STEP`
    - Line 52: `2025-10-31T22:34:49.1630276Z [36;1mif [ "failure" = "success" ]; then[0m`

#### Suggested Fixes

1. Review test cases and ensure code changes haven't broken existing functionality

---

### 30. Notify Test Status

**Status**: ❌ failure
**Category**: Test Failure
**Severity**: HIGH
**Started**: 2025-10-31T22:34:55Z
**Completed**: 2025-10-31T22:34:59Z
**Duration**: 4 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS-AI/actions/runs/18986757601/job/54232143505)

#### Failed Steps

- **Step 3**: Failure Notification

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 1
  - Sample matches:
    - Line -3: `2025-10-31T22:34:56.9631253Z ##[error]Process completed with exit code 1.`

- Pattern: `FAIL|Failed|failed`
  - Occurrences: 3
  - Sample matches:
    - Line -11: `2025-10-31T22:34:56.7177963Z ##[group]Run echo "❌ Unlock tests failed - 'unlock my screen' may be br`
    - Line -10: `2025-10-31T22:34:56.7179036Z [36;1mecho "❌ Unlock tests failed - 'unlock my screen' may be broken!"`
    - Line -5: `2025-10-31T22:34:56.9611937Z ❌ Unlock tests failed - 'unlock my screen' may be broken!`

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

📊 *Report generated on 2025-10-31T22:35:39.011576*
🤖 *JARVIS CI/CD Auto-PR Manager*
