# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: PR Automation & Validation
- **Run Number**: #305
- **Branch**: `fix/ci/websocket-self-healing-validation-run11-20251109-030809`
- **Commit**: `09a2646d35ab892afb692431489b5d8dbd10b4ae`
- **Status**: ❌ FAILED
- **Timestamp**: 2025-11-09T03:08:40Z
- **Triggered By**: @cubic-dev-ai[bot]
- **Workflow URL**: [View Run](https://github.com/drussell23/JARVIS/actions/runs/19202496625)

## Failure Overview

Total Failed Jobs: **1**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
| 1 | Validate PR Title | timeout | high | 6s |

## Detailed Analysis

### 1. Validate PR Title

**Status**: ❌ failure
**Category**: Timeout
**Severity**: HIGH
**Started**: 2025-11-09T03:08:44Z
**Completed**: 2025-11-09T03:08:50Z
**Duration**: 6 seconds
**Job URL**: [View Logs](https://github.com/drussell23/JARVIS/actions/runs/19202496625/job/54892619188)

#### Failed Steps

- **Step 2**: Validate Conventional Commits

#### Error Analysis

**Detected Error Patterns:**

- Pattern: `ERROR|Error|error`
  - Occurrences: 2
  - Sample matches:
    - Line 25: `2025-11-09T03:08:47.1928263Z   subjectPatternError: The PR title must start with a capital letter.`
    - Line 37: `2025-11-09T03:08:48.0279056Z ##[error]No release type found in pull request title "🚨 Fix CI/CD: WebS`

- Pattern: `timeout|timed out`
  - Occurrences: 1
  - Sample matches:
    - Line 29: `- fix: Resolve database connection timeout`

#### Suggested Fixes

1. Consider increasing timeout values or optimizing slow operations
2. Check service availability and network connectivity

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

📊 *Report generated on 2025-11-09T03:09:32.335152*
🤖 *JARVIS CI/CD Auto-PR Manager*
