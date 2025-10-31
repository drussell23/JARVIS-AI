# 🤖 Advanced CI/CD Auto-PR System

## Overview

The JARVIS CI/CD Auto-PR System automatically creates pull requests when CI/CD workflows fail, providing intelligent failure analysis and suggested fixes.

## Features

### 🚀 Core Capabilities

- **Dynamic Workflow Monitoring**: Automatically monitors ALL workflows without hardcoding
- **Intelligent Failure Analysis**: Async analysis of failed jobs with error pattern detection
- **Automated PR Creation**: Creates fix PRs with detailed failure reports
- **Smart Deduplication**: Prevents duplicate PRs for similar failures
- **Critical Failure Alerts**: Creates issues for severe failures (3+ failed jobs)
- **Configuration-Driven**: Fully customizable via YAML configuration

### 🔬 Advanced Features

- **Async Processing**: High-performance async API calls with retry logic
- **Error Pattern Detection**: Regex-based error extraction from logs
- **Categorization**: Automatic failure categorization (test, lint, build, etc.)
- **Suggested Fixes**: Context-aware fix suggestions based on error patterns
- **Rate Limiting**: Built-in safety limits to prevent PR spam
- **Artifact Uploads**: Saves detailed analysis reports as artifacts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Workflow Failure Event                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Pre-flight Checks                           │
│  • Load configuration                                    │
│  • Validate config YAML                                  │
│  • Check exclusion patterns                              │
│  • Verify should process                                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Failure Analysis (Async)                         │
│  • Fetch workflow run details                            │
│  • Get all failed jobs (with pagination)                 │
│  • Download job logs                                     │
│  • Extract error patterns                                │
│  • Categorize failures                                   │
│  • Generate fix suggestions                              │
│  • Check for similar open PRs                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│           PR Creation & Reporting                        │
│  • Generate branch name                                  │
│  • Create detailed failure report                        │
│  • Generate PR body with analysis                        │
│  • Create and push fix branch                            │
│  • Create pull request                                   │
│  • Add commit/PR comments                                │
│  • Upload artifacts                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│        Critical Failure Handling                         │
│  • Check if critical (3+ failures)                       │
│  • Create GitHub issue                                   │
│  • Notify stakeholders                                   │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Configuration File

Location: `.github/workflows/config/ci-auto-pr-config.yml`

Key sections:

```yaml
# Workflow monitoring
monitoring:
  auto_discover_workflows: true  # Monitor all workflows
  exclude_patterns:              # Exclude specific patterns
    - ".*dependabot.*"
    - ".*auto-merge.*"

# PR settings
pr_settings:
  branch_prefix: "fix/ci"
  title_template: "🚨 Fix CI/CD: {workflow_name} (Run #{run_number})"
  labels:
    - "ci-failure"
    - "automated-pr"
  auto_assign:
    enabled: true
    assign_to: "actor"  # actor, author, team, or specific users

# Failure analysis
analysis:
  fetch_logs: true
  max_log_lines: 100
  parse_errors: true
  error_patterns:
    - pattern: "ERROR|Error|error"
      severity: "high"
      emoji: "❌"

# Safety limits
safety:
  max_prs_per_day: 20
  skip_if_similar_pr_exists: true
  similarity_threshold: 80
```

## Components

### 1. Workflow File
**File**: `failed-ci-auto-pr.yml`

Main orchestrator that:
- Triggers on workflow_run completion events
- Performs pre-flight checks
- Calls the Python analysis script
- Creates branches and PRs
- Handles critical failures

### 2. Python Analysis Script
**File**: `scripts/ci_auto_pr_manager.py`

Advanced async processor that:
- Fetches workflow and job data from GitHub API
- Downloads and analyzes job logs
- Detects error patterns
- Categorizes failures
- Generates fix suggestions
- Checks for duplicate PRs

**Key Classes**:
- `ConfigManager`: Loads and validates configuration
- `GitHubAPIClient`: Async GitHub API client with retry logic
- `FailureAnalyzer`: Intelligent failure analysis engine
- `PRManager`: PR generation and management
- `CIAutoPRManager`: Main orchestrator

### 3. Configuration File
**File**: `config/ci-auto-pr-config.yml`

Customizable settings for:
- Workflow monitoring
- PR behavior
- Analysis depth
- Safety limits
- Notifications

## Usage

### Automatic Operation

The system automatically triggers when any workflow fails. No manual intervention required!

### Manual Testing

To test the system with a specific workflow run:

```bash
# Set environment variables
export GITHUB_TOKEN="your-token"
export GITHUB_REPOSITORY="owner/repo"
export WORKFLOW_RUN_ID="123456789"

# Run the analysis script
python3 .github/workflows/scripts/ci_auto_pr_manager.py
```

### Customization

1. **Modify Configuration**:
   ```bash
   vim .github/workflows/config/ci-auto-pr-config.yml
   ```

2. **Add Custom Error Patterns**:
   ```yaml
   analysis:
     error_patterns:
       - pattern: "YourCustomError"
         severity: "high"
         emoji: "🔥"
   ```

3. **Adjust Safety Limits**:
   ```yaml
   safety:
     max_prs_per_day: 50  # Increase limit
     min_time_between_prs: 15  # Reduce cooldown
   ```

## Output

### PR Content

Each auto-created PR includes:

1. **Summary Section**:
   - Failed job count
   - Commit SHA
   - Workflow run link
   - Trigger actor

2. **Failed Jobs Analysis**:
   - Job name with severity indicator
   - Category and classification
   - Duration
   - Failed steps list
   - Suggested fixes

3. **Failure Report** (CI_FAILURE_REPORT.md):
   - Executive summary
   - Detailed job-by-job analysis
   - Error pattern extraction
   - Suggested remediation steps

4. **Machine-Readable Data** (workflow_failure_details.json):
   - Complete failure metadata
   - Analysis results
   - Timestamps

### Artifacts

Each run uploads:
- `CI_FAILURE_REPORT.md`: Human-readable report
- `workflow_failure_details.json`: Machine-readable data
- `pr_body.md`: Generated PR description

Retention: 30 days

## Error Patterns & Categories

### Detected Categories

| Category | Triggers | Suggestions |
|----------|----------|-------------|
| `test_failure` | Test fail, assertion error | Review tests, update expectations |
| `linting_error` | Flake8, pylint, eslint | Run linters locally, fix style |
| `syntax_error` | Parse error, syntax error | Check code syntax |
| `build_error` | Build fail, compilation | Check build config, dependencies |
| `dependency_error` | Import error, module not found | Update requirements.txt |
| `timeout` | Timeout, timed out | Increase timeout, optimize code |
| `permission_error` | Permission denied, forbidden | Check permissions in workflow |
| `network_error` | Connection refused, timeout | Check service availability |

### Error Severity

- **High**: Critical errors that block progress
- **Medium**: Warnings and non-critical issues
- **Low**: Informational messages

## Safety Features

### Rate Limiting

- Max 20 PRs per day (configurable)
- Min 30 minutes between PRs for same workflow
- Automatic cooldown on excessive failures

### Deduplication

- Checks for similar open PRs before creating
- 80% similarity threshold (configurable)
- Prevents PR spam

### Exclusion Patterns

Automatically excludes:
- Dependabot workflows
- Auto-merge workflows
- Cleanup workflows
- The auto-PR workflow itself (prevents recursion)

## Integration

### With Existing Workflows

No changes needed! The system monitors all workflows automatically.

### With Notifications

Configure in `ci-auto-pr-config.yml`:

```yaml
notifications:
  slack:
    enabled: true
    webhook_secret: "SLACK_WEBHOOK_URL"
    channel: "#ci-alerts"

  discord:
    enabled: true
    webhook_secret: "DISCORD_WEBHOOK_URL"
```

### With External Tools

The system outputs JSON data that can be consumed by:
- Monitoring dashboards
- Analytics platforms
- Custom automation scripts

## Troubleshooting

### Common Issues

**Issue**: Workflow not triggering
- **Solution**: Check that workflow_run permissions are granted

**Issue**: Cannot create PR
- **Solution**: Verify GITHUB_TOKEN has `contents: write` and `pull-requests: write`

**Issue**: Logs not fetched
- **Solution**: Ensure `actions: read` permission is granted

**Issue**: Too many PRs created
- **Solution**: Adjust `safety.max_prs_per_day` in config

### Debug Mode

Enable debug output:

```yaml
advanced:
  debug_mode: true
```

This provides verbose logging in workflow runs.

## Performance

### Optimization Features

- **Async Processing**: Concurrent API requests
- **Pagination Handling**: Efficient large dataset processing
- **Caching**: Reduces redundant API calls
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Management**: Prevents hanging operations

### Typical Execution Time

- Simple failure (1-2 jobs): ~30-45 seconds
- Complex failure (5+ jobs): ~60-90 seconds
- Critical failure with logs: ~90-120 seconds

## Future Enhancements

Planned features:

- [ ] AI-powered fix suggestions (OpenAI/Anthropic integration)
- [ ] Auto-fix common issues (dependency updates, cache clearing)
- [ ] Historical trend analysis
- [ ] Slack/Discord notifications
- [ ] Custom webhook support
- [ ] Failure prediction based on patterns
- [ ] Auto-rerun flaky tests
- [ ] Integration with issue tracking systems

## Contributing

To enhance the auto-PR system:

1. Modify configuration in `config/ci-auto-pr-config.yml`
2. Extend analysis logic in `scripts/ci_auto_pr_manager.py`
3. Update workflow in `failed-ci-auto-pr.yml`
4. Test with workflow_dispatch trigger
5. Submit PR with changes

## Security

### Permissions Required

```yaml
permissions:
  contents: write       # Create branches
  pull-requests: write  # Create PRs
  issues: write         # Create issues
  actions: read         # Read workflow data
  checks: read          # Read check results
```

### Secrets

No additional secrets required! Uses built-in `GITHUB_TOKEN`.

Optional secrets for extensions:
- `ANTHROPIC_API_KEY`: For AI suggestions
- `SLACK_WEBHOOK_URL`: For Slack notifications
- `DISCORD_WEBHOOK_URL`: For Discord notifications

## License

Part of the JARVIS-AI-Agent project.

---

📚 **Documentation Version**: 1.0.0
🤖 **Maintained by**: JARVIS CI/CD Team
📅 **Last Updated**: 2025-10-30
