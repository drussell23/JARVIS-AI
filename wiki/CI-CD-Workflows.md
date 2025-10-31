# CI/CD Workflows

Complete GitHub Actions automation documentation for JARVIS AI Agent.

---

## Overview

JARVIS features **20+ GitHub Actions workflows** providing comprehensive CI/CD automation including testing, deployment, security scanning, and code quality enforcement.

**Time Savings:** ~20-28 hours/month across all workflows

---

## Core Workflows

### 1. Test Pipeline (`test.yml`)

**Trigger:** Every push, pull request

**Purpose:** Unit and integration testing

**Features:**
- Multi-Python version testing (3.10, 3.11)
- Multi-platform (Ubuntu, macOS)
- Code coverage with Codecov
- Hybrid architecture validation

**Example:**
```bash
# Automatic on push
git push origin feature-branch

# Check results
gh run list --workflow=test.yml
```

### 2. Deployment Pipeline (`deployment.yml`)

**Trigger:** Push to main, version tags, manual

**Purpose:** Production deployment with environment protection

**Environments:**
- **Staging:** Auto-deploy on main push
- **Production:** Manual approval required

**Flow:**
1. Pre-deployment checks
2. Critical test suite
3. Build artifacts
4. Deploy to staging
5. Production approval gate
6. Deploy to production
7. Create GitHub release

**Example:**
```bash
# Deploy to production
gh workflow run deployment.yml -f environment=production

# Check deployment status
gh run list --workflow=deployment.yml
```

### 3. Code Quality (`code-quality.yml`)

**Trigger:** Push, pull request

**Checks (7 tools):**
- isort - Import sorting
- flake8 - Linting
- bandit - Security scanning
- interrogate - Docstring coverage
- autoflake - Unused code detection
- black - Code formatting
- pylint - Static analysis

**Features:**
- Dynamic Python version detection
- Auto source directory discovery
- Parallel execution
- Zero hardcoding

### 4. Security Scanning

**CodeQL Analysis (`codeql-analysis.yml`):**
- Multi-language (Python, JS/TS)
- Extended security queries
- SARIF upload to GitHub Security
- Daily automated scans

**Features Detected:**
- SQL injection
- XSS vulnerabilities
- Authentication issues
- Hardcoded secrets

### 5. Database Sync (`sync-databases.yml`)

**Trigger:** Every 6 hours, manual

**Purpose:** Sync local ↔ cloud databases

**Process:**
1. Export learning data from local
2. Upload to Cloud SQL
3. Aggregate statistics
4. Clean up old patterns (30+ days)
5. Create backups (7-day retention)

### 6. WebSocket Health Validation

**Trigger:** Manual, scheduled

**Test Suites (6):**
1. Connection Lifecycle
2. Self-Healing & Recovery
3. Message Delivery
4. Heartbeat Monitoring
5. Concurrent Connections
6. Latency & Performance

**Features:**
- Async/await testing
- Chaos engineering
- Stress testing
- Auto-issue creation

**Example:**
```bash
# Basic test
gh workflow run websocket-health-validation.yml

# Stress test
gh workflow run websocket-health-validation.yml \
  -f connection_count=50 \
  -f stress_test=true
```

### 7. PR Automation (`pr-automation.yml`)

**Features:**
- Auto-labeling (40+ label rules)
- Size analysis (XS/S/M/L/XL)
- Title validation (Conventional Commits)
- Description quality checks
- Conflict detection
- Review management

---

## Specialized Workflows

### Voice & Authentication

**Biometric Voice Unlock E2E (`biometric-voice-unlock-e2e.yml`):**
- Tests full voice unlock flow
- Speaker recognition validation
- Cloud SQL integration
- Security verification

**Unlock Integration E2E (`unlock-integration-e2e.yml`):**
- Complete unlock testing
- Voice + password integration
- Keychain access validation
- Error handling

### Monitoring & Alerts

**Live Monitoring Alerts (`live-monitoring-alerts.yml`):**
- Real-time system monitoring
- Auto-alerts on failures
- Slack/Discord notifications
- Health check validation

**Failed CI Auto-PR (`failed-ci-auto-pr.yml`):**
- Detects CI failures
- Creates fix PRs automatically
- Assigns to relevant developers
- Tracks fix progress

### Documentation

**Claude Docs Generator (`claude-docs-generator.yml`):**
- Auto-generates documentation
- Uses Claude AI for descriptions
- Updates API docs
- Maintains consistency

**Claude PR Analyzer (`claude-pr-analyzer.yml`):**
- AI-powered PR review
- Code quality suggestions
- Security analysis
- Performance recommendations

---

## Workflow Configuration

### Required Secrets

Configure in: `Settings > Secrets and variables > Actions`

```bash
# GCP
GCP_SA_KEY                          # Service account JSON
GCP_PROJECT_ID=jarvis-473803
GCP_VM_NAME=jarvis-backend-vm
GCP_ZONE=us-central1-a

# Cloud
ANTHROPIC_API_KEY                   # Claude API
CODECOV_TOKEN                       # Code coverage
GCP_PRODUCTION_SERVICE_ACCOUNT_KEY  # Production deployment

# Optional
SLACK_WEBHOOK_URL                   # Notifications
DISCORD_WEBHOOK_URL                 # Notifications
```

### Environment Protection

**Staging:**
- No protection rules
- Auto-deploy on main push
- Fast feedback loop

**Production:**
- Manual approval required
- Deployment windows (optional)
- Rollback capability

---

## Usage Examples

### Manual Workflow Trigger

```bash
# Deploy to staging
gh workflow run deployment.yml -f environment=staging

# Run security scan
gh workflow run codeql-analysis.yml

# Sync databases now
gh workflow run sync-databases.yml -f force_full_sync=true
```

### Check Workflow Status

```bash
# List recent runs
gh run list --limit 10

# View specific workflow
gh run view <run-id>

# Watch workflow in real-time
gh run watch <run-id>
```

### Download Artifacts

```bash
# List artifacts
gh run view <run-id> --log

# Download artifact
gh run download <run-id>
```

---

## Best Practices

### When to Commit

```
feature-branch → multi-monitor-support → main
     ↓                    ↓                ↓
   tests              tests+deploy      tests+deploy
                      (staging)         (production)
```

### Workflow Dependencies

**Sequential:**
1. Code quality checks
2. Test suite
3. Build artifacts
4. Deploy to staging
5. Integration tests
6. Deploy to production

### Performance Optimization

- Use caching for dependencies
- Run tests in parallel
- Matrix testing for platforms
- Conditional workflow execution

---

## Monitoring

### View Workflow Metrics

```bash
# Check deployment history
gh run list --workflow=deployment.yml --limit 20

# View success rate
gh run list --status=success --limit 100 | wc -l
gh run list --status=failure --limit 100 | wc -l
```

### Workflow Notifications

**Auto-configured:**
- GitHub Actions status badges
- Email notifications (on failure)
- PR comments with results

**Optional:**
- Slack integration
- Discord webhooks
- Custom notifications

---

## Troubleshooting

### Workflow Failed

**Check logs:**
```bash
gh run view <run-id> --log
```

**Common issues:**
- Missing secrets
- Permission errors
- Quota exceeded
- Test failures

### Deployment Rollback

**Automatic:**
- Health checks fail → auto-rollback
- Previous version restored
- Logs capture failure reason

**Manual:**
```bash
# Redeploy previous version
gh workflow run deployment.yml \
  -f environment=production \
  -f version=v17.3.0
```

---

**Full Documentation:**
- [.github/workflows/README.md](../.github/workflows/README.md) - Complete workflow guide
- [GitHub Actions](https://github.com/derekjrussell/JARVIS-AI-Agent/actions) - View runs

---

**Last Updated:** 2025-10-30
