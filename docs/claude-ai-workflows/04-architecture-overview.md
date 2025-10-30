# 🏗️ Architecture Overview - Claude AI Workflows

**Complete system architecture and design principles**

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Overview](#component-overview)
3. [Data Flow](#data-flow)
4. [Security Model](#security-model)
5. [Scaling & Performance](#scaling--performance)
6. [Design Principles](#design-principles)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Code Changes                            │  │
│  │  • Pull Requests  • Commits  • Issues  • Comments        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Webhook Events
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions Platform                       │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                 Workflow Orchestration                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │ │
│  │  │   Standard  │  │     AI      │  │    Security     │   │ │
│  │  │  Workflows  │  │  Workflows  │  │    Scanning     │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   Execution Environment                    │ │
│  │  • Ubuntu Runners  • Python  • Node.js  • Docker         │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ API Calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services Layer                       │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Anthropic API   │  │   GitHub API     │  │  CodeQL DB   │ │
│  │  Claude Sonnet 4 │  │  REST & GraphQL  │  │   Security   │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Results
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Automated Actions                          │
│                                                                   │
│  • PR Comments        • Commits        • Labels                 │
│  • Status Checks      • Issues         • Notifications          │
│  • Deployments        • Reviews        • Analytics              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

#### Layer 1: Event Sources
- **Pull Requests** - Code changes, reviews, comments
- **Issues** - Bug reports, feature requests
- **Commits** - Direct pushes to branches
- **Schedules** - Cron-based triggers
- **Manual** - Workflow dispatch events

#### Layer 2: Workflow Engine
- **GitHub Actions** - Orchestration platform
- **YAML Workflows** - Declarative configuration
- **Runners** - Execution environment
- **Secrets Management** - Secure credential storage

#### Layer 3: Processing
- **Standard Workflows** - Linting, testing, building
- **AI Workflows** - Claude-powered analysis
- **Security Workflows** - Vulnerability scanning
- **Deployment Workflows** - Release automation

#### Layer 4: External Services
- **Anthropic API** - Claude AI models
- **GitHub API** - Repository operations
- **CodeQL** - Security analysis
- **Third-party Tools** - Linters, scanners

#### Layer 5: Actions
- **Commenting** - Post feedback on PRs
- **Committing** - Auto-fix code issues
- **Labeling** - Categorize PRs/issues
- **Status Updates** - Pass/fail checks
- **Deployments** - Trigger releases

---

## Component Overview

### 1. AI-Powered Workflows

#### Claude PR Analyzer
```yaml
Trigger: pull_request [opened, synchronize, reopened]
Input: PR diff, metadata, context
Processing: Claude Sonnet 4 analysis
Output: Code review comment, labels, status
Cost: $0.20-0.50 per PR
Duration: 2-3 minutes
```

**Key Features:**
- Context-aware code review
- Security vulnerability detection
- Architecture analysis
- Performance recommendations
- Testing gap identification

**Technology Stack:**
```
GitHub Actions → Python Script → Anthropic API → Claude Sonnet 4
    ↓                                                    ↓
GitHub API ← Parse Response ← JSON Response ← Analysis
```

#### Claude Auto-Fix
```yaml
Trigger: pull_request [opened], comment with "@claude fix"
Input: Claude PR Analyzer findings
Processing: Generate and apply fixes
Output: New commit with improvements
Cost: $0.25-0.40 per fix
Duration: 3-4 minutes
```

**Fix Categories:**
- Security vulnerabilities
- Code style issues
- Type hints addition
- Docstring generation
- Performance optimizations

#### Claude Test Generator
```yaml
Trigger: pull_request [opened], comment with "@claude generate tests"
Input: Changed code files
Processing: Analyze code, generate tests
Output: New test files committed
Cost: $0.20-0.35 per generation
Duration: 3-4 minutes
```

**Test Types:**
- Unit tests
- Integration tests
- Edge case coverage
- Error condition testing
- Mocking and fixtures

#### Claude Security Analyzer
```yaml
Trigger: pull_request [opened], schedule (daily)
Input: Codebase + security context
Processing: Deep security analysis
Output: Security report, issues, blocks
Cost: $0.15-0.30 per scan
Duration: 2-3 minutes
```

**Security Checks:**
- OWASP Top 10
- CWE common weaknesses
- Dependency vulnerabilities
- Configuration issues
- Secrets detection

#### Claude Docs Generator
```yaml
Trigger: pull_request [opened], comment with "@claude generate docs"
Input: Changed code files
Processing: Generate documentation
Output: Updated docs committed
Cost: $0.10-0.20 per generation
Duration: 2-3 minutes
```

**Documentation Types:**
- Docstrings (Google style)
- README updates
- API documentation
- Code examples
- Type hints

### 2. Standard Workflows

#### Super-Linter
```yaml
Trigger: pull_request
Tools: Black, Flake8, Pylint, ESLint, ShellCheck
Output: Linting report
Cost: Free (GitHub Actions minutes)
Duration: 1-2 minutes
```

#### CodeQL Analysis
```yaml
Trigger: pull_request, push, schedule (daily)
Languages: Python, JavaScript, TypeScript
Output: SARIF security report
Cost: Free (GitHub CodeQL)
Duration: 3-5 minutes
```

#### CI/CD Pipeline
```yaml
Trigger: pull_request
Phases: 6 (quality, test, architecture, performance, security, reporting)
Matrix: Python 3.10/3.11, Ubuntu/macOS
Output: Test results, coverage
Cost: Free (GitHub Actions minutes)
Duration: 5-7 minutes
```

#### Deployment
```yaml
Trigger: Manual, merge to main
Environments: staging, production
Strategy: Blue-green, rolling updates
Output: Deployed application
Cost: Free (GitHub Actions minutes)
Duration: 10-15 minutes
```

### 3. Supporting Components

#### Database Validation
```yaml
Trigger: pull_request, schedule (daily)
Checks: Cloud SQL config, connection strings, migrations
Output: Validation report
```

#### Environment Variable Validation
```yaml
Trigger: pull_request, schedule (daily)
Coverage: 80% documentation required
Output: Coverage report, missing vars
```

#### PR Automation
```yaml
Trigger: pull_request
Actions: Auto-labeling, size analysis, reviewer assignment
Output: Labels, comments, assignments
```

---

## Data Flow

### PR Analysis Flow

```
1. PR Created/Updated
   ↓
2. GitHub Webhook → Workflow Trigger
   ↓
3. Workflow Runner Starts
   ↓
4. Checkout Code (main + PR branch)
   ↓
5. Fetch PR Metadata
   │  • Title, description, author
   │  • Files changed, diff
   │  • Labels, comments
   │  • Commit history
   ↓
6. Build Context Object
   │  {
   │    pr_info: {...},
   │    changes: [...],
   │    codebase_context: {...}
   │  }
   ↓
7. Call Anthropic API
   │  POST /v1/messages
   │  {
   │    model: "claude-sonnet-4-20250514",
   │    system: "You are an expert...",
   │    messages: [{role: "user", content: context}]
   │  }
   ↓
8. Claude Processes Request
   │  • Analyzes code changes
   │  • Identifies issues
   │  • Generates recommendations
   │  • Scores quality
   ↓
9. Receive Response
   │  {
   │    content: [{text: "## Claude AI Review..."}],
   │    usage: {input_tokens: X, output_tokens: Y}
   │  }
   ↓
10. Parse Analysis
    │  • Extract score
    │  • Parse sections
    │  • Identify labels
    ↓
11. Take Actions
    │  • Post comment via GitHub API
    │  • Add labels
    │  • Update check status
    │  • (Optional) Create commits
    ↓
12. Workflow Complete
    │  • Log metrics
    │  • Update dashboard
```

### Auto-Fix Flow

```
1. Trigger Event
   │  • PR opened with issues
   │  • "@claude fix" comment
   ↓
2. Fetch Analysis Results
   │  • Read Claude PR Analyzer output
   │  • Identify fixable issues
   ↓
3. Generate Fix Prompts
   │  For each fixable issue:
   │  {
   │    file: "path/to/file.py",
   │    issue: "SQL injection vulnerability",
   │    current_code: "...",
   │    context: "..."
   │  }
   ↓
4. Request Fixes from Claude
   │  POST /v1/messages
   │  "Generate fix for this issue..."
   ↓
5. Receive Fixed Code
   │  {
   │    content: [{text: "Here's the fixed code:\n```python..."}]
   │  }
   ↓
6. Parse and Validate
   │  • Extract code blocks
   │  • Validate syntax
   │  • Run basic checks
   ↓
7. Apply Fixes
   │  • Checkout PR branch
   │  • Modify files
   │  • Run formatters
   ↓
8. Create Commit
   │  git config user.name "claude-ai[bot]"
   │  git commit -m "fix: Apply Claude AI fixes"
   │  git push
   ↓
9. Post Summary
   │  Comment on PR with fixes applied
```

### Security Scan Flow

```
1. Daily Schedule Trigger (4 AM UTC)
   ↓
2. Full Codebase Checkout
   ↓
3. Parallel Security Scans
   ├─ CodeQL (GitHub native)
   ├─ Bandit (Python)
   ├─ Safety (Dependencies)
   └─ Claude AI Security Analysis
   ↓
4. Claude Deep Analysis
   │  • AI-specific vulnerabilities
   │  • Prompt injection risks
   │  • Model poisoning
   │  • Data leakage
   ↓
5. Aggregate Results
   │  {
   │    codeql: [...],
   │    bandit: [...],
   │    safety: [...],
   │    claude: [...]
   │  }
   ↓
6. Deduplicate Issues
   │  • Merge similar findings
   │  • Prioritize by severity
   ↓
7. Create Security Report
   │  • SARIF format
   │  • Markdown summary
   ↓
8. Take Actions
   │  Critical: Create issue, block deploys
   │  Major: Create issue
   │  Minor: Log for review
   ↓
9. Notify Team
   │  • Slack/Discord webhook
   │  • Email digest
```

---

## Security Model

### Secrets Management

```yaml
Secrets Hierarchy:
  Repository Secrets (Most PRs)
    ↓
  Environment Secrets (Production)
    ↓
  Organization Secrets (Shared)
```

**Used Secrets:**
- `ANTHROPIC_API_KEY` - Claude AI access
- `GITHUB_TOKEN` - Auto-generated, PR operations
- `CODECOV_TOKEN` - (Optional) Coverage reporting
- `SLACK_WEBHOOK_URL` - (Optional) Notifications

**Security Practices:**
```yaml
# Secrets are:
- ✅ Encrypted at rest
- ✅ Not logged
- ✅ Not accessible in forks
- ✅ Scoped per environment
- ✅ Rotatable without code changes
```

### Permission Model

```yaml
GitHub Actions Permissions:
  contents: write       # Create commits
  pull-requests: write  # Comment, label
  issues: write         # Create security issues
  checks: write         # Update check status
  actions: read         # Read workflow runs
```

**Principle of Least Privilege:**
```yaml
# Each workflow only requests needed permissions
jobs:
  analyze:
    permissions:
      contents: read
      pull-requests: write  # Only needs to comment
```

### Data Handling

**What's Sent to Anthropic:**
- ✅ Code changes (diff)
- ✅ PR metadata (title, description)
- ✅ File names and structure
- ❌ NOT secrets (filtered)
- ❌ NOT credentials
- ❌ NOT environment variables

**Data Filtering:**
```python
# Before sending to Claude
def filter_sensitive_data(content):
    """Remove sensitive data from content."""
    patterns = [
        r'api[_-]?key["\s:=]+[A-Za-z0-9]+',
        r'password["\s:=]+[^\s]+',
        r'secret["\s:=]+[^\s]+',
        r'token["\s:=]+[A-Za-z0-9]+',
    ]

    filtered = content
    for pattern in patterns:
        filtered = re.sub(pattern, '[REDACTED]', filtered)

    return filtered
```

**Anthropic Data Policy:**
- ✅ No training on your data
- ✅ Not stored long-term
- ✅ Encrypted in transit
- ✅ SOC 2 Type II certified
- ✅ GDPR compliant

### Access Control

```yaml
Branch Protection (main):
  required_reviews: 1
  required_checks:
    - Claude AI PR Analyzer
    - Claude AI Security Analyzer
    - Super-Linter
    - CodeQL
  restrict_pushes: true
  allow_force_pushes: false

Environment Protection (production):
  required_reviewers: ["lead-dev", "devops-team"]
  wait_timer: 300  # 5 minute delay
  deployment_branches: ["main"]
```

---

## Scaling & Performance

### Concurrency Control

```yaml
# Prevent duplicate runs
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Impact:**
- New commits cancel ongoing analysis
- Saves compute time
- Reduces API costs

### Caching Strategy

```yaml
# Cache dependencies
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      node_modules
    key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements.txt') }}

# Cache improves speed by 60-80%
```

### Parallel Execution

```yaml
# Run independent workflows in parallel
jobs:
  lint:
    runs-on: ubuntu-latest
  test:
    runs-on: ubuntu-latest
  security:
    runs-on: ubuntu-latest

# All run simultaneously
```

### Resource Limits

```yaml
# Prevent runaway workflows
timeout-minutes: 15

# Matrix strategy for large test suites
strategy:
  matrix:
    python-version: [3.10, 3.11]
    os: [ubuntu-latest, macos-latest]
  fail-fast: false
```

### API Rate Limiting

```python
# Handle Anthropic rate limits
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def call_claude_api(context):
    return client.messages.create(...)
```

### Cost Optimization

**Smart Triggering:**
```yaml
# Skip docs-only changes
if: |
  github.event.pull_request.additions + github.event.pull_request.deletions > 10 &&
  !contains(github.event.pull_request.files.*.filename, 'docs/')
```

**Token Management:**
```python
# Truncate large files intelligently
def truncate_file(content, max_tokens=10000):
    """Keep beginning and end, truncate middle."""
    if estimate_tokens(content) <= max_tokens:
        return content

    lines = content.split('\n')
    keep_top = lines[:100]
    keep_bottom = lines[-100:]

    return '\n'.join([
        *keep_top,
        f'\n... [{len(lines) - 200} lines truncated] ...\n',
        *keep_bottom
    ])
```

---

## Design Principles

### 1. Zero Hardcoding

**Traditional Approach:**
```python
# ❌ Hardcoded patterns
if "password" in variable_name and not has_hashing:
    report_issue("Use hashed passwords")
```

**Claude AI Approach:**
```python
# ✅ Context-aware intelligence
context = f"""
Analyze this authentication code considering:
- Industry best practices
- Security implications
- Framework-specific patterns
- Project requirements
"""
analysis = claude.analyze(code, context)
```

### 2. Context-Aware Decisions

```python
# AI considers:
- Project type (API, library, CLI, etc.)
- Language and framework
- Existing patterns in codebase
- Team coding standards
- Security requirements
- Performance needs
```

### 3. Explainable Actions

```markdown
Every AI action includes:
- What was changed
- Why it was changed
- Impact of the change
- References (CWE, OWASP, etc.)
- How to test the change
```

### 4. Fail-Safe Operations

```python
# Never auto-merge without checks
if all_checks_passed and is_safe_update:
    # Still requires approval for production
    if environment == "production":
        require_manual_approval()
    else:
        auto_merge()
```

### 5. Progressive Enhancement

```yaml
# Works without AI if API unavailable
- name: Analyze with Claude
  id: claude
  continue-on-error: true

- name: Fallback to static analysis
  if: steps.claude.outcome == 'failure'
  run: flake8 src/
```

### 6. Observable System

```python
# Every action is logged and traceable
logger.info({
    'event': 'pr_analyzed',
    'pr_number': pr.number,
    'tokens_used': response.usage.input_tokens,
    'cost': calculate_cost(response.usage),
    'duration': time.time() - start_time,
    'score': parsed_score,
    'issues_found': len(issues)
})
```

---

## Technology Stack

### Languages
- **Python 3.11** - AI workflow scripts
- **Bash** - Shell automation
- **YAML** - Workflow configuration
- **JavaScript/TypeScript** - CodeQL queries

### Frameworks & Libraries
- **Anthropic SDK** - Claude AI integration
- **PyGithub** - GitHub API client
- **pytest** - Testing framework
- **Black** - Code formatting
- **Flake8, Pylint** - Linting

### Platforms & Services
- **GitHub Actions** - CI/CD platform
- **Anthropic API** - Claude AI models
- **GitHub API** - Repository operations
- **CodeQL** - Security analysis

### Infrastructure
- **GitHub Runners** - Ubuntu latest
- **Docker** - Containerization
- **Cloud SQL Proxy** - Database connections
- **Redis** - Caching (application layer)

---

## Deployment Architecture

### Workflow Deployment

```
Development:
  ├─ Create feature branch
  ├─ Edit .github/workflows/*.yml
  ├─ Test with workflow_dispatch
  ├─ Create PR
  ├─ Review and test
  └─ Merge to main
       ↓
Production:
  └─ Workflows active immediately
```

### Version Management

```yaml
# Pin action versions for stability
- uses: actions/checkout@v4
- uses: actions/setup-python@v5

# Use commit SHAs for security-critical actions
- uses: anthropics/action@a1b2c3d4e5f6...
```

### Rollback Strategy

```bash
# Rollback workflow to previous version
git revert <commit-hash>
git push origin main

# Or temporarily disable
gh workflow disable <workflow-name>
```

---

## Monitoring & Observability

### Metrics Tracked

```yaml
Per Workflow Run:
  - Duration
  - Token usage
  - API cost
  - Success/failure rate
  - Issues found
  - Fixes applied

Aggregated:
  - Daily/weekly/monthly costs
  - Average PR analysis time
  - Issue detection rate
  - Auto-fix success rate
  - Coverage trends
```

### Alerting

```yaml
Alert Conditions:
  - Workflow failure rate > 10%
  - Daily API cost > $50
  - Critical security issue found
  - Auto-fix validation failed
```

### Dashboards

```
GitHub Actions Insights:
  - Workflow runs timeline
  - Success rate graphs
  - Duration trends
  - Resource usage

Custom Dashboard:
  - AI analysis metrics
  - Cost tracking
  - Issue trends
  - Coverage evolution
```

---

## Future Architecture

### Planned Enhancements

1. **Multi-Model Support**
   - Claude Opus for complex analyses
   - Claude Haiku for quick scans
   - Model selection based on PR complexity

2. **Distributed Analysis**
   - Split large PRs across multiple API calls
   - Parallel file analysis
   - Aggregate results

3. **Learning System**
   - Track analysis accuracy
   - Improve prompts based on feedback
   - Team-specific tuning

4. **Advanced Caching**
   - Cache analysis for unchanged files
   - Incremental analysis
   - Cross-PR insights

---

## 📖 Related Documentation

- [PR Analyzer Deep Dive](./07-pr-analyzer-deep-dive.md)
- [Security & Privacy](./21-security-privacy.md)
- [Performance Tuning](./14-performance-tuning.md)
- [Best Practices](./19-best-practices.md)

---

[← Back to Index](./README.md) | [Next: Workflow Reference →](./05-workflow-reference.md)
