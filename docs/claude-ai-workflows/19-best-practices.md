# ⭐ Best Practices - Claude AI Workflows

**Production-tested guidelines for optimal Claude AI workflow usage**

---

## 📋 Overview

This guide contains battle-tested best practices from running Claude AI workflows in production environments. Follow these guidelines to maximize value, minimize costs, and maintain code quality.

---

## 🎯 General Principles

### 1. Trust But Verify

**AI is Smart, But Not Perfect**

```python
# ✅ Good: Review AI suggestions
def review_ai_changes():
    """Always review AI-generated code before merging."""
    - Read the diff carefully
    - Understand why changes were made
    - Test the changes locally
    - Verify tests pass
    - Check for edge cases

# ❌ Bad: Blindly merge AI commits
git merge --no-verify  # Never do this
```

**Why:** AI can make mistakes, especially with complex business logic or project-specific requirements.

### 2. Provide Rich Context

**The More Context, The Better**

```yaml
# ✅ Good: Detailed PR description
title: "feat: Add user authentication with JWT"
body: |
  ## Context
  - Implementing OAuth 2.0 authentication
  - Using JWT tokens for stateless auth
  - Following OWASP ASVS Level 2 standards

  ## Technical Details
  - Framework: FastAPI 0.104
  - Database: PostgreSQL with async
  - Caching: Redis for token blacklist

  ## Security Considerations
  - Token expiry: 15 minutes (access), 7 days (refresh)
  - Password hashing: bcrypt with cost 12
  - Rate limiting: 5 attempts per minute

  ## Testing
  - Unit tests for all auth functions
  - Integration tests for login/logout flow
  - Security tests for common attacks

# ❌ Bad: Minimal description
title: "add auth"
body: "added authentication"
```

**Impact:** Detailed context improves analysis quality by 40-60%.

### 3. Use Labels Effectively

**Labels Guide AI Focus**

```bash
# Security-critical PRs
gh pr create --label security-review

# Performance-sensitive changes
gh pr create --label performance

# Breaking API changes
gh pr create --label breaking-change

# Skip AI for minor changes
gh pr create --label skip-ai
```

**Best Practice:** Create a labeling guide for your team.

### 4. Iterate on Prompts

**Optimize Prompts Over Time**

```python
# Track prompt effectiveness
prompt_versions = {
    'v1.0': {'avg_score': 7.2, 'cost': '$0.45', 'issues_found': 3.5},
    'v1.1': {'avg_score': 7.8, 'cost': '$0.38', 'issues_found': 4.2},
    'v1.2': {'avg_score': 8.1, 'cost': '$0.30', 'issues_found': 4.8},
}

# Continuously improve based on feedback
```

**Best Practice:** A/B test prompt changes before rolling out.

---

## 🔒 Security Best Practices

### 1. Never Commit Secrets

**Filter Sensitive Data**

```python
# Before sending to Claude
def sanitize_content(content):
    """Remove sensitive data from content."""
    patterns = {
        'api_keys': r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']([A-Za-z0-9_-]+)["\']',
        'passwords': r'["\']?password["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        'tokens': r'["\']?token["\']?\s*[:=]\s*["\']([A-Za-z0-9._-]+)["\']',
        'secrets': r'["\']?secret["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        'private_keys': r'-----BEGIN (RSA |EC )?PRIVATE KEY-----',
    }

    sanitized = content
    for name, pattern in patterns.items():
        sanitized = re.sub(pattern, f'[{name.upper()}_REDACTED]', sanitized)

    return sanitized
```

### 2. Use Environment-Specific Secrets

```yaml
# Production uses production API key
production:
  secrets:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_PROD }}

# Staging uses separate key with lower limits
staging:
  secrets:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_STAGING }}
```

### 3. Rotate API Keys Regularly

```bash
# Quarterly rotation schedule
Q1: Rotate ANTHROPIC_API_KEY
Q2: Review and rotate all secrets
Q3: Rotate ANTHROPIC_API_KEY
Q4: Review and rotate all secrets
```

### 4. Monitor for Anomalies

```python
# Alert on unusual activity
def check_anomalies():
    """Detect unusual API usage patterns."""
    daily_avg = get_average_daily_calls()
    today_calls = get_today_calls()

    if today_calls > daily_avg * 3:
        alert_team("Unusual API activity detected")
```

---

## 💰 Cost Management Best Practices

### 1. Set and Monitor Budgets

```yaml
# .github/workflows/cost-check.yml
env:
  DAILY_BUDGET: "5.00"
  MONTHLY_BUDGET: "100.00"
  ALERT_THRESHOLD: "0.80"  # 80%

jobs:
  check-budget:
    runs-on: ubuntu-latest
    steps:
      - name: Check if under budget
        run: |
          CURRENT=$(get_current_cost)
          if (( $(echo "$CURRENT > $DAILY_BUDGET" | bc -l) )); then
            echo "::error::Budget exceeded"
            exit 1
          fi
```

### 2. Optimize Trigger Conditions

```yaml
# Best practice trigger configuration
on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - 'src/**'
      - 'lib/**'
      - '!docs/**'
      - '!**/*.md'

jobs:
  analyze:
    if: |
      github.event.pull_request.additions +
      github.event.pull_request.deletions > 10 &&
      !contains(github.event.pull_request.labels.*.name, 'skip-ai')
```

### 3. Use Appropriate Models

```python
# Match model to task complexity
def select_model_for_task(task_type, complexity):
    """Choose most cost-effective model."""

    if task_type == 'quick_lint' or complexity == 'low':
        return 'claude-haiku-4'  # Fast, cheap

    elif task_type == 'standard_review' or complexity == 'medium':
        return 'claude-sonnet-4'  # Balanced

    elif task_type == 'architecture_review' or complexity == 'high':
        return 'claude-opus-4'  # Most capable

    return 'claude-sonnet-4'  # Default
```

### 4. Implement Smart Caching

```python
# Cache analysis results
@lru_cache(maxsize=100)
def analyze_file(file_hash, content):
    """Cache analysis for unchanged files."""
    return claude.analyze(content)

# Invalidate cache on file change
def get_file_hash(filename, content):
    """Generate hash for caching."""
    return hashlib.sha256(f"{filename}:{content}".encode()).hexdigest()
```

---

## 🧪 Testing Best Practices

### 1. Review AI-Generated Tests

```python
# ✅ Good: Review and enhance tests
def test_user_authentication():
    """Test generated by Claude, reviewed by human."""
    # AI-generated test
    user = create_user("test@example.com", "password123")
    assert authenticate(user.email, "password123")

    # Human additions for edge cases
    assert not authenticate(user.email, "wrong_password")
    assert not authenticate("nonexistent@example.com", "password123")
    with pytest.raises(ValueError):
        authenticate(None, None)
```

### 2. Verify Test Coverage

```bash
# Ensure AI tests actually improve coverage
# Before AI
pytest --cov=src --cov-report=term
# Coverage: 65%

# After AI-generated tests
pytest --cov=src --cov-report=term
# Coverage: 85%

# ✅ Good: 20% improvement
```

### 3. Test the Tests

```python
# Mutation testing to verify test quality
# Use mutpy or similar tools
mutpy --target src/auth.py --unit-test tests/test_auth.py

# Good tests should catch >80% of mutations
```

---

## 📝 Code Review Best Practices

### 1. Combine AI and Human Review

```mermaid
PR Created
    ↓
Claude AI Analysis (automated)
    ↓
Human Review (required)
    ↓
Discussion & Iteration
    ↓
Approval & Merge
```

**Best Practice:** Require both AI analysis AND human approval.

### 2. Use AI Analysis as a Checklist

```markdown
## Review Checklist (Based on Claude Analysis)

### Security
- [x] No SQL injection vulnerabilities
- [x] Input validation present
- [ ] **Action Required:** Add rate limiting (Claude found issue)

### Performance
- [x] No N+1 queries
- [ ] **Action Required:** Add database indexes (Claude suggestion)

### Testing
- [x] Unit tests present
- [ ] **Action Required:** Add edge case tests (Claude identified gaps)
```

### 3. Provide Feedback to AI

```bash
# Label AI analysis quality
gh pr comment --body "Claude analysis was helpful ✅"
gh pr comment --body "Claude missed security issue ⚠️"

# Track accuracy over time to improve prompts
```

---

## 🚀 Deployment Best Practices

### 1. Gate Deployments on AI Checks

```yaml
# Only deploy if AI checks pass
jobs:
  deploy:
    needs: [claude-security, claude-analysis]
    if: |
      needs.claude-security.result == 'success' &&
      needs.claude-analysis.outputs.score >= 7
    steps:
      - name: Deploy to production
        run: ./deploy.sh
```

### 2. Use Environment Protection

```yaml
# Require manual approval for production
production:
  environment:
    name: production
    url: https://app.example.com
  needs: claude-analysis
  if: needs.claude-analysis.outputs.critical_issues == 0
```

### 3. Implement Rollback on Issues

```yaml
- name: Deploy
  id: deploy
  run: deploy.sh

- name: Verify deployment
  run: smoke-test.sh

- name: Rollback on failure
  if: failure()
  run: rollback.sh
```

---

## 👥 Team Collaboration Best Practices

### 1. Create Team Guidelines

```markdown
# Team AI Workflow Guide

## When to Use @claude
- ✅ New features or major changes
- ✅ Security-sensitive code
- ✅ Performance-critical sections
- ❌ Simple typo fixes
- ❌ Configuration changes
- ❌ Documentation-only PRs

## How to Interpret AI Feedback
- Score 9-10: Excellent, likely ready to merge
- Score 7-8: Good, minor improvements
- Score 5-6: Needs work, address major issues
- Score 1-4: Significant problems, consider redesign

## Response Time
- Review AI feedback within 4 hours
- Address critical issues before requesting human review
- Don't wait for AI to start human review
```

### 2. Train Team on AI Tools

```bash
# Onboarding checklist
- [ ] Read quick start guide
- [ ] Review sample AI analyses
- [ ] Create first PR with AI review
- [ ] Practice using @claude commands
- [ ] Learn to interpret AI scores
- [ ] Understand when to skip AI
```

### 3. Share Knowledge

```bash
# Weekly AI insights sharing
- What did AI catch that we missed?
- What did AI miss that we caught?
- New patterns or issues discovered
- Prompt improvements made
- Cost savings achieved
```

---

## 📊 Monitoring Best Practices

### 1. Track Key Metrics

```python
metrics_to_track = {
    'quality': [
        'average_pr_score',
        'critical_issues_found',
        'bugs_caught_before_production',
        'test_coverage_improvement'
    ],
    'productivity': [
        'time_to_first_review',
        'review_iterations_saved',
        'auto_fixes_applied',
        'tests_generated'
    ],
    'cost': [
        'daily_api_cost',
        'cost_per_pr',
        'token_usage_trend',
        'optimization_savings'
    ],
    'accuracy': [
        'false_positives',
        'missed_issues',
        'ai_suggestion_adoption_rate'
    ]
}
```

### 2. Create Dashboards

```yaml
# Weekly dashboard
## AI Workflow Metrics

### This Week
- PRs Analyzed: 45
- Average Score: 7.8/10
- Critical Issues Found: 3
- Auto-Fixes Applied: 27
- Tests Generated: 18
- Cost: $15.30
- Time Saved: ~22 hours

### Trends
- Score trend: ↑ (from 7.5 last week)
- Cost trend: ↓ (from $18.20 last week)
- Adoption: 95% of PRs use AI

### Top Issues Found
1. SQL injection (3 instances)
2. Missing input validation (8 instances)
3. Inefficient queries (5 instances)
```

### 3. Regular Reviews

```bash
# Monthly review checklist
- [ ] Review cost trends
- [ ] Analyze accuracy metrics
- [ ] Gather team feedback
- [ ] Optimize prompts
- [ ] Update documentation
- [ ] Share successes/learnings
```

---

## 🔧 Maintenance Best Practices

### 1. Keep Dependencies Updated

```yaml
# Auto-update Anthropic SDK
dependabot.yml:
  updates:
    - package-ecosystem: pip
      directory: "/.github/scripts"
      schedule:
        interval: weekly
      groups:
        anthropic:
          patterns:
            - "anthropic"
```

### 2. Version Workflows

```yaml
# workflows/claude-pr-analyzer-v2.yml
name: Claude AI PR Analyzer v2
# Version: 2.1.0
# Last updated: 2025-01-15
# Changelog: Added caching, reduced tokens by 30%
```

### 3. Test Workflow Changes

```bash
# Test workflow changes before merging
1. Create feature branch
2. Update workflow
3. Add workflow_dispatch trigger for testing
4. Run manual test
5. Create PR
6. Test on PR
7. Merge if successful
```

### 4. Document Customizations

```markdown
# CUSTOMIZATIONS.md

## Team-Specific Prompts
- Added focus on FastAPI patterns
- Emphasized async/await best practices
- Included company security standards

## Custom Labels
- `needs-architect-review` - Complex architectural changes
- `db-migration` - Database schema changes
- `breaking-api-change` - API contract modifications

## Skip Patterns
- `migrations/*.sql` - Auto-generated migrations
- `*.generated.py` - Code generation output
```

---

## ⚠️ Common Pitfalls to Avoid

### 1. Over-Reliance on AI

```python
# ❌ Bad: Trust AI completely
if ai_score >= 7:
    auto_merge()  # Don't do this!

# ✅ Good: AI assists, humans decide
if ai_score >= 7 and human_approved:
    merge()
```

### 2. Ignoring AI Feedback

```python
# ❌ Bad: Dismiss AI concerns
"AI doesn't understand our architecture, ignore it"

# ✅ Good: Investigate AI concerns
"AI flagged this, let's understand why and decide if it's valid"
```

### 3. Not Providing Feedback

```python
# ❌ Bad: Silent about AI mistakes
# AI gives false positive, do nothing

# ✅ Good: Track and improve
log_ai_mistake(issue_id, "false_positive", reason)
# Use to improve prompts
```

### 4. Excessive API Calls

```python
# ❌ Bad: Trigger on every commit
on:
  push:
    branches: ['**']

# ✅ Good: Trigger strategically
on:
  pull_request:
    types: [opened, synchronize]
```

### 5. Poor Error Handling

```python
# ❌ Bad: Fail silently
try:
    analysis = claude.analyze(pr)
except Exception:
    pass  # Don't do this

# ✅ Good: Graceful degradation
try:
    analysis = claude.analyze(pr)
except AnthropicAPIError as e:
    logger.error(f"AI analysis failed: {e}")
    fallback_to_static_analysis()
    notify_team("AI temporarily unavailable")
```

---

## 🎯 Success Patterns

### 1. Start Small, Scale Gradually

```
Week 1: Enable on 1 repository
Week 2-3: Gather feedback, iterate
Week 4: Enable on 3 repositories
Month 2: Roll out to team
Month 3: Optimize and scale
```

### 2. Measure and Communicate Value

```markdown
## Monthly Value Report

### Bugs Prevented
- Critical: 2 (estimated cost: $10,000)
- Major: 8 (estimated cost: $5,000)
- Minor: 15 (estimated cost: $1,000)

### Time Saved
- Code review: 25 hours ($2,500)
- Test writing: 18 hours ($1,800)
- Bug fixing: 12 hours ($1,200)

### Total Value: $21,500
### AI Cost: $45
### ROI: 478x
```

### 3. Continuous Improvement

```python
improvement_cycle = [
    'Measure current state',
    'Identify bottlenecks',
    'Test improvements',
    'Roll out changes',
    'Measure impact',
    'Repeat'
]
```

---

## 📚 Additional Resources

### Internal Documentation
- [Quick Start Guide](./01-quick-start.md)
- [Architecture Overview](./04-architecture-overview.md)
- [Cost Optimization](./13-cost-optimization.md)
- [Troubleshooting](./17-troubleshooting.md)

### External Resources
- [Anthropic Best Practices](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [GitHub Actions Security](https://docs.github.com/en/actions/security-guides)
- [OWASP Secure Coding](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

---

## 🎓 Summary Checklist

### Essential Practices
- [ ] Always review AI-generated code
- [ ] Provide detailed PR descriptions
- [ ] Use labels to guide AI focus
- [ ] Set and monitor budgets
- [ ] Filter sensitive data
- [ ] Track metrics and ROI
- [ ] Iterate on prompts
- [ ] Train team on tools
- [ ] Combine AI and human review
- [ ] Test workflow changes

### Nice to Have
- [ ] A/B test prompts
- [ ] Create custom dashboards
- [ ] Implement smart caching
- [ ] Share team learnings
- [ ] Document customizations

---

**Remember:** Claude AI is a powerful assistant, not a replacement for human judgment. Use it to augment your team's capabilities, catch issues early, and improve code quality—but always maintain human oversight and decision-making.

---

[← Back to Index](./README.md)
