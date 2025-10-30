# 💰 Cost Optimization Guide - Claude AI Workflows

**Comprehensive strategies for managing and reducing AI workflow costs**

---

## 📊 Cost Overview

### Baseline Costs

**Anthropic Claude Sonnet 4 Pricing:**
- Input tokens: $0.003 per 1,000 tokens
- Output tokens: $0.015 per 1,000 tokens

**Typical Monthly Costs:**

| Activity | Volume | Cost Range |
|----------|--------|------------|
| Small team (5-10 PRs/week) | 20-40 PRs/month | $10-20/month |
| Medium team (10-25 PRs/week) | 40-100 PRs/month | $20-50/month |
| Large team (25-50 PRs/week) | 100-200 PRs/month | $50-100/month |
| Enterprise (50+ PRs/week) | 200+ PRs/month | $100-300/month |

**Per-Operation Costs:**

| Operation | Token Usage | Typical Cost |
|-----------|-------------|--------------|
| Small PR analysis (1-5 files) | 2k in / 1.5k out | $0.10 |
| Medium PR analysis (6-20 files) | 10k in / 3k out | $0.35 |
| Large PR analysis (21-50 files) | 25k in / 5k out | $0.90 |
| Auto-fix (minor changes) | 5k in / 2k out | $0.20 |
| Test generation | 8k in / 4k out | $0.30 |
| Security scan (full repo) | 15k in / 3k out | $0.50 |
| Documentation generation | 5k in / 3k out | $0.20 |

---

## 🎯 Optimization Strategies

### 1. Smart Triggering

#### Skip Trivial Changes

```yaml
# .github/workflows/claude-pr-analyzer.yml
on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  analyze:
    # Skip very small PRs
    if: |
      github.event.pull_request.additions +
      github.event.pull_request.deletions > 10

    # Skip docs-only PRs
    if: |
      !contains(github.event.pull_request.files.*.filename, 'docs/') ||
      github.event.pull_request.changed_files > 1
```

**Savings:** 20-30% of unnecessary runs

#### Path-Based Triggering

```yaml
on:
  pull_request:
    paths:
      - 'src/**'
      - 'lib/**'
      - '!docs/**'
      - '!*.md'
      - '!.github/**'
```

**Savings:** 15-25% by skipping non-code changes

#### Label-Based Control

```yaml
jobs:
  analyze:
    # Allow opting out
    if: |
      !contains(github.event.pull_request.labels.*.name, 'skip-ai') &&
      !contains(github.event.pull_request.labels.*.name, 'wip')
```

**Usage:**
```bash
# Skip AI analysis for work in progress
gh pr create --label wip

# Re-enable when ready
gh pr edit --remove-label wip
```

**Savings:** 10-20% for draft/WIP PRs

### 2. Content Optimization

#### File Filtering

```python
# Only analyze relevant files
SKIP_PATTERNS = [
    '*.lock',           # Lock files (package-lock.json, poetry.lock)
    '*.min.js',         # Minified JavaScript
    '*.min.css',        # Minified CSS
    'dist/*',           # Build outputs
    'build/*',          # Compiled files
    'node_modules/*',   # Dependencies
    '__pycache__/*',    # Python cache
    '*.pyc',            # Compiled Python
    'venv/*',           # Virtual environments
    '.env',             # Environment files
    '*.png', '*.jpg',   # Images
    '*.pdf',            # Documents
    'migrations/*',     # Database migrations (usually auto-generated)
]

def should_analyze_file(filename):
    """Determine if file should be sent to Claude."""
    for pattern in SKIP_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return False
    return True
```

**Savings:** 30-50% token reduction

#### File Size Limits

```python
# Skip very large files
MAX_FILE_SIZE = 100_000  # 100KB

def filter_large_files(files):
    """Filter out files that are too large."""
    filtered = []
    for file in files:
        if file.size <= MAX_FILE_SIZE:
            filtered.append(file)
        else:
            logger.info(f'Skipping large file: {file.filename} ({file.size} bytes)')
    return filtered
```

**Savings:** 10-20% for repos with large files

#### Intelligent Truncation

```python
def truncate_content(content, max_tokens=10000):
    """
    Intelligently truncate file content while preserving context.

    Strategy:
    - Keep first 100 lines (imports, setup)
    - Keep last 100 lines (recent changes often at end)
    - Truncate middle
    """
    if estimate_tokens(content) <= max_tokens:
        return content

    lines = content.split('\n')

    if len(lines) <= 200:
        return content

    # Keep beginning and end
    keep_top = lines[:100]
    keep_bottom = lines[-100:]

    truncated_count = len(lines) - 200

    return '\n'.join([
        *keep_top,
        f'\n... [{truncated_count} lines truncated for brevity] ...\n',
        *keep_bottom
    ])
```

**Savings:** 20-40% for large files

#### Diff-Only Analysis

```python
# Send only changed lines instead of full files
def get_minimal_context(file):
    """Extract minimal context around changes."""

    if file.patch:
        # For modified files, use git patch
        return {
            'filename': file.filename,
            'status': file.status,
            'patch': file.patch,  # Only changed lines
        }
    elif file.status == 'added':
        # For new files, include full content
        return {
            'filename': file.filename,
            'status': 'added',
            'content': get_file_content(file)
        }
    else:
        # For removed files, just note deletion
        return {
            'filename': file.filename,
            'status': 'removed'
        }
```

**Savings:** 40-60% for large files with small changes

### 3. Model Selection

#### Use Appropriate Models

```python
def select_model(pr_context):
    """Select most cost-effective model for the task."""

    files_changed = pr_context['files_changed']
    total_changes = pr_context['additions'] + pr_context['deletions']

    # Small, simple PRs → Haiku (cheapest)
    if files_changed <= 3 and total_changes <= 50:
        return 'claude-haiku-4-20250514'

    # Medium PRs → Sonnet (balanced)
    elif files_changed <= 20 and total_changes <= 500:
        return 'claude-sonnet-4-20250514'

    # Large, complex PRs → Opus (most capable)
    else:
        return 'claude-opus-4-20250514'

# Cost comparison:
# Haiku: $0.25 per 1M input tokens  (12x cheaper than Sonnet)
# Sonnet: $3 per 1M input tokens
# Opus: $15 per 1M input tokens (5x more expensive than Sonnet)
```

**Savings:** 50-70% for simple PRs using Haiku

#### Specialized Workflows

```yaml
# Use Haiku for simple tasks
claude-quick-check.yml:
  model: claude-haiku-4-20250514
  trigger: Small PRs (< 5 files)
  focus: Quick security scan, obvious issues

# Use Sonnet for standard analysis
claude-pr-analyzer.yml:
  model: claude-sonnet-4-20250514
  trigger: Normal PRs
  focus: Comprehensive review

# Use Opus for critical PRs
claude-deep-review.yml:
  model: claude-opus-4-20250514
  trigger: Label "needs-deep-review"
  focus: Architecture, complex logic
```

**Savings:** 30-50% overall

### 4. Caching Strategies

#### Analysis Result Caching

```python
import hashlib
import json

def get_cache_key(pr_context):
    """Generate cache key based on PR content."""
    content = json.dumps({
        'files': sorted([f['filename'] for f in pr_context['files']]),
        'file_hashes': {
            f['filename']: hashlib.sha256(f['content'].encode()).hexdigest()
            for f in pr_context['files']
        }
    })
    return hashlib.sha256(content.encode()).hexdigest()

def get_cached_analysis(cache_key):
    """Retrieve cached analysis if available."""
    cache_file = f'.github/cache/claude-{cache_key}.json'

    if os.path.exists(cache_file):
        # Check if cache is fresh (< 7 days)
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < 7 * 24 * 3600:
            with open(cache_file) as f:
                return json.load(f)

    return None

def cache_analysis(cache_key, analysis):
    """Store analysis result in cache."""
    os.makedirs('.github/cache', exist_ok=True)
    cache_file = f'.github/cache/claude-{cache_key}.json'

    with open(cache_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'analysis': analysis
        }, f)
```

**Savings:** 80-90% for re-analysis of unchanged code

#### Incremental Analysis

```python
def analyze_incrementally(pr_context, previous_analysis):
    """Analyze only new/changed files since last analysis."""

    if not previous_analysis:
        return full_analysis(pr_context)

    # Get files that changed since last analysis
    new_files = get_new_files(pr_context, previous_analysis)

    if not new_files:
        return previous_analysis

    # Analyze only new files
    new_analysis = analyze_files(new_files)

    # Merge with previous analysis
    return merge_analyses(previous_analysis, new_analysis)
```

**Savings:** 60-80% for PRs with multiple commits

### 5. Token Management

#### Reduce Max Tokens

```yaml
# Default: 8000 tokens output
- name: Analyze PR
  env:
    MAX_TOKENS: 8000

# Optimized: 4000 tokens (still comprehensive)
- name: Analyze PR
  env:
    MAX_TOKENS: 4000
```

**Output quality impact:** Minimal
**Savings:** 50% on output tokens

#### Prompt Optimization

```python
# ❌ Verbose prompt (wastes input tokens)
prompt = f"""
Please carefully review this pull request and provide a very detailed
comprehensive analysis covering all aspects of code quality including
but not limited to security, performance, maintainability, testability,
documentation, and architectural considerations. Be very thorough and
provide extensive explanations for every finding.

Pull Request Title: {title}
Pull Request Description: {description}
Files changed: {files}
...
"""

# ✅ Concise prompt (saves input tokens)
prompt = f"""
Review PR: {title}

Changes:
{format_changes_compact(files)}

Focus: Security, performance, architecture.
Format: Issues → Recommendations
"""
```

**Savings:** 20-40% on input tokens

#### Response Formatting

```python
system_prompt = """
Provide concise, actionable feedback.

Format:
## Score: X/10

### Critical
- Issue 1 (file:line): Description → Fix

### Major
- Issue 2: Description → Fix

### Recommendations
- Improvement 1 → Suggestion

AVOID:
- Lengthy explanations
- Obvious statements
- Repetition
"""
```

**Savings:** 30-50% on output tokens

### 6. Conditional Workflows

#### Run Based on Complexity

```yaml
jobs:
  assess:
    runs-on: ubuntu-latest
    outputs:
      complexity: ${{ steps.check.outputs.complexity }}
    steps:
      - id: check
        run: |
          FILES=${{ github.event.pull_request.changed_files }}
          CHANGES=${{ github.event.pull_request.additions }}

          if [ $FILES -lt 5 ] && [ $CHANGES -lt 100 ]; then
            echo "complexity=low" >> $GITHUB_OUTPUT
          elif [ $FILES -lt 20 ] && [ $CHANGES -lt 500 ]; then
            echo "complexity=medium" >> $GITHUB_OUTPUT
          else
            echo "complexity=high" >> $GITHUB_OUTPUT
          fi

  quick-check:
    needs: assess
    if: needs.assess.outputs.complexity == 'low'
    # Run lightweight checks only

  full-analysis:
    needs: assess
    if: needs.assess.outputs.complexity == 'medium'
    # Run standard Claude analysis

  deep-review:
    needs: assess
    if: needs.assess.outputs.complexity == 'high'
    # Run comprehensive analysis with Opus
```

**Savings:** 40-60% by matching analysis depth to PR complexity

### 7. Batch Processing

#### Combine Multiple Analyses

```python
# ❌ Separate API calls
security_analysis = claude.analyze(code, focus='security')
performance_analysis = claude.analyze(code, focus='performance')
style_analysis = claude.analyze(code, focus='style')

# Cost: 3x API calls

# ✅ Single comprehensive analysis
combined_analysis = claude.analyze(code, focus='security,performance,style')

# Cost: 1x API call
```

**Savings:** 60-70% by reducing API overhead

### 8. Auto-Fix Optimization

#### Selective Auto-Fix

```python
# Only auto-fix safe, low-risk issues
SAFE_FIX_CATEGORIES = [
    'missing_docstring',
    'missing_type_hint',
    'unused_import',
    'trailing_whitespace',
    'line_too_long',
    'missing_comma',
]

def should_auto_fix(issue):
    """Determine if issue is safe for auto-fix."""
    return (
        issue['category'] in SAFE_FIX_CATEGORIES and
        issue['risk_level'] == 'low' and
        issue['confidence'] > 0.9
    )
```

**Savings:** 50-70% by avoiding unnecessary fix generation

#### Batch Fixes

```python
# ❌ Fix issues one by one
for issue in issues:
    fix = generate_fix(issue)
    apply_fix(fix)

# Cost: N API calls

# ✅ Generate all fixes in one request
all_fixes = generate_fixes_batch(issues)
apply_fixes(all_fixes)

# Cost: 1 API call
```

**Savings:** 80-90% for multiple fixes

---

## 📊 Monitoring Costs

### Cost Tracking Script

```python
# .github/scripts/track_costs.py
import json
from datetime import datetime, timedelta
from collections import defaultdict

class CostTracker:
    def __init__(self):
        self.costs = defaultdict(lambda: {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0
        })

    def log_api_call(self, workflow, input_tokens, output_tokens):
        """Log API call and calculate cost."""
        cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)

        self.costs[workflow]['input_tokens'] += input_tokens
        self.costs[workflow]['output_tokens'] += output_tokens
        self.costs[workflow]['total_cost'] += cost
        self.costs[workflow]['api_calls'] += 1

        # Log to file
        with open('.github/costs/log.jsonl', 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'workflow': workflow,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost
            }) + '\n')

    def get_daily_report(self):
        """Generate daily cost report."""
        total_cost = sum(w['total_cost'] for w in self.costs.values())
        total_calls = sum(w['api_calls'] for w in self.costs.values())

        return {
            'total_cost': f'${total_cost:.2f}',
            'total_calls': total_calls,
            'by_workflow': dict(self.costs),
            'average_per_call': f'${total_cost / total_calls:.2f}' if total_calls > 0 else '$0.00'
        }
```

### Cost Dashboard

```python
# Generate weekly cost report
def generate_cost_dashboard():
    """Create cost visualization."""
    logs = load_cost_logs()

    # Calculate metrics
    daily_costs = calculate_daily_costs(logs)
    workflow_breakdown = calculate_workflow_breakdown(logs)
    trend = calculate_cost_trend(logs)

    # Create markdown report
    report = f"""
# 💰 Weekly Cost Report

## Summary
- **Total Spent:** ${sum(daily_costs):.2f}
- **Daily Average:** ${sum(daily_costs) / 7:.2f}
- **Trend:** {"📈 Increasing" if trend > 0 else "📉 Decreasing"}

## By Workflow
| Workflow | Calls | Cost | Avg/Call |
|----------|-------|------|----------|
{format_workflow_table(workflow_breakdown)}

## Daily Costs
{format_daily_chart(daily_costs)}

## Optimization Opportunities
{identify_optimization_opportunities(logs)}
    """

    return report
```

### Budget Alerts

```python
# Set spending limits
DAILY_BUDGET = 5.00    # $5/day
MONTHLY_BUDGET = 100.00  # $100/month

def check_budget_alerts():
    """Alert if approaching budget limits."""
    daily_cost = get_today_cost()
    monthly_cost = get_month_cost()

    alerts = []

    # Daily check
    if daily_cost > DAILY_BUDGET * 0.8:
        alerts.append({
            'level': 'warning',
            'message': f'Daily cost at 80%: ${daily_cost:.2f} / ${DAILY_BUDGET:.2f}'
        })

    if daily_cost > DAILY_BUDGET:
        alerts.append({
            'level': 'critical',
            'message': f'Daily budget exceeded: ${daily_cost:.2f} / ${DAILY_BUDGET:.2f}'
        })

    # Monthly check
    if monthly_cost > MONTHLY_BUDGET * 0.8:
        alerts.append({
            'level': 'warning',
            'message': f'Monthly cost at 80%: ${monthly_cost:.2f} / ${MONTHLY_BUDGET:.2f}'
        })

    return alerts
```

---

## 💡 Best Practices

### 1. Set Clear Budgets

```yaml
# In workflow configuration
env:
  DAILY_COST_LIMIT: "5.00"
  MONTHLY_COST_LIMIT: "100.00"

- name: Check budget
  run: |
    CURRENT_COST=$(calculate_current_cost)
    if (( $(echo "$CURRENT_COST > $DAILY_COST_LIMIT" | bc -l) )); then
      echo "::error::Daily budget exceeded"
      exit 1
    fi
```

### 2. Use Cost-Aware Triggering

```python
# Track costs per PR
def should_run_analysis(pr_number):
    """Decide if analysis should run based on budget."""
    daily_cost = get_today_cost()

    # Always analyze if under 50% budget
    if daily_cost < DAILY_BUDGET * 0.5:
        return True

    # Be selective if approaching limit
    if daily_cost < DAILY_BUDGET * 0.8:
        # Only analyze PRs with certain labels
        return has_label(pr_number, ['security', 'critical'])

    # Stop if over budget
    return False
```

### 3. Optimize Prompts Regularly

```bash
# A/B test prompts
prompt_v1_cost: $0.35/PR (8000 tokens)
prompt_v2_cost: $0.22/PR (5000 tokens)  # 37% savings

# Gradually migrate to more efficient prompt
```

### 4. Review Cost Reports

```bash
# Weekly review
- Which workflows cost the most?
- Which PRs were expensive to analyze?
- Any optimization opportunities?
- Are we getting value for money?
```

---

## 🎯 Cost Reduction Checklist

### Immediate Savings (Implement Today)

- [ ] Enable smart triggering (skip trivial PRs)
- [ ] Add file filtering (skip generated files)
- [ ] Set file size limits (100KB max)
- [ ] Use diff-only analysis
- [ ] Reduce max_tokens from 8000 to 4000
- [ ] Skip docs-only PRs
- [ ] Add skip-ai label option

**Expected Savings: 40-60%**

### Medium-Term Optimizations (This Week)

- [ ] Implement caching for repeated analyses
- [ ] Use Haiku for simple PRs
- [ ] Optimize system prompts
- [ ] Add incremental analysis
- [ ] Batch auto-fixes
- [ ] Set up cost tracking
- [ ] Create budget alerts

**Expected Savings: 60-80% total**

### Long-Term Improvements (This Month)

- [ ] Build cost dashboard
- [ ] Implement A/B testing for prompts
- [ ] Fine-tune model selection
- [ ] Create team cost reports
- [ ] Set up predictive budgeting
- [ ] Optimize based on analysis accuracy

**Expected Savings: 70-85% total**

---

## 📈 ROI Analysis

### Value Calculation

```
Time Saved per PR:
  Manual review: 30 min ($50 value)
  Writing tests: 45 min ($75 value)
  Security review: 20 min ($35 value)
  Documentation: 15 min ($25 value)
  Total: 110 min ($185 value)

Cost per PR: $0.35 (optimized)

ROI per PR: 528x
```

### Monthly ROI

```
20 PRs/month:
  Value: 20 × $185 = $3,700
  Cost: 20 × $0.35 = $7
  Net Value: $3,693
  ROI: 528x
```

**Even at $1/PR, ROI is still 185x**

---

## 📚 Related Documentation

- [Architecture Overview](./04-architecture-overview.md)
- [Performance Tuning](./14-performance-tuning.md)
- [Monitoring & Observability](./16-monitoring-observability.md)
- [Best Practices](./19-best-practices.md)

---

[← Back to Index](./README.md)
