# 🔍 PR Analyzer Deep Dive - Claude AI Workflows

**Complete technical reference for the Claude AI PR Analyzer workflow**

---

## 📋 Overview

The PR Analyzer is the cornerstone of the Claude AI workflow system. It provides comprehensive, context-aware code review that goes far beyond traditional linters or static analysis tools.

**Key Capabilities:**
- 🧠 Understands code architecture and design patterns
- 🔒 Identifies security vulnerabilities with explanations
- ⚡ Suggests performance optimizations
- 📚 Reviews code clarity and maintainability
- 🧪 Identifies testing gaps
- 🏗️ Analyzes architectural decisions

---

## 🎯 How It Works

### High-Level Flow

```
PR Created/Updated
       ↓
Workflow Triggers
       ↓
Fetch PR Changes (git diff)
       ↓
Build Context (files, comments, metadata)
       ↓
Send to Claude Sonnet 4 API
       ↓
Receive Analysis
       ↓
Parse Response
       ↓
Post PR Comment
       ↓
Apply Labels
       ↓
Update PR Status
```

### Detailed Workflow

#### 1. Trigger Conditions

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
  issue_comment:
    types: [created]
```

**Triggers:**
- PR opened
- New commits pushed
- PR reopened after being closed
- Comment containing `@claude`

**Smart Skipping:**
- Skips if PR is from Dependabot (has separate workflow)
- Skips if PR is draft (unless requested)
- Skips if changes are documentation-only (configurable)

#### 2. Context Building

The workflow collects extensive context:

```python
context = {
    'pr_title': pr.title,
    'pr_description': pr.body,
    'pr_author': pr.user.login,
    'base_branch': pr.base.ref,
    'head_branch': pr.head.ref,
    'files_changed': len(files),
    'additions': pr.additions,
    'deletions': pr.deletions,
    'changed_files': [
        {
            'filename': file.filename,
            'status': file.status,  # added, modified, removed
            'additions': file.additions,
            'deletions': file.deletions,
            'changes': file.changes,
            'patch': file.patch,  # actual diff
            'previous_content': get_file_content(base_sha, file.filename),
            'current_content': get_file_content(head_sha, file.filename)
        }
        for file in files
    ],
    'pr_labels': [label.name for label in pr.labels],
    'existing_comments': [comment.body for comment in pr.get_comments()],
    'pr_commits': [
        {
            'sha': commit.sha,
            'message': commit.commit.message,
            'author': commit.commit.author.name
        }
        for commit in pr.get_commits()
    ]
}
```

**Context Optimizations:**

- **File Filtering:** Only includes files that need review (skips generated files, binaries)
- **Content Truncation:** Large files are intelligently truncated
- **Diff Focus:** Emphasizes changed lines rather than entire files
- **Token Management:** Stays within Claude's context window (200k tokens)

#### 3. System Prompt

The system prompt defines Claude's role and capabilities:

```python
system_prompt = """You are an expert code reviewer with deep knowledge of:
- Software architecture and design patterns
- Security best practices (OWASP Top 10, CWE)
- Performance optimization
- Code maintainability and readability
- Testing strategies
- Language-specific idioms and best practices

Your code reviews should be:
1. **Comprehensive** - Cover all aspects of code quality
2. **Actionable** - Provide specific, implementable suggestions
3. **Constructive** - Focus on improvement, not criticism
4. **Context-aware** - Consider the PR's purpose and scope
5. **Security-focused** - Prioritize security issues
6. **Performance-conscious** - Identify optimization opportunities

When reviewing code:
- Explain WHY issues matter, not just WHAT is wrong
- Provide code examples for suggested fixes
- Prioritize issues (Critical → Major → Minor)
- Consider the broader codebase architecture
- Identify testing gaps
- Suggest documentation improvements

Rate the PR from 1-10:
- 1-3: Needs significant rework
- 4-6: Needs improvements
- 7-8: Good, minor issues
- 9-10: Excellent

Use this format:
## 🤖 Claude AI Code Review

### Overall Score: X/10

[Brief summary]

### 🚨 Critical Issues
[Must-fix security, data loss, or breaking issues]

### ⚠️  Major Issues
[Important bugs, performance issues, design flaws]

### 💡 Improvements
[Nice-to-have enhancements]

### 🧪 Testing Gaps
[Missing tests or test scenarios]

### 📊 Metrics
- Lines Added: X
- Files Changed: X
- Complexity: [Low/Medium/High]
- Security Issues: X
- Code Coverage Estimate: X%

### 🎯 Recommendations
[Prioritized action items]

### 🏷️  Suggested Labels
[Relevant labels]
"""
```

#### 4. API Call

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    temperature=0,  # Deterministic for code review
    system=system_prompt,
    messages=[{
        "role": "user",
        "content": f"""Please review this pull request:

**Title:** {context['pr_title']}

**Description:**
{context['pr_description']}

**Changes:**
{format_changes(context['changed_files'])}

**Additional Context:**
- Base Branch: {context['base_branch']}
- Files Changed: {context['files_changed']}
- Additions: {context['additions']}
- Deletions: {context['deletions']}

Please provide a comprehensive code review."""
    }]
)

analysis = message.content[0].text
```

**API Parameters:**

- `model`: `claude-sonnet-4-20250514` - Latest Sonnet model
- `max_tokens`: `8000` - Sufficient for detailed analysis
- `temperature`: `0` - Deterministic output for consistency
- `system`: Defines Claude's expertise and output format
- `messages`: Contains PR context and request

#### 5. Response Parsing

```python
def parse_analysis(analysis_text):
    """Extract structured data from Claude's response."""

    # Extract score
    score_match = re.search(r'Overall Score: (\d+)/10', analysis_text)
    score = int(score_match.group(1)) if score_match else 5

    # Extract sections
    sections = {
        'critical': extract_section(analysis_text, 'Critical Issues'),
        'major': extract_section(analysis_text, 'Major Issues'),
        'improvements': extract_section(analysis_text, 'Improvements'),
        'testing': extract_section(analysis_text, 'Testing Gaps'),
        'metrics': extract_section(analysis_text, 'Metrics'),
        'recommendations': extract_section(analysis_text, 'Recommendations')
    }

    # Extract suggested labels
    labels_section = extract_section(analysis_text, 'Suggested Labels')
    suggested_labels = parse_labels(labels_section)

    return {
        'score': score,
        'analysis': analysis_text,
        'sections': sections,
        'labels': suggested_labels
    }
```

#### 6. Posting Results

```python
# Post comment
pr.create_issue_comment(parsed['analysis'])

# Apply labels
current_labels = [label.name for label in pr.labels]
new_labels = [
    label for label in parsed['labels']
    if label not in current_labels
]
if new_labels:
    pr.add_to_labels(*new_labels)

# Update check status
if parsed['score'] >= 7:
    conclusion = 'success'
elif parsed['score'] >= 5:
    conclusion = 'neutral'
else:
    conclusion = 'failure'

github.create_check_run(
    name='Claude AI PR Analyzer',
    conclusion=conclusion,
    output={
        'title': f'Code Review Score: {parsed["score"]}/10',
        'summary': parsed['analysis']
    }
)
```

---

## 📊 Analysis Examples

### Example 1: Security Vulnerability

**Code Reviewed:**
```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
```

**Claude's Analysis:**

```markdown
### 🚨 Critical Issues

#### SQL Injection Vulnerability (CWE-89)
**File:** `api/users.py:45`
**Severity:** CRITICAL

**Issue:**
The `get_user` function uses string interpolation to build SQL queries, creating
a SQL injection vulnerability. An attacker could provide malicious input like:
`1 OR 1=1; DROP TABLE users;--`

**Impact:**
- Complete database compromise
- Data theft
- Data destruction
- Unauthorized access

**Fix:**
```python
def get_user(user_id: int) -> dict:
    """
    Retrieve user by ID using parameterized query.

    Args:
        user_id: Unique user identifier

    Returns:
        User data dictionary

    Raises:
        ValueError: If user_id is not a valid integer
    """
    # Validate input
    if not isinstance(user_id, int) or user_id < 1:
        raise ValueError("Invalid user ID")

    # Use parameterized query
    query = "SELECT * FROM users WHERE id = %s"
    return db.execute(query, (user_id,))
```

**Testing:**
Add tests to verify injection prevention:
```python
def test_get_user_sql_injection_prevention():
    """Test that SQL injection attempts are prevented."""
    malicious_input = "1 OR 1=1; DROP TABLE users;--"
    with pytest.raises(ValueError):
        get_user(malicious_input)
```

**References:**
- OWASP: https://owasp.org/www-community/attacks/SQL_Injection
- CWE-89: https://cwe.mitre.org/data/definitions/89.html
```

### Example 2: Performance Issue

**Code Reviewed:**
```python
def process_users(users):
    result = []
    for user in users:
        if user['active']:
            result.append(user['name'].upper())
    return result
```

**Claude's Analysis:**

```markdown
### 💡 Improvements

#### Inefficient List Building
**File:** `utils/processors.py:12`
**Severity:** MINOR
**Impact:** Performance

**Current Implementation:**
The function uses a loop with repeated `append()` calls, which is less
efficient than list comprehension in Python.

**Benchmarks:**
- Current approach: ~100μs for 1000 users
- List comprehension: ~60μs for 1000 users
- **40% performance improvement**

**Recommended:**
```python
def process_users(users: list[dict]) -> list[str]:
    """
    Extract uppercase names of active users.

    Args:
        users: List of user dictionaries with 'active' and 'name' keys

    Returns:
        List of uppercase user names

    Example:
        >>> users = [{'active': True, 'name': 'john'}, {'active': False, 'name': 'jane'}]
        >>> process_users(users)
        ['JOHN']
    """
    return [
        user['name'].upper()
        for user in users
        if user.get('active', False)
    ]
```

**Additional Improvements:**
1. Added type hints for clarity
2. Used `.get()` for safe dictionary access
3. Added docstring with example
4. More Pythonic and readable

**When to Use Loop vs Comprehension:**
- Use comprehension: Simple transformations, filtering
- Use loop: Complex logic, multiple operations, error handling
```

### Example 3: Architecture Issue

**Code Reviewed:**
```python
class UserService:
    def __init__(self):
        self.db = PostgresDB()
        self.cache = RedisCache()
        self.logger = Logger()

    def get_user(self, user_id):
        # Check cache
        cached = self.cache.get(f'user:{user_id}')
        if cached:
            return cached

        # Query database
        user = self.db.query('SELECT * FROM users WHERE id = %s', user_id)

        # Update cache
        self.cache.set(f'user:{user_id}', user, ttl=3600)

        return user
```

**Claude's Analysis:**

```markdown
### ⚠️  Major Issues

#### Tight Coupling and Dependency Injection
**File:** `services/user_service.py:5-25`
**Severity:** MAJOR
**Impact:** Testability, Maintainability

**Issue:**
The `UserService` class directly instantiates its dependencies (`PostgresDB`,
`RedisCache`, `Logger`), making it:
- Hard to test (can't mock dependencies)
- Tightly coupled to specific implementations
- Violates Dependency Inversion Principle
- Difficult to configure different environments

**Current Problems:**
```python
# Testing is difficult
def test_get_user():
    service = UserService()  # Creates real DB connection!
    # How do we mock the database?
```

**Recommended: Dependency Injection**
```python
from typing import Protocol

class Database(Protocol):
    """Database interface."""
    def query(self, sql: str, *params) -> dict:
        ...

class Cache(Protocol):
    """Cache interface."""
    def get(self, key: str) -> dict | None:
        ...
    def set(self, key: str, value: dict, ttl: int) -> None:
        ...

class UserService:
    """
    Service for user operations with dependency injection.

    Args:
        db: Database implementation
        cache: Cache implementation
        logger: Logger implementation
    """

    def __init__(
        self,
        db: Database,
        cache: Cache,
        logger: logging.Logger
    ):
        self.db = db
        self.cache = cache
        self.logger = logger

    def get_user(self, user_id: int) -> dict:
        """Retrieve user with caching."""
        cache_key = f'user:{user_id}'

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.info(f'Cache hit for user {user_id}')
            return cached

        # Query database
        self.logger.info(f'Cache miss for user {user_id}')
        user = self.db.query(
            'SELECT * FROM users WHERE id = %s',
            user_id
        )

        if not user:
            raise UserNotFoundError(f'User {user_id} not found')

        # Update cache
        self.cache.set(cache_key, user, ttl=3600)

        return user

# Usage
def create_user_service():
    """Factory function for production."""
    return UserService(
        db=PostgresDB(config.db_url),
        cache=RedisCache(config.redis_url),
        logger=logging.getLogger(__name__)
    )

# Testing becomes easy
def test_get_user():
    """Test with mocked dependencies."""
    mock_db = Mock(spec=Database)
    mock_cache = Mock(spec=Cache)
    mock_logger = Mock(spec=logging.Logger)

    mock_cache.get.return_value = None
    mock_db.query.return_value = {'id': 1, 'name': 'John'}

    service = UserService(mock_db, mock_cache, mock_logger)
    user = service.get_user(1)

    assert user['name'] == 'John'
    mock_db.query.assert_called_once()
    mock_cache.set.assert_called_once()
```

**Benefits:**
- ✅ Easy to test with mocks
- ✅ Flexible (swap implementations)
- ✅ Follows SOLID principles
- ✅ Better error handling
- ✅ Environment-agnostic

**Migration Path:**
1. Create Protocol interfaces
2. Update UserService constructor
3. Create factory function
4. Update all instantiation sites
5. Add tests with mocks
```

---

## 🎛️  Configuration

### Workflow File

`.github/workflows/claude-pr-analyzer.yml`

Key configuration options:

```yaml
env:
  # Model selection
  CLAUDE_MODEL: "claude-sonnet-4-20250514"

  # Token limits
  MAX_TOKENS: 8000  # Increase for more detailed analysis

  # Temperature (0 = deterministic, 1 = creative)
  TEMPERATURE: 0

  # File size limits (bytes)
  MAX_FILE_SIZE: 100000

  # Context limits
  MAX_FILES_TO_ANALYZE: 50

  # Skip patterns
  SKIP_FILES: "*.lock,*.min.js,dist/*,build/*"

  # Auto-labeling
  ENABLE_AUTO_LABELS: true

  # Minimum score for success
  PASSING_SCORE: 7
```

### Customizing the System Prompt

Edit the workflow to customize Claude's focus:

```yaml
- name: Analyze PR
  env:
    CUSTOM_INSTRUCTIONS: |
      Additional focus areas:
      - Follow our internal Python style guide
      - Ensure all public APIs have examples
      - Check for proper logging
      - Verify error messages are user-friendly
      - Ensure database migrations are included
```

### Skip Conditions

```yaml
jobs:
  analyze:
    if: |
      github.event_name != 'pull_request' ||
      (
        !contains(github.event.pull_request.labels.*.name, 'skip-ai') &&
        github.event.pull_request.draft == false &&
        github.event.pull_request.user.login != 'dependabot[bot]'
      )
```

**Add `skip-ai` label to PR** to prevent analysis.

---

## 📈 Performance & Costs

### Typical Performance

| PR Size | Analysis Time | Token Usage | Cost |
|---------|---------------|-------------|------|
| Small (1-5 files) | 30-60s | 2,000 in / 1,500 out | $0.10 |
| Medium (6-20 files) | 1-2 min | 10,000 in / 3,000 out | $0.35 |
| Large (21-50 files) | 2-4 min | 25,000 in / 5,000 out | $0.90 |
| Very Large (50+ files) | 4-8 min | 50,000 in / 7,000 out | $1.80 |

**Pricing (Claude Sonnet 4):**
- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens

### Optimization Strategies

#### 1. Smart File Filtering

```python
# Skip files that don't need review
SKIP_PATTERNS = [
    '*.lock',      # Lock files
    '*.min.js',    # Minified JavaScript
    '*.min.css',   # Minified CSS
    'dist/*',      # Distribution builds
    'build/*',     # Build artifacts
    'node_modules/*',  # Dependencies
    '__pycache__/*',   # Python cache
    '*.pyc',       # Compiled Python
    'venv/*',      # Virtual environments
]
```

#### 2. File Size Limits

```python
# Skip very large files
MAX_FILE_SIZE = 100_000  # 100KB

if file.size > MAX_FILE_SIZE:
    logger.info(f'Skipping {file.filename} (too large: {file.size} bytes)')
    continue
```

#### 3. Diff-Only Analysis

For large files, send only the diff:

```python
# Instead of full file content
content = file.patch  # Just the changes
```

#### 4. Conditional Triggers

```yaml
# Only run on specific paths
on:
  pull_request:
    paths:
      - 'src/**'
      - 'lib/**'
      - '!docs/**'  # Exclude docs
```

#### 5. Batch Small PRs

For very small PRs (1-2 line changes), skip AI:

```yaml
if: github.event.pull_request.additions + github.event.pull_request.deletions > 10
```

---

## 🔧 Troubleshooting

### Issue: Analysis Too Generic

**Symptom:** Claude provides surface-level feedback

**Cause:** Insufficient context or too broad system prompt

**Fix:**
```python
# Add more context
context['codebase_info'] = {
    'primary_language': 'Python',
    'framework': 'FastAPI',
    'database': 'PostgreSQL',
    'style_guide': 'PEP 8 + Black',
    'testing_framework': 'pytest'
}
```

### Issue: Missing Critical Issues

**Symptom:** Claude doesn't catch a security issue you expected

**Cause:** Issue not in changed files or insufficient emphasis

**Fix:**
```python
# Emphasize security in system prompt
system_prompt += """
CRITICAL: Prioritize security analysis above all else.
Specifically check for:
- SQL injection
- XSS vulnerabilities
- Authentication bypass
- Authorization issues
- Cryptographic weaknesses
- Input validation gaps
"""
```

### Issue: Too Verbose

**Symptom:** Analysis is overwhelming or too long

**Cause:** High max_tokens or verbose system prompt

**Fix:**
```yaml
env:
  MAX_TOKENS: 4000  # Reduce from 8000

  CONCISE_MODE: |
    Keep responses concise:
    - List only high/critical issues
    - Limit to top 5 improvements
    - Skip obvious fixes
```

### Issue: Inconsistent Scoring

**Symptom:** Similar PRs get different scores

**Cause:** Temperature > 0 or inconsistent scoring criteria

**Fix:**
```python
# Ensure deterministic behavior
temperature = 0

# Add clear scoring rubric
system_prompt += """
Scoring Rubric:
10: Perfect code, no issues
9:  1-2 minor improvements suggested
8:  3-5 minor improvements needed
7:  1-2 major issues or 5+ minor
6:  3-4 major issues
5:  5+ major issues or 1 critical
4:  Multiple critical issues
3:  Severe security or architectural problems
2:  Code doesn't work or has critical flaws
1:  Unacceptable quality
"""
```

---

## 🧪 Testing the Analyzer

### Test Workflow Locally

```bash
# Install dependencies
pip install anthropic PyGithub

# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
export PR_NUMBER=123

# Run analyzer script
python .github/scripts/claude_pr_analyzer.py
```

### Create Test Cases

```python
# test_pr_analyzer.py
import pytest
from claude_pr_analyzer import analyze_pr, parse_analysis

def test_security_issue_detection():
    """Test that SQL injection is detected."""
    code = '''
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return db.execute(query)
    '''

    result = analyze_pr({'changed_files': [{'patch': code}]})

    assert 'SQL injection' in result['analysis'].lower()
    assert result['score'] < 5
    assert 'security-critical' in result['labels']

def test_scoring_consistency():
    """Test that scoring is deterministic."""
    context = load_test_pr_context('fixtures/test_pr.json')

    result1 = analyze_pr(context)
    result2 = analyze_pr(context)

    assert result1['score'] == result2['score']
    assert result1['analysis'] == result2['analysis']
```

---

## 📚 Best Practices

### 1. Provide Rich Context

```python
# Good: Detailed context
context = {
    'pr_title': title,
    'pr_description': description,
    'project_info': {
        'type': 'REST API',
        'language': 'Python 3.11',
        'framework': 'FastAPI',
        'database': 'PostgreSQL + Redis'
    },
    'team_standards': {
        'style_guide': 'PEP 8 + Black',
        'test_coverage_required': '80%',
        'security_standards': 'OWASP ASVS Level 2'
    }
}

# Bad: Minimal context
context = {'files': files}
```

### 2. Use Consistent Prompts

```python
# Version your system prompts
SYSTEM_PROMPT_VERSION = "2.1.0"

# Track prompt changes in git
# .github/prompts/pr_analyzer_v2.1.0.txt
```

### 3. Monitor Costs

```python
# Track token usage
def log_analysis_metrics(response):
    logger.info({
        'input_tokens': response.usage.input_tokens,
        'output_tokens': response.usage.output_tokens,
        'cost': calculate_cost(response.usage),
        'pr_number': pr.number
    })
```

### 4. Iterate on Feedback

```python
# Collect feedback on analysis quality
def collect_feedback(pr_number, analysis_id):
    """Allow developers to rate AI analysis."""
    # Add reaction buttons to comment
    # Track which analyses were helpful
```

---

## 🎯 Advanced Usage

### Custom Analysis Types

Request specific analysis via PR labels:

```python
# Security-focused analysis
if 'security-review' in pr_labels:
    system_prompt += "\n\nFOCUS: Deep security analysis. Check for all OWASP Top 10 vulnerabilities."

# Performance-focused analysis
if 'performance' in pr_labels:
    system_prompt += "\n\nFOCUS: Performance optimization. Identify bottlenecks and suggest improvements."

# Architecture review
if 'architecture-review' in pr_labels:
    system_prompt += "\n\nFOCUS: Architectural analysis. Evaluate design patterns, SOLID principles, and scalability."
```

### Multi-Model Comparison

Compare analyses from different models:

```python
models = [
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250514',
    'claude-haiku-4-20250514'
]

analyses = {}
for model in models:
    analyses[model] = analyze_with_model(context, model)

# Post comparison
post_comparison_comment(pr, analyses)
```

---

## 📖 Related Documentation

- [Architecture Overview](./04-architecture-overview.md)
- [Auto-Fix Deep Dive](./08-auto-fix-deep-dive.md)
- [Best Practices](./19-best-practices.md)
- [Cost Optimization](./13-cost-optimization.md)

---

[← Back to Index](./README.md) | [Next: Auto-Fix Deep Dive →](./08-auto-fix-deep-dive.md)
