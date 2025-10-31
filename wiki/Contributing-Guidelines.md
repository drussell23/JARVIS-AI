# Contributing Guidelines

How to contribute to JARVIS AI Agent development.

---

## Welcome Contributors!

We appreciate your interest in contributing to JARVIS. This guide will help you get started.

---

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

---

## How to Contribute

### 1. Fork & Clone

```bash
# Fork repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/JARVIS-AI-Agent.git
cd JARVIS-AI-Agent

# Add upstream remote
git remote add upstream https://github.com/derekjrussell/JARVIS-AI-Agent.git
```

### 2. Create Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

**Development Setup:**
```bash
# Install development dependencies
pip install -r backend/requirements.txt
pip install -r backend/requirements-optional.txt

# Install pre-commit hooks
pre-commit install

# Run tests
cd backend
pytest tests/
```

### 4. Test Your Changes

**Required:**
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Code quality
black backend/
flake8 backend/
pylint backend/
```

### 5. Commit Changes

**Use Conventional Commits:**
```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(voice): add ECAPA-TDNN speaker recognition"
git commit -m "fix(database): resolve Cloud SQL connection timeout"
git commit -m "docs(wiki): add API documentation"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Add/update tests
- `chore`: Maintenance

### 6. Push & Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request
gh pr create --fill
```

---

## Pull Request Guidelines

### PR Title

Use Conventional Commits format:
```
feat(voice): Add SpeechBrain STT engine
fix(gcp): Resolve VM auto-scaling issue
docs(setup): Update installation guide
```

### PR Description

Include:
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### PR Size

**Preferred sizes:**
- XS: <10 lines
- S: 10-99 lines
- M: 100-499 lines
- L: 500-999 lines
- XL: 1000+ lines (split if possible)

---

## Code Style

### Python

**Follow PEP 8 with Black formatting:**
```bash
# Format code
black backend/

# Sort imports
isort backend/

# Lint
flake8 backend/
pylint backend/
```

**Type hints required:**
```python
def process_command(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process voice command with context."""
    pass
```

**Docstrings required:**
```python
def analyze_screen(prompt: str) -> Dict[str, Any]:
    """
    Analyze screen content with Claude Vision API.

    Args:
        prompt: Analysis prompt

    Returns:
        Dict containing analysis results

    Raises:
        VisionAPIError: If analysis fails
    """
    pass
```

### JavaScript/TypeScript

**Follow ESLint rules:**
```javascript
// Use const/let, not var
const apiUrl = 'http://localhost:8010';

// Arrow functions for callbacks
ws.onmessage = (event) => {
  handleMessage(JSON.parse(event.data));
};

// Async/await, not callbacks
async function fetchData() {
  const response = await fetch(apiUrl);
  return response.json();
}
```

---

## Testing Requirements

### Unit Tests

**Required for:**
- New features
- Bug fixes
- Refactored code

**Example:**
```python
import pytest
from intelligence.cai import ContextAwarenessIntelligence

def test_intent_prediction():
    cai = ContextAwarenessIntelligence()
    result = cai.predict_intent("unlock my screen")

    assert result['intent'] == 'screen_unlock'
    assert result['confidence'] > 0.7
    assert result['requires_auth'] is True
```

### Integration Tests

**Required for:**
- API endpoints
- Database operations
- External service integrations

### Test Coverage

**Minimum:** 70% overall
**Target:** 85%

```bash
# Check coverage
pytest --cov=backend tests/
```

---

## Documentation

### Update Documentation

**When adding features:**
1. Update relevant Wiki page
2. Add docstrings to code
3. Update API documentation if needed
4. Add examples to README

**When fixing bugs:**
1. Document fix in PR description
2. Add to CHANGELOG.md
3. Update troubleshooting guide if applicable

---

## Review Process

### What We Look For

**Code Quality:**
- Follows style guidelines
- Well-structured and readable
- Properly tested
- No unnecessary complexity

**Functionality:**
- Solves stated problem
- No regressions
- Edge cases handled
- Error handling implemented

**Documentation:**
- Clear commit messages
- Updated documentation
- Code comments where needed
- PR description complete

### Review Timeline

- **Initial Review:** Within 48 hours
- **Follow-up:** Within 24 hours
- **Approval:** Minimum 1 reviewer

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "fix: address PR feedback"
git push origin feature/your-feature-name
```

---

## Issue Guidelines

### Reporting Bugs

**Include:**
1. Clear description
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details
5. Logs/screenshots

**Template:**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Start JARVIS
2. Execute command "..."
3. Observe error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: macOS 13.5 (M1)
- Python: 3.10.12
- JARVIS: v17.4.0

## Logs
```
Paste relevant logs
```
```

### Feature Requests

**Include:**
1. Use case/problem to solve
2. Proposed solution
3. Alternatives considered
4. Additional context

---

## Development Tips

### Local Testing

```bash
# Start JARVIS in dev mode
DEBUG=true python start_system.py --dev

# Watch logs
tail -f jarvis_startup.log

# Test API endpoints
curl http://localhost:8010/health
```

### Debugging

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug info")
logger.info("Info message")
logger.error("Error occurred")
```

### Performance

- Profile code before optimization
- Use async/await for I/O operations
- Cache expensive operations
- Monitor memory usage

---

## Community

### Getting Help

1. Check existing documentation
2. Search GitHub issues
3. Ask in discussions
4. Create detailed issue

### Staying Updated

```bash
# Sync your fork
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

**Thank you for contributing to JARVIS!**

---

**Last Updated:** 2025-10-30
