# JARVIS Testing Guide

Comprehensive guide to testing JARVIS with enhanced tools and strategies.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Test Types](#test-types)
3. [Pytest Plugins](#pytest-plugins)
4. [Property-Based Testing](#property-based-testing)
5. [Pre-Commit Hooks](#pre-commit-hooks)
6. [CI/CD Integration](#cicd-integration)
7. [Best Practices](#best-practices)

---

## Quick Start

### Running Tests

```bash
# Run all tests
cd backend
pytest

# Run specific test file
pytest tests/test_hypothesis_examples.py

# Run with specific markers
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Exclude slow tests

# Run tests in parallel (using pytest-xdist)
pytest -n auto          # Auto-detect CPU cores
pytest -n 4             # Use 4 workers

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_hypothesis_examples.py::test_string_round_trip
```

### Pre-Commit Hooks

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on staged files (automatic on git commit)
git commit -m "Your message"
```

---

## Test Types

### Unit Tests
Fast, isolated tests for individual functions/classes.

```python
import pytest

@pytest.mark.unit
def test_confidence_calculation():
    """Test confidence score calculation"""
    result = calculate_confidence([0.8, 0.9, 0.7])
    assert 0.7 <= result <= 0.9
```

### Integration Tests
Tests that involve multiple components or external services.

```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_integration():
    """Test database operations"""
    async with get_database() as db:
        await db.insert_goal("test goal")
        goals = await db.get_goals()
        assert len(goals) > 0
```

### Property-Based Tests
Tests using Hypothesis to automatically generate test cases.

```python
from hypothesis import given, strategies as st

@given(st.text())
def test_string_processing(text):
    """Test string processing with auto-generated inputs"""
    result = process_string(text)
    assert isinstance(result, str)
```

---

## Pytest Plugins

### 1. pytest-xdist (Parallel Execution)
```bash
# Automatically uses all CPU cores
pytest -n auto

# Manual worker count
pytest -n 4
```

### 2. pytest-sugar (Better Output)
Automatically enabled - provides better test output formatting.

### 3. pytest-clarity (Better Diffs)
Automatically enabled - shows clearer assertion diffs.

### 4. pytest-mock (Mocking)
```python
def test_with_mock(mocker):
    """Test with mocked dependencies"""
    mock_db = mocker.patch('module.database')
    mock_db.return_value = {'result': 'mocked'}

    result = function_using_db()
    assert result == 'mocked'
```

### 5. pytest-timeout (Prevent Hangs)
```python
@pytest.mark.timeout(5)  # 5 second timeout
def test_slow_operation():
    """Test that should complete quickly"""
    result = slow_operation()
    assert result
```

### 6. pytest-cov (Coverage)
```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# Open report
open htmlcov/index.html
```

---

## Property-Based Testing

### What is Property-Based Testing?

Instead of writing specific test cases, you write **properties** that should always be true, and Hypothesis automatically generates hundreds of test cases to verify them.

### Basic Example

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    """Property: addition is commutative"""
    assert a + b == b + a
```

### JARVIS-Specific Examples

#### Testing Goal Patterns
```python
from hypothesis import given, strategies as st

@given(
    st.text(min_size=1, max_size=500),
    st.floats(min_value=0.0, max_value=1.0)
)
def test_goal_pattern_properties(goal_text, confidence):
    """Test goal pattern invariants"""
    pattern = create_goal_pattern(goal_text, confidence)

    # Properties that should always hold
    assert pattern['confidence'] >= 0.0
    assert pattern['confidence'] <= 1.0
    assert len(pattern['goal_text']) > 0
```

#### Custom Strategies
```python
@st.composite
def goal_strategy(draw):
    """Generate realistic goal patterns"""
    return {
        'goal_type': draw(st.sampled_from(['open_app', 'query', 'action'])),
        'confidence': draw(st.floats(min_value=0.5, max_value=1.0)),
        'context': draw(st.lists(st.text(), max_size=5))
    }

@given(goal_strategy())
def test_goal_processing(goal):
    """Test goal processing with realistic data"""
    result = process_goal(goal)
    assert result['processed'] is True
```

#### Stateful Testing
```python
from hypothesis.stateful import RuleBasedStateMachine, rule

class DatabaseStateMachine(RuleBasedStateMachine):
    """Test database maintains consistency across operations"""

    def __init__(self):
        super().__init__()
        self.db = InMemoryDB()

    @rule(key=st.text(), value=st.integers())
    def insert(self, key, value):
        self.db.insert(key, value)

    @invariant()
    def database_consistent(self):
        """Database should always be consistent"""
        assert self.db.check_consistency()
```

---

## Pre-Commit Hooks

Automatically run code quality checks before each commit.

### Installed Hooks

1. **Black** - Code formatting
2. **isort** - Import sorting
3. **flake8** - Linting
4. **bandit** - Security checks
5. **mypy** - Type checking
6. **interrogate** - Docstring coverage
7. **autoflake** - Remove unused imports

### Configuration

Edit `.pre-commit-config.yaml` to customize:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        args: ['--line-length=100']
```

### Skipping Hooks (Emergency Only)

```bash
# Skip all hooks
git commit --no-verify -m "Emergency fix"

# Skip specific hook
SKIP=black git commit -m "Skip black formatting"
```

---

## CI/CD Integration

### GitHub Actions

Tests run automatically on push/PR. See `.github/workflows/test.yml`:

```yaml
- name: Run Tests
  run: |
    cd backend
    pytest tests/ \
      --cov=. \
      --cov-report=xml \
      -n auto \
      -v
```

### Test Markers in CI

```yaml
# Run only fast tests
pytest -m "unit and not slow"

# Run integration tests separately
pytest -m integration
```

---

## Best Practices

### 1. Use Markers

```python
@pytest.mark.unit
@pytest.mark.fast
def test_simple_function():
    """Quick unit test"""
    assert add(1, 2) == 3

@pytest.mark.integration
@pytest.mark.db
@pytest.mark.slow
async def test_database_operation():
    """Slow database test"""
    result = await complex_db_operation()
    assert result
```

### 2. Write Properties, Not Examples

**Bad (Example-based):**
```python
def test_confidence():
    assert calculate_confidence(0.8, 0.9) == 0.85
```

**Good (Property-based):**
```python
@given(st.floats(0, 1), st.floats(0, 1))
def test_confidence_properties(a, b):
    result = calculate_confidence(a, b)
    assert 0 <= result <= 1  # Property: always in range
    assert result >= min(a, b)  # Property: at least minimum
```

### 3. Test Invariants

```python
@given(st.lists(st.integers()))
def test_sort_invariants(lst):
    """Test sorting preserves invariants"""
    sorted_lst = sorted(lst)

    # Invariant 1: Same length
    assert len(sorted_lst) == len(lst)

    # Invariant 2: Same elements
    assert sorted(lst) == sorted_lst

    # Invariant 3: Ordered
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]
```

### 4. Use Fixtures for Setup

```python
@pytest.fixture
async def database():
    """Provide clean database for each test"""
    db = await create_test_database()
    yield db
    await db.cleanup()

async def test_with_database(database):
    """Test uses database fixture"""
    await database.insert("test")
    result = await database.query("test")
    assert result
```

### 5. Mock External Services

```python
def test_api_call(mocker):
    """Test API without making real calls"""
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'status': 'success'}

    mocker.patch('requests.get', return_value=mock_response)

    result = make_api_call()
    assert result['status'] == 'success'
```

### 6. Test Edge Cases

```python
@given(st.integers())
@example(0)           # Edge: zero
@example(-1)          # Edge: negative
@example(2**31 - 1)   # Edge: max int
@example(-2**31)      # Edge: min int
def test_integer_processing(n):
    """Test with edge cases"""
    result = process_integer(n)
    assert isinstance(result, int)
```

---

## Coverage Goals

- **Unit tests**: 90%+ coverage
- **Integration tests**: 70%+ coverage
- **Overall**: 80%+ coverage

```bash
# Check coverage
pytest --cov=. --cov-report=term-missing

# View detailed HTML report
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

### Tests Hanging
```bash
# Add timeout to all tests
pytest --timeout=30
```

### Parallel Test Failures
```bash
# Disable parallel execution
pytest -n 0
```

### Import Errors
```bash
# Ensure you're in backend directory
cd backend
pytest
```

### Coverage Not Working
```bash
# Reinstall coverage plugin
pip install --upgrade pytest-cov
```

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Happy Testing! 🧪**
