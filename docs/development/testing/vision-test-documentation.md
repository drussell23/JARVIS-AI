# JARVIS Multi-Window Intelligence Testing Documentation

## Overview

This comprehensive testing suite validates all aspects of the JARVIS Multi-Window Intelligence system according to the PRD requirements. The suite includes functional, performance, and integration testing with automated reporting and success criteria evaluation.

## Test Structure

```
tests/vision/
├── test_utils.py           # Common utilities and fixtures
├── test_functional.py      # Functional testing suite
├── test_performance.py     # Performance testing suite
├── test_integration.py     # Integration testing suite
├── run_all_tests.py       # Comprehensive test runner
└── test_reports/          # Generated test reports
```

## Running Tests

### Run All Tests
```bash
cd tests/vision
python run_all_tests.py
```

### Run Individual Suites
```bash
# Functional tests only
python test_functional.py

# Performance tests only
python test_performance.py

# Integration tests only
python test_integration.py
```

## Test Categories

### 1. Functional Testing (`test_functional.py`)

Tests core functionality across all features:

- **Single Window Analysis**: Basic window understanding
- **Multi-Window Analysis (2-10 windows)**: Relationship detection
- **Edge Cases**: 20+ windows, minimized windows, full-screen apps
- **Privacy Controls**: Blacklist enforcement, privacy modes
- **Query Types**: All query categories (general, specific, historical)
- **Meeting Preparation**: Meeting detection and layout
- **Workflow Learning**: Pattern recognition and predictions

### 2. Performance Testing (`test_performance.py`)

Validates performance requirements:

- **Response Time Scaling**: Tests with 1, 5, 10, 20, 30, 50 windows
- **Resource Usage**: CPU and memory monitoring
- **Cache Effectiveness**: Hit rate and performance improvement
- **API Cost Tracking**: Cost per query type validation
- **Concurrent Requests**: Parallel query handling
- **Memory Leaks**: Extended operation testing

### 3. Integration Testing (`test_integration.py`)

Tests system integrations:

- **Claude API Integration**: Success, error handling, no-key scenarios
- **macOS API Integration**: Window detection, capture fallbacks
- **End-to-End Workflows**: Complete user journeys
- **Error Handling**: Malformed queries, edge cases
- **Permission Handling**: Screen recording permissions

## Success Criteria

The test suite validates all PRD success criteria:

| Criteria | Target | Test Coverage |
|----------|--------|---------------|
| Test Coverage | 90% | ✅ All feature areas |
| P0 Bugs | 0 | ✅ Critical path validation |
| Response Time | <3s for 95% | ✅ P95 measurement |
| API Cost | <$0.05 for 90% | ✅ Cost tracking |
| Integration | All passing | ✅ API & macOS tests |

## Test Fixtures

The `test_utils.py` provides consistent test data:

### Window Configurations
- `single_window()`: Single VS Code window
- `development_setup()`: IDE + Terminal + Browser
- `meeting_setup()`: Zoom + Calendar + Notes + Sensitive
- `communication_heavy()`: Multiple chat apps
- `edge_case_many_windows(n)`: Generate n windows
- `privacy_sensitive()`: Password managers, banking, etc.

### Query Sets
- `FUNCTIONAL_QUERIES`: Standard user queries
- `EDGE_CASE_QUERIES`: Malformed, empty, very long queries

## Performance Benchmarks

Expected performance metrics:

- **Single Window**: <100ms
- **5 Windows**: <500ms
- **20 Windows**: <1500ms
- **50 Windows**: <3000ms
- **Cache Hit**: 20-50% improvement
- **API Cost Average**: $0.02-0.03

## Test Reports

Reports are generated in `test_reports/`:

- `functional_test_report.json`: Detailed functional results
- `performance_test_report.json`: Performance metrics
- `integration_test_report.json`: Integration test results
- `consolidated_test_report.json`: Combined analysis

### Report Structure
```json
{
  "summary": {
    "total_tests": 50,
    "passed": 48,
    "failed": 2,
    "pass_rate": 0.96,
    "duration_seconds": 120.5
  },
  "by_category": {...},
  "key_metrics": {...},
  "success_criteria": {...},
  "recommendations": [...]
}
```

## Continuous Testing

### Pre-commit Testing
Run minimal test set before commits:
```bash
python test_functional.py --quick
```

### CI/CD Integration
The test runner returns exit codes:
- 0: All tests passed, ready for launch
- 1: Some tests failed, not ready

### Performance Monitoring
Track performance over time:
```python
# Compare reports
python compare_reports.py report1.json report2.json
```

## Debugging Failed Tests

### Enable Verbose Output
```python
# In test files, set logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Run Single Test
```python
# In test file
async def run_single():
    suite = FunctionalTestSuite()
    await suite.test_single_window_analysis()
    
asyncio.run(run_single())
```

### Mock Troubleshooting
- Ensure mock windows have all required fields
- Check mock API responses match expected format
- Verify permission mocks return appropriate values

## Adding New Tests

1. Add test method to appropriate suite class
2. Use consistent naming: `test_feature_scenario()`
3. Return `TestResult` object with metrics
4. Update fixtures if new scenarios needed
5. Document expected behavior

Example:
```python
async def test_new_feature(self):
    """Test new feature functionality"""
    with TestTimer() as timer:
        try:
            # Test implementation
            result = await self.jarvis.new_feature()
            
            passed = validate_result(result)
            
            self.results.append(TestResult(
                test_name="new_feature_basic",
                category="functional",
                passed=passed,
                duration_ms=timer.duration_ms,
                metrics={"custom_metric": value}
            ))
        except Exception as e:
            # Handle errors
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Use mocks for external dependencies
3. **Timing**: Always measure performance
4. **Metrics**: Include relevant metrics in results
5. **Error Handling**: Catch and report all exceptions
6. **Documentation**: Comment complex test logic

## Maintenance

### Regular Tasks
- Update fixtures when UI changes
- Add tests for new features
- Review and update performance thresholds
- Clean old test reports
- Update mock data for API changes

### Version Compatibility
Tests are compatible with:
- Python 3.8+
- macOS 10.15+
- JARVIS v3.7.0+

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure backend is in Python path
2. **Permission Errors**: Run with appropriate permissions
3. **Mock Failures**: Check mock data matches current API
4. **Timeout Errors**: Increase timeout for slow systems
5. **Memory Errors**: Run tests individually if limited RAM

### Getting Help

- Check test output for specific error messages
- Review test reports for patterns
- Enable debug logging for detailed traces
- Run individual tests to isolate issues