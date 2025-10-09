# JARVIS Vision Intelligence Testing Guide

## Overview

This guide explains the comprehensive test suite for JARVIS's intelligent vision system, which has been transformed from a hardcoded pattern-matching system to a dynamic, intelligent vision-based system.

## Test Categories

### 1. Functional Tests (`test_intelligent_vision.py`)
Tests core functionality without requiring actual window detection.

**Key Tests:**
- **Unknown App Detection**: Verifies queries about apps JARVIS has never seen
- **Query Pattern Detection**: Tests intelligent query routing
- **Multi-Language Support**: Tests non-English app names
- **Flexible Query Handling**: Tests various ways users might ask questions

**Example:**
```python
# Tests that "do i have any notifications from SuperNewChatApp" 
# is detected as a system command even though SuperNewChatApp 
# is not in any hardcoded list
```

### 2. Integration Tests (`test_vision_integration.py`)
Tests with actual window detection on macOS.

**Key Tests:**
- **Real Window Detection**: Detects actual open applications
- **Unknown App Handling**: Tests apps not in predefined lists
- **Window Capture Fallback**: Tests graceful handling when screenshots fail
- **Dynamic Categorization**: Tests pattern-based app categorization

**Note:** These tests require macOS and screen recording permissions.

### 3. Dynamic Visual Analysis Tests (`test_dynamic_visual_analysis.py`)
Tests the ability to analyze visual content dynamically.

**Key Tests:**
- **Notification Pattern Detection**: Various badge formats (5), [3], •••, etc.
- **Multi-Language Indicators**: 新消息, Новые, etc.
- **Visual Indicator Recognition**: Dots, exclamations, badges
- **Context-Aware Analysis**: Adapts responses to visible content

## Running the Tests

### Run All Tests
```bash
cd backend/tests
python run_vision_tests.py
```

### Run Specific Category
```bash
# Functional tests only
python run_vision_tests.py functional

# Integration tests only (macOS)
python run_vision_tests.py integration

# Dynamic analysis tests only
python run_vision_tests.py dynamic

# Async tests only
python run_vision_tests.py async
```

### Run Individual Test Files
```bash
# Test unknown app detection
python test_intelligent_vision.py

# Test with real windows (macOS)
python test_vision_integration.py

# Test visual analysis
python test_dynamic_visual_analysis.py
```

## Expected Output

### Successful Test Run
```
🧪 JARVIS VISION INTELLIGENCE TEST SUITE
======================================================================
Platform: Darwin arm64
Python: 3.10.0
Started: 2024-01-20 15:30:00
======================================================================

📋 FUNCTIONAL TESTS
--------------------------------------------------
test_query_detection_for_unknown_apps ... ok
test_app_pattern_detection ... ok
test_flexible_query_handling ... ok
[... more tests ...]

🔗 INTEGRATION TESTS
--------------------------------------------------
test_detect_any_open_app ... ok
📱 Detected 15 windows:
   1. WhatsApp - WhatsApp
   2. SuperNewChatApp - 5 unread messages
   3. CustomWorkTool - 3 notifications
[... more tests ...]

📊 TEST SUMMARY
======================================================================
TOTAL: 45 tests
✅ PASSED: 45 (100.0%)
❌ FAILED: 0
⏱️ Duration: 12.34 seconds

📋 FEATURE COVERAGE:
  ✅ Unknown app detection
  ✅ Multi-language support
  ✅ Dynamic notification detection
  ✅ Pattern-based app categorization
  ✅ Context-aware query routing
  ✅ Visual indicator recognition
  ✅ Fallback analysis (no screenshots)
  ✅ Real window detection
  ✅ Window capture with fallback
======================================================================
🎉 ALL TESTS PASSED! Vision intelligence is working correctly.
```

## Test Scenarios Covered

### 1. Unknown Apps
- Apps JARVIS has never seen before (SuperNewChatApp, CustomWorkTool)
- Apps with unusual names (未知应用, RandomBusinessApp)
- Apps that don't fit any category

### 2. Notification Formats
- Numeric badges: (5), [3], (99+)
- Text indicators: "3 new", "unread messages"
- Visual indicators: •••, !, badges
- Multi-language: 新消息, Новые, رسائل جديدة

### 3. Query Variations
- Direct: "notifications from WhatsApp"
- Indirect: "that new app", "the blue one"
- Generic: "any notifications", "check everything"
- Multi-language: "check 微信"

### 4. Fallback Scenarios
- Window capture fails
- No screenshot available
- Permission denied
- Unknown window types

## Verifying the Intelligence

The tests verify that JARVIS:

1. **Detects ANY app** without hardcoding
2. **Understands context** from window titles
3. **Routes queries intelligently** based on intent
4. **Handles failures gracefully** with fallbacks
5. **Works with any language** or app name

## Troubleshooting

### Common Issues

**"No windows detected"**
- Make sure you have applications open
- On macOS, grant screen recording permission

**"Import errors"**
- Run from the backend directory
- Ensure all dependencies are installed

**"Async tests timeout"**
- This is normal if ML models are loading
- First run may take longer

### macOS Permissions

For integration tests on macOS:
1. System Preferences → Security & Privacy
2. Privacy → Screen Recording
3. Check Terminal/IDE
4. Restart Terminal/IDE

## Adding New Tests

To test a new unknown app scenario:

```python
def test_my_new_app_scenario(self):
    """Test a specific unknown app scenario"""
    
    # Create a mock window for an unknown app
    unknown_window = WindowInfo(
        window_id="1",
        app_name="MyCompletelyNewApp",
        window_title="MyCompletelyNewApp - 7 alerts",
        bounds={"x": 0, "y": 0, "width": 800, "height": 600},
        is_focused=True
    )
    
    # Test query detection
    query = "check MyCompletelyNewApp"
    is_system = self.jarvis._is_system_command(query)
    self.assertTrue(is_system, "Should detect as system command")
    
    # Test routing
    route = self.router.route_query(query, [unknown_window])
    self.assertEqual(route.intent, QueryIntent.SPECIFIC_APP)
    self.assertIn(unknown_window, route.target_windows)
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
test-vision:
  runs-on: macos-latest  # For full tests
  steps:
    - name: Run Vision Tests
      run: |
        cd backend/tests
        python run_vision_tests.py
```

For non-macOS CI:

```yaml
test-vision-functional:
  runs-on: ubuntu-latest
  steps:
    - name: Run Functional Tests
      run: |
        cd backend/tests
        python run_vision_tests.py functional
```

## Success Criteria

The vision intelligence system is working correctly when:

1. ✅ All functional tests pass (no hardcoding)
2. ✅ Integration tests detect real windows (macOS)
3. ✅ Dynamic analysis handles any notification format
4. ✅ Unknown apps are handled as well as known apps
5. ✅ Fallbacks work when screenshots fail

## Conclusion

These tests ensure JARVIS can intelligently analyze ANY content on your screen without needing hardcoded app lists or patterns. The system is truly dynamic and future-proof.