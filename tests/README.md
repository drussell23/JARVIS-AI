# 🧪 JARVIS Test Suite

This directory contains all test files for the JARVIS AI Agent system, organized by component.

## 📁 Test Organization

### 🔧 `/backend/`
Core backend functionality tests:
- `test_imports.py` - Import verification
- `test_jarvis_agent.py` - JARVIS agent tests
- `test_jarvis_commands.py` - Command processing tests
- `test_jarvis_fixed.py` - Bug fix verification
- `test_jarvis_import.py` - Import system tests
- `test_ml_enhanced_jarvis.py` - ML enhancement tests
- `test_close_apps.py` - App closing functionality
- `test_jarvis_close_apps.py` - JARVIS app control
- `test_microphone.py` - Microphone functionality
- `verify_api_key.py` - API key verification utility

### 👁️ `/vision/`
Vision system tests:
- `test_vision_system.py` - Core vision functionality
- `test_jarvis_vision_commands.py` - Vision command processing
- `test_jarvis_vision_integration.py` - Vision integration tests
- `test_enhanced_vision_commands.py` - Enhanced vision features
- `test_claude_vision_debug.py` - Claude vision debugging
- `test_enhanced_vision.py` - Enhanced vision system
- `test_jarvis_vision_response.py` - Vision response testing
- `demo_enhanced_vision.py` - Vision demonstration

### 🎤 `/voice/`
Voice system tests (to be added)

### 🖥️ `/system_control/`
System control tests (to be added)

### 🔗 `/integration/`
Full system integration tests:
- `test_claude_math.py` - Claude math capabilities
- `test_jarvis_voice.py` - Voice integration
- `test_jarvis.py` - Full JARVIS system
- `test_memory_api.py` - Memory API testing

### 🎨 `/frontend/`
Frontend tests (to be added)

## 🚀 Running Tests

### Run all tests:
```bash
cd tests
python -m pytest
```

### Run specific category:
```bash
python -m pytest backend/
python -m pytest vision/
python -m pytest integration/
```

### Run individual test:
```bash
python backend/test_jarvis_agent.py
python vision/test_vision_system.py
```

## 📝 Test Guidelines

When adding new tests:
1. Place in the appropriate subdirectory
2. Follow naming convention: `test_*.py`
3. Include docstrings explaining test purpose
4. Update import paths if needed
5. Add to this README

## ⚠️ Important Notes

- Some tests require API keys (check `.env` configuration)
- Vision tests require screen recording permissions on macOS
- Voice tests require microphone access
- Integration tests may take longer to run

## 🔍 Debugging

For test failures:
1. Check required permissions (screen, microphone)
2. Verify API keys are set
3. Ensure dependencies are installed
4. Run individual tests for detailed output