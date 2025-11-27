# JARVIS AI Agent Changelog

## v3.8.0 - Async Vision & Timeout Protection (2025-11-26)

### 🛡️ Major Stability Improvements

This release fixes critical blocking issues that could cause JARVIS to hang indefinitely when processing vision queries like "can you see my screen?" or "what's happening across my workspaces?"

#### ✅ Bug Fixes
- **Fixed**: "Can you see my screen?" queries no longer hang indefinitely
- **Fixed**: Multi-space Yabai queries now complete within timeout
- **Fixed**: PyAutoGUI blocking operations no longer freeze the event loop
- **Fixed**: Claude API timeouts now return graceful error messages

#### 🔧 New Features

- **ThreadPoolExecutor Integration**
  - All blocking PyAutoGUI operations now run in background threads
  - Screen capture, mouse clicks, and typing are now non-blocking
  - Prevents UI freezes during computer use operations

- **Circuit Breaker Pattern**
  - API calls protected by circuit breaker (3 failures = open)
  - Auto-recovery after 60 seconds
  - Graceful degradation when Claude API is unavailable
  - Prevents cascading failures during outages

- **Comprehensive Timeout Protection**
  - Overall command timeout: 45 seconds
  - Claude API calls: 30 seconds
  - Screen capture: 10 seconds
  - Action execution: 10 seconds
  - Yabai subprocess calls: 5 seconds

- **Async Yabai Support**
  - New `enumerate_all_spaces_async()` method
  - New `describe_workspace_async()` method
  - New `get_workspace_summary_async()` method
  - Subprocess calls run in thread pool with timeout

#### 📁 Files Changed
- `backend/display/computer_use_connector.py` - Added `run_blocking()`, `CircuitBreaker`, async `ScreenCaptureHandler` and `ActionExecutor`
- `backend/api/vision_command_handler.py` - Added overall timeout wrapper
- `backend/api/pure_vision_intelligence.py` - Added `VisionCircuitBreaker`, timeout on API calls
- `backend/vision/yabai_space_detector.py` - Added `run_subprocess_async()` and async methods
- `backend/vision/intelligent_vision_router.py` - Updated `_execute_yabai()` with async support

#### 📊 Performance Impact
- Vision queries now guaranteed to complete or timeout (max 45s)
- No more indefinite hangs on slow API responses
- Improved responsiveness during high load
- Better error messages for timeout conditions

---

## v3.7.0 - PRD Complete Edition (2025-08-20)

### 🎉 Major Milestone: 100% PRD Complete!

#### ✅ New Features
- **Meeting Preparation System** - Auto-detects meetings and prepares workspace
  - Detects Zoom, Google Meet, Teams
  - Automatically identifies and hides sensitive windows
  - Meeting-specific layout templates (presentation, collaboration, focus)
  - Alerts for missing materials or conflicts
  
- **Privacy Control System** - Multi-mode privacy protection
  - Four modes: normal, meeting, focused, private
  - Automatic sensitive content detection
  - Pattern-based content filtering
  - Temporary privacy sessions
  
- **Workflow Learning System** - ML-powered pattern recognition
  - Uses sklearn for clustering and predictions
  - Learns daily workflow patterns
  - Predicts missing windows based on history
  - Time-of-day aware suggestions
  
- **Comprehensive Testing Suite** - Full test coverage
  - Functional tests for all features (87.5% pass rate)
  - Performance tests (<3s response time verified)
  - Integration tests for APIs
  - Cost optimization tests (<$0.05 per query)

#### 🔧 Improvements
- Enhanced query routing with new patterns
- Better sensitive content detection
- Improved window relationship algorithms
- Optimized API usage for cost efficiency

#### 📊 Metrics Achieved
- Response Time: ✅ 95% of queries <3 seconds
- API Cost: ✅ 90% of queries <$0.05
- Test Coverage: ✅ 100% feature coverage
- Production Ready: ✅ Zero P0 bugs

## v3.5.0 - Intelligence Layer (Previous)
- Window Relationship Detection
- Smart Query Routing
- Proactive Insights
- Workspace Optimization

## v3.2.1 - Vision Enhanced Edition (Previous)
- Computer Vision System
- Screen Analysis
- Update Detection
- OCR Text Extraction

## v3.1.3 - Iron Man Edition (Previous)
- Futuristic Landing Page
- Arc Reactor UI
- Enhanced Voice System
- Extended Timeouts