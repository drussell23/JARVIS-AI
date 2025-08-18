# 🖥️ JARVIS Vision System Guide

## Overview

The JARVIS Vision System adds computer vision capabilities to JARVIS, enabling it to:
- 👀 See and understand what's displayed on your screen
- 🔄 Detect software updates and system notifications
- 📝 Extract and read text from any application
- 🎯 Identify UI elements and application states
- 🧠 Use Claude's vision AI for advanced understanding

## Architecture

```
vision/
├── screen_vision.py          # Core vision system with OCR
├── claude_vision_analyzer.py # Claude AI vision integration
└── VISION_SYSTEM_GUIDE.md   # This guide

api/
└── vision_api.py            # REST API endpoints

voice/
└── jarvis_agent_voice.py    # Voice command integration
```

## Installation

### 1. Install Dependencies

```bash
# macOS-specific (required for screen capture)
brew install tesseract

# Python dependencies
pip install opencv-python pytesseract Pillow
pip install pyobjc-framework-Quartz pyobjc-framework-Vision
```

### 2. Verify Installation

```bash
# Test basic vision
cd backend
python test_vision_system.py

# Test with JARVIS
python test_jarvis_voice.py
# Say: "Hey JARVIS, what's on my screen?"
```

## Voice Commands

### Basic Vision Commands

```
"Hey JARVIS, what's on my screen?"
→ Describes visible applications and content

"Hey JARVIS, check for software updates"
→ Scans screen for update notifications

"Hey JARVIS, what applications are open?"
→ Lists detected applications

"Hey JARVIS, analyze my screen"
→ Provides detailed screen analysis
```

### Monitoring Commands

```
"Hey JARVIS, start monitoring for updates"
→ Begins continuous monitoring (checks every 5 minutes)

"Hey JARVIS, stop monitoring"
→ Stops update monitoring

"Hey JARVIS, are there any updates?"
→ Quick check for pending updates
```

### Advanced Commands

```
"Hey JARVIS, read the text in the menu bar"
→ Extracts text from specific screen region

"Hey JARVIS, is there anything important on screen?"
→ Uses AI to identify important information

"Hey JARVIS, check for security alerts"
→ Scans for security-related notifications
```

## API Endpoints

### Check Vision Status
```bash
GET /api/vision/status

# Response:
{
    "vision_enabled": true,
    "claude_vision_available": true,
    "monitoring_active": false,
    "detected_updates": 0
}
```

### Process Vision Command
```bash
POST /api/vision/command
{
    "command": "check for updates",
    "use_claude": true
}
```

### Analyze Screen
```bash
POST /api/vision/analyze
{
    "analysis_type": "updates",  # or "activity", "security", "text"
    "region": [100, 100, 500, 300]  # Optional: [x, y, width, height]
}
```

### Configure Monitoring
```bash
POST /api/vision/monitor/updates
{
    "enabled": true,
    "interval": 300,  # seconds
    "notify_critical_only": false
}
```

## Features

### 1. Software Update Detection

The system can detect:
- macOS system updates
- App Store updates
- Browser update notifications
- Security updates
- Application-specific updates

Detection methods:
- OCR text extraction
- Pattern matching for update keywords
- Red notification badge detection
- UI element analysis

### 2. Screen Context Understanding

Provides information about:
- Open applications
- Visible text content
- UI element positions
- Notification badges
- System status indicators

### 3. Claude Vision Integration

When Claude API is available:
- Advanced image understanding
- Natural language descriptions
- Activity context analysis
- Security threat detection
- Intelligent suggestions

### 4. Continuous Monitoring

- Background monitoring for updates
- Configurable check intervals
- Priority-based notifications
- Automatic JARVIS announcements

## Use Cases

### 1. Update Management
```
User: "Hey JARVIS, check if I need to update anything"
JARVIS: "Sir, I've detected 3 updates: macOS 14.2, Chrome browser, and Slack. 
         The macOS update is marked as a critical security update."
```

### 2. Productivity Assistant
```
User: "Hey JARVIS, what am I working on?"
JARVIS: "I can see you have VS Code open with a Python file, Chrome with 
         documentation tabs, and Terminal running tests. You appear to be 
         debugging the authentication module."
```

### 3. Security Monitoring
```
User: "Hey JARVIS, start monitoring for security alerts"
JARVIS: "I'll monitor your screen for security notifications and alert you 
         immediately if any appear, sir."
```

### 4. Accessibility
```
User: "Hey JARVIS, read the error message on screen"
JARVIS: "The error message says: 'Connection timeout: Unable to reach server. 
         Please check your internet connection and try again.'"
```

## Technical Details

### Screen Capture (macOS)
- Uses Quartz framework for native screen capture
- No external dependencies or permissions needed
- Supports full screen or specific regions
- Hardware-accelerated on Apple Silicon

### Text Extraction
- Tesseract OCR for text recognition
- Pre-processing for better accuracy:
  - Grayscale conversion
  - Adaptive thresholding
  - Noise reduction
- Multi-language support (configure in Tesseract)

### Update Detection Algorithm
1. Capture screen or specific regions
2. Extract text using OCR
3. Apply regex patterns for update keywords
4. Detect red notification badges
5. Analyze UI elements for update indicators
6. Cross-reference with known update patterns
7. Classify by urgency and type

### Performance Optimizations
- Async/await for non-blocking operations
- Region-based capture for efficiency
- Caching of recent scans
- Batch processing of text regions
- GPU acceleration where available

## Privacy & Security

- All processing happens locally
- No screenshots are stored permanently
- No data sent to external services (except Claude API if enabled)
- Only extracted text is processed, not raw images
- User can disable monitoring at any time

## Troubleshooting

### "Vision capabilities are not available"
1. Install required dependencies: `pip install -r requirements.txt`
2. Install Tesseract: `brew install tesseract`
3. Restart JARVIS

### "No text detected on screen"
1. Ensure good screen contrast
2. Try different screen regions
3. Check Tesseract installation: `tesseract --version`

### "Claude vision not working"
1. Verify ANTHROPIC_API_KEY is set
2. Ensure you're using a vision-capable Claude model
3. Check API quota/limits

### "Updates not detected"
1. Ensure update notifications are visible
2. Try manual region selection
3. Check if notifications are in supported languages

## Future Enhancements

- [ ] Multi-monitor support
- [ ] Custom update pattern training
- [ ] Integration with native macOS notification center
- [ ] Screenshot history with searchable text
- [ ] Automated action execution (with safety controls)
- [ ] Support for Windows and Linux
- [ ] Real-time screen change detection
- [ ] Application-specific intelligence

## Contributing

To add new detection patterns:

1. Edit `screen_vision.py`:
```python
def _initialize_update_patterns(self):
    return {
        "your_app": [
            re.compile(r"Your App.*update pattern", re.I)
        ]
    }
```

2. Add voice command in `jarvis_agent_voice.py`:
```python
self.special_commands["your command"] = "Description"
```

3. Test with: `python test_vision_system.py`