# 🤖 JARVIS - Full Autonomy AI Agent (v4.0 - Production Ready)

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Agent-Full%20Autonomy-purple" alt="Full Autonomy">
  <img src="https://img.shields.io/badge/AI-Claude%203-blue" alt="Claude AI">
  <img src="https://img.shields.io/badge/Vision-Real--time%20OCR-green" alt="Vision System">
  <img src="https://img.shields.io/badge/UI-Iron%20Man%20Inspired-red" alt="Iron Man UI">
  <img src="https://img.shields.io/badge/Latency-%3C2s-brightgreen" alt="Fast Response">
  <img src="https://img.shields.io/badge/Status-PRODUCTION-success" alt="Production">
  <img src="https://img.shields.io/badge/Version-4.0-brightgreen" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Enterprise--grade-gold" alt="Enterprise">
</p>

<p align="center">
  <em>"JARVIS, sometimes you gotta run before you can walk." - Tony Stark</em>
</p>

## 🎯 Overview

JARVIS v4.0 represents the pinnacle of autonomous AI technology - a **Fully Autonomous Digital Agent** that monitors, understands, decides, and acts independently across your entire digital workspace. This isn't just an upgrade; it's a complete transformation from a voice-activated assistant to a production-ready autonomous system with enterprise-grade reliability.

> **"Sir, I've detected 3 urgent Slack messages while you were coding. I've also noticed your calendar shows a meeting in 5 minutes. I've prepared your workspace by hiding sensitive windows and muting distractions. The meeting link is ready in your browser. Shall I handle the non-urgent notifications after your meeting?"** - JARVIS v4.0 (Full Autonomy Mode)

### 🚀 What's New in v4.0 - Full Autonomy Edition

- 👁️ **Complete Vision Pipeline** - Real-time screen monitoring with OCR text extraction
- 🧠 **Autonomous Decision Engine** - Context-aware decisions based on screen content
- 📋 **Priority Action Queue** - Smart prioritization with safety controls
- 🛡️ **Enterprise Error Recovery** - Self-healing with multiple recovery strategies
- 📊 **Performance Monitoring** - Real-time metrics and health tracking
- 🔄 **State Management** - System states with automatic transitions
- ⚡ **<2s Latency** - Lightning-fast response times
- 🔐 **User Approval Workflow** - Confirm sensitive actions before execution

### Key Capabilities

**Real-time Monitoring**
- Continuous screen capture every 2 seconds
- OCR text extraction from all visible content
- Window state and application tracking
- Notification detection across all apps

**Intelligent Decision Making**
- Analyzes screen content to identify actionable items
- Prioritizes actions based on urgency and context
- Learns from user preferences over time
- Respects focus time and meeting schedules

**Autonomous Actions**
- Handles routine notifications automatically
- Prepares workspace for meetings
- Manages window organization
- Responds to urgent items first

**Enterprise Features**
- Component health monitoring
- Automatic error recovery
- Performance metrics tracking
- WebSocket real-time updates

## 🏗️ Architecture Overview

JARVIS v4.0 implements a complete 3-layer architecture:

### Layer 1: Vision System (Python)
- **Screen Capture Module** - Cross-platform screen capture with differential mode
- **OCR Processing** - Text extraction with region classification
- **Window Analysis** - Application state and content understanding

### Layer 2: Backend API (FastAPI)
- **Vision Decision Pipeline** - Integrates all vision components
- **System State Manager** - Handles state transitions and health
- **Error Recovery System** - Multiple recovery strategies
- **Monitoring & Metrics** - Real-time performance tracking

### Layer 3: Frontend UI (React)
- **ActionDisplay Component** - Shows pending/executed actions
- **WorkspaceMonitor** - Real-time workspace visualization
- **VisionConnection** - WebSocket communication

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- macOS (for full features)
- Screen Recording permission
- Anthropic API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/JARVIS-AI-Agent.git
cd JARVIS-AI-Agent

# 2. Set up API key
echo "ANTHROPIC_API_KEY=your-key-here" > backend/.env

# 3. Install dependencies
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 4. Grant permissions (macOS)
# System Preferences → Security & Privacy → Screen Recording
# Check the box next to Terminal/IDE

# 5. Start JARVIS
python start_system.py
```

### Enable Full Autonomy

```javascript
// In the UI, click "Enable Autonomous Mode" or say:
"Hey JARVIS, enable autonomous mode"

// JARVIS will start:
// - Monitoring your screen continuously
// - Detecting notifications and urgent items
// - Making intelligent decisions
// - Executing approved actions
```

## 💡 How It Works

### Vision Pipeline Flow

```
Screen Capture (2s) → OCR Processing → Window Analysis → Decision Engine → Action Queue → Execution
     ↓                      ↓                ↓                ↓                ↓            ↓
 Differential         Text Extraction    App States      Smart Decisions   Priority    Safety
   Mode               Classification    Content Type     Context-aware     Ordering    Controls
```

### Example Autonomous Behavior

```python
# Scenario: Multiple notifications while coding
1. JARVIS detects: Slack (3 new), Calendar (meeting in 5 min), 1Password open
2. Analysis: User is coding, meeting approaching, sensitive data visible
3. Decisions:
   - Priority 1: Hide 1Password (meeting prep)
   - Priority 2: Notify about meeting
   - Priority 3: Mute Slack until after meeting
4. Execution: Actions queued and executed with user approval
```

## 🎯 Core Features

### Real-time Vision System
- **Screen Monitoring** - Captures screen every 2 seconds
- **OCR Processing** - Extracts all visible text
- **Window Analysis** - Understands application states
- **Change Detection** - Only processes when screen changes

### Autonomous Decision Engine
- **Context Analysis** - Understands what you're doing
- **Smart Prioritization** - Critical > High > Medium > Low
- **Permission Learning** - Learns what you approve/deny
- **Safety Controls** - Requires approval for sensitive actions

### Action Management
- **Priority Queue** - Executes most important actions first
- **Retry Logic** - Handles failures gracefully
- **User Approval** - Confirms before sensitive actions
- **Action History** - Tracks all executed actions

### Enterprise Features
- **Health Monitoring** - Tracks component status
- **Error Recovery** - Self-healing capabilities
- **Performance Metrics** - Real-time performance data
- **State Management** - Manages system states

## 📊 Performance Metrics

- **Latency**: <2 seconds for decision making
- **OCR Speed**: <500ms for typical screen
- **Queue Processing**: 3 concurrent actions
- **Error Recovery**: Automatic retry with backoff
- **Memory Usage**: ~500MB active
- **CPU Usage**: <15% during monitoring

## 🛠️ Technical Components

### Backend Modules

```
backend/
├── vision/
│   ├── screen_capture_module.py    # Screen capture with optimization
│   ├── ocr_processor.py            # OCR with text classification
│   └── window_analysis.py          # Window state analysis
├── autonomy/
│   ├── vision_decision_pipeline.py # Main processing pipeline
│   ├── system_states.py            # State management
│   ├── error_recovery.py           # Error handling
│   └── monitoring_metrics.py       # Performance tracking
└── api/
    └── vision_api.py               # REST/WebSocket endpoints
```

### Frontend Components

```
frontend/src/components/
├── ActionDisplay.js        # Autonomous action UI
├── WorkspaceMonitor.js     # Workspace visualization
└── VisionConnection.js     # WebSocket handling
```

## 🔧 Configuration

### Autonomous Mode Settings

```javascript
// In backend/autonomy/vision_decision_pipeline.py
config = {
    'enable_ocr': true,              // Enable text extraction
    'confidence_threshold': 0.7,      // Min confidence for actions
    'max_actions_per_cycle': 5,       // Limit concurrent actions
    'capture_interval': 2.0           // Screen capture frequency
}
```

### Component Health Thresholds

```python
# In backend/autonomy/monitoring_metrics.py
alert_thresholds = {
    'error_rate': 0.1,      # 10% error rate triggers alert
    'queue_depth': 100,     # Max queue size
    'response_time': 5.0    # Max response time in seconds
}
```

## 📡 API Endpoints

### Vision Control
```
POST /vision/pipeline/control?action=start|stop
GET  /vision/pipeline/status
GET  /vision/monitoring/report
GET  /vision/monitoring/health
```

### WebSocket
```
WS /ws/vision - Real-time vision updates
```

### Example Response
```json
{
  "type": "workspace_update",
  "timestamp": "2024-01-20T10:30:00Z",
  "workspace": {
    "window_count": 12,
    "notifications": ["Slack: 3 new messages"],
    "focused_app": "Visual Studio Code"
  },
  "autonomous_actions": [
    {
      "type": "handle_notification",
      "priority": "HIGH",
      "confidence": 0.85
    }
  ]
}
```

## 🛡️ Safety & Privacy

- **Local Processing** - All vision processing happens on your machine
- **User Approval** - Sensitive actions require confirmation
- **No Data Storage** - No screenshots or text are permanently stored
- **Permission Learning** - Learns your preferences locally
- **Secure Communication** - All API calls use HTTPS

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Can't see screen" | Grant Screen Recording permission and restart Terminal/IDE |
| "OCR not working" | Install tesseract: `brew install tesseract` |
| "High latency" | Check CPU usage and reduce capture frequency |
| "Actions not executing" | Verify action queue is running in pipeline status |

### Debug Commands

```bash
# Check vision system
curl http://localhost:8000/vision/status

# View pipeline metrics
curl http://localhost:8000/vision/monitoring/report

# Test screen capture
python backend/vision/screen_capture_module.py
```

## 📈 Roadmap

### Completed ✅
- [x] Real-time vision system
- [x] OCR text extraction
- [x] Window analysis engine
- [x] Autonomous decision pipeline
- [x] Priority action queue
- [x] Error recovery system
- [x] Performance monitoring
- [x] State management

### In Progress 🚧
- [ ] Multi-monitor support
- [ ] Custom action plugins
- [ ] Advanced ML predictions

### Planned 📋
- [ ] Mobile app integration
- [ ] Cloud sync for preferences
- [ ] Third-party app integrations
- [ ] Voice command integration

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Anthropic** for Claude AI
- **Marvel/Disney** for JARVIS inspiration
- **Open source community** for amazing tools
- **You** for bringing JARVIS to life!

---

<p align="center">
<strong>⭐ Star this repo if JARVIS helps automate your digital life!</strong><br>
<em>"The truth is... I am Iron Man." - Tony Stark</em>
</p>