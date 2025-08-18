# JARVIS AI Agent - System Control Guide

## 🚀 Overview

JARVIS has evolved from a voice assistant into a full AI agent capable of controlling your macOS environment through natural language commands. This system leverages Anthropic's Claude API for intelligent command interpretation and safe execution of system-level tasks.

## 🎯 Key Features

### 1. Natural Language Understanding
- **Claude-Powered**: Uses Claude AI to interpret complex, natural commands
- **Context-Aware**: Maintains conversation history and system state
- **Multi-Intent**: Handles compound commands like "Open Chrome and search for Python tutorials"
- **Confidence Scoring**: Only executes commands with high confidence

### 2. System Control Capabilities

#### Application Management
```
"Open Chrome"
"Close Spotify"
"Switch to Visual Studio Code"
"Show me all open applications"
"Minimize everything"
```

#### File Operations
```
"Create a new file called notes.txt on my desktop"
"Open the report.pdf from my documents"
"Search for Python files in my projects folder"
"Delete old_file.txt from downloads" (requires confirmation)
```

#### System Settings
```
"Set volume to 50%"
"Mute the sound"
"Take a screenshot"
"Put the display to sleep"
"Turn off WiFi"
```

#### Web Integration
```
"Search Google for machine learning tutorials"
"Open YouTube"
"Go to github.com"
"Search for weather in San Francisco"
```

#### Workflow Automation
```
"Start my morning routine"
"Set up my development environment"
"Prepare for a meeting"
```

## 🛡️ Safety Features

### 1. Application Restrictions
- System-critical apps are blocked (System Preferences, Terminal, etc.)
- Confirmation required for potentially disruptive actions

### 2. File System Protection
- Operations limited to safe directories (Desktop, Documents, Downloads, etc.)
- All deletions require explicit confirmation
- System directories are protected

### 3. Command Validation
- Dangerous shell commands are blocked
- Commands are categorized by safety level
- Low-confidence interpretations are rejected

### 4. User Confirmation
- Destructive actions require verbal confirmation
- Clear feedback on what will be executed
- Easy cancellation with "cancel" or "abort"

## 🔧 Setup & Configuration

### 1. Environment Setup
```bash
# Required: Anthropic API key for Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# Optional: OpenWeatherMap for weather features
export OPENWEATHER_API_KEY="your-weather-key"
```

### 2. Install Dependencies
```bash
cd backend
pip install anthropic psutil
```

### 3. Test System Control
```bash
python test_jarvis_agent.py
```

## 💬 Usage Examples

### Basic Commands
```
User: "Hey JARVIS, open Chrome"
JARVIS: "I've opened Google Chrome for you, sir."

User: "Set volume to 30 percent"
JARVIS: "Volume adjusted to 30%, sir."

User: "Take a screenshot"
JARVIS: "Screenshot captured and saved, sir."
```

### Complex Commands
```
User: "Open Visual Studio Code and search for Python tutorials"
JARVIS: "Opening Visual Studio Code... Searching Google for Python tutorials..."

User: "Start my morning routine"
JARVIS: "Initiating morning routine, sir. Opening your email... Checking your calendar... 
         Getting today's weather... Morning routine complete, sir."
```

### Safety Confirmations
```
User: "Delete test.txt from my desktop"
JARVIS: "This action requires your confirmation, sir. Say 'confirm' to proceed or 'cancel' to abort."
User: "Confirm"
JARVIS: "Task completed successfully, sir. Deleted file: test.txt"
```

## 🎛️ Command Modes

### 1. Conversation Mode (Default)
- Normal JARVIS interactions
- System commands detected automatically
- Mixed conversation and control

### 2. System Control Mode
- Focused on system commands
- Higher sensitivity to control keywords
- Activate: "Switch to system control mode"

### 3. Workflow Mode
- Execute predefined routines
- Batch operations
- Custom workflow creation

## 📋 Predefined Workflows

### Morning Routine
1. Opens email client
2. Opens calendar
3. Checks weather
4. Opens news sites

### Development Setup
1. Launches IDE (VS Code)
2. Opens terminal
3. Starts Docker
4. Opens localhost

### Meeting Prep
1. Sets volume to 50%
2. Closes distracting apps
3. Minimizes all windows
4. Opens video conferencing app

## 🔌 API Integration

### WebSocket Commands
```javascript
// Send command via WebSocket
ws.send(JSON.stringify({
    type: "command",
    text: "Open Chrome"
}));

// Receive response
{
    type: "response",
    text: "I've opened Google Chrome for you, sir.",
    context: {...},
    timestamp: "2024-01-20T10:30:00Z"
}
```

### REST API
```bash
# Check status
curl http://localhost:8000/voice/jarvis/status

# Send command
curl -X POST http://localhost:8000/voice/jarvis/command \
    -H "Content-Type: application/json" \
    -d '{"text": "Set volume to 50%"}'
```

## 🚨 Troubleshooting

### System Control Not Working
1. Check ANTHROPIC_API_KEY is set
2. Verify macOS permissions for accessibility
3. Ensure Python has automation permissions

### Commands Not Recognized
1. Speak clearly near the microphone
2. Use command keywords (open, close, set, etc.)
3. Switch to system control mode for better recognition

### Application Not Opening
1. Check exact application name
2. Verify app is installed
3. Try common aliases (e.g., "Chrome" for "Google Chrome")

## 🔒 Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **Network Security**: Use HTTPS in production
3. **Access Control**: Implement user authentication for remote access
4. **Audit Logging**: All system commands are logged
5. **Permission Scope**: Grant minimal required permissions

## 🚀 Advanced Features

### Custom Commands
Add custom commands to `MacOSController`:
```python
def custom_action(self, params):
    # Your custom logic
    return success, message
```

### New Workflows
Define workflows in `execute_workflow`:
```python
"custom_workflow": [
    ("action1", "param1"),
    ("action2", "param2")
]
```

### Voice Feedback
Customize responses in `agent_responses`:
```python
"custom_response": "Custom message with {parameter}, {user}."
```

## 📚 Architecture

```
JARVIS Agent Voice System
├── Natural Language Layer (Claude API)
│   ├── Intent Recognition
│   ├── Entity Extraction
│   └── Confidence Scoring
├── Command Interpreter
│   ├── Safety Validation
│   ├── Parameter Parsing
│   └── Context Management
├── System Controller
│   ├── AppleScript Execution
│   ├── Shell Commands
│   └── File Operations
└── Safety Layer
    ├── Path Restrictions
    ├── Command Filtering
    └── Confirmation Dialogs
```

## 🎯 Future Enhancements

1. **Multi-OS Support**: Windows and Linux compatibility
2. **Plugin System**: Extensible command framework
3. **Visual Feedback**: Screen overlays for confirmations
4. **Scheduling**: Time-based command execution
5. **Macro Recording**: Record and replay command sequences
6. **Remote Control**: Secure remote system access
7. **AI Learning**: Personalized command predictions

## 📝 Contributing

To add new system control features:
1. Add methods to `MacOSController`
2. Update command patterns in `ClaudeCommandInterpreter`
3. Add voice responses to `JARVISAgentVoice`
4. Include safety checks and confirmations
5. Update this documentation

Remember: With great power comes great responsibility. Always prioritize user safety and system security.