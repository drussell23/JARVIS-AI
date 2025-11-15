# JARVIS macOS HUD

Native macOS desktop overlay interface for JARVIS AI Assistant.

## Overview

This is the macOS HUD implementation that provides a transparent, always-on-top interface for JARVIS, maintaining the exact visual aesthetic from the web app (neon green, terminal-style design).

## Features

- **Transparent Overlay Window**: Borderless, always-on-top HUD that appears over all applications
- **Concentric Pulse Animation**: Animated green rings matching the web app's arc reactor design
- **Real-time Transcript Display**: Shows user speech and JARVIS responses in terminal-style green text
- **Auto-show/Auto-hide**: Appears on command, auto-hides after 10 seconds of inactivity
- **Web App Color Matching**: Uses identical colors from the web frontend
  - Primary Green: `#00ff41` (Matrix green)
  - Cyan Accents: `#00FFFF`, `#00D9FF`
  - Pure Black Background: `#000000`
  - Neon glows and shadows

## Architecture

```
SwiftUI (HUD UI Layer)
      ↓
NSWindow (Transparent Overlay)
      ↓
Swift App Logic
      ↓ WebSocket / HTTP ↓
Python Backend (FastAPI + Multi-agent system)
      ↓
macOS System APIs
```

## Project Structure

```
macos-hud/
├── JARVIS-HUD.xcodeproj/       # Xcode project file
└── JARVIS-HUD/
    ├── JARVISApp.swift          # Main app entry point
    ├── HUDView.swift            # Main HUD interface
    ├── JARVISColors.swift       # Color system (matches web app)
    ├── JARVISPulseView.swift    # Animated pulse rings
    ├── TransparentWindow.swift  # Custom NSWindow for overlay
    └── PythonBridge.swift       # WebSocket/HTTP bridge to Python backend
```

## Requirements

- macOS 13.0+
- Xcode 15.0+
- Swift 5.9+
- Python backend running on `localhost:8000` (from main JARVIS repo)

## Building & Running

### Option 1: Xcode

1. Open `JARVIS-HUD.xcodeproj` in Xcode
2. Select the JARVIS-HUD target
3. Build and run (⌘R)

### Option 2: Command Line

```bash
cd macos-hud
xcodebuild -project JARVIS-HUD.xcodeproj -scheme JARVIS-HUD -configuration Debug build
open build/Debug/JARVIS-HUD.app
```

## Configuration

The HUD connects to the Python backend at:
- WebSocket: `ws://localhost:8000/ws`
- HTTP API: `http://localhost:8000`

To change the backend URL, edit `PythonBridge.swift`:

```swift
init(
    websocketURL: String = "ws://localhost:8000/ws",
    apiBaseURL: String = "http://localhost:8000"
)
```

## Color System

All colors are extracted from the web app's CSS files and defined in `JARVISColors.swift`:

### Primary Colors
- `Color.jarvisGreen` - #00ff41 (primary neon green)
- `Color.jarvisGreenDark` - #00aa2e (darker green)
- `Color.jarvisCyan` - #00FFFF (cyan accent)
- `Color.jarvisBlack` - #000000 (pure black background)

### Status Colors
- `Color.jarvisSuccess` - #00ff88 (success green)
- `Color.jarvisWarning` - #ffaa00 (warning orange)
- `Color.jarvisError` - #f44336 (error red)

### Glow Effects
- `Color.jarvisGreenGlow(opacity:)` - Neon green glow
- `Color.jarvisCyanGlow(opacity:)` - Cyan glow

## HUD States

The HUD displays different visual states:

- **Offline**: Red error color, no pulse
- **Listening**: Cyan pulse animation
- **Processing**: Green pulse animation
- **Speaking**: Active green pulse
- **Idle**: Static green, minimal pulse

## Window Behavior

The HUD window is configured per PRD requirements:

- **Window Level**: `.statusBar` (always on top)
- **Collection Behavior**: `.canJoinAllSpaces` (visible in all Spaces)
- **Transparency**: Fully transparent background
- **Focus**: Does not steal focus from other apps
- **Click-through**: Can be enabled/disabled programmatically

## Auto-hide Logic

The HUD follows these auto-hide rules:

### Shows When:
- Voice command detected
- User says "Hey JARVIS" or "Open the HUD"
- Command is being processed

### Hides When:
- 10 seconds of inactivity (after task completion)
- User says "Close the HUD" or "Dismiss"
- Simple commands like unlock/lock (stays hidden for speed)

## Integration with Python Backend

The HUD expects these message types from the Python backend via WebSocket:

### Transcript Message
```json
{
  "type": "transcript",
  "speaker": "USER" | "JARVIS",
  "text": "Open Safari and search for dogs"
}
```

### State Update
```json
{
  "type": "state",
  "state": "offline" | "listening" | "processing" | "speaking" | "idle"
}
```

### Status Update
```json
{
  "type": "status",
  "message": "System Online"
}
```

## Development Roadmap

### Phase 1 - Foundation ✅
- [x] Create transparent NSWindow
- [x] Render JARVIS pulse animation
- [x] Establish color system matching web app
- [x] Basic Python backend bridge

### Phase 2 - HUD MVP (Current)
- [ ] Implement full transcript streaming
- [ ] Auto-show and auto-hide logic
- [ ] Connect to real Python backend WebSocket
- [ ] Voice command integration

### Phase 3 - System Integration
- [ ] macOS window control
- [ ] Improved animations and transitions
- [ ] Menu bar integration (optional)
- [ ] Keyboard shortcuts

### Future Enhancements
- [ ] Local LLM inference support
- [ ] Gesture-triggered HUD
- [ ] Full visual dashboard
- [ ] Plugin architecture

## Permissions

The app requires the following macOS permissions:

- **Microphone Access**: For voice commands (configured in project settings)
- **Accessibility**: For system-level window control (future)
- **Screen Recording**: For HUD enhancements (future)

## Troubleshooting

### HUD not appearing
- Check that Python backend is running on `localhost:8000`
- Verify WebSocket connection in Console.app logs
- Ensure window level is set to `.statusBar`

### Colors don't match web app
- Compare with `frontend/src/components/JarvisVoice.css`
- Verify hex values in `JARVISColors.swift`
- Check shadow/glow opacity settings

### Window appears in wrong position
- Call `window.center()` to re-center
- Check screen bounds calculations

## License

Part of the JARVIS-AI-Agent project by Derek J. Russell.

## Related Files

- Web App Colors: `frontend/src/components/JarvisVoice.css`
- Web App Design: `frontend/src/App.css`, `frontend/src/index.css`
- PRD Document: See attached PRD in project root
