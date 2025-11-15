# JARVIS macOS HUD

Native macOS desktop overlay interface for JARVIS AI Assistant.

## Overview

This is the macOS HUD implementation that provides a transparent, always-on-top interface for JARVIS, maintaining the exact visual aesthetic from the web app (neon green, terminal-style design).

## Features

- **🎯 True Holographic Overlay**: Fully transparent, click-through window with NO traditional window frame
  - Completely borderless and chromeless
  - Zero blur - pure transparency showing desktop through
  - Only JARVIS UI elements (text, Arc Reactor, buttons) are visible
  - No interference with desktop or other applications

- **🖱️ Revolutionary Click-Through Technology**:
  - **Desktop fully accessible** - click files, folders, and windows behind the HUD
  - **Smart event capture** - interactive elements become clickable when you hover over them
  - **Global mouse tracking** - dynamically enables/disables window events in real-time
  - **Zero blocking** - JARVIS floats above everything without preventing interaction

- **🎨 Loading Screen with Matrix Transition**:
  - Iron Man-style loading animation with Arc Reactor
  - Real-time progress updates from Python backend
  - Matrix code rain transition effect when loading completes
  - Seamless fade to main HUD interface

- **⚛️ Arc Reactor Animation**: Animated concentric rings matching web app design
  - Neon green pulse effect
  - Cyan variant for listening state
  - Smooth CSS-inspired animations

- **💬 Real-time Transcript Display**: Terminal-style transcript of conversations
  - User speech displayed in white
  - JARVIS responses in neon green
  - Auto-scrolling message history

- **🎨 Web App Color Matching**: Pixel-perfect colors from web frontend
  - Primary Green: `#00ff41` (Matrix neon green)
  - Cyan Accents: `#00FFFF`, `#00D9FF` (Arc Reactor glow)
  - Pure Black Background: `#000000`
  - Custom glow and shadow effects

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
├── JARVIS-HUD.xcodeproj/           # Xcode project file
└── JARVIS-HUD/
    ├── JARVISApp.swift              # Main app entry point + click-through system
    ├── HUDView.swift                # Main HUD interface
    ├── LoadingHUDView.swift         # Loading screen with Matrix transition
    ├── JARVISColors.swift           # Color system (matches web app exactly)
    ├── JARVISPulseView.swift        # Animated pulse rings (deprecated - see ArcReactorView)
    ├── ArcReactorView.swift         # Arc Reactor animation with state support
    ├── TransparentWindow.swift      # Custom NSWindow for overlay
    ├── ClickThroughWindow.swift     # Custom click-through window implementation
    └── PythonBridge.swift           # WebSocket/HTTP bridge to Python backend
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

The HUD implements a revolutionary holographic overlay system:

### Transparency & Chrome
- **Zero Window Chrome**: Completely borderless with no title bar, shadow, or traditional window elements
- **Pure Transparency**: `Color.clear` background with no blur effects
- **Full-Screen Overlay**: Covers entire screen while remaining completely transparent
- **Hidden from Dock**: Uses `.accessory` activation policy (not visible in Dock or app switcher)

### Window Level & Spaces
- **Window Level**: `.floating` (always on top, non-intrusive)
- **Collection Behavior**:
  - `.canJoinAllSpaces` - visible across all Mission Control Spaces
  - `.stationary` - doesn't move when switching Spaces
  - `.fullScreenAuxiliary` - works alongside full-screen apps
  - `.ignoresCycle` - not in Cmd+Tab app switcher

### Click-Through Technology
- **Global Mouse Tracking**: Monitors mouse position system-wide in real-time
- **Dynamic Event Switching**:
  - `window.ignoresMouseEvents = true` by default (full click-through)
  - Switches to `false` when mouse hovers over interactive elements
  - Switches back to `true` when mouse leaves interactive areas
- **Interactive Elements Detected**:
  - NSButton (e.g., "SEND" button)
  - Editable NSTextField (command input field)
  - NSControl elements
- **Non-Interactive Pass-Through**:
  - Text labels (JARVIS title, transcript)
  - Arc Reactor animation
  - Empty/transparent space
  - All visual-only elements

### Focus Behavior
- **Non-Activating**: Never steals focus from other applications
- **Keyboard Input**: Only accepts input when user explicitly clicks interactive element
- **Background Operation**: Remains passive until user interaction required

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

### Phase 1 - Foundation ✅ COMPLETE
- [x] Create transparent NSWindow
- [x] Render JARVIS Arc Reactor animation
- [x] Establish color system matching web app exactly
- [x] Basic Python backend bridge (WebSocket + HTTP)
- [x] Loading screen with Arc Reactor and progress bar
- [x] Matrix code rain transition effect
- [x] Pure transparency with zero blur

### Phase 2 - Holographic Overlay ✅ COMPLETE
- [x] Remove ALL window chrome (borderless, no shadow, no title bar)
- [x] Implement revolutionary click-through technology
- [x] Global mouse tracking for dynamic event capture
- [x] Full desktop accessibility (click files/folders/windows through HUD)
- [x] Smart interactive element detection
- [x] Loading progress buffer system in backend
- [x] Real-time progress updates from Python backend

### Phase 3 - HUD Integration (Current)
- [x] Connect to Python backend WebSocket
- [x] Real-time loading progress display
- [ ] Full transcript streaming (partial - demo implemented)
- [ ] Auto-show and auto-hide logic
- [ ] Voice command integration
- [ ] State-based Arc Reactor animations (listening, processing, speaking)

### Phase 4 - Advanced Features
- [ ] macOS window control via JARVIS
- [ ] Improved transition animations
- [ ] Keyboard shortcuts for HUD control
- [ ] Multi-monitor support

### Future Enhancements
- [ ] Local LLM inference support
- [ ] Gesture-triggered HUD activation
- [ ] Full visual dashboard with metrics
- [ ] Plugin architecture for extensions

## Permissions

The app requires the following macOS permissions:

- **Microphone Access**: For voice commands (configured in project settings)
- **Accessibility**: For system-level window control (future)
- **Screen Recording**: For HUD enhancements (future)

## Troubleshooting

### HUD not appearing
- Check that Python backend is running on `localhost:8000`
- Verify WebSocket connection in Console.app logs
- Window should auto-appear on launch (loading screen first)
- Check that window level is set to `.floating`

### Can't click through HUD to desktop
- Verify `window.ignoresMouseEvents = true` is set in AppDelegate
- Check that global mouse tracking is initialized
- Look for "startMouseTracking()" call in Console.app logs
- Ensure interactive element detection is working (hover over buttons)

### Interactive elements (buttons) not clickable
- Mouse over the element - window should switch to `ignoresMouseEvents = false`
- Check that `isInteractiveElement()` correctly identifies buttons/text fields
- Verify global monitor is receiving mouse events

### Colors don't match web app
- Compare with `frontend/src/components/JarvisVoice.css`
- Verify hex values in `JARVISColors.swift`
- Check shadow/glow opacity settings

### Loading screen stuck
- Check Python backend is sending progress updates via WebSocket
- Verify progress buffer system is working in `unified_websocket.py`
- Look for `loading_complete = true` message from backend
- Check Console.app for "Backend signaled completion" message

### Matrix transition not playing
- Ensure `pythonBridge.loadingComplete` triggers properly
- Check that MatrixTransitionView initializes with correct column count
- Verify opacity animations are running

## License

Part of the JARVIS-AI-Agent project by Derek J. Russell.

## Related Files

- Web App Colors: `frontend/src/components/JarvisVoice.css`
- Web App Design: `frontend/src/App.css`, `frontend/src/index.css`
- PRD Document: See attached PRD in project root
