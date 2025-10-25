# Phase 1.1 — Multi-Monitor Support

## Executive Summary 

JARVIS currently assumes a single-display workspace, limiting its vision and context awareness when users operate across multiple monitors. This enhancement introduces Multi-Monitor Support, enabling JARVIS to detect, map, and analyze multiple displays and their active spaces in real time.  

This capability allows JARVIS to understand multi-screen workflows such as: 

**"Code on monitor 1, documentation on monitor 2, Slack on monitor 3."**

By integrating this foundation, subsequent vision layers (Multi-Space Vision Intelligence v2.0, Adaptive Context Routing, and Cognitive Awareness) can operate seamlessly across multiple displays. 

## Goals & Objectives 

| Goal | Description |
|------|-------------|
| G1 | Detect and track all connected monitors using macOS Core Graphics |
| G2 | Map spaces to corresponding displays via Yabai + CG APIs |
| G3 | Capture screenshots per-monitor and per-space for vision analysis |
| G4 | Provide display-aware summaries and context understanding |
| G5 | Enable user queries like "What's on my second monitor?" |

## Current Limitations 

| Issue | Impact |
|-------|--------|
| Single-monitor assumption | JARVIS only tracks space on one display |
| No spatial awareness | Cannot distinguish which monitor a space belongs to |
| Limited context analysis | Vision system reports incomplete activity summaries |

## Proposed Solution 

Introduce a new module, `MultiMonitorDetector`, as the foundation for multi-display intelligence.

### Core Component 

```python
# backend/vision/multi_monitor_detector.py

class MultiMonitorDetector:
    """Detect and track windows across multiple displays"""

    async def detect_displays(self) -> List[Display]:
        """Return all connected displays (ID, resolution, position, spaces)."""

    async def get_space_display_mapping(self) -> Dict[int, int]:
        """Map each Yabai space to its corresponding display."""

    async def capture_all_displays(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Return screenshots for each display and its active spaces."""
```

## Integration Points 

| Component | Update | Purpose |
|-----------|--------|---------|
| `backend/vision/intelligent_orchestrator.py` | Add display-aware workspace scouting | Enable orchestration per-monitor |
| `backend/vision/yabai_space_detector.py` | Extend to include display mapping | Integrate with Yabai CLI (`yabai -m query --displays`) |
| `backend/vision/multi_monitor_detector.py` (new) | Core detector module | Responsible for CG + Yabai integration |
| `backend/api/vision_api.py` | Add endpoint `/vision/displays` | Expose monitor + space data for UI |
| `frontend/mission_control_view.tsx` (future) | Display live monitor layout | Visual feedback for user context |

## User Stories 

| User Story | Description | Priority |
|------------|-------------|----------|
| US-1 | As a user, I want JARVIS to detect all my connected monitors | ⭐⭐⭐⭐⭐ |
| US-2 | As a user, I want to ask "What's on my second monitor?" | ⭐⭐⭐⭐ |
| US-3 | As a developer, I want to map which spaces belong to which displays | ⭐⭐⭐⭐ |
| US-4 | As an analyst, I want to capture screenshots per monitor for visual intelligence | ⭐⭐⭐⭐ |

## Technical Details

### Detection Layer 
- **macOS CoreGraphics API**
  - `Quartz.CGGetActiveDisplayList()`
  - `Quartz.CGDisplayBounds()`
  - `Quartz.CGDisplayIsMain()`
- **Yabai CLI Integration**
  - `yabai -m query --displays`
  - `yabai -m query --spaces`

### Screenshot Layer
- `Quartz.CGWindowListCreateImage(display_bounds, …)` per display
- Store screenshots as NumPy arrays for downstream vision models

### Data Structure  

```python
{
  display_id: {
    "resolution": (width, height),
    "position": (x, y),
    "is_primary": True/False,
    "spaces": {
      space_id: np.ndarray  # screenshot
    }
  }
}
```

## API & CLI Extensions 

| Interface | Description |
|-----------|-------------|
| `GET /vision/displays` | Returns JSON of connected displays, spaces, and resolutions |
| `CLI: jarvis --monitors` | CLI summary: "2 monitors detected – Primary 1920x1080 (Code), Secondary 2560x1440 (Research)" |

## Testing Plan 

| Test Type | Description |
|-----------|-------------|
| Unit Tests | Mock CG + Yabai responses for single and dual monitors |
| Integration Tests | Run on systems with 2–3; verify correct mapping |
| Functional Tests | Query "Show me all my displays" and validate verbal/JSON response |
| Regression Tests | Ensure single-monitor systems still behave correctly |

## Performance & Constraints
- **CPU Load**: Minimal (CG queries < 20ms)
- **Memory Usage**: ~30–50 MB per monitor (screenshots cached)
- **OS Dependency**: macOS only (for now)
- **Extensibility**: Future support for Windows via Win32 APIs and Linux via Xlib

## Example Responses 

**User:** "What's on my second monitor?"
**JARVIS:** "Monitor 2 (2560×1440) is displaying your documentation space – Chrome and Notion are active."

**User:** "Show me all my displays."
**JARVIS:** "You have 2 displays: Primary 1920×1080 (Code workspace), Secondary 2560×1440 (Research workspace)."

## Risks & Mitigations 

| Risk | Description | Mitigation |
|------|-------------|------------|
| CG API permission errors | macOS screen-recording permissions | Pre-check + user prompt |
| Yabai CLI unavailable | Missing dependency | Fallback to CoreGraphics only |
| Performance overhead | Multiple screenshots cause lag | Async capture + throttled refresh intervals |

## Deliverables 

| Deliverable | File | Description |
|-------------|------|-------------|
| Core Detector | `backend/vision/multi_monitor_detector.py` | New display + space detector |
| Integration | `intelligent_orchestrator.py` & `yabai_space_detector.py` | Add display awareness |
| Endpoint | `backend/api/vision_api.py` | `/vision/displays` API for frontend |
| Documentation | `docs/vision/multi_monitor_support.md` | Developer guide + CLI commands |
| Test Suite | `tests/test_multi_monitor_detector.py` | Unit & integration coverage |

## Timeline (Phase 1: Weeks 1–2)

| Week | Task | Owner | Status |
|------|------|-------|--------|
| Week 1 | Implement MultiMonitorDetector + Yabai integration | Derek | Pending |
| Week 2 | API exposure, frontend visualization, testing | Derek | Pending |

## Success Metrics 

| Metric | Target |
|--------|--------|
| Display detection accuracy | 100% |
| Space-to-display mapping accuracy | 95% |
| Screenshot latency | < 300 ms per monitor |
| Query response accuracy | 95% correct context |
| Zero impact on single-monitor systems | ✅ |

## Future Expansion (Phase 2+)
- **Cross-Display Attention Tracking** – track gaze or cursor across screens
- **Display-Context Memory** – remember what each monitor is used for
- **Multi-Display Vision Fusion** – aggregate context from all monitors
- **Virtual Monitor Emulation** – simulate additional spaces for testing

---

*Generated: 2025-01-14*
*Branch: multi-monitor-support*
