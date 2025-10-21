# AdaptiveControlCenterClicker - Quick Reference

**One-page reference for the most common operations**

---

## 🚀 Quick Start (30 seconds)

```python
from display.adaptive_control_center_clicker import get_adaptive_clicker

# Initialize
clicker = get_adaptive_clicker()

# Open Control Center
result = await clicker.open_control_center()

# Done! No configuration needed.
```

---

## 📋 Common Operations

### Open Control Center
```python
result = await clicker.open_control_center()
# Returns: ClickResult with coordinates, method, duration
```

### Click Screen Mirroring
```python
result = await clicker.click_screen_mirroring()
# Note: Control Center must be open first
```

### Connect to Device (Complete Flow)
```python
result = await clicker.connect_to_device("Living Room TV")
# Handles: Control Center → Screen Mirroring → Device
```

### Check Result
```python
if result.success:
    print(f"✅ Coordinates: {result.coordinates}")
    print(f"Method: {result.method_used}")
else:
    print(f"❌ Error: {result.error}")
```

---

## 🔧 Configuration

### Custom Settings
```python
from display.adaptive_control_center_clicker import AdaptiveControlCenterClicker

clicker = AdaptiveControlCenterClicker(
    vision_analyzer=my_vision_analyzer,  # Optional
    cache_ttl=86400,                      # 24 hours (default)
    enable_verification=True              # Enable verification (default)
)
```

### Cache Management
```python
# Clear cache (force re-detection)
clicker.clear_cache()

# Access cache directly
cached = clicker.cache.get("control_center")

# Invalidate specific entry
clicker.cache.invalidate("control_center")
```

---

## 📊 Metrics

### Get Performance Metrics
```python
metrics = clicker.get_metrics()

print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Cache hits: {metrics['cache_hit_rate']:.1%}")
print(f"Methods used: {metrics['method_usage']}")
```

### Key Metrics
- `total_attempts` - Total clicks attempted
- `success_rate` - % successful clicks
- `cache_hit_rate` - % using cached coordinates
- `fallback_uses` - # times fallback chain used
- `method_usage` - Dict of method usage counts

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/unit/display/test_adaptive_control_center_clicker.py -v
```

### Run Integration Tests
```bash
export JARVIS_INTEGRATION_TESTS=1
pytest tests/integration/test_adaptive_clicker_integration.py -v
```

### Run Verification Script
```bash
# Quick verification
python backend/display/verify_adaptive_clicker.py --quick

# Full verification
python backend/display/verify_adaptive_clicker.py --full
```

---

## 🐛 Troubleshooting

### Issue: Detection Failed
**Solution:**
```bash
# 1. Install OCR
brew install tesseract
pip install pytesseract

# 2. Run verification
python backend/display/verify_adaptive_clicker.py --full

# 3. Check logs for details
```

### Issue: Cache Not Working
**Solution:**
```python
# Check cache location
print(clicker.cache.cache_file)
# Default: ~/.jarvis/control_center_cache.json

# Clear and retry
clicker.clear_cache()
```

### Issue: Slow Performance
**Solution:**
```python
# Check cache hit rate
metrics = clicker.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

# Should be >90% after warmup
# If low, cache may not be persisting
```

---

## 📁 File Locations

| File | Location |
|------|----------|
| Main code | `backend/display/adaptive_control_center_clicker.py` |
| Unit tests | `tests/unit/display/test_adaptive_control_center_clicker.py` |
| Integration tests | `tests/integration/test_adaptive_clicker_integration.py` |
| Verification script | `backend/display/verify_adaptive_clicker.py` |
| Examples | `backend/display/example_adaptive_clicker.py` |
| Cache file | `~/.jarvis/control_center_cache.json` |
| Full docs | `backend/display/ADAPTIVE_CLICKER_README.md` |

---

## 🎯 Detection Methods (Fallback Chain)

| Priority | Method | Speed | Success Rate | Best For |
|----------|--------|-------|--------------|----------|
| 1 | Cached | 10ms | 95%+ | Repeat operations |
| 2 | OCR (Tesseract) | 500ms | 85% | Text elements |
| 2 | OCR (Claude) | 1-2s | 95% | Complex UI |
| 3 | Template Match | 300ms | 80% | Exact matches |
| 4 | Edge Detection | 400ms | 70% | Shapes |
| 5 | Accessibility API | - | - | Future |
| 6 | AppleScript | - | - | Future |

---

## 💡 Pro Tips

### Tip 1: Let It Learn
```python
# First run is slower (detection)
# Subsequent runs are instant (cache)
# Don't clear cache unless necessary!
```

### Tip 2: Check Metrics Regularly
```python
# Monitor cache hit rate
# Should be >90% after warmup
metrics = clicker.get_metrics()
if metrics['cache_hit_rate'] < 0.9:
    print("⚠️ Low cache hit rate - investigate")
```

### Tip 3: Use Verification in Production
```python
# Always enable verification in production
clicker = get_adaptive_clicker(enable_verification=True)
# Catches false positives automatically
```

### Tip 4: Clean Up UI State
```python
# Always clean up after operations
import pyautogui

await clicker.open_control_center()
# ... do work ...
pyautogui.press('escape')  # Close menus
```

---

## 🔗 Quick Links

- **Full Documentation:** `ADAPTIVE_CLICKER_README.md`
- **Examples:** `example_adaptive_clicker.py`
- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Verification Script:** `verify_adaptive_clicker.py`

---

## 📞 Getting Help

1. Check `ADAPTIVE_CLICKER_README.md` troubleshooting section
2. Run `verify_adaptive_clicker.py --full` for diagnostics
3. Check logs for detailed error messages
4. Review metrics for performance insights

---

## ✅ Checklist for Production Use

Before deploying to production:

- [ ] Run unit tests: `pytest tests/unit/display/...`
- [ ] Run integration tests: `pytest tests/integration/...`
- [ ] Run verification: `verify_adaptive_clicker.py --full`
- [ ] Verify cache directory exists: `~/.jarvis/`
- [ ] Check screen recording permissions granted
- [ ] Test on target macOS version
- [ ] Monitor metrics after deployment

---

**That's it! You're ready to use AdaptiveControlCenterClicker. 🚀**

For detailed documentation, see `ADAPTIVE_CLICKER_README.md`
