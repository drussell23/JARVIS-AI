# 🚀 SAI Quick Start Guide

Get Situational Awareness Intelligence running in **5 minutes**.

---

## Installation

SAI is included in JARVIS. No additional installation needed.

```bash
# Ensure dependencies are installed
cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend
pip install -r requirements.txt
```

---

## Minimal Example

```python
#!/usr/bin/env python3
import asyncio
from backend.vision.situational_awareness import get_sai_engine

async def main():
    # Initialize SAI with default settings
    engine = get_sai_engine()

    # Start monitoring
    await engine.start_monitoring()
    print("✅ SAI monitoring started")

    # Let it run for 60 seconds
    await asyncio.sleep(60)

    # Stop monitoring
    await engine.stop_monitoring()
    print("✅ SAI monitoring stopped")

    # Show metrics
    metrics = engine.get_metrics()
    print(f"📊 Changes detected: {metrics['changes']['total_detected']}")
    print(f"📊 Cache hit rate: {metrics['cache']['hit_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Save as:** `test_sai_basic.py`

**Run:**
```bash
python test_sai_basic.py
```

---

## SAI-Enhanced Control Center Clicker

```python
#!/usr/bin/env python3
import asyncio
from backend.display.sai_enhanced_control_center_clicker import get_sai_clicker

async def main():
    # Create SAI-enhanced clicker
    async with get_sai_clicker(enable_sai=True) as clicker:
        print("✅ SAI-enhanced clicker ready")

        # Click Control Center
        result = await clicker.click("control_center")

        print(f"✅ Success: {result.success}")
        print(f"📍 Method: {result.method_used}")
        print(f"✅ Verified: {result.verification_passed}")

        # Show SAI metrics
        metrics = clicker.get_metrics()
        print(f"\n📊 SAI Stats:")
        print(f"  Environment changes: {metrics['sai']['environment_changes_detected']}")
        print(f"  Cache invalidations: {metrics['sai']['proactive_cache_invalidations']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Save as:** `test_sai_clicker.py`

**Run:**
```bash
python test_sai_clicker.py
```

---

## Track Custom Element

```python
#!/usr/bin/env python3
import asyncio
from backend.vision.situational_awareness import (
    get_sai_engine,
    UIElementDescriptor,
    ElementType
)

async def main():
    engine = get_sai_engine()

    # Register custom element
    engine.tracker.add_custom_element(UIElementDescriptor(
        element_id="my_app",
        element_type=ElementType.DOCK_ICON,
        display_characteristics={
            'icon_description': 'Your app icon description',
            'location': 'Dock',
            'app_name': 'MyApp'
        }
    ))

    print("✅ Custom element registered")

    # Detect position
    position = await engine.get_element_position("my_app", use_cache=False)

    if position:
        print(f"✅ Found at: {position.coordinates}")
        print(f"📊 Confidence: {position.confidence:.2%}")
    else:
        print("❌ Not found")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Monitor Environment Changes

```python
#!/usr/bin/env python3
import asyncio
from backend.vision.situational_awareness import (
    get_sai_engine,
    ChangeEvent,
    ChangeType
)

async def main():
    engine = get_sai_engine()

    # Register callback
    def on_change(change: ChangeEvent):
        print(f"\n🔔 Change detected!")
        print(f"  Type: {change.change_type.value}")
        print(f"  Element: {change.element_id}")
        print(f"  Old: {change.old_value}")
        print(f"  New: {change.new_value}")

    engine.register_change_callback(on_change)

    # Start monitoring
    await engine.start_monitoring()
    print("👀 Watching for changes...")
    print("   Try moving menu bar icons or changing displays\n")

    # Monitor for 2 minutes
    await asyncio.sleep(120)

    await engine.stop_monitoring()
    print("\n✅ Monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Common Patterns

### Pattern: Auto-Recovery

```python
async with get_sai_clicker(enable_sai=True) as clicker:
    # SAI automatically:
    # 1. Detects UI changes
    # 2. Invalidates stale coordinates
    # 3. Re-detects new positions
    # 4. Updates cache
    # → Clicks ALWAYS work, even after macOS updates!

    result = await clicker.click("control_center")
```

### Pattern: Multi-Step Automation

```python
async with get_sai_clicker(enable_sai=True) as clicker:
    # Connect to AirPlay device
    result = await clicker.connect_to_device("Living Room TV")

    # SAI monitors throughout entire flow
    # Adapts if menu bar shifts mid-automation
    print(f"✅ Connected: {result['success']}")
```

### Pattern: Continuous Monitoring

```python
engine = get_sai_engine(monitoring_interval=5.0)  # Check every 5s

await engine.start_monitoring()

# SAI runs in background, keeping cache fresh
# Your app continues normally

# Later...
await engine.stop_monitoring()
```

---

## Next Steps

1. **Read full documentation:** [README.md](./README.md)
2. **Run comprehensive tests:** `pytest backend/vision/situational_awareness/tests/ -v`
3. **Integrate with your automation:** See [README.md](./README.md) for advanced patterns
4. **Monitor performance:** Use `engine.get_metrics()` for insights

---

## Troubleshooting

### SAI not detecting changes

```python
# Enable debug logging
import logging
logging.getLogger('backend.vision.situational_awareness').setLevel(logging.DEBUG)

# Check environment hash stability
old_hash = engine.current_snapshot.environment_hash
await asyncio.sleep(10)
new_hash = (await engine._capture_environment_snapshot()).environment_hash
print(f"Hash changed: {old_hash != new_hash}")
```

### Cache always missing

```python
# Check cache metrics
metrics = engine.cache.get_metrics()
print(f"Hit rate: {metrics['hit_rate']:.1%}")
print(f"Invalidations: {metrics['invalidations']}")

# Increase TTL
engine.cache.default_ttl = 86400  # 24 hours
```

### Vision detection slow

```python
# Use cached positions when possible
position = await engine.get_element_position(
    "control_center",
    use_cache=True,  # Use cache
    force_detect=False  # Don't force vision
)

# Cache hit = instant (< 1ms)
# Cache miss = vision detection (500-2000ms)
```

---

## Performance Tips

1. **Use appropriate monitoring intervals**
   ```python
   # Fast monitoring (battery intensive)
   engine = get_sai_engine(monitoring_interval=5.0)

   # Balanced (recommended)
   engine = get_sai_engine(monitoring_interval=10.0)

   # Slow monitoring (battery friendly)
   engine = get_sai_engine(monitoring_interval=30.0)
   ```

2. **Enable caching**
   ```python
   # Always use cache for repeated operations
   position = await engine.get_element_position(
       "control_center",
       use_cache=True  # ← Important!
   )
   ```

3. **Use context managers**
   ```python
   # Automatically starts/stops SAI
   async with get_sai_clicker(enable_sai=True) as clicker:
       # SAI active here
       ...
   # SAI automatically stopped
   ```

---

## Support

- **Documentation:** [README.md](./README.md)
- **Tests:** `backend/vision/situational_awareness/tests/`
- **Issues:** GitHub Issues

---

**Built with ❤️ for JARVIS**
