# 🚀 JARVIS Startup Examples - Goal Inference Integration

## Quick Reference Guide

All these examples are now **integrated into start_system.py** and work automatically!

---

## 🎯 Method 1: Simple Startup Script (Recommended)

### Scenario 1: First Time User
**Goal**: Get JARVIS running with zero configuration

```bash
./start_jarvis.sh
```

**What happens:**
- ✅ Creates default `balanced` configuration automatically
- ✅ Initializes SQLite + ChromaDB databases
- ✅ Starts JARVIS with all 10 components
- ✅ Displays configuration summary on startup

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║                   🤖 JARVIS AI ASSISTANT                       ║
║              Goal Inference & Learning System                  ║
╚════════════════════════════════════════════════════════════════╝

Available Configuration Presets:
  [Shows preset options...]

🚀 Starting JARVIS...
✅ Goal Inference + Learning Database loaded
   • Goal Confidence: 0.75
   • Proactive Suggestions: True
   • Automation: False
   • Learning: True
```

---

### Scenario 2: Quick Learning Mode
**Goal**: Quickly teach JARVIS your patterns

```bash
./start_jarvis.sh learning
```

**What happens:**
- 🎯 Applies `learning` preset
- 🔥 Lowers pattern threshold to 2 (vs 3)
- 📈 Increases pattern boost to 0.10 (vs 0.05)
- 🚀 Faster adaptation to your routines

**Best for:**
- First week of using JARVIS
- Teaching new routines
- Onboarding new users

---

### Scenario 3: Aggressive + Automation
**Goal**: Maximum proactivity and automation

```bash
./start_jarvis.sh aggressive --enable-automation
```

Or shorter:
```bash
./start_jarvis.sh aggressive -a
```

**What happens:**
- ⚡ Lowers confidence thresholds (0.65 vs 0.75)
- 🤖 Enables automatic execution of high-confidence actions
- 🔥 Highly proactive suggestions
- ⚠️ **WARNING**: Will auto-execute actions with >95% confidence

**Best for:**
- Power users who trust JARVIS
- Highly repetitive workflows
- Maximum productivity

---

### Scenario 4: Conservative Mode
**Goal**: Only the most confident predictions

```bash
./start_jarvis.sh conservative
```

**What happens:**
- 🛡️ High confidence thresholds (0.85 vs 0.75)
- 🐌 Slower learning (0.02 boost vs 0.05)
- 📊 Only suggests when very certain
- ❌ Automation disabled by default

**Best for:**
- Important presentations
- High-stakes work
- Minimal interruptions needed

---

### Scenario 5: Maximum Performance
**Goal**: Fastest possible responses

```bash
./start_jarvis.sh performance
```

**What happens:**
- 💾 Larger cache (200 entries vs 100)
- ⏱️ Longer TTL (600s vs 300s)
- ⚡ Parallel processing enabled
- 🚀 Resource preloading enabled

**Best for:**
- Powerful machines (16GB+ RAM)
- Speed-critical workflows
- Heavy JARVIS usage

---

## 🎯 Method 2: Direct start_system.py (Advanced)

### Example 1: Basic with Preset
```bash
python start_system.py --goal-preset learning
```

### Example 2: Preset + Automation
```bash
python start_system.py --goal-preset aggressive --enable-automation
```

### Example 3: Multiple Flags
```bash
python start_system.py \
    --goal-preset learning \
    --enable-automation \
    --no-browser \
    --verbose
```

### Example 4: Backend Only + Goal Inference
```bash
python start_system.py \
    --backend-only \
    --goal-preset performance \
    --port 8000
```

---

## 🎯 Method 3: Environment Variables

### Scenario 4: Set Default Preset
**Goal**: Always use same preset without typing it

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export JARVIS_GOAL_PRESET=aggressive
export JARVIS_GOAL_AUTOMATION=true
```

Then just run:
```bash
python start_system.py
# or
./start_jarvis.sh
```

**What happens:**
- ✅ Automatically uses `aggressive` preset every time
- ✅ Automation enabled by default
- ✅ No need to specify flags

---

### Scenario 5: One-Time Override
**Goal**: Use different preset for this session only

```bash
JARVIS_GOAL_PRESET=learning python start_system.py
```

Or:
```bash
JARVIS_GOAL_PRESET=conservative JARVIS_GOAL_AUTOMATION=false python start_system.py
```

---

## 📊 All Command Options

### start_jarvis.sh Options

```bash
./start_jarvis.sh [preset] [automation_flag]
```

**Presets:**
- `aggressive` - Proactive learning and suggestions
- `balanced` - Default, recommended for most users
- `conservative` - High confidence required
- `learning` - Fast pattern learning
- `performance` - Maximum speed

**Automation Flags:**
- `--enable-automation` or `-a` - Enable auto-execution
- `--disable-automation` or `-d` - Suggestions only

---

### start_system.py Options

```bash
python start_system.py [flags]
```

**Goal Inference Flags:**
```bash
--goal-preset {aggressive|balanced|conservative|learning|performance}
--enable-automation    # Enable automatic actions
--disable-automation   # Disable automatic actions
```

**Standard Flags:**
```bash
--no-browser          # Don't open browser
--backend-only        # Start backend only
--frontend-only       # Start frontend only
--verbose             # Verbose logging
--debug               # Debug mode
--port 8000           # Backend port
--restart             # Restart JARVIS
```

---

## 🎓 Best Practices

### Week 1: Learning Mode
```bash
./start_jarvis.sh learning
```
- Let JARVIS observe your patterns
- Review suggestions but don't enable automation yet
- Check metrics weekly

### Week 2-4: Balanced with Selective Automation
```bash
./start_jarvis.sh balanced
```
- Switch to balanced mode
- Keep automation disabled
- JARVIS has learned your patterns now

### Month 2+: Aggressive or Custom
```bash
./start_jarvis.sh aggressive --enable-automation
```
- Enable automation for trusted actions
- JARVIS is now reliable and personalized
- Fine-tune with `configure_goal_inference.py` if needed

---

## 🔍 Verification

After starting JARVIS, check the startup logs:

```
✅ Goal Inference + Learning Database loaded
   • Goal Confidence: 0.65              ← From preset
   • Proactive Suggestions: True        ← Enabled
   • Automation: True                   ← From --enable-automation
   • Learning: True                     ← Always on
   • Database Cache: 200 entries        ← From performance preset
   • Previous session: 45 goals, 67 actions  ← Your history
   • Success rate: 89.2%                ← Learning effectiveness
```

---

## 🆘 Troubleshooting

### Issue: Wrong preset applied
**Solution:**
```bash
# Check environment variable
echo $JARVIS_GOAL_PRESET

# Clear it if set incorrectly
unset JARVIS_GOAL_PRESET

# Or override for this session
JARVIS_GOAL_PRESET=balanced ./start_jarvis.sh
```

### Issue: Automation enabled when I don't want it
**Solution:**
```bash
# Explicitly disable
./start_jarvis.sh balanced --disable-automation

# Or clear environment variable
unset JARVIS_GOAL_AUTOMATION
```

### Issue: Config not updating
**Solution:**
```bash
# Delete config to recreate with new settings
rm backend/config/integration_config.json

# Restart with desired preset
./start_jarvis.sh learning
```

---

## 📝 Quick Comparison

| Scenario | Command | Learning Speed | Suggestions | Automation | Best For |
|----------|---------|---------------|-------------|------------|----------|
| **First Time** | `./start_jarvis.sh` | Normal | Moderate | ❌ | Getting started |
| **Quick Learn** | `./start_jarvis.sh learning` | Fast ⚡ | Many | ❌ | First week |
| **Daily Use** | `./start_jarvis.sh balanced` | Normal | Balanced | ❌ | Most users |
| **Power User** | `./start_jarvis.sh aggressive -a` | Fast ⚡ | Many | ✅ | Trusted workflows |
| **Important Work** | `./start_jarvis.sh conservative` | Slow 🐌 | Few | ❌ | High stakes |
| **Speed Demon** | `./start_jarvis.sh performance` | Normal | Moderate | ❌ | Performance critical |

---

## 🎯 Summary

**Simplest way to start:**
```bash
./start_jarvis.sh
```

**Fastest learning:**
```bash
./start_jarvis.sh learning
```

**Maximum automation:**
```bash
./start_jarvis.sh aggressive --enable-automation
```

**Production ready:**
```bash
# Add to .bashrc/.zshrc
export JARVIS_GOAL_PRESET=balanced

# Then just:
python start_system.py
```

Everything is now **fully integrated** - no manual configuration needed! 🎉
