# 🚀 JARVIS Quick Start Guide

## The Simple Way: Just Run It! ✨

```bash
python start_system.py
```

**That's it!** An interactive menu will guide you through:
1. 🎯 Selecting a Goal Inference preset (or press Enter for default)
2. ⚙️ Enabling/disabling automation (smart defaults)
3. 🚀 Starting JARVIS with your configuration

---

## Interactive Menu Preview

```
╔════════════════════════════════════════════════════════════════╗
║                   🤖 JARVIS AI ASSISTANT                       ║
║              Goal Inference & Learning System                  ║
╚════════════════════════════════════════════════════════════════╝

Available Configuration Presets:

  1. aggressive   - Highly proactive, learns quickly, suggests often
  2. balanced     - Default balanced settings (recommended)
  3. conservative - Cautious, requires high confidence
  4. learning     - Optimized for learning your patterns quickly
  5. performance  - Maximum speed, aggressive caching
  6. skip         - Use existing/default configuration

Select preset [1-6] (or press Enter for default 'balanced'):
```

---

## Skip the Menu (Advanced Users)

If you already know which preset you want:

```bash
# Specific preset
python start_system.py --goal-preset learning

# With automation enabled
python start_system.py --goal-preset aggressive --enable-automation

# With other flags
python start_system.py --goal-preset balanced --no-browser --verbose
```

---

## Set Permanent Defaults

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export JARVIS_GOAL_PRESET=balanced
export JARVIS_GOAL_AUTOMATION=false
```

Then just run:
```bash
python start_system.py
```

---

## Preset Quick Reference

| Preset | Best For | Confidence | Learning Speed | Automation Default |
|--------|----------|------------|----------------|-------------------|
| **aggressive** | Power users, quick learning | 0.65 | Fast ⚡ | ✅ ON |
| **balanced** | Daily use (default) | 0.75 | Normal | ❌ OFF |
| **conservative** | High-stakes work | 0.85 | Slow 🐌 | ❌ OFF |
| **learning** | First week usage | 0.75 | Fast ⚡ | ❌ OFF |
| **performance** | Speed-critical | 0.75 | Normal | ❌ OFF |

---

## What Gets Started?

JARVIS includes **10 core components** + **6 intelligent systems**:

### Core Components
1. **CHATBOTS** - Claude Vision AI
2. **VISION** - Multi-Space Desktop Monitoring
3. **MEMORY** - M1-optimized memory management
4. **VOICE** - Voice activation ("Hey JARVIS")
5. **ML_MODELS** - NLP and sentiment analysis
6. **MONITORING** - System health tracking
7. **VOICE_UNLOCK** - Advanced Screen Unlock
8. **WAKE_WORD** - Hands-free activation
9. **DISPLAY_MONITOR** - External Display Management
10. **GOAL_INFERENCE** - ML-Powered Goal Understanding ⭐

### Intelligent Systems v2.0
1. TemporalQueryHandler v3.0
2. ErrorRecoveryManager v2.0
3. StateIntelligence v2.0
4. StateDetectionPipeline v2.0
5. ComplexComplexityHandler v2.0
6. PredictiveQueryHandler v2.0

---

## Need More Details?

- **Full Usage Examples**: See [STARTUP_EXAMPLES.md](STARTUP_EXAMPLES.md)
- **Goal Inference Guide**: See [GOAL_INFERENCE_GUIDE.md](GOAL_INFERENCE_GUIDE.md)
- **Configuration**: Run `python configure_goal_inference.py --interactive`

---

## First Time Setup

1. **Run JARVIS**:
   ```bash
   python start_system.py
   ```

2. **Select "learning" preset** (Option 4)
   - This helps JARVIS learn your patterns quickly

3. **Keep automation OFF** for the first week
   - Review suggestions to build trust

4. **After 1-2 weeks**, switch to "balanced":
   ```bash
   python start_system.py --goal-preset balanced
   ```

5. **Enable automation** when ready:
   ```bash
   python start_system.py --goal-preset aggressive --enable-automation
   ```

---

## All Methods in One Place

### Method 1: Interactive Menu (Recommended)
```bash
python start_system.py
```
✅ **Shows menu** → Select preset → Choose automation

### Method 2: Command-Line Arguments
```bash
python start_system.py --goal-preset learning --enable-automation
```
✅ **Skips menu** → Uses specified preset

### Method 3: Environment Variables
```bash
export JARVIS_GOAL_PRESET=balanced
python start_system.py
```
✅ **Skips menu** → Uses environment variable

---

## Everything is Unified! 🎉

No more multiple scripts or manual configuration files!

- ✅ One command: `python start_system.py`
- ✅ Interactive menu when needed
- ✅ Skip menu with `--goal-preset`
- ✅ Auto-configuration on first run
- ✅ Environment variable support
- ✅ All features in one place

**Simple. Powerful. Unified.** 🚀
