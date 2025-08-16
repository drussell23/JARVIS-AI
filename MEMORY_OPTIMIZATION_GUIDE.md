# JARVIS Memory Optimization Guide

## Quick Reference

### Check Memory Status
```bash
# Check current memory usage
python start_system.py --memory-status

# Via API (if running)
curl http://localhost:8000/memory/status
```

### Optimize Memory

#### 1. Standard Optimization (Safe)
```bash
# Via API
curl -X POST http://localhost:8000/chat/optimize-memory

# During startup
python start_system.py --optimize-memory
```

#### 2. Aggressive Optimization (Closes Apps)
```bash
# Via API with aggressive mode
curl -X POST http://localhost:8000/chat/optimize-memory \
  -H "Content-Type: application/json" \
  -d '{"aggressive": true}'

# Force LangChain mode at startup
python start_system.py --force-langchain
```

#### 3. Interactive Optimization
```bash
# Run the interactive optimizer
python optimize_memory_advanced.py --interactive

# With specific target
python optimize_memory_advanced.py --target 40
```

## Memory Thresholds

| Memory Usage | Available Features | Recommendation |
|--------------|-------------------|----------------|
| < 45% | Full LangChain + All Features | Optimal |
| 45-50% | LangChain (limited) | Good |
| 50-65% | Intelligent Mode Only | Run optimization |
| 65-80% | Basic Mode Only | Strongly recommend optimization |
| > 80% | Emergency Mode | Critical - must optimize |

## Optimization Strategies

### What Gets Optimized

1. **Safe Optimizations** (Always Run):
   - Python garbage collection
   - Clear system caches
   - Kill helper processes
   - Memory compression

2. **Standard Optimizations** (Memory > 65%):
   - Close browser tabs
   - Suspend background apps
   - Kill non-essential services

3. **Aggressive Optimizations** (Manual Approval):
   - Close IDEs (Cursor, VSCode, IntelliJ)
   - Close messaging apps (WhatsApp, Slack)
   - Close browsers entirely
   - Suspend Docker/VMs

## Common Scenarios

### Enable LangChain Mode
```bash
# Option 1: Force at startup
python start_system.py --force-langchain

# Option 2: Optimize while running
curl -X POST http://localhost:8000/chat/optimize-memory -d '{"aggressive": true}'

# Option 3: Interactive
python optimize_memory_advanced.py --interactive
# Select option 2 (Aggressive optimization)
```

### Quick Memory Free
```bash
# Kill common memory hogs
pkill -f "Cursor Helper"
pkill -f "Chrome Helper"
pkill -f "Code Helper"

# Then run optimization
python optimize_memory_advanced.py --aggressive
```

### Check What's Using Memory
```bash
# Top memory users
ps aux | sort -nrk 4 | head -10

# JARVIS memory report
curl http://localhost:8000/memory/report
```

## Tips

1. **Best Time to Optimize**: Before starting JARVIS
2. **Persistent Settings**: Set `PREFER_LANGCHAIN=1` in your shell profile
3. **Auto-Optimize**: Add `--optimize-memory` to your JARVIS startup alias

## Troubleshooting

### Optimization Not Working?
1. Check for protected processes: `ps aux | grep -E "python|node|uvicorn"`
2. Manually close high-memory apps
3. Try aggressive mode: `{"aggressive": true}`

### Can't Reach Target?
- Target is 45% for LangChain
- Close IDEs and browsers manually
- Check for memory leaks: `sudo leaks <pid>`

### API Not Responding?
```bash
# Check if running
curl http://localhost:8000/health

# Use standalone script instead
python optimize_memory_advanced.py --aggressive
```