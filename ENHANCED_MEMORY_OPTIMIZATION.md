# 🚀 Enhanced Memory Optimization for LangChain

## Overview

The enhanced memory optimization system now intelligently targets specific high-memory applications to enable LangChain features. This addresses the exact scenario where Cursor, Chrome, and WhatsApp are preventing JARVIS from using its advanced capabilities.

## Key Enhancements

### 1. **Targeted App Profiles**

The system now includes profiles for common memory-hungry applications:

```python
# High Priority Apps (closed first)
- Cursor (Priority: 9) - Closes if using > 5% memory
- VS Code (Priority: 9) - Closes if using > 5% memory  
- IntelliJ IDEA (Priority: 9) - Closes if using > 5% memory
- Docker Desktop (Priority: 8) - Closes if using > 3% memory
- WhatsApp (Priority: 8) - Closes if using > 1% memory
- Slack (Priority: 7) - Closes if using > 2% memory
- Discord (Priority: 7) - Closes if using > 2% memory
```

### 2. **Graceful vs Force Close**

- **IDEs (Cursor, VS Code)**: Attempts graceful close with "save work" first
- **Messaging Apps**: Direct termination (no unsaved work)
- **Browsers**: Closes tabs first, then whole browser if needed

### 3. **Configuration-Based System**

New `optimization_config.py` allows customization:
- Adjust memory thresholds
- Add/remove target applications
- Change priorities
- Protect specific processes

### 4. **Two-Stage Optimization**

1. **Standard Mode**: Tries less aggressive strategies first
2. **Aggressive Mode**: Automatically kicks in if memory > 60% after standard optimization

## Usage

### Quick Test

```bash
# Check what would be optimized
python test_langchain_optimization.py

# Choose option 3 to see current status
# Choose option 1 to run optimization
```

### API Endpoint

```bash
# Trigger optimization via API
curl -X POST http://localhost:8000/chat/optimize-memory

# Returns detailed report of what was closed
```

### Automatic Optimization

When JARVIS tries to upgrade to LangChain mode:
1. First attempts standard optimization
2. If memory still > 60%, tries aggressive mode
3. Closes apps based on priority and memory usage
4. Retries LangChain activation

## Example Scenario

**Before Optimization:**
```
Memory: 73.1%
- Cursor Helper: 9.5%
- Chrome: 7.3% (multiple processes)
- WhatsApp: 1.4%
```

**Optimization Process:**
1. Kills Chrome/Cursor helper processes
2. Gracefully closes Cursor IDE (saves work)
3. Terminates WhatsApp
4. Clears system caches

**After Optimization:**
```
Memory: 45.2%
✅ LangChain mode enabled!
```

## Configuration

Edit `backend/memory/optimization_config.py` to customize:

```python
# Add a new app to target
AppProfile(
    name="MyApp",
    patterns=["myapp", "myapp helper"],
    priority=8,  # 1-10, higher = close first
    graceful_close=True,
    min_memory_percent=2.0  # Only close if using > 2%
)
```

## Safety Features

### Protected Processes
Never closes:
- System processes (kernel, launchd, Finder)
- Python/Node (our servers)
- FastAPI/Uvicorn (JARVIS backend)

### Graceful Handling
- IDEs: Attempts to save work before closing
- Browsers: Warns before closing
- Critical apps: Requires confirmation

### Memory Verification
- Checks memory after each action
- Stops when target reached
- Reports exactly what was freed

## Troubleshooting

### Apps Not Closing

Some apps may resist termination:
```bash
# Force kill specific app
pkill -9 -f "Cursor"

# Or use Activity Monitor (macOS)
```

### Memory Still High After Optimization

1. Check for hidden memory users:
   ```bash
   python check_memory.py
   ```

2. Look for system processes:
   - kernel_task (normal, don't close)
   - WindowServer (graphics, don't close)
   - mds_stores (Spotlight indexing)

3. Restart if needed:
   ```bash
   sudo purge  # Clear all caches
   ```

## Best Practices

1. **Save Work First**: Before running optimization, save important work
2. **Close Manually When Possible**: Better than force closing
3. **Regular Monitoring**: Use `--memory-status` flag on startup
4. **Preventive Measures**: 
   - Limit browser tabs
   - Close unused IDEs
   - Quit chat apps when not needed

## Integration with JARVIS

The optimization is fully integrated:

1. **Automatic Trigger**: When upgrading to LangChain fails
2. **Smart Decisions**: Only closes what's necessary
3. **Progressive Strategy**: Tries gentle methods first
4. **User Control**: Can be triggered manually anytime

## Future Enhancements

- [ ] Machine learning to predict memory usage patterns
- [ ] Schedule optimization during low-activity periods
- [ ] Integration with macOS memory pressure API
- [ ] Custom profiles per user
- [ ] Memory usage notifications