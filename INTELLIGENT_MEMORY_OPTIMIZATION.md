# 🧠 Intelligent Memory Optimization for JARVIS

## Overview

JARVIS now includes an intelligent memory optimization system that automatically frees up system memory to enable advanced features like LangChain when memory usage is too high.

## Features

### Automatic Memory Optimization
- **Smart Detection**: Automatically detects when memory is too high for LangChain mode (> 50%)
- **Progressive Strategies**: Applies memory optimization strategies in order of impact
- **Safe Operations**: Only targets non-critical processes and resources
- **Detailed Reporting**: Provides comprehensive reports of actions taken

### Optimization Strategies

1. **Python Memory Optimization**
   - Garbage collection (full GC cycle)
   - Cache clearing (functools, re)
   - Object cleanup

2. **System Cache Clearing** (macOS)
   - DNS cache flush
   - Memory pressure release
   - Inactive memory purging

3. **Helper Process Management**
   - Identifies killable helper processes
   - Targets high-memory helpers (> 0.5% memory)
   - Preserves critical system processes

4. **Background App Suspension**
   - Suspends non-critical apps (Slack, Discord, Spotify, etc.)
   - Releases ~70% of suspended app memory
   - Apps can be resumed later

5. **Browser Memory Optimization**
   - Closes excess browser tabs (keeps 5 most recent)
   - Targets Chrome, Safari, Firefox, Edge, Brave
   - Uses AppleScript for safe tab management

6. **Inactive Memory Purging**
   - Purges macOS inactive memory pages
   - Requires sudo access for maximum effectiveness

## Usage

### Automatic Optimization

When JARVIS tries to upgrade to LangChain mode but memory is too high:

```python
# DynamicChatbot automatically attempts optimization
User: "What is 2+2?"
JARVIS: [Attempting memory optimization...]
JARVIS: [Freed 2GB memory, upgraded to LangChain]
JARVIS: "4"
```

### Manual Optimization via API

```bash
# Trigger memory optimization
curl -X POST http://localhost:8000/chat/optimize-memory

# Response:
{
  "success": true,
  "initial_memory_percent": 74.6,
  "final_memory_percent": 48.2,
  "memory_freed_mb": 2156,
  "actions_taken": [
    {"strategy": "garbage_collection", "freed_mb": 125},
    {"strategy": "kill_helpers", "freed_mb": 1831},
    {"strategy": "clear_caches", "freed_mb": 200}
  ],
  "current_mode": "langchain",
  "can_use_langchain": true
}
```

### Command Line Testing

```bash
# Test intelligent optimization
cd backend
python test_intelligent_optimization.py

# Choose option 1 for optimization only
# Choose option 2 for full DynamicChatbot test
```

### Memory API Endpoints

```bash
# Get optimization suggestions
curl http://localhost:8000/memory/optimize/suggestions

# Optimize for LangChain
curl -X POST http://localhost:8000/memory/optimize/langchain
```

## Configuration

### Memory Thresholds

Default thresholds in `DynamicChatbot`:
- **LangChain Mode**: < 50% memory usage
- **Intelligent Mode**: < 65% memory usage  
- **Simple Mode**: > 80% memory usage

### Target Optimization

The optimizer targets 45% memory usage to provide buffer:
```python
self.target_memory_percent = 45  # 5% buffer below LangChain threshold
```

### Process Patterns

Killable processes:
- Helper, Renderer, GPU Process, Utility
- CrashPad, ReportCrash, mdworker
- WebKit services, Photo analysis
- Various Apple daemons

Protected processes:
- kernel, launchd, systemd, init
- WindowServer, loginwindow, Finder
- Dock, SystemUIServer
- python, node (your apps)

Suspendable apps:
- Slack, Discord, Spotify, Music
- TV, News, Stocks, Weather
- Reminders, Notes, Calendar

## Safety Features

1. **Progressive Application**: Strategies applied from least to most aggressive
2. **Process Protection**: Critical system processes never touched
3. **Atomic Operations**: Each strategy wrapped in try-catch
4. **Detailed Logging**: All actions logged for debugging
5. **Report Generation**: JSON reports saved to `~/.jarvis/memory_reports/`

## Metrics and Monitoring

The system tracks:
- `intelligent_optimizations`: Number of optimization attempts
- `optimization_successes`: Successful optimizations
- `memory_freed_mb`: Total memory freed per session
- `actions_taken`: Detailed log of each action

## Troubleshooting

### Optimization Not Working

1. **Check Current Memory**:
   ```bash
   python check_memory.py
   ```

2. **View Top Processes**:
   ```bash
   # Shows processes using > 1% memory
   python check_memory.py | grep -A 20 "Top Memory Users"
   ```

3. **Manual Process Management**:
   ```bash
   # Kill specific helper
   pkill -f "Cursor Helper"
   
   # Kill all Chrome helpers
   pkill -f "Chrome Helper"
   ```

### Sudo Access Required

Some optimizations (purge, cache clearing) work better with sudo:
```bash
# Grant temporary sudo access
sudo -v

# Then run optimization
python test_intelligent_optimization.py
```

### Memory Still High After Optimization

1. Close heavy applications manually:
   - IDEs (Cursor, VS Code, IntelliJ)
   - Browsers with many tabs
   - Docker Desktop
   - Virtual Machines

2. Check for memory leaks:
   ```bash
   # Monitor memory over time
   watch -n 5 'python check_memory.py | head -20'
   ```

## Integration with JARVIS

The intelligent memory optimizer is fully integrated:

1. **Automatic Triggering**: When upgrading to LangChain fails
2. **API Access**: Available via REST endpoints
3. **Metrics Tracking**: Integrated with DynamicChatbot metrics
4. **Logging**: Uses unified logging system

## Best Practices

1. **Regular Monitoring**: Check memory status periodically
2. **Proactive Management**: Close unused apps before they accumulate
3. **Buffer Maintenance**: Keep memory below 45% for best performance
4. **Report Analysis**: Review optimization reports for patterns

## Future Enhancements

- [ ] Machine learning for optimal strategy selection
- [ ] User preference learning
- [ ] Cross-platform support (Linux, Windows)
- [ ] Integration with system notifications
- [ ] Scheduled optimization tasks
- [ ] Memory usage predictions

## Example Workflow

```python
# 1. Memory too high for LangChain
Memory: 74.6% - Stuck in Simple mode

# 2. User asks complex question
User: "What's the weather in San Francisco?"
JARVIS: [generic response due to Simple mode]

# 3. Trigger optimization
curl -X POST http://localhost:8000/chat/optimize-memory

# 4. Memory freed
Freed 2.1GB - Memory now at 48.2%

# 5. JARVIS upgrades automatically
Mode switched to LangChain

# 6. User gets proper response
User: "What's the weather in San Francisco?"
JARVIS: "The current weather in San Francisco is 68°F with partly cloudy skies..."
```