#!/bin/bash
echo "🚨 Emergency Memory Cleanup for JARVIS"
echo "Current memory usage: $(python3 -c 'import psutil; print(f"{psutil.virtual_memory().percent:.1f}%")')"

echo ""
echo "⚠️  This will close memory-heavy applications!"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "📱 Closing memory-heavy applications..."

# Close browsers
killall "Google Chrome" 2>/dev/null && echo "   ✓ Closed Google Chrome"
killall "Safari" 2>/dev/null && echo "   ✓ Closed Safari"
killall "Firefox" 2>/dev/null && echo "   ✓ Closed Firefox"

# Close communication apps
killall "Slack" 2>/dev/null && echo "   ✓ Closed Slack"
killall "Discord" 2>/dev/null && echo "   ✓ Closed Discord"
killall "Microsoft Teams" 2>/dev/null && echo "   ✓ Closed Microsoft Teams"
killall "zoom.us" 2>/dev/null && echo "   ✓ Closed Zoom"

# Close entertainment apps
killall "Spotify" 2>/dev/null && echo "   ✓ Closed Spotify"
killall "Music" 2>/dev/null && echo "   ✓ Closed Apple Music"

# Close development tools
killall "Cursor" 2>/dev/null && echo "   ✓ Closed Cursor"
killall "Visual Studio Code" 2>/dev/null && echo "   ✓ Closed VS Code"
killall "IntelliJ IDEA" 2>/dev/null && echo "   ✓ Closed IntelliJ IDEA"

# Close other memory-heavy apps
killall "WhatsApp" 2>/dev/null && echo "   ✓ Closed WhatsApp"
killall "Docker Desktop" 2>/dev/null && echo "   ✓ Closed Docker Desktop"
killall "Parallels Desktop" 2>/dev/null && echo "   ✓ Closed Parallels Desktop"

# Wait for apps to close
sleep 3

echo ""
echo "🧹 Clearing system caches..."

# Clear DNS cache (doesn't require sudo)
dscacheutil -flushcache 2>/dev/null && echo "   ✓ Flushed DNS cache"

# Clear memory pressure (if available)
if command -v memory_pressure &> /dev/null; then
    memory_pressure -l warn 2>/dev/null && echo "   ✓ Triggered memory pressure"
fi

# Check if we have sudo access for more aggressive cleanup
if sudo -n true 2>/dev/null; then
    echo ""
    echo "🔧 Running privileged cleanup..."
    
    # Purge memory (macOS specific)
    sudo purge 2>/dev/null && echo "   ✓ Purged inactive memory"
    
    # Clear system caches
    sudo rm -rf /Library/Caches/* 2>/dev/null
    sudo rm -rf ~/Library/Caches/* 2>/dev/null
    echo "   ✓ Cleared system caches"
else
    echo ""
    echo "ℹ️  Run with sudo for more aggressive cleanup:"
    echo "   sudo ./emergency_memory_cleanup.sh"
fi

# Force garbage collection in Python
echo ""
echo "🐍 Running Python memory cleanup..."
python3 -c "
import gc
import psutil

before = psutil.virtual_memory().percent
gc.collect(2)
after = psutil.virtual_memory().percent

print(f'   ✓ Python garbage collection (saved {before-after:.1f}%)')
" 2>/dev/null || echo "   ⚠️  Python cleanup skipped"

# Wait for memory to settle
sleep 2

# Show final memory usage
echo ""
echo "📊 Memory cleanup complete!"
echo "Final memory usage: $(python3 -c 'import psutil; print(f"{psutil.virtual_memory().percent:.1f}%")')"

# Check if memory is suitable for LangChain
python3 -c "
import psutil
mem = psutil.virtual_memory().percent
if mem < 50:
    print('\n✅ Memory is now suitable for LangChain mode!')
elif mem < 65:
    print('\n⚠️  Memory is suitable for Intelligent mode only')
else:
    print('\n❌ Memory is still high. Consider restarting your Mac.')
"

echo ""
echo "💡 Additional tips:"
echo "   - Close any remaining browser tabs"
echo "   - Quit unused applications from the dock"
echo "   - Consider restarting your Mac if memory remains high"
echo ""