#!/bin/bash
#
# Clean Sensitive Data Script
# ==========================
# 
# This script removes any sensitive test files and logs that might contain passwords
# Run this before pushing to GitHub
#

echo "🧹 Cleaning sensitive data from Voice Unlock..."

# Remove test files that might contain passwords
echo "Removing test files..."
rm -f test_*.py debug_*.py fix_*.py *_test.py

# Clean up log files
echo "Removing log files..."
rm -f /tmp/websocket_*.log /tmp/daemon_*.log /tmp/*_debug.log
rm -f *.log

# Clean up any temporary password files
echo "Checking for password files..."
find . -name "*password*.txt" -o -name "*credentials*.txt" | while read file; do
    echo "  Removing: $file"
    rm -f "$file"
done

# Clean up build artifacts that might have debugging info
echo "Cleaning build artifacts..."
cd objc
make clean 2>/dev/null || true
rm -rf bin/ build/
cd ..

# Remove any cache files
echo "Removing cache files..."
rm -rf .jarvis/
rm -f *.cache

# Check if any files still contain sensitive patterns
echo -e "\n🔍 Checking for remaining sensitive data..."
FOUND_SENSITIVE=0

# Don't search binary files or logs
# Search for common password patterns without using the actual password
if grep -r -E "(password|Password|PASSWORD)\s*=\s*['\"][^'\"]{8,}" . --exclude="*.log" --exclude="*.dylib" --exclude="*.o" --exclude="*.sh" --exclude-dir="bin" --exclude-dir="build" 2>/dev/null | grep -v "password =" | grep -v "getpass" | grep -v "find-generic-password"; then
    echo -e "\n⚠️  WARNING: Found files containing hardcoded password patterns!"
    FOUND_SENSITIVE=1
fi

if [ $FOUND_SENSITIVE -eq 0 ]; then
    echo "✅ No sensitive data found in source files"
else
    echo -e "\n❌ Please remove sensitive data from the files listed above"
    exit 1
fi

echo -e "\n✅ Cleanup complete!"
echo "💡 Remember: Your password is securely stored in macOS Keychain, not in the code"