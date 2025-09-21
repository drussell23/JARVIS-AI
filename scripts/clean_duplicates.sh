#!/bin/bash
# Script to clean up duplicate files in the repository

echo "🧹 Cleaning up duplicate files..."
echo "================================"

# Counter for removed files
removed_count=0

# Function to find and remove duplicates
clean_duplicates() {
    local pattern="$1"
    local description="$2"
    
    echo -n "Checking for $description... "
    
    # Find files matching the pattern
    local files=$(find . -type f \( -name "$pattern" \) -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./venv/*" -not -path "./.venv/*" 2>/dev/null)
    
    if [ -z "$files" ]; then
        echo "✅ None found"
    else
        local count=$(echo "$files" | wc -l | tr -d ' ')
        echo "❌ Found $count files"
        echo "$files" | while read -r file; do
            echo "  Removing: $file"
            rm -f "$file"
            ((removed_count++))
        done
    fi
}

# Clean various duplicate patterns
clean_duplicates "* 2.*" "files with ' 2.'"
clean_duplicates "* 2" "files ending with ' 2'"
clean_duplicates "* copy.*" "files with ' copy.'"
clean_duplicates "* copy" "files ending with ' copy'"
clean_duplicates "*-copy.*" "files with '-copy.'"
clean_duplicates "*-copy" "files ending with '-copy'"
clean_duplicates "*(2).*" "files with '(2).'"
clean_duplicates "*(2)" "files ending with '(2)'"

# Also clean common editor temporary files
clean_duplicates "*~" "editor backup files"
clean_duplicates "*.swp" "vim swap files"
clean_duplicates ".*.swp" "hidden vim swap files"

echo ""
echo "================================"
echo "✅ Cleanup complete!"
echo ""

# Check git status
echo "📊 Git status after cleanup:"
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || true
git status --porcelain | grep -E '( 2\.|\ 2$| copy\.| copy$|-copy\.|-copy$|\(2\)\.|\(2\)$)' | wc -l | xargs echo "Duplicate files still tracked by git:"

echo ""
echo "💡 Tips to prevent duplicates:"
echo "1. The .gitignore has been updated to ignore duplicate files"
echo "2. A pre-commit hook has been installed to prevent committing duplicates"
echo "3. Use 'git mv' instead of copying files when you need to rename"
echo "4. Be careful when using 'Save As' in editors - use meaningful names"
echo ""
echo "To test the pre-commit hook, try:"
echo "  touch 'test 2.txt' && git add 'test 2.txt' && git commit -m 'test'"