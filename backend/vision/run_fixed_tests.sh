#!/bin/bash
# Quick test runner for fixed Claude Vision Analyzer tests

echo "🚀 Running Claude Vision Analyzer Fixed Tests"
echo "============================================"
echo ""

# Set memory safety environment
export VISION_MEMORY_SAFETY=true
export VISION_PROCESS_LIMIT_MB=2048
export VISION_MIN_SYSTEM_RAM_GB=2.0
export VISION_REJECT_ON_MEMORY=true

# Run the main test suite
echo "📊 Running main test runner..."
python3 run_vision_tests.py --continue-on-failure

# Check if we should run additional tests
if [ "$1" == "--full" ]; then
    echo ""
    echo "🔍 Running additional test files..."
    
    # Run any additional test files that exist
    for test_file in test_*.py; do
        if [ -f "$test_file" ] && [[ ! " test_enhanced_vision_integration.py test_memory_quick.py test_real_world_scenarios.py " =~ " $test_file " ]]; then
            echo ""
            echo "Running: $test_file"
            python3 "$test_file" || echo "⚠️  $test_file had issues"
        fi
    done
fi

echo ""
echo "✅ Test run complete!"
echo ""
echo "📝 For detailed testing guide, see: TESTING_GUIDE.md"