#!/bin/bash
# Enable JARVIS Goal Inference Automation

# Option 1: Add to your shell configuration for permanent settings
echo "
# JARVIS Goal Inference Settings
export JARVIS_GOAL_PRESET=balanced      # or 'aggressive' for more proactive behavior
export JARVIS_GOAL_AUTOMATION=true      # Enable automatic actions
" >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

echo "✅ JARVIS automation enabled permanently!"
echo "Settings:"
echo "  - Preset: balanced"
echo "  - Automation: enabled"
echo ""
echo "To use different settings, you can override:"
echo "  export JARVIS_GOAL_PRESET=aggressive  # More proactive"
echo "  export JARVIS_GOAL_PRESET=conservative  # Less proactive"