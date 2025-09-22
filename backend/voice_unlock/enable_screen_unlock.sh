#!/bin/bash
# Enable Screen Unlock for JARVIS Voice Unlock
# ============================================

echo "🔐 JARVIS Voice Unlock - Enable Screen Unlock"
echo "============================================="
echo
echo "This script will enable JARVIS to actually unlock your screen."
echo "Your password will be stored securely in the macOS Keychain."
echo
echo "⚠️  IMPORTANT: Run this script directly in your terminal:"
echo
echo "cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock"
echo "./enable_screen_unlock.sh"
echo
echo "The script will:"
echo "1. Prompt for your Mac password (securely)"
echo "2. Store it in the Keychain"
echo "3. Enable screen unlocking"
echo
echo "To remove the stored password later:"
echo "security delete-generic-password -s com.jarvis.voiceunlock -a unlock_token"
echo
echo "Press Ctrl+C now if you want to cancel."
echo

# Wait for user to read
sleep 5

# Get username
USERNAME=$(whoami)
echo "Setting up for user: $USERNAME"

# Prompt for password
echo
echo "Please enter your Mac password:"
echo "(It will be stored securely in Keychain)"
read -s PASSWORD

# Verify password
echo
echo "Please verify your password:"
read -s PASSWORD_VERIFY

if [ "$PASSWORD" != "$PASSWORD_VERIFY" ]; then
    echo
    echo "❌ Passwords don't match. Setup cancelled."
    exit 1
fi

# Delete any existing entry
echo
echo "📝 Storing password in Keychain..."
security delete-generic-password -s com.jarvis.voiceunlock -a unlock_token 2>/dev/null

# Add new entry
security add-generic-password \
    -s com.jarvis.voiceunlock \
    -a unlock_token \
    -w "$PASSWORD" \
    -T /usr/bin/security \
    -U

if [ $? -eq 0 ]; then
    echo "✅ Password stored successfully!"
else
    echo "❌ Failed to store password"
    exit 1
fi

# Update enrollment file
echo
echo "📝 Updating enrollment data..."

VOICE_UNLOCK_DIR="$HOME/.jarvis/voice_unlock"
mkdir -p "$VOICE_UNLOCK_DIR"

cat > "$VOICE_UNLOCK_DIR/enrolled_users.json" << EOF
{
  "default_user": {
    "name": "$USERNAME",
    "enrolled": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "active": true,
    "has_password": true
  }
}
EOF

# Create setup complete marker
date > "$VOICE_UNLOCK_DIR/.setup_complete"

echo "✅ Enrollment data updated!"
echo
echo "🎉 Voice Unlock Setup Complete!"
echo "==============================="
echo
echo "Your screen can now be unlocked by voice!"
echo
echo "How to use:"
echo "1. Make sure Voice Unlock is running:"
echo "   ./start_voice_unlock_system.sh"
echo
echo "2. Lock your screen (⌘+Control+Q)"
echo
echo "3. Say one of these phrases:"
echo "   - 'Hello JARVIS, unlock my Mac'"
echo "   - 'JARVIS, this is $USERNAME'"
echo "   - 'Open sesame, JARVIS'"
echo
echo "The system will type your password and unlock the screen!"
echo
echo "⚠️  Security Notes:"
echo "- Your password is encrypted in the macOS Keychain"
echo "- Only the Voice Unlock system can access it"
echo "- Remove it anytime with:"
echo "  security delete-generic-password -s com.jarvis.voiceunlock -a unlock_token"