#!/bin/bash
#
# Voice Unlock Keychain Password Fix
# ===================================
# This script fixes the missing keychain password issue
#

set -e

echo "🔐 JARVIS Voice Unlock - Keychain Password Setup"
echo "=================================================="
echo ""
echo "Current Issue: Password not found in keychain"
echo "This prevents voice unlock from working properly."
echo ""
echo "This script will:"
echo "  1. Prompt for your macOS password"
echo "  2. Store it securely in macOS Keychain"
echo "  3. Verify the storage was successful"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo ""
echo "Enter your macOS password (the one you use to unlock your Mac):"
read -s PASSWORD
echo ""

# Store in keychain
echo "🔐 Storing password in macOS Keychain..."
security add-generic-password \
    -a "jarvis" \
    -s "jarvis_voice_unlock" \
    -w "$PASSWORD" \
    -U 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Password stored successfully!"
    echo ""

    # Verify it was stored
    echo "🔍 Verifying storage..."
    STORED_PASS=$(security find-generic-password -s "jarvis_voice_unlock" -w 2>&1)

    if [ "$STORED_PASS" = "$PASSWORD" ]; then
        echo "✅ Verification successful!"
        echo ""
        echo "🎉 Voice unlock is now ready to use!"
        echo ""
        echo "Try saying: 'JARVIS, unlock my screen'"
    else
        echo "⚠️  Warning: Stored password doesn't match"
        echo "You may need to run this script again"
    fi
else
    echo "❌ Failed to store password"
    echo "Error: $(security add-generic-password -a jarvis -s jarvis_voice_unlock -w \"$PASSWORD\" 2>&1)"
fi

# Clear password variable
unset PASSWORD
