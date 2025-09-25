# JARVIS Voice Unlock - Quick Reference Guide

## 🔓 Two Ways to Unlock Your Screen

### 1. Manual Unlock - Direct Control
**Command:** `"Hey JARVIS, unlock my screen"`

- **When to use:** When you just want to unlock your screen
- **Response:** "Screen unlocked successfully, Sir"
- **Benefits:** 
  - Works 24/7 (no quiet hours)
  - Instant execution
  - No questions asked

### 2. Context-Aware Unlock - Smart Assistant
**Scenario:** Screen is locked + you ask JARVIS to do something

**Example:** `"Hey JARVIS, open Safari and search for dogs"`

**JARVIS Response Flow:**
1. "I see your screen is locked. I'll unlock it now by typing in your password so I can open Safari and search for dogs."
2. [Unlocks screen automatically]
3. [Opens Safari and searches]
4. "I unlocked your screen, opened Safari, and searched for dogs, Sir"

## 🎯 Key Features

- **No Time Restrictions** - Both modes work anytime, day or night
- **Secure** - Password stored in macOS Keychain
- **Fast** - Unlock in under 2 seconds
- **Smart** - Knows when to unlock automatically
- **Clear Feedback** - Always tells you what it's doing

## 💡 Examples

### Manual Unlock Examples:
- `"unlock my screen"`
- `"unlock screen"`
- `"unlock the screen"`

### Context-Aware Examples (when locked):
- `"open Chrome and go to YouTube"`
- `"check my calendar"`
- `"take a screenshot"`
- `"create a new document"`

## 🔧 Setup Required

1. Run `enable_screen_unlock.sh` to store your password securely
2. Ensure Voice Unlock daemon is running (starts with JARVIS)
3. Say "Hey JARVIS" to activate, then your command

## ⚠️ Troubleshooting

**If unlock doesn't work:**
- Check if daemon is running: `ps aux | grep 8765`
- Re-run setup: `./enable_screen_unlock.sh`
- Check JARVIS logs: `tail -f /tmp/jarvis.log`

**If you get "policy denied" message:**
- This has been fixed! Manual unlock now bypasses all policies
- Restart JARVIS if you still see this error

## 🚀 Pro Tips

1. **Quick unlock when sitting down:** Just say "Hey JARVIS, unlock my screen"
2. **Let JARVIS handle it:** When locked, just tell JARVIS what you want done
3. **Chain commands:** "Unlock my screen and open my email"
4. **Works with multi-commands:** "Unlock my screen, open Safari and search for Python tutorials"

---
*Voice Unlock: The perfect Apple Watch alternative for Mac users!*