# Voice Unlock Security Guide

## Password Storage

Your Mac login password is **NEVER** stored in the source code. It is securely stored in the macOS Keychain, which provides:

- **Encryption at rest** - Passwords are encrypted using your Mac's security infrastructure
- **Access control** - Only the Voice Unlock daemon can retrieve the password
- **No code storage** - The password never appears in any source files

## Before Pushing to GitHub

1. **Run the cleanup script**:
   ```bash
   cd backend/voice_unlock
   ./clean_sensitive_data.sh
   ```

2. **Check .gitignore** - Ensure these patterns are present:
   ```
   backend/voice_unlock/test_*.py
   backend/voice_unlock/debug_*.py
   backend/voice_unlock/*_test.py
   backend/voice_unlock/fix_*.py
   ```

3. **Never commit**:
   - Test files with hardcoded passwords
   - Debug scripts with credentials
   - Log files that might contain sensitive data

## Safe Testing Practices

When testing Voice Unlock:

1. **Use the Keychain** - Always retrieve passwords from Keychain:
   ```python
   result = subprocess.run([
       'security', 'find-generic-password',
       '-s', 'com.jarvis.voiceunlock',
       '-a', 'unlock_token',
       '-w'
   ], capture_output=True, text=True)
   password = result.stdout.strip()
   ```

2. **Don't hardcode** - Never put passwords in test files
3. **Use environment variables** - For test passwords, use env vars:
   ```python
   test_password = os.environ.get('TEST_PASSWORD', 'dummy_pass')
   ```

## Security Best Practices

1. **Regular cleanup** - Run `clean_sensitive_data.sh` before each commit
2. **Check commits** - Review `git diff` before committing
3. **Use pre-commit hook** - The hook will catch most password patterns
4. **Rotate passwords** - If a password is accidentally exposed, change it immediately

## What's Safe to Commit

✅ **Safe to commit**:
- WebSocket server code
- Daemon code
- Voice processing logic
- Configuration files (without passwords)
- Documentation
- Build scripts

❌ **Never commit**:
- Files containing your actual password
- Test files with hardcoded credentials
- Debug logs
- Keychain export files
- Any file with "password = 'your_actual_password'"

## If You Accidentally Commit a Password

1. **Don't push** - If you haven't pushed yet, amend the commit
2. **If pushed** - Change your Mac password immediately
3. **Clean history** - Use `git filter-branch` or BFG Repo Cleaner
4. **Rotate** - Update the password in Keychain with the new one

Remember: The Voice Unlock system is designed to work with passwords stored securely in Keychain, never in code!