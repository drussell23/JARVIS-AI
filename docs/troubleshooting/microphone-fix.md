# JARVIS Microphone Fix Guide

## Quick Fix

Run the automated fix script:
```bash
./fix-microphone.sh
```

This script will:
1. Run comprehensive Python diagnostic
2. Identify apps blocking the microphone
3. Fix browser permissions
4. Restart Core Audio if needed
5. Test microphone access
6. Provide specific solutions

## What's Included

### 1. **Automated Diagnostic System** (`backend/system/microphone_diagnostic.py`)
- Detects blocking applications
- Lists available microphone devices
- Checks browser compatibility
- Tests microphone access
- Applies automatic fixes
- Generates detailed reports

### 2. **Enhanced Fix Script** (`fix-microphone.sh`)
- Runs Python diagnostic first
- Identifies and closes blocking apps (with permission)
- Restarts Core Audio service
- Browser-specific fixes
- Tests microphone with sox
- Offers to start JARVIS after fixing

### 3. **Browser Test Page** (`frontend/public/microphone-test.html`)
- Test browser compatibility
- Request microphone permission
- List audio devices
- Test speech recognition
- Run full diagnostics
- Visual feedback for all tests

### 4. **Integrated into Startup** (`start_system.py`)
- Automatically runs microphone diagnostic on startup
- Shows blocking apps and recommendations
- Continues even if microphone has issues
- Provides clear guidance for fixes

## Common Solutions

### "NotReadableError: Could not start audio source"

**Cause**: Another application is using the microphone

**Solutions**:
1. Close Zoom, Teams, Discord, Slack
2. Run: `./fix-microphone.sh`
3. Restart Core Audio: `sudo killall coreaudiod`
4. Restart your browser

### Browser Permission Issues

**Chrome**:
- Visit: chrome://settings/content/microphone
- Remove and re-add localhost
- Select default microphone

**Safari**:
- Safari → Preferences → Websites → Microphone
- Set localhost to "Allow"

**Firefox**:
- Click lock icon in address bar
- Clear permissions and reload

### Testing

1. **Command Line Test**:
   ```bash
   python3 backend/system/microphone_diagnostic.py
   ```

2. **Browser Test**:
   - Open: http://localhost:3000/microphone-test.html
   - Run all diagnostic tests

3. **Quick Audio Test**:
   ```bash
   # Install sox first
   brew install sox
   
   # Test recording
   sox -d test.wav trim 0 2
   play test.wav
   ```

## How It Works

1. **Startup Check**: When you run `python start_system.py`, it automatically checks microphone
2. **Diagnostic**: Identifies specific issues (blocking apps, permissions, etc.)
3. **Auto-Fix**: Attempts to fix common issues automatically
4. **Manual Fix**: Run `./fix-microphone.sh` for interactive fixes
5. **Verification**: Tests microphone after fixes to confirm it's working

## Troubleshooting

If issues persist after running fixes:

1. **Check Activity Monitor**:
   - Look for apps using high CPU (might be using mic)
   - Force quit suspicious audio apps

2. **Reset Browser**:
   - Completely quit browser (Cmd+Q)
   - Clear browser data for localhost
   - Restart browser

3. **System Reset**:
   - Restart your Mac
   - First app to open should be your browser
   - Test JARVIS immediately

4. **Check Logs**:
   ```bash
   cat logs/microphone_diagnostic.log
   ```

## Prevention

To avoid microphone issues:

1. Close communication apps before using JARVIS
2. Use Chrome or Edge for best compatibility
3. Don't use microphone in multiple browser tabs
4. Grant permissions immediately when prompted

## Support

If you continue to have issues:
- Check the diagnostic log for detailed error information
- The log shows exactly which app is blocking the microphone
- Try a different browser (Chrome recommended)
- Ensure macOS permissions are granted in System Preferences