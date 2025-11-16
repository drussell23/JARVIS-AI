# Apple Developer ID Setup for JARVIS HUD

**Goal:** Properly sign and notarize JARVIS-HUD.app for distribution outside Mac App Store

**Cost:** $99/year (Apple Developer Program)

**Timeline:** 2-4 weeks total (1 day enrollment + 1-2 weeks approval + setup time)

---

## Phase 1: Apple Developer Program Enrollment

### Step 1: Enroll in Apple Developer Program

1. **Go to:** https://developer.apple.com/programs/enroll/
2. **Sign in** with your Apple ID (or create one)
3. **Choose:** "Individual" or "Organization"
   - **Individual:** Faster approval, uses your personal name
   - **Organization:** Requires D-U-N-S number, business verification
4. **Pay:** $99/year (auto-renews)
5. **Wait:** 24-48 hours for approval email

**Status:** ⏳ Pending (do this first)

---

## Phase 2: Request Developer ID Certificate

### Step 2: Generate Certificate Signing Request (CSR)

Run this on your Mac:

```bash
# Open Keychain Access
open /Applications/Utilities/Keychain\ Access.app

# Then in Keychain Access:
# 1. Menu: Keychain Access → Certificate Assistant → Request a Certificate from a Certificate Authority
# 2. Fill in:
#    - User Email Address: your@email.com
#    - Common Name: Derek J. Russell (your name)
#    - CA Email: (leave blank)
#    - Request: "Saved to disk"
# 3. Save as: DeveloperID_Certificate.certSigningRequest
# 4. Location: ~/Desktop
```

**Output:** `DeveloperID_Certificate.certSigningRequest` file on Desktop

### Step 3: Create Developer ID Application Certificate

1. **Go to:** https://developer.apple.com/account/resources/certificates/add
2. **Select:** "Developer ID Application" (for apps outside Mac App Store)
3. **Upload:** The `.certSigningRequest` file from Step 2
4. **Download:** The certificate (`.cer` file)
5. **Double-click** the `.cer` file to install in Keychain

**Verify installation:**
```bash
security find-identity -v -p codesigning
```

You should see:
```
1) XXXXXXXX "Developer ID Application: Derek J. Russell (TEAM_ID)"
```

**Status:** ⏳ Pending (after enrollment approved)

---

## Phase 3: Sign the JARVIS HUD App

### Step 4: Create Entitlements File

Create `JARVIS-HUD.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Enable Hardened Runtime -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>

    <!-- Network access for WebSocket connection to backend -->
    <key>com.apple.security.network.client</key>
    <true/>

    <!-- Allow unsigned executable memory (for JIT if needed) -->
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>

    <!-- Disable library validation (allows loading Python/native libs) -->
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
```

Save to: `JARVIS-HUD/JARVIS-HUD.entitlements`

### Step 5: Sign the App Bundle

```bash
# Navigate to where HUD app is built
cd /Users/derekjrussell/Library/Developer/Xcode/DerivedData

# Find the latest JARVIS-HUD build
JARVIS_HUD=$(find . -name "JARVIS-HUD.app" -type d | head -1)

# Or if built to /Applications:
JARVIS_HUD="/Applications/JARVIS-HUD.app"

# Get your Developer ID (replace with actual identity from Step 3)
IDENTITY="Developer ID Application: Derek J. Russell (XXXXXXXXXX)"

# Sign with Hardened Runtime enabled
codesign \
  --deep \
  --force \
  --verify \
  --verbose \
  --timestamp \
  --options runtime \
  --entitlements JARVIS-HUD/JARVIS-HUD.entitlements \
  --sign "$IDENTITY" \
  "$JARVIS_HUD"

# Verify signature
codesign --verify --deep --strict --verbose=2 "$JARVIS_HUD"
spctl --assess --type execute --verbose=4 "$JARVIS_HUD"
```

**Expected output:**
```
JARVIS-HUD.app: valid on disk
JARVIS-HUD.app: satisfies its Designated Requirement
JARVIS-HUD.app: accepted
source=Developer ID
```

**Status:** ⏳ Pending (after certificate obtained)

---

## Phase 4: Notarize with Apple

### Step 6: Create App-Specific Password

1. **Go to:** https://appleid.apple.com/account/manage
2. **Sign in** with Apple ID
3. **Security → App-Specific Passwords**
4. **Generate** password for "JARVIS Notarization"
5. **Save** the password (you'll need it below)

### Step 7: Create ZIP for Notarization

```bash
# Create a ZIP of the signed app
cd /Applications
ditto -c -k --keepParent JARVIS-HUD.app JARVIS-HUD.zip

# Verify ZIP
unzip -l JARVIS-HUD.zip | head -20
```

### Step 8: Submit for Notarization

```bash
# Store credentials (one-time setup)
xcrun notarytool store-credentials "jarvis-notarization" \
  --apple-id "your@email.com" \
  --team-id "XXXXXXXXXX" \
  --password "xxxx-xxxx-xxxx-xxxx"  # App-specific password from Step 6

# Submit for notarization
xcrun notarytool submit JARVIS-HUD.zip \
  --keychain-profile "jarvis-notarization" \
  --wait

# Expected: "status: Accepted" (takes 5-15 minutes)
```

**If accepted:**
```
Successfully uploaded file
  id: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  path: JARVIS-HUD.zip
Processing started...
  status: Accepted
```

**If rejected:** Check logs:
```bash
xcrun notarytool log <submission-id> --keychain-profile "jarvis-notarization"
```

### Step 9: Staple the Notarization Ticket

```bash
# Staple the ticket to the app
xcrun stapler staple /Applications/JARVIS-HUD.app

# Verify stapling
xcrun stapler validate /Applications/JARVIS-HUD.app
spctl --assess --type execute --verbose /Applications/JARVIS-HUD.app
```

**Expected:**
```
The staple and validate action worked!
/Applications/JARVIS-HUD.app: accepted
source=Notarized Developer ID
```

**Status:** ⏳ Pending (after signing)

---

## Phase 5: Test and Distribute

### Step 10: Test Launch (No Gatekeeper Warnings!)

```bash
# Kill any old instances
killall -9 JARVIS-HUD 2>/dev/null

# Launch - should work instantly with NO warnings
open /Applications/JARVIS-HUD.app

# Or from Python (now trivial):
python3 -c "import subprocess; subprocess.run(['open', '-a', 'JARVIS-HUD'])"
```

**Expected:** HUD launches immediately, no Gatekeeper dialog!

### Step 11: Automated Signing Script

Create `scripts/sign_and_notarize_hud.sh`:

```bash
#!/bin/bash
set -e

IDENTITY="Developer ID Application: Derek J. Russell (XXXXXXXXXX)"
PROFILE="jarvis-notarization"
APP_PATH="/Applications/JARVIS-HUD.app"

echo "🔐 Signing JARVIS-HUD..."
codesign --deep --force --verify --verbose --timestamp \
  --options runtime \
  --entitlements JARVIS-HUD/JARVIS-HUD.entitlements \
  --sign "$IDENTITY" \
  "$APP_PATH"

echo "📦 Creating ZIP..."
rm -f JARVIS-HUD.zip
ditto -c -k --keepParent "$APP_PATH" JARVIS-HUD.zip

echo "🍎 Submitting to Apple for notarization..."
xcrun notarytool submit JARVIS-HUD.zip \
  --keychain-profile "$PROFILE" \
  --wait

echo "📌 Stapling notarization ticket..."
xcrun stapler staple "$APP_PATH"

echo "✅ Done! HUD is signed and notarized."
echo "   Launch: open $APP_PATH"
```

Make executable:
```bash
chmod +x scripts/sign_and_notarize_hud.sh
```

**Usage:**
```bash
# After building HUD in Xcode, run:
./scripts/sign_and_notarize_hud.sh
```

---

## Integration with start_system.py

After notarization, update `start_system.py` to use simple launch:

```python
# BEFORE (multi-strategy launcher):
from macos_hud_launcher import launch_hud_async_safe
launch_success = await launch_hud_async_safe(hud_app_path)

# AFTER (signed app - trivial launch):
subprocess.run(["open", "-a", str(hud_app_path)])
# That's it! No strategies needed, works instantly.
```

---

## Troubleshooting

### Issue: "Developer ID not found"
**Solution:** Run `security find-identity -v -p codesigning` to get exact name

### Issue: "Notarization failed - invalid signature"
**Solution:**
- Ensure Hardened Runtime enabled (`--options runtime`)
- Check entitlements match app capabilities
- Verify all frameworks/dylibs are signed

### Issue: "App-specific password rejected"
**Solution:**
- Use app-specific password, NOT Apple ID password
- Regenerate at appleid.apple.com if expired

### Issue: "Gatekeeper still blocks app"
**Solution:**
- Verify stapling: `xcrun stapler validate JARVIS-HUD.app`
- Clear quarantine: `xattr -cr JARVIS-HUD.app`
- Re-notarize if needed

---

## Timeline & Costs

| Phase | Time | Cost |
|-------|------|------|
| Enrollment | 24-48 hours | $99/year |
| Certificate Setup | 30 minutes | Included |
| First-time Signing | 1 hour (learning) | Free |
| Notarization | 5-15 minutes (Apple review) | Free |
| Future Builds | 5 minutes (automated) | Free |
| **Total First Time** | **2-4 days** | **$99/year** |
| **Each Update** | **5 minutes** | **Free** |

---

## Benefits After Setup

✅ **No Gatekeeper Warnings:** Users never see security dialogs
✅ **Instant Launch:** `open -a JARVIS-HUD` works immediately
✅ **Professional Distribution:** Can share .dmg files publicly
✅ **Auto-updates:** Can use Sparkle framework for auto-updates
✅ **Trust:** macOS shows "verified developer" in About dialog
✅ **Future App Store:** Certificate also works for Mac App Store submission

---

## Next Steps

1. ✅ **Enroll:** https://developer.apple.com/programs/enroll/
2. ⏳ **Wait for approval** (check email)
3. ⏳ **Generate CSR** (Keychain Access)
4. ⏳ **Download certificate** (developer.apple.com)
5. ⏳ **Sign HUD** (codesign command)
6. ⏳ **Notarize** (xcrun notarytool)
7. ✅ **Launch instantly!**

**Current Status:** Ready to enroll - need to purchase $99 Apple Developer membership

---

## References

- [Apple Developer Program](https://developer.apple.com/programs/)
- [Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/)
- [Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Hardened Runtime](https://developer.apple.com/documentation/security/hardened_runtime)

---

**Created:** November 16, 2025
**For:** JARVIS AI Assistant v17.7.2
**Status:** Documentation Complete - Ready to Execute
