# Claude API Setup for JARVIS Document Writer

## Current Status
The document writer feature is working but currently running in **DEMO mode** because the Claude API key is invalid or not configured.

## What's Happening
When you ask JARVIS to write an essay or document:
1. ✅ The command is recognized correctly
2. ✅ The document writer module is triggered
3. ⚠️ The Claude API authentication fails (invalid key)
4. ✅ The system falls back to demo content (generic pre-written text)
5. ✅ The document is created successfully with demo content

## How to Enable Real AI Content Generation

### Option 1: Quick Setup Script (Recommended)
```bash
cd backend
python setup_claude_api.py
```

This script will:
- Check your current API key status
- Help you set up a valid key
- Test the connection
- Create/update your .env file

### Option 2: Manual Setup

1. **Get an API Key**
   - Go to: https://console.anthropic.com/settings/keys
   - Create a new API key
   - Copy the key (starts with `sk-ant-api03-`)

2. **Set the Environment Variable**
   ```bash
   export ANTHROPIC_API_KEY='your-key-here'
   ```

3. **Or Create a .env File**
   Create `backend/.env` with:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

## Testing the Setup

After setting up your API key:

1. **Test directly:**
   ```bash
   cd backend
   python test_claude_api.py
   ```

2. **Test in JARVIS:**
   - Start JARVIS
   - Say: "Write me an essay about climate change"
   - You should see real AI-generated content instead of generic demo text

## How to Identify Demo vs Real Content

### Demo Mode Indicators:
- Generic content about dogs (always the same)
- Warning message: "⚠️ Running in DEMO mode"
- Content always starts with "Dogs have been humanity's faithful companions..."

### Real API Mode:
- Unique, contextual content for each request
- Content specifically about your requested topic
- No demo warning messages
- Dynamic, varied writing style

## Troubleshooting

### Invalid API Key Error
- Verify key starts with `sk-ant-api03-`
- Check for extra spaces or quotes
- Ensure the key is active in your Anthropic console

### Environment Variable Not Found
- Make sure to restart your terminal after setting the variable
- Or source your .env file: `source backend/.env`

### Rate Limiting
- Free tier has limits
- Consider upgrading your Anthropic plan if needed

## Current Implementation Details

The document writer uses:
- **Claude 3.5 Sonnet** (primary model) for content generation
- Automatic fallback to **Claude 3 Haiku** if rate limited
- Demo content fallback if API is unavailable
- Real-time streaming to Google Docs
- Support for multiple document formats (MLA, APA, Chicago, etc.)

## Security Note
Never commit your API key to git! The `.env` file should be in `.gitignore`.