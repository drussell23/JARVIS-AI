# Vision System Timeout Configuration

## Overview
The vision analysis system has been configured with increased timeouts to handle complex screen analysis requests that may take longer to process.

## Timeout Settings

### API Timeouts
- **VISION_API_TIMEOUT**: Default 60 seconds (increased from 30)
  - Controls the timeout for Anthropic API calls
  - Can be overridden with environment variable: `VISION_API_TIMEOUT=90`

- **SIMPLIFIED_VISION_TIMEOUT**: Default 60 seconds (increased from 30)
  - Controls timeout for simplified vision system
  - Can be overridden with environment variable: `SIMPLIFIED_VISION_TIMEOUT=90`

### Application-level Timeouts
- **vision_command_handler.py**: 60-second timeout for asyncio.wait_for calls
- **Anthropic client**: Initialized with timeout from VISION_API_TIMEOUT

## Configuration
To adjust timeouts, set the following environment variables before starting the backend:

```bash
export VISION_API_TIMEOUT=90  # Set to 90 seconds
export SIMPLIFIED_VISION_TIMEOUT=90
```

## Why These Changes Were Made
1. Vision analysis of complex screens can take longer than 30 seconds
2. The Anthropic API needs time to process high-resolution images
3. Network latency and API queue times can add delays
4. Increased timeout reduces "analysis timed out" errors

## Monitoring
If you continue to experience timeouts:
1. Check your internet connection speed
2. Verify the Anthropic API status
3. Consider reducing screen resolution or using compression
4. Enable caching to avoid repeated API calls for similar screens