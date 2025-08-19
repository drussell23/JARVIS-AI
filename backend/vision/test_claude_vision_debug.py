#!/usr/bin/env python3
"""
Debug Claude Vision Integration
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.screen_capture_fallback import capture_screen_fallback, analyze_with_claude_vision
import numpy as np


def test_claude_vision():
    """Test Claude Vision with detailed debugging"""
    
    print("🔍 Claude Vision Debugging")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"1. API Key present: {'✅ YES' if api_key else '❌ NO'}")
    if api_key:
        print(f"   Key starts with: {api_key[:15]}...")
        print(f"   Key length: {len(api_key)}")
    
    # Test screen capture
    print("\n2. Testing screen capture...")
    screenshot = capture_screen_fallback()
    if screenshot is not None:
        print(f"   ✅ Screen captured: {screenshot.shape}")
    else:
        print("   ❌ Screen capture failed")
        return
    
    # Test Claude Vision
    print("\n3. Testing Claude Vision analysis...")
    try:
        # Direct test of analyze_with_claude_vision
        result = analyze_with_claude_vision(
            screenshot,
            "Please describe what you see on the screen in one sentence."
        )
        print("   ✅ Claude Vision SUCCESS!")
        print(f"   Response: {result[:200]}...")
        
    except Exception as e:
        print(f"   ❌ Claude Vision ERROR: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        
        # Additional debugging
        if "anthropic" in str(e).lower():
            print("\n   Troubleshooting:")
            print("   - Check if anthropic package is installed: pip install anthropic")
            print("   - Verify API key is valid")
            print("   - Check internet connection")
        
        if "model" in str(e).lower():
            print("\n   Model issue detected. Testing with different model...")
            # Try with a different model
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                
                # List available models
                print("\n   Testing basic Claude connection...")
                test_response = client.messages.create(
                    model="claude-3-haiku-20240307",  # Try a different model
                    max_tokens=100,
                    messages=[{"role": "user", "content": "Say 'Hello'"}]
                )
                print(f"   ✅ Basic Claude works: {test_response.content[0].text}")
                
            except Exception as e2:
                print(f"   ❌ Basic Claude also failed: {e2}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")


if __name__ == "__main__":
    test_claude_vision()