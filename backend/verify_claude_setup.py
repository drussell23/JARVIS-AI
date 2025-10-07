#!/usr/bin/env python3
"""
Verify Claude API Setup for JARVIS
===================================

Run this script to verify that your Claude API is properly configured
and JARVIS can generate real AI content.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv


def check_env_files():
    """Check for .env files and API keys"""
    print("\n📁 Checking environment files...")

    backend_env = Path(__file__).parent / '.env'
    root_env = Path(__file__).parent.parent / '.env'

    results = []

    if backend_env.exists():
        print(f"  ✅ Backend .env found: {backend_env}")
        load_dotenv(backend_env, override=True)
        key = os.getenv('ANTHROPIC_API_KEY')
        if key:
            print(f"     API Key: {key[:30]}...")
            results.append(('backend', key))
    else:
        print(f"  ❌ No backend .env file")

    if root_env.exists():
        print(f"  ✅ Root .env found: {root_env}")
        load_dotenv(root_env, override=True)
        key = os.getenv('ANTHROPIC_API_KEY')
        if key:
            print(f"     API Key: {key[:30]}...")
            results.append(('root', key))
    else:
        print(f"  ❌ No root .env file")

    return results


def test_api_key(api_key):
    """Test if API key is valid"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        return True, response.content[0].text
    except anthropic.AuthenticationError:
        return False, "Invalid API key"
    except Exception as e:
        return False, str(e)


async def test_document_writer():
    """Test document writer module"""
    print("\n🔧 Testing document writer module...")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from context_intelligence.automation.claude_streamer import ClaudeContentStreamer

        api_key = os.getenv('ANTHROPIC_API_KEY')
        streamer = ClaudeContentStreamer(api_key=api_key)

        # Quick test
        content = ""
        async for chunk in streamer.stream_content(
            "Write the word 'SUCCESS'",
            max_tokens=10
        ):
            content += chunk

        if "SUCCESS" in content.upper():
            return True, "Real API content"
        elif "Dogs" in content:
            return False, "Mock/demo content"
        else:
            return True, f"Generated: {content[:50]}"

    except Exception as e:
        return False, str(e)


async def main():
    """Run all verification checks"""
    print("=" * 60)
    print("JARVIS Claude API Verification")
    print("=" * 60)

    # Check environment files
    env_results = check_env_files()

    # Test API key
    print("\n🔑 Testing API key...")
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if api_key:
        print(f"  Current key: {api_key[:30]}...")
        valid, message = test_api_key(api_key)

        if valid:
            print(f"  ✅ API key is VALID - Claude responded: {message}")
        else:
            print(f"  ❌ API key is INVALID - {message}")
    else:
        print("  ❌ No API key found in environment")
        valid = False

    # Test document writer
    if valid:
        writer_works, writer_msg = await test_document_writer()
        if writer_works:
            print(f"  ✅ Document writer using REAL API: {writer_msg}")
        else:
            print(f"  ⚠️ Document writer in DEMO mode: {writer_msg}")
    else:
        print("  ⏭️ Skipping document writer test (no valid API key)")
        writer_works = False

    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if valid and writer_works:
        print("✅ SUCCESS: JARVIS is configured to use real Claude API!")
        print("\nYour setup is complete. JARVIS will now generate:")
        print("  • Unique, contextual AI content for each request")
        print("  • Properly formatted documents (MLA, APA, Chicago, etc.)")
        print("  • Content specifically about your requested topics")
        print("\nTry these commands in JARVIS:")
        print('  • "Write me an essay about climate change"')
        print('  • "Create a 500 word report on quantum computing"')
        print('  • "Draft an MLA format paper about artificial intelligence"')
    elif valid and not writer_works:
        print("⚠️ PARTIAL SUCCESS: API key works but document writer needs configuration")
        print("\nTroubleshooting:")
        print("  1. Restart JARVIS")
        print("  2. Check that backend/.env has your API key")
    else:
        print("❌ SETUP INCOMPLETE: Valid API key needed")
        print("\nTo fix:")
        print("  1. Get an API key from: https://console.anthropic.com/settings/keys")
        print("  2. Add to backend/.env:")
        print("     ANTHROPIC_API_KEY=your-key-here")
        print("  3. Run this script again to verify")

    print("=" * 60)


if __name__ == "__main__":
    # Check for required packages
    try:
        import anthropic
    except ImportError:
        print("Installing anthropic package...")
        os.system("pip install anthropic")
        print("Please run this script again.")
        sys.exit(1)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv package...")
        os.system("pip install python-dotenv")
        print("Please run this script again.")
        sys.exit(1)

    asyncio.run(main())