#!/usr/bin/env python3
"""
Claude API Setup Helper
========================

This script helps you configure the Claude API for JARVIS document writer.
"""

import os
import sys
from pathlib import Path


def check_current_api_key():
    """Check if API key is currently set"""
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if api_key:
        if api_key.startswith('sk-ant-api03-'):
            print("✅ Valid API key format detected")
            return True
        else:
            print("⚠️  API key found but appears to be invalid format")
            print(f"   Current key starts with: {api_key[:15]}...")
            print("   Valid keys should start with: sk-ant-api03-")
            return False
    else:
        print("❌ No ANTHROPIC_API_KEY found in environment")
        return False


def create_env_file():
    """Create or update .env file with API key"""
    env_path = Path(__file__).parent / '.env'
    env_example_path = Path(__file__).parent / '.env.example'

    # Read example file
    if env_example_path.exists():
        with open(env_example_path, 'r') as f:
            example_content = f.read()
    else:
        example_content = ""

    print("\n📝 Setting up .env file...")

    # Check if .env exists
    if env_path.exists():
        print("   .env file already exists")
        response = input("   Do you want to update it? (y/n): ").lower()
        if response != 'y':
            return

    # Get API key from user
    print("\n🔑 Please enter your Anthropic API key")
    print("   (Get one at: https://console.anthropic.com/settings/keys)")
    api_key = input("   API Key: ").strip()

    # Validate format
    if not api_key.startswith('sk-ant-api03-'):
        print("\n⚠️  Warning: Key doesn't match expected format (sk-ant-api03-...)")
        response = input("   Continue anyway? (y/n): ").lower()
        if response != 'y':
            return

    # Create .env content
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()

        # Update existing key
        if 'ANTHROPIC_API_KEY' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('ANTHROPIC_API_KEY'):
                    lines[i] = f'ANTHROPIC_API_KEY={api_key}'
                    break
            content = '\n'.join(lines)
        else:
            # Add key
            content += f'\n\nANTHROPIC_API_KEY={api_key}\n'
    else:
        # Create new file from example
        if example_content:
            content = example_content.replace('your_api_key_here', api_key)
        else:
            content = f'ANTHROPIC_API_KEY={api_key}\n'

    # Write file
    with open(env_path, 'w') as f:
        f.write(content)

    print(f"\n✅ .env file {'updated' if env_path.exists() else 'created'} successfully!")
    print(f"   Path: {env_path}")


def test_api_connection():
    """Test the API connection"""
    print("\n🧪 Testing API connection...")

    try:
        import anthropic
    except ImportError:
        print("❌ anthropic package not installed")
        print("   Install with: pip install anthropic")
        return False

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        # Try loading from .env file
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
            api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        print("❌ No API key found")
        return False

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'API works'"}]
        )
        print(f"✅ API connection successful!")
        print(f"   Response: {response.content[0].text}")
        return True
    except anthropic.AuthenticationError:
        print("❌ Authentication failed - invalid API key")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    """Main setup flow"""
    print("=" * 60)
    print("JARVIS Claude API Setup")
    print("=" * 60)

    # Check current status
    print("\n1️⃣  Checking current configuration...")
    has_valid_key = check_current_api_key()

    if not has_valid_key:
        print("\n2️⃣  Setting up API key...")
        create_env_file()

        # Reload environment
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)

    # Test connection
    print("\n3️⃣  Testing connection...")
    success = test_api_connection()

    print("\n" + "=" * 60)
    if success:
        print("✅ Setup Complete!")
        print("\nYour document writer will now use the real Claude API.")
        print("\nTry these commands in JARVIS:")
        print('  • "Write me an essay about climate change"')
        print('  • "Create a 500 word report on AI"')
        print('  • "Draft an MLA format paper on renewable energy"')
    else:
        print("❌ Setup incomplete")
        print("\nTo fix:")
        print("1. Get an API key from: https://console.anthropic.com/settings/keys")
        print("2. Run this script again: python setup_claude_api.py")
        print("3. Or manually set: export ANTHROPIC_API_KEY='your-key-here'")
    print("=" * 60)


if __name__ == "__main__":
    # Check for dotenv
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        os.system("pip install python-dotenv")
        print("Please run this script again.")
        sys.exit(1)

    main()