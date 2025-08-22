#!/usr/bin/env python3
"""
Diagnostic script for JARVIS Vision WebSocket issues
"""

import subprocess
import sys
import os
import json
import socket
from pathlib import Path


def check_backend_running():
    """Check if backend is running on port 8000"""
    print("🔍 Checking if backend is running...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8000))
    sock.close()
    
    if result == 0:
        print("✅ Backend is running on port 8000")
        return True
    else:
        print("❌ Backend is NOT running on port 8000")
        return False


def check_env_file():
    """Check if .env file exists and has API key"""
    print("\n🔍 Checking environment configuration...")
    
    env_path = Path("backend/.env")
    if not env_path.exists():
        print("❌ backend/.env file not found!")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if "ANTHROPIC_API_KEY=" in content:
            print("✅ ANTHROPIC_API_KEY found in .env")
            return True
        else:
            print("❌ ANTHROPIC_API_KEY not found in .env")
            return False


def check_imports():
    """Check if all required modules can be imported"""
    print("\n🔍 Checking Python imports...")
    
    modules_to_check = [
        ("fastapi", "FastAPI"),
        ("websockets", "WebSockets"),
        ("anthropic", "Anthropic API"),
        ("cv2", "OpenCV (optional)"),
        ("PIL", "Pillow (optional)"),
        ("pytesseract", "Tesseract OCR (optional)")
    ]
    
    all_good = True
    for module, name in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {name} is installed")
        except ImportError:
            if "optional" in name:
                print(f"⚠️  {name} is not installed")
            else:
                print(f"❌ {name} is not installed")
                all_good = False
    
    return all_good


def test_api_endpoints():
    """Test various API endpoints"""
    print("\n🔍 Testing API endpoints...")
    
    import requests
    
    endpoints = [
        ("http://localhost:8000/", "Root endpoint"),
        ("http://localhost:8000/health", "Health check"),
        ("http://localhost:8000/vision/status", "Vision status"),
        ("http://localhost:8000/voice/jarvis/status", "JARVIS status")
    ]
    
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
            else:
                print(f"⚠️  {name}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")


def check_frontend_running():
    """Check if frontend is running"""
    print("\n🔍 Checking frontend...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 3000))
    sock.close()
    
    if result == 0:
        print("✅ Frontend is running on port 3000")
        return True
    else:
        print("⚠️  Frontend is not running on port 3000")
        return False


def diagnose():
    """Run full diagnostic"""
    print("🤖 JARVIS Vision WebSocket Diagnostic")
    print("=" * 50)
    
    issues = []
    
    # Check environment
    if not check_env_file():
        issues.append("Missing or incomplete .env file")
    
    # Check imports
    if not check_imports():
        issues.append("Missing required Python packages")
    
    # Check backend
    if check_backend_running():
        test_api_endpoints()
    else:
        issues.append("Backend not running")
    
    # Check frontend
    check_frontend_running()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if not issues:
        print("✅ Everything looks good!")
        print("\nIf you're still having issues:")
        print("1. Check browser console for errors")
        print("2. Make sure you've activated full autonomy mode")
        print("3. Try refreshing the browser page")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  • {issue}")
        
        print("\n📋 Next steps:")
        if "Backend not running" in issues:
            print("1. Start the backend:")
            print("   cd backend && python main.py")
        if "Missing or incomplete .env file" in issues:
            print("2. Create backend/.env with:")
            print("   ANTHROPIC_API_KEY=your-api-key-here")
        if "Missing required Python packages" in issues:
            print("3. Install dependencies:")
            print("   pip install -r backend/requirements.txt")


if __name__ == "__main__":
    diagnose()