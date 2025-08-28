#!/usr/bin/env python3
"""Test script to verify the unified WebSocket system components"""

import os
import sys
import json
import subprocess
import time
import asyncio
import websockets
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status messages"""
    if status == 'success':
        print(f"{GREEN}✓ {message}{RESET}")
    elif status == 'error':
        print(f"{RED}✗ {message}{RESET}")
    elif status == 'warning':
        print(f"{YELLOW}⚠ {message}{RESET}")
    else:
        print(f"{BLUE}→ {message}{RESET}")

def check_file_exists(file_path, description):
    """Check if a required file exists"""
    if os.path.exists(file_path):
        print_status(f"{description} found at {file_path}", 'success')
        return True
    else:
        print_status(f"{description} missing at {file_path}", 'error')
        return False

def check_port_available(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

async def test_websocket_connection(url):
    """Test WebSocket connection"""
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps({
                "type": "ping",
                "timestamp": int(time.time())
            }))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            return True, response
    except Exception as e:
        return False, str(e)

def main():
    """Run system verification tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}JARVIS v12.3 - Unified WebSocket System Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_checks_passed = True
    
    # Check 1: Verify TypeScript components
    print_status("Checking TypeScript WebSocket components...", 'info')
    ts_files = [
        ("backend/websocket/UnifiedWebSocketRouter.ts", "TypeScript Router"),
        ("backend/websocket/DynamicWebSocketClient.ts", "Dynamic Client"),
        ("backend/websocket/server.ts", "WebSocket Server"),
        ("backend/websocket/middleware/ErrorHandlingMiddleware.ts", "Error Middleware"),
        ("backend/bridges/WebSocketBridge.ts", "TypeScript Bridge"),
    ]
    
    for file_path, desc in ts_files:
        if not check_file_exists(file_path, desc):
            all_checks_passed = False
    
    # Check 2: Verify Python components
    print_status("\nChecking Python components...", 'info')
    py_files = [
        ("backend/bridges/python_ts_bridge.py", "Python Bridge"),
        ("backend/api/unified_vision_handler.py", "Unified Vision Handler"),
        ("backend/main.py", "Main FastAPI"),
        ("backend/start_unified_backend.sh", "Startup Script"),
    ]
    
    for file_path, desc in py_files:
        if not check_file_exists(file_path, desc):
            all_checks_passed = False
    
    # Check 3: Verify Node.js build
    print_status("\nChecking TypeScript build...", 'info')
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd="backend/websocket",
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_status("TypeScript build successful", 'success')
        else:
            print_status(f"TypeScript build failed: {result.stderr}", 'error')
            all_checks_passed = False
    except Exception as e:
        print_status(f"Could not run TypeScript build: {e}", 'error')
        all_checks_passed = False
    
    # Check 4: Verify ports are available
    print_status("\nChecking port availability...", 'info')
    ports = [
        (8000, "Python FastAPI Backend"),
        (8001, "TypeScript WebSocket Router"),
    ]
    
    for port, desc in ports:
        if check_port_available(port):
            print_status(f"Port {port} is available for {desc}", 'success')
        else:
            print_status(f"Port {port} is already in use ({desc})", 'warning')
    
    # Check 5: Verify configuration files
    print_status("\nChecking configuration files...", 'info')
    config_files = [
        ("backend/websocket/websocket-routes.json", "WebSocket Routes Config"),
        ("backend/websocket/tsconfig.json", "TypeScript Config"),
        ("backend/websocket/package.json", "Node.js Package"),
    ]
    
    for file_path, desc in config_files:
        if check_file_exists(file_path, desc):
            try:
                with open(file_path, 'r') as f:
                    if file_path.endswith('.json'):
                        json.load(f)
                        print_status(f"  → {desc} is valid JSON", 'success')
            except json.JSONDecodeError as e:
                print_status(f"  → {desc} has invalid JSON: {e}", 'error')
                all_checks_passed = False
    
    # Check 6: Verify integration with start_system.py
    print_status("\nChecking start_system.py integration...", 'info')
    try:
        with open("start_system.py", 'r') as f:
            content = f.read()
            if "start_unified_backend.sh" in content and "websocket_router" in content:
                print_status("start_system.py properly integrated with unified backend", 'success')
            else:
                print_status("start_system.py not integrated with unified backend", 'error')
                all_checks_passed = False
    except Exception as e:
        print_status(f"Could not check start_system.py: {e}", 'error')
        all_checks_passed = False
    
    # Check 7: Verify README documentation
    print_status("\nChecking documentation...", 'info')
    try:
        with open("README.md", 'r') as f:
            content = f.read()
            if "v12.3" in content and "Unified WebSocket Architecture" in content:
                print_status("README.md properly documents v12.3", 'success')
            else:
                print_status("README.md missing v12.3 documentation", 'warning')
    except Exception as e:
        print_status(f"Could not check README.md: {e}", 'warning')
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_checks_passed:
        print_status("All system checks passed! ✨", 'success')
        print(f"\n{GREEN}To start the unified system:{RESET}")
        print(f"  1. Run: {BLUE}python start_system.py{RESET}")
        print(f"  2. Or manually: {BLUE}cd backend && ./start_unified_backend.sh{RESET}")
    else:
        print_status("Some checks failed. Please review the errors above.", 'error')
        print(f"\n{YELLOW}Common fixes:{RESET}")
        print(f"  • Install Node.js dependencies: {BLUE}cd backend/websocket && npm install{RESET}")
        print(f"  • Make script executable: {BLUE}chmod +x backend/start_unified_backend.sh{RESET}")
    
    print(f"\n{BLUE}System Information:{RESET}")
    print(f"  • TypeScript Router Port: 8001")
    print(f"  • Python Backend Port: 8000")
    print(f"  • Frontend WebSocket URL: ws://localhost:8001/ws/vision")
    print()

if __name__ == "__main__":
    main()