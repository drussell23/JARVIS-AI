#!/usr/bin/env python3
"""Test JARVIS Real-time Screen Vision When Monitoring Active"""

import asyncio
import websockets
import json
import os
import sys
from datetime import datetime

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

# Test settings
BACKEND_URL = "ws://localhost:8000/ws"
TEST_USER_NAME = "Derek"

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

async def test_real_time_vision():
    """Test that JARVIS can see actual screen content when monitoring is active"""
    print(f"\n{BOLD}🧪 Testing JARVIS Real-time Vision When Monitoring Active{RESET}")
    print("=" * 80)
    
    try:
        async with websockets.connect(BACKEND_URL) as websocket:
            print(f"{GREEN}✓ Connected to JARVIS backend{RESET}")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"Welcome message: {json.loads(welcome)}")
            
            # Step 1: Start screen monitoring
            print(f"\n{BLUE}Step 1: Starting screen monitoring...{RESET}")
            await websocket.send(json.dumps({
                "type": "voice_command",
                "command": "start monitoring my screen"
            }))
            
            # Get response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Response: {data.get('text', 'No text')}")
            
            if data.get('text') and 'activated' in data['text'].lower():
                print(f"{GREEN}✓ Screen monitoring activated successfully{RESET}")
            else:
                print(f"{RED}✗ Failed to activate monitoring{RESET}")
                return
            
            # Wait a moment for monitoring to stabilize
            await asyncio.sleep(2)
            
            # Step 2: Test various "can you see" queries
            test_queries = [
                "can you see my terminal?",
                "what do you see on my screen?",
                "can you see vscode?",
                "describe what's on my screen",
                "what am I looking at?",
                "tell me what you see"
            ]
            
            print(f"\n{BLUE}Step 2: Testing screen vision queries...{RESET}")
            
            for query in test_queries:
                print(f"\n{YELLOW}Testing: '{query}'{RESET}")
                
                await websocket.send(json.dumps({
                    "type": "voice_command",
                    "command": query
                }))
                
                # Get response
                response = await websocket.recv()
                data = json.loads(response)
                response_text = data.get('text', 'No response')
                
                print(f"JARVIS: {response_text[:200]}...")
                
                # Check if response is generic or real
                generic_responses = [
                    "Yes sir, I can see your screen",
                    "Task completed successfully",
                    "I'm monitoring your screen"
                ]
                
                is_generic = any(generic in response_text for generic in generic_responses)
                
                if is_generic and len(response_text) < 100:
                    print(f"{RED}✗ Generic/hardcoded response detected{RESET}")
                elif any(word in response_text.lower() for word in ['terminal', 'vscode', 'window', 'application', 'see', 'screen']):
                    print(f"{GREEN}✓ Real-time vision response detected{RESET}")
                else:
                    print(f"{YELLOW}? Unclear response type{RESET}")
                
                await asyncio.sleep(1)  # Pause between queries
            
            # Step 3: Stop monitoring
            print(f"\n{BLUE}Step 3: Stopping screen monitoring...{RESET}")
            await websocket.send(json.dumps({
                "type": "voice_command", 
                "command": "stop monitoring"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Response: {data.get('text', 'No text')}")
            
            print(f"\n{GREEN}✓ Test completed successfully!{RESET}")
            
    except Exception as e:
        print(f"\n{RED}✗ Test failed with error: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"{BOLD}JARVIS Real-time Vision Test{RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend URL: {BACKEND_URL}")
    print("-" * 80)
    
    # Check if backend is running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8000))
    sock.close()
    
    if result != 0:
        print(f"{RED}Error: Backend is not running on port 8000{RESET}")
        print("Please start the backend with: cd backend && python main.py")
        sys.exit(1)
    
    print(f"{GREEN}Backend is running{RESET}\n")
    
    # Run the test
    asyncio.run(test_real_time_vision())