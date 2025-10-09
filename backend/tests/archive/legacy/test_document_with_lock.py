#!/usr/bin/env python3
"""
Test script to verify context-aware document creation with screen lock detection
"""
import asyncio
import subprocess
import time
import json

async def test_document_creation_with_lock():
    """Test the document creation flow with screen lock"""

    print("=" * 60)
    print("TESTING CONTEXT-AWARE DOCUMENT CREATION")
    print("=" * 60)

    # Step 1: Lock the screen
    print("\n[STEP 1] Locking the screen...")
    subprocess.run([
        'osascript', '-e',
        'tell application "System Events" to key code 12 using {control down, command down}'
    ])
    time.sleep(2)

    # Step 2: Send document creation command through WebSocket
    print("\n[STEP 2] Sending document creation command...")

    import websocket
    import json

    # Connect to WebSocket
    ws_url = "ws://localhost:8000/ws"
    ws = websocket.create_connection(ws_url)

    # Send document creation command
    command = {
        "type": "command",
        "text": "write me an essay on quantum computing"
    }

    print(f"Sending: {command}")
    ws.send(json.dumps(command))

    # Wait for response
    print("\n[STEP 3] Waiting for response...")
    timeout = 30
    start_time = time.time()

    responses = []
    while time.time() - start_time < timeout:
        try:
            result = ws.recv()
            response = json.loads(result)
            print(f"Received: {json.dumps(response, indent=2)}")
            responses.append(response)

            # Check if we got a completion message
            if response.get("type") == "command_response":
                break

        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            print(f"Error: {e}")
            break

    ws.close()

    # Step 4: Check the logs
    print("\n[STEP 4] Checking logs for context-aware behavior...")
    logs = subprocess.run([
        'tail', '-n', '100',
        '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/logs/jarvis_optimized_*.log'
    ], capture_output=True, text=True)

    # Look for context-aware logs
    context_logs = []
    for line in logs.stdout.split('\n'):
        if 'CONTEXT AWARE' in line:
            context_logs.append(line)

    if context_logs:
        print("\n✅ CONTEXT-AWARE LOGS FOUND:")
        for log in context_logs[-10:]:  # Show last 10 context-aware logs
            print(f"  {log}")
    else:
        print("\n❌ NO CONTEXT-AWARE LOGS FOUND!")

    # Check for screen lock detection
    lock_detected = any('screen_locked' in log or 'Screen is LOCKED' in log for log in context_logs)
    if lock_detected:
        print("\n✅ Screen lock was detected!")
    else:
        print("\n❌ Screen lock was NOT detected!")

    # Check for unlock notification
    notification_sent = any('Speaking unlock notification' in log for log in context_logs)
    if notification_sent:
        print("\n✅ Unlock notification was sent!")
    else:
        print("\n❌ Unlock notification was NOT sent!")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return {
        "context_aware_logs": len(context_logs) > 0,
        "screen_lock_detected": lock_detected,
        "notification_sent": notification_sent,
        "responses": responses
    }

if __name__ == "__main__":
    try:
        result = asyncio.run(test_document_creation_with_lock())

        print("\n📊 TEST RESULTS:")
        print(f"  Context-aware handler active: {result['context_aware_logs']}")
        print(f"  Screen lock detected: {result['screen_lock_detected']}")
        print(f"  Notification sent: {result['notification_sent']}")

        if all([result['context_aware_logs'], result['screen_lock_detected'], result['notification_sent']]):
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED - Check the implementation")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()