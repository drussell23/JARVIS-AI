#!/usr/bin/env python3
"""
Comprehensive Test Suite for Context Awareness Integration
==========================================================

Tests the enhanced context awareness in:
- unified_command_processor.py
- async_pipeline.py
"""

import asyncio
import subprocess
import json
import time
import sys
import websocket
from typing import Dict, Any, List, Tuple
from datetime import datetime

class ContextAwarenessTestSuite:
    """
    Complete test suite for context awareness features
    """

    def __init__(self):
        self.ws_url = "ws://localhost:8000/ws"
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def print_header(self, title: str):
        """Print a formatted test header"""
        print("\n" + "=" * 70)
        print(f" {title} ")
        print("=" * 70)

    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Print test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n[{status}] {test_name}")
        if details:
            print(f"  Details: {details}")

        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    async def test_screen_lock_detection(self) -> bool:
        """Test 1: Screen Lock Detection"""
        self.print_header("TEST 1: Screen Lock Detection")

        try:
            # Lock the screen
            print("Locking screen...")
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to key code 12 using {control down, command down}'
            ])
            await asyncio.sleep(2)

            # Check if lock is detected
            from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
            detector = get_screen_lock_detector()
            is_locked = await detector.is_screen_locked()

            self.print_result(
                "Screen lock detection",
                is_locked,
                f"Screen lock status: {'LOCKED' if is_locked else 'UNLOCKED'}"
            )

            # Unlock for next tests
            if is_locked:
                print("Unlocking screen for next tests...")
                await detector.handle_screen_lock_context("test")
                await asyncio.sleep(2)

            return is_locked

        except Exception as e:
            self.print_result("Screen lock detection", False, str(e))
            return False

    async def test_unified_processor_context(self) -> bool:
        """Test 2: Unified Command Processor Context Awareness"""
        self.print_header("TEST 2: Unified Processor Context Integration")

        try:
            from api.unified_command_processor import UnifiedCommandProcessor
            processor = UnifiedCommandProcessor()

            # Test simple command with context
            print("Testing simple command processing with context...")
            result = await processor.process_command("what time is it")

            # Check if context was gathered
            has_context = False
            if isinstance(result, dict):
                # Look for context-related keys
                if any(key in result for key in ['context', 'system_context', 'steps_taken']):
                    has_context = True

            self.print_result(
                "Context gathering in unified processor",
                has_context,
                f"Result contains context: {has_context}"
            )

            # Test compound command with context
            print("\nTesting compound command with context...")
            compound_result = await processor.process_command(
                "lock my screen and then tell me the weather"
            )

            is_compound_aware = False
            if isinstance(compound_result, dict):
                if compound_result.get('steps_taken') or compound_result.get('context'):
                    is_compound_aware = True

            self.print_result(
                "Compound command context awareness",
                is_compound_aware,
                f"Compound commands use context: {is_compound_aware}"
            )

            return has_context and is_compound_aware

        except Exception as e:
            self.print_result("Unified processor context", False, str(e))
            return False

    async def test_async_pipeline_context(self) -> bool:
        """Test 3: Async Pipeline Enhanced Context"""
        self.print_header("TEST 3: Async Pipeline Context Intelligence")

        try:
            from core.async_pipeline import AdvancedAsyncPipeline
            pipeline = AdvancedAsyncPipeline()

            # Test document type detection
            print("Testing document type detection...")

            test_commands = [
                ("create a presentation about AI", "presentation"),
                ("write me a spreadsheet for budget", "spreadsheet"),
                ("draft an email to the team", "correspondence"),
                ("write an essay on climate change", "text_document")
            ]

            detection_success = True
            for command, expected_type in test_commands:
                detected_type = pipeline._detect_document_type(command)
                matches = detected_type == expected_type
                if not matches:
                    detection_success = False
                print(f"  '{command}' → {detected_type} {'✓' if matches else '✗'}")

            self.print_result(
                "Document type detection",
                detection_success,
                "All document types detected correctly" if detection_success else "Some detections failed"
            )

            # Test enhanced system context
            print("\nTesting enhanced system context gathering...")
            context = await pipeline._get_enhanced_system_context()

            has_all_fields = all(
                field in context for field in
                ['screen_locked', 'active_apps', 'network_connected', 'system_load']
            )

            self.print_result(
                "Enhanced system context fields",
                has_all_fields,
                f"Context keys: {list(context.keys())}"
            )

            return detection_success and has_all_fields

        except Exception as e:
            self.print_result("Async pipeline context", False, str(e))
            return False

    async def test_context_aware_handler(self) -> bool:
        """Test 4: Context-Aware Handler Integration"""
        self.print_header("TEST 4: Context-Aware Handler")

        try:
            from context_intelligence.handlers.context_aware_handler import get_context_aware_handler
            handler = get_context_aware_handler()

            # Test with mock command
            print("Testing context-aware command handling...")

            async def mock_execute(cmd: str, context: Dict[str, Any] = None):
                """Mock execution callback"""
                return {
                    "success": True,
                    "message": f"Executed: {cmd}",
                    "used_context": context is not None
                }

            result = await handler.handle_command_with_context(
                "test command",
                execute_callback=mock_execute
            )

            # Check result structure
            has_required_fields = all(
                field in result for field in
                ['success', 'command', 'messages', 'steps_taken', 'context']
            )

            self.print_result(
                "Context handler structure",
                has_required_fields,
                f"Steps taken: {len(result.get('steps_taken', []))}"
            )

            # Check if steps were recorded
            has_steps = len(result.get('steps_taken', [])) > 0

            self.print_result(
                "Step tracking",
                has_steps,
                f"Number of steps: {len(result.get('steps_taken', []))}"
            )

            return has_required_fields and has_steps

        except Exception as e:
            self.print_result("Context-aware handler", False, str(e))
            return False

    async def test_websocket_integration(self) -> bool:
        """Test 5: WebSocket Context Integration"""
        self.print_header("TEST 5: WebSocket Context Flow")

        try:
            print("Connecting to WebSocket...")
            ws = websocket.create_connection(self.ws_url, timeout=5)

            # Send test command
            command = {
                "type": "command",
                "text": "what's the weather"
            }

            print(f"Sending command: {command['text']}")
            ws.send(json.dumps(command))

            # Collect responses
            responses = []
            timeout = 10
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    ws.settimeout(0.5)
                    result = ws.recv()
                    response = json.loads(result)
                    responses.append(response)

                    # Check for context in response
                    if response.get("type") == "command_response":
                        break

                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    break

            ws.close()

            # Check if responses contain context
            has_context_response = any(
                'context' in str(r) or 'steps_taken' in str(r)
                for r in responses
            )

            self.print_result(
                "WebSocket context flow",
                has_context_response,
                f"Received {len(responses)} responses"
            )

            return has_context_response

        except Exception as e:
            self.print_result("WebSocket integration", False, str(e))
            return False

    async def test_full_flow_with_lock(self) -> bool:
        """Test 6: Full Document Creation with Lock"""
        self.print_header("TEST 6: Complete Flow - Document Creation with Lock")

        try:
            # Lock the screen
            print("Locking screen for full flow test...")
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to key code 12 using {control down, command down}'
            ])
            await asyncio.sleep(3)

            # Connect to WebSocket
            print("Connecting to WebSocket...")
            ws = websocket.create_connection(self.ws_url, timeout=5)

            # Send document creation command
            command = {
                "type": "command",
                "text": "write me a short essay on artificial intelligence"
            }

            print(f"Sending: {command['text']}")
            ws.send(json.dumps(command))

            # Monitor responses
            responses = []
            unlock_detected = False
            context_aware_flow = False
            timeout = 30
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    ws.settimeout(0.5)
                    result = ws.recv()
                    response = json.loads(result)
                    responses.append(response)

                    # Check for unlock notification
                    if 'unlock' in str(response).lower() or 'locked' in str(response).lower():
                        unlock_detected = True

                    # Check for context awareness
                    if 'context' in str(response) or 'steps_taken' in str(response):
                        context_aware_flow = True

                    if response.get("type") == "command_response":
                        break

                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    break

            ws.close()

            self.print_result(
                "Screen unlock notification",
                unlock_detected,
                "User was notified about screen lock" if unlock_detected else "No lock notification"
            )

            self.print_result(
                "Context-aware document creation",
                context_aware_flow,
                f"Total responses: {len(responses)}"
            )

            return unlock_detected or context_aware_flow

        except Exception as e:
            self.print_result("Full flow with lock", False, str(e))
            return False

    async def check_log_integration(self) -> bool:
        """Test 7: Log Verification"""
        self.print_header("TEST 7: Log Integration Verification")

        try:
            # Read recent logs
            print("Checking logs for context awareness...")

            logs = subprocess.run([
                'tail', '-n', '200',
                '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/logs/jarvis_optimized_*.log'
            ], capture_output=True, text=True, shell=True)

            log_lines = logs.stdout.split('\n') if logs.stdout else []

            # Check for key indicators
            indicators = {
                "context_aware": "[CONTEXT AWARE]" in logs.stdout,
                "system_context": "_get_full_system_context" in logs.stdout or "_get_enhanced_system_context" in logs.stdout,
                "screen_detection": "screen_locked" in logs.stdout.lower(),
                "document_type": "_detect_document_type" in logs.stdout,
                "pipeline_context": "[PIPELINE]" in logs.stdout and "context" in logs.stdout.lower()
            }

            for key, found in indicators.items():
                self.print_result(
                    f"Log indicator: {key}",
                    found,
                    "Found in logs" if found else "Not found in logs"
                )

            return all(indicators.values())

        except Exception as e:
            self.print_result("Log verification", False, str(e))
            return False

    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "█" * 70)
        print(" JARVIS CONTEXT AWARENESS TEST SUITE ")
        print("█" * 70)
        print(f"\nStarting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run all tests
        tests = [
            ("Screen Lock Detection", self.test_screen_lock_detection),
            ("Unified Processor Context", self.test_unified_processor_context),
            ("Async Pipeline Context", self.test_async_pipeline_context),
            ("Context-Aware Handler", self.test_context_aware_handler),
            ("WebSocket Integration", self.test_websocket_integration),
            ("Full Flow with Lock", self.test_full_flow_with_lock),
            ("Log Integration", self.check_log_integration)
        ]

        for test_name, test_func in tests:
            try:
                await test_func()
            except Exception as e:
                print(f"\n⚠️ Test '{test_name}' encountered error: {e}")
                self.failed_tests += 1
                self.total_tests += 1

        # Print summary
        self.print_header("TEST SUMMARY")
        print(f"\nTotal Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")

        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Context awareness is fully integrated.")
        else:
            print(f"\n⚠️ {self.failed_tests} tests failed. Please check implementation.")

        print("\n" + "█" * 70)


async def main():
    """Main test runner"""
    # Check if backend is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("⚠️ Backend appears to be down. Starting it...")
            subprocess.Popen(["python3", "main.py"],
                           cwd="/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend")
            await asyncio.sleep(5)
    except:
        print("⚠️ Backend not responding. Please ensure it's running.")
        print("Starting backend...")
        subprocess.Popen(["python3", "main.py"],
                       cwd="/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend")
        await asyncio.sleep(5)

    # Run test suite
    suite = ContextAwarenessTestSuite()
    await suite.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()