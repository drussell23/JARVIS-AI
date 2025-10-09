#!/usr/bin/env python3
"""
Test Voice Unlock Integration
============================

Quick test script to verify voice unlock is properly integrated with JARVIS.
"""

import asyncio
import requests
import json
import sys
from colorama import init, Fore, Style

init(autoreset=True)

# Configuration
BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
VOICE_UNLOCK_STATUS = f"{BASE_URL}/api/voice-unlock/status"
VOICE_UNLOCK_CONFIG = f"{BASE_URL}/api/voice-unlock/config"


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}{text.center(60)}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


def print_status(label, status, details=""):
    """Print status with color coding"""
    if status:
        color = Fore.GREEN
        symbol = "✓"
    else:
        color = Fore.RED
        symbol = "✗"
    
    print(f"{color}{symbol} {label}{Style.RESET_ALL}")
    if details:
        print(f"  {Fore.YELLOW}{details}{Style.RESET_ALL}")


def check_health():
    """Check JARVIS health endpoint"""
    print_header("Checking JARVIS Health")
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Check overall health
            print_status("JARVIS Backend", data.get("status") == "healthy")
            
            # Check components
            components = data.get("components", {})
            print(f"\n{Fore.CYAN}Components:{Style.RESET_ALL}")
            for comp, loaded in components.items():
                print_status(f"  {comp}", loaded)
            
            # Check voice unlock specifically
            voice_unlock = data.get("voice_unlock", {})
            print(f"\n{Fore.CYAN}Voice Unlock Status:{Style.RESET_ALL}")
            print_status("  Available", voice_unlock.get("enabled", False))
            print_status("  Initialized", voice_unlock.get("initialized", False))
            print_status("  API Available", voice_unlock.get("api_available", False))
            
            return True
        else:
            print_status("Health Check", False, f"Status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_status("Backend Connection", False, "Cannot connect to JARVIS backend")
        print(f"\n{Fore.YELLOW}Make sure JARVIS is running:")
        print(f"{Fore.YELLOW}  cd backend && python start_system.py{Style.RESET_ALL}")
        return False
    except Exception as e:
        print_status("Health Check", False, str(e))
        return False


def check_voice_unlock_api():
    """Check voice unlock API endpoints"""
    print_header("Checking Voice Unlock API")
    
    # Check status endpoint
    try:
        response = requests.get(VOICE_UNLOCK_STATUS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_status("Voice Unlock API", True)
            
            print(f"\n{Fore.CYAN}Service Status:{Style.RESET_ALL}")
            print_status("  Initialized", data.get("initialized", False))
            
            # Check services
            services = data.get("services", {})
            if services:
                print(f"\n{Fore.CYAN}Services:{Style.RESET_ALL}")
                for service, available in services.items():
                    print_status(f"  {service}", available)
            
            # Check enrolled users
            enrolled = data.get("enrolled_users", 0)
            print(f"\n{Fore.CYAN}Enrolled Users: {enrolled}{Style.RESET_ALL}")
            
            return True
        else:
            print_status("Voice Unlock API", False, f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_status("Voice Unlock API", False, str(e))
        return False


def check_configuration():
    """Check voice unlock configuration"""
    print_header("Voice Unlock Configuration")
    
    try:
        response = requests.get(VOICE_UNLOCK_CONFIG, timeout=5)
        if response.status_code == 200:
            config = response.json()
            
            print(f"{Fore.CYAN}Enrollment Settings:{Style.RESET_ALL}")
            enrollment = config.get("enrollment", {})
            print(f"  Min samples: {enrollment.get('min_samples', 'N/A')}")
            print(f"  Max samples: {enrollment.get('max_samples', 'N/A')}")
            print(f"  Min quality: {enrollment.get('min_quality_score', 'N/A')}")
            
            print(f"\n{Fore.CYAN}Security Settings:{Style.RESET_ALL}")
            security = config.get("security", {})
            print(f"  Anti-spoofing level: {security.get('anti_spoofing_level', 'N/A')}")
            print(f"  Encrypt voiceprints: {security.get('encrypt_voiceprints', 'N/A')}")
            print(f"  Storage backend: {security.get('storage_backend', 'N/A')}")
            
            print(f"\n{Fore.CYAN}System Integration:{Style.RESET_ALL}")
            system = config.get("system", {})
            print(f"  Integration mode: {system.get('integration_mode', 'N/A')}")
            print(f"  JARVIS responses: {system.get('jarvis_responses', 'N/A')}")
            
            return True
        else:
            print_status("Configuration", False, f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_status("Configuration", False, str(e))
        return False


def test_command(command):
    """Test a voice unlock command through the unified processor"""
    print_header(f"Testing Command: '{command}'")
    
    try:
        # Use the unified command endpoint
        response = requests.post(
            f"{BASE_URL}/api/command",
            json={"command": command},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print_status("Command Processed", True)
            
            print(f"\n{Fore.CYAN}Response:{Style.RESET_ALL}")
            print(f"  Type: {result.get('type', 'unknown')}")
            print(f"  Action: {result.get('action', 'N/A')}")
            print(f"  Message: {result.get('message', 'N/A')}")
            
            if result.get('available_commands'):
                print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
                for cmd in result['available_commands']:
                    print(f"  • {cmd}")
            
            return True
        else:
            print_status("Command Test", False, f"Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_status("Command Test", False, str(e))
        return False


async def test_audio_system():
    """Test audio capture system"""
    print_header("Testing Audio System")
    
    try:
        # Import and test locally
        from voice_unlock.utils.audio_capture import AudioCapture
        
        audio_capture = AudioCapture()
        
        # List devices
        devices = audio_capture.list_devices()
        print(f"{Fore.CYAN}Available Audio Devices:{Style.RESET_ALL}")
        for device in devices:
            print(f"  [{device['index']}] {device['name']} ({device['channels']} channels)")
        
        # Test noise calibration
        print(f"\n{Fore.CYAN}Calibrating noise floor...{Style.RESET_ALL}")
        noise_level = audio_capture.calibrate_noise_floor()
        print_status("Noise Calibration", True, f"Noise floor: {noise_level:.4f}")
        
        return True
        
    except ImportError:
        print_status("Audio System", False, "Voice unlock module not installed")
        print(f"\n{Fore.YELLOW}To install dependencies:")
        print(f"{Fore.YELLOW}  cd backend/voice_unlock && ./install_dependencies.sh{Style.RESET_ALL}")
        return False
    except Exception as e:
        print_status("Audio System", False, str(e))
        return False


def main():
    """Run all integration tests"""
    print_header("JARVIS Voice Unlock Integration Test")
    
    # Track test results
    results = []
    
    # Test 1: Check JARVIS health
    results.append(("JARVIS Health", check_health()))
    
    # Only continue if JARVIS is healthy
    if results[0][1]:
        # Test 2: Check voice unlock API
        results.append(("Voice Unlock API", check_voice_unlock_api()))
        
        # Test 3: Check configuration
        results.append(("Configuration", check_configuration()))
        
        # Test 4: Test command processing
        results.append(("Command Processing", test_command("voice unlock status")))
        
        # Test 5: Test audio system (async)
        loop = asyncio.get_event_loop()
        results.append(("Audio System", loop.run_until_complete(test_audio_system())))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"{Fore.CYAN}Tests Run: {total}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {total - passed}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Results:{Style.RESET_ALL}")
    for test_name, passed in results:
        print_status(f"  {test_name}", passed)
    
    if passed == total:
        print(f"\n{Fore.GREEN}✨ Voice Unlock integration successful!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Next Steps:{Style.RESET_ALL}")
        print("1. Test enrollment: Say 'Hey JARVIS, enroll my voice'")
        print("2. Enable monitoring: Say 'Hey JARVIS, enable voice unlock'")
        print("3. Test unlock: Lock your screen and say your unlock phrase")
    else:
        print(f"\n{Fore.RED}❌ Some tests failed. Please check the errors above.{Style.RESET_ALL}")
        
        if not results[0][1]:  # JARVIS not running
            print(f"\n{Fore.YELLOW}Start JARVIS first:")
            print(f"  cd backend && python start_system.py{Style.RESET_ALL}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())