#!/usr/bin/env python3
"""
Test script to verify the complete monitoring integration flow:
1. Command classification
2. Purple indicator activation  
3. Vision status updates
4. WebSocket broadcasting
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_monitoring_integration():
    """Test the complete monitoring integration flow"""
    
    print("\n🧪 Testing Monitoring Integration\n")
    
    # Test 1: Command Classification
    print("1️⃣ Testing Command Classification...")
    try:
        from vision.monitoring_command_classifier import classify_monitoring_command
        
        test_commands = [
            ("start monitoring my screen", False),
            ("stop monitoring", True),
            ("what do you see?", False),
            ("is monitoring active?", False),
            ("enable screen monitoring capabilities", False)
        ]
        
        for command, current_state in test_commands:
            result = classify_monitoring_command(command, current_state)
            print(f"   Command: '{command}' (monitoring_active={current_state})")
            print(f"   → Type: {result['type'].value}, Action: {result['action'].value}, Confidence: {result['confidence']:.2f}")
            print()
        
        print("✅ Command classification working\n")
    except Exception as e:
        print(f"❌ Command classification failed: {e}\n")
        return
    
    # Test 2: State Manager
    print("2️⃣ Testing State Manager...")
    try:
        from vision.monitoring_state_manager import get_state_manager, MonitoringState
        
        state_manager = get_state_manager()
        
        # Test state transitions
        print(f"   Initial state: {state_manager.current_state.value}")
        
        # Check if we can start
        can_start, reason = state_manager.can_start_monitoring()
        print(f"   Can start monitoring: {can_start} {f'({reason})' if reason else ''}")
        
        if can_start:
            # Transition to activating
            await state_manager.transition_to(MonitoringState.ACTIVATING)
            print(f"   Transitioned to: {state_manager.current_state.value}")
            
            # Simulate component activation
            state_manager.update_component_status('vision_intelligence', True)
            state_manager.update_component_status('macos_indicator', True)
            print("   Components activated")
            
            # Check state (should auto-transition to active)
            await asyncio.sleep(0.1)  # Give time for auto-transition
            print(f"   Final state: {state_manager.current_state.value}")
        
        print("✅ State manager working\n")
    except Exception as e:
        print(f"❌ State manager failed: {e}\n")
        return
    
    # Test 3: macOS Indicator Controller
    print("3️⃣ Testing macOS Indicator Controller...")
    try:
        from vision.macos_indicator_controller import get_indicator_controller
        
        indicator = get_indicator_controller()
        
        # Check permissions
        print("   Checking permissions...")
        perm_result = await indicator.ensure_permissions()
        print(f"   Permissions granted: {perm_result['granted']}")
        
        if not perm_result['granted']:
            print("   ⚠️  Screen recording permission required")
            print("   Instructions:")
            for instruction in perm_result.get('instructions', []):
                print(f"     - {instruction}")
        else:
            # Test activation
            print("   Activating indicator...")
            result = await indicator.activate_indicator()
            print(f"   Activation result: {'✅ Success' if result['success'] else '❌ Failed'}")
            
            if result['success']:
                # Check status
                status = indicator.get_indicator_status()
                print(f"   Indicator active: {status['active']}")
                
                # Wait a bit
                print("   Waiting 3 seconds...")
                await asyncio.sleep(3)
                
                # Deactivate
                print("   Deactivating indicator...")
                result = await indicator.deactivate_indicator()
                print(f"   Deactivation result: {'✅ Success' if result['success'] else '❌ Failed'}")
        
        print("✅ Indicator controller working\n")
    except Exception as e:
        print(f"❌ Indicator controller failed: {e}\n")
    
    # Test 4: Vision Command Handler Integration
    print("4️⃣ Testing Vision Command Handler Integration...")
    try:
        from api.vision_command_handler import vision_command_handler
        
        # Initialize with mock API key
        await vision_command_handler.initialize_intelligence(api_key="test-key")
        
        # Test monitoring control command
        print("   Testing 'start monitoring my screen' command...")
        result = await vision_command_handler.handle_command("start monitoring my screen")
        
        print(f"   Handled: {result.get('handled', False)}")
        print(f"   Response: {result.get('response', 'No response')[:100]}...")
        print(f"   Monitoring active: {result.get('monitoring_active', False)}")
        
        print("✅ Vision command handler working\n")
    except Exception as e:
        print(f"❌ Vision command handler failed: {e}\n")
    
    # Test 5: Complete Flow
    print("5️⃣ Testing Complete Integration Flow...")
    try:
        # Reset state
        state_manager = get_state_manager()
        if state_manager.current_state != MonitoringState.INACTIVE:
            await state_manager.transition_to(MonitoringState.INACTIVE)
        
        print("   Starting complete flow test...")
        
        # Simulate full command flow
        command = "start monitoring my screen"
        print(f"   User command: '{command}'")
        
        # Process through vision handler
        result = await vision_command_handler.handle_command(command)
        
        # Check results
        print(f"   → Response: {result.get('response', 'No response')}")
        print(f"   → Monitoring active: {result.get('monitoring_active', False)}")
        print(f"   → State: {state_manager.current_state.value}")
        print(f"   → Purple indicator: {result.get('indicator_active', False)}")
        
        # Stop monitoring
        print("\n   Stopping monitoring...")
        result = await vision_command_handler.handle_command("stop monitoring")
        print(f"   → Response: {result.get('response', 'No response')}")
        print(f"   → State: {state_manager.current_state.value}")
        
        print("\n✅ Complete integration flow working!")
        
    except Exception as e:
        print(f"\n❌ Complete flow failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Integration test complete!\n")

if __name__ == "__main__":
    asyncio.run(test_monitoring_integration())