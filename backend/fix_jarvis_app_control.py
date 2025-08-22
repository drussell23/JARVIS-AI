#!/usr/bin/env python3
"""
Fix JARVIS App Control Integration
Connects JARVIS AI Core to System Control for actual command execution
"""

import os
import sys
import asyncio
import json
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.jarvis_ai_core import get_jarvis_ai_core
from system_control.macos_controller import MacOSController
from system_control.claude_command_interpreter import ClaudeCommandInterpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_safari_command():
    """Test opening Safari through JARVIS"""
    print("\n🧪 Testing JARVIS App Control Integration")
    print("=" * 50)
    
    # Initialize components
    print("\n1️⃣ Initializing JARVIS AI Core...")
    ai_core = get_jarvis_ai_core()
    
    print("2️⃣ Initializing System Controller...")
    controller = MacOSController()
    
    print("3️⃣ Initializing Command Interpreter...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found!")
        return
    
    interpreter = ClaudeCommandInterpreter(api_key)
    
    # Test command processing
    print("\n4️⃣ Testing command: 'open Safari'")
    
    # Process through AI Core
    command_analysis = await ai_core.process_speech_command("open Safari")
    print(f"\n📊 AI Core Analysis:")
    print(json.dumps(command_analysis, indent=2))
    
    # Process through Command Interpreter
    intent = await interpreter.interpret_command("open Safari")
    print(f"\n🎯 Command Intent:")
    print(f"  Action: {intent.action}")
    print(f"  Target: {intent.target}")
    print(f"  Category: {intent.category}")
    print(f"  Confidence: {intent.confidence}")
    
    # Execute the command
    if intent.confidence > 0.5:
        print(f"\n5️⃣ Executing: {intent.action} {intent.target}")
        result = await interpreter.execute_intent(intent)
        print(f"\n✅ Result: {result.message}")
    else:
        print("\n❌ Low confidence, not executing")
    
    # Test direct controller
    print("\n6️⃣ Testing direct controller...")
    success, message = controller.open_application("Safari")
    print(f"Direct result: {message}")


async def create_integrated_executor():
    """Create an integrated command executor for JARVIS"""
    
    class IntegratedCommandExecutor:
        def __init__(self):
            self.ai_core = get_jarvis_ai_core()
            self.controller = MacOSController()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.interpreter = ClaudeCommandInterpreter(api_key)
            
        async def execute_command(self, command: str) -> dict:
            """Execute a command through the full pipeline"""
            
            # 1. Process through AI Core for analysis
            logger.info(f"Processing command: {command}")
            analysis = await self.ai_core.process_speech_command(command)
            
            # 2. Interpret command for system execution
            intent = await self.interpreter.interpret_command(command)
            
            # 3. Execute if confident
            if intent.confidence > 0.5:
                result = await self.interpreter.execute_intent(intent)
                
                # 4. Update AI Core with result
                await self.ai_core.execute_task({
                    "action": intent.action,
                    "target": intent.target,
                    "result": result.success,
                    "message": result.message
                })
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "analysis": analysis,
                    "intent": {
                        "action": intent.action,
                        "target": intent.target,
                        "confidence": intent.confidence
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Low confidence in command interpretation",
                    "analysis": analysis,
                    "confidence": intent.confidence
                }
    
    return IntegratedCommandExecutor()


def create_jarvis_integration_patch():
    """Create a patch file to integrate system control into JARVIS AI Core"""
    
    patch_content = '''
# Add this to jarvis_ai_core.py after imports

from system_control.macos_controller import MacOSController
from system_control.claude_command_interpreter import ClaudeCommandInterpreter

# Add to __init__ method:
        # Initialize system control
        self.controller = MacOSController()
        self.command_interpreter = ClaudeCommandInterpreter(api_key)
        logger.info("System control integration initialized")

# Replace execute_task method with:
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using Claude's intelligence and system control"""
        try:
            # If task has a direct command, execute it
            if "command" in task:
                intent = await self.command_interpreter.interpret_command(task["command"])
                if intent.confidence > 0.5:
                    result = await self.command_interpreter.execute_intent(intent)
                    return {
                        "task": task,
                        "executed": True,
                        "success": result.success,
                        "message": result.message,
                        "executed_at": datetime.now().isoformat()
                    }
            
            # Otherwise use existing execution plan logic
            # ... existing code ...
'''
    
    with open("jarvis_ai_core_integration.patch", "w") as f:
        f.write(patch_content)
    
    print("\n📝 Created integration patch: jarvis_ai_core_integration.patch")
    print("Apply this patch to backend/core/jarvis_ai_core.py to enable system control")


async def main():
    """Run tests and create integration components"""
    
    # Test Safari command
    await test_safari_command()
    
    # Create integrated executor
    print("\n\n🔧 Creating Integrated Command Executor...")
    executor = await create_integrated_executor()
    
    # Test integrated execution
    print("\n\n🧪 Testing Integrated Execution...")
    result = await executor.execute_command("open Safari")
    print(f"\nIntegrated Result:")
    print(json.dumps(result, indent=2))
    
    # Create patch file
    create_jarvis_integration_patch()
    
    print("\n\n✅ Integration test complete!")
    print("\n📋 Next steps:")
    print("1. Apply the patch to jarvis_ai_core.py")
    print("2. Restart JARVIS backend")
    print("3. Test with 'Hey JARVIS, open Safari'")


if __name__ == "__main__":
    asyncio.run(main())