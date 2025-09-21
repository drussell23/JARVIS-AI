#!/usr/bin/env python3
"""
Fix JARVIS Weather Hang Issue
This script patches the weather handling to prevent hanging
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def apply_weather_fix():
    """Apply fix to prevent weather hanging"""
    print("🔧 Applying JARVIS Weather Hang Fix")
    print("=" * 60)
    
    # Read the jarvis_agent_voice.py file
    file_path = "voice/jarvis_agent_voice.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Ensure imports are at module level, not inside function
    import_fix = """# Weather workflow imports (moved to module level)
try:
    from workflows.weather_app_vision_unified import execute_weather_app_workflow
    from system_control.macos_controller import MacOSController
    WEATHER_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Weather imports not available: {e}")
    WEATHER_IMPORTS_AVAILABLE = False
    execute_weather_app_workflow = None
    MacOSController = None
"""
    
    # Add imports after the logger definition if not already there
    if "WEATHER_IMPORTS_AVAILABLE" not in content:
        import_pos = content.find("logger = logging.getLogger(__name__)")
        if import_pos > -1:
            end_pos = content.find("\n", import_pos) + 1
            content = content[:end_pos] + "\n" + import_fix + "\n" + content[end_pos:]
            print("✅ Added weather imports at module level")
    
    # Fix 2: Simplify the weather handler to remove nested timeouts
    weather_handler_fix = '''    async def _handle_weather_command(self, text: str) -> str:
        """Handle weather-related commands using VISION to read Weather app"""
        logger.info(f"[WEATHER HANDLER] Starting weather command processing: {text}")
        
        # Check if weather imports are available
        if not WEATHER_IMPORTS_AVAILABLE:
            logger.error("[WEATHER HANDLER] Weather imports not available!")
            return f"I'm unable to check the weather - system not properly configured, {self.user_name}."
        
        # Check vision handler
        if not hasattr(self, 'vision_handler') or not self.vision_handler:
            logger.error("[WEATHER HANDLER] No vision handler available!")
            return f"I need my vision capabilities to check the weather, {self.user_name}."
        
        try:
            # Get or create controller
            controller = self.controller if hasattr(self, 'controller') else MacOSController()
            logger.info(f"[WEATHER HANDLER] Using controller: {controller}")
            
            # Single timeout for the entire operation
            logger.info("[WEATHER HANDLER] Calling weather workflow...")
            vision_response = await asyncio.wait_for(
                execute_weather_app_workflow(controller, self.vision_handler, text),
                timeout=20.0  # 20 second timeout for everything
            )
            
            logger.info(f"[WEATHER HANDLER] Got response: {vision_response[:100] if vision_response else 'None'}...")
            
            # Add personalization if we got a good response
            if vision_response and len(vision_response) > 10:
                if self.user_name and self.user_name != "User":
                    vision_response += f", {self.user_name}"
                return vision_response
            else:
                # Open Weather app as fallback
                logger.warning("[WEATHER HANDLER] Empty response, opening Weather app")
                try:
                    import subprocess
                    subprocess.run(['open', '-a', 'Weather'], check=False)
                    return f"I've opened the Weather app for you to check the forecast, {self.user_name}."
                except:
                    return f"Please check the Weather app for the forecast, {self.user_name}."
                    
        except asyncio.TimeoutError:
            logger.error("[WEATHER HANDLER] Weather operation timed out after 20s")
            # Try to at least open the Weather app
            try:
                import subprocess
                subprocess.run(['open', '-a', 'Weather'], check=False)
                return f"The weather check is taking too long. I've opened the Weather app for you, {self.user_name}."
            except:
                return f"I'm having trouble accessing the weather. Please check the Weather app, {self.user_name}."
                
        except Exception as e:
            logger.error(f"[WEATHER HANDLER] Error: {e}", exc_info=True)
            return f"I encountered an error checking the weather. Please try again, {self.user_name}."
'''
    
    # Find and replace the weather handler
    start_marker = "    async def _handle_weather_command(self, text: str) -> str:"
    end_marker = "    async def _force_vision_weather_read"
    
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos > -1 and end_pos > -1:
        # Replace the entire weather handler
        content = content[:start_pos] + weather_handler_fix + "\n" + content[end_pos:]
        print("✅ Replaced weather handler with simplified version")
    else:
        print("⚠️  Could not find weather handler to replace")
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("\n✅ Weather hang fix applied successfully!")
    print("\nThe fix includes:")
    print("1. Module-level imports to prevent import errors")
    print("2. Simplified timeout structure (single 20s timeout)")
    print("3. Better error handling and logging")
    print("4. Automatic Weather app opening as fallback")
    print("\nRestart your JARVIS server for the fix to take effect.")


if __name__ == "__main__":
    apply_weather_fix()