#!/usr/bin/env python3
"""
Apply Context Awareness to JARVIS
=================================

This script modifies the JARVIS backend to add context awareness
for handling locked screen scenarios.
"""

import os
import re
import shutil
from pathlib import Path

def apply_context_awareness():
    """Apply context awareness patches to main.py"""
    
    main_path = Path("main.py")
    if not main_path.exists():
        print("Error: main.py not found!")
        return False
        
    # Read the current main.py
    with open(main_path, 'r') as f:
        content = f.read()
        
    # Backup original
    backup_path = Path("main.py.backup_context")
    if not backup_path.exists():
        shutil.copy(main_path, backup_path)
        print(f"Created backup: {backup_path}")
        
    # Check if already patched
    if "context_aware_integration" in content:
        print("Context awareness already integrated!")
        return True
        
    # Find the process_command function
    pattern = r'(@app\.post\("/api/command"\)\s*async def process_command.*?)(try:\s*from api\.unified_command_processor import UnifiedCommandProcessor\s*processor = UnifiedCommandProcessor\(\)\s*result = await processor\.process_command\(command\))'
    
    replacement = r'''\1try:
        from api.unified_command_processor import UnifiedCommandProcessor
        from api.context_aware_integration import wrap_command_processor_with_context
        
        # Create processor with context awareness
        processor = UnifiedCommandProcessor()
        context_processor = wrap_command_processor_with_context(processor)
        
        # Process with context awareness
        result = await context_processor.process_command_with_context(command)'''
    
    # Apply the patch
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print("Warning: Could not find the pattern to patch. Trying alternative approach...")
        
        # Alternative: Find the specific lines and replace
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'from api.unified_command_processor import UnifiedCommandProcessor' in line and i > 0 and 'try:' in lines[i-1]:
                # Found the import line
                new_lines.append(line)
                new_lines.append('        from api.context_aware_integration import wrap_command_processor_with_context')
                i += 1
                
                # Skip to processor creation
                while i < len(lines) and 'processor = UnifiedCommandProcessor()' not in lines[i]:
                    new_lines.append(lines[i])
                    i += 1
                    
                if i < len(lines):
                    new_lines.append('        ')
                    new_lines.append('        # Create processor with context awareness')
                    new_lines.append('        processor = UnifiedCommandProcessor()')
                    new_lines.append('        context_processor = wrap_command_processor_with_context(processor)')
                    i += 1
                    
                    # Replace the process_command call
                    while i < len(lines) and 'result = await processor.process_command(command)' not in lines[i]:
                        new_lines.append(lines[i])
                        i += 1
                        
                    if i < len(lines):
                        new_lines.append('        ')
                        new_lines.append('        # Process with context awareness')
                        new_lines.append('        result = await context_processor.process_command_with_context(command)')
                        i += 1
            else:
                new_lines.append(line)
                i += 1
                
        new_content = '\n'.join(new_lines)
        
    # Write the modified content
    with open(main_path, 'w') as f:
        f.write(new_content)
        
    print("✅ Context awareness integration applied successfully!")
    print("\nThe following changes were made:")
    print("1. Added import for context_aware_integration")
    print("2. Wrapped UnifiedCommandProcessor with context awareness")
    print("3. Commands will now check for locked screen and handle accordingly")
    print("\nRestart the backend for changes to take effect.")
    
    return True


def create_example_config():
    """Create example configuration for context awareness"""
    
    config_dir = Path("backend/context_intelligence/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "context_config.py"
    if not config_path.exists():
        config_content = '''"""
Context Intelligence Configuration
=================================
"""

from enum import Enum
from typing import Dict, Any, Optional


class ContextMode(Enum):
    """Context intelligence operation modes"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ContextConfig:
    """Configuration for context intelligence"""
    
    def __init__(self):
        self.mode = ContextMode.STANDARD
        self.monitoring_enabled = True
        self.proactive_enabled = False
        self.config = {
            "monitoring.enabled": True,
            "monitoring.poll_interval": 0.5,
            "proactive.enabled": False,
            "screen_lock.auto_unlock": True,
            "screen_lock.confirm_before_unlock": True
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any, persist: bool = True):
        """Set configuration value"""
        self.config[key] = value
        
    def get_mode_config(self) -> Dict[str, Any]:
        """Get mode-specific configuration"""
        if self.mode == ContextMode.MINIMAL:
            return {
                "monitoring.enabled": False,
                "proactive.enabled": False
            }
        elif self.mode == ContextMode.ADVANCED:
            return {
                "monitoring.enabled": True,
                "proactive.enabled": True,
                "monitoring.poll_interval": 0.3
            }
        else:  # STANDARD
            return {
                "monitoring.enabled": True,
                "proactive.enabled": False,
                "monitoring.poll_interval": 0.5
            }


# Global configuration instance
_config_manager = None


def get_config_manager() -> ContextConfig:
    """Get or create configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ContextConfig()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config_manager().get(key, default)
'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Created configuration file: {config_path}")


if __name__ == "__main__":
    print("Applying context awareness to JARVIS...")
    
    # Create config if needed
    create_example_config()
    
    # Apply the patch
    if apply_context_awareness():
        print("\n✅ Context awareness successfully integrated!")
        print("\nJARVIS will now:")
        print("- Detect when the screen is locked")
        print("- Inform you before unlocking")
        print("- Unlock the screen automatically when needed")
        print("- Proceed with your command after unlocking")
        print("- Provide step-by-step confirmation")
    else:
        print("\n❌ Failed to apply context awareness")