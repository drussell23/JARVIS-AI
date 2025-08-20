#!/usr/bin/env python3
"""
Test imports to debug the import error
"""
import sys
import os

# Add backend to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

print(f"Python path: {sys.path[:3]}")
print(f"Current dir: {os.getcwd()}")
print(f"Backend dir: {backend_dir}")

try:
    print("\nTesting imports...")
    
    # Test basic imports
    print("1. Importing memory manager...")
    from memory.memory_manager import M1MemoryManager
    print("   ✓ Success")
    
    print("2. Importing optimization config...")
    from memory.optimization_config import optimization_config
    print("   ✓ Success")
    
    print("3. Importing intelligent optimizer...")
    from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
    print("   ✓ Success")
    
    print("4. Importing dynamic chatbot...")
    from backend.chatbots.dynamic_chatbot import DynamicChatbot
    print("   ✓ Success")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()