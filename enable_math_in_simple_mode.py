#!/usr/bin/env python3
"""
Patch to enable math in Simple mode using quantized LLM
"""

import os
import sys
import shutil
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def patch_simple_chatbot():
    """Add math capability to simple chatbot"""
    
    chatbot_path = "backend/chatbots/simple_chatbot.py"
    backup_path = f"backend/chatbots/simple_chatbot.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create backup
    shutil.copy(chatbot_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read the file
    with open(chatbot_path, 'r') as f:
        content = f.read()
    
    # Find the line where we handle "what" questions
    search_str = '''if "what" in user_lower:
                response = "That's a great question! Based on what you're asking, I'd be happy to help explain or provide information."'''
    
    replacement_str = '''if "what" in user_lower:
                # Check if it's a math question
                if any(op in user_input for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided', 'add', 'subtract', 'multiply']):
                    try:
                        # Try to use quantized LLM for math
                        from chatbots.quantized_llm_wrapper import get_quantized_llm
                        llm = get_quantized_llm()
                        if llm.initialized or llm.initialize():
                            math_response = llm.generate(user_input, max_tokens=50, temperature=0.1)
                            response = math_response.strip()
                        else:
                            response = "I'd love to help with that calculation, but I need the language model to be loaded first. Try running: python setup_m1_optimized_llm.py"
                    except:
                        response = "I can help with math, but the calculation engine isn't available right now."
                else:
                    response = "That's a great question! Based on what you're asking, I'd be happy to help explain or provide information."'''
    
    # Replace the content
    if search_str in content:
        content = content.replace(search_str, replacement_str)
        
        # Write back
        with open(chatbot_path, 'w') as f:
            f.write(content)
        
        print("✅ Successfully patched simple_chatbot.py to handle math!")
        print("🧮 Math questions will now use the quantized LLM")
        print("\n📝 To test, restart JARVIS and ask math questions like:")
        print("   - What is 2 + 2?")
        print("   - Calculate 15 * 8")
        print("   - What's 100 divided by 4?")
    else:
        print("❌ Could not find the target code to patch")
        print("The simple chatbot may have been modified")

if __name__ == "__main__":
    patch_simple_chatbot()