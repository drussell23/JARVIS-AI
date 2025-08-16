#!/usr/bin/env python3
"""
Test the quantized LLM directly for math capabilities
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from chatbots.quantized_llm_wrapper import get_quantized_llm

def test_quantized_math():
    """Test math with quantized LLM directly"""
    print("🤖 Testing Quantized LLM Math Capabilities")
    print("=" * 50)
    
    # Get the quantized LLM
    llm = get_quantized_llm()
    
    # Initialize it
    print("⏳ Initializing quantized model...")
    if not llm.initialize():
        print("❌ Failed to initialize model. Please run: python setup_m1_optimized_llm.py")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test math queries
    math_queries = [
        ("What is 2 + 2?", "4"),
        ("Calculate 15 * 8", "120"),
        ("What is 100 divided by 4?", "25"),
        ("What's 7 plus 13?", "20"),
        ("What is the square root of 144?", "12"),
        ("Calculate 10% of 250", "25")
    ]
    
    print("\n🧮 Math Test Results:")
    print("-" * 50)
    
    for query, expected in math_queries:
        print(f"\n📝 Query: {query}")
        print(f"📊 Expected: {expected}")
        
        # Generate response
        response = llm.generate(query, max_tokens=50, temperature=0.1)
        
        # Clean up response
        response = response.strip()
        
        print(f"🤖 Response: {response}")
        
        # Check if answer is correct
        if expected in response:
            print("✅ Correct!")
        else:
            print("❌ Incorrect")
    
    # Show memory usage
    stats = llm.get_memory_usage()
    print(f"\n💾 Memory Usage:")
    print(f"   - Model Size: {stats['model_size_mb']:.1f} MB")
    print(f"   - System Memory: {stats['system_memory_percent']:.1f}%")
    print(f"   - Available: {stats['available_memory_gb']:.1f} GB")

if __name__ == "__main__":
    test_quantized_math()