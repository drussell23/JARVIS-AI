#!/usr/bin/env python3
"""
Test JARVIS math capabilities with real LLMs
"""

import asyncio
import sys
sys.path.append('backend')

async def test_jarvis_math():
    from core import JARVISCore
    
    print("🧮 Testing JARVIS Math Capabilities")
    print("=" * 50)
    
    jarvis = JARVISCore()
    
    # Test queries
    math_queries = [
        "What is 2+2?",
        "Calculate 15 * 8",
        "What is 100 divided by 4?",
        "What's 7 plus 13?",
        "Compute the square root of 144",
        "What is 10% of 250?"
    ]
    
    for query in math_queries:
        print(f"\n📝 Query: {query}")
        
        try:
            result = await jarvis.process_query(query)
            response = result['response']
            model_tier = result['metadata']['model_tier']
            
            # Extract just the answer part if it's long
            if len(response) > 100:
                # Try to find the number in the response
                import re
                numbers = re.findall(r'\b\d+\.?\d*\b', response)
                if numbers:
                    print(f"✅ Answer: {numbers[0]} (from {response[:50]}...)")
                else:
                    print(f"📄 Response: {response[:100]}...")
            else:
                print(f"✅ Answer: {response}")
            
            print(f"🤖 Model: {model_tier}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("💡 Note: Language models can do basic math, but may")
    print("   occasionally make errors. For critical calculations,")
    print("   use LangChain mode (when memory < 50%) for exact results.")

if __name__ == "__main__":
    asyncio.run(test_jarvis_math())