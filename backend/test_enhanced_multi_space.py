#!/usr/bin/env python3
"""Test enhanced multi-space intelligence system"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension

async def test_enhanced_detection():
    """Test the enhanced multi-space detection"""
    print("🚀 Testing Enhanced Multi-Space Intelligence")
    print("=" * 80)
    
    extension = MultiSpaceIntelligenceExtension()
    
    # Comprehensive test queries
    test_queries = [
        # Original issue
        "can you see the Cursor IDE in the other desktop space?",
        
        # Edge cases
        "is Xcode running anywhere?",
        "might Chrome be open somewhere?",
        "check if Terminal is active across my screens",
        "compare what's on desktop 1 and desktop 2",
        "switch to the space with VS Code",
        "list all open applications everywhere",
        "what am I working on across all workspaces?",
        "show me the difference between my desktops",
        "is there a Spotify window I forgot about?",
        "locate my Figma design file",
        
        # Ambiguous cases
        "where's my work?",
        "find my code",
        "check my setup",
        "what's happening?",
        
        # Complex queries
        "can you see if I have both Chrome and Safari open on different spaces?",
        "which desktop has the most windows?",
        "are there any error messages in my IDEs?",
        "move focus to the Terminal on the other screen",
    ]
    
    print("\n📊 Detection Analysis:")
    print("-" * 80)
    print(f"{'Query':<60} {'Multi-Space':<12} {'Confidence':<10} {'Type'}")
    print("-" * 80)
    
    for query in test_queries:
        should_use = extension.should_use_multi_space(query)
        intent = extension.query_detector.detect_intent(query)
        
        # Show decision factors for transparency
        factors = []
        if intent.target_app:
            factors.append("app")
        if intent.space_references:
            factors.append("space_ref")
        if intent.context_hints:
            factors.append("context")
            
        print(f"{query[:60]:<60} {'YES ✅' if should_use else 'NO ❌':<12} "
              f"{intent.confidence:.2f}        {intent.query_type.value}")
        
        if factors:
            print(f"  └─ Factors: {', '.join(factors)}")
        if intent.target_app:
            print(f"  └─ App: {intent.target_app}")
            
    # Test robustness with variations
    print("\n\n🔍 Variation Robustness Test:")
    print("-" * 80)
    
    base_queries = [
        "can you see {} in the other desktop space?",
        "is {} running on another screen?",
        "where is {}?",
        "check {} across all spaces"
    ]
    
    test_apps = ["IntelliJ IDEA", "Discord", "Notion", "Adobe Photoshop", "OmniGraffle"]
    
    success_count = 0
    total_count = 0
    
    for template in base_queries:
        for app in test_apps:
            query = template.format(app)
            should_use = extension.should_use_multi_space(query)
            intent = extension.query_detector.detect_intent(query)
            
            total_count += 1
            if should_use and intent.target_app:
                success_count += 1
                
    print(f"Robustness Score: {success_count}/{total_count} "
          f"({success_count/total_count*100:.1f}%) correctly detected")
    
    # Edge case stress test
    print("\n\n⚡ Edge Case Stress Test:")
    print("-" * 80)
    
    edge_cases = [
        "s p a c e s",  # Spaced out
        "SHOW ME ALL DESKTOPS",  # All caps
        "whatsontheotherdesktop",  # No spaces
        "can u c cursor on other scrn?",  # Abbreviated
        "🖥️ where is my app? 🤔",  # Emojis
        "desktop/space/screen with code",  # Slashes
        "the-other-workspace",  # Hyphens
    ]
    
    for query in edge_cases:
        should_use = extension.should_use_multi_space(query)
        intent = extension.query_detector.detect_intent(query)
        print(f"{query:<40} → Multi-space: {'YES' if should_use else 'NO'} "
              f"(confidence: {intent.confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(test_enhanced_detection())