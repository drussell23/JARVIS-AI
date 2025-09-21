#!/usr/bin/env python3
"""Demonstrate comprehensive edge case coverage"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension

async def test_edge_cases():
    """Test various edge case scenarios"""
    print("🎯 Comprehensive Edge Case Coverage Test")
    print("=" * 80)
    
    extension = MultiSpaceIntelligenceExtension()
    
    # Categories of edge cases
    edge_case_categories = {
        "🔤 Typos & Misspellings": [
            "can you see Curor in the othr desktop?",  # Misspelled app and 'other'
            "is VSCde running on anothr screen?",       # Misspelled VSCode
            "where is Termnal?",                        # Missing letter
            "show me Spoitfy on different workspace",   # Misspelled Spotify
        ],
        
        "🗣️ Natural Language Variations": [
            "yo where's my Discord at?",               # Casual speech
            "um, is like Chrome open somewhere?",      # Filler words
            "could you perhaps check if Notion is running?",  # Polite/verbose
            "Slack... where?",                         # Minimal query
            "I think I left Figma open on another desktop",  # Conversational
        ],
        
        "🎨 Unusual App Names": [
            "where is 1Password?",                     # Starts with number
            "can you find OBS Studio on other screen?", # Acronym + word
            "is IntelliJ IDEA running anywhere?",      # Multi-word with caps
            "locate Microsoft Teams in my spaces",     # Company + product
            "find iTerm2 across desktops",            # Mixed case with number
        ],
        
        "💬 Ambiguous Queries": [
            "that app I was using",                    # No specific app
            "the thing on the other screen",           # Vague reference
            "my work stuff",                           # Generic description
            "whatever's on desktop 3",                 # Non-specific content
            "you know, that window",                   # Unclear reference
        ],
        
        "🔢 Numeric & Special Cases": [
            "what's on desktop number two?",           # Written number
            "space #3 contents",                       # Hash notation
            "workspace (2) apps",                      # Parentheses
            "desktop two vs desktop three",            # Comparison with words
            "1st screen applications",                 # Ordinal numbers
        ],
        
        "🌐 Multi-language Elements": [
            "où est Chrome?",                          # French
            "find 微信 on other desktop",              # Chinese app name
            "locate café app",                         # Accented characters
            "where is α-editor?",                      # Greek letter
        ],
        
        "🔀 Complex Compound Queries": [
            "is either Chrome or Safari or Firefox open anywhere?",
            "find all my code editors (VS Code, Cursor, Sublime) across spaces",
            "which desktop has Slack but not Discord?",
            "compare Figma on space 1 with Sketch on space 2",
            "any Adobe apps (Photoshop/Illustrator/XD) running?",
        ],
        
        "📱 Modern App Patterns": [
            "where's the Zoom meeting?",               # Video conf apps
            "find my Obsidian vault",                  # Knowledge management
            "is Docker Desktop running?",              # Dev tools
            "locate Linear app",                       # Project management
            "check if Raycast is open",              # Productivity tools
        ],
        
        "🤖 Technical Queries": [
            "which space has port 3000 running?",     # Developer context
            "find the terminal with npm start",       # Process-specific
            "where's my localhost:8080 tab?",         # Web dev
            "locate the Jupyter notebook",            # Data science
            "find my database GUI",                   # Generic technical
        ],
        
        "😵 Extreme Edge Cases": [
            "!!!WHERE IS CHROME!!!",                   # Excessive punctuation
            "c h r o m e ?",                          # Spaced letters
            "WhErE iS sLaCk?",                        # Mixed case chaos
            "find...um...you know...the...thing",     # Excessive hesitation
            "→ other desktop ←",                      # Special characters
        ]
    }
    
    # Test each category
    for category, queries in edge_case_categories.items():
        print(f"\n{category}")
        print("-" * 80)
        
        success_count = 0
        for query in queries:
            should_use = extension.should_use_multi_space(query)
            intent = extension.query_detector.detect_intent(query)
            
            # Determine if it was handled well
            handled_well = (
                should_use or  # Detected as multi-space
                intent.target_app or  # Found an app
                intent.confidence > 0.5  # Reasonable confidence
            )
            
            if handled_well:
                success_count += 1
            
            status = "✅" if handled_well else "❌"
            print(f"{status} {query:<50} | MS: {'Y' if should_use else 'N'} | "
                  f"Conf: {intent.confidence:.2f} | App: {intent.target_app or 'None'}")
        
        print(f"   → Category Success Rate: {success_count}/{len(queries)} "
              f"({success_count/len(queries)*100:.0f}%)")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("📊 Overall Edge Case Handling Statistics:")
    print("=" * 80)
    
    total_queries = sum(len(queries) for queries in edge_case_categories.values())
    handled_queries = 0
    
    for queries in edge_case_categories.values():
        for query in queries:
            should_use = extension.should_use_multi_space(query)
            intent = extension.query_detector.detect_intent(query)
            if should_use or intent.target_app or intent.confidence > 0.5:
                handled_queries += 1
    
    print(f"Total Edge Cases Tested: {total_queries}")
    print(f"Successfully Handled: {handled_queries}")
    print(f"Success Rate: {handled_queries/total_queries*100:.1f}%")
    
    # Special test: Dynamic app discovery
    print("\n🔍 Dynamic App Discovery Test:")
    print("-" * 80)
    
    novel_apps = [
        "can you see Bruno API client in the other space?",
        "where is Warp terminal?",
        "find TablePlus on my desktops",
        "is Insomnia running anywhere?",
        "locate Paw API tool"
    ]
    
    print("Testing with apps not in the original codebase:")
    for query in novel_apps:
        intent = extension.query_detector.detect_intent(query)
        app = intent.target_app
        print(f"Query: {query}")
        print(f"  → Extracted App: {app or 'None'}")
        print(f"  → Success: {'✅' if app else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_edge_cases())