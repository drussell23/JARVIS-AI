#!/usr/bin/env python3
"""
Test script for JARVIS Intelligent Command Routing
Demonstrates Swift-based classification without hardcoding
"""

import asyncio
import os
import sys
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add swift_bridge to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'swift_bridge'))

try:
    from python_bridge import IntelligentCommandRouter, SWIFT_AVAILABLE
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    SWIFT_AVAILABLE = False
    logger.warning("⚠️  Classifier not available")


class TestColors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


async def test_intelligent_routing():
    """Test the intelligent command routing system"""
    
    if not CLASSIFIER_AVAILABLE:
        print(f"{TestColors.RED}❌ Classifier not available{TestColors.ENDC}")
        return
    
    print(f"{TestColors.BOLD}🧠 JARVIS Intelligent Command Routing Test{TestColors.ENDC}")
    print("=" * 60)
    
    if SWIFT_AVAILABLE:
        print(f"\n{TestColors.CYAN}Using Swift-based classification with ZERO hardcoding{TestColors.ENDC}")
        print(f"{TestColors.CYAN}The classifier uses NLP to understand intent, not keywords!{TestColors.ENDC}\n")
    else:
        print(f"\n{TestColors.YELLOW}Using Python fallback classifier (Swift unavailable){TestColors.ENDC}")
        print(f"{TestColors.YELLOW}Still provides intelligent routing without hardcoding!{TestColors.ENDC}\n")
    
    # Initialize router
    router = IntelligentCommandRouter()
    
    # Test commands showing various patterns
    test_cases: List[Tuple[str, str, str]] = [
        # (command, expected_type, description)
        
        # Clear system commands
        ("close whatsapp", "system", "Direct action command"),
        ("quit discord", "system", "Alternative action verb"),
        ("open safari", "system", "Launch application"),
        ("terminate chrome", "system", "Strong action verb"),
        
        # Clear vision commands
        ("what's on my screen", "vision", "Question about screen"),
        ("show me my notifications", "vision", "Request to see"),
        ("what applications are running", "vision", "Query about state"),
        ("describe my workspace", "vision", "Analysis request"),
        
        # Edge cases that demonstrate intelligence
        ("can you close spotify", "system", "Polite action request"),
        ("please open terminal", "system", "Polite command"),
        ("where is whatsapp", "vision", "Location query"),
        ("what's in discord", "vision", "Content query"),
        
        # Complex cases
        ("i want to close all apps", "system", "Intent statement"),
        ("show me how to quit safari", "vision", "Help request, not action"),
        ("whatsapp", "system", "Single word - learned from context"),
        
        # Natural language variations
        ("could you please close whatsapp for me", "system", "Very polite request"),
        ("what messages do i have in discord", "vision", "Specific content query"),
        ("launch visual studio code", "system", "Multi-word app name"),
        ("analyze what's happening in terminal", "vision", "Analysis request"),
    ]
    
    # Track results
    correct = 0
    total = len(test_cases)
    
    print(f"{TestColors.BOLD}Testing {total} commands...{TestColors.ENDC}\n")
    
    for command, expected_type, description in test_cases:
        # Classify command
        handler_type, details = await router.route_command(command)
        
        # Check if correct
        is_correct = handler_type == expected_type
        if is_correct:
            correct += 1
            symbol = f"{TestColors.GREEN}✓{TestColors.ENDC}"
        else:
            symbol = f"{TestColors.RED}✗{TestColors.ENDC}"
        
        # Display result
        print(f"{symbol} \"{command}\"")
        print(f"   Expected: {TestColors.YELLOW}{expected_type}{TestColors.ENDC}, "
              f"Got: {TestColors.YELLOW}{handler_type}{TestColors.ENDC} "
              f"(confidence: {details['confidence']:.2f})")
        print(f"   Intent: {details['intent']}")
        print(f"   Reasoning: {TestColors.CYAN}{details['reasoning']}{TestColors.ENDC}")
        if details.get('entities'):
            print(f"   Entities: {details['entities']}")
        print(f"   {TestColors.BLUE}→ {description}{TestColors.ENDC}")
        print()
        
        # Small delay for readability
        await asyncio.sleep(0.1)
    
    # Summary
    accuracy = (correct / total) * 100
    print(f"\n{TestColors.BOLD}Results Summary:{TestColors.ENDC}")
    print(f"Correct: {correct}/{total} ({accuracy:.1f}% accuracy)")
    
    if accuracy >= 80:
        print(f"{TestColors.GREEN}✅ Excellent classification accuracy!{TestColors.ENDC}")
    elif accuracy >= 60:
        print(f"{TestColors.YELLOW}⚠️  Good accuracy, but can improve with learning{TestColors.ENDC}")
    else:
        print(f"{TestColors.RED}❌ Low accuracy - classifier needs training{TestColors.ENDC}")
    
    # Demonstrate learning
    print(f"\n{TestColors.BOLD}Testing Learning Capability...{TestColors.ENDC}")
    
    # Test ambiguous command
    ambiguous = "handle whatsapp"
    handler1, details1 = await router.route_command(ambiguous)
    confidence1 = details1['confidence']
    print(f"\nFirst classification of '{ambiguous}':")
    print(f"  Type: {handler1}, Confidence: {confidence1:.2f}")
    
    # Teach it
    await router.provide_feedback(ambiguous, "system", True)
    print(f"\n{TestColors.GREEN}✓ Taught classifier that '{ambiguous}' is a system command{TestColors.ENDC}")
    
    # Test again
    handler2, details2 = await router.route_command(ambiguous)
    confidence2 = details2['confidence']
    print(f"\nSecond classification of '{ambiguous}':")
    print(f"  Type: {handler2}, Confidence: {confidence2:.2f}")
    
    if handler2 == "system" and confidence2 > confidence1:
        print(f"{TestColors.GREEN}✅ Learning successful! Confidence increased by {(confidence2-confidence1):.2f}{TestColors.ENDC}")
    else:
        print(f"{TestColors.YELLOW}⚠️  Learning in progress...{TestColors.ENDC}")


async def interactive_test():
    """Interactive testing mode"""
    if not CLASSIFIER_AVAILABLE:
        return
    
    print(f"\n{TestColors.BOLD}Interactive Testing Mode{TestColors.ENDC}")
    print("Type commands to see classification. Type 'quit' to exit.")
    print("-" * 60)
    
    router = IntelligentCommandRouter()
    
    while True:
        try:
            command = input(f"\n{TestColors.CYAN}Enter command: {TestColors.ENDC}")
            
            if command.lower() in ['quit', 'exit']:
                break
            
            if not command.strip():
                continue
            
            # Classify
            handler_type, details = await router.route_command(command)
            
            # Display results
            print(f"\n{TestColors.BOLD}Classification:{TestColors.ENDC}")
            print(f"  Handler: {TestColors.YELLOW}{handler_type}{TestColors.ENDC}")
            print(f"  Confidence: {details['confidence']:.2f}")
            print(f"  Intent: {details['intent']}")
            print(f"  Reasoning: {TestColors.CYAN}{details['reasoning']}{TestColors.ENDC}")
            
            # Ask for feedback
            feedback = input(f"\nWas this correct? (y/n/skip): ").lower()
            if feedback == 'n':
                correct_type = input("What should it be? (system/vision): ").lower()
                if correct_type in ['system', 'vision']:
                    await router.provide_feedback(command, correct_type, True)
                    print(f"{TestColors.GREEN}✓ Thanks! I'll remember that.{TestColors.ENDC}")
            elif feedback == 'y':
                await router.provide_feedback(command, handler_type, True)
                print(f"{TestColors.GREEN}✓ Great! Reinforcing this classification.{TestColors.ENDC}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{TestColors.RED}Error: {e}{TestColors.ENDC}")
    
    print(f"\n{TestColors.YELLOW}👋 Thanks for testing!{TestColors.ENDC}")


async def main():
    """Main test function"""
    # Run automated tests
    await test_intelligent_routing()
    
    # Offer interactive mode
    print(f"\n{TestColors.BOLD}Would you like to try interactive mode?{TestColors.ENDC}")
    response = input("(y/n): ").lower()
    
    if response == 'y':
        await interactive_test()


if __name__ == "__main__":
    print(f"\n{TestColors.BOLD}🚀 JARVIS Intelligent Routing Test{TestColors.ENDC}")
    
    if CLASSIFIER_AVAILABLE:
        if SWIFT_AVAILABLE:
            print(f"{TestColors.CYAN}Powered by Swift NLP - No Hardcoding!{TestColors.ENDC}\n")
        else:
            print(f"{TestColors.YELLOW}Using Python NLP Fallback - Still No Hardcoding!{TestColors.ENDC}\n")
        asyncio.run(main())
    else:
        print(f"{TestColors.RED}Classifier module not found{TestColors.ENDC}")
        print(f"\nPlease check installation")