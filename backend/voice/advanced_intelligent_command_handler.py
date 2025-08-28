#!/usr/bin/env python3
"""
Advanced Intelligent Command Handler with Zero Hardcoding
Every decision is learned and adaptive - no hardcoded patterns
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

# Import our advanced learning components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "swift_bridge"))

from advanced_python_bridge import AdvancedIntelligentCommandRouter, LearningFeedback
from learning_components import LearningDatabase

# Import existing handlers to maintain compatibility
from system_control.claude_command_interpreter import ClaudeCommandInterpreter
from chatbots.claude_chatbot import ClaudeChatbot
from api.anthropic_client import get_anthropic_client

logger = logging.getLogger(__name__)

class AdvancedIntelligentCommandHandler:
    """
    Advanced command handler that learns and adapts with zero hardcoding
    All routing decisions are based on learned patterns
    """
    
    def __init__(self, user_name: str = "Sir"):
        self.user_name = user_name
        self.router = AdvancedIntelligentCommandRouter()
        
        # Initialize handlers (these remain for backward compatibility)
        self.system_handler = ClaudeCommandInterpreter(user_name=user_name)
        
        # Initialize Claude for conversation handling
        client = get_anthropic_client()
        self.conversation_handler = ClaudeChatbot(
            api_key=client.api_key,
            model="claude-3-opus-20240229"
        )
        
        # Performance tracking
        self.command_history = []
        self.learning_enabled = True
        
        # Initialize with some basic patterns if database is empty
        self._initialize_if_needed()
        
        logger.info("Advanced Intelligent Command Handler initialized with zero hardcoding")
    
    async def handle_command(self, command: str) -> Tuple[str, str]:
        """
        Handle command with intelligent routing based on learned patterns
        
        Returns: (response, handler_used)
        """
        start_time = datetime.now()
        
        try:
            # Route command using advanced ML
            handler_type, classification = await self.router.route_command(command)
            
            # Log classification for debugging
            logger.info(f"Command: '{command}' -> Type: {classification.type}, "
                       f"Intent: {classification.intent}, Confidence: {classification.confidence:.2f}")
            logger.debug(f"Reasoning: {classification.reasoning}")
            
            # Execute based on classification
            response = await self._execute_command(
                command, 
                handler_type, 
                classification
            )
            
            # Track performance
            elapsed = (datetime.now() - start_time).total_seconds()
            self._track_command(command, classification, handler_type, elapsed)
            
            # Learn from successful execution
            if self.learning_enabled:
                self._learn_from_execution(command, classification, True)
            
            return response, handler_type
            
        except Exception as e:
            logger.error(f"Error handling command '{command}': {e}")
            
            # Learn from error
            if self.learning_enabled:
                self._learn_from_error(command, str(e))
            
            # Fallback response
            return f"I encountered an error processing that command: {str(e)}", "error"
    
    async def _execute_command(
        self, 
        command: str, 
        handler_type: str,
        classification: Any
    ) -> str:
        """Execute command based on classification"""
        
        # Route to appropriate handler based on learned classification
        if handler_type == "system":
            return await self._handle_system_command(command, classification)
        elif handler_type == "vision":
            return await self._handle_vision_command(command, classification)
        elif handler_type == "conversation":
            return await self._handle_conversation(command, classification)
        elif handler_type == "automation":
            return await self._handle_automation(command, classification)
        else:
            # Unknown type - use conversation as fallback but learn from it
            logger.warning(f"Unknown handler type: {handler_type}")
            return await self._handle_conversation(command, classification)
    
    async def _handle_system_command(self, command: str, classification: Any) -> str:
        """Handle system commands"""
        
        try:
            # Extract key information from classification
            entities = classification.entities
            intent = classification.intent
            
            # Use the system handler with the command
            response = await self.system_handler.interpret_and_execute(command)
            
            # If we have high confidence and specific intent, we can optimize
            if classification.confidence > 0.8 and intent:
                logger.info(f"High confidence system command: {intent}")
            
            return response
            
        except Exception as e:
            logger.error(f"System command error: {e}")
            return f"I couldn't execute that system command: {str(e)}"
    
    async def _handle_vision_command(self, command: str, classification: Any) -> str:
        """Handle vision-related commands"""
        
        # This would integrate with the vision system
        # For now, return a placeholder that indicates vision analysis
        entities_text = ", ".join([e["text"] for e in classification.entities])
        
        return (f"I understand you want me to analyze something visually. "
                f"Detected elements: {entities_text}. "
                f"Vision analysis would be performed here.")
    
    async def _handle_conversation(self, command: str, classification: Any) -> str:
        """Handle conversational commands"""
        
        try:
            # Add context about the classification
            enhanced_prompt = f"{command}\n\n[Context: Classified as {classification.type} with intent '{classification.intent}']"
            
            response = await self.conversation_handler.generate_response(enhanced_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            return "I'm having trouble processing that request. Could you please rephrase?"
    
    async def _handle_automation(self, command: str, classification: Any) -> str:
        """Handle automation commands"""
        
        # This would integrate with an automation system
        # For now, acknowledge the automation request
        return (f"I understand you want to automate something: {classification.intent}. "
                f"Automation features are being developed.")
    
    def provide_feedback(self, command: str, was_correct: bool, correct_type: Optional[str] = None):
        """
        Provide feedback to improve future classifications
        
        Args:
            command: The original command
            was_correct: Whether the classification was correct
            correct_type: If incorrect, what the correct type should have been
        """
        
        if not self.learning_enabled:
            return
        
        # Find the most recent classification for this command
        recent_classification = self._find_recent_classification(command)
        
        if not recent_classification:
            logger.warning(f"No recent classification found for feedback on: {command}")
            return
        
        # Create feedback
        feedback = LearningFeedback(
            command=command,
            classified_as=recent_classification["type"],
            should_be=correct_type or recent_classification["type"],
            user_rating=1.0 if was_correct else 0.0,
            timestamp=datetime.now(),
            context=recent_classification.get("context", {})
        )
        
        # Send to router for learning
        self.router.provide_feedback(feedback)
        
        logger.info(f"Feedback provided: {command} -> {'Correct' if was_correct else f'Should be {correct_type}'}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = self.router.get_performance_metrics()
        insights = self.router.get_learning_insights()
        
        return {
            "performance": metrics,
            "learning": insights,
            "command_count": len(self.command_history),
            "handlers": {
                "system": self._count_handler_usage("system"),
                "vision": self._count_handler_usage("vision"),
                "conversation": self._count_handler_usage("conversation"),
                "automation": self._count_handler_usage("automation")
            }
        }
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning"""
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
    
    def _initialize_if_needed(self):
        """Initialize with basic patterns if database is empty"""
        
        # Check if we have any patterns
        insights = self.router.get_learning_insights()
        
        if insights["total_patterns_learned"] == 0:
            logger.info("No patterns found - initializing with basic examples")
            
            # These are just examples to bootstrap - the system will learn and adapt
            examples = [
                ("open safari", True, "system"),
                ("close whatsapp", True, "system"),
                ("what's on my screen", True, "vision"),
                ("how are you", True, "conversation"),
                ("remind me in 5 minutes", True, "automation")
            ]
            
            for command, _, correct_type in examples:
                # Simulate feedback to learn these patterns
                feedback = LearningFeedback(
                    command=command,
                    classified_as=correct_type,
                    should_be=correct_type,
                    user_rating=0.7,  # Medium confidence for bootstrap data
                    timestamp=datetime.now(),
                    context={"bootstrap": True}
                )
                self.router.provide_feedback(feedback)
    
    def _track_command(
        self, 
        command: str, 
        classification: Any,
        handler_type: str,
        elapsed_time: float
    ):
        """Track command for history and analysis"""
        
        self.command_history.append({
            "command": command,
            "classification": {
                "type": classification.type,
                "intent": classification.intent,
                "confidence": classification.confidence
            },
            "handler": handler_type,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now()
        })
        
        # Keep history manageable
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-500:]
    
    def _find_recent_classification(self, command: str) -> Optional[Dict[str, Any]]:
        """Find the most recent classification for a command"""
        
        # Search from most recent
        for entry in reversed(self.command_history):
            if entry["command"].lower() == command.lower():
                return entry["classification"]
        
        return None
    
    def _count_handler_usage(self, handler_type: str) -> int:
        """Count how many times a handler was used"""
        
        return sum(1 for entry in self.command_history 
                  if entry["handler"] == handler_type)
    
    def _learn_from_execution(self, command: str, classification: Any, success: bool):
        """Learn from command execution"""
        
        # Reinforce successful classifications
        if success and classification.confidence < 0.9:
            # Provide positive feedback to increase confidence
            feedback = LearningFeedback(
                command=command,
                classified_as=classification.type,
                should_be=classification.type,
                user_rating=min(1.0, classification.confidence + 0.1),
                timestamp=datetime.now(),
                context={"execution": "success"}
            )
            self.router.provide_feedback(feedback)
    
    def _learn_from_error(self, command: str, error: str):
        """Learn from execution errors"""
        
        # Log error patterns for learning
        logger.info(f"Learning from error - Command: {command}, Error: {error}")
        
        # This could be enhanced to detect patterns in errors
        # and adjust classifications accordingly
    
    async def analyze_command_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in command history"""
        
        if not self.command_history:
            return {"message": "No command history available"}
        
        # Analyze patterns
        total_commands = len(self.command_history)
        
        # Confidence distribution
        confidences = [entry["classification"]["confidence"] 
                      for entry in self.command_history]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Response time analysis
        response_times = [entry["elapsed_time"] for entry in self.command_history]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Type distribution
        type_counts = {}
        for entry in self.command_history:
            cmd_type = entry["classification"]["type"]
            type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1
        
        # Intent patterns
        intent_counts = {}
        for entry in self.command_history:
            intent = entry["classification"]["intent"]
            if intent:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_commands": total_commands,
            "average_confidence": avg_confidence,
            "average_response_time": avg_response_time,
            "type_distribution": type_counts,
            "top_intents": dict(sorted(intent_counts.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10]),
            "learning_insights": self.router.get_learning_insights()
        }

# Convenience function for backward compatibility
def create_intelligent_handler(user_name: str = "Sir") -> AdvancedIntelligentCommandHandler:
    """Create an instance of the advanced intelligent command handler"""
    return AdvancedIntelligentCommandHandler(user_name=user_name)

# Example usage and testing
async def test_advanced_handler():
    """Test the advanced handler with various commands"""
    
    handler = create_intelligent_handler()
    
    test_commands = [
        "open WhatsApp",
        "close Safari", 
        "what's on my screen",
        "how's the weather today",
        "remind me to call mom in 10 minutes",
        "show me my calendar",
        "play some music"
    ]
    
    print("\n🧠 Testing Advanced Intelligent Command Handler (Zero Hardcoding)\n")
    
    for command in test_commands:
        print(f"\n📝 Command: '{command}'")
        response, handler_type = await handler.handle_command(command)
        print(f"🤖 Response: {response}")
        print(f"📍 Handler: {handler_type}")
        
        # Get metrics after each command
        metrics = handler.router.get_performance_metrics()
        print(f"📊 Confidence: {metrics.accuracy:.2%}")
    
    # Show learning insights
    print("\n📈 Learning Insights:")
    analysis = await handler.analyze_command_patterns()
    print(f"Total patterns learned: {analysis['learning_insights']['total_patterns_learned']}")
    print(f"Average confidence: {analysis['average_confidence']:.2%}")
    print(f"Adaptation rate: {analysis['learning_insights']['adaptation_rate']:.2f}")
    
    # Test feedback
    print("\n🔄 Testing Feedback System:")
    handler.provide_feedback("open WhatsApp", True)  # Correct classification
    handler.provide_feedback("play some music", False, "system")  # Wrong, should be system
    
    print("\n✅ Advanced handler test complete!")

if __name__ == "__main__":
    asyncio.run(test_advanced_handler())