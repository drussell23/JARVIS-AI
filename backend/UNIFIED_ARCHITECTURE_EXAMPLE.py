#!/usr/bin/env python3
"""
Unified Intelligence Architecture - Concrete Implementation Example
This shows how JARVIS should be restructured to solve the multi-interpreter problems
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime


class IntentType(Enum):
    """Types of user intents"""
    VISUAL_ANALYSIS = "visual_analysis"
    SYSTEM_CONTROL = "system_control" 
    COMMUNICATION = "communication"
    COMPOUND = "compound"
    QUERY = "query"
    MONITORING = "monitoring"
    IMPLICIT = "implicit"
    META_COMMAND = "meta_command"


class CapabilityType(Enum):
    """Available capabilities"""
    VISION = "vision"
    SYSTEM = "system"
    VOICE = "voice"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    MONITORING = "monitoring"


@dataclass
class UnifiedContext:
    """Single source of truth for ALL context"""
    conversation_history: List[Dict[str, Any]]
    visual_context: Dict[str, Any]  # What's on screen NOW
    entity_memory: Dict[str, Any]   # it/this/that resolution
    system_state: Dict[str, Any]    # Running processes, windows
    action_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    temporal_context: Dict[str, Any]  # Time-based references
    
    def resolve_reference(self, reference: str) -> Tuple[Any, float]:
        """Resolves ambiguous references like 'it', 'that', 'this'
        Returns: (resolved_entity, confidence_score)
        """
        if reference.lower() in ['it', 'that', 'this']:
            # Check visual context first
            if 'focused_element' in self.visual_context:
                return self.visual_context['focused_element'], 0.9
            
            # Check recent entities
            if reference in self.entity_memory:
                entity = self.entity_memory[reference]
                age = datetime.now() - entity['timestamp']
                confidence = 0.8 if age.seconds < 60 else 0.5
                return entity['value'], confidence
                
            # Check conversation history
            for entry in reversed(self.conversation_history[-5:]):
                if 'entities' in entry:
                    return entry['entities'][0], 0.6
                    
        return None, 0.0
    
    def update_from_capability(self, capability: str, update: Dict[str, Any]):
        """Any capability can update the unified context"""
        timestamp = datetime.now()
        
        if capability == CapabilityType.VISION.value:
            self.visual_context.update(update)
            # Extract entities for reference resolution
            if 'identified_elements' in update:
                for element in update['identified_elements']:
                    self.entity_memory['that'] = {
                        'value': element,
                        'timestamp': timestamp
                    }
                    
        elif capability == CapabilityType.SYSTEM.value:
            self.system_state.update(update)
            
        # Record in history
        self.action_history.append({
            'capability': capability,
            'update': update,
            'timestamp': timestamp
        })


@dataclass  
class Intent:
    """Resolved user intent"""
    type: IntentType
    primary_action: str
    required_capabilities: List[CapabilityType]
    parameters: Dict[str, Any]
    confidence: float
    ambiguities: List[str]
    implicit_requirements: List[str]
    

class IntentResolutionEngine:
    """Resolves user commands to actionable intents"""
    
    def __init__(self, claude_api):
        self.claude = claude_api
        self.intent_patterns = self._build_intent_patterns()
        
    async def resolve_intent(self, command: str, context: UnifiedContext) -> Intent:
        """Resolves command to intent using context"""
        
        # First try pattern matching for common commands
        quick_match = self._quick_pattern_match(command)
        if quick_match and quick_match.confidence > 0.8:
            return quick_match
            
        # Use Claude for complex understanding
        prompt = f"""
        Analyze this command in context and determine the user's intent.
        
        Command: "{command}"
        
        Context:
        - Visual: {context.visual_context.get('summary', 'No visual context')}
        - Recent conversation: {self._summarize_conversation(context)}
        - System state: {context.system_state.get('summary', 'Normal')}
        
        Determine:
        1. Primary intent type
        2. Required capabilities
        3. Any ambiguities
        4. Implicit requirements
        5. Parameters needed
        
        Consider edge cases like:
        - Compound commands requiring multiple actions
        - Ambiguous references (it, that, this)
        - Temporal dependencies
        - Implicit actions
        """
        
        # Parse Claude's response into Intent
        response = await self.claude.analyze(prompt)
        return self._parse_intent_response(response, command, context)
        
    def _parse_intent_response(self, response: Dict, command: str, context: UnifiedContext) -> Intent:
        """Parse Claude's response into structured Intent"""
        # Example parsing logic
        return Intent(
            type=IntentType.COMPOUND,
            primary_action="screenshot_and_send",
            required_capabilities=[CapabilityType.VISION, CapabilityType.COMMUNICATION],
            parameters={
                'target': 'error_dialog',
                'recipient': 'Mike',
                'format': 'screenshot'
            },
            confidence=0.85,
            ambiguities=[],
            implicit_requirements=['identify_error', 'find_contact']
        )


class CapabilityOrchestra:
    """Orchestrates multiple capabilities to fulfill intents"""
    
    def __init__(self, context: UnifiedContext):
        self.context = context
        self.capabilities = {}
        self._init_capabilities()
        
    def _init_capabilities(self):
        """Initialize all available capabilities"""
        from capabilities import (
            VisionCapability, SystemCapability, 
            VoiceCapability, CommunicationCapability
        )
        
        self.capabilities = {
            CapabilityType.VISION: VisionCapability(self.context),
            CapabilityType.SYSTEM: SystemCapability(self.context),
            CapabilityType.VOICE: VoiceCapability(self.context),
            CapabilityType.COMMUNICATION: CommunicationCapability(self.context)
        }
        
    async def execute_intent(self, intent: Intent) -> Dict[str, Any]:
        """Execute an intent by orchestrating capabilities"""
        
        # Build execution pipeline
        pipeline = self._build_pipeline(intent)
        
        # Execute with transaction support
        results = []
        rollback_stack = []
        
        try:
            for step in pipeline:
                capability_type, method, params = step
                capability = self.capabilities[capability_type]
                
                # Execute step
                result = await getattr(capability, method)(**params)
                results.append(result)
                
                # Track for rollback
                if hasattr(capability, f"rollback_{method}"):
                    rollback_stack.append((capability, method, params))
                    
                # Update context
                self.context.update_from_capability(capability_type.value, result)
                
            return {
                'success': True,
                'results': results,
                'intent': intent
            }
            
        except Exception as e:
            # Rollback on failure
            for capability, method, params in reversed(rollback_stack):
                try:
                    await getattr(capability, f"rollback_{method}")(**params)
                except:
                    pass
                    
            return {
                'success': False,
                'error': str(e),
                'partial_results': results
            }
            
    def _build_pipeline(self, intent: Intent) -> List[Tuple]:
        """Build execution pipeline from intent"""
        
        # Example: Screenshot error and send to Mike
        if intent.primary_action == "screenshot_and_send":
            return [
                (CapabilityType.VISION, 'capture_screenshot', {}),
                (CapabilityType.VISION, 'analyze_content', {'target': 'error'}),
                (CapabilityType.COMMUNICATION, 'find_contact', {'name': intent.parameters['recipient']}),
                (CapabilityType.COMMUNICATION, 'send_message', {'attachment': 'screenshot'})
            ]
            
        # Add more pipeline builders for different intents
        return []


class FeedbackLoopSystem:
    """Learns from successes and failures to improve intent resolution"""
    
    def __init__(self):
        self.outcome_history = []
        self.pattern_corrections = {}
        self.user_preferences = {}
        
    def record_outcome(self, 
                      command: str, 
                      intent: Intent, 
                      result: Dict,
                      user_satisfied: bool = True):
        """Record command outcome for learning"""
        self.outcome_history.append({
            'command': command,
            'intent': intent,
            'result': result,
            'satisfied': user_satisfied,
            'timestamp': datetime.now()
        })
        
        # Learn patterns
        if not user_satisfied:
            self._learn_correction(command, intent)
            
    def _learn_correction(self, command: str, failed_intent: Intent):
        """Learn from failures to improve future routing"""
        # Track what went wrong
        key = f"{command[:20]}_{failed_intent.type.value}"
        if key not in self.pattern_corrections:
            self.pattern_corrections[key] = []
        self.pattern_corrections[key].append(failed_intent)
        
    def suggest_intent_correction(self, command: str, proposed_intent: Intent) -> Optional[Intent]:
        """Suggest corrections based on learned patterns"""
        key = f"{command[:20]}_{proposed_intent.type.value}"
        if key in self.pattern_corrections:
            # This pattern has failed before
            # Suggest alternative based on what worked
            return self._find_successful_alternative(command)
        return None


class UnifiedJARVIS:
    """Main JARVIS class using unified architecture"""
    
    def __init__(self, claude_api_key: str):
        self.context = UnifiedContext(
            conversation_history=[],
            visual_context={},
            entity_memory={},
            system_state={},
            action_history=[],
            user_preferences={},
            temporal_context={}
        )
        
        self.intent_resolver = IntentResolutionEngine(claude_api_key)
        self.orchestra = CapabilityOrchestra(self.context)
        self.feedback_loop = FeedbackLoopSystem()
        
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process user command through unified pipeline"""
        
        # 1. Resolve intent with full context
        intent = await self.intent_resolver.resolve_intent(command, self.context)
        
        # 2. Check feedback loop for corrections
        corrected_intent = self.feedback_loop.suggest_intent_correction(command, intent)
        if corrected_intent:
            intent = corrected_intent
            
        # 3. Handle ambiguities if needed
        if intent.ambiguities:
            clarification = await self._request_clarification(intent.ambiguities)
            intent = await self._refine_intent(intent, clarification)
            
        # 4. Execute through orchestra
        result = await self.orchestra.execute_intent(intent)
        
        # 5. Record outcome
        self.feedback_loop.record_outcome(command, intent, result)
        
        # 6. Update conversation history
        self.context.conversation_history.append({
            'command': command,
            'intent': intent,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result
        
    async def _request_clarification(self, ambiguities: List[str]) -> str:
        """Request clarification for ambiguous commands"""
        # In real implementation, this would interact with user
        pass
        
    async def _refine_intent(self, intent: Intent, clarification: str) -> Intent:
        """Refine intent based on clarification"""
        # Re-resolve with additional context
        pass


# Example usage showing how edge cases are handled

async def demonstrate_unified_architecture():
    """Show how the unified architecture handles problematic commands"""
    
    jarvis = UnifiedJARVIS(api_key="test")
    
    # Edge Case 1: Compound Command
    print("=== Compound Command ===")
    result = await jarvis.process_command("Screenshot this error and send it to Mike")
    # Intent: COMPOUND type
    # Capabilities: [VISION, COMMUNICATION]  
    # Pipeline: capture → analyze → find_contact → send
    # Result: All steps executed in order with shared context
    
    # Edge Case 2: Ambiguous Reference
    print("\n=== Ambiguous Reference ===")
    # First establish context
    await jarvis.process_command("What application is using the most CPU?")
    # Updates context.visual_context with Chrome info
    
    result = await jarvis.process_command("Close it")
    # Context resolves "it" → Chrome (confidence: 0.9)
    # Intent: SYSTEM_CONTROL
    # Action: close_application(target="Chrome")
    
    # Edge Case 3: Implicit Action
    print("\n=== Implicit Action ===")  
    result = await jarvis.process_command("This is wrong")
    # Context checks: visual_context shows error dialog
    # Intent: IMPLICIT type, likely error correction
    # Clarification requested: "Would you like me to debug the error?"
    
    # Edge Case 4: Temporal Dependency
    print("\n=== Temporal Dependency ===")
    await jarvis.process_command("Watch for updates")
    result = await jarvis.process_command("When one appears, install it")
    # Intent: MONITORING with conditional action
    # Creates persistent monitoring task with trigger
    
    # Edge Case 5: Meta Command
    print("\n=== Meta Command ===")
    await jarvis.process_command("Close Chrome") 
    result = await jarvis.process_command("Wait, not that one, close Safari instead")
    # Intent: META_COMMAND type
    # Rolls back Chrome close, executes Safari close


if __name__ == "__main__":
    # This shows the conceptual flow
    asyncio.run(demonstrate_unified_architecture())