#!/usr/bin/env python3
"""
Capability Generator for Vision System v2.0
Analyzes failed requests and automatically generates new capabilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import ast
import inspect
import textwrap
from collections import defaultdict, deque
import asyncio
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class FailedRequest:
    """Represents a failed request that needs capability generation"""
    request_id: str
    timestamp: datetime
    command: str
    intent: str
    confidence: float
    error_type: str
    error_message: str
    context: Dict[str, Any]
    user_id: Optional[str] = None
    attempted_handlers: List[str] = field(default_factory=list)
    similar_failures: List[str] = field(default_factory=list)


@dataclass
class GeneratedCapability:
    """Represents a newly generated capability"""
    capability_id: str
    name: str
    description: str
    intent_patterns: List[str]
    handler_code: str
    dependencies: List[str]
    safety_score: float
    complexity: str  # simple, moderate, complex
    created_at: datetime
    training_examples: List[Dict[str, Any]] = field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityTemplate:
    """Template for generating new capabilities"""
    template_id: str
    template_type: str  # vision_analysis, action_execution, data_processing
    base_code: str
    required_imports: List[str]
    parameter_slots: Dict[str, Any]
    safety_constraints: List[str]
    example_usage: str


class CapabilityAnalyzer:
    """Analyzes failed requests to identify capability gaps"""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.capability_gaps = []
        self.analysis_history = deque(maxlen=1000)
        
    async def analyze_failure(self, failed_request: FailedRequest) -> Dict[str, Any]:
        """Analyze a failed request to identify missing capabilities"""
        analysis = {
            'failure_type': self._categorize_failure(failed_request),
            'missing_capability': self._identify_missing_capability(failed_request),
            'similar_failures': self._find_similar_failures(failed_request),
            'complexity_estimate': self._estimate_complexity(failed_request),
            'safety_concerns': self._identify_safety_concerns(failed_request)
        }
        
        # Update failure patterns
        pattern_key = f"{failed_request.intent}_{failed_request.error_type}"
        self.failure_patterns[pattern_key].append(failed_request)
        
        # Add to history
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'request_id': failed_request.request_id,
            'analysis': analysis
        })
        
        return analysis
    
    def _categorize_failure(self, request: FailedRequest) -> str:
        """Categorize the type of failure"""
        error_lower = request.error_message.lower()
        
        if "not implemented" in error_lower or "no handler" in error_lower:
            return "missing_handler"
        elif "confidence too low" in error_lower:
            return "low_confidence"
        elif "timeout" in error_lower:
            return "performance"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission"
        else:
            return "unknown"
    
    def _identify_missing_capability(self, request: FailedRequest) -> Dict[str, Any]:
        """Identify what capability is missing"""
        # Analyze command structure
        command_tokens = request.command.lower().split()
        
        # Common capability patterns
        if any(word in command_tokens for word in ['analyze', 'detect', 'identify']):
            capability_type = 'vision_analysis'
        elif any(word in command_tokens for word in ['click', 'type', 'move', 'drag']):
            capability_type = 'action_execution'
        elif any(word in command_tokens for word in ['extract', 'parse', 'convert']):
            capability_type = 'data_processing'
        else:
            capability_type = 'general'
        
        return {
            'type': capability_type,
            'action': self._extract_action(command_tokens),
            'target': self._extract_target(command_tokens),
            'modifiers': self._extract_modifiers(command_tokens)
        }
    
    def _extract_action(self, tokens: List[str]) -> str:
        """Extract the main action from command tokens"""
        action_words = ['analyze', 'detect', 'identify', 'find', 'show', 'highlight',
                       'click', 'type', 'move', 'drag', 'select', 'open', 'close',
                       'extract', 'parse', 'convert', 'save', 'copy']
        
        for token in tokens:
            if token in action_words:
                return token
        
        # Use first verb-like word
        return tokens[0] if tokens else 'unknown'
    
    def _extract_target(self, tokens: List[str]) -> str:
        """Extract the target object from command tokens"""
        # Simple heuristic: noun after action word
        for i, token in enumerate(tokens[:-1]):
            if token in ['the', 'a', 'an', 'this', 'that']:
                if i + 1 < len(tokens):
                    return tokens[i + 1]
        
        return 'element'
    
    def _extract_modifiers(self, tokens: List[str]) -> List[str]:
        """Extract modifiers from command tokens"""
        modifiers = []
        modifier_words = ['red', 'blue', 'green', 'large', 'small', 'first', 'last',
                         'all', 'selected', 'highlighted', 'active']
        
        for token in tokens:
            if token in modifier_words:
                modifiers.append(token)
                
        return modifiers
    
    def _find_similar_failures(self, request: FailedRequest) -> List[FailedRequest]:
        """Find similar failed requests"""
        similar = []
        
        for pattern_key, failures in self.failure_patterns.items():
            if request.intent in pattern_key or request.error_type in pattern_key:
                similar.extend([f for f in failures if f.request_id != request.request_id])
                
        return similar[:5]  # Return top 5 similar failures
    
    def _estimate_complexity(self, request: FailedRequest) -> str:
        """Estimate complexity of the missing capability"""
        # Based on command structure and context
        command_len = len(request.command.split())
        has_modifiers = len(self._extract_modifiers(request.command.lower().split())) > 0
        
        if command_len < 5 and not has_modifiers:
            return "simple"
        elif command_len < 10:
            return "moderate"
        else:
            return "complex"
    
    def _identify_safety_concerns(self, request: FailedRequest) -> List[str]:
        """Identify potential safety concerns"""
        concerns = []
        command_lower = request.command.lower()
        
        # Check for potentially dangerous operations
        if any(word in command_lower for word in ['delete', 'remove', 'destroy']):
            concerns.append("destructive_operation")
        
        if any(word in command_lower for word in ['password', 'secret', 'key', 'token']):
            concerns.append("sensitive_data")
            
        if any(word in command_lower for word in ['system', 'admin', 'root']):
            concerns.append("elevated_privileges")
            
        if any(word in command_lower for word in ['all', 'everything', 'entire']):
            concerns.append("bulk_operation")
            
        return concerns


class CodeGenerator:
    """Generates code for new capabilities"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.generation_history = []
        
    def _load_templates(self) -> Dict[str, CapabilityTemplate]:
        """Load capability templates"""
        templates = {}
        
        # Vision Analysis Template
        templates['vision_analysis'] = CapabilityTemplate(
            template_id='vision_analysis_v1',
            template_type='vision_analysis',
            base_code='''
async def {handler_name}(self, command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    {description}
    
    Args:
        command: The user command
        context: Optional context information
        
    Returns:
        Dict containing the analysis results
    """
    try:
        # Capture screen
        screen_data = await self._capture_screen(context)
        
        # Perform analysis
        {analysis_code}
        
        # Return results
        return {{
            'success': True,
            'result': result,
            'confidence': confidence,
            'metadata': {{
                'handler': '{handler_name}',
                'timestamp': datetime.now().isoformat()
            }}
        }}
        
    except Exception as e:
        logger.error(f"Error in {handler_name}: {{e}}")
        return {{
            'success': False,
            'error': str(e)
        }}
''',
            required_imports=['from datetime import datetime', 'import logging'],
            parameter_slots={
                'handler_name': 'str',
                'description': 'str',
                'analysis_code': 'str'
            },
            safety_constraints=['read_only', 'no_external_calls'],
            example_usage='await handler("analyze red buttons on screen")'
        )
        
        # Action Execution Template
        templates['action_execution'] = CapabilityTemplate(
            template_id='action_execution_v1',
            template_type='action_execution',
            base_code='''
async def {handler_name}(self, command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    {description}
    
    Args:
        command: The user command
        context: Optional context information
        
    Returns:
        Dict containing the execution results
    """
    try:
        # Safety check
        if not self._is_safe_action(command, context):
            return {{
                'success': False,
                'error': 'Action blocked by safety constraints'
            }}
        
        # Parse command
        {parse_code}
        
        # Execute action
        {action_code}
        
        # Verify result
        {verification_code}
        
        return {{
            'success': True,
            'result': result,
            'confidence': confidence,
            'metadata': {{
                'handler': '{handler_name}',
                'timestamp': datetime.now().isoformat(),
                'action_type': action_type
            }}
        }}
        
    except Exception as e:
        logger.error(f"Error in {handler_name}: {{e}}")
        return {{
            'success': False,
            'error': str(e)
        }}
''',
            required_imports=['from datetime import datetime', 'import logging'],
            parameter_slots={
                'handler_name': 'str',
                'description': 'str',
                'parse_code': 'str',
                'action_code': 'str',
                'verification_code': 'str'
            },
            safety_constraints=['user_confirmation', 'action_logging', 'rollback_capable'],
            example_usage='await handler("click the submit button")'
        )
        
        return templates
    
    async def generate_capability(
        self,
        analysis: Dict[str, Any],
        failed_request: FailedRequest
    ) -> GeneratedCapability:
        """Generate a new capability based on failure analysis"""
        capability_type = analysis['missing_capability']['type']
        
        # Select appropriate template
        template = self.templates.get(capability_type, self.templates['vision_analysis'])
        
        # Generate handler name
        action = analysis['missing_capability']['action']
        target = analysis['missing_capability']['target']
        handler_name = f"handle_{action}_{target}".replace(' ', '_').lower()
        
        # Generate description
        description = f"Handles '{action}' operations on '{target}' elements"
        
        # Generate code based on template
        if capability_type == 'vision_analysis':
            code = self._generate_vision_analysis_code(analysis, failed_request)
        elif capability_type == 'action_execution':
            code = self._generate_action_execution_code(analysis, failed_request)
        else:
            code = self._generate_generic_code(analysis, failed_request)
            
        # Fill template
        handler_code = template.base_code.format(
            handler_name=handler_name,
            description=description,
            **code
        )
        
        # Generate intent patterns
        intent_patterns = self._generate_intent_patterns(failed_request, analysis)
        
        # Create capability
        capability = GeneratedCapability(
            capability_id=hashlib.sha256(handler_name.encode()).hexdigest()[:16],
            name=handler_name,
            description=description,
            intent_patterns=intent_patterns,
            handler_code=handler_code,
            dependencies=template.required_imports,
            safety_score=self._calculate_safety_score(analysis),
            complexity=analysis['complexity_estimate'],
            created_at=datetime.now(),
            training_examples=[{
                'command': failed_request.command,
                'expected_result': 'success',
                'context': failed_request.context
            }]
        )
        
        # Add to history
        self.generation_history.append({
            'timestamp': datetime.now(),
            'capability_id': capability.capability_id,
            'failed_request_id': failed_request.request_id
        })
        
        return capability
    
    def _generate_vision_analysis_code(
        self,
        analysis: Dict[str, Any],
        request: FailedRequest
    ) -> Dict[str, str]:
        """Generate code for vision analysis capability"""
        action = analysis['missing_capability']['action']
        target = analysis['missing_capability']['target']
        modifiers = analysis['missing_capability']['modifiers']
        
        # Generate analysis code
        analysis_code = f'''
        # Extract target elements
        elements = self._extract_elements(screen_data, target='{target}')
        
        # Apply filters
        filtered_elements = elements'''
        
        for modifier in modifiers:
            analysis_code += f'''
        filtered_elements = [e for e in filtered_elements if self._matches_modifier(e, '{modifier}')]'''
        
        analysis_code += f'''
        
        # Perform {action} analysis
        result = []
        confidence = 0.0
        
        for element in filtered_elements:
            analysis_result = self._analyze_element(element, action='{action}')
            result.append(analysis_result)
            confidence = max(confidence, analysis_result.get('confidence', 0))
        '''
        
        return {
            'analysis_code': analysis_code
        }
    
    def _generate_action_execution_code(
        self,
        analysis: Dict[str, Any],
        request: FailedRequest
    ) -> Dict[str, str]:
        """Generate code for action execution capability"""
        action = analysis['missing_capability']['action']
        target = analysis['missing_capability']['target']
        
        parse_code = f'''
        # Parse action parameters
        action_type = '{action}'
        target_type = '{target}'
        parameters = self._extract_parameters(command)'''
        
        action_code = f'''
        # Find target element
        target_element = self._find_element(target_type, parameters)
        
        if not target_element:
            raise ValueError(f"Could not find {{target_type}} element")
        
        # Execute action
        result = await self._execute_action(action_type, target_element, parameters)
        confidence = 0.85'''
        
        verification_code = '''
        # Verify action completed successfully
        if not await self._verify_action_result(action_type, target_element, result):
            raise RuntimeError("Action verification failed")'''
        
        return {
            'parse_code': parse_code,
            'action_code': action_code,
            'verification_code': verification_code
        }
    
    def _generate_generic_code(
        self,
        analysis: Dict[str, Any],
        request: FailedRequest
    ) -> Dict[str, str]:
        """Generate generic capability code"""
        return {
            'analysis_code': '''
        # Generic analysis
        result = {"status": "analyzed", "details": "Generic capability"}
        confidence = 0.5'''
        }
    
    def _generate_intent_patterns(
        self,
        request: FailedRequest,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate intent patterns for the capability"""
        patterns = [request.command]
        
        # Generate variations
        action = analysis['missing_capability']['action']
        target = analysis['missing_capability']['target']
        
        variations = [
            f"{action} {target}",
            f"{action} the {target}",
            f"please {action} {target}",
            f"can you {action} the {target}",
            f"I need to {action} {target}"
        ]
        
        patterns.extend(variations)
        
        return patterns[:10]  # Limit to 10 patterns
    
    def _calculate_safety_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate safety score for generated capability"""
        score = 1.0
        
        # Reduce score based on safety concerns
        concerns = analysis['safety_concerns']
        
        if 'destructive_operation' in concerns:
            score -= 0.4
        if 'sensitive_data' in concerns:
            score -= 0.3
        if 'elevated_privileges' in concerns:
            score -= 0.3
        if 'bulk_operation' in concerns:
            score -= 0.2
            
        return max(0.0, score)


class CapabilityGenerator:
    """
    Main capability generator that coordinates analysis and code generation
    """
    
    def __init__(self):
        self.analyzer = CapabilityAnalyzer()
        self.code_generator = CodeGenerator()
        self.generated_capabilities: Dict[str, GeneratedCapability] = {}
        self.failure_threshold = 3  # Number of similar failures before generation
        
        # Storage
        self.storage_path = Path("backend/data/generated_capabilities")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        logger.info("Capability Generator initialized")
    
    async def analyze_failed_request(self, failed_request: FailedRequest) -> Optional[GeneratedCapability]:
        """Analyze a failed request and potentially generate new capability"""
        # Analyze the failure
        analysis = await self.analyzer.analyze_failure(failed_request)
        
        # Check if we should generate a capability
        if self._should_generate_capability(failed_request, analysis):
            # Generate new capability
            capability = await self.code_generator.generate_capability(analysis, failed_request)
            
            # Store capability
            self.generated_capabilities[capability.capability_id] = capability
            await self._save_capability(capability)
            
            logger.info(f"Generated new capability: {capability.name}")
            
            return capability
        
        return None
    
    def _should_generate_capability(
        self,
        request: FailedRequest,
        analysis: Dict[str, Any]
    ) -> bool:
        """Determine if we should generate a new capability"""
        # Check failure type
        if analysis['failure_type'] != 'missing_handler':
            return False
        
        # Check safety score threshold
        safety_concerns = len(analysis['safety_concerns'])
        if safety_concerns > 2:  # Too many safety concerns
            logger.warning(f"Too many safety concerns ({safety_concerns}) for capability generation")
            return False
        
        # Check if we have enough similar failures
        similar_failures = analysis['similar_failures']
        if len(similar_failures) < self.failure_threshold - 1:
            return False
        
        # Check complexity
        if analysis['complexity_estimate'] == 'complex':
            logger.info("Capability too complex for automatic generation")
            return False
        
        return True
    
    async def _save_capability(self, capability: GeneratedCapability):
        """Save generated capability to disk"""
        # Save capability metadata
        metadata_file = self.storage_path / f"{capability.capability_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'capability_id': capability.capability_id,
                'name': capability.name,
                'description': capability.description,
                'intent_patterns': capability.intent_patterns,
                'dependencies': capability.dependencies,
                'safety_score': capability.safety_score,
                'complexity': capability.complexity,
                'created_at': capability.created_at.isoformat(),
                'training_examples': capability.training_examples
            }, f, indent=2)
        
        # Save handler code
        code_file = self.storage_path / f"{capability.capability_id}_handler.py"
        with open(code_file, 'w') as f:
            f.write(capability.handler_code)
    
    def get_pending_capabilities(self) -> List[GeneratedCapability]:
        """Get capabilities pending validation"""
        return [
            cap for cap in self.generated_capabilities.values()
            if cap.test_results is None
        ]
    
    def get_validated_capabilities(self) -> List[GeneratedCapability]:
        """Get capabilities that passed validation"""
        return [
            cap for cap in self.generated_capabilities.values()
            if cap.test_results and cap.test_results.get('passed', False)
        ]
    
    async def combine_capabilities(
        self,
        capability_ids: List[str],
        combination_type: str = "sequential"
    ) -> Optional[GeneratedCapability]:
        """Combine multiple capabilities for complex tasks"""
        if len(capability_ids) < 2:
            return None
        
        capabilities = [
            self.generated_capabilities.get(cap_id)
            for cap_id in capability_ids
        ]
        
        if any(cap is None for cap in capabilities):
            return None
        
        # Generate combined capability
        if combination_type == "sequential":
            return await self._generate_sequential_capability(capabilities)
        elif combination_type == "parallel":
            return await self._generate_parallel_capability(capabilities)
        elif combination_type == "conditional":
            return await self._generate_conditional_capability(capabilities)
        
        return None
    
    async def _generate_sequential_capability(
        self,
        capabilities: List[GeneratedCapability]
    ) -> GeneratedCapability:
        """Generate capability that executes others in sequence"""
        # Extract handler names
        handler_names = [cap.name for cap in capabilities]
        
        # Generate combined handler code
        combined_code = f'''
async def handle_combined_{"_and_".join(handler_names)}(self, command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Combined capability that executes: {", ".join(handler_names)}
    """
    results = []
    overall_success = True
    
    try:
        # Execute capabilities in sequence
'''
        
        for i, cap in enumerate(capabilities):
            combined_code += f'''
        # Step {i+1}: {cap.description}
        result_{i} = await self.{cap.name}(command, context)
        results.append(result_{i})
        
        if not result_{i}.get('success', False):
            overall_success = False
            logger.warning(f"Step {i+1} failed: {cap.name}")
            # Continue with remaining steps or abort based on criticality
'''
        
        combined_code += '''
        return {
            'success': overall_success,
            'results': results,
            'metadata': {
                'combined_capability': True,
                'steps': len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in combined capability: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_results': results
        }
'''
        
        # Create combined capability
        return GeneratedCapability(
            capability_id=hashlib.sha256(f"combined_{'_'.join(handler_names)}".encode()).hexdigest()[:16],
            name=f"handle_combined_{'_and_'.join(handler_names)}",
            description=f"Combined capability: {' then '.join([cap.description for cap in capabilities])}",
            intent_patterns=[],  # Will be generated based on component patterns
            handler_code=combined_code,
            dependencies=list(set(sum([cap.dependencies for cap in capabilities], []))),
            safety_score=min([cap.safety_score for cap in capabilities]),
            complexity="complex",
            created_at=datetime.now(),
            metadata={
                'combination_type': 'sequential',
                'component_capabilities': [cap.capability_id for cap in capabilities]
            }
        )
    
    async def _generate_parallel_capability(
        self,
        capabilities: List[GeneratedCapability]
    ) -> GeneratedCapability:
        """Generate capability that executes others in parallel"""
        # Implementation for parallel execution
        pass
    
    async def _generate_conditional_capability(
        self,
        capabilities: List[GeneratedCapability]
    ) -> GeneratedCapability:
        """Generate capability with conditional logic"""
        # Implementation for conditional execution
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get capability generation statistics"""
        return {
            'total_generated': len(self.generated_capabilities),
            'pending_validation': len(self.get_pending_capabilities()),
            'validated': len(self.get_validated_capabilities()),
            'failure_patterns': len(self.analyzer.failure_patterns),
            'capability_types': {
                'vision_analysis': sum(1 for cap in self.generated_capabilities.values() 
                                     if 'analyze' in cap.name),
                'action_execution': sum(1 for cap in self.generated_capabilities.values() 
                                      if any(word in cap.name for word in ['click', 'type', 'move'])),
                'data_processing': sum(1 for cap in self.generated_capabilities.values() 
                                     if any(word in cap.name for word in ['extract', 'parse']))
            }
        }


# Singleton instance
_capability_generator: Optional[CapabilityGenerator] = None


def get_capability_generator() -> CapabilityGenerator:
    """Get singleton instance of capability generator"""
    global _capability_generator
    if _capability_generator is None:
        _capability_generator = CapabilityGenerator()
    return _capability_generator