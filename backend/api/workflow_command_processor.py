"""
JARVIS Workflow Command Processor - Integration Layer
Processes multi-command workflows through JARVIS voice system
"""

import asyncio
import re
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .workflow_parser import WorkflowParser
from .workflow_engine import WorkflowExecutionEngine
from .jarvis_voice_api import JARVISCommand

logger = logging.getLogger(__name__)


class WorkflowCommandProcessor:
    """Processes workflow commands and integrates with JARVIS voice system"""
    
    # Patterns that indicate multi-command workflows
    WORKFLOW_INDICATORS = [
        r'\band\b',
        r'\bthen\b',
        r'\bafter that\b',
        r'\bfollowed by\b',
        r'\bnext\b',
        r'\balso\b',
        r'\bplus\b',
        r'[,;]',  # Comma or semicolon separated commands
        r'\bstep \d+',  # Numbered steps
    ]
    
    def __init__(self):
        """Initialize workflow processor"""
        self.parser = WorkflowParser()
        self.engine = WorkflowExecutionEngine()
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.WORKFLOW_INDICATORS]
        
    def is_workflow_command(self, command_text: str) -> bool:
        """Check if command contains multiple actions"""
        # Quick checks
        if len(command_text.split()) < 5:  # Too short for multi-command
            return False
            
        # Check for workflow indicators
        for pattern in self._compiled_patterns:
            if pattern.search(command_text):
                return True
                
        # Check for multiple action verbs
        action_verbs = ['open', 'close', 'search', 'check', 'create', 'send', 
                       'unlock', 'launch', 'start', 'prepare', 'mute', 'set']
        verb_count = sum(1 for verb in action_verbs if verb in command_text.lower())
        
        return verb_count >= 2
        
    async def process_workflow_command(self, command: JARVISCommand, 
                                     user_id: str = "default",
                                     websocket: Optional[Any] = None) -> Dict[str, Any]:
        """Process a multi-command workflow"""
        try:
            logger.info(f"Processing workflow command: '{command.text}'")
            
            # Parse the command into workflow
            workflow = self.parser.parse_command(command.text)
            
            if not workflow.actions:
                return {
                    'success': False,
                    'response': "I couldn't understand the workflow steps. Please try rephrasing.",
                    'command_type': 'workflow_parse_error'
                }
                
            # Log workflow details
            logger.info(f"Parsed workflow with {len(workflow.actions)} actions:")
            for i, action in enumerate(workflow.actions):
                logger.info(f"  {i+1}. {action.action_type.value}: {action.target}")
                
            # Send initial response
            if websocket:
                await websocket.send_json({
                    'type': 'workflow_analysis',
                    'message': f"I'll help you with that. I've identified {len(workflow.actions)} tasks to complete.",
                    'workflow': {
                        'total_actions': len(workflow.actions),
                        'complexity': workflow.complexity,
                        'estimated_duration': workflow.estimated_duration,
                        'actions': [
                            {
                                'type': action.action_type.value,
                                'description': action.description or f"{action.action_type.value} {action.target}"
                            }
                            for action in workflow.actions
                        ]
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            # Execute the workflow
            result = await self.engine.execute_workflow(workflow, user_id, websocket)
            
            # Generate response based on results
            response = self._generate_response(workflow, result)
            
            return {
                'success': result.success_rate > 0.5,
                'response': response,
                'command_type': 'workflow',
                'workflow_result': {
                    'workflow_id': result.workflow_id,
                    'status': result.status.value,
                    'success_rate': result.success_rate,
                    'total_duration': result.total_duration,
                    'actions_completed': sum(1 for r in result.action_results 
                                           if r.status.value == 'completed'),
                    'actions_failed': sum(1 for r in result.action_results 
                                        if r.status.value == 'failed')
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow processing error: {e}", exc_info=True)
            return {
                'success': False,
                'response': "I encountered an error while processing your workflow. Let me try a different approach.",
                'command_type': 'workflow_error',
                'error': str(e)
            }
            
    def _generate_response(self, workflow, result) -> str:
        """Generate natural language response for workflow result"""
        completed = sum(1 for r in result.action_results if r.status.value == 'completed')
        failed = sum(1 for r in result.action_results if r.status.value == 'failed')
        total = len(result.action_results)
        
        if completed == total:
            # All succeeded
            responses = [
                f"Sir, I've successfully completed all {total} tasks.",
                f"All done! I've finished the {total} tasks you requested.",
                f"Mission accomplished. All {total} actions have been completed successfully.",
                f"I've executed your workflow perfectly. All {total} steps are complete."
            ]
            response = responses[hash(workflow.original_command) % len(responses)]
            
            # Add summary of what was done
            if total <= 3:
                actions = [a.description or f"{a.action_type.value} {a.target}" 
                          for a in workflow.actions]
                response += f" I've {', '.join(actions)}."
                
        elif completed > 0:
            # Partial success
            response = f"I've completed {completed} out of {total} tasks."
            if failed > 0:
                response += f" {failed} task(s) encountered issues."
                
                # Mention which failed
                failed_actions = [
                    workflow.actions[r.action_index].description or 
                    f"{workflow.actions[r.action_index].action_type.value} {workflow.actions[r.action_index].target}"
                    for r in result.action_results if r.status.value == 'failed'
                ]
                if len(failed_actions) <= 2:
                    response += f" I couldn't {' or '.join(failed_actions)}."
                    
        else:
            # All failed
            response = "I'm sorry, but I couldn't complete the workflow. "
            if result.action_results:
                first_error = result.action_results[0].error
                if first_error:
                    response += f"The issue was: {first_error}"
                    
        # Add timing info for longer workflows
        if result.total_duration > 10:
            response += f" The entire process took {result.total_duration:.1f} seconds."
            
        return response
        
    async def get_workflow_examples(self) -> List[Dict[str, Any]]:
        """Get example workflow commands for user guidance"""
        return [
            {
                "category": "Productivity",
                "examples": [
                    {
                        "command": "Hey JARVIS, open Safari and search for Python tutorials",
                        "description": "Opens browser and performs search"
                    },
                    {
                        "command": "Hey JARVIS, check my email and calendar for today",
                        "description": "Reviews email and calendar"
                    },
                    {
                        "command": "Hey JARVIS, prepare for my meeting by opening Zoom and muting notifications",
                        "description": "Meeting preparation workflow"
                    }
                ]
            },
            {
                "category": "Document Creation",
                "examples": [
                    {
                        "command": "Hey JARVIS, open Word and create a new document",
                        "description": "Starts document creation"
                    },
                    {
                        "command": "Hey JARVIS, create a new presentation and add a title slide",
                        "description": "PowerPoint workflow"
                    }
                ]
            },
            {
                "category": "Research",
                "examples": [
                    {
                        "command": "Hey JARVIS, search for machine learning on the web and open the top results",
                        "description": "Research workflow"
                    },
                    {
                        "command": "Hey JARVIS, find documents about project alpha and open them",
                        "description": "File search workflow"
                    }
                ]
            }
        ]
        
    def extract_workflow_intents(self, workflow) -> List[str]:
        """Extract high-level intents from workflow for analytics"""
        intents = []
        
        # Check for common workflow patterns
        action_types = [a.action_type.value for a in workflow.actions]
        
        if 'open_app' in action_types and 'mute' in action_types:
            intents.append('focus_mode')
            
        if 'check' in action_types and any('email' in a.target.lower() or 'calendar' in a.target.lower() 
                                          for a in workflow.actions):
            intents.append('daily_review')
            
        if 'open_app' in action_types and 'create' in action_types:
            intents.append('content_creation')
            
        if 'search' in action_types:
            intents.append('research')
            
        if 'unlock' in action_types:
            intents.append('system_access')
            
        return intents or ['general_workflow']


# Global instance for easy access
workflow_processor = WorkflowCommandProcessor()


async def handle_workflow_command(command: JARVISCommand, user_id: str = "default", 
                                websocket: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Helper function to check and process workflow commands"""
    if workflow_processor.is_workflow_command(command.text):
        return await workflow_processor.process_workflow_command(command, user_id, websocket)
    return None