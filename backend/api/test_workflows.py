#!/usr/bin/env python3
"""
Test script for JARVIS Multi-Command Workflow System
Tests various workflow scenarios to ensure proper functionality
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from workflow_parser import WorkflowParser
from workflow_engine import WorkflowExecutionEngine
from workflow_command_processor import WorkflowCommandProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowTester:
    """Test harness for workflow system"""
    
    def __init__(self):
        self.parser = WorkflowParser()
        self.engine = WorkflowExecutionEngine()
        self.processor = WorkflowCommandProcessor()
        self.test_results = []
        
    async def run_test_suite(self):
        """Run complete test suite"""
        logger.info("Starting JARVIS Workflow Test Suite")
        logger.info("=" * 50)
        
        # Test 1: Simple two-action workflow
        await self.test_simple_workflow()
        
        # Test 2: Complex multi-action workflow
        await self.test_complex_workflow()
        
        # Test 3: Workflow with dependencies
        await self.test_dependency_workflow()
        
        # Test 4: Error handling and recovery
        await self.test_error_recovery()
        
        # Test 5: Parallel execution
        await self.test_parallel_execution()
        
        # Test 6: Natural language variations
        await self.test_nlp_variations()
        
        # Test 7: Real-world scenarios
        await self.test_real_world_scenarios()
        
        # Print summary
        self.print_test_summary()
        
    async def test_simple_workflow(self):
        """Test simple two-action workflow"""
        logger.info("\n[TEST 1] Simple Two-Action Workflow")
        logger.info("-" * 40)
        
        test_commands = [
            "Hey JARVIS, open Safari and search for Python tutorials",
            "JARVIS, check my email and calendar",
            "Open Word and create a new document"
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "simple_workflow")
            
    async def test_complex_workflow(self):
        """Test complex multi-action workflow"""
        logger.info("\n[TEST 2] Complex Multi-Action Workflow")
        logger.info("-" * 40)
        
        test_commands = [
            "Hey JARVIS, prepare for my meeting by opening Zoom, checking my calendar, and muting notifications",
            "JARVIS, research machine learning by opening Safari, searching for ML tutorials, and creating a new document for notes",
            "Help me organize my work by checking emails, opening my task list, and setting a reminder for the important ones"
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "complex_workflow")
            
    async def test_dependency_workflow(self):
        """Test workflows with action dependencies"""
        logger.info("\n[TEST 3] Dependency Workflow")
        logger.info("-" * 40)
        
        test_commands = [
            "JARVIS, unlock my screen, then open Mail and check for new messages",
            "Open PowerPoint, create a new presentation, and add a title slide about AI",
            "Find documents about project alpha, open them, and create a summary"
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "dependency_workflow")
            
    async def test_error_recovery(self):
        """Test error handling and recovery"""
        logger.info("\n[TEST 4] Error Recovery")
        logger.info("-" * 40)
        
        test_commands = [
            "Open NonExistentApp and search for something",  # Should handle missing app
            "Search for files in /invalid/path and open them",  # Should handle invalid path
            "Create a new document in ReadOnlyApp"  # Should handle permission errors
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "error_recovery")
            
    async def test_parallel_execution(self):
        """Test parallel execution of independent actions"""
        logger.info("\n[TEST 5] Parallel Execution")
        logger.info("-" * 40)
        
        test_commands = [
            "JARVIS, open Safari, Chrome, and Firefox simultaneously",
            "Check my email, calendar, and weather at the same time",
            "Open Word, Excel, and PowerPoint all at once"
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "parallel_execution")
            
    async def test_nlp_variations(self):
        """Test natural language variations"""
        logger.info("\n[TEST 6] Natural Language Variations")
        logger.info("-" * 40)
        
        test_commands = [
            # Different connectors
            "Open Safari and then search for dogs",
            "Open Safari, after that search for dogs",
            "First open Safari, next search for dogs",
            "Open Safari followed by searching for dogs",
            
            # Different phrasings
            "Could you please open Mail and check my messages?",
            "I need you to launch the Mail app and see if I have new emails",
            "Would you mind opening Mail and looking at my inbox?",
            
            # Comma-separated
            "Open Word, create document, add title",
            "Check email, calendar, weather",
            
            # Complex sentences
            "JARVIS, I need to prepare for my presentation, so please open PowerPoint, search for design templates, and create a new slide deck"
        ]
        
        for cmd in test_commands:
            await self._test_command(cmd, "nlp_variations", quick=True)
            
    async def test_real_world_scenarios(self):
        """Test real-world workflow scenarios"""
        logger.info("\n[TEST 7] Real-World Scenarios")
        logger.info("-" * 40)
        
        scenarios = [
            {
                "name": "Morning Routine",
                "command": "JARVIS, start my day by checking the weather, reviewing my calendar, and opening my email"
            },
            {
                "name": "Research Task",
                "command": "Help me research quantum computing by searching for articles, opening Wikipedia, and creating a notes document"
            },
            {
                "name": "Meeting Preparation",
                "command": "Prepare for my 2pm meeting by opening the meeting link, muting notifications, and pulling up the agenda"
            },
            {
                "name": "Document Creation",
                "command": "Create a report by opening Word, setting up a professional template, and adding a table of contents"
            },
            {
                "name": "Email Management",
                "command": "Process my inbox by checking for urgent emails, flagging important ones, and archiving the read messages"
            }
        ]
        
        for scenario in scenarios:
            logger.info(f"\nScenario: {scenario['name']}")
            await self._test_command(scenario['command'], "real_world")
            
    async def _test_command(self, command: str, test_type: str, quick: bool = False):
        """Test a single command"""
        logger.info(f"\nTesting: '{command}'")
        start_time = datetime.now()
        
        try:
            # Parse the workflow
            workflow = self.parser.parse_command(command)
            logger.info(f"  Parsed: {len(workflow.actions)} actions, complexity: {workflow.complexity}")
            
            # Log actions
            for i, action in enumerate(workflow.actions):
                deps = f" (depends on: {action.dependencies})" if action.dependencies else ""
                logger.info(f"    {i+1}. {action.action_type.value}: {action.target}{deps}")
                
            if quick:
                # Just test parsing for NLP variations
                result = {
                    'test_type': test_type,
                    'command': command,
                    'success': True,
                    'actions': len(workflow.actions),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            else:
                # Execute the workflow (mock execution for testing)
                result = await self._mock_execute_workflow(workflow, test_type)
                result['command'] = command
                result['duration'] = (datetime.now() - start_time).total_seconds()
                
            self.test_results.append(result)
            logger.info(f"  Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"  ERROR: {str(e)}")
            self.test_results.append({
                'test_type': test_type,
                'command': command,
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - start_time).total_seconds()
            })
            
    async def _mock_execute_workflow(self, workflow, test_type: str) -> Dict[str, Any]:
        """Mock workflow execution for testing"""
        # Simulate execution with mock results
        mock_results = []
        
        for i, action in enumerate(workflow.actions):
            # Simulate some failures for error recovery testing
            if test_type == "error_recovery":
                if "NonExistentApp" in action.target:
                    mock_results.append({
                        'action_index': i,
                        'status': 'failed',
                        'error': 'Application not found'
                    })
                elif "invalid" in str(action.parameters.get('path', '')):
                    mock_results.append({
                        'action_index': i,
                        'status': 'failed',
                        'error': 'Invalid path'
                    })
                else:
                    mock_results.append({
                        'action_index': i,
                        'status': 'completed'
                    })
            else:
                # Normal execution
                mock_results.append({
                    'action_index': i,
                    'status': 'completed',
                    'duration': 0.5 + (i * 0.2)  # Simulate varying execution times
                })
                
        # Determine overall success
        failed_count = sum(1 for r in mock_results if r['status'] == 'failed')
        success = failed_count == 0 or (test_type == "error_recovery" and failed_count < len(mock_results))
        
        return {
            'test_type': test_type,
            'success': success,
            'actions': len(workflow.actions),
            'completed': sum(1 for r in mock_results if r['status'] == 'completed'),
            'failed': failed_count,
            'results': mock_results
        }
        
    def print_test_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        
        # Group results by test type
        by_type = {}
        for result in self.test_results:
            test_type = result['test_type']
            if test_type not in by_type:
                by_type[test_type] = {'total': 0, 'passed': 0, 'failed': 0}
                
            by_type[test_type]['total'] += 1
            if result.get('success'):
                by_type[test_type]['passed'] += 1
            else:
                by_type[test_type]['failed'] += 1
                
        # Print summary by type
        for test_type, stats in by_type.items():
            success_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"\n{test_type.upper()}:")
            logger.info(f"  Total: {stats['total']}")
            logger.info(f"  Passed: {stats['passed']} ({success_rate:.1f}%)")
            logger.info(f"  Failed: {stats['failed']}")
            
        # Overall summary
        total_tests = len(self.test_results)
        total_passed = sum(1 for r in self.test_results if r.get('success'))
        overall_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info("\n" + "-" * 50)
        logger.info(f"OVERALL: {total_passed}/{total_tests} passed ({overall_rate:.1f}%)")
        
        # Performance stats
        avg_duration = sum(r.get('duration', 0) for r in self.test_results) / total_tests if total_tests > 0 else 0
        logger.info(f"Average test duration: {avg_duration:.2f}s")
        
        # Failed tests detail
        failed_tests = [r for r in self.test_results if not r.get('success')]
        if failed_tests:
            logger.info("\nFAILED TESTS:")
            for test in failed_tests:
                logger.info(f"  - '{test['command']}': {test.get('error', 'Unknown error')}")


async def main():
    """Main test execution"""
    tester = WorkflowTester()
    await tester.run_test_suite()


if __name__ == "__main__":
    asyncio.run(main())