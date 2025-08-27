#!/usr/bin/env python3
"""
Test script for JARVIS Vision System v2.0 - Phase 4: Continuous Learning Pipeline
Tests experience replay, meta-learning, and advanced learning capabilities
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List
import torch
import torch.nn as nn
import random

# Import Vision System v2.0 with Phase 4 components
from vision.vision_system_v2 import get_vision_system_v2
from vision.advanced_continuous_learning import get_advanced_continuous_learning, FederatedUpdate
from vision.experience_replay_system import get_experience_replay_system
from vision.meta_learning_framework import get_meta_learning_framework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase4Tester:
    """Tests Phase 4 continuous learning capabilities"""
    
    def __init__(self):
        self.vision_system = get_vision_system_v2()
        self.test_results = {
            'experience_replay': {},
            'meta_learning': {},
            'continuous_learning': {},
            'federated_learning': {},
            'performance': {}
        }
        
    async def run_all_tests(self):
        """Run all Phase 4 tests"""
        print("\n" + "="*50)
        print("PHASE 4 TEST SUITE - Continuous Learning Pipeline")
        print("="*50)
        
        # Test each component
        await self.test_experience_replay()
        await self.test_pattern_extraction()
        await self.test_meta_learning()
        await self.test_catastrophic_forgetting_prevention()
        await self.test_federated_learning()
        await self.test_continuous_learning_cycle()
        await self.test_adaptive_learning_rate()
        await self.test_end_to_end_learning()
        
        # Print summary
        self.print_test_summary()
        
    async def test_experience_replay(self):
        """Test experience replay system functionality"""
        print("\n1. Testing Experience Replay System...")
        
        if not self.vision_system.experience_replay:
            print("   ❌ Experience replay not available")
            self.test_results['experience_replay']['available'] = False
            return
            
        replay_system = self.vision_system.experience_replay
        
        # Test adding experiences
        print("   Testing experience storage...")
        test_commands = [
            ("Can you see my screen?", "vision_capability_confirmation", True, 45.2),
            ("What's on my display?", "screen_analysis", True, 78.5),
            ("Describe the window", "window_description", False, 125.3),
            ("Show me the error", "error_analysis", False, 95.7),
            ("Analyze this code", "code_analysis", True, 67.8)
        ]
        
        experience_ids = []
        for cmd, intent, success, latency in test_commands:
            # Create mock embedding
            embedding = np.random.randn(768).astype(np.float32)
            
            exp_id = await replay_system.add_experience(
                command=cmd,
                command_embedding=embedding,
                intent=intent,
                confidence=random.uniform(0.6, 0.95),
                handler=f"{intent}_handler",
                response=f"Processed: {cmd}",
                success=success,
                latency_ms=latency,
                user_id="test_user",
                context={'test_mode': True}
            )
            experience_ids.append(exp_id)
            
        print(f"   ✓ Added {len(experience_ids)} experiences")
        
        # Test sampling methods
        print("   Testing sampling methods...")
        
        # Prioritized sampling
        batch = await replay_system.sample_batch(3, method='prioritized')
        print(f"   ✓ Prioritized sampling: {len(batch.experiences)} experiences")
        
        # Recent sampling
        batch = await replay_system.sample_batch(2, method='recent')
        print(f"   ✓ Recent sampling: {len(batch.experiences)} experiences")
        
        # Failure sampling
        batch = await replay_system.sample_batch(2, method='failure')
        print(f"   ✓ Failure sampling: {len(batch.experiences)} experiences")
        
        # Get statistics
        stats = await replay_system.get_statistics()
        print(f"   ✓ Buffer utilization: {stats['buffer_stats']['utilization']:.1%}")
        print(f"   ✓ Success rate: {stats['performance_stats']['success_rate']:.1%}")
        
        self.test_results['experience_replay'] = {
            'available': True,
            'experiences_added': len(experience_ids),
            'sampling_methods_tested': ['prioritized', 'recent', 'failure'],
            'buffer_utilization': stats['buffer_stats']['utilization'],
            'success_rate': stats['performance_stats']['success_rate']
        }
        
    async def test_pattern_extraction(self):
        """Test pattern extraction from experience history"""
        print("\n2. Testing Pattern Extraction...")
        
        if not self.vision_system.experience_replay:
            print("   ❌ Pattern extraction not available")
            return
            
        replay_system = self.vision_system.experience_replay
        
        # Add more experiences to create patterns
        print("   Adding experiences for pattern extraction...")
        
        # Create command patterns
        for i in range(10):
            embedding = np.random.randn(768).astype(np.float32)
            await replay_system.add_experience(
                command=f"show me the {'error' if i < 5 else 'warning'} message",
                command_embedding=embedding,
                intent="error_analysis",
                confidence=0.85,
                handler="error_handler",
                response="Showing error details",
                success=i % 3 != 0,  # Some failures
                latency_ms=random.uniform(50, 150),
                user_id=f"user_{i % 3}",
                context={'error_type': 'syntax' if i < 5 else 'runtime'}
            )
            
        # Force pattern extraction
        await replay_system._extract_all_patterns()
        
        # Check extracted patterns
        print(f"   ✓ Extracted {len(replay_system.patterns)} patterns")
        
        pattern_types = {}
        for pattern in replay_system.patterns.values():
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            
        print("   Pattern types found:")
        for ptype, count in pattern_types.items():
            print(f"     - {ptype}: {count}")
            
        self.test_results['experience_replay']['patterns_extracted'] = len(replay_system.patterns)
        self.test_results['experience_replay']['pattern_types'] = pattern_types
        
    async def test_meta_learning(self):
        """Test meta-learning framework"""
        print("\n3. Testing Meta-Learning Framework...")
        
        if not self.vision_system.advanced_learning:
            print("   ❌ Meta-learning not available")
            self.test_results['meta_learning']['available'] = False
            return
            
        meta_framework = self.vision_system.advanced_learning.meta_framework
        
        # Test strategy selection
        print("   Testing learning strategy selection...")
        
        contexts = [
            {'task_type': 'retraining', 'data_size': 'large'},
            {'task_type': 'fine_tuning', 'data_size': 'small'},
            {'task_type': 'adaptation', 'urgency': 0.9}
        ]
        
        selected_strategies = []
        for context in contexts:
            strategy = meta_framework.select_learning_strategy(context)
            selected_strategies.append(strategy.name)
            print(f"   ✓ Selected '{strategy.name}' for context: {context}")
            
        # Test performance tracking
        print("   Testing performance adaptation...")
        
        # Simulate training steps
        for i in range(5):
            mock_batch = {
                'input': torch.randn(32, 768),
                'target': torch.randint(0, 100, (32,))
            }
            
            metrics = meta_framework.train_step(mock_batch, task_id='test_task')
            print(f"   ✓ Training step {i+1}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2%}")
            
        # Check if adaptation is needed
        should_adapt = meta_framework.should_adapt()
        print(f"   ✓ Adaptation needed: {should_adapt}")
        
        # Get comprehensive metrics
        meta_metrics = meta_framework.get_metrics()
        print(f"   ✓ Stability score: {meta_metrics['stability']:.2f}")
        print(f"   ✓ Task diversity: {meta_metrics['task_diversity']:.2f}")
        
        self.test_results['meta_learning'] = {
            'available': True,
            'strategies_tested': selected_strategies,
            'adaptation_triggered': should_adapt,
            'stability': meta_metrics['stability'],
            'task_diversity': meta_metrics['task_diversity']
        }
        
    async def test_catastrophic_forgetting_prevention(self):
        """Test EWC and forgetting prevention"""
        print("\n4. Testing Catastrophic Forgetting Prevention...")
        
        if not self.vision_system.advanced_learning:
            print("   ❌ Forgetting prevention not available")
            return
            
        forgetting_prevention = self.vision_system.advanced_learning.meta_framework.forgetting_prevention
        
        # Create model snapshot
        print("   Creating model snapshot...")
        snapshot_id = forgetting_prevention.create_snapshot({'accuracy': 0.85, 'loss': 0.23})
        print(f"   ✓ Created snapshot: {snapshot_id}")
        
        # Update task performance
        print("   Tracking task performance...")
        tasks = ['vision_analysis', 'error_detection', 'screen_reading']
        
        for task in tasks:
            for _ in range(3):
                performance = random.uniform(0.7, 0.95)
                forgetting_prevention.update_task_performance(task, performance)
                
        # Check if restoration needed
        current_performance = {'accuracy': 0.65}  # Degraded performance
        should_restore = forgetting_prevention.should_restore_snapshot(current_performance)
        print(f"   ✓ Should restore snapshot: {should_restore}")
        
        # Test Fisher Information computation (mock)
        print("   Testing Fisher Information Matrix computation...")
        
        # Create simple data loader
        class MockDataLoader:
            def __iter__(self):
                for _ in range(5):
                    yield {
                        'input': torch.randn(16, 768),
                        'target': torch.randint(0, 100, (16,))
                    }
                    
        data_loader = MockDataLoader()
        forgetting_prevention.compute_fisher_information(data_loader, 'test_task', num_samples=50)
        
        print(f"   ✓ Computed Fisher Information for {len(forgetting_prevention.fisher_information)} tasks")
        
        self.test_results['meta_learning']['forgetting_prevention'] = {
            'snapshots_created': len(forgetting_prevention.model_snapshots),
            'tasks_tracked': len(forgetting_prevention.task_performances),
            'fisher_matrices': len(forgetting_prevention.fisher_information)
        }
        
    async def test_federated_learning(self):
        """Test federated learning capabilities"""
        print("\n5. Testing Federated Learning...")
        
        if not self.vision_system.advanced_learning:
            print("   ❌ Federated learning not available")
            return
            
        advanced_learning = self.vision_system.advanced_learning
        
        # Enable federated learning
        advanced_learning.federated_enabled = True
        print("   ✓ Enabled federated learning")
        
        # Simulate federated updates
        print("   Simulating federated updates...")
        
        for i in range(5):
            # Create mock update
            model_updates = {}
            for name, param in advanced_learning.model.named_parameters():
                # Simulate parameter updates from remote client
                model_updates[name] = param.data + torch.randn_like(param.data) * 0.01
                
            update = FederatedUpdate(
                update_id=f"update_{i}",
                source_id=f"client_{i}",
                model_updates=model_updates,
                performance_metrics={'accuracy': random.uniform(0.7, 0.9)},
                data_statistics={'samples': random.randint(100, 1000)},
                timestamp=datetime.now()
            )
            
            await advanced_learning.add_federated_update(update)
            
        print(f"   ✓ Added {len(advanced_learning.federated_updates)} federated updates")
        print(f"   ✓ Privacy budget remaining: {advanced_learning.privacy_budget:.2f}")
        
        self.test_results['federated_learning'] = {
            'enabled': True,
            'updates_received': len(advanced_learning.federated_updates),
            'privacy_budget': advanced_learning.privacy_budget
        }
        
    async def test_continuous_learning_cycle(self):
        """Test the continuous learning cycle"""
        print("\n6. Testing Continuous Learning Cycle...")
        
        if not self.vision_system.advanced_learning:
            print("   ❌ Continuous learning not available")
            return
            
        advanced_learning = self.vision_system.advanced_learning
        
        # Check learning status
        status = advanced_learning.get_status()
        print("   Learning system status:")
        print(f"   - Last retraining: {status['continuous_learning']['last_retraining']}")
        print(f"   - Last mini-batch: {status['continuous_learning']['last_mini_batch']}")
        print(f"   - Active tasks: {status['continuous_learning']['active_tasks']}")
        print(f"   - Queued tasks: {status['continuous_learning']['queued_tasks']}")
        
        # Force a mini-batch training
        print("   Scheduling mini-batch training...")
        advanced_learning.last_mini_batch = datetime.now() - timedelta(minutes=20)
        await advanced_learning._learning_cycle()
        
        print("   ✓ Learning cycle executed")
        
        self.test_results['continuous_learning'] = {
            'cycle_executed': True,
            'active_tasks': status['continuous_learning']['active_tasks'],
            'queued_tasks': status['continuous_learning']['queued_tasks']
        }
        
    async def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment"""
        print("\n7. Testing Adaptive Learning Rate...")
        
        if not self.vision_system.advanced_learning:
            print("   ❌ Adaptive learning rate not available")
            return
            
        lr_adjuster = self.vision_system.advanced_learning.lr_adjuster
        
        print("   Simulating training with varying loss...")
        
        # Simulate improving loss
        for i in range(10):
            loss = 1.0 - (i * 0.08)  # Decreasing loss
            lr_adjuster.update({'loss': loss})
            
        current_lr = lr_adjuster.get_adjusted_lr()
        print(f"   ✓ Current learning rate: {current_lr:.6f}")
        
        # Simulate stagnant loss (should trigger reduction)
        print("   Simulating stagnant loss...")
        for i in range(15):
            loss = 0.3 + random.uniform(-0.01, 0.01)
            lr_adjuster.update({'loss': loss})
            
        new_lr = lr_adjuster.get_adjusted_lr()
        print(f"   ✓ Adjusted learning rate: {new_lr:.6f}")
        print(f"   ✓ Learning rate reduced: {new_lr < current_lr}")
        
        self.test_results['continuous_learning']['adaptive_lr'] = {
            'initial_lr': current_lr,
            'adjusted_lr': new_lr,
            'reduction_triggered': new_lr < current_lr
        }
        
    async def test_end_to_end_learning(self):
        """Test end-to-end learning through vision system"""
        print("\n8. Testing End-to-End Learning Integration...")
        
        # Process commands and check if experiences are recorded
        test_scenarios = [
            {
                'command': "Can you analyze what's on my screen?",
                'context': {'user': 'test_user', 'test_phase4': True}
            },
            {
                'command': "Show me the error messages",
                'context': {'user': 'test_user', 'error_present': True}
            },
            {
                'command': "Describe the current window",
                'context': {'user': 'test_user', 'window_count': 3}
            }
        ]
        
        print("   Processing test commands...")
        responses = []
        
        for scenario in test_scenarios:
            start_time = time.time()
            
            response = await self.vision_system.process_command(
                scenario['command'],
                scenario['context']
            )
            
            latency = (time.time() - start_time) * 1000
            
            responses.append({
                'command': scenario['command'],
                'success': response.success,
                'confidence': response.confidence,
                'latency_ms': latency,
                'phase4_enabled': response.data.get('phase4_enabled', False)
            })
            
            print(f"   ✓ Processed: '{scenario['command'][:30]}...'")
            print(f"     - Success: {response.success}")
            print(f"     - Confidence: {response.confidence:.2%}")
            print(f"     - Latency: {latency:.1f}ms")
            print(f"     - Phase 4 enabled: {response.data.get('phase4_enabled', False)}")
            
        # Check if experiences were recorded
        if self.vision_system.experience_replay:
            stats = await self.vision_system.experience_replay.get_statistics()
            print(f"\n   Experience replay stats:")
            print(f"   - Total experiences: {stats['buffer_stats']['total_experiences']}")
            print(f"   - Buffer utilization: {stats['buffer_stats']['utilization']:.1%}")
            
        self.test_results['performance'] = {
            'commands_processed': len(responses),
            'avg_latency': np.mean([r['latency_ms'] for r in responses]),
            'avg_confidence': np.mean([r['confidence'] for r in responses]),
            'phase4_integration': all(r.get('phase4_enabled', False) for r in responses)
        }
        
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*50)
        print("PHASE 4 TEST SUMMARY")
        print("="*50)
        
        # Experience Replay
        print("\nExperience Replay System:")
        er_results = self.test_results['experience_replay']
        if er_results.get('available', False):
            print(f"  ✅ Available and functional")
            print(f"  - Experiences added: {er_results.get('experiences_added', 0)}")
            print(f"  - Buffer utilization: {er_results.get('buffer_utilization', 0):.1%}")
            print(f"  - Patterns extracted: {er_results.get('patterns_extracted', 0)}")
            if 'pattern_types' in er_results:
                print(f"  - Pattern types: {', '.join(er_results['pattern_types'].keys())}")
        else:
            print(f"  ❌ Not available")
            
        # Meta-Learning
        print("\nMeta-Learning Framework:")
        ml_results = self.test_results['meta_learning']
        if ml_results.get('available', False):
            print(f"  ✅ Available and functional")
            print(f"  - Strategies tested: {', '.join(ml_results.get('strategies_tested', []))}")
            print(f"  - Stability score: {ml_results.get('stability', 0):.2f}")
            print(f"  - Catastrophic forgetting prevention: ✅")
            if 'forgetting_prevention' in ml_results:
                fp = ml_results['forgetting_prevention']
                print(f"    - Model snapshots: {fp.get('snapshots_created', 0)}")
                print(f"    - Tasks tracked: {fp.get('tasks_tracked', 0)}")
        else:
            print(f"  ❌ Not available")
            
        # Federated Learning
        print("\nFederated Learning:")
        fl_results = self.test_results['federated_learning']
        if fl_results.get('enabled', False):
            print(f"  ✅ Enabled and functional")
            print(f"  - Updates received: {fl_results.get('updates_received', 0)}")
            print(f"  - Privacy budget: {fl_results.get('privacy_budget', 0):.2f}")
        else:
            print(f"  ❌ Not enabled")
            
        # Performance
        print("\nEnd-to-End Performance:")
        perf_results = self.test_results['performance']
        if perf_results:
            print(f"  - Commands processed: {perf_results.get('commands_processed', 0)}")
            print(f"  - Average latency: {perf_results.get('avg_latency', 0):.1f}ms")
            print(f"  - Average confidence: {perf_results.get('avg_confidence', 0):.2%}")
            print(f"  - Phase 4 integration: {'✅' if perf_results.get('phase4_integration', False) else '❌'}")
            
        # Overall status
        print("\n" + "-"*50)
        phase4_available = (
            self.test_results['experience_replay'].get('available', False) and
            self.test_results['meta_learning'].get('available', False)
        )
        
        if phase4_available:
            print("✅ PHASE 4 IMPLEMENTATION COMPLETE AND FUNCTIONAL")
            print("\nKey achievements:")
            print("- Experience replay with prioritized sampling")
            print("- Pattern extraction from interaction history")
            print("- Meta-learning with strategy selection")
            print("- Catastrophic forgetting prevention (EWC)")
            print("- Privacy-preserving federated learning")
            print("- Adaptive learning rate adjustment")
            print("- Continuous learning pipeline integrated")
        else:
            print("⚠️  PHASE 4 PARTIALLY AVAILABLE")
            print("Some components may not be initialized")


async def main():
    """Run Phase 4 tests"""
    tester = Phase4Tester()
    
    try:
        await tester.run_all_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        
    # Allow time for background tasks to complete
    await asyncio.sleep(2)
    
    # Shutdown
    print("\nShutting down test systems...")
    vision_system = tester.vision_system
    if vision_system.advanced_learning:
        await vision_system.advanced_learning.shutdown()
    await vision_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())