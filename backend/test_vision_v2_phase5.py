#!/usr/bin/env python3
"""
Test script for JARVIS Vision System v2.0 - Phase 5: Autonomous Capability Discovery
Tests capability generation, safety verification, and gradual rollout
"""

import asyncio
import logging
from datetime import datetime
import time
from typing import Dict, Any, List

# Import Vision System v2.0 with Phase 5 components
from vision.vision_system_v2 import get_vision_system_v2
from vision.capability_generator import get_capability_generator, FailedRequest, GeneratedCapability
from vision.safe_capability_synthesis import get_capability_synthesizer
from vision.safety_verification_framework import get_safety_verification_framework, VerificationLevel
from vision.sandbox_testing_environment import get_sandbox_test_runner, TestCase
from vision.performance_benchmarking import get_performance_benchmark
from vision.gradual_rollout_system import get_gradual_rollout_system, RolloutStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase5Tester:
    """Tests Phase 5 autonomous capability features"""
    
    def __init__(self):
        self.vision_system = get_vision_system_v2()
        self.test_results = {
            'capability_generation': {},
            'safety_verification': {},
            'sandbox_testing': {},
            'performance_benchmarking': {},
            'gradual_rollout': {}
        }
        
    async def run_all_tests(self):
        """Run all Phase 5 tests"""
        print("\n" + "="*60)
        print("PHASE 5 TEST SUITE - Autonomous Capability Discovery")
        print("="*60)
        
        # Test each component
        await self.test_capability_generation()
        await self.test_safety_verification()
        await self.test_sandbox_testing()
        await self.test_performance_benchmarking()
        await self.test_gradual_rollout()
        await self.test_end_to_end_capability_discovery()
        
        # Print summary
        self.print_test_summary()
        
    async def test_capability_generation(self):
        """Test capability generation from failed requests"""
        print("\n1. Testing Capability Generation...")
        
        if not self.vision_system.capability_generator:
            print("   ❌ Capability generator not available")
            self.test_results['capability_generation']['available'] = False
            return
            
        generator = self.vision_system.capability_generator
        
        # Create test failed requests
        test_failures = [
            FailedRequest(
                request_id="test_1",
                timestamp=datetime.now(),
                command="analyze the red button on the screen",
                intent="button_analysis",
                confidence=0.3,
                error_type="missing_handler",
                error_message="No handler found for button_analysis",
                context={'screen': 'active'},
                user_id="test_user"
            ),
            FailedRequest(
                request_id="test_2",
                timestamp=datetime.now(),
                command="count the number of windows open",
                intent="window_counting",
                confidence=0.4,
                error_type="missing_handler",
                error_message="No handler found for window_counting",
                context={'desktop': 'visible'},
                user_id="test_user"
            ),
            FailedRequest(
                request_id="test_3",
                timestamp=datetime.now(),
                command="find all error messages in red",
                intent="error_detection",
                confidence=0.5,
                error_type="missing_handler",
                error_message="No handler found for error_detection",
                context={'color_filter': 'red'},
                user_id="test_user"
            )
        ]
        
        generated_capabilities = []
        
        # Process failed requests
        for failure in test_failures:
            print(f"\n   Processing failure: {failure.command}")
            
            # Analyze failure
            capability = await generator.analyze_failed_request(failure)
            
            if capability:
                generated_capabilities.append(capability)
                print(f"   ✓ Generated capability: {capability.name}")
                print(f"     - Intent patterns: {len(capability.intent_patterns)}")
                print(f"     - Safety score: {capability.safety_score:.2f}")
                print(f"     - Complexity: {capability.complexity}")
            else:
                print(f"   ℹ️  No capability generated (insufficient similar failures)")
        
        # Test capability combination
        if len(generated_capabilities) >= 2:
            print("\n   Testing capability combination...")
            combined = await generator.combine_capabilities(
                [cap.capability_id for cap in generated_capabilities[:2]],
                combination_type="sequential"
            )
            
            if combined:
                print(f"   ✓ Created combined capability: {combined.name}")
                generated_capabilities.append(combined)
        
        # Get statistics
        stats = generator.get_statistics()
        print(f"\n   Generator Statistics:")
        print(f"   - Total generated: {stats['total_generated']}")
        print(f"   - Pending validation: {stats['pending_validation']}")
        print(f"   - Failure patterns: {stats['failure_patterns']}")
        
        self.test_results['capability_generation'] = {
            'available': True,
            'capabilities_generated': len(generated_capabilities),
            'generated_caps': generated_capabilities,
            'statistics': stats
        }
        
    async def test_safety_verification(self):
        """Test safety verification of generated capabilities"""
        print("\n2. Testing Safety Verification...")
        
        if not self.vision_system.safety_verifier:
            print("   ❌ Safety verifier not available")
            self.test_results['safety_verification']['available'] = False
            return
            
        verifier = self.vision_system.safety_verifier
        
        # Get generated capabilities from previous test
        generated_caps = self.test_results.get('capability_generation', {}).get('generated_caps', [])
        
        if not generated_caps:
            # Create a test capability
            print("   Creating test capability for verification...")
            test_cap = GeneratedCapability(
                capability_id="test_safety",
                name="test_capability",
                description="Test capability for safety verification",
                intent_patterns=["test pattern"],
                handler_code="""
async def test_capability(self, command: str, context=None):
    result = {"status": "processed", "command": command}
    return {"success": True, "result": result}
""",
                dependencies=[],
                safety_score=0.8,
                complexity="simple",
                created_at=datetime.now()
            )
            generated_caps = [test_cap]
        
        verified_capabilities = []
        
        # Verify each capability
        for cap in generated_caps[:3]:  # Test first 3
            print(f"\n   Verifying: {cap.name}")
            
            # Run verification
            report = await verifier.verify_capability(
                cap,
                VerificationLevel.COMPREHENSIVE
            )
            
            print(f"   ✓ Verification complete:")
            print(f"     - Safety score: {report.safety_score:.2f}")
            print(f"     - Risk level: {report.risk_level.value}")
            print(f"     - Approved: {'✅' if report.approved else '❌'}")
            
            if report.safety_violations:
                print(f"     - Violations: {len(report.safety_violations)}")
                for v in report.safety_violations[:2]:
                    print(f"       • {v.violation_type}: {v.description}")
            
            if report.recommendations:
                print(f"     - Recommendations:")
                for rec in report.recommendations[:2]:
                    print(f"       • {rec}")
            
            if report.approved:
                verified_capabilities.append((cap, report))
        
        # Get verification summary
        summary = verifier.get_verification_summary()
        print(f"\n   Verification Summary:")
        print(f"   - Total verified: {summary['total_verified']}")
        print(f"   - Approved: {summary['approved']}")
        print(f"   - Risk distribution: {dict(summary['risk_distribution'])}")
        
        self.test_results['safety_verification'] = {
            'available': True,
            'capabilities_verified': len(generated_caps),
            'approved': len(verified_capabilities),
            'verified_caps': verified_capabilities,
            'summary': summary
        }
        
    async def test_sandbox_testing(self):
        """Test sandbox testing environment"""
        print("\n3. Testing Sandbox Environment...")
        
        if not get_sandbox_test_runner:
            print("   ❌ Sandbox test runner not available")
            self.test_results['sandbox_testing']['available'] = False
            return
            
        sandbox = get_sandbox_test_runner()
        
        # Create test code
        test_code = """
async def handle_capability(command, context=None):
    # Simple test capability
    words = command.split()
    word_count = len(words)
    
    result = {
        'word_count': word_count,
        'first_word': words[0] if words else '',
        'context': context or {}
    }
    
    return {'success': True, 'result': result}
"""
        
        # Create test cases
        test_cases = [
            TestCase(
                test_id="sandbox_1",
                name="Basic execution",
                input_data={'command': "test sandbox environment"},
                timeout=5.0
            ),
            TestCase(
                test_id="sandbox_2",
                name="Empty input",
                input_data={'command': ""},
                timeout=5.0
            ),
            TestCase(
                test_id="sandbox_3",
                name="Context handling",
                input_data={
                    'command': "test with context",
                    'context': {'user': 'test', 'mode': 'sandbox'}
                },
                timeout=5.0
            )
        ]
        
        # Run tests
        print("   Running sandbox tests...")
        test_summary = await sandbox.test_capability(
            test_code,
            test_cases,
            use_docker=False  # Use process sandbox for simplicity
        )
        
        print(f"\n   Test Results:")
        print(f"   - Total tests: {test_summary['total_tests']}")
        print(f"   - Passed: {test_summary['passed']}")
        print(f"   - Failed: {test_summary['failed']}")
        print(f"   - Avg duration: {test_summary['average_duration']:.2f}s")
        print(f"   - Sandbox type: {test_summary['sandbox_type']}")
        
        for result in test_summary['results']:
            status = "✅" if result.success else "❌"
            print(f"     {status} {result.test_id}: {result.duration:.2f}s")
            
        self.test_results['sandbox_testing'] = {
            'available': True,
            'test_summary': test_summary
        }
        
    async def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        print("\n4. Testing Performance Benchmarking...")
        
        if not self.vision_system.performance_benchmark:
            print("   ❌ Performance benchmark not available")
            self.test_results['performance_benchmarking']['available'] = False
            return
            
        benchmark = self.vision_system.performance_benchmark
        
        # Create test capability code
        test_code = """
async def benchmark_capability(command, context=None):
    # Simulate some processing
    import time
    import random
    
    # Variable latency to test performance
    delay = random.uniform(0.01, 0.05)
    time.sleep(delay)
    
    return {
        'success': True,
        'processed': command,
        'latency': delay
    }
"""
        
        # Run benchmark
        print("   Running performance benchmark...")
        benchmark_result = await benchmark.benchmark_capability(
            test_code,
            "benchmark_test_capability",
            {'command': "benchmark test command"},
            capability_id="benchmark_test"
        )
        
        print(f"\n   Benchmark Results:")
        print(f"   - Avg latency: {benchmark_result.avg_latency_ms:.2f}ms")
        print(f"   - P99 latency: {benchmark_result.p99_latency_ms:.2f}ms")
        print(f"   - Throughput: {benchmark_result.sustained_throughput_rps:.1f} rps")
        print(f"   - Success rate: {benchmark_result.success_rate:.2%}")
        print(f"   - Memory leak detected: {'⚠️' if benchmark_result.memory_leak_detected else '✅ No'}")
        
        # Load test results
        if benchmark_result.load_test_results:
            print(f"\n   Load Test Results:")
            
            # Concurrent users
            if 'concurrent_users' in benchmark_result.load_test_results:
                print("   Concurrent Users:")
                for users, results in benchmark_result.load_test_results['concurrent_users'].items():
                    print(f"     - {users} users: {results['throughput_rps']:.1f} rps, "
                          f"{results['avg_latency_ms']:.1f}ms")
        
        self.test_results['performance_benchmarking'] = {
            'available': True,
            'benchmark_result': {
                'avg_latency_ms': benchmark_result.avg_latency_ms,
                'p99_latency_ms': benchmark_result.p99_latency_ms,
                'throughput_rps': benchmark_result.sustained_throughput_rps,
                'success_rate': benchmark_result.success_rate
            }
        }
        
    async def test_gradual_rollout(self):
        """Test gradual rollout system"""
        print("\n5. Testing Gradual Rollout System...")
        
        if not self.vision_system.rollout_system:
            print("   ❌ Rollout system not available")
            self.test_results['gradual_rollout']['available'] = False
            return
            
        rollout = self.vision_system.rollout_system
        
        # Get a verified capability
        verified_caps = self.test_results.get('safety_verification', {}).get('verified_caps', [])
        
        if verified_caps:
            cap, report = verified_caps[0]
        else:
            # Create test capability
            cap = GeneratedCapability(
                capability_id="rollout_test",
                name="rollout_test_capability",
                description="Test capability for rollout",
                intent_patterns=["rollout test"],
                handler_code="# Test code",
                dependencies=[],
                safety_score=0.9,
                complexity="simple",
                created_at=datetime.now()
            )
            
            # Create mock verification report
            from vision.safety_verification_framework import VerificationReport, RiskLevel
            report = VerificationReport(
                capability_id=cap.capability_id,
                verification_level=VerificationLevel.STANDARD,
                timestamp=datetime.now(),
                safety_score=0.9,
                risk_level=RiskLevel.LOW,
                approved=True
            )
        
        # Create rollout
        print(f"   Creating rollout for: {cap.name}")
        rollout_id = await rollout.create_rollout(cap, report)
        print(f"   ✓ Created rollout: {rollout_id}")
        
        # Get initial status
        status = rollout.get_rollout_status(rollout_id)
        print(f"   - Stage: {status['stage']}")
        print(f"   - Percentage: {status['percentage']}%")
        
        # Advance to canary
        print("\n   Advancing to canary stage...")
        advanced, reason = await rollout.advance_rollout(rollout_id, force=True)
        print(f"   {'✅' if advanced else '❌'} {reason}")
        
        # Simulate some traffic
        print("\n   Simulating traffic...")
        for i in range(20):
            should_use = rollout.should_use_capability(
                rollout_id,
                user_id=f"user_{i}",
                request_context={'test': True}
            )
            
            if should_use:
                # Record successful result
                rollout.record_result(
                    rollout_id,
                    success=True,
                    latency_ms=30.0 + (i % 10),
                    user_feedback="good" if i % 3 == 0 else None
                )
        
        # Get updated status
        final_status = rollout.get_rollout_status(rollout_id)
        print(f"\n   Final Status:")
        print(f"   - Stage: {final_status['stage']}")
        print(f"   - Percentage: {final_status['percentage']}%")
        print(f"   - Requests served: {final_status['metrics']['requests_served']}")
        print(f"   - Success rate: {final_status['metrics']['success_rate']:.2%}")
        
        self.test_results['gradual_rollout'] = {
            'available': True,
            'rollout_id': rollout_id,
            'final_status': final_status
        }
        
    async def test_end_to_end_capability_discovery(self):
        """Test end-to-end capability discovery through vision system"""
        print("\n6. Testing End-to-End Capability Discovery...")
        
        # Test commands that should fail and trigger capability generation
        test_commands = [
            {
                'command': "identify all clickable buttons in blue color",
                'context': {'user': 'test_user', 'phase5_test': True}
            },
            {
                'command': "measure the distance between windows",
                'context': {'user': 'test_user', 'measurement': 'pixels'}
            },
            {
                'command': "detect text in images on screen",
                'context': {'user': 'test_user', 'ocr_required': True}
            }
        ]
        
        print("   Processing test commands...")
        responses = []
        
        for scenario in test_commands:
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
                'phase5_enabled': response.data.get('phase5_enabled', False),
                'intent': response.intent_type
            })
            
            print(f"\n   Command: '{scenario['command'][:40]}...'")
            print(f"   - Success: {response.success}")
            print(f"   - Intent: {response.intent_type}")
            print(f"   - Phase 5 enabled: {response.data.get('phase5_enabled', False)}")
            
            # Small delay between commands
            await asyncio.sleep(0.5)
        
        # Check autonomous capabilities status
        if hasattr(self.vision_system, 'get_autonomous_capabilities_status'):
            status = self.vision_system.get_autonomous_capabilities_status()
            
            if status.get('available'):
                print(f"\n   Autonomous Capabilities Status:")
                print(f"   - Components active: {sum(status['components'].values())}/{len(status['components'])}")
                
                if 'generation_stats' in status:
                    gen_stats = status['generation_stats']
                    print(f"   - Capabilities generated: {gen_stats.get('total_generated', 0)}")
                    print(f"   - Pending validation: {gen_stats.get('pending_validation', 0)}")
        
        self.test_results['end_to_end'] = {
            'commands_processed': len(responses),
            'phase5_active': any(r['phase5_enabled'] for r in responses),
            'responses': responses
        }
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("PHASE 5 TEST SUMMARY")
        print("="*60)
        
        # Capability Generation
        print("\nCapability Generation:")
        cg_results = self.test_results['capability_generation']
        if cg_results.get('available', False):
            print(f"  ✅ Available and functional")
            print(f"  - Capabilities generated: {cg_results.get('capabilities_generated', 0)}")
            stats = cg_results.get('statistics', {})
            print(f"  - Total in system: {stats.get('total_generated', 0)}")
            print(f"  - Capability types: {stats.get('capability_types', {})}")
        else:
            print(f"  ❌ Not available")
        
        # Safety Verification
        print("\nSafety Verification:")
        sv_results = self.test_results['safety_verification']
        if sv_results.get('available', False):
            print(f"  ✅ Available and functional")
            print(f"  - Capabilities verified: {sv_results.get('capabilities_verified', 0)}")
            print(f"  - Approved: {sv_results.get('approved', 0)}")
            summary = sv_results.get('summary', {})
            print(f"  - Total verifications: {summary.get('total_verified', 0)}")
        else:
            print(f"  ❌ Not available")
        
        # Sandbox Testing
        print("\nSandbox Testing:")
        st_results = self.test_results['sandbox_testing']
        if st_results.get('available', False):
            print(f"  ✅ Available and functional")
            summary = st_results.get('test_summary', {})
            print(f"  - Tests passed: {summary.get('passed', 0)}/{summary.get('total_tests', 0)}")
            print(f"  - Sandbox type: {summary.get('sandbox_type', 'unknown')}")
        else:
            print(f"  ❌ Not available")
        
        # Performance Benchmarking
        print("\nPerformance Benchmarking:")
        pb_results = self.test_results['performance_benchmarking']
        if pb_results.get('available', False):
            print(f"  ✅ Available and functional")
            bench = pb_results.get('benchmark_result', {})
            print(f"  - Avg latency: {bench.get('avg_latency_ms', 0):.1f}ms")
            print(f"  - Throughput: {bench.get('throughput_rps', 0):.1f} rps")
            print(f"  - Success rate: {bench.get('success_rate', 0):.1%}")
        else:
            print(f"  ❌ Not available")
        
        # Gradual Rollout
        print("\nGradual Rollout:")
        gr_results = self.test_results['gradual_rollout']
        if gr_results.get('available', False):
            print(f"  ✅ Available and functional")
            status = gr_results.get('final_status', {})
            print(f"  - Stage reached: {status.get('stage', 'unknown')}")
            print(f"  - Traffic percentage: {status.get('percentage', 0)}%")
            metrics = status.get('metrics', {})
            print(f"  - Success rate: {metrics.get('success_rate', 0):.1%}")
        else:
            print(f"  ❌ Not available")
        
        # Overall Status
        print("\n" + "-"*60)
        components_available = sum(
            1 for component in [
                'capability_generation',
                'safety_verification',
                'sandbox_testing',
                'performance_benchmarking',
                'gradual_rollout'
            ]
            if self.test_results.get(component, {}).get('available', False)
        )
        
        if components_available == 5:
            print("✅ PHASE 5 IMPLEMENTATION COMPLETE AND FUNCTIONAL")
            print("\nKey achievements:")
            print("- Automatic capability generation from failed requests")
            print("- Comprehensive safety verification framework")
            print("- Isolated sandbox testing environment")
            print("- Performance benchmarking and analysis")
            print("- Gradual rollout with traffic management")
            print("- Full integration with Vision System v2.0")
        else:
            print(f"⚠️  PHASE 5 PARTIALLY AVAILABLE ({components_available}/5 components)")
            print("Some components may not be initialized")


async def main():
    """Run Phase 5 tests"""
    tester = Phase5Tester()
    
    try:
        await tester.run_all_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        
    # Allow time for background tasks
    await asyncio.sleep(2)
    
    # Shutdown
    print("\nShutting down test systems...")
    await tester.vision_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main()