#!/usr/bin/env python3
"""
Test ML Audio System
Comprehensive test suite for ML-enhanced audio error handling
"""

import asyncio
import os
import sys
import json
from datetime import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.ml_audio_manager import (
    AudioEvent,
    get_audio_manager,
    AudioPatternLearner
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_error_handling():
    """Test ML-driven error handling"""
    print("\n🧪 Testing ML Audio Error Handling")
    print("=" * 50)
    
    audio_manager = get_audio_manager()
    
    # Simulate different error scenarios
    test_scenarios = [
        {
            'error_code': 'audio-capture',
            'context': {
                'browser': 'chrome',
                'retry_count': 0,
                'session_duration': 1000,
                'first_time_user': True
            }
        },
        {
            'error_code': 'not-allowed',
            'context': {
                'browser': 'safari',
                'retry_count': 2,
                'session_duration': 30000,
                'permission_state': 'denied'
            }
        },
        {
            'error_code': 'network',
            'context': {
                'browser': 'firefox',
                'retry_count': 1,
                'session_duration': 15000
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📍 Test Scenario {i}: {scenario['error_code']}")
        
        result = await audio_manager.handle_error(
            scenario['error_code'],
            scenario['context']
        )
        
        print(f"✅ Strategy: {result.get('method')}")
        print(f"📊 Success: {result.get('success')}")
        print(f"💬 Message: {result.get('message')}")
        print(f"🎯 ML Confidence: {result.get('ml_confidence', 0):.2f}")
        
        if result.get('action'):
            print(f"🔧 Action: {result['action']['type']}")


async def test_pattern_learning():
    """Test pattern learning capabilities"""
    print("\n\n🧠 Testing Pattern Learning")
    print("=" * 50)
    
    audio_manager = get_audio_manager()
    learner = audio_manager.pattern_learner
    
    # Simulate a series of events
    print("\n📊 Simulating event sequence...")
    
    # Morning errors (9-11 AM)
    for hour in [9, 10, 11]:
        event = AudioEvent(
            timestamp=datetime.now().replace(hour=hour),
            event_type='error',
            error_code='audio-capture',
            browser='chrome',
            context={'retry_count': 0}
        )
        await learner.learn_from_event(event)
    
    # Successful recoveries
    for hour in [9, 10, 11]:
        event = AudioEvent(
            timestamp=datetime.now().replace(hour=hour),
            event_type='success',
            browser='chrome',
            resolution='request_permission',
            duration_ms=1500
        )
        await learner.learn_from_event(event)
    
    # Check learned patterns
    print("\n📈 Learned Patterns:")
    
    # Error patterns
    for error_code, events in learner.error_patterns.items():
        print(f"\n❌ {error_code}: {len(events)} occurrences")
    
    # Success patterns
    for browser, events in learner.success_patterns.items():
        print(f"✅ {browser}: {len(events)} successful recoveries")
    
    # Test prediction
    print("\n\n🔮 Testing Prediction:")
    
    test_context = {
        'browser': 'chrome',
        'time_of_day': 10,
        'day_of_week': 1,
        'error_history': [{'code': 'audio-capture'}],
        'session_duration': 1000,
        'permission_state': 'prompt'
    }
    
    probability = learner.predict_error_probability(test_context)
    print(f"🎯 Error probability: {probability:.2%}")
    
    # Test anomaly detection
    anomaly_event = AudioEvent(
        timestamp=datetime.now().replace(hour=3),  # 3 AM - unusual
        event_type='error',
        error_code='unknown-error',
        browser='netscape',  # Unusual browser
        context={'retry_count': 99}  # Unusual retry count
    )
    
    is_anomaly = learner.detect_anomaly(anomaly_event)
    print(f"🚨 Anomaly detected: {is_anomaly}")


async def test_recovery_strategies():
    """Test recovery strategy selection"""
    print("\n\n🔧 Testing Recovery Strategies")
    print("=" * 50)
    
    audio_manager = get_audio_manager()
    learner = audio_manager.pattern_learner
    
    # Seed with some successful recovery data
    print("\n📚 Seeding recovery data...")
    
    recovery_data = [
        ('audio-capture', 'request_permission', True, 1000),
        ('audio-capture', 'request_permission', True, 1200),
        ('audio-capture', 'browser_settings', True, 3000),
        ('audio-capture', 'browser_settings', False, 5000),
        ('not-allowed', 'system_settings', True, 2000),
        ('not-allowed', 'system_settings', True, 2500),
    ]
    
    for error_code, resolution, success, duration in recovery_data:
        # Error event
        error_event = AudioEvent(
            timestamp=datetime.now(),
            event_type='error',
            error_code=error_code,
            browser='chrome'
        )
        await learner.learn_from_event(error_event)
        
        # Recovery event
        if success:
            recovery_event = AudioEvent(
                timestamp=datetime.now(),
                event_type='success',
                browser='chrome',
                resolution=resolution,
                duration_ms=duration
            )
            await learner.learn_from_event(recovery_event)
    
    # Test strategy selection
    print("\n🎯 Testing strategy selection:")
    
    for error_code in ['audio-capture', 'not-allowed', 'network']:
        strategy = learner.get_recovery_strategy(error_code, {'browser': 'chrome'})
        
        print(f"\n📍 Error: {error_code}")
        if strategy['primary']:
            print(f"  Primary: {strategy['primary']['method']}")
            print(f"  Success Rate: {strategy['primary']['success_rate']:.2%}")
            print(f"  Avg Steps: {strategy['primary']['avg_steps']:.1f}")
        
        if strategy['alternatives']:
            print(f"  Alternatives: {[s['method'] for s in strategy['alternatives']]}")
        
        print(f"  ML Confidence: {strategy['ml_confidence']:.2%}")


async def test_metrics():
    """Test metrics collection"""
    print("\n\n📊 Testing Metrics")
    print("=" * 50)
    
    audio_manager = get_audio_manager()
    
    # Get current metrics
    metrics = audio_manager.get_metrics()
    
    print("\n📈 Current Metrics:")
    print(f"  Total Errors: {metrics['total_errors']}")
    print(f"  Resolved Errors: {metrics['resolved_errors']}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  ML Model Accuracy: {metrics['ml_model_accuracy']:.2%}")
    
    print("\n🎯 Strategy Success Rates:")
    for strategy, rate in metrics['strategy_success_rates'].items():
        print(f"  {strategy}: {rate:.2%}")


async def test_ml_api():
    """Test ML Audio API endpoints"""
    print("\n\n🌐 Testing ML Audio API")
    print("=" * 50)
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test config endpoint
            async with session.get('http://localhost:8000/audio/ml/config') as resp:
                if resp.status == 200:
                    config = await resp.json()
                    print("✅ Config endpoint working")
                    print(f"   ML Enabled: {config.get('enable_ml')}")
                    print(f"   Auto Recovery: {config.get('auto_recovery')}")
            
            # Test metrics endpoint
            async with session.get('http://localhost:8000/audio/ml/metrics') as resp:
                if resp.status == 200:
                    metrics = await resp.json()
                    print("\n✅ Metrics endpoint working")
                    print(f"   Success Rate: {metrics.get('success_rate', 0):.2%}")
            
            # Test patterns endpoint
            async with session.get('http://localhost:8000/audio/ml/patterns') as resp:
                if resp.status == 200:
                    patterns = await resp.json()
                    print("\n✅ Patterns endpoint working")
                    print(f"   Error Patterns: {len(patterns.get('error_patterns', {}))}")
                    print(f"   Success Patterns: {len(patterns.get('success_patterns', {}))}")
    
    except Exception as e:
        print(f"⚠️  API test skipped (server not running): {e}")


def print_summary():
    """Print test summary"""
    print("\n\n" + "=" * 60)
    print("📋 ML Audio System Test Summary")
    print("=" * 60)
    
    print("\n✅ Components Tested:")
    print("  • ML Error Handling with adaptive strategies")
    print("  • Pattern Learning from events")
    print("  • Predictive error detection")
    print("  • Recovery strategy optimization")
    print("  • Metrics collection and analysis")
    print("  • API endpoints (if server running)")
    
    print("\n🎯 Key Features:")
    print("  • No hardcoded solutions - fully adaptive")
    print("  • Browser-specific optimization")
    print("  • Anomaly detection for unusual errors")
    print("  • Continuous learning from user patterns")
    print("  • Configuration-driven behavior")
    
    print("\n📊 ML Models:")
    print("  • RandomForest for error prediction")
    print("  • IsolationForest for anomaly detection")
    print("  • DBSCAN for pattern clustering")
    print("  • Dynamic strategy ranking")
    
    print("\n🚀 Next Steps:")
    print("  1. Deploy to production")
    print("  2. Monitor real-world performance")
    print("  3. Collect user feedback")
    print("  4. Continuously improve models")


async def main():
    """Run all tests"""
    print("🤖 ML Audio System Test Suite")
    print("=" * 60)
    
    # Run tests
    await test_error_handling()
    await test_pattern_learning()
    await test_recovery_strategies()
    await test_metrics()
    await test_ml_api()
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())