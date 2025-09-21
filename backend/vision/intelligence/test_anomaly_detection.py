#!/usr/bin/env python3
"""
Test script for Anomaly Detection Framework
Demonstrates baseline establishment, real-time monitoring, and anomaly response
"""

import asyncio
import time
from datetime import datetime, timedelta
import random
from pathlib import Path

async def test_anomaly_detection():
    """Test the Anomaly Detection Framework"""
    print("🚨 Testing Anomaly Detection Framework")
    print("=" * 60)
    
    try:
        from anomaly_detection_framework import (
            AnomalyDetectionFramework, AnomalyType, AnomalySeverity,
            get_anomaly_detection_framework
        )
        
        # Initialize framework
        framework = get_anomaly_detection_framework()
        print("\n✅ Initialized Anomaly Detection Framework")
        
        # Test 1: Establish Baselines
        print("\n1️⃣ Establishing baselines from normal observations...")
        
        # Generate normal visual observations
        normal_visual_observations = []
        for i in range(50):
            obs = {
                'category': 'visual',
                'timestamp': datetime.now() - timedelta(minutes=50-i),
                'layout': {
                    'elements': [{'x': j*100, 'y': j*50, 'type': 'button'} 
                               for j in range(5)],
                    'anomaly_score': random.uniform(0.1, 0.3)
                },
                'colors': {
                    'dominant_hue': random.uniform(180, 220),
                    'palette': ['blue', 'white', 'gray']
                },
                'window': {
                    'windows': ['main_window'],
                    'has_modal': False
                }
            }
            normal_visual_observations.append(obs)
        
        visual_baseline = await framework.establish_baseline(
            normal_visual_observations, 'visual'
        )
        print(f"   Visual baseline established with {visual_baseline.sample_count} samples")
        
        # Generate normal behavioral observations  
        normal_behavioral_observations = []
        for i in range(50):
            obs = {
                'category': 'behavioral',
                'timestamp': datetime.now() - timedelta(minutes=50-i),
                'action_sequence': ['click', 'type', 'submit', 'wait'],
                'timing': {
                    'duration': random.uniform(5, 15),
                    'actions_per_minute': random.uniform(10, 30),
                    'idle_ratio': random.uniform(0.1, 0.3)
                },
                'navigation': {
                    'back_count': random.randint(0, 2),
                    'loop_score': random.uniform(0.0, 0.2)
                }
            }
            normal_behavioral_observations.append(obs)
        
        behavioral_baseline = await framework.establish_baseline(
            normal_behavioral_observations, 'behavioral'
        )
        print(f"   Behavioral baseline established with {behavioral_baseline.sample_count} samples")
        
        # Test 2: Real-time Monitoring - Normal Observations
        print("\n2️⃣ Testing real-time monitoring with normal observations...")
        
        normal_obs = {
            'category': 'visual',
            'layout': {
                'elements': [{'x': j*100, 'y': j*50} for j in range(5)],
                'anomaly_score': 0.2
            },
            'colors': {
                'dominant_hue': 200
            },
            'window': {
                'windows': ['main_window'],
                'has_modal': False
            }
        }
        
        anomalies = await framework.monitor_realtime(normal_obs)
        print(f"   Normal observation: {len(anomalies)} anomalies detected")
        
        # Test 3: Detect Visual Anomalies
        print("\n3️⃣ Testing visual anomaly detection...")
        
        # Unexpected popup
        popup_obs = {
            'category': 'visual',
            'window': {
                'has_modal': True,
                'modal_unexpected': True,
                'new_windows': [{'type': 'popup', 'expected': False}]
            }
        }
        anomalies = await framework.monitor_realtime(popup_obs)
        if anomalies:
            print(f"   ✅ Detected popup anomaly: {anomalies[0].description}")
            print(f"      Severity: {anomalies[0].severity.name}")
        
        # Error dialog
        error_obs = {
            'category': 'visual',
            'text': {
                'content': 'Error: Failed to save file. Access denied.'
            },
            'window': {
                'has_modal': True
            }
        }
        anomalies = await framework.monitor_realtime(error_obs)
        if anomalies:
            print(f"   ✅ Detected error dialog: {anomalies[0].description}")
        
        # Unusual layout
        layout_obs = {
            'category': 'visual',
            'layout': {
                'anomaly_score': 0.9,
                'missing_elements': ['save_button', 'cancel_button'],
                'overlap_count': 5
            }
        }
        anomalies = await framework.monitor_realtime(layout_obs)
        if anomalies:
            print(f"   ✅ Detected layout anomaly: {anomalies[0].description}")
        
        # Test 4: Detect Behavioral Anomalies
        print("\n4️⃣ Testing behavioral anomaly detection...")
        
        # Simulate repeated failures
        for i in range(5):
            failure_obs = {
                'category': 'behavioral',
                'outcome': 'failure',
                'action_sequence': ['click_submit'],
                'timestamp': datetime.now()
            }
            await framework.monitor_realtime(failure_obs)
        
        # Check for repeated failure anomaly
        check_obs = {
            'category': 'behavioral',
            'action_sequence': ['click_submit'],
            'outcome': 'failure'
        }
        anomalies = await framework.monitor_realtime(check_obs)
        if anomalies:
            failure_anomaly = next((a for a in anomalies 
                                  if a.anomaly_type == AnomalyType.REPEATED_FAILED_ATTEMPTS), None)
            if failure_anomaly:
                print(f"   ✅ Detected repeated failures: {failure_anomaly.description}")
        
        # Stuck state
        stuck_obs = {
            'category': 'behavioral',
            'state': {
                'state_id': 'loading_screen',
                'duration_seconds': 400  # Over 5 minutes
            }
        }
        anomalies = await framework.monitor_realtime(stuck_obs)
        if anomalies:
            print(f"   ✅ Detected stuck state: {anomalies[0].description}")
        
        # Circular pattern
        for i in range(3):
            for action in ['page_a', 'page_b', 'page_c', 'page_a']:
                circular_obs = {
                    'category': 'behavioral',
                    'navigation': {
                        'loop_score': 0.9 if i == 2 else 0.5
                    },
                    'state_sequence': ['page_a', 'page_b', 'page_c', 'page_a']
                }
                await framework.monitor_realtime(circular_obs)
        
        # Test 5: Detect System Anomalies
        print("\n5️⃣ Testing system anomaly detection...")
        
        # High CPU
        cpu_obs = {
            'category': 'system',
            'resources': {
                'cpu_percent': 95,
                'memory_percent': 60
            }
        }
        anomalies = await framework.monitor_realtime(cpu_obs)
        if anomalies:
            print(f"   ✅ Detected resource warning: {anomalies[0].description}")
        
        # Crash indicator
        crash_obs = {
            'category': 'system',
            'processes': {
                'crashed': 2
            },
            'text': {
                'content': 'Application quit unexpectedly'
            },
            'state_transition': {
                'unexpected': True
            }
        }
        anomalies = await framework.monitor_realtime(crash_obs)
        if anomalies:
            print(f"   ✅ Detected crash indicator: {anomalies[0].description}")
        
        # Test 6: Anomaly Response
        print("\n6️⃣ Testing anomaly response system...")
        
        if framework.active_anomalies:
            # Get a critical anomaly
            critical_anomaly = next((a for a in framework.active_anomalies.values() 
                                   if a.severity == AnomalySeverity.CRITICAL), None)
            
            if critical_anomaly:
                response = await framework.respond_to_anomaly(critical_anomaly, "auto")
                print(f"   Auto-response to critical anomaly: {response['action_taken']}")
                print(f"   Details: {response.get('details', 'N/A')}")
                
                # Investigate anomaly
                investigation = await framework.respond_to_anomaly(critical_anomaly, "investigate")
                if 'investigation' in investigation:
                    print(f"   Investigation results:")
                    print(f"   - Related anomalies: {len(investigation['investigation']['related_anomalies'])}")
                    print(f"   - Pattern frequency: {investigation['investigation']['pattern_analysis'].get('frequency', 0)}")
        
        # Test 7: Statistics
        print("\n7️⃣ Anomaly detection statistics...")
        
        stats = framework.get_statistics()
        print(f"   Total checks: {stats['total_checks']}")
        print(f"   Total anomalies: {stats['total_anomalies']}")
        print(f"   Detection rate: {stats['detection_rate']:.2%}")
        print(f"   Active anomalies: {stats['active_anomalies']}")
        
        if stats['anomaly_types_distribution']:
            print("\n   Anomaly type distribution:")
            for atype, count in stats['anomaly_types_distribution'].items():
                print(f"   - {atype}: {count}")
        
        if stats['severity_distribution']:
            print("\n   Severity distribution:")
            for severity, count in stats['severity_distribution'].items():
                print(f"   - {severity}: {count}")
        
        # Memory usage
        memory = framework.get_memory_usage()
        print(f"\n   Memory usage:")
        print(f"   - Baselines: {memory['baseline_models'] / 1024:.1f} KB")
        print(f"   - Rules: {memory['detection_rules'] / 1024:.1f} KB")
        print(f"   - History: {memory['anomaly_history'] / 1024:.1f} KB")
        print(f"   - Total: {memory['total'] / 1024:.1f} KB")
        
        # Test 8: ML-based Anomaly Detection
        print("\n8️⃣ Testing ML-based anomaly detection...")
        
        # Generate anomalous observation with extreme values
        anomalous_obs = {
            'category': 'visual',
            'layout': {
                'elements': [{'x': j*500, 'y': j*200} for j in range(20)],  # Way more elements
                'anomaly_score': 0.95
            },
            'colors': {
                'dominant_hue': 50  # Very different from baseline
            },
            'window': {
                'windows': ['window1', 'window2', 'window3', 'popup1', 'popup2'],  # Many windows
                'has_modal': True
            }
        }
        
        anomalies = await framework.monitor_realtime(anomalous_obs)
        ml_anomalies = [a for a in anomalies if 'ml_anomaly' in a.anomaly_id]
        if ml_anomalies:
            print(f"   ✅ ML model detected {len(ml_anomalies)} anomalies")
            for anomaly in ml_anomalies:
                print(f"      - {anomaly.description} (confidence: {anomaly.confidence:.2f})")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_rust_integration():
    """Test Rust anomaly detection integration"""
    print("\n🦀 Testing Rust Anomaly Detection Integration")
    print("=" * 50)
    
    try:
        # This would require the Rust library to be compiled
        print("   ⚠️  Rust integration requires compiled library")
        print("   To compile: cd jarvis-rust-core && cargo build --release")
        
        # If available, test the Rust detector
        # from jarvis_rust_core import PyAnomalyDetector
        # detector = PyAnomalyDetector()
        # ... test detector methods
        
    except Exception as e:
        print(f"   Error: {e}")


async def test_macos_integration():
    """Test macOS native anomaly detection"""
    print("\n🍎 Testing macOS Native Anomaly Detection")
    print("=" * 50)
    
    try:
        # This requires Swift compilation
        print("   ⚠️  macOS integration requires Swift compilation")
        print("   To compile: swiftc -o anomaly_detector anomaly_detection_macos.swift")
        
        # If available, test the macOS detector
        # Could use subprocess to call Swift binary or create Python bindings
        
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_anomaly_detection())
    # asyncio.run(test_rust_integration())
    # asyncio.run(test_macos_integration())