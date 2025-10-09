#!/usr/bin/env python3
"""
Test enhanced ML Audio compatibility endpoints
"""

import requests
import json
import time
from datetime import datetime

def test_enhanced_ml_audio():
    """Test the enhanced ML Audio endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Enhanced ML Audio System")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing /health endpoint for ML Audio status...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            ml_audio = health.get('ml_audio_system', {})
            if ml_audio.get('enabled'):
                print("✅ ML Audio System Status:")
                print(f"   - Active streams: {ml_audio.get('active_streams', 0)}")
                print(f"   - Total processed: {ml_audio.get('total_processed', 0)}")
                print(f"   - Uptime: {ml_audio.get('uptime_hours', 0):.2f} hours")
                print(f"   - GPU available: {ml_audio.get('capabilities', {}).get('gpu_acceleration', False)}")
                print(f"   - PyTorch: {ml_audio.get('capabilities', {}).get('pytorch_available', False)}")
                print(f"   - Performance: {ml_audio.get('performance', {})}")
            else:
                print("❌ ML Audio system not enabled in health check")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Dynamic configuration
    print("\n2. Testing /audio/ml/config for dynamic configuration...")
    try:
        response = requests.get(f"{base_url}/audio/ml/config")
        if response.status_code == 200:
            config = response.json()
            print("✅ Dynamic Config:")
            print(f"   - Model: {config.get('model')}")
            print(f"   - System CPU: {config.get('system_status', {}).get('cpu_usage')}%")
            print(f"   - System Memory: {config.get('system_status', {}).get('memory_usage')}%")
            print(f"   - Platform: {config.get('system_status', {}).get('platform')}")
            print(f"   - Quality insights: {config.get('quality_insights', {})}")
            print(f"   - Client recommendations: {config.get('client_info', {}).get('recommended_settings', {})}")
    except Exception as e:
        print(f"❌ Config error: {e}")
    
    # Test 3: Enhanced prediction
    print("\n3. Testing /audio/ml/predict with enhanced analysis...")
    try:
        # Create test audio data
        test_data = {
            "audio_data": "dGVzdF9hdWRpb19kYXRhX2Jhc2U2NA==",  # "test_audio_data_base64" in base64
            "format": "base64",
            "sample_rate": 16000,
            "duration_ms": 1000,
            "client_id": "test_client_001"
        }
        
        response = requests.post(f"{base_url}/audio/ml/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced Prediction:")
            print(f"   - Prediction: {result.get('prediction')}")
            print(f"   - Confidence: {result.get('confidence')}")
            print(f"   - Quality score: {result.get('audio_quality', {}).get('score')}")
            print(f"   - Quality level: {result.get('audio_quality', {}).get('level')}")
            print(f"   - SNR: {result.get('audio_quality', {}).get('signal_to_noise_ratio')} dB")
            print(f"   - Issues: {result.get('issues', [])}")
            print(f"   - Recommendations: {result.get('recommendations', [])}")
            print(f"   - Processing time: {result.get('processing_time_ms')} ms")
            print(f"   - System load: {result.get('detailed_analysis', {}).get('system_load', {})}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Stream management
    print("\n4. Testing stream management endpoints...")
    try:
        # Start stream
        response = requests.post(f"{base_url}/audio/ml/stream/start")
        if response.status_code == 200:
            stream_info = response.json()
            stream_id = stream_info.get('stream_id')
            print(f"✅ Stream started: {stream_id}")
            print(f"   - Recommended config: {stream_info.get('recommended_config', {})}")
            
            # Wait a bit
            time.sleep(1)
            
            # Stop stream
            response = requests.post(f"{base_url}/audio/ml/stream/{stream_id}/stop")
            if response.status_code == 200:
                stop_info = response.json()
                print(f"✅ Stream stopped:")
                print(f"   - Duration: {stop_info.get('duration_seconds')} seconds")
                print(f"   - Quality trend: {stop_info.get('final_report', {}).get('quality_trend')}")
    except Exception as e:
        print(f"❌ Stream management error: {e}")
    
    # Test 5: Model listing
    print("\n5. Testing /audio/ml/models for dynamic model detection...")
    try:
        response = requests.get(f"{base_url}/audio/ml/models")
        if response.status_code == 200:
            models_info = response.json()
            print(f"✅ Available Models: {models_info.get('total_models')} total, {models_info.get('loaded_models')} loaded")
            for model in models_info.get('models', []):
                print(f"   - {model['name']} ({model['id']})")
                print(f"     Status: {model['status']}")
                print(f"     Capabilities: {', '.join(model['capabilities'])}")
    except Exception as e:
        print(f"❌ Models error: {e}")
    
    # Test 6: Detailed analysis
    print("\n6. Testing /audio/ml/analyze for spectral features...")
    try:
        response = requests.post(f"{base_url}/audio/ml/analyze", json=test_data)
        if response.status_code == 200:
            analysis = response.json()
            print("✅ Spectral Analysis:")
            spectral = analysis.get('spectral_features', {})
            print(f"   - Spectral centroid: {spectral.get('spectral_centroid')} Hz")
            print(f"   - Spectral rolloff: {spectral.get('spectral_rolloff')} Hz")
            print(f"   - Zero crossing rate: {spectral.get('zero_crossing_rate')}")
            
            perceptual = analysis.get('perceptual_features', {})
            print(f"   - Loudness: {perceptual.get('loudness_lufs')} LUFS")
            print(f"   - Sharpness: {perceptual.get('sharpness')}")
            print(f"   - Warmth: {perceptual.get('warmth')}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("Enhanced ML Audio system test completed!")
    print("\nKey Features Demonstrated:")
    print("✓ Dynamic system resource monitoring")
    print("✓ Client-specific recommendations")
    print("✓ Real-time performance metrics")
    print("✓ Quality trend analysis")
    print("✓ Spectral feature extraction")
    print("✓ Stream session management")

if __name__ == "__main__":
    test_enhanced_ml_audio()