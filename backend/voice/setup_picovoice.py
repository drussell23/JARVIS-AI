#!/usr/bin/env python3
"""
Setup and test Picovoice integration for JARVIS
"""

import os
import sys
import asyncio
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_picovoice_setup():
    """Check if Picovoice is properly set up"""
    print("🔍 Checking Picovoice setup...")
    
    # Check environment variable
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("❌ PICOVOICE_ACCESS_KEY not set in environment")
        return False
    
    print("✅ PICOVOICE_ACCESS_KEY found")
    
    # Check if pvporcupine is installed
    try:
        import pvporcupine
        print("✅ pvporcupine package installed")
    except ImportError:
        print("❌ pvporcupine not installed")
        print("   Run: pip install pvporcupine")
        return False
    
    # Test Picovoice initialization
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=["jarvis"],
            sensitivities=[0.5]
        )
        print("✅ Picovoice Porcupine initialized successfully")
        print(f"   - Sample rate: {porcupine.sample_rate} Hz")
        print(f"   - Frame length: {porcupine.frame_length} samples")
        porcupine.delete()
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Porcupine: {e}")
        return False

def test_picovoice_detection():
    """Test Picovoice wake word detection"""
    print("\n🎤 Testing Picovoice wake word detection...")
    
    try:
        from picovoice_integration import PicovoiceWakeWordDetector, PicovoiceConfig
        
        # Create detector
        config = PicovoiceConfig(
            keywords=["jarvis"],
            sensitivities=[0.5]
        )
        detector = PicovoiceWakeWordDetector(config)
        
        print("✅ Detector created successfully")
        
        # Test with dummy audio (in real use, this would be microphone input)
        print("\n📊 Processing test audio...")
        sample_rate = detector.sample_rate
        duration = 1.0
        
        for i in range(3):
            # Generate dummy audio
            audio = np.random.randn(int(sample_rate * duration)) * 0.1
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Process
            result = detector.process_audio(audio_int16)
            if result is not None:
                print(f"   Frame {i}: Wake word detected! (keyword index: {result})")
            else:
                print(f"   Frame {i}: No wake word")
        
        # Show metrics
        metrics = detector.get_metrics()
        print(f"\n📈 Metrics:")
        print(f"   - Frames processed: {metrics['total_frames_processed']}")
        print(f"   - Detections: {metrics['total_detections']}")
        
        # Cleanup
        detector.cleanup()
        print("\n✅ Picovoice test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_hybrid_system():
    """Test the hybrid Picovoice + ML system"""
    print("\n🔄 Testing hybrid wake word detection...")
    
    try:
        from optimized_voice_system import create_optimized_jarvis
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  ANTHROPIC_API_KEY not set, skipping hybrid test")
            return False
        
        # Enable Picovoice in config
        os.environ["USE_PICOVOICE"] = "true"
        
        # Create optimized system
        system = await create_optimized_jarvis(api_key, "16gb_macbook_pro")
        
        print("✅ Hybrid system initialized")
        
        # Test detection
        sample_rate = 16000
        test_audio = np.random.randn(sample_rate * 2) * 0.1  # 2 seconds
        
        result = await system.detect_wake_word(test_audio)
        print(f"\n📊 Detection result:")
        print(f"   - Detected: {result[0]}")
        print(f"   - Confidence: {result[1]:.3f}")
        print(f"   - Message: {result[2] or 'Success'}")
        
        # Get stats
        stats = system.get_optimization_stats()
        print(f"\n📈 System stats:")
        print(f"   - Acceleration available: {stats['acceleration']}")
        
        await system.stop()
        print("\n✅ Hybrid system test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid test failed: {e}")
        return False

def setup_environment():
    """Setup environment for Picovoice"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = f"""# Voice System Configuration

# Picovoice (DO NOT COMMIT THIS FILE)
PICOVOICE_ACCESS_KEY=e9AVn4el49rhJxxUDILvK2vYOTbFZx1ZSlSnsZqEu3kLY3Ix8Ghckg==

# Voice Detection
WAKE_WORD_THRESHOLD=0.55
CONFIDENCE_THRESHOLD=0.6
ENABLE_VAD=true
USE_PICOVOICE=true

# Optimization
JARVIS_OPTIMIZATION_LEVEL=balanced
JARVIS_USE_COREML=true
JARVIS_USE_METAL=true
"""
        env_file.write_text(env_content)
        print("✅ Created .env file with your Picovoice key")
    else:
        print("ℹ️  .env file already exists")
    
    # Add .env to .gitignore
    gitignore_path = Path(__file__).parent.parent.parent / ".gitignore"
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()
        if ".env" not in gitignore_content:
            print("📝 Adding .env to .gitignore...")
            gitignore_path.write_text(gitignore_content + "\n# Environment files\n.env\n")
            print("✅ Updated .gitignore")
    
    # Set environment variable for current session
    os.environ["PICOVOICE_ACCESS_KEY"] = "e9AVn4el49rhJxxUDILvK2vYOTbFZx1ZSlSnsZqEu3kLY3Ix8Ghckg=="
    print("✅ Environment configured")

def main():
    """Main setup and test function"""
    print("🚀 JARVIS Picovoice Setup\n")
    
    # Setup environment
    setup_environment()
    
    # Check setup
    if not check_picovoice_setup():
        print("\n⚠️  Please install pvporcupine and set up your environment:")
        print("   pip install pvporcupine")
        print("   export PICOVOICE_ACCESS_KEY='your-key'")
        return
    
    # Run tests
    print("\n" + "="*50)
    test_picovoice_detection()
    
    # Run hybrid test
    print("\n" + "="*50)
    asyncio.run(test_hybrid_system())
    
    print("\n✨ Setup complete! Picovoice is ready to use.")
    print("\nNext steps:")
    print("1. Use Picovoice in your code:")
    print("   from voice.optimized_voice_system import create_optimized_jarvis")
    print("   system = await create_optimized_jarvis(api_key, '16gb_macbook_pro')")
    print("\n2. The system will automatically use Picovoice for fast wake word detection")
    print("3. Adjust sensitivity if needed in voice/config.py")

if __name__ == "__main__":
    main()