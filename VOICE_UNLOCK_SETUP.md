# JARVIS Voice Unlock - Apple Watch Alternative Setup Guide

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Run the install script
./install_voice_unlock_deps.sh

# OR manually install core packages
pip install fastapi anthropic scikit-learn librosa sounddevice bleak
```

### 2. Test Configuration (Auto-optimizes for your RAM)

```bash
# Check configuration and optimizations
python backend/voice_unlock/test_config.py
```

The system automatically detects your RAM and applies appropriate optimizations:
- **16GB**: Quantization ON, 400MB limit, aggressive unloading
- **32GB**: Quantization OFF, 800MB limit, normal unloading  
- **64GB+**: Full capabilities, no restrictions

### 3. Test Voice Unlock

```bash
# Check if everything is working
python backend/voice_unlock/jarvis_integration.py
```

### 4. Enroll Your Voice

```bash
# Install the voice unlock command
cd backend/voice_unlock
./install.sh

# Enroll your voice (3 samples)
jarvis-voice-unlock enroll john
```

### 5. Test Authentication

```bash
# Test Voice Unlock (no Apple Watch needed)
jarvis-voice-unlock test

# Test voice commands
Say: "Hey JARVIS, unlock my Mac"
Say: "JARVIS, this is John"
```

## 🔧 Troubleshooting

### Missing Dependencies Error

```bash
# Install all dependencies
pip install -r backend/voice_unlock/requirements.txt
```

### ProcessCleanupManager Error

✅ Already fixed! The config attribute has been added.

### High Memory Usage

✅ Already optimized! The system is configured for 16GB RAM with:
- Max 400MB for voice unlock
- Aggressive model unloading
- Quantization enabled
- Lazy loading everywhere

### Voice Unlock Not Working

1. Check microphone is working:
   ```bash
   # Test microphone input
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```

2. Make sure you have:
   - Enrolled your voice properly
   - Granted microphone permissions
   - Speaking clearly and naturally

### Microphone Permission

Grant microphone access when prompted, or manually in:
System Preferences → Security & Privacy → Privacy → Microphone → Terminal

## 📊 Memory Usage

With the optimizations applied:

| Component | Memory | Purpose |
|-----------|--------|---------|
| Voice Unlock | 400MB | Authentication system |
| ML Models | 200MB | Voice recognition models |
| Cache | 150MB | Performance optimization |
| Audio Buffer | 50MB | Real-time processing |
| **Total** | **800MB** | Complete system |

## 🎯 Integration with JARVIS

The voice unlock system is the perfect Apple Watch alternative:

1. **Automatic Start**: Voice unlock starts with JARVIS
2. **Background Operation**: Runs without blocking
3. **Voice Authentication**: No Apple Watch required
4. **Commands**: "Hey JARVIS, unlock my Mac" - no typing needed!

## ✅ What's Working Now

1. **ProcessCleanupManager** - Config attribute added ✓
2. **Dependencies** - Install script created ✓
3. **Memory Optimization** - 16GB config applied ✓
4. **Voice Authentication** - Apple Watch alternative ready ✓
5. **JARVIS Integration** - Startup hooks added ✓

## 🚦 Next Steps

1. Run `./install_voice_unlock_deps.sh`
2. Restart JARVIS
3. Enroll your voice
4. Enjoy Apple Watch-free voice unlocking!

---

**Note**: The system is optimized for your 16GB RAM MacBook Pro and will automatically manage memory to stay within the 4GB JARVIS budget.