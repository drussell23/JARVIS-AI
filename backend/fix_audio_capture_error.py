#!/usr/bin/env python3
"""
Fix JARVIS Audio Capture Error
Diagnoses and provides solutions for microphone access issues
"""

import os
import sys
import subprocess
import platform

def check_microphone_permissions():
    """Check macOS microphone permissions"""
    print("\n🎤 Checking Microphone Permissions")
    print("=" * 50)
    
    if platform.system() != 'Darwin':
        print("This script is for macOS only")
        return
    
    # Check if any browser has microphone access
    print("\n1️⃣ Checking browser microphone permissions...")
    
    browsers = {
        'Google Chrome': 'com.google.Chrome',
        'Safari': 'com.apple.Safari',
        'Firefox': 'org.mozilla.firefox',
        'Brave': 'com.brave.Browser',
        'Edge': 'com.microsoft.edgemac'
    }
    
    for browser_name, bundle_id in browsers.items():
        result = subprocess.run(
            ['osascript', '-e', f'tell application "System Events" to get exists (processes where bundle identifier is "{bundle_id}")'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip() == 'true':
            print(f"✅ {browser_name} is running")
            
    print("\n2️⃣ macOS Microphone Permission Status:")
    print("Go to: System Preferences → Security & Privacy → Privacy → Microphone")
    print("Ensure your browser is checked ✅")
    
    print("\n3️⃣ Browser-Specific Fixes:")
    print("\n🔵 Chrome/Brave:")
    print("1. Click the lock icon 🔒 in address bar")
    print("2. Set Microphone to 'Allow'")
    print("3. Or go to: chrome://settings/content/microphone")
    print("4. Add http://localhost:3000 to allowed sites")
    
    print("\n🟠 Firefox:")
    print("1. Click the lock icon 🔒 in address bar")
    print("2. Click '>' next to 'Connection Secure'")
    print("3. Set Microphone to 'Allow'")
    
    print("\n🔵 Safari:")
    print("1. Safari → Preferences → Websites → Microphone")
    print("2. Set localhost to 'Allow'")
    
    print("\n4️⃣ Quick Fix Commands:")
    print("```bash")
    print("# Kill all audio processes and restart")
    print("sudo killall coreaudiod")
    print("")
    print("# Reset audio permissions")
    print("tccutil reset Microphone")
    print("```")


def create_enhanced_jarvis_voice_fix():
    """Create an enhanced version of JarvisVoice.js with better error handling"""
    
    fix_content = '''// Add this to JarvisVoice.js to handle audio-capture errors better

      recognitionRef.current.onerror = async (event) => {
        console.error('Speech recognition error:', event.error, event);
        
        // Handle different error types
        switch(event.error) {
          case 'audio-capture':
            setError('🎤 Microphone access denied. Please grant permission and reload.');
            setMicStatus('error');
            
            // Try to request permissions again
            try {
              const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
              stream.getTracks().forEach(track => track.stop());
              
              // If we get here, permissions were granted
              setError('');
              setMicStatus('ready');
              startListening();
            } catch (e) {
              console.error('Failed to get microphone access:', e);
              
              // Show detailed instructions
              const instructions = `
                🎤 Microphone Access Required
                
                Please follow these steps:
                1. Click the lock icon 🔒 in your browser's address bar
                2. Set Microphone to "Allow"
                3. Reload the page
                
                Or check System Preferences → Security & Privacy → Microphone
              `;
              setError(instructions);
            }
            break;
            
          case 'not-allowed':
            setError('🚫 Microphone permission denied. Please enable in browser settings.');
            setMicStatus('error');
            break;
            
          case 'no-speech':
            // Silently restart for no-speech
            if (continuousListening && isListening) {
              setTimeout(() => {
                try {
                  recognitionRef.current.start();
                } catch (e) {
                  console.log('Restarting recognition...');
                }
              }, 100);
            }
            break;
            
          case 'network':
            setError('🌐 Network error. Check your connection.');
            break;
            
          default:
            if (event.error !== 'aborted') {
              setError(`Speech recognition error: ${event.error}`);
            }
        }
      };
'''
    
    with open('jarvis_voice_audio_fix.js', 'w') as f:
        f.write(fix_content)
    
    print("\n📝 Created enhanced error handling code: jarvis_voice_audio_fix.js")


def test_microphone_access():
    """Test microphone access with Python"""
    print("\n5️⃣ Testing Python Microphone Access...")
    
    try:
        import pyaudio
        
        # Try to open microphone
        p = pyaudio.PyAudio()
        
        # List audio devices
        print("\nAvailable audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        
        # Try to open default microphone
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            stream.close()
            print("\n✅ Python can access the microphone!")
        except Exception as e:
            print(f"\n❌ Python cannot access microphone: {e}")
        
        p.terminate()
        
    except ImportError:
        print("PyAudio not installed. Install with: pip install pyaudio")
    except Exception as e:
        print(f"Error testing microphone: {e}")


def create_browser_permission_test():
    """Create a simple HTML page to test microphone permissions"""
    
    test_html = '''<!DOCTYPE html>
<html>
<head>
    <title>JARVIS Microphone Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .status {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 18px;
        }
        .success { background: #27ae60; }
        .error { background: #e74c3c; }
        .warning { background: #f39c12; }
        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover { background: #2980b9; }
        #audioLevel {
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        #audioBar {
            height: 100%;
            background: #3498db;
            width: 0%;
            transition: width 0.1s;
        }
    </style>
</head>
<body>
    <h1>🎤 JARVIS Microphone Test</h1>
    
    <div id="status" class="status warning">
        Click "Test Microphone" to check permissions
    </div>
    
    <button onclick="testMicrophone()">Test Microphone</button>
    
    <div id="audioLevel" style="display: none;">
        <div id="audioBar"></div>
    </div>
    
    <div id="instructions" style="display: none; margin-top: 30px;">
        <h2>✅ Microphone Working!</h2>
        <p>Audio level indicator shows your microphone input.</p>
        <p>You can now use JARVIS voice commands!</p>
    </div>
    
    <script>
        let audioContext;
        let analyser;
        let microphone;
        let javascriptNode;
        
        async function testMicrophone() {
            const statusDiv = document.getElementById('status');
            
            try {
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                statusDiv.className = 'status success';
                statusDiv.textContent = '✅ Microphone access granted!';
                
                // Set up audio visualization
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
                
                analyser.smoothingTimeConstant = 0.8;
                analyser.fftSize = 1024;
                
                microphone.connect(analyser);
                analyser.connect(javascriptNode);
                javascriptNode.connect(audioContext.destination);
                
                javascriptNode.onaudioprocess = function() {
                    const array = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(array);
                    const average = array.reduce((a, b) => a + b) / array.length;
                    
                    // Update audio level bar
                    const audioBar = document.getElementById('audioBar');
                    audioBar.style.width = Math.min(100, average * 2) + '%';
                }
                
                document.getElementById('audioLevel').style.display = 'block';
                document.getElementById('instructions').style.display = 'block';
                
                // Test speech recognition
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    statusDiv.textContent += ' Speech recognition available!';
                } else {
                    statusDiv.textContent += ' (Speech recognition not supported in this browser)';
                }
                
            } catch (error) {
                console.error('Microphone error:', error);
                statusDiv.className = 'status error';
                
                if (error.name === 'NotAllowedError') {
                    statusDiv.innerHTML = `
                        ❌ Microphone access denied<br><br>
                        Please grant permission:<br>
                        1. Click the 🔒 icon in the address bar<br>
                        2. Set Microphone to "Allow"<br>
                        3. Reload the page
                    `;
                } else if (error.name === 'NotFoundError') {
                    statusDiv.textContent = '❌ No microphone found. Please connect a microphone.';
                } else {
                    statusDiv.textContent = '❌ Error: ' + error.message;
                }
            }
        }
        
        // Check current permission status
        if (navigator.permissions) {
            navigator.permissions.query({ name: 'microphone' }).then(result => {
                const statusDiv = document.getElementById('status');
                if (result.state === 'granted') {
                    statusDiv.className = 'status success';
                    statusDiv.textContent = '✅ Microphone permission already granted. Click to test.';
                } else if (result.state === 'denied') {
                    statusDiv.className = 'status error';
                    statusDiv.textContent = '❌ Microphone permission denied. Please enable in browser settings.';
                }
            });
        }
    </script>
</body>
</html>'''
    
    with open('test_microphone.html', 'w') as f:
        f.write(test_html)
    
    print("\n📝 Created microphone test page: test_microphone.html")
    print("Open this file in your browser to test microphone access")


def main():
    """Run all diagnostic checks"""
    print("🎤 JARVIS Audio Capture Error Fix")
    print("=" * 60)
    
    # Run checks
    check_microphone_permissions()
    create_enhanced_jarvis_voice_fix()
    test_microphone_access()
    create_browser_permission_test()
    
    print("\n\n✅ Diagnostic complete!")
    print("\n📋 To fix the audio-capture error:")
    print("1. Open test_microphone.html in your browser")
    print("2. Grant microphone permission when prompted")
    print("3. Check browser settings (see instructions above)")
    print("4. Apply the enhanced error handling from jarvis_voice_audio_fix.js")
    print("5. Restart JARVIS")
    
    print("\n🚀 Quick fix:")
    print("1. Go to http://localhost:3000")
    print("2. Click the 🔒 icon in address bar")
    print("3. Set Microphone to 'Allow'")
    print("4. Reload the page")


if __name__ == "__main__":
    main()