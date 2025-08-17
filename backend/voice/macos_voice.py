"""
macOS native voice support using the 'say' command
Provides a British accent for JARVIS
"""

import subprocess
import os

class MacOSVoice:
    """Simple TTS using macOS 'say' command"""
    
    def __init__(self):
        # Get available voices
        result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
        self.voices = {}
        
        for line in result.stdout.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    voice_name = parts[0]
                    # Rest is the language/description
                    lang_desc = ' '.join(parts[1:])
                    self.voices[voice_name] = lang_desc
        
        # Find a British voice
        self.british_voice = None
        for voice, desc in self.voices.items():
            if 'en_GB' in desc or 'British' in desc:
                self.british_voice = voice
                break
        
        # Default to Daniel if no British voice found
        if not self.british_voice:
            self.british_voice = 'Daniel'  # British voice on macOS
            
        self.rate = 175  # Words per minute
        
    def say(self, text):
        """Speak the given text"""
        cmd = ['say', '-v', self.british_voice, '-r', str(self.rate)]
        subprocess.Popen(cmd + [text])
        
    def say_and_wait(self, text):
        """Speak the given text and wait for completion"""
        cmd = ['say', '-v', self.british_voice, '-r', str(self.rate)]
        subprocess.run(cmd + [text])
        
    def setProperty(self, name, value):
        """Set voice properties"""
        if name == 'rate':
            self.rate = int(value)
        elif name == 'voice':
            if value in self.voices:
                self.british_voice = value
                
    def getProperty(self, name):
        """Get voice properties"""
        if name == 'rate':
            return self.rate
        elif name == 'voice':
            return self.british_voice
        elif name == 'voices':
            return list(self.voices.keys())
            
    def runAndWait(self):
        """Compatibility method - does nothing since say_and_wait handles this"""
        pass