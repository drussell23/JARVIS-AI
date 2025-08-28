"""
macOS native voice support using the 'say' command
Enhanced with voice variations and speech rate optimization
"""

import subprocess
import os
import tempfile
import threading
import queue
import time
from typing import Optional, List, Dict

class MacOSVoice:
    """Enhanced TTS using macOS 'say' command with professional features"""
    
    def __init__(self):
        # Get available voices
        self.voices = self._get_available_voices()
        
        # Find British voices (prioritized)
        self.british_voices = self._find_british_voices()
        
        # Select primary voice
        self.primary_voice = self._select_best_voice()
        
        # Speech settings
        self.rate = 175  # Words per minute (slightly faster for JARVIS)
        self.pitch_adjustment = 0  # Can be adjusted for effect
        
        # Queue for smooth speech delivery
        self.speech_queue = queue.Queue()
        self.speaking = False
        self.speech_thread = None
        
        # Voice variations for different contexts
        self.voice_modes = {
            'normal': {'rate': 175, 'voice': self.primary_voice},
            'urgent': {'rate': 200, 'voice': self.primary_voice},
            'thoughtful': {'rate': 160, 'voice': self.primary_voice},
            'quiet': {'rate': 150, 'voice': self.primary_voice}
        }
        
        # Start speech processing thread
        self._start_speech_thread()
        
    def _get_available_voices(self) -> Dict[str, str]:
        """Get all available system voices"""
        result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
        voices = {}
        
        for line in result.stdout.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    voice_name = parts[0]
                    # Rest is the language/description
                    lang_desc = ' '.join(parts[1:])
                    voices[voice_name] = lang_desc
        
        return voices
    
    def _find_british_voices(self) -> List[str]:
        """Find British English voices"""
        british_voices = []
        preferred_voices = ['Daniel', 'Oliver', 'Serena', 'Kate']  # Known British voices
        
        for voice, desc in self.voices.items():
            # Check for British indicators
            if any(indicator in desc for indicator in ['en_GB', 'British', 'United Kingdom']):
                british_voices.append(voice)
            # Also check known British voice names
            elif voice in preferred_voices:
                british_voices.append(voice)
        
        # Sort to prioritize male voices for JARVIS
        british_voices.sort(key=lambda v: 0 if v in ['Daniel', 'Oliver'] else 1)
        
        return british_voices
    
    def _select_best_voice(self) -> str:
        """Select the best available voice for JARVIS"""
        # Try British voices first
        if self.british_voices:
            return self.british_voices[0]
        
        # Fallback to any English voice
        english_voices = [v for v, d in self.voices.items() if 'en_' in d]
        if english_voices:
            # Prefer male voices
            male_keywords = ['Daniel', 'Oliver', 'Alex', 'Fred']
            for voice in english_voices:
                if any(keyword in voice for keyword in male_keywords):
                    return voice
            return english_voices[0]
        
        # Last resort - system default
        return 'Daniel'  # macOS default British voice
    
    def _start_speech_thread(self):
        """Start background thread for speech processing"""
        self.speech_thread = threading.Thread(target=self._speech_processor, daemon=True)
        self.speech_thread.start()
    
    def _speech_processor(self):
        """Process speech queue in background"""
        while True:
            try:
                text, mode, wait = self.speech_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                
                self.speaking = True
                self._speak_text(text, mode)
                self.speaking = False
                
                if wait:
                    # Signal completion
                    self.speech_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech processor error: {e}")
                self.speaking = False
    
    def _speak_text(self, text: str, mode: str = 'normal'):
        """Speak text with specified mode"""
        voice_config = self.voice_modes.get(mode, self.voice_modes['normal'])
        
        # Prepare say command
        cmd = [
            'say',
            '-v', voice_config['voice'],
            '-r', str(voice_config['rate'])
        ]
        
        # Add text processing for better speech
        processed_text = self._process_text_for_speech(text)
        
        # Execute speech
        subprocess.run(cmd + [processed_text])
    
    def _process_text_for_speech(self, text: str) -> str:
        """Process text for better speech synthesis"""
        # Add pauses for better rhythm
        replacements = {
            ', ': ', ... ',  # Short pause after commas
            '. ': '. ... ',  # Longer pause after periods
            '? ': '? ... ',  # Pause after questions
            '! ': '! ... ',  # Pause after exclamations
            'JARVIS': '[[rate -20]]JARVIS[[rate +20]]',  # Slow down for name
        }
        
        processed = text
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed
    
    def say(self, text: str, mode: str = 'normal'):
        """Speak the given text (non-blocking)"""
        self.speech_queue.put((text, mode, False))
    
    def say_and_wait(self, text: str, mode: str = 'normal'):
        """Speak the given text and wait for completion"""
        self.speech_queue.put((text, mode, True))
        self.speech_queue.join()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speaking or not self.speech_queue.empty()
    
    def stop_speaking(self):
        """Stop current speech and clear queue"""
        # Clear the queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        # Kill any running say process
        subprocess.run(['killall', 'say'], capture_output=True)
        self.speaking = False
    
    def set_voice_mode(self, mode: str):
        """Set voice mode for subsequent speech"""
        if mode in self.voice_modes:
            # Update default rate and voice
            self.rate = self.voice_modes[mode]['rate']
            self.primary_voice = self.voice_modes[mode]['voice']
    
    def setProperty(self, name: str, value):
        """Set voice properties (compatibility with pyttsx3)"""
        if name == 'rate':
            self.rate = int(value)
            # Update all voice modes
            for mode in self.voice_modes.values():
                mode['rate'] = self.rate
        elif name == 'voice':
            if value in self.voices:
                self.primary_voice = value
                # Update all voice modes
                for mode in self.voice_modes.values():
                    mode['voice'] = value
    
    def getProperty(self, name: str):
        """Get voice properties (compatibility with pyttsx3)"""
        if name == 'rate':
            return self.rate
        elif name == 'voice':
            return self.primary_voice
        elif name == 'voices':
            # Return voice objects similar to pyttsx3
            return [type('Voice', (), {'id': v, 'name': v})() for v in self.voices.keys()]
        return None
    
    def runAndWait(self):
        """Compatibility method - does nothing since say_and_wait handles this"""
        pass
    
    def save_to_file(self, text: str, filename: str):
        """Save speech to audio file"""
        cmd = [
            'say',
            '-v', self.primary_voice,
            '-r', str(self.rate),
            '-o', filename,
            '--data-format=LEF32@22050'  # Linear PCM format
        ]
        
        processed_text = self._process_text_for_speech(text)
        subprocess.run(cmd + [processed_text])
    
    def get_voice_info(self) -> Dict[str, any]:
        """Get information about current voice configuration"""
        return {
            'primary_voice': self.primary_voice,
            'available_voices': list(self.voices.keys()),
            'british_voices': self.british_voices,
            'current_rate': self.rate,
            'voice_modes': list(self.voice_modes.keys()),
            'is_speaking': self.is_speaking()
        }
    
    def __del__(self):
        """Cleanup speech thread on deletion"""
        if hasattr(self, 'speech_queue'):
            self.speech_queue.put((None, None, None))  # Shutdown signal
            if hasattr(self, 'speech_thread') and self.speech_thread:
                self.speech_thread.join(timeout=1)

# Enhanced testing and demo
if __name__ == "__main__":
    print("Testing Enhanced macOS Voice System...")
    
    voice = MacOSVoice()
    
    # Show voice info
    info = voice.get_voice_info()
    print(f"\nVoice Configuration:")
    print(f"  Primary Voice: {info['primary_voice']}")
    print(f"  British Voices Found: {info['british_voices']}")
    print(f"  Speech Rate: {info['current_rate']} WPM")
    
    # Test different voice modes
    print("\n--- Testing Voice Modes ---")
    
    # Normal mode
    print("Normal mode:")
    voice.say_and_wait("Good evening, sir. All systems are operational.", mode='normal')
    time.sleep(0.5)
    
    # Urgent mode
    print("Urgent mode:")
    voice.say_and_wait("Sir, we have an urgent situation requiring your attention!", mode='urgent')
    time.sleep(0.5)
    
    # Thoughtful mode
    print("Thoughtful mode:")
    voice.say_and_wait("Hmm, that's an interesting question. Let me consider the possibilities.", mode='thoughtful')
    time.sleep(0.5)
    
    # Test JARVIS personality
    print("\n--- JARVIS Personality Test ---")
    jarvis_phrases = [
        "Welcome home, sir. Shall I prepare the workshop?",
        "The weather is partly cloudy, 72 degrees. Perfect for flying, if I may say so, sir.",
        "Sir, your heart rate suggests you haven't taken a break in 3 hours. Might I suggest a brief respite?",
        "Of course, sir. Shall I also cancel your 3 o'clock? You do have a tendency to lose track of time."
    ]
    
    for phrase in jarvis_phrases:
        print(f"Speaking: {phrase[:50]}...")
        voice.say_and_wait(phrase)
        time.sleep(1)
    
    print("\n✅ Enhanced macOS Voice System Test Complete!")