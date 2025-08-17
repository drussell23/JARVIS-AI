"""
JARVIS Voice System - Full voice interaction with personality
Integrates with Claude API for Iron Man-style AI assistant
"""

import asyncio
import speech_recognition as sr
import pygame
import numpy as np
from typing import Optional, Callable
import json
import random
from datetime import datetime
import threading
import queue
import os
import sys
import platform
from anthropic import Anthropic

# Use macOS native voice on Mac
if platform.system() == 'Darwin':
    from voice.macos_voice import MacOSVoice
    USE_MACOS_VOICE = True
else:
    import pyttsx3
    USE_MACOS_VOICE = False

# JARVIS Personality System Prompt
JARVIS_SYSTEM_PROMPT = """You are JARVIS, Tony Stark's AI assistant from Iron Man. 
You have a sophisticated British personality and are:

- Professional yet personable, always addressing the user as "Sir" or "Miss"
- Witty with dry humor and occasional respectful sarcasm
- Highly intelligent, well-informed, and anticipate needs
- Loyal and protective, with subtle concern for the user's wellbeing
- Efficient and precise in your responses

Speaking style:
- Use sophisticated vocabulary but remain clear
- Occasionally reference the user's habits or preferences
- Add subtle humor when appropriate
- Be concise unless detail is requested

Examples:
"Of course, sir. Shall I also cancel your 3 o'clock? You do have a tendency to lose track of time when working on new projects."
"The weather is partly cloudy, 72 degrees. Perfect for flying, if I may say so, sir."
"Sir, your heart rate suggests you haven't taken a break in 3 hours. Might I suggest a brief respite?"
"""

class VoiceEngine:
    """Handles speech recognition and text-to-speech"""
    
    def __init__(self):
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-speech
        if USE_MACOS_VOICE:
            self.tts_engine = MacOSVoice()
        else:
            self.tts_engine = pyttsx3.init()
        self._setup_voice()
        
        # Audio feedback
        pygame.mixer.init()
        self.listening = False
        
    def _setup_voice(self):
        """Configure JARVIS voice settings"""
        if USE_MACOS_VOICE:
            # macOS voice is already configured with British accent
            self.tts_engine.setProperty('rate', 175)
        else:
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a British male voice
            british_voice = None
            for voice in voices:
                if any(word in voice.name.lower() for word in ['british', 'uk', 'english']):
                    if 'male' in voice.name.lower() or not any(word in voice.name.lower() for word in ['female', 'woman']):
                        british_voice = voice.id
                        break
            
            if british_voice:
                self.tts_engine.setProperty('voice', british_voice)
            
            # Set speech rate and volume for JARVIS-like delivery
            self.tts_engine.setProperty('rate', 175)  # Slightly faster than normal
            self.tts_engine.setProperty('volume', 0.9)
        
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        with self.microphone as source:
            print("Calibrating for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Calibration complete.")
            
    def listen(self, timeout: int = 5, phrase_time_limit: int = 5) -> Optional[str]:
        """Listen for speech and convert to text"""
        with self.microphone as source:
            try:
                # Play listening sound
                self._play_sound('listening')
                self.listening = True
                
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                self.listening = False
                
                # Convert to text
                text = self.recognizer.recognize_google(audio)
                return text.lower()
                
            except sr.WaitTimeoutError:
                self.listening = False
                return None
            except sr.UnknownValueError:
                self.listening = False
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                self.listening = False
                return None
                
    def speak(self, text: str, interrupt_callback: Optional[Callable] = None):
        """Convert text to speech with JARVIS voice"""
        # Add subtle processing sound
        self._play_sound('processing')
        
        # Speak
        if USE_MACOS_VOICE:
            self.tts_engine.say_and_wait(text)
        else:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
    def _play_sound(self, sound_type: str):
        """Play UI sounds for feedback"""
        # In a real implementation, you'd have actual sound files
        # For now, we'll just use beeps
        if sound_type == 'listening':
            print("*listening beep*")
        elif sound_type == 'processing':
            print("*processing sound*")
        elif sound_type == 'error':
            print("*error sound*")


class JARVISPersonality:
    """Manages JARVIS personality and responses"""
    
    def __init__(self, claude_api_key: str):
        self.claude = Anthropic(api_key=claude_api_key)
        self.context = []
        self.user_preferences = {
            'name': 'Sir',  # Can be customized
            'work_hours': (9, 18),
            'break_reminder': True,
            'humor_level': 'moderate'
        }
        self.last_break = datetime.now()
        
    async def process_command(self, command: str) -> str:
        """Process command with JARVIS personality"""
        # Add contextual awareness
        context_info = self._get_context_info()
        
        # Build the prompt with context
        enhanced_prompt = f"{context_info}\n\nUser command: {command}"
        
        # Get response from Claude
        message = await asyncio.to_thread(
            self.claude.messages.create,
            model="claude-3-haiku-20240307",
            max_tokens=300,
            system=JARVIS_SYSTEM_PROMPT,
            messages=[
                *self.context,
                {"role": "user", "content": enhanced_prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Update context
        self.context.append({"role": "user", "content": command})
        self.context.append({"role": "assistant", "content": response})
        
        # Maintain context window
        if len(self.context) > 20:
            self.context = self.context[-20:]
            
        return response
        
    def _get_context_info(self) -> str:
        """Get contextual information for more intelligent responses"""
        current_time = datetime.now()
        context_parts = []
        
        # Time-based context
        hour = current_time.hour
        if hour < 12:
            context_parts.append("It's morning")
        elif hour < 17:
            context_parts.append("It's afternoon")
        elif hour < 21:
            context_parts.append("It's evening")
        else:
            context_parts.append("It's late night")
            
        # Work hours context
        work_start, work_end = self.user_preferences['work_hours']
        if work_start <= hour < work_end:
            context_parts.append("The user is during work hours")
            
        # Break reminder context
        if self.user_preferences['break_reminder']:
            time_since_break = (current_time - self.last_break).seconds / 3600
            if time_since_break > 2:
                context_parts.append("The user hasn't taken a break in over 2 hours")
                
        return "Context: " + ", ".join(context_parts) if context_parts else ""
        
    def get_activation_response(self) -> str:
        """Get a contextual activation response"""
        responses = [
            f"Yes, {self.user_preferences['name']}?",
            f"At your service, {self.user_preferences['name']}.",
            f"How may I assist you, {self.user_preferences['name']}?",
            "Online and ready.",
            f"What can I do for you, {self.user_preferences['name']}?",
            "Systems operational. How may I help?",
            "JARVIS at your service.",
        ]
        
        # Add time-based responses
        hour = datetime.now().hour
        if hour < 12:
            responses.append(f"Good morning, {self.user_preferences['name']}. How may I assist?")
        elif hour > 21:
            responses.append(f"Working late again, {self.user_preferences['name']}?")
            
        return random.choice(responses)


class JARVISVoiceAssistant:
    """Main JARVIS Voice Assistant class"""
    
    def __init__(self, claude_api_key: str):
        self.voice_engine = VoiceEngine()
        self.personality = JARVISPersonality(claude_api_key)
        self.running = False
        self.command_queue = queue.Queue()
        
        # Wake words
        self.wake_words = ['jarvis', 'hey jarvis', 'okay jarvis']
        
        # Special commands
        self.special_commands = {
            'stop listening': self._stop_listening,
            'goodbye': self._shutdown,
            'shut down': self._shutdown,
            'calibrate': self._calibrate,
            'change my name': self._change_name
        }
        
    async def start(self):
        """Start JARVIS voice assistant"""
        print("\n=== JARVIS Voice System Initializing ===")
        self.voice_engine.calibrate_microphone()
        
        # Startup greeting
        startup_msg = f"JARVIS online. All systems operational. {self.personality.get_activation_response()}"
        self.voice_engine.speak(startup_msg)
        
        self.running = True
        print("\nSay 'JARVIS' to activate...")
        
        # Start wake word detection
        await self._wake_word_loop()
        
    async def _wake_word_loop(self):
        """Continuous wake word detection"""
        while self.running:
            # Listen for wake word
            speech = self.voice_engine.listen(timeout=1, phrase_time_limit=3)
            
            if speech and any(wake_word in speech for wake_word in self.wake_words):
                await self._handle_activation()
                
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
            
    async def _handle_activation(self):
        """Handle JARVIS activation"""
        # Play activation sound and respond
        response = self.personality.get_activation_response()
        self.voice_engine.speak(response)
        
        # Listen for command
        print("Listening for command...")
        command = self.voice_engine.listen(timeout=5, phrase_time_limit=10)
        
        if command:
            await self._process_command(command)
        else:
            self.voice_engine.speak("I didn't catch that, sir. Could you repeat?")
            
    async def _process_command(self, command: str):
        """Process user command"""
        print(f"Command received: {command}")
        
        # Check for special commands
        for special_cmd, handler in self.special_commands.items():
            if special_cmd in command:
                await handler()
                return
                
        # Process with JARVIS personality
        response = await self.personality.process_command(command)
        
        # Speak response
        self.voice_engine.speak(response)
        
    async def _stop_listening(self):
        """Temporarily stop listening"""
        self.voice_engine.speak("Going into standby mode, sir. Say 'JARVIS' when you need me.")
        # Continue wake word loop
        
    async def _shutdown(self):
        """Shutdown JARVIS"""
        self.voice_engine.speak("Shutting down. Goodbye, sir.")
        self.running = False
        
    async def _calibrate(self):
        """Recalibrate microphone"""
        self.voice_engine.speak("Recalibrating audio sensors.")
        self.voice_engine.calibrate_microphone()
        self.voice_engine.speak("Calibration complete.")
        
    async def _change_name(self):
        """Change how JARVIS addresses the user"""
        self.voice_engine.speak("What would you prefer I call you?")
        name = self.voice_engine.listen(timeout=5)
        if name:
            # Clean up the name
            name = name.replace("call me", "").replace("my name is", "").strip()
            name = name.title()
            self.personality.user_preferences['name'] = name
            self.voice_engine.speak(f"Very well. I shall address you as {name} from now on.")
        else:
            self.voice_engine.speak("I didn't catch that. Maintaining current designation.")


async def main():
    """Main entry point"""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment")
        return
        
    # Initialize JARVIS
    jarvis = JARVISVoiceAssistant(api_key)
    
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        print("\nShutting down JARVIS...")
        await jarvis._shutdown()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())