"""
JARVIS Voice System - Enhanced with professional-grade accuracy
Integrates with Claude API for intelligent voice command processing
"""

import asyncio
import speech_recognition as sr
import pygame
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple, Union
import json
import random
from datetime import datetime
import threading
import queue
import os
import sys
import platform
from anthropic import Anthropic
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML trainer
try:
    from voice.voice_ml_trainer import VoiceMLTrainer, VoicePattern
    ML_TRAINING_AVAILABLE = True
except ImportError:
    ML_TRAINING_AVAILABLE = False
    logger.warning("ML training not available. Install required libraries for adaptive learning.")

# Import weather service
try:
    from services.weather_service import WeatherService
    WEATHER_SERVICE_AVAILABLE = True
except ImportError:
    WEATHER_SERVICE_AVAILABLE = False
    logger.warning("Weather service not available")

# Use macOS native voice on Mac
if platform.system() == 'Darwin':
    from voice.macos_voice import MacOSVoice
    USE_MACOS_VOICE = True
else:
    import pyttsx3
    USE_MACOS_VOICE = False

# JARVIS Personality System Prompt
JARVIS_SYSTEM_PROMPT = """You are JARVIS, Tony Stark's AI assistant. Be concise and helpful.

CRITICAL RULES:
1. Keep responses SHORT (1-2 sentences max unless explaining something complex)
2. Be direct and to the point - no flowery language
3. Don't describe the weather or time unless specifically asked
4. Don't add context about the day/afternoon/evening unless relevant
5. Address the user as "Sir" but don't overuse it
6. Don't use sound effects like "chimes softly" or stage directions
7. No long greetings or farewells

Examples of GOOD responses:
- "Yes, sir?" (when activated)
- "The weather is 24 degrees, overcast clouds."
- "Certainly. The calculation equals 42."
- "I'll check that for you."
- "Done. Anything else?"

Examples of BAD responses (avoid these):
- "Good afternoon, sir. The weather is lovely today..."
- "A gentle chime sounds as I activate..."
- Multiple paragraphs of context
- Asking multiple questions at once
Busy period: "Ready when you are, sir. What's the priority?"

Remember: You're not just an AI following a script - you're JARVIS, a sophisticated assistant with genuine personality. Each interaction should feel fresh and authentic."""

# Voice-specific system prompt for Anthropic
VOICE_OPTIMIZATION_PROMPT = """You are processing voice commands for JARVIS. Voice commands differ from typed text:

Context: This was spoken aloud and may contain:
- Recognition errors
- Informal speech patterns
- Missing punctuation
- Homophones (e.g., "to/too/two")

Previous context: {context}
Voice command: "{command}"
Confidence: {confidence}
Detected intent: {intent}

If confidence is low or the command seems unclear:
1. Provide your best interpretation
2. Ask a clarifying question if needed

Natural response guidelines:
- Vary your greetings - don't use the same pattern
- Keep responses conversational length (what feels natural to say aloud)
- Add personality through word choice and observations
- Reference context when relevant (time, previous conversations, etc.)

Examples of natural variations:
Instead of "How may I assist you?" try:
- "What can I do for you?"
- "What's on your mind?"
- "Need something?"
- "I'm listening."

Respond as JARVIS would - sophisticated, natural, and genuinely helpful."""

class VoiceConfidence(Enum):
    """Confidence levels for voice detection"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class VoiceCommand:
    """Structured voice command data"""
    raw_text: str
    confidence: float
    intent: str
    needs_clarification: bool = False
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EnhancedVoiceEngine:
    """Enhanced speech recognition with confidence scoring, noise reduction, and ML training"""
    
    def __init__(self, ml_trainer: Optional['VoiceMLTrainer'] = None):
        # Speech recognition with multiple engines
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Configure recognizer for better accuracy
        self.recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8  # Seconds of silence before phrase is considered complete
        
        # Text-to-speech
        if USE_MACOS_VOICE:
            self.tts_engine: Union[MacOSVoice, Any] = MacOSVoice()
        else:
            self.tts_engine = pyttsx3.init()
        self._setup_voice()
        
        # Audio feedback
        pygame.mixer.init()
        self.listening = False
        
        # Noise profile for reduction
        self.noise_profile = None
        
        # ML trainer for adaptive learning
        self.ml_trainer = ml_trainer
        self.last_audio_data = None  # Store for ML training
        
        # Intent patterns for better recognition
        self.intent_patterns = {
            "question": ["what", "when", "where", "who", "why", "how", "is", "are", "can", "could"],
            "action": ["open", "close", "start", "stop", "play", "pause", "set", "turn", "activate", "launch"],
            "information": ["tell", "show", "find", "search", "look", "get", "fetch", "explain"],
            "system": ["system", "status", "diagnostic", "check", "monitor", "analyze", "report"],
            "conversation": ["chat", "talk", "discuss", "explain", "describe", "hello", "hi"]
        }
        
    def _setup_voice(self):
        """Configure JARVIS voice settings"""
        if USE_MACOS_VOICE:
            # macOS voice is already configured with British accent
            self.tts_engine.setProperty('rate', 175)
        else:
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a British male voice
            british_voice = None
            if voices:
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
    
    def calibrate_microphone(self, duration: int = 3):
        """Enhanced calibration with noise profiling"""
        with self.microphone as source:
            print("🎤 Calibrating for ambient noise... Please remain quiet.")
            
            # Adjust for ambient noise with longer duration
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            
            # Record noise sample for profile
            try:
                print("📊 Creating noise profile...")
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=2)
                # Store noise profile for future noise reduction
                self.noise_profile = audio.get_raw_data()
                print("✅ Calibration complete. Noise profile created.")
            except:
                print("✅ Calibration complete.")
    
    def listen_with_confidence(self, timeout: int = 5, phrase_time_limit: int = 5) -> Tuple[Optional[str], float]:
        """Listen for speech and return text with confidence score"""
        with self.microphone as source:
            try:
                # Play listening sound
                self._play_sound('listening')
                self.listening = True
                
                # Clear the buffer
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                self.listening = False
                
                # Store audio data for ML training
                self.last_audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                
                # Try multiple recognition methods for better accuracy
                recognition_results = []
                confidence = 0.0
                
                # Google Speech Recognition with alternatives
                try:
                    google_result = self.recognizer.recognize_google(
                        audio, 
                        show_all=True,
                        language="en-US"
                    )
                    
                    if google_result and 'alternative' in google_result:
                        for i, alternative in enumerate(google_result['alternative']):
                            text = alternative.get('transcript', '').lower()
                            # Google provides confidence only for the first alternative
                            conf = alternative.get('confidence', 0.8 - (i * 0.1))
                            recognition_results.append((text, conf))
                except Exception as e:
                    logger.debug(f"Google recognition failed: {e}")
                
                # If we have results, return the best one
                if recognition_results:
                    # Sort by confidence
                    recognition_results.sort(key=lambda x: x[1], reverse=True)
                    best_text, best_confidence = recognition_results[0]
                    
                    # Apply confidence adjustments based on audio quality
                    adjusted_confidence = self._adjust_confidence(audio, best_confidence)
                    
                    # Check ML predictions if available
                    if self.ml_trainer and best_text:
                        predicted_correction = self.ml_trainer.predict_correction(
                            best_text, adjusted_confidence,
                            self.ml_trainer.extract_audio_features(self.last_audio_data)
                        )
                        if predicted_correction:
                            logger.info(f"ML prediction: '{best_text}' -> '{predicted_correction}'")
                            # You might want to return the prediction with higher confidence
                            # For now, we'll just log it
                    
                    return best_text, adjusted_confidence
                
                return None, 0.0
                
            except sr.WaitTimeoutError:
                self.listening = False
                return None, 0.0
            except sr.UnknownValueError:
                self.listening = False
                return None, 0.0
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                self.listening = False
                return None, 0.0
    
    def _adjust_confidence(self, audio: sr.AudioData, base_confidence: float) -> float:
        """Adjust confidence based on audio quality metrics"""
        try:
            # Convert audio to numpy array
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            
            # Calculate audio quality metrics
            energy = np.sqrt(np.mean(raw_data**2))  # RMS energy
            
            # Very quiet audio is less reliable
            if energy < 100:
                base_confidence *= 0.7
            elif energy > 5000:  # Very loud (possible distortion)
                base_confidence *= 0.9
            
            # Check for clipping
            if np.any(np.abs(raw_data) > 32000):  # Near max int16 value
                base_confidence *= 0.8
            
            return min(base_confidence, 1.0)
        except:
            return base_confidence
    
    def detect_intent(self, text: str) -> str:
        """Detect the intent of the command"""
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # Check against intent patterns
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in text_lower.split() for keyword in keywords):
                return intent
        
        return "conversation"  # Default intent
    
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
            print("🎤 *listening*")
        elif sound_type == 'processing':
            print("⚡ *processing*")
        elif sound_type == 'error':
            print("❌ *error*")
        elif sound_type == 'success':
            print("✅ *success*")


class EnhancedJARVISPersonality:
    """Enhanced JARVIS personality with voice-specific intelligence and ML integration"""
    
    def __init__(self, claude_api_key: str, ml_trainer: Optional['VoiceMLTrainer'] = None):
        self.claude = Anthropic(api_key=claude_api_key)
        self.context = []
        self.voice_context = []  # Separate context for voice commands
        self.user_preferences = {
            'name': 'Sir',  # Can be customized
            'work_hours': (9, 18),
            'break_reminder': True,
            'humor_level': 'moderate'
        }
        self.last_break = datetime.now()
        
        # Voice command history for learning patterns
        self.command_history = []
        
        # ML trainer for adaptive learning
        self.ml_trainer = ml_trainer
        
        # Initialize weather service
        if WEATHER_SERVICE_AVAILABLE:
            self.weather_service = WeatherService()
        else:
            self.weather_service = None
        
    async def process_voice_command(self, command: VoiceCommand) -> str:
        """Process voice command with enhanced intelligence"""
        # Add to command history
        self.command_history.append(command)
        if len(self.command_history) > 50:
            self.command_history = self.command_history[-50:]
        
        # Get context
        context_info = self._get_context_info()
        recent_context = self._get_recent_voice_context()
        
        # Log for debugging
        logger.info(f"Processing command: '{command.raw_text}' (confidence: {command.confidence})")
        
        # Determine if we need to use voice optimization
        if command.confidence < VoiceConfidence.HIGH.value or command.needs_clarification:
            return await self._optimize_voice_command(command, context_info, recent_context)
        else:
            return await self._process_clear_command(command.raw_text, context_info)
    
    async def _optimize_voice_command(self, command: VoiceCommand, context_info: str, recent_context: str) -> str:
        """Use Anthropic to interpret unclear voice commands"""
        # Build voice-specific prompt
        prompt = VOICE_OPTIMIZATION_PROMPT.format(
            context=recent_context,
            command=command.raw_text,
            confidence=f"{command.confidence:.2f}",
            intent=command.intent
        )
        
        # Add any specific context
        if context_info:
            prompt = f"{context_info}\n\n{prompt}"
        
        # Get interpretation from Claude
        message = await asyncio.to_thread(
            self.claude.messages.create,
            model="claude-3-haiku-20240307",  # Fast model for voice processing
            max_tokens=200,
            temperature=0.3,  # Lower temperature for accuracy
            system=JARVIS_SYSTEM_PROMPT,
            messages=[
                *self.voice_context[-5:],  # Include recent voice context
                {"role": "user", "content": prompt}
            ]
        )
        
        response = message.content[0].text
        
        # Update voice context
        self.voice_context.append({"role": "user", "content": command.raw_text})
        self.voice_context.append({"role": "assistant", "content": response})
        
        # Maintain voice context window
        if len(self.voice_context) > 20:
            self.voice_context = self.voice_context[-20:]
        
        return response
    
    async def _process_clear_command(self, command: str, context_info: str) -> str:
        """Process clear commands normally"""
        # Check if this is a weather request FIRST - before adding context
        if await self._is_weather_request(command):
            return await self._handle_weather_request(command)
        
        # For non-weather requests, use simple prompt without context
        enhanced_prompt = f"User command: {command}"
        
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
    
    def _get_recent_voice_context(self) -> str:
        """Get recent voice command context"""
        if not self.command_history:
            return "No recent voice commands"
        
        recent = self.command_history[-3:]
        context_parts = []
        
        for cmd in recent:
            time_ago = (datetime.now() - cmd.timestamp).seconds
            if time_ago < 60:
                context_parts.append(f"{time_ago}s ago: '{cmd.raw_text}'")
            elif time_ago < 3600:
                context_parts.append(f"{time_ago//60}m ago: '{cmd.raw_text}'")
        
        return "Recent commands: " + "; ".join(context_parts) if context_parts else "No recent commands"
    
    def _get_context_info(self) -> str:
        """Get contextual information for more intelligent responses"""
        current_time = datetime.now()
        context_parts = []
        
        # Time and day context
        hour = current_time.hour
        day_name = current_time.strftime("%A")
        
        # Add natural time context
        if hour < 6:
            context_parts.append("Very early morning hours")
        elif hour < 9:
            context_parts.append("Early morning")
        elif hour < 12:
            context_parts.append("Morning")
        elif hour < 14:
            context_parts.append("Midday")
        elif hour < 17:
            context_parts.append("Afternoon")
        elif hour < 20:
            context_parts.append("Evening")
        elif hour < 23:
            context_parts.append("Late evening")
        else:
            context_parts.append("Late night")
        
        # Weekend context
        if day_name in ['Saturday', 'Sunday']:
            context_parts.append("Weekend")
        
        # Work hours context - more natural
        work_start, work_end = self.user_preferences['work_hours']
        if day_name not in ['Saturday', 'Sunday'] and work_start <= hour < work_end:
            context_parts.append("During typical work hours")
        elif hour >= work_end and day_name not in ['Saturday', 'Sunday']:
            context_parts.append("After work hours")
            
        # Break reminder context - more natural
        if self.user_preferences['break_reminder']:
            time_since_break = (current_time - self.last_break).seconds / 3600
            if time_since_break > 3:
                context_parts.append("User has been active for extended period")
        
        # Recent interaction patterns
        if self.command_history:
            # Check if user just started interacting
            if len(self.command_history) == 1:
                context_parts.append("First interaction of this session")
            # Check for rapid commands
            elif len(self.command_history) > 2:
                recent_times = [cmd.timestamp for cmd in self.command_history[-3:]]
                time_diffs = [(recent_times[i+1] - recent_times[i]).seconds for i in range(len(recent_times)-1)]
                if all(diff < 30 for diff in time_diffs):
                    context_parts.append("User is actively engaged")
                
        return "Natural context: " + "; ".join(context_parts) if context_parts else ""
    
    def get_activation_response(self, confidence: float = 1.0) -> str:
        """Get a contextual activation response based on confidence"""
        if confidence < VoiceConfidence.MEDIUM.value:
            # Low confidence responses
            return random.choice([
                f"I think I heard you, {self.user_preferences['name']}. How may I assist?",
                f"Apologies if I misheard, {self.user_preferences['name']}. What can I do for you?",
                "Pardon me, sir. Could you repeat that?"
            ])
        
        # Keep responses short and simple
        responses = [
            f"Yes, {self.user_preferences['name']}?",
            "Yes?",
            "Sir?",
            "Listening.",
            "Go ahead.",
            "I'm here.",
        ]
            
        return random.choice(responses)
    
    async def _is_weather_request(self, command: str) -> bool:
        """Check if command is asking about weather - FAST"""
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 
                          'cold', 'hot', 'warm', 'degrees', 'celsius', 'fahrenheit',
                          'outside', 'today']
        command_lower = command.lower()
        # Quick check for common patterns
        if 'weather' in command_lower or 'temperature' in command_lower:
            logger.info(f"Weather request detected: '{command}'")
            return True
        is_weather = any(keyword in command_lower for keyword in weather_keywords)
        if is_weather:
            logger.info(f"Weather request detected via keywords: '{command}'")
        return is_weather
    
    async def _handle_weather_request(self, command: str) -> str:
        """Handle weather requests with real data"""
        logger.info(f"Handling weather request. Weather service available: {self.weather_service is not None}")
        if not self.weather_service:
            # Fallback to Claude if weather service not available
            enhanced_prompt = f"User is asking about weather: {command}"
            message = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=150,
                system="You are JARVIS. Give a brief, direct weather response. Be concise.",
                messages=[
                    {"role": "user", "content": enhanced_prompt}
                ]
            )
            return message.content[0].text
        
        try:
            # Determine if asking about specific location or current location
            command_lower = command.lower()
            
            # Check for specific cities
            city = None
            common_cities = ['toronto', 'new york', 'london', 'paris', 'tokyo', 'los angeles', 
                           'chicago', 'vancouver', 'montreal', 'calgary', 'ottawa']
            
            for c in common_cities:
                if c in command_lower:
                    city = c.title()
                    break
            
            # Get weather data
            if city:
                weather_data = await self.weather_service.get_weather_by_city(city)
            else:
                # Use current location
                weather_data = await self.weather_service.get_current_weather()
            
            # Format response in JARVIS style
            location = weather_data.get('location', 'your location')
            temp = weather_data.get('temperature', 0)
            feels_like = weather_data.get('feels_like', temp)
            description = weather_data.get('description', 'unknown conditions')
            wind = weather_data.get('wind_speed', 0)
            
            # Build JARVIS-style response
            response = f"Currently in {location}, we have {description} "
            response += f"with a temperature of {temp} degrees Celsius"
            
            if abs(feels_like - temp) > 2:
                response += f", though it feels like {feels_like}"
            
            response += f". Wind speed is {wind} kilometers per hour. "
            
            # Add personalized suggestions based on conditions
            hour = datetime.now().hour
            if temp > 25:
                response += "Quite warm today, sir. Perhaps consider lighter attire."
            elif temp < 10:
                response += "Rather chilly, sir. I'd recommend a jacket."
            elif 'rain' in description.lower():
                response += "Don't forget an umbrella if you're heading out, sir."
            elif 'clear' in description.lower() and temp > 18 and hour < 18:
                response += "Beautiful weather for any outdoor activities you might have planned."
            
            # Update context with actual weather info
            self.context.append({"role": "user", "content": command})
            self.context.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            # Fallback to Claude
            return await self._process_clear_command(command, self._get_context_info())


class EnhancedJARVISVoiceAssistant:
    """Enhanced JARVIS Voice Assistant with professional-grade accuracy and ML training"""
    
    def __init__(self, claude_api_key: str, enable_ml_training: bool = True):
        # Initialize ML trainer if enabled
        self.ml_trainer = None
        if enable_ml_training and ML_TRAINING_AVAILABLE:
            try:
                self.ml_trainer = VoiceMLTrainer(claude_api_key)
                logger.info("ML training system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ML trainer: {e}")
                self.ml_trainer = None
        
        # Initialize components with ML trainer - OPTIMIZED
        # Start personality initialization early (it pre-loads weather)
        self.personality = EnhancedJARVISPersonality(claude_api_key, ml_trainer=self.ml_trainer)
        
        # Initialize voice engine in parallel
        self.voice_engine = EnhancedVoiceEngine(ml_trainer=self.ml_trainer)
        self.running = False
        self.command_queue = queue.Queue()
        
        # Enhanced wake words with variations
        self.wake_words = {
            'primary': ['jarvis', 'hey jarvis', 'okay jarvis'],
            'variations': ['jar vis', 'hey jar vis', 'jarv'],  # Handle speech breaks
            'urgent': ['jarvis emergency', 'jarvis urgent']
        }
        
        # Confidence thresholds
        self.wake_word_threshold = 0.6
        self.command_threshold = 0.7
        
        # Special commands
        self.special_commands = {
            'stop listening': self._stop_listening,
            'goodbye': self._shutdown,
            'shut down': self._shutdown,
            'calibrate': self._calibrate,
            'change my name': self._change_name,
            'improve accuracy': self._improve_accuracy,
            'show my voice stats': self._show_voice_stats,
            'export my voice model': self._export_voice_model,
            'personalized tips': self._get_personalized_tips
        }
        
    def _check_wake_word(self, text: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """Enhanced wake word detection with fuzzy matching"""
        if not text:
            return False, None
        
        text_lower = text.lower()
        
        # Check urgent wake words first
        for wake_word in self.wake_words['urgent']:
            if wake_word in text_lower:
                return True, 'urgent'
        
        # Check primary wake words
        for wake_word in self.wake_words['primary']:
            if wake_word in text_lower:
                # Boost confidence if wake word is at the beginning
                if text_lower.startswith(wake_word):
                    confidence += 0.1
                if confidence >= self.wake_word_threshold:
                    return True, 'primary'
        
        # Check variations (with lower threshold)
        for variation in self.wake_words['variations']:
            if variation in text_lower and confidence >= (self.wake_word_threshold - 0.1):
                return True, 'variation'
        
        return False, None
    
    async def start(self):
        """Start enhanced JARVIS voice assistant"""
        print("\n=== JARVIS Enhanced Voice System Initializing ===")
        print("🚀 Loading professional-grade voice processing...")
        
        # Enhanced calibration
        self.voice_engine.calibrate_microphone(duration=3)
        
        # Startup greeting
        startup_msg = "JARVIS enhanced voice system online. All systems operational."
        self.voice_engine.speak(startup_msg)
        
        self.running = True
        print("\n🎤 Say 'JARVIS' to activate...")
        print("💡 Tip: For better accuracy, speak clearly and wait for the listening indicator")
        
        # Start wake word detection
        await self._wake_word_loop()
    
    async def _wake_word_loop(self):
        """Enhanced wake word detection loop"""
        consecutive_failures = 0
        
        while self.running:
            # Listen for wake word with confidence
            text, confidence = self.voice_engine.listen_with_confidence(timeout=1, phrase_time_limit=3)
            
            if text:
                # Check for wake word
                detected, wake_type = self._check_wake_word(text, confidence)
                
                if detected:
                    logger.info(f"Wake word detected: '{text}' (confidence: {confidence:.2f}, type: {wake_type})")
                    consecutive_failures = 0
                    await self._handle_activation(confidence, wake_type, text)
                else:
                    # Log near-misses for debugging
                    if any(word in text.lower() for sublist in self.wake_words.values() for word in (sublist if isinstance(sublist, list) else [sublist])):
                        logger.debug(f"Near-miss wake word: '{text}' (confidence: {confidence:.2f})")
            
            # Recalibrate if we're getting too many failures
            consecutive_failures += 1
            if consecutive_failures > 30:  # About 30 seconds of failures
                logger.info("Recalibrating due to consecutive failures")
                self.voice_engine.calibrate_microphone(duration=1)
                consecutive_failures = 0
            
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
    
    async def _handle_activation(self, wake_confidence: float, wake_type: str, full_text: str = None):
        """Enhanced activation handling"""
        # Check if command was included with wake word
        command_text = None
        command_confidence = wake_confidence
        
        if full_text:
            # Extract command after wake word
            text_lower = full_text.lower()
            for wake_list in self.wake_words.values():
                for wake_word in (wake_list if isinstance(wake_list, list) else [wake_list]):
                    if wake_word in text_lower:
                        # Find the wake word position and extract everything after it
                        wake_pos = text_lower.find(wake_word)
                        if wake_pos != -1:
                            potential_command = full_text[wake_pos + len(wake_word):].strip()
                            if potential_command:
                                command_text = potential_command
                                print(f"📝 Command detected with wake word: '{command_text}'")
                                break
                if command_text:
                    break
        
        if not command_text:
            # No command with wake word, so respond and listen
            if wake_type == 'urgent':
                self.voice_engine.speak("Emergency protocol activated. What's the situation?")
            else:
                response = self.personality.get_activation_response(wake_confidence)
                self.voice_engine.speak(response)
            
            # Listen for command with confidence scoring
            print("🎤 Listening for command...")
            command_text, command_confidence = self.voice_engine.listen_with_confidence(
                timeout=5, 
                phrase_time_limit=10
            )
        
        if command_text:
            # Detect intent
            intent = self.voice_engine.detect_intent(command_text)
            
            # Create structured command
            command = VoiceCommand(
                raw_text=command_text,
                confidence=command_confidence,
                intent=intent,
                needs_clarification=command_confidence < self.command_threshold
            )
            
            await self._process_command(command)
        else:
            # Different responses based on context
            if wake_confidence < 0.7:
                self.voice_engine.speak("I'm having trouble hearing you clearly, sir. Could you speak up?")
            else:
                self.voice_engine.speak("I didn't catch that, sir. Could you repeat?")
    
    async def _process_command(self, command: VoiceCommand, audio_data: Optional[np.ndarray] = None):
        """Process enhanced voice command with ML training"""
        logger.info(f"Command: '{command.raw_text}' (confidence: {command.confidence:.2f}, intent: {command.intent})")
        
        # Store original command for ML training
        original_command = command.raw_text
        success = True
        corrected_text = None
        
        # Check for special commands first
        for special_cmd, handler in self.special_commands.items():
            if special_cmd in command.raw_text.lower():
                await handler()
                # Train ML on successful special command
                if self.ml_trainer and audio_data is not None:
                    await self.ml_trainer.learn_from_interaction(
                        recognized_text=original_command,
                        confidence=command.confidence,
                        audio_data=audio_data,
                        corrected_text=None,
                        success=True,
                        context="special_command"
                    )
                return
        
        # Process with enhanced personality
        response = await self.personality.process_voice_command(command)
        
        # Check if response indicates a clarification was needed
        if "?" in response and command.needs_clarification:
            # Wait for clarification
            self.voice_engine.speak(response)
            
            # Listen for clarification
            clarification_text, clarification_confidence = self.voice_engine.listen_with_confidence(
                timeout=5, phrase_time_limit=10
            )
            
            if clarification_text:
                corrected_text = clarification_text
                # Re-process with clarification
                clarified_command = VoiceCommand(
                    raw_text=clarification_text,
                    confidence=clarification_confidence,
                    intent=self.voice_engine.detect_intent(clarification_text),
                    needs_clarification=False
                )
                response = await self.personality.process_voice_command(clarified_command)
        
        # Speak final response
        self.voice_engine.speak(response)
        
        # Train ML system with the interaction
        if self.ml_trainer:
            # Get audio data from voice engine if not provided
            if audio_data is None and hasattr(self.voice_engine, 'last_audio_data'):
                audio_data = self.voice_engine.last_audio_data
            
            if audio_data is not None:
                await self.ml_trainer.learn_from_interaction(
                    recognized_text=original_command,
                    confidence=command.confidence,
                    audio_data=audio_data,
                    corrected_text=corrected_text,
                    success=success,
                    context=f"intent:{command.intent}"
                )
                
                # Log ML insights periodically
                user_profile = self.ml_trainer.user_profiles.get(self.ml_trainer.current_user, {})
                voice_patterns = user_profile.get('voice_patterns', []) if isinstance(user_profile, dict) else []
                if len(voice_patterns) % 20 == 0:
                    insights = self.ml_trainer.get_user_insights()
                    logger.info(f"ML Insights - Accuracy: {insights['recent_accuracy']:.2%}, Total interactions: {insights['total_interactions']}")
    
    async def _improve_accuracy(self):
        """Guide user through accuracy improvement"""
        self.voice_engine.speak("Let's improve my accuracy. I'll guide you through a quick calibration.")
        await asyncio.sleep(1)
        
        # Recalibrate with user guidance
        self.voice_engine.speak("First, please remain quiet while I calibrate for background noise.")
        self.voice_engine.calibrate_microphone(duration=4)
        
        self.voice_engine.speak("Excellent. Now, please say 'Hey JARVIS' three times, pausing between each.")
        
        # Collect samples
        samples = []
        for i in range(3):
            self.voice_engine.speak(f"Sample {i+1} of 3. Please say 'Hey JARVIS'.")
            text, confidence = self.voice_engine.listen_with_confidence(timeout=5)
            if text:
                samples.append((text, confidence))
                self.voice_engine.speak("Got it.")
            else:
                self.voice_engine.speak("I didn't catch that. Let's try again.")
                i -= 1
        
        # Analyze samples
        if samples:
            avg_confidence = np.mean([s[1] for s in samples])
            if avg_confidence > 0.8:
                self.voice_engine.speak(f"Excellent! Your voice is coming through clearly with {avg_confidence*100:.0f}% confidence.")
            elif avg_confidence > 0.6:
                self.voice_engine.speak(f"Good. I'm detecting your voice with {avg_confidence*100:.0f}% confidence. Try speaking a bit louder or clearer.")
            else:
                self.voice_engine.speak(f"I'm having some difficulty. Only {avg_confidence*100:.0f}% confidence. You may want to check your microphone or reduce background noise.")
        
        self.voice_engine.speak("Calibration complete. My accuracy should be improved.")
    
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
        self.voice_engine.calibrate_microphone(duration=3)
        self.voice_engine.speak("Calibration complete.")
        
    async def _change_name(self):
        """Change how JARVIS addresses the user"""
        self.voice_engine.speak("What would you prefer I call you?")
        name_text, confidence = self.voice_engine.listen_with_confidence(timeout=5)
        
        if name_text and confidence > 0.5:
            # Clean up the name using AI if confidence is low
            if confidence < 0.8:
                command = VoiceCommand(
                    raw_text=name_text,
                    confidence=confidence,
                    intent="name_change",
                    needs_clarification=True
                )
                # Process through AI for clarification
                processed = await self.personality.process_voice_command(command)
                # Extract name from response (simplified)
                name = name_text.replace("call me", "").replace("my name is", "").strip().title()
            else:
                # High confidence - process directly
                name = name_text.replace("call me", "").replace("my name is", "").strip().title()
            
            self.personality.user_preferences['name'] = name
            self.voice_engine.speak(f"Very well. I shall address you as {name} from now on.")
        else:
            self.voice_engine.speak("I didn't catch that. Maintaining current designation.")
    
    async def _show_voice_stats(self):
        """Show user's voice interaction statistics"""
        if not self.ml_trainer:
            self.voice_engine.speak("Voice statistics are not available. ML training is not enabled.")
            return
        
        insights = self.ml_trainer.get_user_insights()
        
        if 'error' in insights:
            self.voice_engine.speak("No voice statistics available yet. Keep using voice commands to build your profile.")
            return
        
        # Prepare summary
        stats_summary = f"""Your voice interaction statistics:
        
Total interactions: {insights['total_interactions']}
Recent accuracy: {insights['recent_accuracy']:.0%}
Most used command: {insights['top_commands'][0][0] if insights['top_commands'] else 'None'}

You've used voice commands {insights['total_interactions']} times with {insights['recent_accuracy']:.0%} accuracy recently."""
        
        self.voice_engine.speak(stats_summary)
        
        # Log detailed stats
        logger.info(f"Voice Stats: {json.dumps(insights, indent=2, default=str)}")
    
    async def _export_voice_model(self):
        """Export user's voice model"""
        if not self.ml_trainer:
            self.voice_engine.speak("Voice model export is not available. ML training is not enabled.")
            return
        
        try:
            export_path = self.ml_trainer.export_user_model()
            if export_path:
                self.voice_engine.speak(f"Your voice model has been exported successfully. Check the models directory.")
                logger.info(f"Voice model exported to: {export_path}")
            else:
                self.voice_engine.speak("Unable to export voice model. No data available.")
        except Exception as e:
            logger.error(f"Error exporting voice model: {e}")
            self.voice_engine.speak("There was an error exporting your voice model.")
    
    async def _get_personalized_tips(self):
        """Get personalized tips based on ML analysis"""
        if not self.ml_trainer:
            self.voice_engine.speak("Personalized tips are not available. ML training is not enabled.")
            return
        
        self.voice_engine.speak("Analyzing your voice patterns to generate personalized tips...")
        
        try:
            tips = await self.ml_trainer.generate_personalized_tips()
            self.voice_engine.speak(tips)
        except Exception as e:
            logger.error(f"Error generating tips: {e}")
            self.voice_engine.speak("I encountered an error while generating tips. Please try again later.")


async def main():
    """Main entry point"""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment")
        return
        
    # Initialize enhanced JARVIS
    jarvis = EnhancedJARVISVoiceAssistant(api_key)
    
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        print("\nShutting down JARVIS...")
        await jarvis._shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())