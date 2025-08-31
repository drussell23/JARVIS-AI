/**
 * Speech Synthesis Utilities
 * Handles text-to-speech functionality with better error handling and voice selection
 */

class SpeechSynthesisManager {
  constructor() {
    this.selectedVoice = null;
    this.isInitialized = false;
    this.speechQueue = [];
    this.isSpeaking = false;
    this.onSpeakingChange = null;
    
    // Initialize voices
    this.initializeVoices();
  }

  initializeVoices() {
    if (!('speechSynthesis' in window)) {
      console.error('Speech synthesis not supported in this browser');
      return;
    }

    // Load voices
    const loadVoices = () => {
      const voices = speechSynthesis.getVoices();
      console.log(`Found ${voices.length} voices`);
      
      if (voices.length === 0) {
        // Try again after a delay
        setTimeout(loadVoices, 100);
        return;
      }

      // Select the best voice for JARVIS
      this.selectedVoice = this.selectBestVoice(voices);
      this.isInitialized = true;
      console.log('Selected voice:', this.selectedVoice?.name || 'default');
    };

    // Initial attempt
    loadVoices();

    // Also listen for voice changes
    speechSynthesis.onvoiceschanged = loadVoices;
  }

  selectBestVoice(voices) {
    // Preferred voices for JARVIS (British/male voices preferred)
    const preferredVoices = [
      'Daniel', 'Oliver', 'Google UK English Male',
      'Microsoft David - English (United States)', 'Alex',
      'Google US English', 'Microsoft Mark', 'Fred',
      'Microsoft Zira Desktop', 'Google UK English Female'
    ];

    // Try to find a preferred voice
    for (const preferredName of preferredVoices) {
      const voice = voices.find(v =>
        v.name.includes(preferredName) && !v.name.includes('Siri')
      );
      if (voice) {
        return voice;
      }
    }

    // If no preferred voice found, use any English voice
    const englishVoice = voices.find(v => v.lang.startsWith('en'));
    return englishVoice || voices[0];
  }

  speak(text, options = {}) {
    if (!('speechSynthesis' in window)) {
      console.error('Speech synthesis not supported');
      return Promise.reject(new Error('Speech synthesis not supported'));
    }

    return new Promise((resolve, reject) => {
      // Add to queue
      this.speechQueue.push({ text, options, resolve, reject });
      
      // Process queue if not already speaking
      if (!this.isSpeaking) {
        this.processQueue();
      }
    });
  }

  async processQueue() {
    if (this.speechQueue.length === 0) {
      return;
    }

    const { text, options, resolve, reject } = this.speechQueue.shift();
    this.isSpeaking = true;
    
    if (this.onSpeakingChange) {
      this.onSpeakingChange(true);
    }

    try {
      // Cancel any ongoing speech
      speechSynthesis.cancel();
      
      // Wait a bit after cancel to avoid issues
      await new Promise(r => setTimeout(r, 100));

      const utterance = new SpeechSynthesisUtterance(text);

      // Set voice
      if (this.selectedVoice) {
        utterance.voice = this.selectedVoice;
      }

      // Set options
      utterance.rate = options.rate || 1.0;
      utterance.pitch = options.pitch || 0.95;
      utterance.volume = options.volume || 1.0;

      // Set up event handlers
      utterance.onstart = () => {
        console.log('Speech started:', text.substring(0, 50) + '...');
      };

      utterance.onend = () => {
        console.log('Speech ended');
        this.isSpeaking = false;
        
        if (this.onSpeakingChange) {
          this.onSpeakingChange(false);
        }
        
        resolve();
        
        // Process next in queue
        setTimeout(() => this.processQueue(), 100);
      };

      utterance.onerror = (event) => {
        console.error('Speech error:', event.error);
        this.isSpeaking = false;
        
        if (this.onSpeakingChange) {
          this.onSpeakingChange(false);
        }

        // Handle specific errors
        if (event.error === 'canceled' || event.error === 'interrupted') {
          // These are often not real errors
          resolve();
        } else {
          reject(new Error(`Speech synthesis error: ${event.error}`));
        }
        
        // Process next in queue
        setTimeout(() => this.processQueue(), 100);
      };

      // Speak
      speechSynthesis.speak(utterance);
      
    } catch (error) {
      console.error('Error in processQueue:', error);
      this.isSpeaking = false;
      
      if (this.onSpeakingChange) {
        this.onSpeakingChange(false);
      }
      
      reject(error);
      
      // Process next in queue
      setTimeout(() => this.processQueue(), 100);
    }
  }

  stop() {
    speechSynthesis.cancel();
    this.speechQueue = [];
    this.isSpeaking = false;
    
    if (this.onSpeakingChange) {
      this.onSpeakingChange(false);
    }
  }

  pause() {
    speechSynthesis.pause();
  }

  resume() {
    speechSynthesis.resume();
  }

  getVoices() {
    return speechSynthesis.getVoices();
  }

  setVoice(voice) {
    this.selectedVoice = voice;
  }

  setSpeakingChangeCallback(callback) {
    this.onSpeakingChange = callback;
  }
}

// Create singleton instance
const speechManager = new SpeechSynthesisManager();

export default speechManager;