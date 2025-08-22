/**
 * Speech Recognition Manager
 * Handles speech recognition state properly to avoid conflicts
 */

class SpeechRecognitionManager {
  constructor() {
    this.recognition = null;
    this.isListening = false;
    this.isInitialized = false;
    this.retryCount = 0;
    this.maxRetries = 3;
    this.callbacks = {
      onResult: null,
      onError: null,
      onEnd: null,
      onStart: null
    };
    
    this.initializeRecognition();
  }

  initializeRecognition() {
    try {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      
      if (!SpeechRecognition) {
        console.error('Speech recognition not supported');
        return;
      }

      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 1;
      this.recognition.lang = 'en-US';

      // Set up event handlers
      this.recognition.onstart = () => {
        console.log('Speech recognition started');
        this.isListening = true;
        this.retryCount = 0;
        if (this.callbacks.onStart) {
          this.callbacks.onStart();
        }
      };

      this.recognition.onend = () => {
        console.log('Speech recognition ended');
        this.isListening = false;
        
        if (this.callbacks.onEnd) {
          this.callbacks.onEnd();
        }

        // Auto-restart if it ended unexpectedly
        if (this.shouldAutoRestart) {
          setTimeout(() => {
            this.start();
          }, 1000);
        }
      };

      this.recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        this.isListening = false;

        if (this.callbacks.onError) {
          this.callbacks.onError(event);
        }

        // Handle specific errors
        switch (event.error) {
          case 'no-speech':
            // Normal - user stopped speaking
            break;
          case 'audio-capture':
            console.error('Microphone access issue');
            break;
          case 'not-allowed':
            console.error('Microphone permission denied');
            break;
          case 'network':
            console.error('Network error');
            if (this.retryCount < this.maxRetries) {
              this.retryCount++;
              setTimeout(() => this.start(), 2000);
            }
            break;
          default:
            // Try to restart for other errors
            if (this.retryCount < this.maxRetries && this.shouldAutoRestart) {
              this.retryCount++;
              setTimeout(() => this.start(), 1000);
            }
        }
      };

      this.recognition.onresult = (event) => {
        if (this.callbacks.onResult) {
          this.callbacks.onResult(event);
        }
      };

      this.isInitialized = true;
      console.log('Speech recognition initialized');

    } catch (error) {
      console.error('Failed to initialize speech recognition:', error);
    }
  }

  start() {
    if (!this.isInitialized) {
      console.error('Speech recognition not initialized');
      return false;
    }

    if (this.isListening) {
      console.log('Already listening, restarting...');
      this.stop();
      setTimeout(() => this.start(), 100);
      return false;
    }

    try {
      this.recognition.start();
      this.shouldAutoRestart = true;
      return true;
    } catch (error) {
      console.error('Failed to start recognition:', error);
      this.isListening = false;
      
      // If already started error, stop and retry
      if (error.message && error.message.includes('already started')) {
        this.stop();
        setTimeout(() => this.start(), 500);
      }
      
      return false;
    }
  }

  stop() {
    this.shouldAutoRestart = false;
    
    if (!this.isInitialized || !this.isListening) {
      return;
    }

    try {
      this.recognition.stop();
      this.isListening = false;
    } catch (error) {
      console.error('Error stopping recognition:', error);
    }
  }

  abort() {
    this.shouldAutoRestart = false;
    
    if (!this.isInitialized) {
      return;
    }

    try {
      this.recognition.abort();
      this.isListening = false;
    } catch (error) {
      console.error('Error aborting recognition:', error);
    }
  }

  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  isActive() {
    return this.isListening;
  }

  reset() {
    this.abort();
    this.retryCount = 0;
    setTimeout(() => {
      this.initializeRecognition();
    }, 500);
  }
}

export default SpeechRecognitionManager;