/**
 * Hybrid STT Client - WebSocket-based audio transcription
 *
 * Replaces browser SpeechRecognition with backend hybrid STT system:
 * - Sends audio chunks to backend via WebSocket
 * - Receives transcriptions with confidence scores
 * - Handles retry logic for low-confidence results
 * - Displays STT engine info (Wav2Vec, Vosk, Whisper)
 * - Learning-enabled (every transcription recorded to database)
 */

class HybridSTTClient {
  constructor(websocket, options = {}) {
    this.ws = websocket;
    this.options = {
      strategy: options.strategy || 'balanced', // speed, accuracy, balanced, cost, adaptive
      speakerName: options.speakerName || null, // Let backend auto-detect speaker
      confidenceThreshold: options.confidenceThreshold || 0.6,
      retryOnLowConfidence: options.retryOnLowConfidence !== false,
      continuous: options.continuous !== false,
      interimResults: options.interimResults !== false,
      maxRetries: options.maxRetries || 2,
      ...options
    };

    // MediaRecorder for audio capture
    this.mediaRecorder = null;
    this.audioStream = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isPaused = false;

    // Event handlers (compatible with SpeechRecognition API)
    this.onstart = null;
    this.onend = null;
    this.onresult = null;
    this.onerror = null;
    this.onspeechstart = null;
    this.onspeechend = null;
    this.onaudiostart = null;
    this.onaudioend = null;

    // Retry tracking
    this.retryCount = 0;
    this.lastTranscript = null;

    // STT metrics
    this.stats = {
      totalRequests: 0,
      successfulTranscriptions: 0,
      lowConfidenceResults: 0,
      retries: 0,
      averageConfidence: 0,
      averageLatency: 0,
      enginesUsed: {}
    };

    console.log('🎤 [HybridSTT] Client initialized', { options: this.options });
  }

  /**
   * Start audio capture and transcription
   */
  async start() {
    if (this.isRecording) {
      console.warn('🎤 [HybridSTT] Already recording');
      return;
    }

    try {
      // Get microphone access
      this.audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,  // Mono audio
          sampleRate: 16000, // 16kHz (optimal for STT)
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Create MediaRecorder for audio capture
      // Try different MIME types for compatibility
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
        'audio/wav'
      ];

      let selectedMimeType = null;
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          selectedMimeType = mimeType;
          break;
        }
      }

      if (!selectedMimeType) {
        throw new Error('No supported audio MIME type found');
      }

      this.mediaRecorder = new MediaRecorder(this.audioStream, {
        mimeType: selectedMimeType
      });

      // Collect audio chunks
      this.audioChunks = [];
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
          console.log(`🎤 [HybridSTT] Audio chunk captured: ${event.data.size} bytes`);
        }
      };

      // Handle recording stop
      this.mediaRecorder.onstop = async () => {
        console.log('🎤 [HybridSTT] Recording stopped, processing audio...');

        if (this.audioChunks.length === 0) {
          console.warn('🎤 [HybridSTT] No audio data captured');
          this._triggerEvent('error', { error: 'no-audio' });
          return;
        }

        // Create audio blob
        const audioBlob = new Blob(this.audioChunks, { type: selectedMimeType });
        console.log(`🎤 [HybridSTT] Audio blob created: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

        // Convert to base64
        const audioBase64 = await this._blobToBase64(audioBlob);
        console.log(`🎤 [HybridSTT] Audio converted to base64: ${audioBase64.length} chars`);

        // Send to backend for transcription
        this._sendAudioForTranscription(audioBase64);

        this._triggerEvent('audioend');
      };

      // Start recording
      this.mediaRecorder.start();
      this.isRecording = true;

      console.log(`🎤 [HybridSTT] Recording started (${selectedMimeType})`);
      this._triggerEvent('start');
      this._triggerEvent('audiostart');

      // Auto-stop after silence or time limit (if continuous is false)
      if (!this.options.continuous) {
        setTimeout(() => {
          if (this.isRecording && !this.isPaused) {
            this.stop();
          }
        }, 10000); // 10 second max recording
      }

    } catch (error) {
      console.error('🎤 [HybridSTT] Failed to start recording:', error);
      this._triggerEvent('error', { error: error.message });
    }
  }

  /**
   * Stop audio capture
   */
  stop() {
    if (!this.isRecording) {
      console.warn('🎤 [HybridSTT] Not recording');
      return;
    }

    console.log('🎤 [HybridSTT] Stopping recording...');

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => track.stop());
      this.audioStream = null;
    }

    this.isRecording = false;
    this._triggerEvent('end');
  }

  /**
   * Abort current transcription
   */
  abort() {
    console.log('🎤 [HybridSTT] Aborting...');
    this.audioChunks = [];
    this.stop();
  }

  /**
   * Send audio to backend for transcription
   */
  _sendAudioForTranscription(audioBase64) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('🎤 [HybridSTT] WebSocket not connected');
      this._triggerEvent('error', { error: 'websocket-not-connected' });
      return;
    }

    console.log(`🎤 [HybridSTT] Sending audio to backend (strategy: ${this.options.strategy})`);
    this.stats.totalRequests++;

    // Send audio via WebSocket
    this.ws.send(JSON.stringify({
      type: 'audio',
      data: audioBase64,
      strategy: this.options.strategy,
      speaker_name: this.options.speakerName,
      timestamp: Date.now()
    }));
  }

  /**
   * Handle transcription result from backend
   */
  handleTranscriptionResult(data) {
    const {
      text,
      confidence,
      engine,
      model_name,
      latency_ms,
      speaker_identified
    } = data;

    console.log(`🎤 [HybridSTT] Transcription received:`, {
      text: text.substring(0, 50) + '...',
      confidence: (confidence * 100).toFixed(1) + '%',
      engine,
      model_name,
      latency_ms: latency_ms.toFixed(0) + 'ms',
      speaker_identified
    });

    // Update stats
    this.stats.successfulTranscriptions++;
    this.stats.averageConfidence = (this.stats.averageConfidence * (this.stats.successfulTranscriptions - 1) + confidence) / this.stats.successfulTranscriptions;
    this.stats.averageLatency = (this.stats.averageLatency * (this.stats.successfulTranscriptions - 1) + latency_ms) / this.stats.successfulTranscriptions;
    this.stats.enginesUsed[engine] = (this.stats.enginesUsed[engine] || 0) + 1;

    // Check confidence threshold
    if (confidence < this.options.confidenceThreshold && this.options.retryOnLowConfidence) {
      this.stats.lowConfidenceResults++;

      // Retry logic
      if (this.retryCount < this.options.maxRetries) {
        this.retryCount++;
        this.stats.retries++;
        console.warn(`🎤 [HybridSTT] Low confidence (${(confidence * 100).toFixed(1)}%), retry ${this.retryCount}/${this.options.maxRetries}`);

        // Trigger event for low confidence (UI can show prompt to retry)
        this._triggerEvent('lowconfidence', {
          text,
          confidence,
          retryCount: this.retryCount,
          maxRetries: this.options.maxRetries
        });

        return; // Don't emit onresult yet
      } else {
        console.warn(`🎤 [HybridSTT] Max retries reached, using low-confidence result`);
      }
    }

    // Reset retry count for next transcription
    this.retryCount = 0;
    this.lastTranscript = text;

    // Create SpeechRecognition-compatible result object
    const result = {
      isFinal: true,
      confidence,
      transcript: text,
      // Additional hybrid STT metadata
      engine,
      model_name,
      latency_ms,
      speaker_identified
    };

    // Emit result event (compatible with SpeechRecognition API)
    const event = {
      results: [[result]],
      resultIndex: 0,
      // Convenience properties
      text,
      confidence,
      engine,
      model_name
    };

    this._triggerEvent('result', event);
  }

  /**
   * Handle transcription error from backend
   */
  handleTranscriptionError(data) {
    console.error('🎤 [HybridSTT] Transcription error:', data.message);
    this._triggerEvent('error', {
      error: data.message,
      type: 'transcription-error'
    });
  }

  /**
   * Convert blob to base64
   */
  _blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1]; // Remove data:audio/...;base64, prefix
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Trigger event handler (if defined)
   */
  _triggerEvent(eventName, data = {}) {
    const handlerName = `on${eventName}`;
    if (typeof this[handlerName] === 'function') {
      this[handlerName](data);
    }
  }

  /**
   * Get statistics
   */
  getStats() {
    return {
      ...this.stats,
      averageConfidence: (this.stats.averageConfidence * 100).toFixed(1) + '%',
      averageLatency: this.stats.averageLatency.toFixed(0) + 'ms',
      successRate: this.stats.totalRequests > 0
        ? ((this.stats.successfulTranscriptions / this.stats.totalRequests) * 100).toFixed(1) + '%'
        : '0%'
    };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      totalRequests: 0,
      successfulTranscriptions: 0,
      lowConfidenceResults: 0,
      retries: 0,
      averageConfidence: 0,
      averageLatency: 0,
      enginesUsed: {}
    };
  }
}

export default HybridSTTClient;
