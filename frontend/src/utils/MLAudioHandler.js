/**
 * ML-Enhanced Audio Handler for JARVIS
 * Integrates with backend ML audio manager for intelligent error recovery
 */

import configService from '../services/DynamicConfigService';

class MLAudioHandler {
    constructor() {
        this.ws = null;
        this.errorHistory = [];
        this.recoveryStrategies = new Map();
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 10;
        this.backoffMultiplier = 1.5;
        this.currentBackoffDelay = 1000;
        this.isConnecting = false;
        this.metrics = {
            errors: 0,
            recoveries: 0,
            startTime: Date.now()
        };

        // Configuration (loaded dynamically)
        this.config = {
            enableML: true,
            autoRecovery: true,
            maxRetries: 5,
            retryDelays: [100, 500, 1000, 2000, 5000],
            anomalyThreshold: 0.8,
            predictionThreshold: 0.7
        };

        // Wait for config service before initializing
        this.initializeWhenReady();

        // Browser detection
        this.browserInfo = this.detectBrowser();

        // Permission state tracking
        this.permissionState = 'unknown';
        this.checkPermissionState();
    }

    async initializeWhenReady() {
        try {
            await configService.waitForConfig();
            console.log('ML Audio: Config service ready, initializing...');
            
            // Load configuration from backend
            await this.loadConfiguration();
            
            // Initialize WebSocket connection to ML backend
            setTimeout(() => this.connectToMLBackend(), 2000);
        } catch (error) {
            console.error('ML Audio: Failed to initialize', error);
            // Retry initialization
            setTimeout(() => this.initializeWhenReady(), 5000);
        }
    }

    async loadConfiguration() {
        try {
            const apiUrl = configService.getApiUrl('audio/ml/config');
            if (!apiUrl) {
                console.warn('ML Audio: API URL not available yet');
                return;
            }
            
            const response = await fetch(apiUrl);
            if (response.ok) {
                const config = await response.json();
                this.config = { ...this.config, ...config };
                console.log('Loaded ML audio configuration:', this.config);
            }
        } catch (error) {
            console.warn('Using default ML audio configuration');
        }
    }

    async connectToMLBackend() {
        if (this.isConnecting || this.connectionAttempts >= this.maxConnectionAttempts) {
            if (this.connectionAttempts >= this.maxConnectionAttempts) {
                console.warn('ML Audio: Max connection attempts reached, stopping reconnection');
            }
            return;
        }

        this.isConnecting = true;
        this.connectionAttempts++;

        try {
            // Try unified WebSocket first, fall back to ML-specific endpoint
            let wsUrl = configService.getWebSocketUrl('ws');
            if (!wsUrl) {
                // Try the ML-specific endpoint
                wsUrl = configService.getWebSocketUrl('audio/ml/stream');
            }
            if (!wsUrl) {
                console.warn('ML Audio: WebSocket URL not available yet');
                this.isConnecting = false;
                setTimeout(() => this.connectToMLBackend(), 5000);
                return;
            }
            
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('Connected to ML Audio Backend');
                this.connectionAttempts = 0;
                this.currentBackoffDelay = 1000;
                this.isConnecting = false;
                this.sendTelemetry('connection', { status: 'connected' });
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMLMessage(data);
            };

            this.ws.onerror = (error) => {
                // Don't log every error to avoid spam
                if (this.connectionAttempts === 1) {
                    console.warn('ML Audio WebSocket not available, will retry...');
                }
            };

            this.ws.onclose = () => {
                this.isConnecting = false;
                if (this.connectionAttempts < this.maxConnectionAttempts) {
                    const delay = Math.min(this.currentBackoffDelay, 30000); // Max 30s
                    console.log(`ML Audio WebSocket closed, retrying in ${delay / 1000}s (attempt ${this.connectionAttempts}/${this.maxConnectionAttempts})`);
                    setTimeout(() => this.connectToMLBackend(), delay);
                    this.currentBackoffDelay = Math.floor(this.currentBackoffDelay * this.backoffMultiplier);
                }
            };
        } catch (error) {
            this.isConnecting = false;
            console.error('Failed to connect to ML backend:', error);
        }
    }

    handleMLMessage(data) {
        switch (data.type) {
            case 'prediction':
                this.handlePrediction(data.prediction);
                break;
            case 'strategy':
                this.handleStrategy(data.strategy);
                break;
            case 'anomaly':
                this.handleAnomaly(data.anomaly);
                break;
            case 'metrics':
                this.updateMetrics(data.metrics);
                break;
        }
    }

    async handleAudioError(error, recognition) {
        // Handle "no-speech" errors quietly - they're expected during silence
        if (error.error === 'no-speech') {
            // Don't log to console - this is normal behavior
            // Just update metrics silently
            this.metrics.noSpeechEvents = (this.metrics.noSpeechEvents || 0) + 1;
        } else {
            // Log actual errors
            console.error('Audio error detected:', error);
            this.metrics.errors++;
        }

        // Create error context
        const context = {
            error_code: error.error || error.name,
            browser: this.browserInfo.name,
            browser_version: this.browserInfo.version,
            timestamp: new Date().toISOString(),
            session_duration: Date.now() - this.metrics.startTime,
            retry_count: this.getRetryCount(error.error),
            permission_state: this.permissionState,
            user_agent: navigator.userAgent,
            audio_context_state: this.getAudioContextState(),
            previous_errors: this.getRecentErrors(5)
        };

        // Record error
        this.errorHistory.push({
            ...context,
            resolved: false
        });

        // Send to ML backend
        const response = await this.sendErrorToBackend(context);

        if (response) {
            // Check if response contains a strategy object with action
            if (response.strategy && response.strategy.action) {
                // Execute ML-recommended strategy
                return await this.executeStrategy(response.strategy, error, recognition);
            } else if (response.success === false) {
                // Backend couldn't provide a strategy, log the response
                console.log('ML backend response:', response);
                // Fallback to local strategy
                return await this.executeLocalStrategy(error, recognition);
            }
        } else {
            // No response from backend, fallback to local strategy
            return await this.executeLocalStrategy(error, recognition);
        }
        
        // Default return if nothing else worked
        return { success: false, message: 'No recovery strategy available' };
    }

    async executeLocalStrategy(error, recognition) {
        // Local fallback strategies when ML backend is unavailable
        console.log('Executing local recovery strategy for:', error.error);
        
        switch (error.error) {
            case 'no-speech':
                // No-speech is normal, just return success
                return { success: true, message: 'No speech detected (normal)' };
                
            case 'audio-capture':
            case 'not-allowed':
                // Try to request permission
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    stream.getTracks().forEach(track => track.stop());
                    return { success: true, message: 'Permission granted', newContext: true };
                } catch (e) {
                    return { success: false, message: 'Permission denied' };
                }
                
            case 'network':
                // Network error - implement retry logic
                console.log('Handling network error with retry strategy');
                
                // Stop current recognition
                try {
                    recognition.stop();
                } catch (e) {
                    console.log('Recognition already stopped');
                }
                
                // Wait a bit then restart
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                try {
                    recognition.start();
                    return { 
                        success: true, 
                        message: 'Network error recovered - recognition restarted',
                        strategy: 'network_retry'
                    };
                } catch (e) {
                    console.error('Failed to restart after network error:', e);
                    return { 
                        success: false, 
                        message: 'Network error - unable to restart recognition',
                        needsManualRestart: true
                    };
                }
                
            default:
                // Unknown error
                return { success: false, message: `Unknown error: ${error.error}` };
        }
    }

    async sendErrorToBackend(context) {
        try {
            const apiUrl = configService.getApiUrl('audio/ml/error');
            if (!apiUrl) {
                console.warn('ML Audio: API URL not available for error endpoint');
                return null;
            }
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(context)
            });

            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Failed to send error to ML backend:', error);
        }
        return null;
    }

    async executeStrategy(strategy, error, recognition) {
        console.log('Executing ML strategy:', strategy);

        // Handle case where strategy doesn't have an action property
        if (!strategy || !strategy.action) {
            console.warn('Invalid strategy format:', strategy);
            return { success: false, message: 'Invalid strategy format' };
        }

        const action = strategy.action;

        // Ensure action has a type
        if (!action.type) {
            console.warn('Strategy action missing type:', action);
            return { success: false, message: 'Strategy action missing type' };
        }

        switch (action.type) {
            case 'request_media_permission':
                return await this.requestPermissionWithRetry(action.params);

            case 'show_instructions':
                return this.showInstructions(action.params);

            case 'restart_audio_context':
                return await this.restartAudioContext(recognition);

            case 'enable_text_fallback':
                return this.enableTextFallback(action.params);

            case 'show_system_settings':
                return this.showSystemSettings(action.params);

            default:
                console.warn('Unknown strategy action:', action.type);
                return { success: false };
        }
    }

    async executeLocalStrategy(error, recognition) {
        // Only log for non-no-speech errors
        if (error.error !== 'no-speech') {
            console.log('Executing local fallback strategy for error:', error.error);
        }

        // Local strategy based on error type
        switch (error.error) {
            case 'not-allowed':
            case 'permission-denied':
                return await this.requestPermissionWithRetry({});

            case 'no-speech':
                // No speech detected is often not a critical error
                return {
                    success: true,
                    message: 'No speech detected - continuing to listen'
                };

            case 'audio-capture':
                // Try to restart audio context
                return await this.restartAudioContext(recognition);

            case 'network':
            case 'service-not-allowed':
                // Network errors might resolve on their own
                return {
                    success: false,
                    message: 'Network or service error - will retry automatically'
                };

            case 'aborted':
                // Recognition was aborted (likely due to restart)
                // This is expected during microphone restarts, treat as success
                return {
                    success: true,
                    message: 'Recognition aborted - normal during restart cycle',
                    skipRestart: true  // Don't trigger another restart
                };

            default:
                console.warn('No local strategy for error type:', error.error);
                return {
                    success: false,
                    message: `Unhandled error type: ${error.error}`
                };
        }
    }

    async requestPermissionWithRetry(params) {
        const { retryDelays = this.config.retryDelays } = params;

        for (let i = 0; i < retryDelays.length; i++) {
            try {
                console.log(`Requesting microphone permission (attempt ${i + 1})`);

                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                // Success! Clean up and report
                stream.getTracks().forEach(track => track.stop());

                this.sendTelemetry('recovery', {
                    method: 'request_permission',
                    attempts: i + 1,
                    success: true
                });

                this.metrics.recoveries++;

                return {
                    success: true,
                    message: 'Microphone permission granted',
                    attempts: i + 1
                };

            } catch (error) {
                console.error(`Permission attempt ${i + 1} failed:`, error);

                if (i < retryDelays.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, retryDelays[i]));
                }
            }
        }

        return {
            success: false,
            message: 'Failed to obtain microphone permission after retries'
        };
    }

    showInstructions(params) {
        const { instructions, browser } = params;

        // Create or update instruction UI
        let instructionDiv = document.getElementById('ml-audio-instructions');
        if (!instructionDiv) {
            instructionDiv = document.createElement('div');
            instructionDiv.id = 'ml-audio-instructions';
            instructionDiv.className = 'ml-audio-instructions';
            document.body.appendChild(instructionDiv);
        }

        // Generate instruction HTML
        const html = `
            <div class="ml-instructions-container">
                <div class="ml-instructions-header">
                    <span class="ml-icon">🎤</span>
                    <h3>Microphone Permission Required</h3>
                    <button class="ml-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="ml-instructions-body">
                    <p>To use voice commands, please grant microphone access:</p>
                    <ol>
                        ${instructions.map(step => `<li>${step}</li>`).join('')}
                    </ol>
                    <div class="ml-browser-specific">
                        <img src="/images/browsers/${browser}.svg" alt="${browser}" />
                        <span>Instructions for ${browser}</span>
                    </div>
                </div>
                <div class="ml-instructions-footer">
                    <button class="ml-retry-button" onclick="window.mlAudioHandler.retryPermission()">
                        🔄 Retry Permission
                    </button>
                    <button class="ml-text-mode-button" onclick="window.mlAudioHandler.enableTextMode()">
                        ⌨️ Use Text Mode
                    </button>
                </div>
            </div>
        `;

        instructionDiv.innerHTML = html;
        instructionDiv.style.display = 'block';

        // Add animation
        requestAnimationFrame(() => {
            instructionDiv.classList.add('ml-show');
        });

        return { success: true, message: 'Instructions displayed' };
    }

    async restartAudioContext(recognition) {
        try {
            // Stop current recognition
            if (recognition && recognition.stop) {
                recognition.stop();
            }

            // Create new audio context
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const newContext = new AudioContext();

            // Test with getUserMedia
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = newContext.createMediaStreamSource(stream);

            // Verify it works
            const analyser = newContext.createAnalyser();
            source.connect(analyser);

            // Clean up test
            stream.getTracks().forEach(track => track.stop());

            this.sendTelemetry('recovery', {
                method: 'restart_audio_context',
                success: true
            });

            return {
                success: true,
                message: 'Audio context restarted successfully',
                newContext
            };

        } catch (error) {
            console.error('Failed to restart audio context:', error);
            return {
                success: false,
                message: 'Audio context restart failed'
            };
        }
    }

    enableTextFallback(params) {
        // Emit event for UI to handle
        window.dispatchEvent(new CustomEvent('enableTextFallback', {
            detail: params
        }));

        this.sendTelemetry('fallback', {
            mode: 'text',
            reason: 'audio_error'
        });

        return {
            success: true,
            message: 'Text input mode enabled'
        };
    }

    showSystemSettings(params) {
        const { os, setting } = params;

        // OS-specific guidance
        const guides = {
            macos: {
                microphone_permissions: [
                    'Open System Preferences',
                    'Go to Security & Privacy → Privacy',
                    'Select Microphone from the left sidebar',
                    'Ensure your browser is checked ✓',
                    'Restart your browser after changes'
                ]
            },
            windows: {
                microphone_permissions: [
                    'Open Settings (Win + I)',
                    'Go to Privacy → Microphone',
                    'Ensure "Allow apps to access your microphone" is ON',
                    'Scroll down and ensure your browser is allowed',
                    'Restart your browser after changes'
                ]
            }
        };

        const instructions = guides[os]?.[setting] || ['Check system microphone settings'];

        return this.showInstructions({
            instructions,
            browser: 'system'
        });
    }

    // Predictive capabilities
    async predictAudioIssue() {
        if (!this.config.enableML) return null;

        const context = {
            browser: this.browserInfo.name,
            time_of_day: new Date().getHours(),
            day_of_week: new Date().getDay(),
            error_history: this.getRecentErrors(10),
            session_duration: Date.now() - this.metrics.startTime,
            permission_state: this.permissionState
        };

        try {
            const apiUrl = configService.getApiUrl('audio/ml/predict');
            if (!apiUrl) {
                console.warn('ML Audio: API URL not available for predict endpoint');
                return null;
            }
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(context)
            });

            if (response.ok) {
                const prediction = await response.json();

                if (prediction.probability > this.config.predictionThreshold) {
                    this.handlePrediction(prediction);
                }

                return prediction;
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
        }

        return null;
    }

    handlePrediction(prediction) {
        console.log('ML Prediction:', prediction);

        if (prediction.probability > this.config.predictionThreshold) {
            // Proactive mitigation
            window.dispatchEvent(new CustomEvent('audioIssuePredicted', {
                detail: {
                    prediction,
                    suggestedAction: prediction.recommended_action
                }
            }));
        }
    }

    handleAnomaly(anomaly) {
        console.warn('Audio anomaly detected:', anomaly);

        // Log for analysis
        this.sendTelemetry('anomaly', anomaly);

        // Notify UI
        window.dispatchEvent(new CustomEvent('audioAnomaly', {
            detail: anomaly
        }));
    }

    // Utility methods
    detectBrowser() {
        const ua = navigator.userAgent;
        let name = 'unknown';
        let version = '';

        if (ua.includes('Chrome') && !ua.includes('Edg')) {
            name = 'chrome';
            version = ua.match(/Chrome\/(\d+)/)?.[1] || '';
        } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
            name = 'safari';
            version = ua.match(/Version\/(\d+)/)?.[1] || '';
        } else if (ua.includes('Firefox')) {
            name = 'firefox';
            version = ua.match(/Firefox\/(\d+)/)?.[1] || '';
        } else if (ua.includes('Edg')) {
            name = 'edge';
            version = ua.match(/Edg\/(\d+)/)?.[1] || '';
        }

        return { name, version, ua };
    }

    async checkPermissionState() {
        if ('permissions' in navigator) {
            try {
                const result = await navigator.permissions.query({ name: 'microphone' });
                this.permissionState = result.state;

                result.addEventListener('change', () => {
                    this.permissionState = result.state;
                    this.sendTelemetry('permission_change', { state: result.state });
                });
            } catch (error) {
                console.warn('Permission API not fully supported');
            }
        }
    }

    getAudioContextState() {
        if (window.AudioContext || window.webkitAudioContext) {
            try {
                const context = new (window.AudioContext || window.webkitAudioContext)();
                const state = context.state;
                context.close();
                return state;
            } catch (error) {
                return 'error';
            }
        }
        return 'unsupported';
    }

    getRetryCount(errorCode) {
        return this.errorHistory.filter(e => e.error_code === errorCode).length;
    }

    getRecentErrors(count) {
        return this.errorHistory.slice(-count).map(e => ({
            code: e.error_code,
            timestamp: e.timestamp,
            resolved: e.resolved
        }));
    }

    sendTelemetry(event, data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Use ml_audio_stream type for unified WebSocket compatibility
            this.ws.send(JSON.stringify({
                type: 'ml_audio_stream',
                subtype: 'telemetry',
                event,
                data,
                timestamp: new Date().toISOString()
            }));
        }
    }

    updateMetrics(metrics) {
        console.log('ML Audio Metrics:', metrics);
        this.metrics = { ...this.metrics, ...metrics };

        // Emit metrics update event
        window.dispatchEvent(new CustomEvent('audioMetricsUpdate', {
            detail: this.metrics
        }));
    }

    // Public methods for UI integration
    async retryPermission() {
        return await this.requestPermissionWithRetry({});
    }

    enableTextMode() {
        return this.enableTextFallback({ showKeyboard: true });
    }

    getMetrics() {
        return {
            ...this.metrics,
            errorRate: this.metrics.errors / Math.max((Date.now() - this.metrics.startTime) / 60000, 1),
            recoveryRate: this.metrics.recoveries / Math.max(this.metrics.errors, 1)
        };
    }
}

// Create singleton instance
const mlAudioHandler = new MLAudioHandler();

// Make available globally for debugging
window.mlAudioHandler = mlAudioHandler;

export default mlAudioHandler;