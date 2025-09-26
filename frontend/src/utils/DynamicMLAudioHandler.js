/**
 * Dynamic ML-Enhanced Audio Handler
 * =================================
 * Self-configuring audio handler that dynamically discovers
 * and connects to backend services with zero hardcoding.
 */

import configService from '../services/DynamicConfigService';

class DynamicMLAudioHandler {
    constructor() {
        this.ws = null;
        this.errorHistory = [];
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 10;
        this.isConnecting = false;
        this.isReady = false;
        
        // Metrics for self-healing
        this.metrics = {
            errors: 0,
            recoveries: 0,
            startTime: Date.now(),
            lastError: null,
            connectionHealth: 1.0
        };

        // Dynamic configuration
        this.config = {
            enableML: true,
            autoRecovery: true,
            maxRetries: 5,
            retryDelays: [100, 500, 1000, 2000, 5000],
            anomalyThreshold: 0.8,
            predictionThreshold: 0.7
        };

        // Browser detection
        this.browserInfo = this.detectBrowser();
        this.permissionState = 'unknown';

        // Initialize when config is ready
        this.initialize();
    }

    async initialize() {
        console.log('🎤 Initializing Dynamic ML Audio Handler...');
        
        try {
            // Wait for configuration service
            await configService.waitForConfig();
            
            // Load ML configuration from discovered backend
            await this.loadConfiguration();
            
            // Connect to ML backend
            this.connectToMLBackend();
            
            // Check permissions
            this.checkPermissionState();
            
            // Listen for service changes
            configService.on('service-relocated', (event) => {
                console.log('🔄 Backend relocated, reconnecting ML audio...');
                this.reconnect();
            });
            
        } catch (error) {
            console.error('Failed to initialize ML Audio:', error);
            // Retry initialization
            setTimeout(() => this.initialize(), 5000);
        }
    }

    async loadConfiguration() {
        try {
            const configUrl = configService.getApiUrl('ml_audio_config');
            if (!configUrl) {
                console.warn('ML audio config endpoint not discovered yet');
                return;
            }
            
            const response = await fetch(configUrl);
            if (response.ok) {
                const config = await response.json();
                this.config = { ...this.config, ...config };
                console.log('✅ Loaded ML audio configuration:', this.config);
                
                // Store config health
                this.metrics.lastConfigLoad = Date.now();
            }
        } catch (error) {
            console.warn('Using default ML audio configuration');
        }
    }

    connectToMLBackend() {
        if (this.isConnecting || this.connectionAttempts >= this.maxConnectionAttempts) {
            return;
        }

        const wsUrl = configService.getWebSocketUrl('audio/ml/stream');
        if (!wsUrl) {
            console.warn('ML audio WebSocket URL not discovered yet');
            setTimeout(() => this.connectToMLBackend(), 2000);
            return;
        }

        this.isConnecting = true;
        this.connectionAttempts++;

        try {
            console.log(`🔌 Connecting to ML Audio: ${wsUrl}`);
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('✅ Connected to ML Audio Backend');
                this.connectionAttempts = 0;
                this.isConnecting = false;
                this.isReady = true;
                this.metrics.connectionHealth = 1.0;
                
                this.sendTelemetry('connection', { 
                    status: 'connected',
                    attempts: this.connectionAttempts,
                    url: wsUrl
                });
                
                // Notify listeners
                this.emit('connected');
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMLMessage(data);
            };

            this.ws.onerror = (error) => {
                this.metrics.errors++;
                this.metrics.lastError = Date.now();
                this.metrics.connectionHealth *= 0.9;
                
                if (this.connectionAttempts === 1) {
                    console.warn('ML Audio WebSocket error, attempting recovery...');
                }
                
                // Trigger self-healing
                this.attemptSelfHealing();
            };

            this.ws.onclose = () => {
                this.isConnecting = false;
                this.isReady = false;
                
                if (this.connectionAttempts < this.maxConnectionAttempts) {
                    const delay = this.calculateBackoffDelay();
                    console.log(`🔄 ML Audio reconnecting in ${delay}ms...`);
                    setTimeout(() => this.connectToMLBackend(), delay);
                } else {
                    console.error('❌ ML Audio max reconnection attempts reached');
                    this.emit('connection-failed');
                }
            };

        } catch (error) {
            this.isConnecting = false;
            console.error('ML Audio connection error:', error);
            this.attemptSelfHealing();
        }
    }

    calculateBackoffDelay() {
        // Intelligent backoff based on connection health
        const baseDelay = this.config.retryDelays[
            Math.min(this.connectionAttempts - 1, this.config.retryDelays.length - 1)
        ];
        
        // Adjust based on health score
        const healthMultiplier = 2 - this.metrics.connectionHealth;
        return Math.min(baseDelay * healthMultiplier, 30000);
    }

    async attemptSelfHealing() {
        console.log('🔧 Attempting ML Audio self-healing...');
        
        // Strategy 1: Re-discover backend
        await configService.discover();
        
        // Strategy 2: Reset connection state
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        // Strategy 3: Reload configuration
        await this.loadConfiguration();
        
        // Strategy 4: Clear error history if too many
        if (this.errorHistory.length > 100) {
            this.errorHistory = this.errorHistory.slice(-50);
        }
        
        this.metrics.recoveries++;
    }

    handleMLMessage(data) {
        switch (data.type) {
            case 'error_prediction':
                this.handleErrorPrediction(data);
                break;
                
            case 'recovery_suggestion':
                this.applyRecoverySuggestion(data);
                break;
                
            case 'anomaly_detected':
                this.handleAnomaly(data);
                break;
                
            case 'config_update':
                this.updateConfiguration(data.config);
                break;
                
            case 'health_check':
                this.respondToHealthCheck();
                break;
                
            default:
                console.log('ML Audio message:', data);
        }
    }

    handleAudioError(error, context = {}) {
        // Track error
        const errorData = {
            timestamp: Date.now(),
            type: error.name || 'UnknownError',
            message: error.message,
            code: error.code,
            context: {
                ...context,
                browser: this.browserInfo.name,
                platform: navigator.platform,
                permissionState: this.permissionState,
                connectionHealth: this.metrics.connectionHealth
            }
        };

        this.errorHistory.push(errorData);
        this.metrics.errors++;

        // Send to ML backend if connected
        if (this.isReady && this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'error_report',
                error: errorData,
                history: this.errorHistory.slice(-10)
            }));
        }

        // Local recovery attempt
        return this.attemptLocalRecovery(error, context);
    }

    async attemptLocalRecovery(error, context) {
        // Dynamic recovery strategies based on error type
        const strategies = this.getRecoveryStrategies(error);
        
        for (const strategy of strategies) {
            try {
                console.log(`🔧 Trying recovery strategy: ${strategy.name}`);
                const result = await strategy.execute(context);
                
                if (result.success) {
                    console.log(`✅ Recovery successful: ${strategy.name}`);
                    this.metrics.recoveries++;
                    
                    // Report success to ML backend
                    this.sendTelemetry('recovery_success', {
                        strategy: strategy.name,
                        error: error.name
                    });
                    
                    return result;
                }
            } catch (strategyError) {
                console.warn(`Strategy ${strategy.name} failed:`, strategyError);
            }
        }
        
        return { success: false };
    }

    getRecoveryStrategies(error) {
        const strategies = [];
        
        // Permission errors
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            strategies.push({
                name: 'permission_prompt',
                execute: async () => {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        stream.getTracks().forEach(track => track.stop());
                        return { success: true };
                    } catch {
                        return { success: false };
                    }
                }
            });
        }
        
        // Device errors
        if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            strategies.push({
                name: 'device_refresh',
                execute: async () => {
                    // Re-enumerate devices
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const hasAudioInput = devices.some(d => d.kind === 'audioinput');
                    return { success: hasAudioInput };
                }
            });
        }
        
        // Network errors
        if (error.name === 'NetworkError' || !navigator.onLine) {
            strategies.push({
                name: 'wait_for_network',
                execute: async () => {
                    return new Promise(resolve => {
                        const checkNetwork = () => {
                            if (navigator.onLine) {
                                resolve({ success: true });
                            } else {
                                setTimeout(checkNetwork, 1000);
                            }
                        };
                        checkNetwork();
                    });
                }
            });
        }
        
        // Generic fallback
        strategies.push({
            name: 'reset_audio_context',
            execute: async (context) => {
                if (context.audioContext) {
                    await context.audioContext.close();
                    context.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    return { success: true };
                }
                return { success: false };
            }
        });
        
        return strategies;
    }

    detectBrowser() {
        const ua = navigator.userAgent;
        if (ua.includes('Chrome')) return { name: 'Chrome', version: ua.match(/Chrome\/(\d+)/)?.[1] };
        if (ua.includes('Safari')) return { name: 'Safari', version: ua.match(/Safari\/(\d+)/)?.[1] };
        if (ua.includes('Firefox')) return { name: 'Firefox', version: ua.match(/Firefox\/(\d+)/)?.[1] };
        return { name: 'Unknown', version: '0' };
    }

    async checkPermissionState() {
        if ('permissions' in navigator) {
            try {
                const result = await navigator.permissions.query({ name: 'microphone' });
                this.permissionState = result.state;
                
                result.addEventListener('change', () => {
                    this.permissionState = result.state;
                    console.log('Microphone permission state changed:', this.permissionState);
                });
            } catch {
                // Fallback for browsers that don't support permission query
                this.permissionState = 'unknown';
            }
        }
    }

    sendTelemetry(event, data) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'telemetry',
                event,
                data,
                timestamp: Date.now()
            }));
        }
    }

    // Event emitter functionality
    emit(event, data) {
        window.dispatchEvent(new CustomEvent(`mlaudio:${event}`, { detail: data }));
    }

    on(event, callback) {
        window.addEventListener(`mlaudio:${event}`, (e) => callback(e.detail));
    }

    reconnect() {
        this.connectionAttempts = 0;
        if (this.ws) {
            this.ws.close();
        }
        this.connectToMLBackend();
    }

    getMetrics() {
        return {
            ...this.metrics,
            uptime: Date.now() - this.metrics.startTime,
            errorRate: this.metrics.errors / ((Date.now() - this.metrics.startTime) / 1000),
            isReady: this.isReady
        };
    }
}

// Create singleton instance
const mlAudioHandler = new DynamicMLAudioHandler();

export default mlAudioHandler;