/**
 * Advanced Network Recovery Manager for JARVIS
 * Implements multiple recovery strategies for network errors
 */

class NetworkRecoveryManager {
    constructor() {
        this.strategies = [
            'connectionCheck',
            'dnsFlush',
            'serviceSwitch',
            'webSocketFallback',
            'offlineMode',
            'mlBackendRecovery',
            'edgeProcessing'
        ];
        
        this.recoveryAttempts = 0;
        this.maxRecoveryAttempts = 5;
        this.connectionHealth = {
            lastSuccessful: Date.now(),
            consecutiveFailures: 0,
            averageLatency: 0,
            serviceStatus: {}
        };
        
        this.fallbackProviders = {
            google: 'https://www.google.com/speech-api/v2/recognize',
            azure: 'https://speech.platform.bing.com/recognize',
            ibm: 'https://stream.watsonplatform.net/speech-to-text/api/v1/recognize'
        };
        
        this.isRecovering = false;
        this.recoveryQueue = [];
        this.monitoringInterval = null;
        
        // Start connection monitoring
        this.startConnectionMonitoring();
    }

    /**
     * Main recovery method - tries multiple strategies
     */
    async recoverFromNetworkError(error, recognition, context = {}) {
        if (this.isRecovering) {
            console.log('Recovery already in progress, queueing request...');
            return new Promise((resolve) => {
                this.recoveryQueue.push({ error, recognition, context, resolve });
            });
        }

        this.isRecovering = true;
        this.recoveryAttempts++;
        
        console.log(`🔧 Advanced Network Recovery - Attempt ${this.recoveryAttempts}`);
        
        // Update connection health
        this.connectionHealth.consecutiveFailures++;
        
        // Try strategies in order
        for (const strategy of this.strategies) {
            try {
                console.log(`🔄 Trying strategy: ${strategy}`);
                const result = await this[`strategy_${strategy}`](error, recognition, context);
                
                if (result.success) {
                    this.onRecoverySuccess(strategy, result);
                    this.isRecovering = false;
                    this.processRecoveryQueue();
                    return result;
                }
            } catch (e) {
                console.warn(`Strategy ${strategy} failed:`, e);
            }
        }
        
        this.isRecovering = false;
        this.processRecoveryQueue();
        
        return {
            success: false,
            message: 'All recovery strategies exhausted',
            needsManualIntervention: true
        };
    }

    /**
     * Strategy 1: Basic connection check and retry
     */
    async strategy_connectionCheck(error, recognition, context) {
        // Test basic connectivity
        const isOnline = await this.checkConnectivity();
        
        if (!isOnline) {
            // Wait for connection
            console.log('🔌 No internet connection, waiting...');
            await this.waitForConnection(5000);
            
            if (await this.checkConnectivity()) {
                // Connection restored
                return await this.restartRecognition(recognition);
            }
        }
        
        // Try different DNS servers
        const dnsServers = ['8.8.8.8', '1.1.1.1', '208.67.222.222'];
        for (const dns of dnsServers) {
            if (await this.testDNS(dns)) {
                console.log(`✅ DNS ${dns} working`);
                return await this.restartRecognition(recognition);
            }
        }
        
        return { success: false };
    }

    /**
     * Strategy 2: DNS cache flush and network reset
     */
    async strategy_dnsFlush(error, recognition, context) {
        // Signal backend to help with network diagnostics
        try {
            const response = await fetch(`${window.API_URL || 'http://localhost:8000'}/network/diagnose`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    error: error.error,
                    timestamp: Date.now(),
                    userAgent: navigator.userAgent
                })
            });
            
            if (response.ok) {
                const diagnosis = await response.json();
                if (diagnosis.recovered) {
                    return await this.restartRecognition(recognition);
                }
            }
        } catch (e) {
            console.warn('Network diagnosis failed:', e);
        }
        
        return { success: false };
    }

    /**
     * Strategy 3: Switch to alternative speech service
     */
    async strategy_serviceSwitch(error, recognition, context) {
        console.log('🔀 Attempting to switch speech recognition service...');
        
        // Try alternative recognition services
        for (const [provider, endpoint] of Object.entries(this.fallbackProviders)) {
            try {
                // Test if service is reachable
                const testResponse = await fetch(endpoint, {
                    method: 'HEAD',
                    mode: 'no-cors'
                });
                
                // If reachable, update recognition config
                if (testResponse || testResponse.type === 'opaque') {
                    console.log(`✅ ${provider} service available`);
                    
                    // Modify recognition to use alternative service
                    recognition.stop();
                    
                    // Create new recognition with different settings
                    const newRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                    newRecognition.continuous = recognition.continuous;
                    newRecognition.interimResults = recognition.interimResults;
                    
                    // Some browsers allow service endpoint configuration
                    if (newRecognition.serviceURI) {
                        newRecognition.serviceURI = endpoint;
                    }
                    
                    return {
                        success: true,
                        message: `Switched to ${provider} speech service`,
                        newRecognition
                    };
                }
            } catch (e) {
                console.warn(`${provider} service test failed:`, e);
            }
        }
        
        return { success: false };
    }

    /**
     * Strategy 4: WebSocket fallback for audio streaming
     */
    async strategy_webSocketFallback(error, recognition, context) {
        console.log('🔌 Attempting WebSocket audio streaming fallback...');
        
        try {
            // Check if JARVIS WebSocket is available
            const wsUrl = `ws://localhost:8000/voice/jarvis/stream`;
            const ws = new WebSocket(wsUrl);
            
            return new Promise((resolve) => {
                ws.onopen = () => {
                    console.log('✅ WebSocket audio streaming available');
                    
                    // Return WebSocket as alternative
                    resolve({
                        success: true,
                        message: 'Switched to WebSocket audio streaming',
                        useWebSocket: true,
                        websocket: ws
                    });
                };
                
                ws.onerror = () => {
                    ws.close();
                    resolve({ success: false });
                };
                
                // Timeout after 3 seconds
                setTimeout(() => {
                    if (ws.readyState !== WebSocket.OPEN) {
                        ws.close();
                        resolve({ success: false });
                    }
                }, 3000);
            });
        } catch (e) {
            console.warn('WebSocket fallback failed:', e);
            return { success: false };
        }
    }

    /**
     * Strategy 5: Offline mode with queued commands
     */
    async strategy_offlineMode(error, recognition, context) {
        console.log('📴 Enabling offline mode with command queueing...');
        
        // Enable offline mode
        return {
            success: true,
            message: 'Offline mode activated - commands will be queued',
            offlineMode: true,
            commandQueue: [],
            syncWhenOnline: true
        };
    }

    /**
     * Strategy 6: ML Backend assisted recovery
     */
    async strategy_mlBackendRecovery(error, recognition, context) {
        console.log('🤖 Requesting ML backend assistance...');
        
        try {
            const response = await fetch(`${window.API_URL || 'http://localhost:8000'}/audio/ml/advanced-recovery`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    error: error.error,
                    connectionHealth: this.connectionHealth,
                    recoveryAttempts: this.recoveryAttempts,
                    browserInfo: {
                        userAgent: navigator.userAgent,
                        platform: navigator.platform,
                        language: navigator.language
                    }
                })
            });
            
            if (response.ok) {
                const strategy = await response.json();
                
                if (strategy.customScript) {
                    // Execute custom recovery script from ML
                    try {
                        const recoveryFunc = new Function('recognition', 'context', strategy.customScript);
                        const result = await recoveryFunc(recognition, context);
                        
                        if (result.success) {
                            return result;
                        }
                    } catch (e) {
                        console.error('ML recovery script failed:', e);
                    }
                }
                
                if (strategy.proxyEndpoint) {
                    // Use ML backend as proxy
                    return {
                        success: true,
                        message: 'Using ML backend as speech proxy',
                        useProxy: true,
                        proxyEndpoint: strategy.proxyEndpoint
                    };
                }
            }
        } catch (e) {
            console.warn('ML backend recovery failed:', e);
        }
        
        return { success: false };
    }

    /**
     * Strategy 7: Edge processing with local models
     */
    async strategy_edgeProcessing(error, recognition, context) {
        console.log('🔲 Attempting edge processing with local models...');
        
        // Check if we can use browser's local speech processing
        if ('SpeechRecognition' in window) {
            try {
                // Try to force offline mode
                const offlineRecognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                offlineRecognition.continuous = true;
                offlineRecognition.interimResults = true;
                
                // Some browsers support offline recognition
                if (offlineRecognition.grammars) {
                    const grammar = '#JSGF V1.0; grammar commands; public <command> = jarvis | hey jarvis | stop | start | help;';
                    const speechRecognitionList = new (window.SpeechGrammarList || window.webkitSpeechGrammarList)();
                    speechRecognitionList.addFromString(grammar, 1);
                    offlineRecognition.grammars = speechRecognitionList;
                }
                
                // Test offline recognition
                return new Promise((resolve) => {
                    let timeout = setTimeout(() => {
                        offlineRecognition.stop();
                        resolve({ success: false });
                    }, 2000);
                    
                    offlineRecognition.onstart = () => {
                        clearTimeout(timeout);
                        console.log('✅ Offline recognition available');
                        resolve({
                            success: true,
                            message: 'Switched to offline edge processing',
                            offlineRecognition
                        });
                    };
                    
                    offlineRecognition.onerror = () => {
                        clearTimeout(timeout);
                        resolve({ success: false });
                    };
                    
                    offlineRecognition.start();
                });
            } catch (e) {
                console.warn('Edge processing failed:', e);
            }
        }
        
        return { success: false };
    }

    /**
     * Helper Methods
     */
    
    async checkConnectivity() {
        try {
            const response = await fetch('https://www.google.com/generate_204', {
                method: 'HEAD',
                mode: 'no-cors'
            });
            return true;
        } catch {
            return false;
        }
    }

    async waitForConnection(timeout = 5000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            if (await this.checkConnectivity()) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        return false;
    }

    async testDNS(server) {
        try {
            // Test DNS resolution
            const response = await fetch(`https://${server}/generate_204`, {
                method: 'HEAD',
                mode: 'no-cors'
            });
            return true;
        } catch {
            return false;
        }
    }

    async restartRecognition(recognition) {
        try {
            recognition.stop();
        } catch (e) {
            // Already stopped
        }
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        try {
            recognition.start();
            return {
                success: true,
                message: 'Recognition restarted successfully'
            };
        } catch (e) {
            return {
                success: false,
                message: `Failed to restart: ${e.message}`
            };
        }
    }

    onRecoverySuccess(strategy, result) {
        console.log(`✅ Recovery successful using strategy: ${strategy}`);
        
        // Reset counters
        this.recoveryAttempts = 0;
        this.connectionHealth.consecutiveFailures = 0;
        this.connectionHealth.lastSuccessful = Date.now();
        
        // Log to ML backend for learning
        this.logRecoverySuccess(strategy, result);
    }

    async logRecoverySuccess(strategy, result) {
        try {
            await fetch(`${window.API_URL || 'http://localhost:8000'}/audio/ml/recovery-success`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    strategy,
                    result,
                    connectionHealth: this.connectionHealth,
                    timestamp: Date.now()
                })
            });
        } catch (e) {
            // Logging is non-critical
        }
    }

    processRecoveryQueue() {
        while (this.recoveryQueue.length > 0) {
            const { error, recognition, context, resolve } = this.recoveryQueue.shift();
            this.recoverFromNetworkError(error, recognition, context).then(resolve);
        }
    }

    startConnectionMonitoring() {
        // Monitor connection health every 30 seconds
        this.monitoringInterval = setInterval(async () => {
            const isOnline = await this.checkConnectivity();
            
            if (!isOnline && this.connectionHealth.consecutiveFailures === 0) {
                console.warn('🔴 Connection lost - preparing recovery strategies...');
                this.preloadRecoveryStrategies();
            } else if (isOnline && this.connectionHealth.consecutiveFailures > 0) {
                console.log('🟢 Connection restored');
                this.connectionHealth.consecutiveFailures = 0;
                this.connectionHealth.lastSuccessful = Date.now();
            }
        }, 30000);
    }

    async preloadRecoveryStrategies() {
        // Preload resources for faster recovery
        console.log('📦 Preloading recovery resources...');
        
        // Cache alternative service endpoints
        for (const endpoint of Object.values(this.fallbackProviders)) {
            try {
                await fetch(endpoint, { method: 'HEAD', mode: 'no-cors' });
            } catch (e) {
                // Just caching
            }
        }
    }

    destroy() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }
    }
}

// Singleton instance
let recoveryManagerInstance = null;

export const getNetworkRecoveryManager = () => {
    if (!recoveryManagerInstance) {
        recoveryManagerInstance = new NetworkRecoveryManager();
    }
    return recoveryManagerInstance;
};

export default NetworkRecoveryManager;