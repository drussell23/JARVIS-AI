/**
 * JARVIS Advanced Loading Manager v2.0
 *
 * Features:
 * - Smooth 1-100% progress with real-time updates
 * - Robust error handling with automatic retry
 * - WebSocket + HTTP polling fallback
 * - Automatic transition to main app at 100%
 */

class JARVISLoadingManager {
    constructor() {
        this.config = {
            loadingServerPort: 3001,
            mainAppPort: 3000,
            wsProtocol: window.location.protocol === 'https:' ? 'wss:' : 'ws:',
            httpProtocol: window.location.protocol,
            hostname: window.location.hostname || 'localhost',
            reconnect: {
                enabled: true,
                initialDelay: 500,
                maxDelay: 5000,
                maxAttempts: 20,
                backoffMultiplier: 1.3
            },
            polling: {
                enabled: true,
                interval: 500,  // Poll every 500ms for smooth updates
                timeout: 3000
            },
            smoothProgress: {
                enabled: true,
                incrementDelay: 100,  // Update every 100ms for smooth animation
                maxAutoProgress: 95   // Don't auto-progress past 95%
            }
        };

        this.state = {
            ws: null,
            connected: false,
            progress: 0,
            targetProgress: 0,
            stage: 'initializing',
            message: 'Initializing JARVIS...',
            reconnectAttempts: 0,
            pollingInterval: null,
            smoothProgressInterval: null,
            startTime: Date.now(),
            lastUpdate: Date.now()
        };

        this.elements = this.cacheElements();
        this.init();
    }

    cacheElements() {
        return {
            statusText: document.getElementById('status-text'),
            subtitle: document.getElementById('subtitle'),
            progressBar: document.getElementById('progress-bar'),
            progressPercentage: document.getElementById('progress-percentage'),
            statusMessage: document.getElementById('status-message'),
            errorContainer: document.getElementById('error-container'),
            errorMessage: document.getElementById('error-message'),
            reactor: document.querySelector('.arc-reactor')
        };
    }

    async init() {
        console.log('[JARVIS] Loading Manager v2.0 starting...');
        console.log(`[Config] Loading Server: ${this.config.hostname}:${this.config.loadingServerPort}`);

        // Create particle background
        this.createParticles();

        // Start smooth progress animation
        this.startSmoothProgress();

        // Try WebSocket first
        await this.connectWebSocket();

        // Also start HTTP polling as backup
        this.startPolling();

        // Monitor for stalls
        this.startHealthMonitoring();
    }

    createParticles() {
        const container = document.getElementById('particles');
        if (!container) return;

        const particleCount = 50;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';

            const size = Math.random() * 3 + 1;
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${Math.random() * 100}%`;

            const duration = Math.random() * 10 + 10;
            const delay = Math.random() * 5;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;

            container.appendChild(particle);
        }
    }

    async connectWebSocket() {
        if (this.state.reconnectAttempts >= this.config.reconnect.maxAttempts) {
            console.warn('[WebSocket] Max reconnection attempts reached');
            return;
        }

        try {
            const wsUrl = `${this.config.wsProtocol}//${this.config.hostname}:${this.config.loadingServerPort}/ws/startup-progress`;
            console.log(`[WebSocket] Connecting to ${wsUrl}...`);

            this.state.ws = new WebSocket(wsUrl);

            this.state.ws.onopen = () => {
                console.log('[WebSocket] ✓ Connected');
                this.state.connected = true;
                this.state.reconnectAttempts = 0;
                this.updateStatusText('Connected', 'connected');
            };

            this.state.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type !== 'pong') {
                        this.handleProgressUpdate(data);
                    }
                } catch (error) {
                    console.error('[WebSocket] Parse error:', error);
                }
            };

            this.state.ws.onerror = (error) => {
                console.error('[WebSocket] Error:', error);
            };

            this.state.ws.onclose = () => {
                console.log('[WebSocket] Disconnected');
                this.state.connected = false;
                this.scheduleReconnect();
            };

        } catch (error) {
            console.error('[WebSocket] Connection failed:', error);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (!this.config.reconnect.enabled) return;
        if (this.state.reconnectAttempts >= this.config.reconnect.maxAttempts) return;

        this.state.reconnectAttempts++;
        const delay = Math.min(
            this.config.reconnect.initialDelay * Math.pow(
                this.config.reconnect.backoffMultiplier,
                this.state.reconnectAttempts - 1
            ),
            this.config.reconnect.maxDelay
        );

        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.state.reconnectAttempts})...`);
        setTimeout(() => this.connectWebSocket(), delay);
    }

    startPolling() {
        if (!this.config.polling.enabled) return;

        console.log('[Polling] Starting HTTP polling...');
        this.state.pollingInterval = setInterval(async () => {
            try {
                const url = `${this.config.httpProtocol}//${this.config.hostname}:${this.config.loadingServerPort}/api/startup-progress`;
                const response = await fetch(url, {
                    method: 'GET',
                    cache: 'no-cache',
                    signal: AbortSignal.timeout(this.config.polling.timeout)
                });

                if (response.ok) {
                    const data = await response.json();
                    this.handleProgressUpdate(data);
                }
            } catch (error) {
                // Silently fail - WebSocket is primary
            }
        }, this.config.polling.interval);
    }

    handleProgressUpdate(data) {
        this.state.lastUpdate = Date.now();

        const { stage, message, progress, metadata } = data;

        console.log(`[Progress] ${progress}% - ${stage}: ${message}`);

        // Update target progress
        if (typeof progress === 'number' && progress >= 0 && progress <= 100) {
            this.state.targetProgress = progress;
        }

        // Update stage and message
        if (stage) this.state.stage = stage;
        if (message) this.state.message = message;

        // Update UI immediately
        this.updateUI();

        // Handle completion
        if (stage === 'complete' || progress >= 100) {
            const success = metadata?.success !== false;
            const redirectUrl = metadata?.redirect_url || `${this.config.httpProtocol}//${this.config.hostname}:${this.config.mainAppPort}`;
            this.handleCompletion(success, redirectUrl, message);
        }

        // Handle failure
        if (stage === 'failed' || metadata?.success === false) {
            this.showError(message || 'Startup failed');
        }
    }

    startSmoothProgress() {
        if (!this.config.smoothProgress.enabled) return;

        // Smooth progress animation - gradually increment to target
        this.state.smoothProgressInterval = setInterval(() => {
            if (this.state.progress < this.state.targetProgress) {
                // Increment progress smoothly
                const diff = this.state.targetProgress - this.state.progress;
                const increment = Math.max(0.5, diff / 10);  // Speed up when far from target
                this.state.progress = Math.min(
                    this.state.progress + increment,
                    this.state.targetProgress
                );
                this.updateProgressBar();
            }
        }, this.config.smoothProgress.incrementDelay);
    }

    updateUI() {
        this.updateProgressBar();

        if (this.state.message) {
            this.elements.statusMessage.textContent = this.state.message;
        }
    }

    updateProgressBar() {
        const displayProgress = Math.round(this.state.progress);
        this.elements.progressBar.style.width = `${displayProgress}%`;
        this.elements.progressPercentage.textContent = `${displayProgress}%`;

        // Add visual feedback at milestones
        if (displayProgress >= 25 && displayProgress < 50) {
            this.elements.progressBar.style.background = 'linear-gradient(90deg, #00ff41 0%, #00aa2e 100%)';
        } else if (displayProgress >= 50 && displayProgress < 75) {
            this.elements.progressBar.style.background = 'linear-gradient(90deg, #00aa2e 0%, #00ff41 100%)';
        } else if (displayProgress >= 75) {
            this.elements.progressBar.style.background = 'linear-gradient(90deg, #00ff41 0%, #00ff41 100%)';
            this.elements.progressBar.style.boxShadow = '0 0 20px rgba(0, 255, 65, 0.8)';
        }
    }

    updateStatusText(text, status) {
        if (this.elements.statusText) {
            this.elements.statusText.textContent = text;
            this.elements.statusText.className = `status-text ${status}`;
        }
    }

    handleCompletion(success, redirectUrl, message) {
        if (!success) {
            this.showError(message || 'Startup completed with errors');
            return;
        }

        console.log('[Complete] ✓ Startup successful!');

        // Stop all intervals
        this.cleanup();

        // Update UI
        this.elements.subtitle.textContent = 'SYSTEM READY';
        this.elements.statusMessage.textContent = message || 'JARVIS is online!';
        this.state.progress = 100;
        this.state.targetProgress = 100;
        this.updateProgressBar();

        // Smooth transition animation
        const container = document.querySelector('.loading-container');

        // 1. Pulse reactor
        if (this.elements.reactor) {
            this.elements.reactor.style.animation = 'pulse 0.5s ease-in-out 3';
        }

        // 2. Wait a moment
        setTimeout(() => {
            // 3. Fade out
            if (container) {
                container.style.transition = 'opacity 1s ease-out, transform 1s ease-out';
                container.style.opacity = '0';
                container.style.transform = 'scale(0.95)';
            }

            // 4. Green gradient overlay
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, #000 0%, #003300 100%);
                opacity: 0;
                transition: opacity 0.5s ease-in;
                z-index: 10000;
            `;
            document.body.appendChild(overlay);

            setTimeout(() => {
                overlay.style.opacity = '1';
            }, 10);

            // 5. Redirect
            setTimeout(() => {
                console.log(`[Redirect] → ${redirectUrl}`);
                window.location.href = redirectUrl;
            }, 1500);
        }, 1500);
    }

    showError(message) {
        console.error('[Error]', message);

        this.cleanup();

        if (this.elements.errorContainer) {
            this.elements.errorContainer.classList.add('visible');
        }
        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
        }
        if (this.elements.subtitle) {
            this.elements.subtitle.textContent = 'INITIALIZATION FAILED';
        }
        if (this.elements.reactor) {
            this.elements.reactor.style.opacity = '0.3';
        }
    }

    startHealthMonitoring() {
        setInterval(() => {
            const timeSinceUpdate = Date.now() - this.state.lastUpdate;
            const totalTime = Date.now() - this.state.startTime;

            // If no updates for 30 seconds and not at 100%, show warning
            if (timeSinceUpdate > 30000 && this.state.progress < 100) {
                console.warn('[Health] No updates for 30 seconds');

                // Try reconnecting if WebSocket is dead
                if (!this.state.connected) {
                    this.connectWebSocket();
                }
            }

            // Timeout after 5 minutes
            if (totalTime > 300000 && this.state.progress < 100) {
                this.showError('Startup timed out. Please check terminal logs and try again.');
            }
        }, 5000);
    }

    cleanup() {
        if (this.state.ws) {
            this.state.ws.close();
            this.state.ws = null;
        }
        if (this.state.pollingInterval) {
            clearInterval(this.state.pollingInterval);
            this.state.pollingInterval = null;
        }
        if (this.state.smoothProgressInterval) {
            clearInterval(this.state.smoothProgressInterval);
            this.state.smoothProgressInterval = null;
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.jarvisLoader = new JARVISLoadingManager();
    });
} else {
    window.jarvisLoader = new JARVISLoadingManager();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.jarvisLoader) {
        window.jarvisLoader.cleanup();
    }
});
