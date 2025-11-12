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

        // 🎬 EPIC CINEMATIC COMPLETION SEQUENCE 🎬
        this.playEpicCompletionAnimation(redirectUrl);
    }

    async playEpicCompletionAnimation(redirectUrl) {
        const container = document.querySelector('.loading-container');
        const reactor = this.elements.reactor;
        const progressBar = this.elements.progressBar;

        // Configuration with matrix transition
        const totalDuration = 3500; // 3.5 seconds total
        const config = {
            phases: {
                powerSurge: {
                    duration: 600,
                    rings: 3
                },
                matrix: {
                    duration: 1500  // Matrix rain effect
                },
                fade: {
                    duration: 800
                },
                totalDuration: totalDuration
            },
            reactor: {
                powerUpScale: 1.5,
                glowIntensity: 80,
                maintainBrightness: true
            },
            effects: {
                ringColor: '#00ff41',
                matrixColumns: Math.floor(window.innerWidth / 20),
                matrixSpeed: 50
            }
        };

        // === PHASE 1: REACTOR POWER SURGE ===
        console.log('[Animation] Phase 1: Reactor power surge');

        // Reactor pulse
        if (reactor) {
            reactor.style.transition = 'all 0.3s ease-out';
            reactor.style.transform = 'translate(-50%, -50%) scale(1.5)';
            reactor.style.filter = 'drop-shadow(0 0 80px rgba(0, 255, 65, 1)) brightness(2)';
            reactor.style.opacity = '1';

            // Create 3 expanding rings
            for (let i = 0; i < 3; i++) {
                setTimeout(() => {
                    this.createEnergyRing(reactor, '#00ff41', i);
                }, i * 200);
            }
        }

        // Progress bar glow
        if (progressBar) {
            progressBar.style.boxShadow = '0 0 40px rgba(0, 255, 65, 1), 0 0 80px rgba(0, 255, 65, 0.8)';
            progressBar.style.background = 'linear-gradient(90deg, #00ff41 0%, #00ff88 100%)';
        }

        await this.sleep(config.phases.powerSurge.duration);

        // === PHASE 2: MATRIX TRANSITION ===
        console.log('[Animation] Phase 2: Matrix code rain');

        // Start fading out the container and scale reactor
        if (container) {
            container.style.transition = 'opacity 1s ease-out';
            container.style.opacity = '0';
        }

        if (reactor) {
            reactor.style.transition = 'all 1s ease-out';
            reactor.style.transform = 'translate(-50%, -50%) scale(2)';
            reactor.style.opacity = '0';
        }

        // Create matrix rain canvas
        const matrixCanvas = this.createMatrixCanvas();
        matrixCanvas.style.opacity = '1';

        // Start matrix animation
        const matrixInterval = this.startMatrixRain(matrixCanvas, config.effects.matrixColumns);

        // Wait for matrix effect
        await this.sleep(config.phases.matrix.duration);

        // Stop matrix animation
        clearInterval(matrixInterval);

        // === PHASE 3: FADE TO BLACK ===
        console.log('[Animation] Phase 3: Final fade');

        // Fade out matrix
        matrixCanvas.style.transition = `opacity ${config.phases.fade.duration / 1000}s ease-out`;
        matrixCanvas.style.opacity = '0';

        // Create black overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #000000;
            opacity: 0;
            transition: opacity ${config.phases.fade.duration / 1000}s ease-in;
            z-index: 10001;
        `;
        document.body.appendChild(overlay);

        // Trigger fade
        setTimeout(() => {
            overlay.style.opacity = '1';
        }, 10);

        // Wait for fade to complete
        await this.sleep(config.phases.fade.duration);

        // === PHASE 4: NAVIGATE TO MAIN PAGE ===
        console.log(`[Transition] Navigating to ${redirectUrl}`);

        // Clean navigation
        window.location.href = redirectUrl;
    }

    // === HELPER METHODS FOR DYNAMIC EFFECTS ===

    createEnergyRing(reactor, color, index) {
        const ring = document.createElement('div');
        ring.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 300px;
            height: 300px;
            border: 3px solid ${color};
            border-radius: 50%;
            opacity: 1;
            animation: expandRing 1s ease-out forwards;
            pointer-events: none;
        `;
        reactor.parentElement.appendChild(ring);
        setTimeout(() => ring.remove(), 1000);
    }

    createHolographicScan(height, duration) {
        const scanLine = document.createElement('div');
        scanLine.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: ${height}px;
            background: linear-gradient(90deg,
                transparent 0%,
                #00ff41 50%,
                transparent 100%);
            box-shadow: 0 0 20px #00ff41;
            opacity: 1;
            animation: scanDown ${duration / 1000}s ease-in-out;
            z-index: 10000;
        `;
        document.body.appendChild(scanLine);
        setTimeout(() => scanLine.remove(), duration);
    }

    preloadMainPage(redirectUrl, config) {
        return new Promise((resolve) => {
            console.log('[Preload] Starting iframe preload in background...');

            // Create black background layer to prevent white flash
            const blackBackground = document.createElement('div');
            blackBackground.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: #000000;
                z-index: 9998;
            `;
            document.body.insertBefore(blackBackground, document.body.firstChild);

            // Preload main page in hidden iframe
            const iframe = document.createElement('iframe');
            iframe.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
                opacity: 0;
                z-index: 10001;
                background: #000000;
            `;
            iframe.src = redirectUrl;
            document.body.appendChild(iframe);

            // Show loading progress if enabled
            if (config.showProgress) {
                this.elements.statusMessage.textContent = 'Loading main interface...';
            }

            // Wait for iframe to load
            iframe.onload = () => {
                console.log('[Preload] ✓ Main page loaded in iframe');
                resolve(iframe);
            };

            // Fallback timeout
            setTimeout(() => {
                console.warn('[Preload] Timeout - proceeding anyway');
                resolve(iframe);
            }, 3000);
        });
    }

    createTransitionOverlay(fadeDuration = 1) {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center,
                #001a00 0%,
                #003300 50%,
                #000000 100%);
            opacity: 0;
            transition: opacity ${fadeDuration}s ease-in;
            z-index: 9999;
        `;
        document.body.appendChild(overlay);
        return overlay;
    }

    createMatrixCanvas() {
        const matrixCanvas = document.createElement('canvas');
        matrixCanvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10000;
            opacity: 0.3;
            pointer-events: none;
        `;
        document.body.appendChild(matrixCanvas);
        return matrixCanvas;
    }

    injectAnimationStyles(config) {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes scanDown {
                0% { top: 0; opacity: 1; }
                100% { top: 100%; opacity: 0; }
            }
            @keyframes expandRing {
                0% { transform: translate(-50%, -50%) scale(0); opacity: 1; }
                100% { transform: translate(-50%, -50%) scale(3); opacity: 0; }
            }
            @keyframes reactorBreathing {
                0%, 100% {
                    transform: scale(${config.reactor.powerUpScale});
                    filter: drop-shadow(0 0 ${config.reactor.glowIntensity}px rgba(0, 255, 65, 1)) brightness(2);
                }
                50% {
                    transform: scale(${config.reactor.powerUpScale + 0.1});
                    filter: drop-shadow(0 0 ${config.reactor.glowIntensity + 20}px rgba(0, 255, 65, 1)) brightness(2.2);
                }
            }
            @keyframes fadeOutUp {
                0% { opacity: 1; transform: translateY(0) scale(1); }
                100% { opacity: 0; transform: translateY(-50px) scale(0.9); }
            }
        `;
        document.head.appendChild(style);
    }

    async playVoiceAnnouncement() {
        try {
            // Use backend API to play voice with macOS Daniel voice (same as JARVIS)
            console.log('[Voice] Requesting backend voice announcement...');

            // Visual feedback during speech
            this.elements.statusMessage.style.animation = 'pulse 0.5s ease-in-out infinite';

            const response = await fetch(
                `${this.config.httpProtocol}//${this.config.hostname}:${this.config.mainAppPort}/api/startup-voice/announce-online`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    signal: AbortSignal.timeout(5000)
                }
            );

            if (response.ok) {
                const data = await response.json();
                console.log(`[Voice] ✓ ${data.voice} speaking: "${data.text}"`);
            } else {
                console.warn('[Voice] Backend voice API failed, falling back to browser TTS');
                // Fallback to browser's speech synthesis
                const utterance = new SpeechSynthesisUtterance('JARVIS is online. Ready for your command.');
                utterance.rate = 0.95;
                speechSynthesis.speak(utterance);
            }

            // Wait for speech to complete (approximate duration)
            await this.sleep(2500);

            // Stop visual feedback
            this.elements.statusMessage.style.animation = '';

        } catch (error) {
            console.error('[Voice] Error:', error);

            // Fallback: Try browser speech synthesis
            try {
                const utterance = new SpeechSynthesisUtterance('JARVIS is online. Ready for your command.');
                utterance.rate = 0.95;
                speechSynthesis.speak(utterance);
                await this.sleep(2500);
            } catch (fallbackError) {
                console.error('[Voice] Fallback also failed:', fallbackError);
            }

            // Continue animation even if voice fails completely
            this.elements.statusMessage.style.animation = '';
        }
    }

    createParticleBurst(centerElement, velocityMin = 200, velocityMax = 300, particleCount = 30) {
        const centerRect = centerElement.getBoundingClientRect();
        const centerX = centerRect.left + centerRect.width / 2;
        const centerY = centerRect.top + centerRect.height / 2;

        // Create dynamic particle burst
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            const angle = (Math.PI * 2 * i) / particleCount;
            const velocity = velocityMin + Math.random() * (velocityMax - velocityMin);
            const endX = Math.cos(angle) * velocity;
            const endY = Math.sin(angle) * velocity;

            // Random particle size for variety
            const size = 3 + Math.random() * 3;

            particle.style.cssText = `
                position: fixed;
                left: ${centerX}px;
                top: ${centerY}px;
                width: ${size}px;
                height: ${size}px;
                background: #00ff41;
                border-radius: 50%;
                box-shadow: 0 0 10px #00ff41, 0 0 20px rgba(0, 255, 65, 0.5);
                transform: translate(-50%, -50%);
                animation: particleBurst${i} 1s ease-out forwards;
                pointer-events: none;
                z-index: 10001;
            `;

            // Create unique animation for each particle
            const style = document.createElement('style');
            style.textContent = `
                @keyframes particleBurst${i} {
                    0% {
                        transform: translate(-50%, -50%) translate(0, 0) scale(1);
                        opacity: 1;
                    }
                    100% {
                        transform: translate(-50%, -50%) translate(${endX}px, ${endY}px) scale(0);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);

            document.body.appendChild(particle);
            setTimeout(() => {
                particle.remove();
                style.remove();
            }, 1000);
        }
    }

    startMatrixRain(canvas, columnCount) {
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const columns = columnCount || Math.floor(canvas.width / 20);
        const drops = Array(columns).fill(0).map(() => Math.random() * -100);

        const matrix = 'JARVIS01アイウエオカキクケコサシスセソ';
        const fontSize = 16;
        const columnWidth = canvas.width / columns;
        ctx.font = `${fontSize}px monospace`;

        const draw = () => {
            // Fade effect for trail
            ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw matrix characters
            drops.forEach((y, i) => {
                const text = matrix[Math.floor(Math.random() * matrix.length)];
                const x = i * columnWidth;

                // Brighter color for leading character
                ctx.fillStyle = '#00ff41';
                ctx.fillText(text, x, y * fontSize);

                // Dimmer trail
                ctx.fillStyle = 'rgba(0, 255, 65, 0.5)';
                if (y > 1) {
                    const trailText = matrix[Math.floor(Math.random() * matrix.length)];
                    ctx.fillText(trailText, x, (y - 1) * fontSize);
                }

                // Reset drop to top with random chance
                if (y * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            });
        };

        return setInterval(draw, 50);
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
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

            // Timeout after 10 minutes (allows time for minimal-to-full mode upgrade)
            if (totalTime > 600000 && this.state.progress < 100) {
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
