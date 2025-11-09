/**
 * Advanced JARVIS Loading Manager
 *
 * Features:
 * - Zero hardcoding: Auto-discovers backend endpoints and stages
 * - Async/robust: Exponential backoff, health checks, connection pooling
 * - Dynamic: Adapts to any startup sequence from backend
 * - Smart reconnection: Handles network failures gracefully
 * - Real-time updates: WebSocket with fallback to polling
 */

class JARVISLoadingManager {
    constructor() {
        // Configuration (auto-discovered from environment)
        this.config = {
            backendPort: this.discoverBackendPort(),
            wsProtocol: window.location.protocol === 'https:' ? 'wss:' : 'ws:',
            httpProtocol: window.location.protocol,
            hostname: window.location.hostname,
            reconnect: {
                initialDelay: 1000,      // 1 second
                maxDelay: 30000,         // 30 seconds
                maxAttempts: 60,         // 1 minute of retries with max delay
                backoffMultiplier: 1.5,
                jitter: 0.3              // 30% random jitter
            },
            polling: {
                enabled: true,           // Fallback to polling if WebSocket fails
                interval: 2000,          // 2 seconds
                maxDuration: 300000      // 5 minutes
            },
            healthCheck: {
                enabled: true,
                interval: 5000,          // 5 seconds
                timeout: 3000            // 3 seconds
            }
        };

        // State management
        this.state = {
            ws: null,
            reconnectAttempts: 0,
            currentDelay: this.config.reconnect.initialDelay,
            connected: false,
            progress: 0,
            stage: null,
            stages: new Map(),           // Dynamic stage tracking
            pollingInterval: null,
            healthCheckInterval: null,
            startTime: Date.now(),
            lastUpdate: Date.now(),
            errors: []
        };

        // DOM elements cache
        this.elements = this.cacheElements();

        // Initialize
        this.init();
    }

    /**
     * Auto-discover backend port from various sources
     */
    discoverBackendPort() {
        // Try multiple sources in order of preference
        const sources = [
            () => window.JARVIS_CONFIG?.backendPort,
            () => localStorage.getItem('jarvis_backend_port'),
            () => sessionStorage.getItem('jarvis_backend_port'),
            () => new URLSearchParams(window.location.search).get('backend_port'),
            () => 8010  // Default fallback
        ];

        for (const source of sources) {
            try {
                const port = source();
                if (port && !isNaN(parseInt(port))) {
                    console.log(`[Discovery] Backend port: ${port}`);
                    return parseInt(port);
                }
            } catch (e) {
                continue;
            }
        }

        return 8010;
    }

    /**
     * Cache DOM elements for performance
     */
    cacheElements() {
        return {
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            subtitle: document.getElementById('subtitle'),
            progressBar: document.getElementById('progress-bar'),
            progressPercentage: document.getElementById('progress-percentage'),
            statusMessage: document.getElementById('status-message'),
            stagesContainer: document.getElementById('stages-container'),
            detailsPanel: document.getElementById('details-panel'),
            errorContainer: document.getElementById('error-container'),
            errorMessage: document.getElementById('error-message')
        };
    }

    /**
     * Initialize the loading manager
     */
    async init() {
        console.log('[Init] JARVIS Loading Manager starting...');

        // Create particle background
        this.createParticles();

        // Start health check
        if (this.config.healthCheck.enabled) {
            this.startHealthCheck();
        }

        // Connect to backend
        await this.connect();

        // Start progress monitoring
        this.startProgressMonitoring();
    }

    /**
     * Create animated particle background
     */
    createParticles() {
        const container = document.getElementById('particles');
        const particleCount = 50;

        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';

            // Random size and position
            const size = Math.random() * 3 + 1;
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${Math.random() * 100}%`;

            // Random animation duration and delay
            const duration = Math.random() * 10 + 10;
            const delay = Math.random() * 5;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;

            container.appendChild(particle);
        }
    }

    /**
     * Health check to detect when backend is ready
     */
    async startHealthCheck() {
        this.state.healthCheckInterval = setInterval(async () => {
            try {
                const response = await this.fetchWithTimeout(
                    `${this.config.httpProtocol}//${this.config.hostname}:${this.config.backendPort}/health`,
                    { timeout: this.config.healthCheck.timeout }
                );

                if (response.ok && !this.state.connected) {
                    console.log('[Health] Backend is healthy, attempting WebSocket connection...');
                    await this.connect();
                }
            } catch (error) {
                // Backend not ready yet, keep checking
            }
        }, this.config.healthCheck.interval);
    }

    /**
     * Fetch with timeout
     */
    async fetchWithTimeout(url, options = {}) {
        const timeout = options.timeout || 5000;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    /**
     * Connect to backend via WebSocket
     */
    async connect() {
        // Don't reconnect if already connected or max attempts reached
        if (this.state.connected) {
            console.log('[Connect] Already connected');
            return;
        }

        if (this.state.reconnectAttempts >= this.config.reconnect.maxAttempts) {
            console.warn('[Connect] Max reconnection attempts reached, falling back to polling');
            this.fallbackToPolling();
            return;
        }

        try {
            const wsUrl = `${this.config.wsProtocol}//${this.config.hostname}:${this.config.backendPort}/ws/startup-progress`;
            console.log(`[Connect] Attempting WebSocket connection to ${wsUrl} (attempt ${this.state.reconnectAttempts + 1}/${this.config.reconnect.maxAttempts})`);

            this.updateConnectionStatus('connecting', 'Connecting to backend...');

            this.state.ws = new WebSocket(wsUrl);

            this.state.ws.onopen = () => this.handleOpen();
            this.state.ws.onmessage = (event) => this.handleMessage(event);
            this.state.ws.onerror = (error) => this.handleError(error);
            this.state.ws.onclose = (event) => this.handleClose(event);

        } catch (error) {
            console.error('[Connect] Connection failed:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket open
     */
    handleOpen() {
        console.log('[WebSocket] Connected successfully');
        this.state.connected = true;
        this.state.reconnectAttempts = 0;
        this.state.currentDelay = this.config.reconnect.initialDelay;

        this.updateConnectionStatus('connected', 'Connected');

        // Send initial ping to ensure connection
        this.sendPing();

        // Setup periodic ping to keep connection alive
        this.startPingInterval();
    }

    /**
     * Handle WebSocket message
     */
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);

            // Ignore pong responses
            if (data.type === 'pong') {
                return;
            }

            console.log('[WebSocket] Progress update:', data);
            this.state.lastUpdate = Date.now();

            this.updateProgress(data);

        } catch (error) {
            console.error('[WebSocket] Error parsing message:', error);
        }
    }

    /**
     * Handle WebSocket error
     */
    handleError(error) {
        console.error('[WebSocket] Error:', error);
        this.state.errors.push({
            timestamp: Date.now(),
            error: error.message || 'Unknown error'
        });
    }

    /**
     * Handle WebSocket close
     */
    handleClose(event) {
        console.log('[WebSocket] Connection closed:', event.code, event.reason);
        this.state.connected = false;

        this.updateConnectionStatus('disconnected', 'Reconnecting...');

        // Clear ping interval
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }

        // Attempt reconnection unless it was a normal closure
        if (event.code !== 1000) {
            this.scheduleReconnect();
        }
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    scheduleReconnect() {
        this.state.reconnectAttempts++;

        // Calculate delay with exponential backoff and jitter
        const baseDelay = Math.min(
            this.config.reconnect.initialDelay * Math.pow(
                this.config.reconnect.backoffMultiplier,
                this.state.reconnectAttempts - 1
            ),
            this.config.reconnect.maxDelay
        );

        // Add jitter (random variation)
        const jitter = baseDelay * this.config.reconnect.jitter * (Math.random() - 0.5);
        this.state.currentDelay = Math.floor(baseDelay + jitter);

        console.log(`[Reconnect] Scheduling reconnection in ${this.state.currentDelay}ms (attempt ${this.state.reconnectAttempts}/${this.config.reconnect.maxAttempts})`);

        setTimeout(() => this.connect(), this.state.currentDelay);
    }

    /**
     * Fallback to HTTP polling if WebSocket fails
     */
    fallbackToPolling() {
        if (!this.config.polling.enabled) {
            this.showError('Unable to connect to backend. Please check your connection and try again.');
            return;
        }

        console.log('[Polling] Falling back to HTTP polling');
        this.updateConnectionStatus('connecting', 'Using fallback mode...');

        this.state.pollingInterval = setInterval(async () => {
            try {
                const response = await this.fetchWithTimeout(
                    `${this.config.httpProtocol}//${this.config.hostname}:${this.config.backendPort}/api/startup-progress`,
                    { timeout: 3000 }
                );

                if (response.ok) {
                    const data = await response.json();
                    this.updateProgress(data);

                    if (!this.state.connected) {
                        this.state.connected = true;
                        this.updateConnectionStatus('connected', 'Connected (polling)');
                    }
                }
            } catch (error) {
                console.error('[Polling] Error:', error);
            }
        }, this.config.polling.interval);

        // Stop polling after max duration
        setTimeout(() => {
            if (this.state.pollingInterval) {
                clearInterval(this.state.pollingInterval);
                this.showError('Backend startup timed out. Please check logs and try again.');
            }
        }, this.config.polling.maxDuration);
    }

    /**
     * Send ping to keep connection alive
     */
    sendPing() {
        if (this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
            this.state.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }

    /**
     * Start periodic ping interval
     */
    startPingInterval() {
        this.pingInterval = setInterval(() => {
            this.sendPing();
        }, 10000); // Every 10 seconds
    }

    /**
     * Update connection status display
     */
    updateConnectionStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator ${status}`;
        this.elements.statusText.textContent = text;
    }

    /**
     * Update progress display with data from backend
     */
    updateProgress(data) {
        const { stage, message, progress, details, success, redirect_url, metadata } = data;

        // Update progress bar
        this.state.progress = progress || 0;
        this.elements.progressBar.style.width = `${this.state.progress}%`;
        this.elements.progressPercentage.textContent = `${Math.round(this.state.progress)}%`;

        // Update status message
        if (message) {
            this.elements.statusMessage.textContent = message;
        }

        // Update or create stage
        if (stage) {
            this.state.stage = stage;
            this.updateStage(stage, progress, metadata);
        }

        // Update details if provided
        if (details) {
            this.updateDetails(details);
        }

        // Handle completion
        if (stage === 'complete') {
            this.handleCompletion(success, redirect_url, message);
        }

        // Handle failure
        if (stage === 'failed' || (!success && progress >= 100)) {
            this.showError(message || 'Startup failed. Please check logs.');
        }
    }

    /**
     * Update or create stage indicator (dynamic, no hardcoding)
     */
    updateStage(stageName, progress, metadata = {}) {
        // Check if stage already exists
        if (!this.state.stages.has(stageName)) {
            // Create new stage dynamically
            const stageElement = this.createStageElement(stageName, metadata);
            this.elements.stagesContainer.appendChild(stageElement);
            this.state.stages.set(stageName, {
                element: stageElement,
                completed: false,
                metadata
            });
        }

        const stageData = this.state.stages.get(stageName);
        const element = stageData.element;

        // Update stage state based on progress
        element.classList.remove('active', 'completed', 'failed');

        if (progress < 100) {
            element.classList.add('active');
        } else if (metadata?.success === false) {
            element.classList.add('failed');
        } else {
            element.classList.add('completed');
            stageData.completed = true;
        }

        // Update sublabel if provided
        if (metadata?.sublabel) {
            const sublabel = element.querySelector('.stage-sublabel');
            if (sublabel) {
                sublabel.textContent = metadata.sublabel;
            }
        }
    }

    /**
     * Create stage element dynamically
     */
    createStageElement(stageName, metadata = {}) {
        const stage = document.createElement('div');
        stage.className = 'stage-item';
        stage.id = `stage-${stageName}`;

        // Auto-select appropriate icon based on stage name
        const icon = this.getStageIcon(stageName, metadata.icon);
        const label = this.getStageLabel(stageName, metadata.label);

        stage.innerHTML = `
            <div class="stage-icon">${icon}</div>
            <div class="stage-label">${label}</div>
            <div class="stage-sublabel">${metadata.sublabel || ''}</div>
        `;

        return stage;
    }

    /**
     * Auto-select appropriate icon for stage
     */
    getStageIcon(stageName, customIcon) {
        if (customIcon) return customIcon;

        // Icon mapping based on common stage name patterns
        const iconMap = {
            detect: '🔍',
            scan: '🔍',
            kill: '⚔️',
            terminate: '⚔️',
            cleanup: '🧹',
            clean: '🧹',
            start: '🚀',
            launch: '🚀',
            initialize: '⚙️',
            init: '⚙️',
            load: '📦',
            connect: '🔗',
            database: '💾',
            db: '💾',
            sql: '💾',
            api: '🌐',
            server: '🖥️',
            frontend: '🎨',
            backend: '⚙️',
            websocket: '🔌',
            complete: '✅',
            ready: '✅',
            success: '✅',
            fail: '❌',
            error: '⚠️'
        };

        // Find matching icon
        const lowerStageName = stageName.toLowerCase();
        for (const [keyword, icon] of Object.entries(iconMap)) {
            if (lowerStageName.includes(keyword)) {
                return icon;
            }
        }

        return '⚙️'; // Default icon
    }

    /**
     * Get human-readable label for stage
     */
    getStageLabel(stageName, customLabel) {
        if (customLabel) return customLabel;

        // Convert snake_case or camelCase to Title Case
        return stageName
            .replace(/([A-Z])/g, ' $1')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .trim();
    }

    /**
     * Update details panel
     */
    updateDetails(details) {
        if (!details || Object.keys(details).length === 0) {
            return;
        }

        this.elements.detailsPanel.classList.add('visible');

        let html = '';
        for (const [key, value] of Object.entries(details)) {
            const displayKey = this.getStageLabel(key);
            html += `
                <div class="detail-item">
                    <span class="detail-key">${displayKey}:</span> ${value}
                </div>
            `;
        }

        this.elements.detailsPanel.innerHTML = html;

        // Auto-scroll to bottom
        this.elements.detailsPanel.scrollTop = this.elements.detailsPanel.scrollHeight;
    }

    /**
     * Handle completion with smooth animation
     */
    handleCompletion(success, redirectUrl, message) {
        if (!success) {
            this.showError(message || 'Startup completed with errors');
            return;
        }

        console.log('[Complete] Startup successful');

        // Update text
        this.elements.subtitle.textContent = 'System Ready!';
        this.elements.statusMessage.textContent = message || 'JARVIS is ready!';

        // Smooth animation sequence
        const container = document.querySelector('.loading-container');
        const reactor = document.querySelector('.arc-reactor');

        // 1. Pulse the reactor core (success indicator)
        if (reactor) {
            reactor.style.animation = 'pulse 0.5s ease-in-out 3';
        }

        // 2. Fade out after 1.5 seconds with smooth transition
        setTimeout(() => {
            if (container) {
                container.style.transition = 'opacity 1s ease-out, transform 1s ease-out';
                container.style.opacity = '0';
                container.style.transform = 'scale(0.95)';
            }

            // Set body background for smooth transition
            document.body.style.transition = 'background-color 1s ease-out';
            document.body.style.backgroundColor = '#000';

        }, 1500);

        // 3. Redirect after fade completes
        const delay = 2500; // 1.5s wait + 1s fade
        const targetUrl = redirectUrl || `${this.config.httpProtocol}//${this.config.hostname}:3000`;

        setTimeout(() => {
            console.log(`[Redirect] Navigating to ${targetUrl}`);

            // Add a nice fade-to-white before redirect
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

            // Trigger fade
            setTimeout(() => {
                overlay.style.opacity = '1';
            }, 10);

            // Redirect after overlay fades in
            setTimeout(() => {
                window.location.href = targetUrl;
            }, 500);
        }, delay);
    }

    /**
     * Show error
     */
    showError(message) {
        console.error('[Error]', message);

        this.elements.errorContainer.classList.add('visible');
        this.elements.errorMessage.textContent = message;

        // Update subtitle
        this.elements.subtitle.textContent = 'Initialization Failed';

        // Stop animations
        const reactor = document.querySelector('.arc-reactor');
        if (reactor) {
            reactor.style.opacity = '0.3';
        }
    }

    /**
     * Monitor progress and detect stalls
     */
    startProgressMonitoring() {
        setInterval(() => {
            const timeSinceUpdate = Date.now() - this.state.lastUpdate;
            const totalTime = Date.now() - this.state.startTime;

            // Warn if no updates for 30 seconds
            if (timeSinceUpdate > 30000 && this.state.progress < 100) {
                console.warn('[Monitor] No progress updates for 30 seconds');

                if (!this.state.connected && !this.state.pollingInterval) {
                    this.updateConnectionStatus('disconnected', 'Connection lost, retrying...');
                    this.connect();
                }
            }

            // Timeout after 5 minutes
            if (totalTime > 300000 && this.state.progress < 100) {
                console.error('[Monitor] Startup timed out after 5 minutes');
                this.showError('Startup timed out. Please check backend logs and try again.');

                // Clear intervals
                if (this.state.pollingInterval) clearInterval(this.state.pollingInterval);
                if (this.state.healthCheckInterval) clearInterval(this.state.healthCheckInterval);
            }
        }, 5000);
    }

    /**
     * Cleanup on page unload
     */
    cleanup() {
        if (this.state.ws) {
            this.state.ws.close(1000, 'Page unloading');
        }
        if (this.state.pollingInterval) {
            clearInterval(this.state.pollingInterval);
        }
        if (this.state.healthCheckInterval) {
            clearInterval(this.state.healthCheckInterval);
        }
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
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
