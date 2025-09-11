/**
 * JARVIS Auto Configuration Client
 * Automatically discovers and configures API endpoints
 * Include this script in your frontend to enable auto-configuration
 * 
 * Usage:
 * <script src="http://localhost:8000/static/jarvis-auto-config.js"></script>
 * 
 * Or in React/Vue/Angular:
 * import 'http://localhost:8000/static/jarvis-auto-config.js';
 */

(function() {
    'use strict';
    
    // Default configuration
    const DEFAULT_PORTS = [8000, 8010, 8080, 5000];
    const DISCOVERY_TIMEOUT = 2000; // 2 seconds
    
    // JARVIS API Configuration object
    window.JARVIS_API = {
        discovered: false,
        baseUrl: null,
        wsUrl: null,
        endpoints: {},
        
        // Auto-discovery function
        discoverBackend: async function() {
            console.log('🔍 JARVIS Auto-Config: Starting backend discovery...');
            
            // Try each port
            for (const port of DEFAULT_PORTS) {
                try {
                    const testUrl = `http://localhost:${port}/auto-config`;
                    const response = await fetch(testUrl, {
                        method: 'GET',
                        headers: {
                            'Accept': 'application/json',
                        },
                        mode: 'cors',
                        credentials: 'include',
                        signal: AbortSignal.timeout(DISCOVERY_TIMEOUT)
                    });
                    
                    if (response.ok) {
                        const config = await response.json();
                        console.log(`✅ JARVIS Backend found on port ${port}!`, config);
                        
                        // Configure API
                        this.discovered = true;
                        this.baseUrl = config.server.base_url;
                        this.wsUrl = config.server.ws_url;
                        this.endpoints = config.endpoints;
                        this.serverInfo = config.server;
                        
                        // Check for port mismatch warnings
                        const portWarning = response.headers.get('X-Port-Mismatch-Warning');
                        if (portWarning) {
                            console.warn('⚠️ Port mismatch detected:', portWarning);
                            const correctUrl = response.headers.get('X-Correct-Base-URL');
                            if (correctUrl) {
                                console.info('💡 Correct API URL:', correctUrl);
                                this.baseUrl = correctUrl;
                            }
                        }
                        
                        // Store in localStorage for faster subsequent loads
                        localStorage.setItem('jarvis_api_config', JSON.stringify({
                            baseUrl: this.baseUrl,
                            wsUrl: this.wsUrl,
                            endpoints: this.endpoints,
                            discoveredAt: new Date().toISOString()
                        }));
                        
                        return true;
                    }
                } catch (error) {
                    console.debug(`Port ${port} not responding:`, error.message);
                }
            }
            
            console.error('❌ JARVIS Backend not found on any port:', DEFAULT_PORTS);
            return false;
        },
        
        // Get configured fetch function with correct base URL
        fetch: async function(endpoint, options = {}) {
            if (!this.discovered) {
                await this.discoverBackend();
            }
            
            const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;
            
            // Add CORS headers
            const defaultHeaders = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            };
            
            return fetch(url, {
                ...options,
                headers: {
                    ...defaultHeaders,
                    ...(options.headers || {})
                },
                mode: 'cors',
                credentials: 'include'
            });
        },
        
        // Create WebSocket with correct URL
        createWebSocket: function(endpoint = '/ws') {
            if (!this.discovered) {
                throw new Error('Backend not discovered. Call discoverBackend() first.');
            }
            
            const url = endpoint.startsWith('ws') ? endpoint : `${this.wsUrl}${endpoint}`;
            return new WebSocket(url);
        },
        
        // Update configuration for specific frameworks
        updateFrameworkConfig: function() {
            // React
            if (window.React || window.process?.env?.REACT_APP_NAME) {
                console.log('🔧 Configuring for React...');
                window.process = window.process || {};
                window.process.env = window.process.env || {};
                window.process.env.REACT_APP_API_URL = this.baseUrl;
                window.process.env.REACT_APP_WS_URL = this.wsUrl;
            }
            
            // Vue
            if (window.Vue || window.__VUE__) {
                console.log('🔧 Configuring for Vue...');
                window.VUE_APP_API_URL = this.baseUrl;
                window.VUE_APP_WS_URL = this.wsUrl;
            }
            
            // Angular
            if (window.ng || window.angular) {
                console.log('🔧 Configuring for Angular...');
                window.ANGULAR_API_URL = this.baseUrl;
                window.ANGULAR_WS_URL = this.wsUrl;
            }
        }
    };
    
    // Check localStorage for cached config
    const cachedConfig = localStorage.getItem('jarvis_api_config');
    if (cachedConfig) {
        try {
            const config = JSON.parse(cachedConfig);
            const discoveredAt = new Date(config.discoveredAt);
            const age = Date.now() - discoveredAt.getTime();
            
            // Use cache if less than 1 hour old
            if (age < 3600000) {
                console.log('📦 Using cached JARVIS configuration');
                window.JARVIS_API.discovered = true;
                window.JARVIS_API.baseUrl = config.baseUrl;
                window.JARVIS_API.wsUrl = config.wsUrl;
                window.JARVIS_API.endpoints = config.endpoints;
                window.JARVIS_API.updateFrameworkConfig();
                return;
            }
        } catch (e) {
            console.error('Failed to parse cached config:', e);
        }
    }
    
    // Auto-discover on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', async () => {
            await window.JARVIS_API.discoverBackend();
            window.JARVIS_API.updateFrameworkConfig();
        });
    } else {
        // DOM already loaded
        window.JARVIS_API.discoverBackend().then(() => {
            window.JARVIS_API.updateFrameworkConfig();
        });
    }
    
    // Export for module systems
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = window.JARVIS_API;
    }
    
})();