"use strict";
/**
 * TypeScript-Python Bridge for WebSocket Communication
 * Provides seamless integration between TypeScript frontend and Python backend
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebSocketBridge = void 0;
exports.getWebSocketBridge = getWebSocketBridge;
const DynamicWebSocketClient_1 = require("../websocket/DynamicWebSocketClient");
/**
 * Bridge between TypeScript WebSocket client and Python backend
 */
class WebSocketBridge {
    constructor(config = {}) {
        this.pythonTypes = new Map();
        this.transformers = new Map();
        this.correlationMap = new Map();
        this.config = {
            typeValidation: true,
            messageTransformation: true,
            errorHandling: 'retry',
            ...config
        };
        this.client = new DynamicWebSocketClient_1.DynamicWebSocketClient({
            autoDiscover: true,
            messageValidation: this.config.typeValidation
        });
        this.setupBridge();
    }
    /**
     * Setup bridge handlers and transformations
     */
    setupBridge() {
        // Register Python type mappings
        this.registerPythonTypes();
        // Setup message transformers
        this.setupTransformers();
        // Handle all incoming messages
        this.client.on('*', (message) => {
            this.handlePythonMessage(message);
        });
    }
    /**
     * Register Python type mappings for proper serialization
     */
    registerPythonTypes() {
        // Map Python types to TypeScript types
        this.pythonTypes.set('Dict[str, Any]', Object);
        this.pythonTypes.set('List[Any]', Array);
        this.pythonTypes.set('Optional[str]', String);
        this.pythonTypes.set('bool', Boolean);
        this.pythonTypes.set('int', Number);
        this.pythonTypes.set('float', Number);
        this.pythonTypes.set('str', String);
        this.pythonTypes.set('datetime', Date);
        this.pythonTypes.set('VisionActionResult', {
            success: Boolean,
            description: String,
            data: Object,
            error: String,
            confidence: Number
        });
    }
    /**
     * Setup message transformers for Python<->TypeScript conversion
     */
    setupTransformers() {
        // Transform datetime strings to Date objects
        this.transformers.set('datetime', (value) => {
            return new Date(value);
        });
        // Transform numpy arrays to regular arrays
        this.transformers.set('ndarray', (value) => {
            if (value.__type__ === 'ndarray' && value.data) {
                return value.data;
            }
            return value;
        });
        // Transform PIL Images to base64
        this.transformers.set('PIL.Image', (value) => {
            if (value.__type__ === 'PIL.Image' && value.base64) {
                return `data:image/png;base64,${value.base64}`;
            }
            return value;
        });
    }
    /**
     * Handle incoming Python messages with type conversion
     */
    handlePythonMessage(message) {
        if (this.config.messageTransformation) {
            message = this.transformPythonMessage(message);
        }
        // Handle correlation if present
        if (message.correlation_id && this.correlationMap.has(message.correlation_id)) {
            const handler = this.correlationMap.get(message.correlation_id);
            handler(message);
            this.correlationMap.delete(message.correlation_id);
        }
    }
    /**
     * Transform Python message to TypeScript-friendly format
     */
    transformPythonMessage(message) {
        const transformed = { ...message };
        // Recursively transform the message
        this.deepTransform(transformed);
        return transformed;
    }
    /**
     * Deep transform object properties
     */
    deepTransform(obj) {
        if (obj === null || typeof obj !== 'object')
            return;
        for (const key in obj) {
            const value = obj[key];
            // Check for Python type indicators
            if (value && typeof value === 'object' && value.__type__) {
                const transformer = this.transformers.get(value.__type__);
                if (transformer) {
                    obj[key] = transformer(value);
                }
            }
            else if (typeof value === 'string') {
                // Check if it's a datetime string
                if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value)) {
                    obj[key] = new Date(value);
                }
            }
            else if (typeof value === 'object') {
                this.deepTransform(value);
            }
        }
    }
    /**
     * Call a Python function via WebSocket
     */
    async callPythonFunction(module, functionName, args = [], kwargs = {}) {
        const correlationId = this.generateCorrelationId();
        const message = {
            type: 'python_function_call',
            correlation_id: correlationId,
            module,
            function: functionName,
            args,
            kwargs,
            timestamp: new Date().toISOString()
        };
        // Send the message
        await this.client.send(message);
        // Wait for response
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.correlationMap.delete(correlationId);
                reject(new Error('Python function call timeout'));
            }, 30000);
            this.correlationMap.set(correlationId, (response) => {
                clearTimeout(timeout);
                if (response.error) {
                    reject(new Error(response.error));
                }
                else {
                    resolve(response.result);
                }
            });
        });
    }
    /**
     * Subscribe to Python events
     */
    subscribeToPythonEvent(eventType, handler) {
        this.client.on(`python_event_${eventType}`, (event) => {
            handler(this.transformPythonMessage(event));
        });
    }
    /**
     * Execute Python code dynamically
     */
    async executePythonCode(code, context = {}) {
        return this.callPythonFunction('__main__', 'exec', [code], { context });
    }
    /**
     * Connect to Python backend with specific capability
     */
    async connectToPython(capability) {
        await this.client.connect(capability || 'python');
    }
    /**
     * Send raw message to Python
     */
    async sendToPython(message) {
        // Transform TypeScript types to Python-compatible format
        const pythonMessage = this.transformToPythonFormat(message);
        await this.client.send(pythonMessage);
    }
    /**
     * Transform TypeScript message to Python format
     */
    transformToPythonFormat(message) {
        const transformed = { ...message };
        // Add Python-specific metadata
        transformed.__from_typescript__ = true;
        transformed.__timestamp__ = new Date().toISOString();
        // Transform specific types
        this.deepTransformToPython(transformed);
        return transformed;
    }
    /**
     * Deep transform to Python format
     */
    deepTransformToPython(obj) {
        if (obj === null || typeof obj !== 'object')
            return;
        for (const key in obj) {
            const value = obj[key];
            if (value instanceof Date) {
                obj[key] = value.toISOString();
            }
            else if (value instanceof RegExp) {
                obj[key] = {
                    __type__: 'regex',
                    pattern: value.source,
                    flags: value.flags
                };
            }
            else if (ArrayBuffer.isView(value)) {
                obj[key] = {
                    __type__: 'buffer',
                    data: Array.from(value)
                };
            }
            else if (typeof value === 'object') {
                this.deepTransformToPython(value);
            }
        }
    }
    /**
     * Generate unique correlation ID
     */
    generateCorrelationId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
    /**
     * Get bridge statistics
     */
    getStats() {
        return {
            clientStats: this.client.getStats(),
            pendingCorrelations: this.correlationMap.size,
            registeredTypes: Array.from(this.pythonTypes.keys()),
            transformers: Array.from(this.transformers.keys())
        };
    }
    /**
     * Cleanup
     */
    destroy() {
        this.correlationMap.clear();
        this.client.destroy();
    }
}
exports.WebSocketBridge = WebSocketBridge;
// Singleton instance for easy access
let bridgeInstance = null;
function getWebSocketBridge() {
    if (!bridgeInstance) {
        bridgeInstance = new WebSocketBridge();
    }
    return bridgeInstance;
}
// Export for JavaScript usage
exports.default = WebSocketBridge;
//# sourceMappingURL=WebSocketBridge.js.map