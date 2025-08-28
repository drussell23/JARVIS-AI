/**
 * TypeScript-Python Bridge for WebSocket Communication
 * Provides seamless integration between TypeScript frontend and Python backend
 */
interface BridgeConfig {
    pythonEndpoint?: string;
    typeValidation?: boolean;
    messageTransformation?: boolean;
    errorHandling?: 'retry' | 'fallback' | 'throw';
}
/**
 * Bridge between TypeScript WebSocket client and Python backend
 */
export declare class WebSocketBridge {
    private client;
    private pythonTypes;
    private transformers;
    private correlationMap;
    private config;
    constructor(config?: BridgeConfig);
    /**
     * Setup bridge handlers and transformations
     */
    private setupBridge;
    /**
     * Register Python type mappings for proper serialization
     */
    private registerPythonTypes;
    /**
     * Setup message transformers for Python<->TypeScript conversion
     */
    private setupTransformers;
    /**
     * Handle incoming Python messages with type conversion
     */
    private handlePythonMessage;
    /**
     * Transform Python message to TypeScript-friendly format
     */
    private transformPythonMessage;
    /**
     * Deep transform object properties
     */
    private deepTransform;
    /**
     * Call a Python function via WebSocket
     */
    callPythonFunction(module: string, functionName: string, args?: any[], kwargs?: any): Promise<any>;
    /**
     * Subscribe to Python events
     */
    subscribeToPythonEvent(eventType: string, handler: Function): void;
    /**
     * Execute Python code dynamically
     */
    executePythonCode(code: string, context?: any): Promise<any>;
    /**
     * Connect to Python backend with specific capability
     */
    connectToPython(capability?: string): Promise<void>;
    /**
     * Send raw message to Python
     */
    sendToPython(message: any): Promise<void>;
    /**
     * Transform TypeScript message to Python format
     */
    private transformToPythonFormat;
    /**
     * Deep transform to Python format
     */
    private deepTransformToPython;
    /**
     * Generate unique correlation ID
     */
    private generateCorrelationId;
    /**
     * Get bridge statistics
     */
    getStats(): any;
    /**
     * Cleanup
     */
    destroy(): void;
}
export declare function getWebSocketBridge(): WebSocketBridge;
export default WebSocketBridge;
//# sourceMappingURL=WebSocketBridge.d.ts.map