/**
 * Vision Connection Handler for JARVIS
 * Enables real-time workspace monitoring and autonomous actions
 */

class VisionConnection {
    constructor(onWorkspaceUpdate, onActionExecuted) {
        this.visionSocket = null;
        this.workspaceData = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        
        // Callbacks
        this.onWorkspaceUpdate = onWorkspaceUpdate || (() => {});
        this.onActionExecuted = onActionExecuted || (() => {});
        
        // Monitoring state
        this.monitoringActive = false;
        this.autonomousMode = false;
    }
    
    async connect() {
        try {
            console.log('🔌 Connecting to Vision WebSocket...');
            
            const wsUrl = `ws://localhost:8000/vision/ws/vision`;
            this.visionSocket = new WebSocket(wsUrl);
            
            this.visionSocket.onopen = () => {
                console.log('✅ Vision WebSocket connected!');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                
                // Request initial workspace analysis
                this.requestWorkspaceAnalysis();
            };
            
            this.visionSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleVisionMessage(data);
                } catch (error) {
                    console.error('Error parsing vision message:', error);
                }
            };
            
            this.visionSocket.onerror = (error) => {
                console.error('❌ Vision WebSocket error:', error);
            };
            
            this.visionSocket.onclose = () => {
                console.log('🔌 Vision WebSocket disconnected');
                this.isConnected = false;
                this.attemptReconnect();
            };
            
        } catch (error) {
            console.error('Failed to connect to Vision WebSocket:', error);
            this.attemptReconnect();
        }
    }
    
    handleVisionMessage(data) {
        console.log('👁️ Vision Update:', data.type);
        
        switch (data.type) {
            case 'initial_state':
                this.handleInitialState(data);
                break;
                
            case 'workspace_update':
                this.handleWorkspaceUpdate(data);
                break;
                
            case 'action_executed':
                this.handleActionExecuted(data);
                break;
                
            case 'workspace_analysis':
                this.handleWorkspaceAnalysis(data);
                break;
                
            case 'config_updated':
                console.log('⚙️ Config updated:', data);
                break;
                
            case 'error':
                console.error('Vision error:', data.message);
                break;
                
            default:
                console.log('Unknown vision message type:', data.type);
        }
    }
    
    handleInitialState(data) {
        console.log('📊 Initial workspace state:', data.workspace);
        this.monitoringActive = data.monitoring_active;
        this.autonomousMode = data.autonomous_mode;
        
        // Update UI with initial state
        this.onWorkspaceUpdate({
            type: 'initial',
            workspace: data.workspace,
            timestamp: data.timestamp
        });
    }
    
    handleWorkspaceUpdate(data) {
        console.log(`🔄 Workspace update: ${data.windows.length} windows, ${data.notifications.length} notifications`);
        
        // Store latest workspace data
        this.workspaceData = data;
        
        // Notify UI of changes
        this.onWorkspaceUpdate({
            type: 'update',
            windows: data.windows,
            notifications: data.notifications,
            suggestions: data.suggestions,
            autonomousActions: data.autonomous_actions,
            stats: data.stats,
            timestamp: data.timestamp
        });
        
        // If we have notifications, announce them
        if (data.notifications.length > 0 && this.autonomousMode) {
            this.announceNotifications(data.notifications);
        }
    }
    
    handleActionExecuted(data) {
        console.log('⚡ Autonomous action executed:', data.action);
        
        // Notify UI of executed action
        this.onActionExecuted({
            action: data.action,
            timestamp: data.timestamp
        });
    }
    
    handleWorkspaceAnalysis(data) {
        console.log('🔍 Workspace analysis received:', data.analysis);
        
        this.onWorkspaceUpdate({
            type: 'analysis',
            analysis: data.analysis,
            timestamp: data.timestamp
        });
    }
    
    announceNotifications(notifications) {
        // Create a summary of notifications
        const summary = notifications.slice(0, 3).join(', ');
        const message = `I've detected ${notifications.length} notification${notifications.length > 1 ? 's' : ''}: ${summary}`;
        
        // Use speech synthesis to announce
        if (window.speechSynthesis) {
            const utterance = new SpeechSynthesisUtterance(message);
            window.speechSynthesis.speak(utterance);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('❌ Max reconnection attempts reached. Vision system offline.');
        }
    }
    
    requestWorkspaceAnalysis() {
        if (this.isConnected && this.visionSocket.readyState === WebSocket.OPEN) {
            this.visionSocket.send(JSON.stringify({
                type: 'request_workspace_analysis'
            }));
        }
    }
    
    setMonitoringInterval(interval) {
        if (this.isConnected && this.visionSocket.readyState === WebSocket.OPEN) {
            this.visionSocket.send(JSON.stringify({
                type: 'set_monitoring_interval',
                interval: interval
            }));
        }
    }
    
    executeAction(action) {
        if (this.isConnected && this.visionSocket.readyState === WebSocket.OPEN) {
            this.visionSocket.send(JSON.stringify({
                type: 'execute_action',
                action: action
            }));
        }
    }
    
    disconnect() {
        if (this.visionSocket) {
            this.visionSocket.close();
            this.visionSocket = null;
            this.isConnected = false;
        }
    }
    
    isConnected() {
        return this.isConnected && this.visionSocket && this.visionSocket.readyState === WebSocket.OPEN;
    }
    
    getWorkspaceStats() {
        if (this.workspaceData && this.workspaceData.stats) {
            return this.workspaceData.stats;
        }
        return {
            window_count: 0,
            notification_count: 0,
            action_count: 0
        };
    }
    
    getLatestNotifications() {
        if (this.workspaceData && this.workspaceData.notifications) {
            return this.workspaceData.notifications;
        }
        return [];
    }
    
    getAutonomousActions() {
        if (this.workspaceData && this.workspaceData.autonomous_actions) {
            return this.workspaceData.autonomous_actions;
        }
        return [];
    }
}

export default VisionConnection;