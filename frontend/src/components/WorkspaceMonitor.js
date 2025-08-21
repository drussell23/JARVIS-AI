import React, { useState, useEffect } from 'react';
import './WorkspaceMonitor.css';

const WorkspaceMonitor = ({ visionData, autonomousMode }) => {
    const [notifications, setNotifications] = useState([]);
    const [windows, setWindows] = useState([]);
    const [actions, setActions] = useState([]);
    const [stats, setStats] = useState({
        window_count: 0,
        notification_count: 0,
        action_count: 0
    });
    
    useEffect(() => {
        if (visionData) {
            // Handle workspace update structure
            if (visionData.workspace) {
                const workspace = visionData.workspace;
                
                // Extract notifications
                if (workspace.notifications) {
                    setNotifications(workspace.notifications);
                }
                
                // Extract windows
                if (workspace.windows) {
                    setWindows(workspace.windows);
                }
                
                // Update stats
                setStats({
                    window_count: workspace.window_count || 0,
                    notification_count: workspace.notification_details ? 
                        Object.values(workspace.notification_details).reduce((a, b) => a + b, 0) : 0,
                    action_count: workspace.actionable_items || 0
                });
            }
            
            // Handle autonomous actions
            if (visionData.autonomous_actions) {
                setActions(visionData.autonomous_actions);
            }
        }
    }, [visionData]);
    
    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'CRITICAL': return '#ff4444';
            case 'HIGH': return '#ff8844';
            case 'MEDIUM': return '#ffaa44';
            case 'LOW': return '#44aa44';
            default: return '#888888';
        }
    };
    
    const getConfidenceBar = (confidence) => {
        const percentage = Math.round(confidence * 100);
        return (
            <div className="confidence-bar">
                <div 
                    className="confidence-fill" 
                    style={{ width: `${percentage}%` }}
                />
                <span className="confidence-text">{percentage}%</span>
            </div>
        );
    };
    
    return (
        <div className="workspace-monitor">
            <div className="monitor-header">
                <h3>
                    <span className="vision-icon">👁️</span>
                    Workspace Intelligence
                </h3>
                <div className="monitor-status">
                    {autonomousMode ? (
                        <span className="status-active">● Autonomous Mode Active</span>
                    ) : (
                        <span className="status-inactive">● Manual Mode</span>
                    )}
                </div>
            </div>
            
            <div className="monitor-stats">
                <div className="stat-item">
                    <span className="stat-value">{stats.window_count}</span>
                    <span className="stat-label">Windows</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value">{stats.notification_count}</span>
                    <span className="stat-label">Notifications</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value">{stats.action_count}</span>
                    <span className="stat-label">Actions</span>
                </div>
            </div>
            
            {notifications.length > 0 && (
                <div className="monitor-section">
                    <h4>🔔 Active Notifications</h4>
                    <div className="notification-list">
                        {notifications.slice(0, 5).map((notif, index) => (
                            <div key={index} className="notification-item">
                                <span className="notification-text">{notif}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {windows.length > 0 && (
                <div className="monitor-section">
                    <h4>🖥️ Active Windows</h4>
                    <div className="window-list">
                        {windows.slice(0, 5).map((window, index) => (
                            <div key={window.id || index} className="window-item">
                                <span className="window-app">{window.app || window.application}</span>
                                <span className="window-title">{window.title || window.window_title || 'Untitled'}</span>
                                {window.focused && <span className="window-focused">●</span>}
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {autonomousMode && actions.length > 0 && (
                <div className="monitor-section">
                    <h4>🤖 Autonomous Actions</h4>
                    <div className="action-list">
                        {actions.map((action, index) => (
                            <div key={index} className="action-item">
                                <div className="action-header">
                                    <span className="action-type">{action.type}</span>
                                    <span 
                                        className="action-priority" 
                                        style={{ color: getPriorityColor(action.priority) }}
                                    >
                                        {action.priority}
                                    </span>
                                </div>
                                <div className="action-target">Target: {action.target}</div>
                                <div className="action-reasoning">{action.reasoning}</div>
                                {getConfidenceBar(action.confidence)}
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {!visionData && (
                <div className="monitor-empty">
                    <p>Waiting for workspace data...</p>
                    <p className="monitor-hint">Enable autonomous mode to start monitoring</p>
                </div>
            )}
        </div>
    );
};

export default WorkspaceMonitor;