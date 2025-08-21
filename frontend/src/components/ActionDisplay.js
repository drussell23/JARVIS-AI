import React, { useState, useEffect } from 'react';
import './ActionDisplay.css';

const ActionDisplay = ({ actions, onApprove, onReject }) => {
  const [pendingActions, setPendingActions] = useState([]);
  const [executedActions, setExecutedActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);

  useEffect(() => {
    // Separate pending and executed actions
    const pending = actions.filter(a => a.status === 'pending' || a.status === 'queued');
    const executed = actions.filter(a => a.status === 'executed' || a.status === 'completed');
    
    setPendingActions(pending);
    setExecutedActions(executed.slice(-5)); // Keep last 5 executed
  }, [actions]);

  const getPriorityClass = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'critical': return 'priority-critical';
      case 'high': return 'priority-high';
      case 'medium': return 'priority-medium';
      case 'low': return 'priority-low';
      default: return 'priority-medium';
    }
  };

  const getCategoryIcon = (category) => {
    switch (category?.toLowerCase()) {
      case 'communication': return '💬';
      case 'notification': return '🔔';
      case 'security': return '🔒';
      case 'workflow': return '⚙️';
      case 'organization': return '📁';
      case 'maintenance': return '🔧';
      case 'calendar': return '📅';
      default: return '🤖';
    }
  };

  const handleApprove = (action) => {
    if (onApprove) {
      onApprove(action);
    }
    setSelectedAction(null);
  };

  const handleReject = (action) => {
    if (onReject) {
      onReject(action);
    }
    setSelectedAction(null);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <div className="action-display">
      <div className="action-header">
        <h3>Autonomous Actions</h3>
        <div className="action-stats">
          <span className="stat-pending">{pendingActions.length} pending</span>
          <span className="stat-executed">{executedActions.length} executed</span>
        </div>
      </div>

      {/* Pending Actions */}
      {pendingActions.length > 0 && (
        <div className="pending-actions">
          <h4>Pending Approval</h4>
          <div className="action-list">
            {pendingActions.map((action, index) => (
              <div
                key={action.id || index}
                className={`action-item ${getPriorityClass(action.priority)} ${
                  selectedAction?.id === action.id ? 'selected' : ''
                }`}
                onClick={() => setSelectedAction(action)}
              >
                <div className="action-icon">{getCategoryIcon(action.category)}</div>
                <div className="action-content">
                  <div className="action-type">{action.action_type || action.type}</div>
                  <div className="action-target">Target: {action.target}</div>
                  {action.reasoning && (
                    <div className="action-reasoning">{action.reasoning}</div>
                  )}
                  <div className="action-meta">
                    <span className="confidence">
                      Confidence: {Math.round((action.confidence || 0) * 100)}%
                    </span>
                    {action.requires_permission && (
                      <span className="permission-required">🔐 Permission Required</span>
                    )}
                  </div>
                </div>
                {selectedAction?.id === action.id && (
                  <div className="action-controls">
                    <button
                      className="btn-approve"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleApprove(action);
                      }}
                    >
                      Approve
                    </button>
                    <button
                      className="btn-reject"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReject(action);
                      }}
                    >
                      Reject
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Executed Actions */}
      {executedActions.length > 0 && (
        <div className="executed-actions">
          <h4>Recently Executed</h4>
          <div className="action-history">
            {executedActions.map((action, index) => (
              <div key={action.id || index} className="history-item">
                <div className="history-icon">{getCategoryIcon(action.category)}</div>
                <div className="history-content">
                  <span className="history-type">{action.action_type || action.type}</span>
                  <span className="history-target">on {action.target}</span>
                  <span className="history-time">{formatTimestamp(action.timestamp)}</span>
                </div>
                <div className={`history-status ${action.success ? 'success' : 'failed'}`}>
                  {action.success ? '✓' : '✗'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {pendingActions.length === 0 && executedActions.length === 0 && (
        <div className="action-empty">
          <p>No autonomous actions at the moment</p>
          <p className="empty-hint">JARVIS will suggest actions based on your workspace activity</p>
        </div>
      )}
    </div>
  );
};

export default ActionDisplay;