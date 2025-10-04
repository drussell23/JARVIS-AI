import React, { useState, useEffect } from 'react';
import adaptiveVoiceDetection from '../utils/AdaptiveVoiceDetection';
import './VoiceStatsDisplay.css';

const VoiceStatsDisplay = ({ show }) => {
  const [stats, setStats] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!show) return;

    // Update stats every 2 seconds
    const interval = setInterval(() => {
      const currentStats = adaptiveVoiceDetection.getStats();
      setStats(currentStats);
    }, 2000);

    // Get initial stats
    setStats(adaptiveVoiceDetection.getStats());

    return () => clearInterval(interval);
  }, [show]);

  if (!show || !stats) return null;

  return (
    <div className={`voice-stats-container ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="voice-stats-header" onClick={() => setExpanded(!expanded)}>
        <span className="voice-stats-icon">🧠</span>
        <span className="voice-stats-title">Adaptive Voice Learning</span>
        <span className="voice-stats-toggle">{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div className="voice-stats-content">
          <div className="voice-stats-grid">
            <div className="voice-stat-item">
              <div className="voice-stat-label">Success Rate</div>
              <div className="voice-stat-value success-rate">{stats.successRate}</div>
              <div className="voice-stat-bar">
                <div
                  className="voice-stat-bar-fill success"
                  style={{ width: stats.successRate }}
                />
              </div>
            </div>

            <div className="voice-stat-item">
              <div className="voice-stat-label">Commands</div>
              <div className="voice-stat-value">{stats.totalCommands}</div>
            </div>

            <div className="voice-stat-item">
              <div className="voice-stat-label">Avg Confidence</div>
              <div className="voice-stat-value">{stats.averageConfidence}</div>
              <div className="voice-stat-bar">
                <div
                  className="voice-stat-bar-fill confidence"
                  style={{ width: stats.averageConfidence }}
                />
              </div>
            </div>

            <div className="voice-stat-item">
              <div className="voice-stat-label">Current Threshold</div>
              <div className="voice-stat-value threshold">{stats.currentThreshold}</div>
              <div className="voice-stat-hint">
                {parseFloat(stats.adaptiveBonus) < 0 ? '⚡ More Aggressive' :
                 parseFloat(stats.adaptiveBonus) > 0 ? '🛡️ More Conservative' :
                 '⚖️ Balanced'}
              </div>
            </div>

            <div className="voice-stat-item">
              <div className="voice-stat-label">Learned Phrases</div>
              <div className="voice-stat-value">{stats.learnedPhrases}</div>
            </div>

            <div className="voice-stat-item">
              <div className="voice-stat-label">Adaptive Bonus</div>
              <div className={`voice-stat-value ${parseFloat(stats.adaptiveBonus) < 0 ? 'bonus-positive' : 'bonus-negative'}`}>
                {stats.adaptiveBonus}
              </div>
            </div>
          </div>

          {stats.predictedNextCommand && (
            <div className="voice-prediction">
              <div className="voice-prediction-icon">🔮</div>
              <div className="voice-prediction-text">
                Predicted next: <strong>"{stats.predictedNextCommand}"</strong>
              </div>
            </div>
          )}

          <div className="voice-stats-footer">
            <div className="voice-stats-explanation">
              🧠 JARVIS is learning your voice patterns and getting faster with each command
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceStatsDisplay;
