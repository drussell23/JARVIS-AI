import React from 'react';
import './DeveloperMetricsOverlay.css';

/**
 * Developer Metrics Overlay
 *
 * Displays detailed biometric and performance metrics for voice unlock
 * Only visible in development/debug mode - not announced by JARVIS
 */
const DeveloperMetricsOverlay = ({ devMetrics, visible }) => {
  if (!visible || !devMetrics) {
    return null;
  }

  const { biometrics, performance, quality_indicators } = devMetrics;

  // Determine confidence bar color
  const getConfidenceColor = (confidence, threshold) => {
    const margin = confidence - threshold;
    if (margin > 0.2) return '#00ff00'; // Bright green - well above threshold
    if (margin > 0.1) return '#7cfc00'; // Green - above threshold
    if (margin > 0) return '#ffd700'; // Gold - just above threshold
    return '#ff4444'; // Red - below threshold
  };

  const confidenceBarColor = biometrics ? getConfidenceColor(
    biometrics.speaker_confidence,
    biometrics.threshold
  ) : '#888';

  return (
    <div className="dev-metrics-overlay">
      <div className="dev-metrics-header">
        <span className="dev-metrics-title">🔬 Developer Metrics</span>
        <span className="dev-metrics-subtitle">(UI only - not announced)</span>
      </div>

      {biometrics && (
        <div className="dev-metrics-section">
          <h4>🎤 Biometric Confidence</h4>
          <div className="metric-row">
            <span className="metric-label">Speaker Match:</span>
            <span className="metric-value">{biometrics.confidence_percentage}</span>
            <span className={`metric-badge ${biometrics.above_threshold ? 'badge-success' : 'badge-error'}`}>
              {biometrics.above_threshold ? '✓ PASS' : '✗ FAIL'}
            </span>
          </div>
          <div className="confidence-bar-container">
            <div
              className="confidence-bar"
              style={{
                width: `${biometrics.speaker_confidence * 100}%`,
                backgroundColor: confidenceBarColor
              }}
            />
            <div
              className="confidence-threshold"
              style={{ left: `${biometrics.threshold * 100}%` }}
              title={`Threshold: ${biometrics.threshold * 100}%`}
            />
          </div>
          <div className="metric-row">
            <span className="metric-label">Threshold:</span>
            <span className="metric-value">{(biometrics.threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Margin:</span>
            <span className={`metric-value ${biometrics.confidence_margin > 0 ? 'text-success' : 'text-error'}`}>
              {biometrics.confidence_margin > 0 ? '+' : ''}{(biometrics.confidence_margin * 100).toFixed(1)}%
            </span>
          </div>
          <div className="metric-row">
            <span className="metric-label">STT Confidence:</span>
            <span className="metric-value">{(biometrics.stt_confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}

      {quality_indicators && (
        <div className="dev-metrics-section">
          <h4>📊 Quality Indicators</h4>
          <div className="metric-row">
            <span className="metric-label">Audio Quality:</span>
            <span className="metric-badge badge-info">{quality_indicators.audio_quality}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Voice Match:</span>
            <span className="metric-badge badge-info">{quality_indicators.voice_match_quality}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Overall:</span>
            <span className="metric-value">{(quality_indicators.overall_confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}

      {performance && (
        <div className="dev-metrics-section">
          <h4>⚡ Performance</h4>
          <div className="metric-row">
            <span className="metric-label">Total Latency:</span>
            <span className="metric-value">{performance.total_latency_ms?.toFixed(0) || 'N/A'}ms</span>
          </div>
          {performance.transcription_time_ms && (
            <div className="metric-row">
              <span className="metric-label">Transcription:</span>
              <span className="metric-value">{performance.transcription_time_ms.toFixed(0)}ms</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DeveloperMetricsOverlay;
