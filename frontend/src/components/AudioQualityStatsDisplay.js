import React, { useState, useEffect } from 'react';
import audioQualityAdaptation from '../utils/AudioQualityAdaptation';
import './AudioQualityStatsDisplay.css';

const AudioQualityStatsDisplay = ({ show }) => {
  const [stats, setStats] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!show) return;

    // Update stats every 2 seconds
    const interval = setInterval(() => {
      const currentStats = audioQualityAdaptation.getStats();
      setStats(currentStats);
    }, 2000);

    // Get initial stats
    setStats(audioQualityAdaptation.getStats());

    return () => clearInterval(interval);
  }, [show]);

  if (!show || !stats) return null;

  // Quality color coding
  const getQualityColor = (quality) => {
    switch (quality) {
      case 'excellent': return '#00ff88';
      case 'good': return '#00d4ff';
      case 'fair': return '#ffaa00';
      case 'poor': return '#ff6600';
      default: return '#888';
    }
  };

  const getDistanceColor = (distance) => {
    switch (distance) {
      case 'close': return '#00ff88';
      case 'normal': return '#00d4ff';
      case 'far': return '#ffaa00';
      case 'very-far': return '#ff6600';
      default: return '#888';
    }
  };

  return (
    <div className={`audio-quality-stats-container ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="audio-quality-stats-header" onClick={() => setExpanded(!expanded)}>
        <span className="audio-quality-stats-icon">🎚️</span>
        <span className="audio-quality-stats-title">Audio Quality</span>
        <span className="audio-quality-stats-toggle">{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div className="audio-quality-stats-content">
          {/* Microphone Quality */}
          <div className="audio-quality-section">
            <div className="audio-quality-section-title">🎤 Microphone</div>
            <div className="audio-quality-stats-grid">
              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Quality</div>
                <div
                  className="audio-quality-stat-value"
                  style={{ color: getQualityColor(stats.micQuality) }}
                >
                  {stats.micQuality.charAt(0).toUpperCase() + stats.micQuality.slice(1)}
                </div>
                <div className="audio-quality-stat-bar">
                  <div
                    className="audio-quality-stat-bar-fill quality"
                    style={{
                      width: stats.micQualityScore + '%',
                      background: `linear-gradient(90deg, ${getQualityColor(stats.micQuality)}, #00d4ff)`
                    }}
                  />
                </div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Type</div>
                <div className="audio-quality-stat-value">
                  {stats.micType === 'laptop-builtin' ? '💻 Built-in' :
                   stats.micType === 'bluetooth' ? '📡 Bluetooth' :
                   stats.micType === 'headset' ? '🎧 Headset' :
                   stats.micType === 'external' ? '🎤 External' :
                   'Unknown'}
                </div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Calibrated</div>
                <div className={`audio-quality-stat-value ${stats.micCalibrated ? 'calibrated' : 'calibrating'}`}>
                  {stats.micCalibrated ? '✓ Yes' : 'In Progress...'}
                </div>
              </div>
            </div>
          </div>

          {/* Distance */}
          <div className="audio-quality-section">
            <div className="audio-quality-section-title">📏 Distance</div>
            <div className="audio-quality-stats-grid">
              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Position</div>
                <div
                  className="audio-quality-stat-value"
                  style={{ color: getDistanceColor(stats.distance) }}
                >
                  {stats.distance.charAt(0).toUpperCase() + stats.distance.slice(1).replace('-', ' ')}
                </div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Estimated</div>
                <div className="audio-quality-stat-value distance">
                  {stats.distanceMeters} m
                </div>
              </div>
            </div>
          </div>

          {/* Signal Quality */}
          <div className="audio-quality-section">
            <div className="audio-quality-section-title">📊 Signal Quality</div>
            <div className="audio-quality-stats-grid">
              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">SNR</div>
                <div className="audio-quality-stat-value snr">{stats.currentSNR}</div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Intelligibility</div>
                <div className="audio-quality-stat-value">{stats.speechIntelligibility}</div>
                <div className="audio-quality-stat-bar">
                  <div
                    className="audio-quality-stat-bar-fill intelligibility"
                    style={{ width: stats.speechIntelligibility }}
                  />
                </div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Clarity</div>
                <div className="audio-quality-stat-value">{stats.spectralClarity}</div>
                <div className="audio-quality-stat-bar">
                  <div
                    className="audio-quality-stat-bar-fill clarity"
                    style={{ width: stats.spectralClarity }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Processing */}
          <div className="audio-quality-section">
            <div className="audio-quality-section-title">⚙️ Processing</div>
            <div className="audio-quality-stats-grid">
              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">AGC Gain</div>
                <div className="audio-quality-stat-value agc">{stats.agcGain}</div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Compression</div>
                <div className="audio-quality-stat-value">{stats.compressionRatio}</div>
              </div>
            </div>
          </div>

          {/* Quality Issues */}
          {stats.issues && Object.values(stats.issues).some(issue => issue) && (
            <div className="audio-quality-section">
              <div className="audio-quality-section-title">⚠️ Quality Issues</div>
              <div className="audio-quality-issues-grid">
                {stats.issues.poorLowFreq && (
                  <div className="audio-quality-issue-badge">
                    🔉 Poor Bass
                  </div>
                )}
                {stats.issues.poorHighFreq && (
                  <div className="audio-quality-issue-badge">
                    🔈 Poor Treble
                  </div>
                )}
                {stats.issues.narrowBandwidth && (
                  <div className="audio-quality-issue-badge">
                    📉 Narrow Bandwidth
                  </div>
                )}
                {stats.issues.highNoise && (
                  <div className="audio-quality-issue-badge">
                    📢 High Noise
                  </div>
                )}
                {stats.issues.compression && (
                  <div className="audio-quality-issue-badge">
                    🔄 Compression Detected
                  </div>
                )}
                {stats.issues.clipping && (
                  <div className="audio-quality-issue-badge">
                    ⚡ Clipping
                  </div>
                )}
                {stats.issues.dropout && (
                  <div className="audio-quality-issue-badge">
                    📡 Dropouts
                  </div>
                )}
                {stats.issues.jitter && (
                  <div className="audio-quality-issue-badge">
                    📊 Level Jitter
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Performance */}
          <div className="audio-quality-section">
            <div className="audio-quality-section-title">⚡ Performance</div>
            <div className="audio-quality-stats-grid">
              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Latency</div>
                <div className="audio-quality-stat-value">{stats.averageLatency}</div>
              </div>

              <div className="audio-quality-stat-item">
                <div className="audio-quality-stat-label">Samples</div>
                <div className="audio-quality-stat-value">{stats.totalSamples}</div>
              </div>
            </div>
          </div>

          <div className="audio-quality-stats-footer">
            <div className="audio-quality-stats-explanation">
              🎚️ Real-time audio quality enhancement with AGC, EQ, and distance compensation
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioQualityStatsDisplay;
