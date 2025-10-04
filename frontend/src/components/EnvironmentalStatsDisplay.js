import React, { useState, useEffect } from 'react';
import environmentalAdaptation from '../utils/EnvironmentalAdaptation';
import './EnvironmentalStatsDisplay.css';

const EnvironmentalStatsDisplay = ({ show }) => {
  const [stats, setStats] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!show) return;

    // Update stats every 2 seconds
    const interval = setInterval(() => {
      const currentStats = environmentalAdaptation.getStats();
      setStats(currentStats);
    }, 2000);

    // Get initial stats
    setStats(environmentalAdaptation.getStats());

    return () => clearInterval(interval);
  }, [show]);

  if (!show || !stats) return null;

  return (
    <div className={`env-stats-container ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="env-stats-header" onClick={() => setExpanded(!expanded)}>
        <span className="env-stats-icon">🌍</span>
        <span className="env-stats-title">Environmental Adaptation</span>
        <span className="env-stats-toggle">{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div className="env-stats-content">
          {/* Primary User Enrollment */}
          <div className="env-section">
            <div className="env-section-title">👤 Speaker Recognition</div>
            <div className="env-stats-grid">
              <div className="env-stat-item">
                <div className="env-stat-label">Primary User</div>
                <div className={`env-stat-value ${stats.primaryUserEnrolled ? 'enrolled' : 'enrolling'}`}>
                  {stats.primaryUserEnrolled ? '✓ Enrolled' : 'Learning...'}
                </div>
                {!stats.primaryUserEnrolled && (
                  <div className="env-stat-bar">
                    <div
                      className="env-stat-bar-fill enrollment"
                      style={{ width: stats.enrollmentProgress }}
                    />
                  </div>
                )}
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Other Speakers</div>
                <div className="env-stat-value">{stats.otherSpeakersCount}</div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Current Speaker</div>
                <div className={`env-stat-value ${stats.currentPrimaryUser ? 'primary' : 'other'}`}>
                  {stats.currentSpeech ? (stats.currentPrimaryUser ? 'You' : 'Other') : 'None'}
                </div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Speech Clarity</div>
                <div className="env-stat-value clarity">{stats.currentClarity}</div>
              </div>
            </div>
          </div>

          {/* Environmental Conditions */}
          <div className="env-section">
            <div className="env-section-title">🔊 Acoustic Environment</div>
            <div className="env-stats-grid">
              <div className="env-stat-item">
                <div className="env-stat-label">Noise Floor</div>
                <div className="env-stat-value">{stats.noiseFloor}</div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Signal/Noise</div>
                <div className="env-stat-value snr">{stats.averageSNR}</div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Stability</div>
                <div className="env-stat-value">{stats.spectralStability}</div>
                <div className="env-stat-bar">
                  <div
                    className="env-stat-bar-fill stability"
                    style={{ width: stats.spectralStability }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Detection Flags */}
          <div className="env-section">
            <div className="env-section-title">🚨 Active Detections</div>
            <div className="env-detection-grid">
              {stats.tvRadioDetected && (
                <div className="env-detection-badge tv">
                  📺 TV/Radio
                </div>
              )}
              {stats.musicDetected && (
                <div className="env-detection-badge music">
                  🎵 Music
                </div>
              )}
              {stats.echoDetected && (
                <div className="env-detection-badge echo">
                  🔁 Echo/Reverb
                </div>
              )}
              {stats.multiSpeakerDetected && (
                <div className="env-detection-badge multi">
                  👥 Multiple Speakers
                </div>
              )}
              {stats.suddenNoiseDetected && (
                <div className="env-detection-badge noise">
                  ⚡ Noise Spike
                </div>
              )}
              {!stats.tvRadioDetected && !stats.musicDetected && !stats.echoDetected &&
               !stats.multiSpeakerDetected && !stats.suddenNoiseDetected && (
                <div className="env-detection-badge clear">
                  ✅ Clear Environment
                </div>
              )}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="env-section">
            <div className="env-section-title">⚡ Performance</div>
            <div className="env-stats-grid">
              <div className="env-stat-item">
                <div className="env-stat-label">Avg Latency</div>
                <div className="env-stat-value">{stats.averageLatency}</div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Total Frames</div>
                <div className="env-stat-value">{stats.totalFrames}</div>
              </div>

              <div className="env-stat-item">
                <div className="env-stat-label">Drop Rate</div>
                <div className="env-stat-value">{stats.dropRate}</div>
              </div>
            </div>
          </div>

          <div className="env-stats-footer">
            <div className="env-stats-explanation">
              🌍 Advanced environmental adaptation with speaker isolation, noise cancellation, and acoustic analysis
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnvironmentalStatsDisplay;
