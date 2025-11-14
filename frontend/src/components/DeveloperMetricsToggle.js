import React from 'react';
import './DeveloperMetricsOverlay.css';

/**
 * Simple toggle button for developer metrics
 * Separated for easier debugging
 */
const DeveloperMetricsToggle = ({ visible, onToggle }) => {
  return (
    <button
      className={`dev-metrics-toggle ${visible ? 'active' : ''}`}
      onClick={onToggle}
      title="Toggle Developer Metrics"
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 99999,
        display: 'block !important',
        visibility: 'visible !important'
      }}
    >
      <span style={{ marginRight: '6px' }}>🔬</span>
      {visible ? 'Hide' : 'Show'} Dev Metrics
    </button>
  );
};

export default DeveloperMetricsToggle;
