/**
 * Command Detection Banner - Visual feedback for detected commands
 *
 * Displays a banner when the backend's streaming safeguard detects
 * a command (like "unlock") and stops the audio stream.
 */

import React, { useState, useEffect } from 'react';

const CommandDetectionBanner = ({ command, onDismiss, autoDismiss = 3000 }) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (autoDismiss && autoDismiss > 0) {
      const timer = setTimeout(() => {
        setVisible(false);
        if (onDismiss) onDismiss();
      }, autoDismiss);

      return () => clearTimeout(timer);
    }
  }, [autoDismiss, onDismiss]);

  if (!visible || !command) return null;

  return (
    <div className="command-detection-banner">
      <div className="banner-content">
        <div className="banner-icon">🛡️</div>
        <div className="banner-text">
          <div className="banner-title">Command Detected</div>
          <div className="banner-command">"{command}"</div>
          <div className="banner-subtitle">Audio stream stopped</div>
        </div>
        <button
          className="banner-close"
          onClick={() => {
            setVisible(false);
            if (onDismiss) onDismiss();
          }}
          aria-label="Close"
        >
          ✕
        </button>
      </div>
      <div className="banner-progress"></div>
    </div>
  );
};

export default CommandDetectionBanner;
