/**
 * Config Diagnostic Component
 * ==========================
 * Shows real-time config status and allows manual refresh
 */

import React, { useState, useEffect } from 'react';
import configService from '../services/DynamicConfigService';
import logger from '../utils/DebugLogger';

const ConfigDiagnostic = () => {
  const [config, setConfig] = useState({
    API_BASE_URL: null,
    WS_BASE_URL: null,
    discovered: false,
    services: {}
  });
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [showDiagnostic, setShowDiagnostic] = useState(true);

  useEffect(() => {
    // Get initial config
    updateConfig();

    // Listen for config changes
    configService.on('config-ready', handleConfigUpdate);
    configService.on('config-updated', handleConfigUpdate);
    configService.on('discovery-failed', handleDiscoveryFailed);

    return () => {
      configService.off('config-ready', handleConfigUpdate);
      configService.off('config-updated', handleConfigUpdate);
      configService.off('discovery-failed', handleDiscoveryFailed);
    };
  }, []);

  const updateConfig = () => {
    const currentConfig = {
      API_BASE_URL: configService.getApiUrl(),
      WS_BASE_URL: configService.getWebSocketUrl(),
      discovered: configService.config.discovered,
      services: configService.config.SERVICES || {}
    };
    setConfig(currentConfig);
    logger.config('Current config state:', currentConfig);
  };

  const handleConfigUpdate = (newConfig) => {
    logger.success('Config updated:', newConfig);
    updateConfig();
    setIsDiscovering(false);
  };

  const handleDiscoveryFailed = (error) => {
    logger.error('Discovery failed:', error);
    setIsDiscovering(false);
  };

  const clearCacheAndRediscover = () => {
    logger.config('Clearing cache and rediscovering...');
    localStorage.removeItem('jarvis_dynamic_config');
    setIsDiscovering(true);
    configService.discover();
  };

  const forceRediscover = () => {
    logger.config('Force rediscovering services...');
    setIsDiscovering(true);
    configService.discover();
  };

  const testEndpoint = async (endpoint) => {
    const url = configService.getApiUrl(endpoint);
    if (!url) {
      logger.error('No API URL available');
      return;
    }

    try {
      logger.api(`Testing endpoint: ${url}`);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        logger.success(`Endpoint ${endpoint} is working:`, data);
      } else {
        logger.error(`Endpoint ${endpoint} returned ${response.status}`);
      }
    } catch (error) {
      logger.error(`Failed to test endpoint ${endpoint}:`, error);
    }
  };

  if (!showDiagnostic) {
    return (
      <button
        onClick={() => setShowDiagnostic(true)}
        style={{
          position: 'fixed',
          bottom: '10px',
          right: '10px',
          padding: '5px 10px',
          background: '#2196F3',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          zIndex: 9999
        }}
      >
        Show Config
      </button>
    );
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '10px',
      right: '10px',
      background: 'rgba(0, 0, 0, 0.9)',
      color: 'white',
      padding: '15px',
      borderRadius: '10px',
      maxWidth: '400px',
      fontSize: '12px',
      fontFamily: 'monospace',
      zIndex: 9999,
      border: '1px solid #333'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <h4 style={{ margin: 0, color: '#ffd700' }}>🔧 Config Diagnostic</h4>
        <button
          onClick={() => setShowDiagnostic(false)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#999',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          ✕
        </button>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <strong>Status:</strong> {' '}
        <span style={{ color: config.discovered ? '#4CAF50' : '#f44336' }}>
          {config.discovered ? '✅ Discovered' : '❌ Not Discovered'}
        </span>
        {isDiscovering && <span style={{ color: '#FF9800' }}> 🔄 Discovering...</span>}
      </div>

      <div style={{ marginBottom: '10px' }}>
        <strong>API URL:</strong> {' '}
        <code style={{ color: '#4CAF50' }}>{config.API_BASE_URL || 'Not set'}</code>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <strong>WS URL:</strong> {' '}
        <code style={{ color: '#4CAF50' }}>{config.WS_BASE_URL || 'Not set'}</code>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <strong>Services Found:</strong>
        {Object.keys(config.services).length > 0 ? (
          <ul style={{ margin: '5px 0 0 20px', padding: 0 }}>
            {Object.entries(config.services).map(([type, service]) => (
              <li key={type} style={{ listStyle: 'none', color: '#4CAF50' }}>
                ✅ {type} on port {service.port}
              </li>
            ))}
          </ul>
        ) : (
          <span style={{ color: '#f44336' }}> None</span>
        )}
      </div>

      <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '10px' }}>
        <button
          onClick={clearCacheAndRediscover}
          style={{
            padding: '5px 10px',
            background: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '11px'
          }}
        >
          Clear Cache & Rediscover
        </button>
        <button
          onClick={forceRediscover}
          style={{
            padding: '5px 10px',
            background: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '11px'
          }}
        >
          Force Rediscover
        </button>
        <button
          onClick={updateConfig}
          style={{
            padding: '5px 10px',
            background: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '11px'
          }}
        >
          Refresh
        </button>
      </div>

      <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #333' }}>
        <strong>Test Endpoints:</strong>
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginTop: '5px' }}>
          <button
            onClick={() => testEndpoint('health')}
            style={{
              padding: '3px 8px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '10px'
            }}
          >
            /health
          </button>
          <button
            onClick={() => testEndpoint('voice/jarvis/status')}
            style={{
              padding: '3px 8px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '10px'
            }}
          >
            /jarvis/status
          </button>
          <button
            onClick={() => testEndpoint('api/wake-word/status')}
            style={{
              padding: '3px 8px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '10px'
            }}
          >
            /wake-word
          </button>
        </div>
      </div>

      <div style={{ marginTop: '10px', fontSize: '10px', color: '#666' }}>
        Open console for detailed logs (F12)
      </div>
    </div>
  );
};

export default ConfigDiagnostic;