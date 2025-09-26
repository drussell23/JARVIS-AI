/**
 * Debug Logger for Frontend
 * ========================
 * Enhanced logging with color coding and filtering
 */

class DebugLogger {
  constructor() {
    this.enabled = true;
    this.filters = {
      config: true,
      connection: true,
      api: true,
      websocket: true,
      error: true,
      success: true
    };
    
    // Color schemes for different log types
    this.colors = {
      config: 'color: #4CAF50; font-weight: bold;',
      connection: 'color: #2196F3; font-weight: bold;',
      api: 'color: #FF9800; font-weight: bold;',
      websocket: 'color: #9C27B0; font-weight: bold;',
      error: 'color: #f44336; font-weight: bold; background: #ffebee; padding: 2px 5px;',
      success: 'color: #4CAF50; font-weight: bold; background: #e8f5e9; padding: 2px 5px;',
      warning: 'color: #ff9800; font-weight: bold; background: #fff3e0; padding: 2px 5px;',
      info: 'color: #2196F3;',
      debug: 'color: #9E9E9E;'
    };
    
    // Check localStorage for debug settings
    this.loadSettings();
    
    // Add console methods
    this.setupConsoleMethods();
  }
  
  loadSettings() {
    try {
      const settings = localStorage.getItem('jarvis_debug_settings');
      if (settings) {
        const parsed = JSON.parse(settings);
        this.enabled = parsed.enabled ?? true;
        this.filters = { ...this.filters, ...parsed.filters };
      }
    } catch (error) {
      console.error('Failed to load debug settings:', error);
    }
  }
  
  saveSettings() {
    try {
      localStorage.setItem('jarvis_debug_settings', JSON.stringify({
        enabled: this.enabled,
        filters: this.filters
      }));
    } catch (error) {
      console.error('Failed to save debug settings:', error);
    }
  }
  
  setupConsoleMethods() {
    // Add a global debug object
    if (typeof window !== 'undefined') {
      window.jarvisDebug = {
        enable: () => this.enable(),
        disable: () => this.disable(),
        filter: (type, enabled) => this.setFilter(type, enabled),
        showFilters: () => this.showFilters(),
        clear: () => console.clear(),
        test: () => this.testAllTypes()
      };
      
      console.log(
        '%c🤖 JARVIS Debug Logger Ready',
        'color: #ffd700; font-size: 16px; font-weight: bold; text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);'
      );
      console.log('Use window.jarvisDebug for debug controls');
    }
  }
  
  enable() {
    this.enabled = true;
    this.saveSettings();
    console.log('%c✅ Debug logging enabled', this.colors.success);
  }
  
  disable() {
    this.enabled = false;
    this.saveSettings();
    console.log('%c🔇 Debug logging disabled', this.colors.warning);
  }
  
  setFilter(type, enabled) {
    if (this.filters.hasOwnProperty(type)) {
      this.filters[type] = enabled;
      this.saveSettings();
      console.log(`%c${enabled ? '✅' : '❌'} ${type} logging ${enabled ? 'enabled' : 'disabled'}`, this.colors.info);
    } else {
      console.error(`Unknown filter type: ${type}`);
    }
  }
  
  showFilters() {
    console.log('%c📋 Debug Filters:', this.colors.info);
    Object.entries(this.filters).forEach(([type, enabled]) => {
      console.log(`  ${enabled ? '✅' : '❌'} ${type}`);
    });
  }
  
  log(type, message, ...args) {
    if (!this.enabled || !this.filters[type]) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const prefix = `[${timestamp}] [${type.toUpperCase()}]`;
    const style = this.colors[type] || this.colors.info;
    
    console.log(`%c${prefix} ${message}`, style, ...args);
  }
  
  // Convenience methods
  config(message, ...args) {
    this.log('config', message, ...args);
  }
  
  connection(message, ...args) {
    this.log('connection', message, ...args);
  }
  
  api(message, ...args) {
    this.log('api', message, ...args);
  }
  
  websocket(message, ...args) {
    this.log('websocket', message, ...args);
  }
  
  error(message, ...args) {
    this.log('error', message, ...args);
    console.trace('Error trace:');
  }
  
  success(message, ...args) {
    this.log('success', message, ...args);
  }
  
  warning(message, ...args) {
    console.warn(`%c[WARNING] ${message}`, this.colors.warning, ...args);
  }
  
  info(message, ...args) {
    if (!this.enabled) return;
    console.info(`%c[INFO] ${message}`, this.colors.info, ...args);
  }
  
  debug(message, ...args) {
    if (!this.enabled) return;
    console.debug(`%c[DEBUG] ${message}`, this.colors.debug, ...args);
  }
  
  group(title, fn) {
    if (!this.enabled) {
      fn();
      return;
    }
    
    console.group(`%c${title}`, this.colors.info);
    fn();
    console.groupEnd();
  }
  
  table(data, columns) {
    if (!this.enabled) return;
    console.table(data, columns);
  }
  
  testAllTypes() {
    console.log('%c🧪 Testing all log types:', this.colors.info);
    this.config('Config message example');
    this.connection('Connection message example');
    this.api('API message example');
    this.websocket('WebSocket message example');
    this.error('Error message example');
    this.success('Success message example');
    this.warning('Warning message example');
    this.info('Info message example');
    this.debug('Debug message example');
  }
}

// Create singleton instance
const logger = new DebugLogger();

// Export both the instance and the class
export { DebugLogger };
export default logger;