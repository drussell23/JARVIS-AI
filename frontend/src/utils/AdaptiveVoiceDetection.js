/**
 * Adaptive Voice Detection System
 * Stub implementation - adaptive learning disabled
 */

class AdaptiveVoiceDetection {
  constructor() {
    this.enabled = false;
  }

  initialize() {
    // Stub - no adaptive learning
    return false;
  }

  getStats() {
    return {
      enabled: false,
      message: 'Adaptive voice detection not available'
    };
  }

  disable() {
    // Stub
  }

  shouldProcessResult(result, context) {
    // Always process results (no filtering)
    return {
      shouldProcess: true,
      confidence: 1.0,
      reason: 'adaptive_learning_disabled'
    };
  }

  recordCommandExecution(command, metadata) {
    // Stub - no recording
  }
}

const adaptiveVoiceDetection = new AdaptiveVoiceDetection();
export default adaptiveVoiceDetection;
