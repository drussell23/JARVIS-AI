/**
 * Adaptive Voice Detection System
 * ================================
 *
 * Advanced voice detection that learns from each interaction and gets stronger over time.
 * Features:
 * - User voice pattern learning
 * - Dynamic confidence thresholding
 * - Command success tracking
 * - Acoustic environment adaptation
 * - Predictive command recognition
 * - Real-time performance optimization
 */

class AdaptiveVoiceDetection {
  constructor() {
    // User voice profile
    this.voiceProfile = {
      // Voice characteristics
      pitchRange: { min: null, max: null, average: null },
      speakingRate: { min: null, max: null, average: null },
      volumeProfile: { min: null, max: null, average: null },

      // Learned patterns
      commonPhrases: new Map(), // phrase -> frequency
      commandPatterns: new Map(), // command -> success rate
      confidenceHistory: [], // Recent confidence scores

      // Success metrics
      totalCommands: 0,
      successfulCommands: 0,
      failedCommands: 0,
      retryCommands: 0,

      // Timing patterns
      averageCommandLength: 0,
      averagePauseAfterWakeWord: 0,
      preferredSpeechSpeed: 'normal', // slow, normal, fast

      // Environment adaptation
      noiseLevel: 0,
      environmentType: 'quiet', // quiet, moderate, noisy
      timeOfDayPatterns: new Map(), // hour -> success rate
    };

    // Dynamic thresholds (start conservative, get more aggressive as confidence builds)
    this.thresholds = {
      minimumConfidence: 0.85, // Will decrease as user voice is learned
      highConfidenceBonus: 0.0, // Bonus added based on voice profile match
      wakeWordConfidence: 0.75, // Lower for wake word (more permissive)
      commandConfidence: 0.85, // Higher for commands (more conservative initially)
      adaptiveAdjustment: 0.0, // Dynamic adjustment based on success rate
    };

    // Performance tracking
    this.sessionStats = {
      startTime: Date.now(),
      commandsThisSession: 0,
      successRate: 1.0,
      averageConfidence: 0.0,
      improvementTrend: 0.0,
    };

    // Command prediction
    this.commandPredictor = {
      recentCommands: [], // Last 10 commands for pattern detection
      predictedNextCommand: null,
      predictionConfidence: 0.0,
      commonSequences: new Map(), // command1 -> [command2, command3, ...]
    };

    // Real-time optimization
    this.optimizationEngine = {
      processingTimes: [],
      bottlenecks: [],
      optimizations: [],
    };

    // Load saved profile from localStorage
    this.loadVoiceProfile();

    // Auto-save every 30 seconds
    setInterval(() => this.saveVoiceProfile(), 30000);
  }

  /**
   * Analyze incoming speech result and decide if it should be processed
   */
  shouldProcessResult(result, context = {}) {
    const { transcript, confidence, isFinal, isWaitingForCommand, timestamp } = context;

    // Calculate enhanced confidence score
    const enhancedConfidence = this.calculateEnhancedConfidence(result, context);

    // Determine appropriate threshold based on context
    const threshold = this.getDynamicThreshold(context);

    // Check if we should process
    const shouldProcess = enhancedConfidence >= threshold;

    // Log decision for learning
    this.logProcessingDecision({
      transcript,
      originalConfidence: confidence,
      enhancedConfidence,
      threshold,
      shouldProcess,
      isFinal,
      timestamp,
    });

    return {
      shouldProcess,
      enhancedConfidence,
      threshold,
      reason: this.getProcessingReason(enhancedConfidence, threshold, context),
    };
  }

  /**
   * Calculate enhanced confidence by analyzing voice patterns and history
   */
  calculateEnhancedConfidence(result, context) {
    const { transcript, confidence } = context;
    let enhancedConfidence = confidence;

    // 1. Voice Pattern Match Bonus (+0.05 to +0.15)
    const voiceMatchBonus = this.calculateVoiceMatchBonus(context);
    enhancedConfidence += voiceMatchBonus;

    // 2. Phrase Familiarity Bonus (+0.03 to +0.10)
    const familiarityBonus = this.calculateFamiliarityBonus(transcript);
    enhancedConfidence += familiarityBonus;

    // 3. Context Prediction Bonus (+0.05 to +0.12)
    const predictionBonus = this.calculatePredictionBonus(transcript);
    enhancedConfidence += predictionBonus;

    // 4. Time-of-Day Pattern Bonus (+0.02 to +0.08)
    const timeBonus = this.calculateTimeBonus();
    enhancedConfidence += timeBonus;

    // 5. Recent Success Streak Bonus (+0.03 to +0.10)
    const streakBonus = this.calculateStreakBonus();
    enhancedConfidence += streakBonus;

    // 6. Environment Adjustment (-0.10 to +0.05)
    const environmentAdjustment = this.calculateEnvironmentAdjustment();
    enhancedConfidence += environmentAdjustment;

    // Cap at 1.0
    enhancedConfidence = Math.min(1.0, enhancedConfidence);

    console.log(`🧠 Enhanced Confidence: ${(confidence * 100).toFixed(1)}% → ${(enhancedConfidence * 100).toFixed(1)}%`, {
      voiceMatch: `+${(voiceMatchBonus * 100).toFixed(1)}%`,
      familiarity: `+${(familiarityBonus * 100).toFixed(1)}%`,
      prediction: `+${(predictionBonus * 100).toFixed(1)}%`,
      time: `+${(timeBonus * 100).toFixed(1)}%`,
      streak: `+${(streakBonus * 100).toFixed(1)}%`,
      environment: `${environmentAdjustment >= 0 ? '+' : ''}${(environmentAdjustment * 100).toFixed(1)}%`,
    });

    return enhancedConfidence;
  }

  /**
   * Calculate bonus based on voice pattern matching
   */
  calculateVoiceMatchBonus(context) {
    if (!this.voiceProfile.pitchRange.average) return 0;

    // This would analyze pitch, rate, volume against learned profile
    // For now, return bonus based on command history
    const successRate = this.voiceProfile.successfulCommands / Math.max(1, this.voiceProfile.totalCommands);
    return successRate > 0.9 ? 0.10 : successRate > 0.8 ? 0.07 : successRate > 0.7 ? 0.05 : 0.02;
  }

  /**
   * Calculate bonus for familiar phrases
   */
  calculateFamiliarityBonus(transcript) {
    if (!transcript) return 0;

    const transcriptLower = transcript.toLowerCase();

    // Check if this phrase or similar has been used before
    for (const [phrase, frequency] of this.voiceProfile.commonPhrases.entries()) {
      if (transcriptLower.includes(phrase) || phrase.includes(transcriptLower)) {
        // More familiar = higher bonus
        const bonus = Math.min(0.10, frequency * 0.01);
        return bonus;
      }
    }

    return 0;
  }

  /**
   * Calculate bonus based on command prediction
   */
  calculatePredictionBonus(transcript) {
    if (!this.commandPredictor.predictedNextCommand) return 0;

    const transcriptLower = transcript.toLowerCase();
    const predicted = this.commandPredictor.predictedNextCommand.toLowerCase();

    // Check if current transcript matches prediction
    if (transcriptLower.includes(predicted) || predicted.includes(transcriptLower)) {
      return this.commandPredictor.predictionConfidence * 0.12;
    }

    return 0;
  }

  /**
   * Calculate bonus based on time of day patterns
   */
  calculateTimeBonus() {
    const hour = new Date().getHours();
    const timePattern = this.voiceProfile.timeOfDayPatterns.get(hour);

    if (timePattern && timePattern.count > 5) {
      // Higher success rate at this hour = bonus
      const successRate = timePattern.successes / timePattern.count;
      return successRate > 0.9 ? 0.08 : successRate > 0.8 ? 0.05 : 0.02;
    }

    return 0;
  }

  /**
   * Calculate bonus based on recent success streak
   */
  calculateStreakBonus() {
    // Check last 5 commands
    const recentCommands = this.commandPredictor.recentCommands.slice(-5);
    if (recentCommands.length < 3) return 0;

    const recentSuccesses = recentCommands.filter(cmd => cmd.success).length;
    const streakRate = recentSuccesses / recentCommands.length;

    return streakRate >= 1.0 ? 0.10 : streakRate >= 0.8 ? 0.07 : streakRate >= 0.6 ? 0.04 : 0;
  }

  /**
   * Calculate environment-based adjustment
   */
  calculateEnvironmentAdjustment() {
    const noiseLevel = this.voiceProfile.noiseLevel;

    if (noiseLevel < 0.2) {
      return 0.05; // Very quiet, boost confidence
    } else if (noiseLevel < 0.5) {
      return 0.02; // Moderate, small boost
    } else if (noiseLevel < 0.7) {
      return -0.05; // Noisy, reduce confidence
    } else {
      return -0.10; // Very noisy, significantly reduce
    }
  }

  /**
   * Get dynamic threshold based on context
   */
  getDynamicThreshold(context) {
    const { isWaitingForCommand, transcript } = context;

    // Start with base threshold
    let threshold = isWaitingForCommand
      ? this.thresholds.commandConfidence
      : this.thresholds.wakeWordConfidence;

    // Apply adaptive adjustment based on recent success
    threshold += this.thresholds.adaptiveAdjustment;

    // Lower threshold for very familiar commands
    if (transcript && this.isVeryFamiliarCommand(transcript)) {
      threshold -= 0.10;
    }

    // Adjust based on overall success rate
    const successRate = this.voiceProfile.successfulCommands / Math.max(1, this.voiceProfile.totalCommands);
    if (successRate > 0.95) {
      threshold -= 0.05; // Very high success, be more aggressive
    } else if (successRate < 0.70) {
      threshold += 0.05; // Low success, be more conservative
    }

    // Ensure threshold stays in reasonable range
    threshold = Math.max(0.60, Math.min(0.95, threshold));

    return threshold;
  }

  /**
   * Check if command is very familiar
   */
  isVeryFamiliarCommand(transcript) {
    const transcriptLower = transcript.toLowerCase();

    // Check common commands
    const veryFamiliarCommands = [
      'lock my screen', 'unlock my screen', 'lock screen', 'unlock screen',
      'what time is it', 'what\'s the weather', 'open', 'close'
    ];

    for (const cmd of veryFamiliarCommands) {
      if (transcriptLower.includes(cmd)) {
        const usage = this.voiceProfile.commonPhrases.get(cmd) || 0;
        return usage >= 5; // Used 5+ times = very familiar
      }
    }

    return false;
  }

  /**
   * Record command execution and result
   */
  recordCommandExecution(command, result) {
    const { success, confidence, executionTime, wasRetry } = result;

    // Update totals
    this.voiceProfile.totalCommands++;
    if (success) {
      this.voiceProfile.successfulCommands++;
    } else {
      this.voiceProfile.failedCommands++;
    }
    if (wasRetry) {
      this.voiceProfile.retryCommands++;
    }

    // Update phrase frequency
    const commandLower = command.toLowerCase();
    const currentCount = this.voiceProfile.commonPhrases.get(commandLower) || 0;
    this.voiceProfile.commonPhrases.set(commandLower, currentCount + 1);

    // Update command pattern success rate
    const pattern = this.voiceProfile.commandPatterns.get(commandLower) || { successes: 0, attempts: 0 };
    pattern.attempts++;
    if (success) pattern.successes++;
    this.voiceProfile.commandPatterns.set(commandLower, pattern);

    // Update confidence history (keep last 50)
    this.voiceProfile.confidenceHistory.push(confidence);
    if (this.voiceProfile.confidenceHistory.length > 50) {
      this.voiceProfile.confidenceHistory.shift();
    }

    // Update time-of-day patterns
    const hour = new Date().getHours();
    const timePattern = this.voiceProfile.timeOfDayPatterns.get(hour) || { count: 0, successes: 0 };
    timePattern.count++;
    if (success) timePattern.successes++;
    this.voiceProfile.timeOfDayPatterns.set(hour, timePattern);

    // Update command predictor
    this.commandPredictor.recentCommands.push({
      command: commandLower,
      success,
      timestamp: Date.now(),
    });
    if (this.commandPredictor.recentCommands.length > 10) {
      this.commandPredictor.recentCommands.shift();
    }

    // Update command sequences for prediction
    if (this.commandPredictor.recentCommands.length >= 2) {
      const prevCommand = this.commandPredictor.recentCommands[this.commandPredictor.recentCommands.length - 2].command;
      const sequences = this.commandPredictor.commonSequences.get(prevCommand) || [];
      sequences.push(commandLower);
      this.commandPredictor.commonSequences.set(prevCommand, sequences);
    }

    // Update session stats
    this.sessionStats.commandsThisSession++;
    this.sessionStats.successRate = this.voiceProfile.successfulCommands / this.voiceProfile.totalCommands;

    // Adjust thresholds based on performance
    this.adjustThresholds();

    // Predict next command
    this.predictNextCommand(commandLower);

    // Save profile
    this.saveVoiceProfile();

    console.log('📊 Voice Profile Updated:', {
      totalCommands: this.voiceProfile.totalCommands,
      successRate: `${(this.sessionStats.successRate * 100).toFixed(1)}%`,
      currentThreshold: `${(this.thresholds.commandConfidence * 100).toFixed(1)}%`,
      adaptiveBonus: `${(this.thresholds.adaptiveAdjustment * 100).toFixed(1)}%`,
    });
  }

  /**
   * Dynamically adjust thresholds based on performance
   */
  adjustThresholds() {
    const successRate = this.voiceProfile.successfulCommands / Math.max(1, this.voiceProfile.totalCommands);

    // If success rate is very high, become more aggressive (lower threshold)
    if (successRate >= 0.95 && this.voiceProfile.totalCommands >= 10) {
      this.thresholds.adaptiveAdjustment = Math.max(-0.15, this.thresholds.adaptiveAdjustment - 0.01);
      console.log('🚀 Increasing aggressiveness (high success rate)');
    }
    // If success rate is low, become more conservative (higher threshold)
    else if (successRate < 0.75 && this.voiceProfile.totalCommands >= 5) {
      this.thresholds.adaptiveAdjustment = Math.min(0.10, this.thresholds.adaptiveAdjustment + 0.02);
      console.log('⚠️ Increasing conservativeness (low success rate)');
    }
    // Gradually return to neutral if performance is stable
    else if (successRate >= 0.85 && successRate < 0.95) {
      this.thresholds.adaptiveAdjustment *= 0.95; // Slowly decay towards 0
    }
  }

  /**
   * Predict next likely command
   */
  predictNextCommand(lastCommand) {
    const sequences = this.commandPredictor.commonSequences.get(lastCommand);
    if (!sequences || sequences.length === 0) {
      this.commandPredictor.predictedNextCommand = null;
      this.commandPredictor.predictionConfidence = 0;
      return;
    }

    // Find most common next command
    const frequency = new Map();
    for (const cmd of sequences) {
      frequency.set(cmd, (frequency.get(cmd) || 0) + 1);
    }

    let maxFreq = 0;
    let predictedCmd = null;
    for (const [cmd, freq] of frequency.entries()) {
      if (freq > maxFreq) {
        maxFreq = freq;
        predictedCmd = cmd;
      }
    }

    this.commandPredictor.predictedNextCommand = predictedCmd;
    this.commandPredictor.predictionConfidence = maxFreq / sequences.length;

    if (this.commandPredictor.predictionConfidence > 0.5) {
      console.log(`🔮 Predicting next command: "${predictedCmd}" (${(this.commandPredictor.predictionConfidence * 100).toFixed(0)}% confidence)`);
    }
  }

  /**
   * Get processing reason for logging
   */
  getProcessingReason(enhancedConfidence, threshold, context) {
    if (enhancedConfidence >= threshold) {
      if (context.isFinal) {
        return 'Final result with sufficient confidence';
      } else {
        return 'High-confidence interim result (adaptive boost applied)';
      }
    } else {
      return `Confidence too low (${(enhancedConfidence * 100).toFixed(1)}% < ${(threshold * 100).toFixed(1)}%)`;
    }
  }

  /**
   * Log processing decision for analysis
   */
  logProcessingDecision(decision) {
    // Could be sent to analytics or stored for debugging
    // For now, just console log important decisions
    if (decision.shouldProcess && !decision.isFinal) {
      console.log('⚡ Fast-tracking interim result:', decision);
    }
  }

  /**
   * Save voice profile to localStorage
   */
  saveVoiceProfile() {
    try {
      const profile = {
        voiceProfile: {
          ...this.voiceProfile,
          commonPhrases: Array.from(this.voiceProfile.commonPhrases.entries()),
          commandPatterns: Array.from(this.voiceProfile.commandPatterns.entries()),
          timeOfDayPatterns: Array.from(this.voiceProfile.timeOfDayPatterns.entries()),
        },
        thresholds: this.thresholds,
        commandPredictor: {
          ...this.commandPredictor,
          commonSequences: Array.from(this.commandPredictor.commonSequences.entries()),
        },
        lastSaved: Date.now(),
      };

      localStorage.setItem('jarvis_adaptive_voice_profile', JSON.stringify(profile));
    } catch (error) {
      console.warn('Failed to save voice profile:', error);
    }
  }

  /**
   * Load voice profile from localStorage
   */
  loadVoiceProfile() {
    try {
      const saved = localStorage.getItem('jarvis_adaptive_voice_profile');
      if (!saved) return;

      const profile = JSON.parse(saved);

      // Restore maps
      this.voiceProfile = {
        ...this.voiceProfile,
        ...profile.voiceProfile,
        commonPhrases: new Map(profile.voiceProfile.commonPhrases),
        commandPatterns: new Map(profile.voiceProfile.commandPatterns),
        timeOfDayPatterns: new Map(profile.voiceProfile.timeOfDayPatterns),
      };

      this.thresholds = { ...this.thresholds, ...profile.thresholds };

      this.commandPredictor = {
        ...this.commandPredictor,
        commonSequences: new Map(profile.commandPredictor.commonSequences),
      };

      console.log('✅ Loaded voice profile:', {
        totalCommands: this.voiceProfile.totalCommands,
        successRate: `${((this.voiceProfile.successfulCommands / Math.max(1, this.voiceProfile.totalCommands)) * 100).toFixed(1)}%`,
        learnedPhrases: this.voiceProfile.commonPhrases.size,
      });
    } catch (error) {
      console.warn('Failed to load voice profile:', error);
    }
  }

  /**
   * Get current stats for display
   */
  getStats() {
    const successRate = this.voiceProfile.successfulCommands / Math.max(1, this.voiceProfile.totalCommands);
    const avgConfidence = this.voiceProfile.confidenceHistory.length > 0
      ? this.voiceProfile.confidenceHistory.reduce((a, b) => a + b, 0) / this.voiceProfile.confidenceHistory.length
      : 0;

    return {
      totalCommands: this.voiceProfile.totalCommands,
      successRate: (successRate * 100).toFixed(1) + '%',
      averageConfidence: (avgConfidence * 100).toFixed(1) + '%',
      learnedPhrases: this.voiceProfile.commonPhrases.size,
      currentThreshold: (this.getDynamicThreshold({ isWaitingForCommand: true }) * 100).toFixed(1) + '%',
      adaptiveBonus: (this.thresholds.adaptiveAdjustment * 100).toFixed(1) + '%',
      predictedNextCommand: this.commandPredictor.predictedNextCommand,
    };
  }

  /**
   * Reset profile (for testing or if user wants fresh start)
   */
  resetProfile() {
    localStorage.removeItem('jarvis_adaptive_voice_profile');
    window.location.reload();
  }
}

// Export singleton instance
const adaptiveVoiceDetection = new AdaptiveVoiceDetection();
export default adaptiveVoiceDetection;
