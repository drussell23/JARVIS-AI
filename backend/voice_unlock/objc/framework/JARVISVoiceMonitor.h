/**
 * JARVISVoiceMonitor.h
 * JARVIS Voice Unlock System
 *
 * Continuous background voice monitoring service that captures
 * and processes audio input when screen is locked.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

// Audio processing modes
typedef NS_ENUM(NSInteger, JARVISAudioProcessingMode) {
    JARVISAudioProcessingModeNormal = 0,
    JARVISAudioProcessingModeHighSensitivity,
    JARVISAudioProcessingModeLowLatency,
    JARVISAudioProcessingModeNoiseCancellation
};

// Voice activity detection state
typedef NS_ENUM(NSInteger, JARVISVoiceActivityState) {
    JARVISVoiceActivityStateIdle = 0,
    JARVISVoiceActivityStateListening,
    JARVISVoiceActivityStateDetecting,
    JARVISVoiceActivityStateProcessing,
    JARVISVoiceActivityStateTimeout
};

// Audio buffer info
@interface JARVISAudioBufferInfo : NSObject
@property (nonatomic, readonly) NSTimeInterval timestamp;
@property (nonatomic, readonly) NSTimeInterval duration;
@property (nonatomic, readonly) float averagePower;
@property (nonatomic, readonly) float peakPower;
@property (nonatomic, readonly) BOOL containsVoice;
@property (nonatomic, readonly) float voiceConfidence;
@end

// Voice monitor delegate
@protocol JARVISVoiceMonitorDelegate <NSObject>
@optional
- (void)voiceMonitorDidStartListening;
- (void)voiceMonitorDidStopListening;
- (void)voiceMonitorDidDetectVoice:(JARVISAudioBufferInfo *)bufferInfo;
- (void)voiceMonitorDidTimeout;
- (void)voiceMonitorDidEncounterError:(NSError *)error;
- (void)voiceMonitorAudioLevel:(float)level;
@end

// Main voice monitor interface
@interface JARVISVoiceMonitor : NSObject

@property (nonatomic, weak, nullable) id<JARVISVoiceMonitorDelegate> delegate;
@property (nonatomic, copy, nullable) void (^audioDetectedBlock)(NSData *audioData);

// State
@property (nonatomic, readonly) BOOL isMonitoring;
@property (nonatomic, readonly) JARVISVoiceActivityState activityState;
@property (nonatomic, readonly) float currentAudioLevel;

// Configuration
@property (nonatomic, assign) JARVISAudioProcessingMode processingMode;
@property (nonatomic, assign) NSTimeInterval silenceTimeout;
@property (nonatomic, assign) NSTimeInterval maxRecordingDuration;
@property (nonatomic, assign) float voiceDetectionThreshold;
@property (nonatomic, assign) float noiseFloorThreshold;
@property (nonatomic, assign) BOOL enableVoiceActivityDetection;
@property (nonatomic, assign) BOOL enableNoiseReduction;

// Control methods
- (BOOL)startMonitoring;
- (void)stopMonitoring;
- (void)pauseMonitoring;
- (void)resumeMonitoring;

// Mode management
- (void)setHighSensitivityMode:(BOOL)enabled;
- (void)setNoiseCancellationMode:(BOOL)enabled;

// Audio processing
- (void)processAudioBuffer:(AVAudioPCMBuffer *)buffer atTime:(AVAudioTime *)when;
- (NSData *)getAudioDataFromBuffer:(AVAudioPCMBuffer *)buffer;

// Voice activity detection
- (BOOL)detectVoiceInBuffer:(AVAudioPCMBuffer *)buffer;
- (float)calculateVoiceConfidence:(AVAudioPCMBuffer *)buffer;

// Audio level monitoring
- (float)calculateAverageLevel:(AVAudioPCMBuffer *)buffer;
- (float)calculatePeakLevel:(AVAudioPCMBuffer *)buffer;

// Configuration presets
- (void)applyPresetForEnvironment:(NSString *)environment; // "quiet", "normal", "noisy"

@end

NS_ASSUME_NONNULL_END