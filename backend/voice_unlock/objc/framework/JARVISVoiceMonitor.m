/**
 * JARVISVoiceMonitor.m
 * JARVIS Voice Unlock System
 *
 * Implementation of continuous background voice monitoring.
 */

#import "JARVISVoiceMonitor.h"
#import <Accelerate/Accelerate.h>
#import <os/log.h>

// Private constants
static const NSTimeInterval kDefaultSilenceTimeout = 3.0;
static const NSTimeInterval kDefaultMaxRecordingDuration = 30.0;
static const float kDefaultVoiceDetectionThreshold = 0.3f;
static const float kDefaultNoiseFloorThreshold = 0.01f;
static const NSUInteger kAudioBufferSize = 4096;
static const NSUInteger kVADWindowSize = 20; // 20ms windows for VAD

// Audio buffer info implementation
@interface JARVISAudioBufferInfo ()
@property (nonatomic, readwrite) NSTimeInterval timestamp;
@property (nonatomic, readwrite) NSTimeInterval duration;
@property (nonatomic, readwrite) float averagePower;
@property (nonatomic, readwrite) float peakPower;
@property (nonatomic, readwrite) BOOL containsVoice;
@property (nonatomic, readwrite) float voiceConfidence;
@end

@implementation JARVISAudioBufferInfo
@end

// Main implementation
@interface JARVISVoiceMonitor ()

@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) AVAudioInputNode *inputNode;
@property (nonatomic, strong) AVAudioFormat *recordingFormat;
@property (nonatomic, strong) AVAudioMixerNode *mixerNode;

@property (nonatomic, strong) dispatch_queue_t processingQueue;
@property (nonatomic, strong) NSMutableData *audioBuffer;
@property (nonatomic, strong) NSTimer *silenceTimer;
@property (nonatomic, strong) NSDate *recordingStartTime;

@property (nonatomic, readwrite) BOOL isMonitoring;
@property (nonatomic, readwrite) JARVISVoiceActivityState activityState;
@property (nonatomic, readwrite) float currentAudioLevel;

@property (nonatomic, strong) os_log_t logger;

// VAD (Voice Activity Detection) properties
@property (nonatomic, strong) NSMutableArray<NSNumber *> *energyHistory;
@property (nonatomic, assign) float adaptiveThreshold;
@property (nonatomic, assign) NSUInteger voiceFrameCount;
@property (nonatomic, assign) NSUInteger silenceFrameCount;

@end

@implementation JARVISVoiceMonitor

- (instancetype)init {
    self = [super init];
    if (self) {
        _audioEngine = [[AVAudioEngine alloc] init];
        _inputNode = [_audioEngine inputNode];
        _mixerNode = [[AVAudioMixerNode alloc] init];
        [_audioEngine attachNode:_mixerNode];
        
        _processingQueue = dispatch_queue_create("com.jarvis.voicemonitor", DISPATCH_QUEUE_SERIAL);
        _audioBuffer = [NSMutableData data];
        _logger = os_log_create("com.jarvis.voiceunlock", "monitor");
        
        // Default configuration
        _processingMode = JARVISAudioProcessingModeNormal;
        _silenceTimeout = kDefaultSilenceTimeout;
        _maxRecordingDuration = kDefaultMaxRecordingDuration;
        _voiceDetectionThreshold = kDefaultVoiceDetectionThreshold;
        _noiseFloorThreshold = kDefaultNoiseFloorThreshold;
        _enableVoiceActivityDetection = YES;
        _enableNoiseReduction = YES;
        
        // VAD initialization
        _energyHistory = [NSMutableArray array];
        _adaptiveThreshold = kDefaultVoiceDetectionThreshold;
        
        // Register for audio session notifications
        [self setupAudioSession];
        [self registerForNotifications];
    }
    return self;
}

- (void)dealloc {
    [self stopMonitoring];
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

#pragma mark - Audio Session Setup

- (void)setupAudioSession {
    // macOS doesn't use AVAudioSession - audio routing is handled differently
    // Just log that we're ready
    os_log_info(self.logger, "Audio setup complete for macOS");
}

- (void)registerForNotifications {
    // Register for macOS-specific audio notifications
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(handleAudioDeviceChange:)
                                                 name:@"AVAudioEngineConfigurationChangeNotification"
                                               object:nil];
}

#pragma mark - Monitoring Control

- (BOOL)startMonitoring {
    if (self.isMonitoring) {
        return YES;
    }
    
    os_log_info(self.logger, "Starting voice monitoring");
    
    NSError *error = nil;
    
    // Get the audio format
    self.recordingFormat = [self.inputNode inputFormatForBus:0];
    
    // Connect nodes
    [self.audioEngine connect:self.inputNode
                           to:self.mixerNode
                       format:self.recordingFormat];
    
    // Install tap on mixer node
    __weak typeof(self) weakSelf = self;
    [self.mixerNode installTapOnBus:0
                         bufferSize:kAudioBufferSize
                             format:self.recordingFormat
                              block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
        [weakSelf processAudioBuffer:buffer atTime:when];
    }];
    
    // Start audio engine
    [self.audioEngine prepare];
    if (![self.audioEngine startAndReturnError:&error]) {
        os_log_error(self.logger, "Failed to start audio engine: %@", error);
        if ([self.delegate respondsToSelector:@selector(voiceMonitorDidEncounterError:)]) {
            [self.delegate voiceMonitorDidEncounterError:error];
        }
        return NO;
    }
    
    self.isMonitoring = YES;
    self.activityState = JARVISVoiceActivityStateListening;
    self.recordingStartTime = [NSDate date];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([self.delegate respondsToSelector:@selector(voiceMonitorDidStartListening)]) {
            [self.delegate voiceMonitorDidStartListening];
        }
    });
    
    return YES;
}

- (void)stopMonitoring {
    if (!self.isMonitoring) {
        return;
    }
    
    os_log_info(self.logger, "Stopping voice monitoring");
    
    [self.silenceTimer invalidate];
    self.silenceTimer = nil;
    
    [self.mixerNode removeTapOnBus:0];
    [self.audioEngine stop];
    
    self.isMonitoring = NO;
    self.activityState = JARVISVoiceActivityStateIdle;
    [self.audioBuffer setLength:0];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([self.delegate respondsToSelector:@selector(voiceMonitorDidStopListening)]) {
            [self.delegate voiceMonitorDidStopListening];
        }
    });
}

- (void)pauseMonitoring {
    if (self.isMonitoring) {
        [self.audioEngine pause];
        self.activityState = JARVISVoiceActivityStateIdle;
    }
}

- (void)resumeMonitoring {
    if (self.isMonitoring) {
        NSError *error = nil;
        if ([self.audioEngine startAndReturnError:&error]) {
            self.activityState = JARVISVoiceActivityStateListening;
        } else {
            os_log_error(self.logger, "Failed to resume monitoring: %@", error);
        }
    }
}

#pragma mark - Mode Management

- (void)setHighSensitivityMode:(BOOL)enabled {
    if (enabled) {
        self.processingMode = JARVISAudioProcessingModeHighSensitivity;
        self.voiceDetectionThreshold = 0.1f;
        self.noiseFloorThreshold = 0.005f;
        self.silenceTimeout = 5.0;
    } else {
        self.processingMode = JARVISAudioProcessingModeNormal;
        self.voiceDetectionThreshold = kDefaultVoiceDetectionThreshold;
        self.noiseFloorThreshold = kDefaultNoiseFloorThreshold;
        self.silenceTimeout = kDefaultSilenceTimeout;
    }
}

- (void)setNoiseCancellationMode:(BOOL)enabled {
    self.enableNoiseReduction = enabled;
    if (enabled) {
        self.processingMode = JARVISAudioProcessingModeNoiseCancellation;
    }
}

#pragma mark - Audio Processing

- (void)processAudioBuffer:(AVAudioPCMBuffer *)buffer atTime:(AVAudioTime *)when {
    dispatch_async(self.processingQueue, ^{
        // Calculate audio levels
        float averageLevel = [self calculateAverageLevel:buffer];
        float peakLevel = [self calculatePeakLevel:buffer];
        self.currentAudioLevel = averageLevel;
        
        // Update audio level on main thread
        dispatch_async(dispatch_get_main_queue(), ^{
            if ([self.delegate respondsToSelector:@selector(voiceMonitorAudioLevel:)]) {
                [self.delegate voiceMonitorAudioLevel:averageLevel];
            }
        });
        
        // Check for voice activity
        BOOL hasVoice = [self detectVoiceInBuffer:buffer];
        float confidence = hasVoice ? [self calculateVoiceConfidence:buffer] : 0.0f;
        
        // Create buffer info
        JARVISAudioBufferInfo *bufferInfo = [[JARVISAudioBufferInfo alloc] init];
        bufferInfo.timestamp = when.sampleTime / buffer.format.sampleRate;
        bufferInfo.duration = buffer.frameLength / buffer.format.sampleRate;
        bufferInfo.averagePower = averageLevel;
        bufferInfo.peakPower = peakLevel;
        bufferInfo.containsVoice = hasVoice;
        bufferInfo.voiceConfidence = confidence;
        
        if (hasVoice) {
            // Voice detected
            self.activityState = JARVISVoiceActivityStateDetecting;
            self.voiceFrameCount++;
            self.silenceFrameCount = 0;
            
            // Add audio data to buffer
            NSData *audioData = [self getAudioDataFromBuffer:buffer];
            [self.audioBuffer appendData:audioData];
            
            // Reset silence timer
            [self resetSilenceTimer];
            
            // Notify delegate
            dispatch_async(dispatch_get_main_queue(), ^{
                if ([self.delegate respondsToSelector:@selector(voiceMonitorDidDetectVoice:)]) {
                    [self.delegate voiceMonitorDidDetectVoice:bufferInfo];
                }
            });
            
            // Check recording duration
            NSTimeInterval recordingDuration = [[NSDate date] timeIntervalSinceDate:self.recordingStartTime];
            if (recordingDuration >= self.maxRecordingDuration) {
                [self handleRecordingComplete];
            }
            
        } else {
            // No voice detected
            self.silenceFrameCount++;
            
            if (self.activityState == JARVISVoiceActivityStateDetecting) {
                // Was detecting voice, now silence
                if (self.silenceFrameCount > 10) { // ~200ms of silence
                    self.activityState = JARVISVoiceActivityStateListening;
                }
            }
        }
    });
}

- (NSData *)getAudioDataFromBuffer:(AVAudioPCMBuffer *)buffer {
    const AudioBufferList *audioBufferList = buffer.audioBufferList;
    NSMutableData *data = [NSMutableData data];
    
    for (UInt32 i = 0; i < audioBufferList->mNumberBuffers; i++) {
        AudioBuffer audioBuffer = audioBufferList->mBuffers[i];
        [data appendBytes:audioBuffer.mData length:audioBuffer.mDataByteSize];
    }
    
    return data;
}

#pragma mark - Voice Activity Detection

- (BOOL)detectVoiceInBuffer:(AVAudioPCMBuffer *)buffer {
    if (!self.enableVoiceActivityDetection) {
        return YES; // Always assume voice if VAD is disabled
    }
    
    float energy = [self calculateAverageLevel:buffer];
    
    // Update energy history
    [self.energyHistory addObject:@(energy)];
    if (self.energyHistory.count > kVADWindowSize) {
        [self.energyHistory removeObjectAtIndex:0];
    }
    
    // Calculate adaptive threshold
    if (self.energyHistory.count >= kVADWindowSize) {
        float sum = 0.0f;
        for (NSNumber *value in self.energyHistory) {
            sum += [value floatValue];
        }
        float average = sum / self.energyHistory.count;
        
        // Update adaptive threshold (slowly)
        self.adaptiveThreshold = self.adaptiveThreshold * 0.95f + average * 1.5f * 0.05f;
    }
    
    // Check if energy exceeds threshold
    BOOL hasVoice = energy > MAX(self.voiceDetectionThreshold, self.adaptiveThreshold);
    
    // Additional checks for voice characteristics
    if (hasVoice && self.processingMode == JARVISAudioProcessingModeHighSensitivity) {
        // Check for voice-like patterns (simplified)
        float zcr = [self calculateZeroCrossingRate:buffer];
        hasVoice = hasVoice && (zcr > 0.1f && zcr < 0.5f);
    }
    
    return hasVoice;
}

- (float)calculateVoiceConfidence:(AVAudioPCMBuffer *)buffer {
    float energy = [self calculateAverageLevel:buffer];
    float zcr = [self calculateZeroCrossingRate:buffer];
    
    // Simple confidence calculation based on energy and ZCR
    float energyConfidence = MIN(1.0f, energy / 0.5f);
    float zcrConfidence = 1.0f - fabsf(zcr - 0.3f) / 0.3f; // Optimal ZCR around 0.3
    
    return (energyConfidence * 0.7f + zcrConfidence * 0.3f);
}

- (float)calculateZeroCrossingRate:(AVAudioPCMBuffer *)buffer {
    const float *samples = buffer.floatChannelData[0];
    NSUInteger frameLength = buffer.frameLength;
    
    NSUInteger crossings = 0;
    float previousSample = 0.0f;
    
    for (NSUInteger i = 0; i < frameLength; i++) {
        float currentSample = samples[i];
        if ((previousSample >= 0 && currentSample < 0) ||
            (previousSample < 0 && currentSample >= 0)) {
            crossings++;
        }
        previousSample = currentSample;
    }
    
    return (float)crossings / (float)frameLength;
}

#pragma mark - Audio Level Monitoring

- (float)calculateAverageLevel:(AVAudioPCMBuffer *)buffer {
    const float *samples = buffer.floatChannelData[0];
    NSUInteger frameLength = buffer.frameLength;
    
    float sum = 0.0f;
    vDSP_measqv(samples, 1, &sum, frameLength);
    
    return sqrtf(sum / frameLength);
}

- (float)calculatePeakLevel:(AVAudioPCMBuffer *)buffer {
    const float *samples = buffer.floatChannelData[0];
    NSUInteger frameLength = buffer.frameLength;
    
    float peak = 0.0f;
    vDSP_maxmgv(samples, 1, &peak, frameLength);
    
    return peak;
}

#pragma mark - Timer Management

- (void)resetSilenceTimer {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.silenceTimer invalidate];
        self.silenceTimer = [NSTimer scheduledTimerWithTimeInterval:self.silenceTimeout
                                                              target:self
                                                            selector:@selector(handleSilenceTimeout)
                                                            userInfo:nil
                                                             repeats:NO];
    });
}

- (void)handleSilenceTimeout {
    os_log_info(self.logger, "Silence timeout reached");
    
    if (self.audioBuffer.length > 0) {
        [self handleRecordingComplete];
    } else {
        self.activityState = JARVISVoiceActivityStateTimeout;
        if ([self.delegate respondsToSelector:@selector(voiceMonitorDidTimeout)]) {
            [self.delegate voiceMonitorDidTimeout];
        }
    }
}

- (void)handleRecordingComplete {
    if (self.audioBuffer.length == 0) {
        return;
    }
    
    self.activityState = JARVISVoiceActivityStateProcessing;
    
    // Copy audio data
    NSData *audioData = [self.audioBuffer copy];
    [self.audioBuffer setLength:0];
    self.recordingStartTime = [NSDate date];
    
    // Process in background
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        if (self.audioDetectedBlock) {
            self.audioDetectedBlock(audioData);
        }
        
        self.activityState = JARVISVoiceActivityStateListening;
    });
}

#pragma mark - Configuration Presets

- (void)applyPresetForEnvironment:(NSString *)environment {
    if ([environment isEqualToString:@"quiet"]) {
        self.voiceDetectionThreshold = 0.05f;
        self.noiseFloorThreshold = 0.001f;
        self.enableNoiseReduction = NO;
        self.processingMode = JARVISAudioProcessingModeHighSensitivity;
        
    } else if ([environment isEqualToString:@"normal"]) {
        self.voiceDetectionThreshold = kDefaultVoiceDetectionThreshold;
        self.noiseFloorThreshold = kDefaultNoiseFloorThreshold;
        self.enableNoiseReduction = YES;
        self.processingMode = JARVISAudioProcessingModeNormal;
        
    } else if ([environment isEqualToString:@"noisy"]) {
        self.voiceDetectionThreshold = 0.5f;
        self.noiseFloorThreshold = 0.1f;
        self.enableNoiseReduction = YES;
        self.processingMode = JARVISAudioProcessingModeNoiseCancellation;
    }
    
    os_log_info(self.logger, "Applied preset for %@ environment", environment);
}

#pragma mark - Audio Device Notifications

- (void)handleAudioDeviceChange:(NSNotification *)notification {
    os_log_info(self.logger, "Audio device configuration changed");
    
    // On macOS, we might need to reconfigure the audio engine
    if (self.isMonitoring) {
        // Restart monitoring with new configuration
        dispatch_async(self.processingQueue, ^{
            [self stopMonitoring];
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                [self startMonitoring];
            });
        });
    }
}

@end