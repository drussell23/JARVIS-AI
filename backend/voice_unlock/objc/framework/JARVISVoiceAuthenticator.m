/**
 * JARVISVoiceAuthenticator.m
 * JARVIS Voice Unlock System
 *
 * Implementation of the voice authentication engine.
 */

#import "JARVISVoiceAuthenticator.h"
#import <Accelerate/Accelerate.h>
#import <AudioToolbox/AudioToolbox.h>
#import <os/log.h>

// Result keys
NSString *const JARVISAuthResultSuccessKey = @"success";
NSString *const JARVISAuthResultConfidenceKey = @"confidence";
NSString *const JARVISAuthResultUserIDKey = @"userID";
NSString *const JARVISAuthResultReasonKey = @"reason";
NSString *const JARVISAuthResultLivenessScoreKey = @"livenessScore";
NSString *const JARVISAuthResultAntispoofingScoreKey = @"antispoofingScore";

// Private constants
static const NSInteger kFeatureVectorSize = 128;
static const float kDefaultConfidenceThreshold = 0.85f;
static const float kDefaultLivenessThreshold = 0.7f;
static const float kDefaultAntispoofingThreshold = 0.8f;

#pragma mark - Voiceprint Implementation

@interface JARVISVoiceprint ()
@property (nonatomic, readwrite) NSString *userID;
@property (nonatomic, readwrite) NSString *userName;
@property (nonatomic, readwrite) NSDate *createdDate;
@property (nonatomic, readwrite) NSDate *lastUpdated;
@property (nonatomic, readwrite) NSArray<NSNumber *> *features;
@property (nonatomic, readwrite) NSUInteger sampleCount;
@property (nonatomic, readwrite) float qualityScore;
@end

@implementation JARVISVoiceprint

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (instancetype)initWithUserID:(NSString *)userID
                      userName:(NSString *)userName
                      features:(NSArray<NSNumber *> *)features {
    self = [super init];
    if (self) {
        _userID = [userID copy];
        _userName = [userName copy];
        _features = [features copy];
        _createdDate = [NSDate date];
        _lastUpdated = _createdDate;
        _sampleCount = 1;
        _qualityScore = 1.0f;
    }
    return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder {
    self = [super init];
    if (self) {
        _userID = [coder decodeObjectOfClass:[NSString class] forKey:@"userID"];
        _userName = [coder decodeObjectOfClass:[NSString class] forKey:@"userName"];
        _createdDate = [coder decodeObjectOfClass:[NSDate class] forKey:@"createdDate"];
        _lastUpdated = [coder decodeObjectOfClass:[NSDate class] forKey:@"lastUpdated"];
        _features = [coder decodeObjectOfClass:[NSArray class] forKey:@"features"];
        _sampleCount = [coder decodeIntegerForKey:@"sampleCount"];
        _qualityScore = [coder decodeFloatForKey:@"qualityScore"];
    }
    return self;
}

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeObject:self.userID forKey:@"userID"];
    [coder encodeObject:self.userName forKey:@"userName"];
    [coder encodeObject:self.createdDate forKey:@"createdDate"];
    [coder encodeObject:self.lastUpdated forKey:@"lastUpdated"];
    [coder encodeObject:self.features forKey:@"features"];
    [coder encodeInteger:self.sampleCount forKey:@"sampleCount"];
    [coder encodeFloat:self.qualityScore forKey:@"qualityScore"];
}

- (BOOL)updateWithNewFeatures:(NSArray<NSNumber *> *)features {
    if (features.count != self.features.count) {
        return NO;
    }
    
    // Adaptive averaging with existing features
    NSMutableArray *updatedFeatures = [NSMutableArray arrayWithCapacity:features.count];
    float alpha = 0.1f; // Learning rate
    
    for (NSUInteger i = 0; i < features.count; i++) {
        float oldValue = [self.features[i] floatValue];
        float newValue = [features[i] floatValue];
        float updated = oldValue * (1.0f - alpha) + newValue * alpha;
        [updatedFeatures addObject:@(updated)];
    }
    
    self.features = [updatedFeatures copy];
    self.lastUpdated = [NSDate date];
    self.sampleCount++;
    
    return YES;
}

@end

#pragma mark - Voice Authenticator Implementation

@interface JARVISVoiceAuthenticator ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, JARVISVoiceprint *> *voiceprints;
@property (nonatomic, strong) dispatch_queue_t processingQueue;
@property (nonatomic, strong) os_log_t logger;
@end

@implementation JARVISVoiceAuthenticator

- (instancetype)init {
    self = [super init];
    if (self) {
        _voiceprints = [NSMutableDictionary dictionary];
        _processingQueue = dispatch_queue_create("com.jarvis.voiceauth.processing", DISPATCH_QUEUE_SERIAL);
        _logger = os_log_create("com.jarvis.voiceunlock", "authenticator");
        
        // Set default thresholds
        _confidenceThreshold = kDefaultConfidenceThreshold;
        _livenessThreshold = kDefaultLivenessThreshold;
        _antispoofingThreshold = kDefaultAntispoofingThreshold;
        
        // Enable features by default
        _enableAdaptiveLearning = YES;
        _enableLivenessDetection = YES;
        _enableAntispoofing = YES;
        
        // Load all voiceprints
        [self loadAllVoiceprints];
    }
    return self;
}

#pragma mark - Voiceprint Management

- (void)loadAllVoiceprints {
    NSString *voiceprintDir = [@"~/.jarvis/voice_unlock" stringByExpandingTildeInPath];
    NSFileManager *fm = [NSFileManager defaultManager];
    
    NSError *error = nil;
    NSArray *files = [fm contentsOfDirectoryAtPath:voiceprintDir error:&error];
    
    if (error) {
        os_log_error(self.logger, "Failed to list voiceprint directory: %@", error);
        return;
    }
    
    for (NSString *filename in files) {
        if ([filename hasSuffix:@"_voiceprint.json"]) {
            NSString *userID = [filename stringByReplacingOccurrencesOfString:@"_voiceprint.json" withString:@""];
            [self loadVoiceprintForUser:userID error:nil];
        }
    }
}

- (BOOL)loadVoiceprintForUser:(NSString *)userID error:(NSError **)error {
    NSString *path = [NSString stringWithFormat:@"~/.jarvis/voice_unlock/%@_voiceprint.json", userID];
    path = [path stringByExpandingTildeInPath];
    
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISVoiceAuth" 
                                         code:404 
                                     userInfo:@{NSLocalizedDescriptionKey: @"Voiceprint file not found"}];
        }
        return NO;
    }
    
    NSError *jsonError = nil;
    NSDictionary *voiceprintData = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
    
    if (jsonError) {
        if (error) *error = jsonError;
        return NO;
    }
    
    // Convert JSON to voiceprint
    NSDictionary *vpData = voiceprintData[@"voiceprint"];
    NSArray *features = vpData[@"features"];
    
    if (!features || features.count != kFeatureVectorSize) {
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISVoiceAuth" 
                                         code:400 
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid voiceprint format"}];
        }
        return NO;
    }
    
    JARVISVoiceprint *voiceprint = [[JARVISVoiceprint alloc] initWithUserID:userID
                                                                    userName:voiceprintData[@"name"]
                                                                    features:features];
    
    self.voiceprints[userID] = voiceprint;
    os_log_info(self.logger, "Loaded voiceprint for user: %@", userID);
    
    return YES;
}

- (BOOL)hasVoiceprintForUser:(NSString *)userID {
    return self.voiceprints[userID] != nil;
}

- (NSArray<NSString *> *)enrolledUsers {
    return [self.voiceprints allKeys];
}

- (BOOL)deleteVoiceprintForUser:(NSString *)userID {
    [self.voiceprints removeObjectForKey:userID];
    
    NSString *path = [NSString stringWithFormat:@"~/.jarvis/voice_unlock/%@_voiceprint.json", userID];
    path = [path stringByExpandingTildeInPath];
    
    NSError *error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:path error:&error];
    
    return error == nil;
}

#pragma mark - Authentication

- (NSDictionary *)authenticateVoice:(NSData *)audioData forUser:(nullable NSString *)userID {
    // Check audio quality first
    if (![self isAudioSuitableForAuthentication:audioData]) {
        return @{
            JARVISAuthResultSuccessKey: @NO,
            JARVISAuthResultReasonKey: @(JARVISAuthFailureReasonAudioQuality)
        };
    }
    
    // Extract features from audio
    NSArray<NSNumber *> *features = [self extractFeaturesFromAudio:audioData];
    if (!features || features.count != kFeatureVectorSize) {
        return @{
            JARVISAuthResultSuccessKey: @NO,
            JARVISAuthResultReasonKey: @(JARVISAuthFailureReasonUnknown)
        };
    }
    
    // If no specific user, try all enrolled users
    NSArray<NSString *> *usersToCheck = userID ? @[userID] : [self.voiceprints allKeys];
    
    float bestConfidence = 0.0f;
    NSString *bestMatchUserID = nil;
    
    for (NSString *checkUserID in usersToCheck) {
        JARVISVoiceprint *voiceprint = self.voiceprints[checkUserID];
        if (!voiceprint) continue;
        
        float confidence = [self calculateConfidence:features against:voiceprint.features];
        if (confidence > bestConfidence) {
            bestConfidence = confidence;
            bestMatchUserID = checkUserID;
        }
    }
    
    // Check if we found a match
    if (!bestMatchUserID || bestConfidence < self.confidenceThreshold) {
        return @{
            JARVISAuthResultSuccessKey: @NO,
            JARVISAuthResultReasonKey: @(JARVISAuthFailureReasonNoMatch),
            JARVISAuthResultConfidenceKey: @(bestConfidence)
        };
    }
    
    NSMutableDictionary *result = [NSMutableDictionary dictionaryWithDictionary:@{
        JARVISAuthResultSuccessKey: @YES,
        JARVISAuthResultUserIDKey: bestMatchUserID,
        JARVISAuthResultConfidenceKey: @(bestConfidence)
    }];
    
    // Perform liveness detection
    if (self.enableLivenessDetection) {
        float livenessScore = [self calculateLivenessScore:audioData];
        result[JARVISAuthResultLivenessScoreKey] = @(livenessScore);
        
        if (livenessScore < self.livenessThreshold) {
            result[JARVISAuthResultSuccessKey] = @NO;
            result[JARVISAuthResultReasonKey] = @(JARVISAuthFailureReasonLivenessFailed);
            return result;
        }
    }
    
    // Perform anti-spoofing
    if (self.enableAntispoofing) {
        float antispoofingScore = [self calculateAntispoofingScore:audioData];
        result[JARVISAuthResultAntispoofingScoreKey] = @(antispoofingScore);
        
        if (antispoofingScore < self.antispoofingThreshold) {
            result[JARVISAuthResultSuccessKey] = @NO;
            result[JARVISAuthResultReasonKey] = @(JARVISAuthFailureReasonSpoofingDetected);
            return result;
        }
    }
    
    // Update voiceprint if adaptive learning is enabled
    if (self.enableAdaptiveLearning && [result[JARVISAuthResultSuccessKey] boolValue]) {
        dispatch_async(self.processingQueue, ^{
            [self updateVoiceprintWithSuccessfulAuth:audioData forUser:bestMatchUserID];
        });
    }
    
    return result;
}

- (NSDictionary *)authenticateAudioBuffer:(AVAudioPCMBuffer *)buffer forUser:(nullable NSString *)userID {
    // Convert buffer to NSData
    NSData *audioData = [self dataFromAudioBuffer:buffer];
    return [self authenticateVoice:audioData forUser:userID];
}

#pragma mark - Feature Extraction

- (NSArray<NSNumber *> *)extractFeaturesFromAudio:(NSData *)audioData {
    // Convert audio data to float samples
    const int16_t *samples = (const int16_t *)audioData.bytes;
    NSUInteger sampleCount = audioData.length / sizeof(int16_t);
    
    if (sampleCount < 1024) {
        return nil; // Not enough samples
    }
    
    // Normalize to float
    float *floatSamples = malloc(sampleCount * sizeof(float));
    vDSP_vflt16(samples, 1, floatSamples, 1, sampleCount);
    float scale = 1.0f / 32768.0f;
    vDSP_vsmul(floatSamples, 1, &scale, floatSamples, 1, sampleCount);
    
    // Extract MFCC features (simplified version)
    NSMutableArray<NSNumber *> *features = [NSMutableArray arrayWithCapacity:kFeatureVectorSize];
    
    // Window size for FFT
    const NSUInteger fftSize = 1024;
    const NSUInteger hopSize = 512;
    const NSUInteger numFrames = (sampleCount - fftSize) / hopSize + 1;
    
    // Process frames and extract features
    float *mfccFeatures = calloc(kFeatureVectorSize, sizeof(float));
    
    for (NSUInteger frame = 0; frame < MIN(numFrames, 20); frame++) {
        NSUInteger startIdx = frame * hopSize;
        
        // Apply Hamming window
        float *windowedFrame = malloc(fftSize * sizeof(float));
        vDSP_hamm_window(windowedFrame, fftSize, 0);
        vDSP_vmul(&floatSamples[startIdx], 1, windowedFrame, 1, windowedFrame, 1, fftSize);
        
        // Compute power spectrum
        [self computeMFCCFeatures:windowedFrame length:fftSize output:mfccFeatures];
        
        free(windowedFrame);
    }
    
    // Normalize features
    float maxVal = 0.0f;
    vDSP_maxv(mfccFeatures, 1, &maxVal, kFeatureVectorSize);
    if (maxVal > 0) {
        vDSP_vsdiv(mfccFeatures, 1, &maxVal, mfccFeatures, 1, kFeatureVectorSize);
    }
    
    // Convert to NSArray
    for (NSUInteger i = 0; i < kFeatureVectorSize; i++) {
        [features addObject:@(mfccFeatures[i])];
    }
    
    free(floatSamples);
    free(mfccFeatures);
    
    return features;
}

- (NSArray<NSNumber *> *)extractFeaturesFromBuffer:(AVAudioPCMBuffer *)buffer {
    NSData *audioData = [self dataFromAudioBuffer:buffer];
    return [self extractFeaturesFromAudio:audioData];
}

- (void)computeMFCCFeatures:(float *)frame length:(NSUInteger)length output:(float *)features {
    // Simplified MFCC computation
    // In a production system, use a proper DSP library
    
    // For now, compute basic spectral features
    float *spectrum = malloc(length * sizeof(float));
    
    // Simple FFT magnitude
    vDSP_vsq(frame, 1, spectrum, 1, length);
    
    // Mel-scale filterbank (simplified)
    const NSUInteger numFilters = 26;
    float *filterOutput = malloc(numFilters * sizeof(float));
    
    for (NSUInteger i = 0; i < numFilters; i++) {
        NSUInteger startBin = (i * length) / (numFilters * 2);
        NSUInteger endBin = ((i + 1) * length) / (numFilters * 2);
        
        float sum = 0.0f;
        vDSP_sve(&spectrum[startBin], 1, &sum, endBin - startBin);
        filterOutput[i] = logf(sum + 1e-10f);
    }
    
    // DCT to get MFCCs (simplified)
    for (NSUInteger i = 0; i < MIN(numFilters, kFeatureVectorSize); i++) {
        features[i] += filterOutput[i] * 0.1f; // Accumulate across frames
    }
    
    free(spectrum);
    free(filterOutput);
}

#pragma mark - Similarity Calculation

- (float)calculateConfidence:(NSArray<NSNumber *> *)features1 against:(NSArray<NSNumber *> *)features2 {
    if (features1.count != features2.count) {
        return 0.0f;
    }
    
    // Convert to float arrays
    NSUInteger count = features1.count;
    float *f1 = malloc(count * sizeof(float));
    float *f2 = malloc(count * sizeof(float));
    
    for (NSUInteger i = 0; i < count; i++) {
        f1[i] = [features1[i] floatValue];
        f2[i] = [features2[i] floatValue];
    }
    
    // Calculate cosine similarity
    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    vDSP_dotpr(f1, 1, f2, 1, &dotProduct, count);
    vDSP_svesq(f1, 1, &norm1, count);
    vDSP_svesq(f2, 1, &norm2, count);
    
    free(f1);
    free(f2);
    
    if (norm1 > 0 && norm2 > 0) {
        return dotProduct / (sqrtf(norm1) * sqrtf(norm2));
    }
    
    return 0.0f;
}

#pragma mark - Audio Quality

- (NSDictionary *)analyzeAudioQuality:(NSData *)audioData {
    const int16_t *samples = (const int16_t *)audioData.bytes;
    NSUInteger sampleCount = audioData.length / sizeof(int16_t);
    
    if (sampleCount == 0) {
        return @{@"suitable": @NO, @"reason": @"No audio data"};
    }
    
    // Calculate RMS (root mean square) for volume check
    float rms = 0.0f;
    vDSP_measqv((const float *)samples, 1, &rms, sampleCount);
    rms = sqrtf(rms / sampleCount);
    
    // Calculate SNR estimate
    float signal = 0.0f;
    float noise = 0.0f;
    
    // Simple noise floor estimation (first 10% of samples)
    NSUInteger noiseLength = sampleCount / 10;
    vDSP_measqv((const float *)samples, 1, &noise, noiseLength);
    noise = sqrtf(noise / noiseLength);
    
    // Signal estimation (middle 50% of samples)
    NSUInteger signalStart = sampleCount / 4;
    NSUInteger signalLength = sampleCount / 2;
    vDSP_measqv((const float *)&samples[signalStart], 1, &signal, signalLength);
    signal = sqrtf(signal / signalLength);
    
    float snr = (noise > 0) ? 20 * log10f(signal / noise) : 0.0f;
    
    return @{
        @"rms": @(rms),
        @"snr": @(snr),
        @"duration": @(sampleCount / 16000.0), // Assuming 16kHz
        @"suitable": @(rms > 0.01f && snr > 10.0f)
    };
}

- (BOOL)isAudioSuitableForAuthentication:(NSData *)audioData {
    NSDictionary *quality = [self analyzeAudioQuality:audioData];
    return [quality[@"suitable"] boolValue];
}

#pragma mark - Anti-spoofing

- (float)calculateLivenessScore:(NSData *)audioData {
    // Simplified liveness detection
    // In production, use advanced techniques like:
    // - Formant analysis
    // - Pitch variation
    // - Breathing patterns
    // - Micro-modulations
    
    NSDictionary *quality = [self analyzeAudioQuality:audioData];
    float snr = [quality[@"snr"] floatValue];
    
    // Higher SNR often indicates real speech vs playback
    float livenessScore = MIN(1.0f, snr / 40.0f);
    
    // Add some randomness for now (replace with real analysis)
    float variation = (arc4random_uniform(100) / 100.0f) * 0.2f;
    livenessScore = livenessScore * 0.8f + variation;
    
    return MIN(1.0f, MAX(0.0f, livenessScore));
}

- (float)calculateAntispoofingScore:(NSData *)audioData {
    // Simplified anti-spoofing detection
    // In production, implement:
    // - Replay attack detection
    // - Voice synthesis detection
    // - Channel characteristics analysis
    
    float livenessScore = [self calculateLivenessScore:audioData];
    
    // For now, use liveness score with some modification
    float antispoofingScore = livenessScore * 0.9f + 0.1f;
    
    return MIN(1.0f, MAX(0.0f, antispoofingScore));
}

#pragma mark - Adaptive Learning

- (void)updateVoiceprintWithSuccessfulAuth:(NSData *)audioData forUser:(NSString *)userID {
    JARVISVoiceprint *voiceprint = self.voiceprints[userID];
    if (!voiceprint || !self.enableAdaptiveLearning) {
        return;
    }
    
    // Extract features from successful authentication
    NSArray<NSNumber *> *newFeatures = [self extractFeaturesFromAudio:audioData];
    if (!newFeatures || newFeatures.count != kFeatureVectorSize) {
        return;
    }
    
    // Update voiceprint with new features
    if ([voiceprint updateWithNewFeatures:newFeatures]) {
        os_log_info(self.logger, "Updated voiceprint for user %@ (sample %lu)", 
                    userID, (unsigned long)voiceprint.sampleCount);
        
        // Optionally save updated voiceprint to disk
        [self saveVoiceprint:voiceprint];
    }
}

- (void)saveVoiceprint:(JARVISVoiceprint *)voiceprint {
    NSString *path = [NSString stringWithFormat:@"~/.jarvis/voice_unlock/%@_voiceprint.json", voiceprint.userID];
    path = [path stringByExpandingTildeInPath];
    
    NSDictionary *data = @{
        @"user_id": voiceprint.userID,
        @"name": voiceprint.userName,
        @"created": [self.dateFormatter stringFromDate:voiceprint.createdDate],
        @"updated": [self.dateFormatter stringFromDate:voiceprint.lastUpdated],
        @"voiceprint": @{
            @"features": voiceprint.features,
            @"sample_count": @(voiceprint.sampleCount),
            @"quality_score": @(voiceprint.qualityScore)
        }
    };
    
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:data options:NSJSONWritingPrettyPrinted error:&error];
    
    if (!error && jsonData) {
        [jsonData writeToFile:path atomically:YES];
    }
}

#pragma mark - Helpers

- (NSData *)dataFromAudioBuffer:(AVAudioPCMBuffer *)buffer {
    const AudioBufferList *abl = buffer.audioBufferList;
    NSMutableData *data = [NSMutableData data];
    
    for (UInt32 i = 0; i < abl->mNumberBuffers; i++) {
        [data appendBytes:abl->mBuffers[i].mData length:abl->mBuffers[i].mDataByteSize];
    }
    
    return data;
}

- (NSDateFormatter *)dateFormatter {
    static NSDateFormatter *formatter = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        formatter = [[NSDateFormatter alloc] init];
        formatter.dateFormat = @"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'";
        formatter.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
    });
    return formatter;
}

@end