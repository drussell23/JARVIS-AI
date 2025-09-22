/**
 * JARVISVoiceAuthenticator.h
 * JARVIS Voice Unlock System
 *
 * Core voice authentication engine that performs voiceprint matching
 * and anti-spoofing detection.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

// Authentication result keys
extern NSString *const JARVISAuthResultSuccessKey;
extern NSString *const JARVISAuthResultConfidenceKey;
extern NSString *const JARVISAuthResultUserIDKey;
extern NSString *const JARVISAuthResultReasonKey;
extern NSString *const JARVISAuthResultLivenessScoreKey;
extern NSString *const JARVISAuthResultAntispoofingScoreKey;

// Authentication failure reasons
typedef NS_ENUM(NSInteger, JARVISAuthFailureReason) {
    JARVISAuthFailureReasonUnknown = 0,
    JARVISAuthFailureReasonNoMatch,
    JARVISAuthFailureReasonLowConfidence,
    JARVISAuthFailureReasonLivenessFailed,
    JARVISAuthFailureReasonSpoofingDetected,
    JARVISAuthFailureReasonNoiseLevel,
    JARVISAuthFailureReasonAudioQuality,
    JARVISAuthFailureReasonTimeout
};

// Voiceprint model
@interface JARVISVoiceprint : NSObject <NSSecureCoding>

@property (nonatomic, readonly) NSString *userID;
@property (nonatomic, readonly) NSString *userName;
@property (nonatomic, readonly) NSDate *createdDate;
@property (nonatomic, readonly) NSDate *lastUpdated;
@property (nonatomic, readonly) NSArray<NSNumber *> *features;
@property (nonatomic, readonly) NSUInteger sampleCount;
@property (nonatomic, readonly) float qualityScore;

- (instancetype)initWithUserID:(NSString *)userID
                      userName:(NSString *)userName
                      features:(NSArray<NSNumber *> *)features;

- (BOOL)updateWithNewFeatures:(NSArray<NSNumber *> *)features;

@end

// Main authenticator interface
@interface JARVISVoiceAuthenticator : NSObject

// Configuration
@property (nonatomic, assign) float confidenceThreshold;
@property (nonatomic, assign) float livenessThreshold;
@property (nonatomic, assign) float antispoofingThreshold;
@property (nonatomic, assign) BOOL enableAdaptiveLearning;
@property (nonatomic, assign) BOOL enableLivenessDetection;
@property (nonatomic, assign) BOOL enableAntispoofing;

// Voiceprint management
- (BOOL)loadVoiceprintForUser:(NSString *)userID error:(NSError **)error;
- (BOOL)hasVoiceprintForUser:(NSString *)userID;
- (NSArray<NSString *> *)enrolledUsers;
- (BOOL)deleteVoiceprintForUser:(NSString *)userID;

// Authentication
- (NSDictionary *)authenticateVoice:(NSData *)audioData forUser:(nullable NSString *)userID;
- (NSDictionary *)authenticateAudioBuffer:(AVAudioPCMBuffer *)buffer forUser:(nullable NSString *)userID;

// Feature extraction (for enrollment)
- (NSArray<NSNumber *> *)extractFeaturesFromAudio:(NSData *)audioData;
- (NSArray<NSNumber *> *)extractFeaturesFromBuffer:(AVAudioPCMBuffer *)buffer;

// Audio quality checks
- (NSDictionary *)analyzeAudioQuality:(NSData *)audioData;
- (BOOL)isAudioSuitableForAuthentication:(NSData *)audioData;

// Anti-spoofing
- (float)calculateLivenessScore:(NSData *)audioData;
- (float)calculateAntispoofingScore:(NSData *)audioData;

// Adaptive learning
- (void)updateVoiceprintWithSuccessfulAuth:(NSData *)audioData forUser:(NSString *)userID;

@end

NS_ASSUME_NONNULL_END