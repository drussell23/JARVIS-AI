/**
 * JARVISVoiceUnlockDaemon.h
 * JARVIS Voice Unlock System
 *
 * Main daemon that runs in the background to monitor for voice unlock phrases
 * when the screen is locked. Integrates with macOS Security Framework.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <Security/Security.h>
#import <LocalAuthentication/LocalAuthentication.h>

NS_ASSUME_NONNULL_BEGIN

// Forward declarations
@class JARVISVoiceAuthenticator;
@class JARVISScreenUnlockManager;
@class JARVISVoiceMonitor;
@class JARVISPythonBridge;
@class JARVISPermissionManager;
@class JARVISWebSocketBridge;

// Notification constants
extern NSString *const JARVISVoiceUnlockStatusChangedNotification;
extern NSString *const JARVISVoiceUnlockAuthenticationFailedNotification;
extern NSString *const JARVISVoiceUnlockAuthenticationSucceededNotification;

// Error domain
extern NSString *const JARVISVoiceUnlockErrorDomain;

// Voice unlock state
typedef NS_ENUM(NSInteger, JARVISVoiceUnlockState) {
    JARVISVoiceUnlockStateInactive = 0,
    JARVISVoiceUnlockStateMonitoring,
    JARVISVoiceUnlockStateProcessing,
    JARVISVoiceUnlockStateUnlocking,
    JARVISVoiceUnlockStateError
};

// Configuration options
typedef NS_OPTIONS(NSUInteger, JARVISVoiceUnlockOptions) {
    JARVISVoiceUnlockOptionNone = 0,
    JARVISVoiceUnlockOptionEnableLivenessDetection = 1 << 0,
    JARVISVoiceUnlockOptionEnableAntiSpoofing = 1 << 1,
    JARVISVoiceUnlockOptionEnableAdaptiveThresholds = 1 << 2,
    JARVISVoiceUnlockOptionEnableContinuousAuthentication = 1 << 3,
    JARVISVoiceUnlockOptionEnableDebugLogging = 1 << 4
};

/**
 * Main daemon interface
 */
@interface JARVISVoiceUnlockDaemon : NSObject

// Singleton instance
+ (instancetype)sharedDaemon;

// Core properties
@property (nonatomic, readonly) JARVISVoiceUnlockState state;
@property (nonatomic, readonly) BOOL isMonitoring;
@property (nonatomic, readonly) BOOL isScreenLocked;
@property (nonatomic, readonly) NSString *enrolledUserIdentifier;
@property (nonatomic, readonly) NSDate *lastUnlockAttempt;
@property (nonatomic, readonly) NSUInteger failedAttemptCount;

// Configuration
@property (nonatomic, assign) JARVISVoiceUnlockOptions options;
@property (nonatomic, assign) NSTimeInterval authenticationTimeout;
@property (nonatomic, assign) NSUInteger maxFailedAttempts;
@property (nonatomic, assign) NSTimeInterval lockoutDuration;

// Unlock phrases (dynamically loaded from configuration)
@property (nonatomic, readonly) NSArray<NSString *> *unlockPhrases;

// Core methods
- (BOOL)startMonitoringWithError:(NSError **)error;
- (void)stopMonitoring;
- (BOOL)isUserEnrolled;
- (NSDictionary *)getStatus;
- (void)resetFailedAttempts;

// Configuration methods
- (void)loadConfigurationFromFile:(NSString *)path;
- (void)updateConfiguration:(NSDictionary *)config;
- (NSDictionary *)currentConfiguration;

// Test methods (for development)
- (void)simulateVoiceUnlock:(NSString *)phrase;
- (void)simulateScreenLock;
- (void)simulateScreenUnlock;

@end

NS_ASSUME_NONNULL_END