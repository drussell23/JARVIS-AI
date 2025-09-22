/**
 * JARVISScreenUnlockManager.h
 * JARVIS Voice Unlock System
 *
 * Manages screen lock detection and unlock operations using
 * macOS Security Framework and private APIs.
 */

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <LocalAuthentication/LocalAuthentication.h>

NS_ASSUME_NONNULL_BEGIN

// Screen state
typedef NS_ENUM(NSInteger, JARVISScreenState) {
    JARVISScreenStateUnknown = 0,
    JARVISScreenStateUnlocked,
    JARVISScreenStateLocked,
    JARVISScreenStateScreensaver,
    JARVISScreenStateSleeping
};

// Unlock method
typedef NS_ENUM(NSInteger, JARVISUnlockMethod) {
    JARVISUnlockMethodPassword = 0,
    JARVISUnlockMethodBiometric,
    JARVISUnlockMethodVoice,
    JARVISUnlockMethodEmergency
};

// Unlock result
@interface JARVISUnlockResult : NSObject
@property (nonatomic, readonly) BOOL success;
@property (nonatomic, readonly) JARVISUnlockMethod method;
@property (nonatomic, readonly) NSTimeInterval duration;
@property (nonatomic, readonly, nullable) NSError *error;
@end

// Screen unlock delegate
@protocol JARVISScreenUnlockDelegate <NSObject>
@optional
- (void)screenStateDidChange:(JARVISScreenState)newState;
- (void)screenUnlockDidBegin;
- (void)screenUnlockDidComplete:(JARVISUnlockResult *)result;
- (void)screenUnlockDidFail:(NSError *)error;
@end

// Main screen unlock manager
@interface JARVISScreenUnlockManager : NSObject

@property (nonatomic, weak, nullable) id<JARVISScreenUnlockDelegate> delegate;
@property (nonatomic, readonly) JARVISScreenState currentScreenState;
@property (nonatomic, readonly) BOOL canUnlockScreen;
@property (nonatomic, readonly) BOOL hasSecureToken;

// Screen state detection
- (BOOL)isScreenLocked;
- (BOOL)isScreensaverActive;
- (BOOL)isSystemSleeping;
- (JARVISScreenState)detectScreenState;

// Unlock operations
- (BOOL)unlockScreenWithError:(NSError **)error;
- (void)unlockScreenAsync:(void (^)(JARVISUnlockResult *result))completion;
- (BOOL)unlockScreenWithPassword:(NSString *)password error:(NSError **)error;

// Authentication
- (BOOL)authenticateWithVoice:(NSData *)voiceData error:(NSError **)error;
- (BOOL)verifyUserPassword:(NSString *)password error:(NSError **)error;

// Keychain integration for secure token
- (BOOL)storeSecureTokenForUnlock:(NSString *)password error:(NSError **)error;
- (BOOL)hasStoredSecureToken;
- (void)clearSecureToken;

// System integration
- (BOOL)requestScreenUnlockPermission;
- (BOOL)hasScreenUnlockPermission;
- (void)simulateUserPresence;

// Wake system
- (void)wakeDisplayIfNeeded;
- (void)preventSystemSleep:(BOOL)prevent;

@end

NS_ASSUME_NONNULL_END