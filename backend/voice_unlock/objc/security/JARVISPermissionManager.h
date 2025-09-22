/**
 * JARVISPermissionManager.h
 * JARVIS Voice Unlock System
 *
 * Manages macOS permissions required for voice unlock functionality
 * including microphone, accessibility, and security permissions.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

// Permission types
typedef NS_ENUM(NSInteger, JARVISPermissionType) {
    JARVISPermissionTypeMicrophone = 0,
    JARVISPermissionTypeAccessibility,
    JARVISPermissionTypeScreenRecording,
    JARVISPermissionTypeFullDiskAccess,
    JARVISPermissionTypeInputMonitoring,
    JARVISPermissionTypeSystemEvents,
    JARVISPermissionTypeKeychain
};

// Permission status
typedef NS_ENUM(NSInteger, JARVISPermissionStatus) {
    JARVISPermissionStatusNotDetermined = 0,
    JARVISPermissionStatusDenied,
    JARVISPermissionStatusAuthorized,
    JARVISPermissionStatusRestricted
};

// Permission info
@interface JARVISPermissionInfo : NSObject
@property (nonatomic, assign) JARVISPermissionType type;
@property (nonatomic, assign) JARVISPermissionStatus status;
@property (nonatomic, strong) NSString *displayName;
@property (nonatomic, strong) NSString *explanation;
@property (nonatomic, assign) BOOL isRequired;
@property (nonatomic, assign) BOOL canRequestInApp;
@end

// Permission delegate
@protocol JARVISPermissionManagerDelegate <NSObject>
@optional
- (void)permissionStatusChanged:(JARVISPermissionType)permission status:(JARVISPermissionStatus)status;
- (void)allRequiredPermissionsGranted;
- (void)missingRequiredPermissions:(NSArray<JARVISPermissionInfo *> *)permissions;
@end

// Main permission manager interface
@interface JARVISPermissionManager : NSObject

@property (nonatomic, weak, nullable) id<JARVISPermissionManagerDelegate> delegate;
@property (nonatomic, readonly) BOOL hasAllRequiredPermissions;
@property (nonatomic, readonly) NSArray<JARVISPermissionInfo *> *allPermissions;
@property (nonatomic, readonly) NSArray<JARVISPermissionInfo *> *missingPermissions;

// Permission checking
- (JARVISPermissionStatus)statusForPermission:(JARVISPermissionType)permission;
- (BOOL)checkAndRequestPermissions:(NSError **)error;
- (void)checkAllPermissionsWithCompletion:(void (^)(BOOL allGranted, NSArray<JARVISPermissionInfo *> *missing))completion;

// Individual permission requests
- (void)requestMicrophonePermission:(void (^)(BOOL granted))completion;
- (BOOL)requestAccessibilityPermission;
- (BOOL)requestScreenRecordingPermission;
- (BOOL)requestFullDiskAccessPermission;
- (BOOL)requestInputMonitoringPermission;

// System preferences
- (void)openSystemPreferencesForPermission:(JARVISPermissionType)permission;
- (void)openPrivacyPreferences;
- (void)openAccessibilityPreferences;
- (void)openSecurityPreferences;

// Permission explanations
- (NSString *)explanationForPermission:(JARVISPermissionType)permission;
- (NSString *)instructionsForPermission:(JARVISPermissionType)permission;

// Monitoring
- (void)startMonitoringPermissions;
- (void)stopMonitoringPermissions;

// Utility
- (BOOL)isRunningInSandbox;
- (BOOL)hasEntitlement:(NSString *)entitlement;

@end

NS_ASSUME_NONNULL_END