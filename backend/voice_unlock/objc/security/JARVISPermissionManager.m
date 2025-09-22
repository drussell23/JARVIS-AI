/**
 * JARVISPermissionManager.m
 * JARVIS Voice Unlock System
 *
 * Implementation of macOS permission management.
 */

#import "JARVISPermissionManager.h"
#import <AVFoundation/AVFoundation.h>
#import <Carbon/Carbon.h>
#import <IOKit/IOKitLib.h>
#import <AppKit/AppKit.h>
#import <os/log.h>

// Permission info implementation
@implementation JARVISPermissionInfo
@end

// Main implementation
@interface JARVISPermissionManager ()

@property (nonatomic, strong) NSMutableDictionary<NSNumber *, JARVISPermissionInfo *> *permissionInfoCache;
@property (nonatomic, strong) NSTimer *monitoringTimer;
@property (nonatomic, strong) os_log_t logger;
@property (nonatomic, assign) BOOL isMonitoring;

@end

@implementation JARVISPermissionManager

- (instancetype)init {
    self = [super init];
    if (self) {
        _permissionInfoCache = [NSMutableDictionary dictionary];
        _logger = os_log_create("com.jarvis.voiceunlock", "permissions");
        
        [self setupPermissionInfo];
        [self checkInitialPermissions];
    }
    return self;
}

- (void)dealloc {
    [self stopMonitoringPermissions];
}

#pragma mark - Setup

- (void)setupPermissionInfo {
    // Microphone
    JARVISPermissionInfo *mic = [[JARVISPermissionInfo alloc] init];
    mic.type = JARVISPermissionTypeMicrophone;
    mic.displayName = @"Microphone";
    mic.explanation = @"Required to capture voice commands for unlocking your Mac";
    mic.isRequired = YES;
    mic.canRequestInApp = YES;
    self.permissionInfoCache[@(JARVISPermissionTypeMicrophone)] = mic;
    
    // Accessibility
    JARVISPermissionInfo *accessibility = [[JARVISPermissionInfo alloc] init];
    accessibility.type = JARVISPermissionTypeAccessibility;
    accessibility.displayName = @"Accessibility";
    accessibility.explanation = @"Required to simulate keyboard input for unlocking the screen";
    accessibility.isRequired = YES;
    accessibility.canRequestInApp = YES;
    self.permissionInfoCache[@(JARVISPermissionTypeAccessibility)] = accessibility;
    
    // Screen Recording
    JARVISPermissionInfo *screenRecording = [[JARVISPermissionInfo alloc] init];
    screenRecording.type = JARVISPermissionTypeScreenRecording;
    screenRecording.displayName = @"Screen Recording";
    screenRecording.explanation = @"Optional - Allows detection of screen lock state";
    screenRecording.isRequired = NO;
    screenRecording.canRequestInApp = NO;
    self.permissionInfoCache[@(JARVISPermissionTypeScreenRecording)] = screenRecording;
    
    // Full Disk Access
    JARVISPermissionInfo *fullDisk = [[JARVISPermissionInfo alloc] init];
    fullDisk.type = JARVISPermissionTypeFullDiskAccess;
    fullDisk.displayName = @"Full Disk Access";
    fullDisk.explanation = @"Optional - Allows access to voice training data";
    fullDisk.isRequired = NO;
    fullDisk.canRequestInApp = NO;
    self.permissionInfoCache[@(JARVISPermissionTypeFullDiskAccess)] = fullDisk;
    
    // Input Monitoring
    JARVISPermissionInfo *inputMonitoring = [[JARVISPermissionInfo alloc] init];
    inputMonitoring.type = JARVISPermissionTypeInputMonitoring;
    inputMonitoring.displayName = @"Input Monitoring";
    inputMonitoring.explanation = @"Optional - Enhances security by detecting unauthorized access attempts";
    inputMonitoring.isRequired = NO;
    inputMonitoring.canRequestInApp = NO;
    self.permissionInfoCache[@(JARVISPermissionTypeInputMonitoring)] = inputMonitoring;
    
    // System Events
    JARVISPermissionInfo *systemEvents = [[JARVISPermissionInfo alloc] init];
    systemEvents.type = JARVISPermissionTypeSystemEvents;
    systemEvents.displayName = @"System Events";
    systemEvents.explanation = @"Optional - Enhanced interaction with the lock screen";
    systemEvents.isRequired = NO;  // Make optional for now
    systemEvents.canRequestInApp = NO;
    self.permissionInfoCache[@(JARVISPermissionTypeSystemEvents)] = systemEvents;
    
    // Keychain
    JARVISPermissionInfo *keychain = [[JARVISPermissionInfo alloc] init];
    keychain.type = JARVISPermissionTypeKeychain;
    keychain.displayName = @"Keychain Access";
    keychain.explanation = @"Required to securely store authentication tokens";
    keychain.isRequired = YES;
    keychain.canRequestInApp = YES;
    self.permissionInfoCache[@(JARVISPermissionTypeKeychain)] = keychain;
}

- (void)checkInitialPermissions {
    for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
        info.status = [self checkPermissionStatus:info.type];
    }
}

#pragma mark - Permission Checking

- (JARVISPermissionStatus)statusForPermission:(JARVISPermissionType)permission {
    JARVISPermissionInfo *info = self.permissionInfoCache[@(permission)];
    if (info) {
        info.status = [self checkPermissionStatus:permission];
        return info.status;
    }
    return JARVISPermissionStatusNotDetermined;
}

- (JARVISPermissionStatus)checkPermissionStatus:(JARVISPermissionType)permission {
    switch (permission) {
        case JARVISPermissionTypeMicrophone:
            return [self checkMicrophonePermission];
            
        case JARVISPermissionTypeAccessibility:
            return [self checkAccessibilityPermission];
            
        case JARVISPermissionTypeScreenRecording:
            return [self checkScreenRecordingPermission];
            
        case JARVISPermissionTypeFullDiskAccess:
            return [self checkFullDiskAccessPermission];
            
        case JARVISPermissionTypeInputMonitoring:
            return [self checkInputMonitoringPermission];
            
        case JARVISPermissionTypeSystemEvents:
            return [self checkSystemEventsPermission];
            
        case JARVISPermissionTypeKeychain:
            return [self checkKeychainPermission];
    }
}

- (JARVISPermissionStatus)checkMicrophonePermission {
    if (@available(macOS 10.14, *)) {
        AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        switch (status) {
            case AVAuthorizationStatusAuthorized:
                return JARVISPermissionStatusAuthorized;
            case AVAuthorizationStatusDenied:
                return JARVISPermissionStatusDenied;
            case AVAuthorizationStatusRestricted:
                return JARVISPermissionStatusRestricted;
            case AVAuthorizationStatusNotDetermined:
                return JARVISPermissionStatusNotDetermined;
        }
    }
    return JARVISPermissionStatusAuthorized; // Pre-10.14 doesn't require permission
}

- (JARVISPermissionStatus)checkAccessibilityPermission {
    if (AXIsProcessTrusted()) {
        return JARVISPermissionStatusAuthorized;
    }
    return JARVISPermissionStatusDenied;
}

- (JARVISPermissionStatus)checkScreenRecordingPermission {
    if (@available(macOS 10.15, *)) {
        // Check if we can access screen content
        // This is a simple check - actual screen recording would need ScreenCaptureKit
        CFArrayRef windowList = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
        if (windowList) {
            CFIndex count = CFArrayGetCount(windowList);
            CFRelease(windowList);
            
            // If we can get window info with real content, we have permission
            // Without permission, the count would be very limited
            if (count > 0) {
                return JARVISPermissionStatusAuthorized;
            }
        }
        return JARVISPermissionStatusDenied;
    }
    return JARVISPermissionStatusAuthorized;
}

- (JARVISPermissionStatus)checkFullDiskAccessPermission {
    // Check if we can access a protected location
    NSString *testPath = [@"~/Library/Safari/Bookmarks.plist" stringByExpandingTildeInPath];
    if ([[NSFileManager defaultManager] isReadableFileAtPath:testPath]) {
        return JARVISPermissionStatusAuthorized;
    }
    
    // Try to read TCC database
    NSString *tccPath = @"/Library/Application Support/com.apple.TCC/TCC.db";
    if ([[NSFileManager defaultManager] isReadableFileAtPath:tccPath]) {
        return JARVISPermissionStatusAuthorized;
    }
    
    return JARVISPermissionStatusDenied;
}

- (JARVISPermissionStatus)checkInputMonitoringPermission {
    // Check if we can monitor global events
    if (@available(macOS 10.15, *)) {
        // Try to create an event tap
        CFMachPortRef eventTap = CGEventTapCreate(kCGSessionEventTap,
                                                  kCGHeadInsertEventTap,
                                                  kCGEventTapOptionListenOnly,
                                                  kCGEventMaskForAllEvents,
                                                  NULL, NULL);
        if (eventTap) {
            CFRelease(eventTap);
            return JARVISPermissionStatusAuthorized;
        }
        return JARVISPermissionStatusDenied;
    }
    return JARVISPermissionStatusAuthorized;
}

- (JARVISPermissionStatus)checkSystemEventsPermission {
    // Check if we can send Apple Events to System Events
    NSAppleEventDescriptor *targetDescriptor = [NSAppleEventDescriptor descriptorWithBundleIdentifier:@"com.apple.systemevents"];
    
    NSAppleEventDescriptor *appleEvent = [NSAppleEventDescriptor appleEventWithEventClass:'ascr'
                                                                                  eventID:'gdte'
                                                                         targetDescriptor:targetDescriptor
                                                                                 returnID:kAutoGenerateReturnID
                                                                            transactionID:kAnyTransactionID];
    
    AEDesc reply = {typeNull, NULL};
    OSStatus status = AESendMessage([appleEvent aeDesc], &reply, kAENoReply | kAECanInteract, kAEDefaultTimeout);
    
    if (reply.descriptorType != typeNull) {
        AEDisposeDesc(&reply);
    }
    
    if (status == noErr) {
        return JARVISPermissionStatusAuthorized;
    } else if (status == -1743) { // errAEEventNotPermitted
        return JARVISPermissionStatusDenied;
    }
    
    return JARVISPermissionStatusNotDetermined;
}

- (JARVISPermissionStatus)checkKeychainPermission {
    // Keychain access is generally available on macOS
    // Try a simple keychain operation to verify access
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: @"com.jarvis.test",
        (__bridge id)kSecReturnData: @NO,
        (__bridge id)kSecMatchLimit: (__bridge id)kSecMatchLimitOne
    };
    
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, NULL);
    
    // errSecItemNotFound is fine - it means we can access keychain but item doesn't exist
    // errSecInteractionNotAllowed or other errors mean we don't have access
    if (status == errSecSuccess || status == errSecItemNotFound) {
        return JARVISPermissionStatusAuthorized;
    }
    
    return JARVISPermissionStatusDenied;
}

#pragma mark - Permission Requests

- (BOOL)checkAndRequestPermissions:(NSError **)error {
    NSMutableArray<JARVISPermissionInfo *> *missingRequired = [NSMutableArray array];
    
    // Check all permissions
    for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
        info.status = [self checkPermissionStatus:info.type];
        
        if (info.isRequired && info.status != JARVISPermissionStatusAuthorized) {
            [missingRequired addObject:info];
        }
    }
    
    // If all required permissions are granted
    if (missingRequired.count == 0) {
        if ([self.delegate respondsToSelector:@selector(allRequiredPermissionsGranted)]) {
            [self.delegate allRequiredPermissionsGranted];
        }
        return YES;
    }
    
    // Request permissions that can be requested in-app
    for (JARVISPermissionInfo *info in missingRequired) {
        if (info.canRequestInApp && info.status == JARVISPermissionStatusNotDetermined) {
            switch (info.type) {
                case JARVISPermissionTypeMicrophone:
                    [self requestMicrophonePermission:nil];
                    break;
                    
                case JARVISPermissionTypeAccessibility:
                    [self requestAccessibilityPermission];
                    break;
                    
                default:
                    break;
            }
        }
    }
    
    // Notify about missing permissions
    if ([self.delegate respondsToSelector:@selector(missingRequiredPermissions:)]) {
        [self.delegate missingRequiredPermissions:missingRequired];
    }
    
    if (error) {
        NSString *missingList = [[missingRequired valueForKey:@"displayName"] componentsJoinedByString:@", "];
        *error = [NSError errorWithDomain:@"JARVISPermissions"
                                     code:1001
                                 userInfo:@{
            NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Missing required permissions: %@", missingList]
        }];
    }
    
    return NO;
}

- (void)checkAllPermissionsWithCompletion:(void (^)(BOOL, NSArray<JARVISPermissionInfo *> *))completion {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSMutableArray<JARVISPermissionInfo *> *missing = [NSMutableArray array];
        BOOL allGranted = YES;
        
        for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
            info.status = [self checkPermissionStatus:info.type];
            
            if (info.isRequired && info.status != JARVISPermissionStatusAuthorized) {
                [missing addObject:info];
                allGranted = NO;
            }
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) {
                completion(allGranted, missing);
            }
        });
    });
}

#pragma mark - Individual Permission Requests

- (void)requestMicrophonePermission:(void (^)(BOOL))completion {
    if (@available(macOS 10.14, *)) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio 
                                 completionHandler:^(BOOL granted) {
            JARVISPermissionInfo *info = self.permissionInfoCache[@(JARVISPermissionTypeMicrophone)];
            info.status = granted ? JARVISPermissionStatusAuthorized : JARVISPermissionStatusDenied;
            
            dispatch_async(dispatch_get_main_queue(), ^{
                if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
                    [self.delegate permissionStatusChanged:JARVISPermissionTypeMicrophone 
                                                    status:info.status];
                }
                
                if (completion) {
                    completion(granted);
                }
            });
        }];
    } else {
        if (completion) {
            completion(YES);
        }
    }
}

- (BOOL)requestAccessibilityPermission {
    NSDictionary *options = @{(__bridge id)kAXTrustedCheckOptionPrompt: @YES};
    BOOL trusted = AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
    
    JARVISPermissionInfo *info = self.permissionInfoCache[@(JARVISPermissionTypeAccessibility)];
    info.status = trusted ? JARVISPermissionStatusAuthorized : JARVISPermissionStatusDenied;
    
    if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
        [self.delegate permissionStatusChanged:JARVISPermissionTypeAccessibility 
                                        status:info.status];
    }
    
    return trusted;
}

- (BOOL)requestScreenRecordingPermission {
    // Screen recording permission cannot be requested programmatically
    // Open system preferences instead
    [self openSystemPreferencesForPermission:JARVISPermissionTypeScreenRecording];
    return NO;
}

- (BOOL)requestFullDiskAccessPermission {
    // Full disk access cannot be requested programmatically
    [self openSystemPreferencesForPermission:JARVISPermissionTypeFullDiskAccess];
    return NO;
}

- (BOOL)requestInputMonitoringPermission {
    // Input monitoring permission cannot be requested programmatically
    [self openSystemPreferencesForPermission:JARVISPermissionTypeInputMonitoring];
    return NO;
}

#pragma mark - System Preferences

- (void)openSystemPreferencesForPermission:(JARVISPermissionType)permission {
    NSString *prefPane = nil;
    
    switch (permission) {
        case JARVISPermissionTypeMicrophone:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone";
            break;
            
        case JARVISPermissionTypeAccessibility:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
            break;
            
        case JARVISPermissionTypeScreenRecording:
            if (@available(macOS 10.15, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";
            }
            break;
            
        case JARVISPermissionTypeFullDiskAccess:
            if (@available(macOS 10.14, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles";
            }
            break;
            
        case JARVISPermissionTypeInputMonitoring:
            if (@available(macOS 10.15, *)) {
                prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";
            }
            break;
            
        case JARVISPermissionTypeSystemEvents:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy_Automation";
            break;
            
        default:
            prefPane = @"x-apple.systempreferences:com.apple.preference.security?Privacy";
            break;
    }
    
    if (prefPane) {
        [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:prefPane]];
    }
}

- (void)openPrivacyPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.security?Privacy";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

- (void)openAccessibilityPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.universalaccess";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

- (void)openSecurityPreferences {
    NSString *urlString = @"x-apple.systempreferences:com.apple.preference.security";
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:urlString]];
}

#pragma mark - Explanations

- (NSString *)explanationForPermission:(JARVISPermissionType)permission {
    JARVISPermissionInfo *info = self.permissionInfoCache[@(permission)];
    return info ? info.explanation : @"This permission is required for JARVIS Voice Unlock to function properly.";
}

- (NSString *)instructionsForPermission:(JARVISPermissionType)permission {
    switch (permission) {
        case JARVISPermissionTypeMicrophone:
            return @"Click 'OK' when prompted to grant microphone access.";
            
        case JARVISPermissionTypeAccessibility:
            return @"1. Click 'Open System Preferences' when prompted\n"
                   @"2. Click the lock icon to make changes\n"
                   @"3. Check the box next to JARVIS Voice Unlock\n"
                   @"4. Close System Preferences";
            
        case JARVISPermissionTypeScreenRecording:
            return @"1. Open System Preferences > Security & Privacy > Privacy\n"
                   @"2. Select 'Screen Recording' from the list\n"
                   @"3. Click the lock icon to make changes\n"
                   @"4. Check the box next to JARVIS Voice Unlock\n"
                   @"5. Restart JARVIS Voice Unlock";
            
        case JARVISPermissionTypeFullDiskAccess:
            return @"1. Open System Preferences > Security & Privacy > Privacy\n"
                   @"2. Select 'Full Disk Access' from the list\n"
                   @"3. Click the lock icon to make changes\n"
                   @"4. Click '+' and add JARVIS Voice Unlock\n"
                   @"5. Restart JARVIS Voice Unlock";
            
        default:
            return @"Please grant the requested permission in System Preferences.";
    }
}

#pragma mark - Monitoring

- (void)startMonitoringPermissions {
    if (self.isMonitoring) {
        return;
    }
    
    self.isMonitoring = YES;
    
    // Monitor permission changes every 2 seconds
    self.monitoringTimer = [NSTimer scheduledTimerWithTimeInterval:2.0
                                                            target:self
                                                          selector:@selector(checkPermissionChanges)
                                                          userInfo:nil
                                                           repeats:YES];
}

- (void)stopMonitoringPermissions {
    self.isMonitoring = NO;
    [self.monitoringTimer invalidate];
    self.monitoringTimer = nil;
}

- (void)checkPermissionChanges {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        BOOL changesDetected = NO;
        
        for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
            JARVISPermissionStatus oldStatus = info.status;
            JARVISPermissionStatus newStatus = [self checkPermissionStatus:info.type];
            
            if (oldStatus != newStatus) {
                info.status = newStatus;
                changesDetected = YES;
                
                dispatch_async(dispatch_get_main_queue(), ^{
                    if ([self.delegate respondsToSelector:@selector(permissionStatusChanged:status:)]) {
                        [self.delegate permissionStatusChanged:info.type status:newStatus];
                    }
                });
                
                os_log_info(self.logger, "Permission %@ changed from %ld to %ld",
                           info.displayName, (long)oldStatus, (long)newStatus);
            }
        }
        
        if (changesDetected) {
            [self checkAndRequestPermissions:nil];
        }
    });
}

#pragma mark - Properties

- (BOOL)hasAllRequiredPermissions {
    for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
        if (info.isRequired && info.status != JARVISPermissionStatusAuthorized) {
            return NO;
        }
    }
    return YES;
}

- (NSArray<JARVISPermissionInfo *> *)allPermissions {
    return [self.permissionInfoCache.allValues sortedArrayUsingComparator:^NSComparisonResult(JARVISPermissionInfo *a, JARVISPermissionInfo *b) {
        if (a.isRequired != b.isRequired) {
            return a.isRequired ? NSOrderedAscending : NSOrderedDescending;
        }
        return [a.displayName compare:b.displayName];
    }];
}

- (NSArray<JARVISPermissionInfo *> *)missingPermissions {
    NSMutableArray *missing = [NSMutableArray array];
    
    for (JARVISPermissionInfo *info in self.permissionInfoCache.allValues) {
        if (info.status != JARVISPermissionStatusAuthorized) {
            [missing addObject:info];
        }
    }
    
    return [missing sortedArrayUsingComparator:^NSComparisonResult(JARVISPermissionInfo *a, JARVISPermissionInfo *b) {
        if (a.isRequired != b.isRequired) {
            return a.isRequired ? NSOrderedAscending : NSOrderedDescending;
        }
        return [a.displayName compare:b.displayName];
    }];
}

#pragma mark - Utility

- (BOOL)isRunningInSandbox {
    // Check if app is sandboxed
    NSString *homeDir = NSHomeDirectory();
    return [homeDir containsString:@"/Library/Containers/"];
}

- (BOOL)hasEntitlement:(NSString *)entitlement {
    SecTaskRef task = SecTaskCreateFromSelf(kCFAllocatorDefault);
    if (task == NULL) {
        return NO;
    }
    
    CFTypeRef value = SecTaskCopyValueForEntitlement(task, (__bridge CFStringRef)entitlement, NULL);
    CFRelease(task);
    
    BOOL hasEntitlement = NO;
    if (value != NULL) {
        if (CFGetTypeID(value) == CFBooleanGetTypeID()) {
            hasEntitlement = CFBooleanGetValue(value);
        }
        CFRelease(value);
    }
    
    return hasEntitlement;
}

@end