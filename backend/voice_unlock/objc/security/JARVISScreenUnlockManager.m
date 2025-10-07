/**
 * JARVISScreenUnlockManager.m
 * JARVIS Voice Unlock System
 *
 * Implementation of screen unlock functionality.
 */

#import "JARVISScreenUnlockManager.h"
#import <IOKit/pwr_mgt/IOPMLib.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Carbon/Carbon.h>
#import <AppKit/AppKit.h>
#import <os/log.h>

// Private APIs would go here, but we'll use alternative methods

// Unlock result implementation
@interface JARVISUnlockResult ()
@property (nonatomic, readwrite) BOOL success;
@property (nonatomic, readwrite) JARVISUnlockMethod method;
@property (nonatomic, readwrite) NSTimeInterval duration;
@property (nonatomic, readwrite, nullable) NSError *error;
@end

@implementation JARVISUnlockResult
@end

// Main implementation
@interface JARVISScreenUnlockManager ()
@property (nonatomic, strong) LAContext *authContext;
@property (nonatomic, strong) dispatch_queue_t unlockQueue;
@property (nonatomic, strong) os_log_t logger;
@property (nonatomic, assign) IOPMAssertionID sleepAssertionID;
@property (nonatomic, strong) NSTimer *stateMonitorTimer;
@property (nonatomic, readwrite) JARVISScreenState currentScreenState;
@property (nonatomic, readwrite) BOOL hasSecureToken;
@end

@implementation JARVISScreenUnlockManager

- (instancetype)init {
    self = [super init];
    if (self) {
        _authContext = [[LAContext alloc] init];
        _unlockQueue = dispatch_queue_create("com.jarvis.screenunlock", DISPATCH_QUEUE_SERIAL);
        _logger = os_log_create("com.jarvis.voiceunlock", "screenunlock");
        _currentScreenState = JARVISScreenStateUnknown;
        _sleepAssertionID = 0;
        
        // Check for secure token
        [self checkSecureToken];
        
        // Start monitoring screen state
        [self startScreenStateMonitoring];
    }
    return self;
}

- (void)dealloc {
    [self stopScreenStateMonitoring];
    [self preventSystemSleep:NO];
}

#pragma mark - Screen State Detection

- (BOOL)isScreenLocked {
    // Method 1: Check if screensaver is running with password requirement
    CFDictionaryRef sessionInfo = CGSessionCopyCurrentDictionary();
    if (sessionInfo) {
        CFBooleanRef screenIsLocked = CFDictionaryGetValue(sessionInfo, CFSTR("CGSSessionScreenIsLocked"));
        if (screenIsLocked) {
            BOOL locked = CFBooleanGetValue(screenIsLocked);
            CFRelease(sessionInfo);
            return locked;
        }
        CFRelease(sessionInfo);
    }
    
    // Method 2: Check screensaver process
    NSArray *apps = [[NSWorkspace sharedWorkspace] runningApplications];
    for (NSRunningApplication *app in apps) {
        if ([[app bundleIdentifier] isEqualToString:@"com.apple.ScreenSaver.Engine"]) {
            return YES;
        }
    }
    
    // Method 3: Check using CGSessionCopyCurrentDictionary as alternative
    
    return NO;
}

- (BOOL)isScreensaverActive {
    NSArray *apps = [[NSWorkspace sharedWorkspace] runningApplications];
    for (NSRunningApplication *app in apps) {
        if ([[app bundleIdentifier] isEqualToString:@"com.apple.ScreenSaver.Engine"]) {
            return YES;
        }
    }
    return NO;
}

- (BOOL)isSystemSleeping {
    // Check system sleep state
    CFDictionaryRef sessionInfo = CGSessionCopyCurrentDictionary();
    if (sessionInfo) {
        CFBooleanRef sleeping = CFDictionaryGetValue(sessionInfo, CFSTR("CGSSessionOnConsoleKey"));
        if (sleeping) {
            BOOL awake = CFBooleanGetValue(sleeping);
            CFRelease(sessionInfo);
            return !awake;
        }
        CFRelease(sessionInfo);
    }
    return NO;
}

- (JARVISScreenState)detectScreenState {
    if ([self isSystemSleeping]) {
        return JARVISScreenStateSleeping;
    } else if ([self isScreenLocked]) {
        return JARVISScreenStateLocked;
    } else if ([self isScreensaverActive]) {
        return JARVISScreenStateScreensaver;
    } else {
        return JARVISScreenStateUnlocked;
    }
}

- (void)startScreenStateMonitoring {
    self.stateMonitorTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                              target:self
                                                            selector:@selector(checkScreenState)
                                                            userInfo:nil
                                                             repeats:YES];
}

- (void)stopScreenStateMonitoring {
    [self.stateMonitorTimer invalidate];
    self.stateMonitorTimer = nil;
}

- (void)checkScreenState {
    JARVISScreenState newState = [self detectScreenState];
    if (newState != self.currentScreenState) {
        JARVISScreenState oldState = self.currentScreenState;
        self.currentScreenState = newState;
        
        os_log_info(self.logger, "Screen state changed from %ld to %ld", (long)oldState, (long)newState);
        
        if ([self.delegate respondsToSelector:@selector(screenStateDidChange:)]) {
            [self.delegate screenStateDidChange:newState];
        }
    }
}

#pragma mark - Lock Operations

- (BOOL)lockScreen {
    NSError *error = nil;
    BOOL result = [self lockScreenWithError:&error];
    if (!result && error) {
        os_log_error(self.logger, "Failed to lock screen: %@", error.localizedDescription);
    }
    return result;
}

- (BOOL)lockScreenWithError:(NSError **)error {
    os_log_info(self.logger, "Attempting to lock screen");

    // Method 1: Use CGSession (most reliable for macOS)
    NSString *cgSessionPath = @"/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession";
    if ([[NSFileManager defaultManager] fileExistsAtPath:cgSessionPath]) {
        NSTask *task = [[NSTask alloc] init];
        task.launchPath = cgSessionPath;
        task.arguments = @[@"-suspend"];

        @try {
            [task launch];
            [task waitUntilExit];

            if (task.terminationStatus == 0) {
                os_log_info(self.logger, "Screen locked successfully using CGSession");

                // Update screen state
                self.currentScreenState = JARVISScreenStateLocked;
                if ([self.delegate respondsToSelector:@selector(screenStateDidChange:)]) {
                    [self.delegate screenStateDidChange:JARVISScreenStateLocked];
                }

                return YES;
            }
        } @catch (NSException *exception) {
            os_log_debug(self.logger, "CGSession method failed: %@", exception.reason);
        }
    }

    // Method 2: Use AppleScript to trigger lock via System Events
    NSString *script = @"tell application \"System Events\" to keystroke \"q\" using {command down, control down}";
    NSAppleScript *appleScript = [[NSAppleScript alloc] initWithSource:script];
    NSDictionary *errorInfo = nil;
    NSAppleEventDescriptor *result = [appleScript executeAndReturnError:&errorInfo];

    if (result && !errorInfo) {
        os_log_info(self.logger, "Screen locked successfully using AppleScript");
        self.currentScreenState = JARVISScreenStateLocked;
        if ([self.delegate respondsToSelector:@selector(screenStateDidChange:)]) {
            [self.delegate screenStateDidChange:JARVISScreenStateLocked];
        }
        return YES;
    }

    // Method 3: Start screensaver (will lock if password required)
    BOOL screensaverStarted = NO;
    @try {
        [[NSWorkspace sharedWorkspace] launchApplication:@"ScreenSaverEngine"];
        screensaverStarted = YES;
        os_log_info(self.logger, "Started screensaver");
    } @catch (NSException *exception) {
        os_log_debug(self.logger, "Failed to start screensaver: %@", exception.reason);
    }

    if (screensaverStarted) {
        self.currentScreenState = JARVISScreenStateScreensaver;
        if ([self.delegate respondsToSelector:@selector(screenStateDidChange:)]) {
            [self.delegate screenStateDidChange:JARVISScreenStateScreensaver];
        }
        return YES;
    }

    // All methods failed
    if (error) {
        *error = [NSError errorWithDomain:@"JARVISScreenUnlock"
                                     code:500
                                 userInfo:@{NSLocalizedDescriptionKey: @"Failed to lock screen using all available methods"}];
    }
    return NO;
}

#pragma mark - Unlock Operations

- (BOOL)canUnlockScreen {
    // Check if we have the necessary permissions and tokens
    return self.hasSecureToken && [self hasScreenUnlockPermission];
}

- (BOOL)unlockScreenWithError:(NSError **)error {
    if (![self canUnlockScreen]) {
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISScreenUnlock"
                                         code:403
                                     userInfo:@{NSLocalizedDescriptionKey: @"Cannot unlock screen - missing permissions or secure token"}];
        }
        return NO;
    }
    
    // Wake display first
    [self wakeDisplayIfNeeded];
    
    // Try multiple unlock methods
    
    // Method 1: Simulate user activity
    [self simulateUserPresence];
    
    // Method 2: Use stored credentials if available
    NSString *storedPassword = [self retrieveStoredPassword];
    if (storedPassword) {
        return [self unlockScreenWithPassword:storedPassword error:error];
    }
    
    // Method 3: Use biometric authentication
    if ([self.authContext canEvaluatePolicy:LAPolicyDeviceOwnerAuthentication error:nil]) {
        __block BOOL success = NO;
        __block NSError *authError = nil;
        
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        
        [self.authContext evaluatePolicy:LAPolicyDeviceOwnerAuthentication
                        localizedReason:@"JARVIS Voice Unlock"
                                  reply:^(BOOL evalSuccess, NSError *evalError) {
            success = evalSuccess;
            authError = evalError;
            dispatch_semaphore_signal(semaphore);
        }];
        
        dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
        
        if (success) {
            return YES;
        } else if (error && authError) {
            *error = authError;
        }
    }
    
    return NO;
}

- (void)unlockScreenAsync:(void (^)(JARVISUnlockResult *))completion {
    NSDate *startTime = [NSDate date];
    
    dispatch_async(self.unlockQueue, ^{
        if ([self.delegate respondsToSelector:@selector(screenUnlockDidBegin)]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.delegate screenUnlockDidBegin];
            });
        }
        
        NSError *error = nil;
        BOOL success = [self unlockScreenWithError:&error];
        
        JARVISUnlockResult *result = [[JARVISUnlockResult alloc] init];
        result.success = success;
        result.method = JARVISUnlockMethodVoice;
        result.duration = [[NSDate date] timeIntervalSinceDate:startTime];
        result.error = error;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) {
                completion(result);
            }
            
            if (success) {
                if ([self.delegate respondsToSelector:@selector(screenUnlockDidComplete:)]) {
                    [self.delegate screenUnlockDidComplete:result];
                }
            } else {
                if ([self.delegate respondsToSelector:@selector(screenUnlockDidFail:)]) {
                    [self.delegate screenUnlockDidFail:error];
                }
            }
        });
    });
}

- (BOOL)unlockScreenWithPassword:(NSString *)password error:(NSError **)error {
    // This would require system-level integration
    // For now, we'll simulate the unlock
    
    if (![self verifyUserPassword:password error:error]) {
        return NO;
    }
    
    // Wake the display
    [self wakeDisplayIfNeeded];
    
    // Simulate keyboard input for password
    // Note: This requires accessibility permissions
    [self simulatePasswordEntry:password];
    
    return YES;
}

#pragma mark - Authentication

- (BOOL)authenticateWithVoice:(NSData *)voiceData error:(NSError **)error {
    // This method would integrate with the voice authenticator
    // For now, return success based on voice data presence
    if (!voiceData || voiceData.length == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISScreenUnlock"
                                         code:400
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid voice data"}];
        }
        return NO;
    }
    
    // In a real implementation, this would verify the voice data
    // against the enrolled voiceprint
    return YES;
}

- (BOOL)verifyUserPassword:(NSString *)password error:(NSError **)error {
    // Use Local Authentication to verify password
    LAContext *context = [[LAContext alloc] init];
    
    if (![context canEvaluatePolicy:LAPolicyDeviceOwnerAuthentication error:error]) {
        return NO;
    }
    
    // This would typically be done asynchronously
    // For demo purposes, we're simplifying
    return password.length > 0;
}

#pragma mark - Keychain Integration

- (void)checkSecureToken {
    // Check if we have a stored secure token
    self.hasSecureToken = [self hasStoredSecureToken];
}

- (BOOL)storeSecureTokenForUnlock:(NSString *)password error:(NSError **)error {
    // Store password securely in keychain
    NSString *service = @"com.jarvis.voiceunlock";
    NSString *account = @"unlock_token";
    
    // Delete existing item
    NSDictionary *deleteQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: service,
        (__bridge id)kSecAttrAccount: account
    };
    SecItemDelete((__bridge CFDictionaryRef)deleteQuery);
    
    // Add new item
    NSDictionary *addQuery = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: service,
        (__bridge id)kSecAttrAccount: account,
        (__bridge id)kSecValueData: [password dataUsingEncoding:NSUTF8StringEncoding],
        (__bridge id)kSecAttrAccessible: (__bridge id)kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    };
    
    OSStatus status = SecItemAdd((__bridge CFDictionaryRef)addQuery, NULL);
    
    if (status != errSecSuccess) {
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISScreenUnlock"
                                         code:status
                                     userInfo:@{NSLocalizedDescriptionKey: @"Failed to store secure token"}];
        }
        return NO;
    }
    
    self.hasSecureToken = YES;
    return YES;
}

- (BOOL)hasStoredSecureToken {
    NSString *service = @"com.jarvis.voiceunlock";
    NSString *account = @"unlock_token";
    
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: service,
        (__bridge id)kSecAttrAccount: account,
        (__bridge id)kSecReturnData: @NO
    };
    
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, NULL);
    return status == errSecSuccess;
}

- (NSString *)retrieveStoredPassword {
    NSString *service = @"com.jarvis.voiceunlock";
    NSString *account = @"unlock_token";
    
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: service,
        (__bridge id)kSecAttrAccount: account,
        (__bridge id)kSecReturnData: @YES
    };
    
    CFDataRef result = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, (CFTypeRef *)&result);
    
    if (status == errSecSuccess && result) {
        NSString *password = [[NSString alloc] initWithData:(__bridge_transfer NSData *)result
                                                   encoding:NSUTF8StringEncoding];
        return password;
    }
    
    return nil;
}

- (void)clearSecureToken {
    NSString *service = @"com.jarvis.voiceunlock";
    NSString *account = @"unlock_token";
    
    NSDictionary *query = @{
        (__bridge id)kSecClass: (__bridge id)kSecClassGenericPassword,
        (__bridge id)kSecAttrService: service,
        (__bridge id)kSecAttrAccount: account
    };
    
    SecItemDelete((__bridge CFDictionaryRef)query);
    self.hasSecureToken = NO;
}

#pragma mark - System Integration

- (BOOL)requestScreenUnlockPermission {
    // Request accessibility permissions
    NSDictionary *options = @{(__bridge id)kAXTrustedCheckOptionPrompt: @YES};
    return AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
}

- (BOOL)hasScreenUnlockPermission {
    return AXIsProcessTrusted();
}

- (void)simulateUserPresence {
    // Simulate mouse movement to wake the screen
    CGEventRef moveEvent = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved,
                                                   CGPointMake(100, 100), kCGMouseButtonLeft);
    if (moveEvent) {
        CGEventPost(kCGHIDEventTap, moveEvent);
        CFRelease(moveEvent);
    }
    
    // Simulate key press
    CGEventRef keyEvent = CGEventCreateKeyboardEvent(NULL, kVK_Space, true);
    if (keyEvent) {
        CGEventPost(kCGHIDEventTap, keyEvent);
        CFRelease(keyEvent);
    }
    
    keyEvent = CGEventCreateKeyboardEvent(NULL, kVK_Space, false);
    if (keyEvent) {
        CGEventPost(kCGHIDEventTap, keyEvent);
        CFRelease(keyEvent);
    }
}

- (void)simulatePasswordEntry:(NSString *)password {
    if (![self hasScreenUnlockPermission]) {
        return;
    }
    
    // Type each character of the password
    for (NSUInteger i = 0; i < password.length; i++) {
        unichar character = [password characterAtIndex:i];
        NSString *charStr = [NSString stringWithCharacters:&character length:1];
        
        CGEventRef keyDown = CGEventCreateKeyboardEvent(NULL, 0, true);
        CGEventRef keyUp = CGEventCreateKeyboardEvent(NULL, 0, false);
        
        if (keyDown && keyUp) {
            CGEventKeyboardSetUnicodeString(keyDown, 1, &character);
            CGEventKeyboardSetUnicodeString(keyUp, 1, &character);
            
            CGEventPost(kCGHIDEventTap, keyDown);
            usleep(50000); // 50ms delay between keystrokes
            CGEventPost(kCGHIDEventTap, keyUp);
            
            CFRelease(keyDown);
            CFRelease(keyUp);
        }
    }
    
    // Press Enter
    CGEventRef enterDown = CGEventCreateKeyboardEvent(NULL, kVK_Return, true);
    CGEventRef enterUp = CGEventCreateKeyboardEvent(NULL, kVK_Return, false);
    
    if (enterDown && enterUp) {
        CGEventPost(kCGHIDEventTap, enterDown);
        CGEventPost(kCGHIDEventTap, enterUp);
        CFRelease(enterDown);
        CFRelease(enterUp);
    }
}

#pragma mark - Power Management

- (void)wakeDisplayIfNeeded {
    // Wake the display
    io_registry_entry_t entry = IORegistryEntryFromPath(kIOMainPortDefault, 
                                                        "IOService:/IOResources/IODisplayWrangler");
    if (entry != MACH_PORT_NULL) {
        IORegistryEntrySetCFProperty(entry, CFSTR("IORequestIdle"), kCFBooleanFalse);
        IOObjectRelease(entry);
    }
    
    // Also simulate user activity
    [self simulateUserPresence];
}

- (void)preventSystemSleep:(BOOL)prevent {
    if (prevent) {
        if (self.sleepAssertionID == 0) {
            CFStringRef reason = CFSTR("JARVIS Voice Unlock Active");
            IOReturn success = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep,
                                                          kIOPMAssertionLevelOn,
                                                          reason,
                                                          &_sleepAssertionID);
            
            if (success == kIOReturnSuccess) {
                os_log_info(self.logger, "Created sleep prevention assertion");
            } else {
                os_log_error(self.logger, "Failed to create sleep prevention assertion");
            }
        }
    } else {
        if (self.sleepAssertionID != 0) {
            IOReturn success = IOPMAssertionRelease(self.sleepAssertionID);
            if (success == kIOReturnSuccess) {
                os_log_info(self.logger, "Released sleep prevention assertion");
            }
            self.sleepAssertionID = 0;
        }
    }
}

@end