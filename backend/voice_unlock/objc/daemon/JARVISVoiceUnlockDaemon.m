/**
 * JARVISVoiceUnlockDaemon.m
 * JARVIS Voice Unlock System
 *
 * Implementation of the main daemon that monitors for voice unlock phrases
 * when the screen is locked.
 */

#import "JARVISVoiceUnlockDaemon.h"
#import "JARVISVoiceAuthenticator.h"
#import "JARVISScreenUnlockManager.h"
#import "JARVISVoiceMonitor.h"
#import "JARVISPythonBridge.h"
#import "JARVISPermissionManager.h"
#import "JARVISWebSocketBridge.h"
#import "../server/JARVISWebSocketServer.h"

#import <os/log.h>
#import <dispatch/dispatch.h>

// Notification constants
NSString *const JARVISVoiceUnlockStatusChangedNotification = @"JARVISVoiceUnlockStatusChanged";
NSString *const JARVISVoiceUnlockAuthenticationFailedNotification = @"JARVISVoiceUnlockAuthenticationFailed";
NSString *const JARVISVoiceUnlockAuthenticationSucceededNotification = @"JARVISVoiceUnlockAuthenticationSucceeded";

// Error domain
NSString *const JARVISVoiceUnlockErrorDomain = @"com.jarvis.voiceunlock.error";

// Private interface
@interface JARVISVoiceUnlockDaemon ()

@property (nonatomic, strong) JARVISVoiceAuthenticator *authenticator;
@property (nonatomic, strong) JARVISScreenUnlockManager *unlockManager;
@property (nonatomic, strong) JARVISVoiceMonitor *voiceMonitor;
@property (nonatomic, strong) JARVISPythonBridge *pythonBridge;
@property (nonatomic, strong) JARVISPermissionManager *permissionManager;
@property (nonatomic, strong) JARVISWebSocketBridge *webSocketBridge;
@property (nonatomic, strong) JARVISWebSocketServer *webSocketServer;

@property (nonatomic, strong) dispatch_queue_t processingQueue;
@property (nonatomic, strong) os_log_t logger;

@property (nonatomic, readwrite) JARVISVoiceUnlockState state;
@property (nonatomic, readwrite) BOOL isMonitoring;
@property (nonatomic, readwrite) BOOL isScreenLocked;
@property (nonatomic, readwrite) NSString *enrolledUserIdentifier;
@property (nonatomic, readwrite) NSDate *lastUnlockAttempt;
@property (nonatomic, readwrite) NSUInteger failedAttemptCount;

@property (nonatomic, strong) NSMutableDictionary *configuration;
@property (nonatomic, strong) NSMutableArray<NSString *> *dynamicUnlockPhrases;

@property (nonatomic, strong) NSTimer *lockCheckTimer;
@property (nonatomic, strong) NSDate *lockoutEndDate;

@end

@implementation JARVISVoiceUnlockDaemon

#pragma mark - Singleton

+ (instancetype)sharedDaemon {
    static JARVISVoiceUnlockDaemon *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

#pragma mark - Initialization

- (instancetype)init {
    self = [super init];
    if (self) {
        // Initialize logger
        _logger = os_log_create("com.jarvis.voiceunlock", "daemon");
        
        // Initialize processing queue
        _processingQueue = dispatch_queue_create("com.jarvis.voiceunlock.processing", DISPATCH_QUEUE_SERIAL);
        
        // Initialize components
        _authenticator = [[JARVISVoiceAuthenticator alloc] init];
        _unlockManager = [[JARVISScreenUnlockManager alloc] init];
        _voiceMonitor = [[JARVISVoiceMonitor alloc] init];
        _pythonBridge = [[JARVISPythonBridge alloc] init];
        _permissionManager = [[JARVISPermissionManager alloc] init];
        _webSocketBridge = [[JARVISWebSocketBridge alloc] init];
        _webSocketServer = [[JARVISWebSocketServer alloc] initWithPort:8765];
        
        // Set default configuration
        _options = JARVISVoiceUnlockOptionEnableAntiSpoofing | JARVISVoiceUnlockOptionEnableAdaptiveThresholds;
        _authenticationTimeout = 10.0;
        _maxFailedAttempts = 5;
        _lockoutDuration = 300.0; // 5 minutes
        
        // Initialize state
        _state = JARVISVoiceUnlockStateInactive;
        _failedAttemptCount = 0;
        
        // Initialize configuration
        _configuration = [NSMutableDictionary dictionary];
        _dynamicUnlockPhrases = [NSMutableArray arrayWithObjects:
            @"Hello JARVIS, unlock my Mac",
            @"JARVIS, this is Derek",
            @"Open sesame, JARVIS",
            nil];
        
        // Set up voice monitor delegate
        __weak typeof(self) weakSelf = self;
        _voiceMonitor.audioDetectedBlock = ^(NSData *audioData) {
            [weakSelf processAudioData:audioData];
        };
        
        // Load configuration
        [self loadDefaultConfiguration];
        
        // Register for screen lock notifications
        [self registerForScreenLockNotifications];
        
        os_log_info(self.logger, "JARVISVoiceUnlockDaemon initialized");
    }
    return self;
}

#pragma mark - Configuration

- (void)loadDefaultConfiguration {
    NSString *configPath = [@"~/.jarvis/voice_unlock/config.json" stringByExpandingTildeInPath];
    [self loadConfigurationFromFile:configPath];
}

- (void)loadConfigurationFromFile:(NSString *)path {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path];
    
    if (data) {
        NSDictionary *config = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (config && !error) {
            [self updateConfiguration:config];
            os_log_info(self.logger, "Loaded configuration from %@", path);
        } else {
            os_log_error(self.logger, "Failed to parse configuration: %@", error);
        }
    }
}

- (void)updateConfiguration:(NSDictionary *)config {
    dispatch_sync(self.processingQueue, ^{
        [self.configuration addEntriesFromDictionary:config];
        
        // Update unlock phrases if provided
        NSArray *phrases = config[@"unlockPhrases"];
        if ([phrases isKindOfClass:[NSArray class]]) {
            [self.dynamicUnlockPhrases removeAllObjects];
            [self.dynamicUnlockPhrases addObjectsFromArray:phrases];
        }
        
        // Update options
        NSDictionary *options = config[@"options"];
        if ([options isKindOfClass:[NSDictionary class]]) {
            if ([options[@"livenessDetection"] boolValue]) {
                self.options |= JARVISVoiceUnlockOptionEnableLivenessDetection;
            }
            if ([options[@"continuousAuth"] boolValue]) {
                self.options |= JARVISVoiceUnlockOptionEnableContinuousAuthentication;
            }
            if ([options[@"debug"] boolValue]) {
                self.options |= JARVISVoiceUnlockOptionEnableDebugLogging;
            }
        }
        
        // Update thresholds
        NSNumber *timeout = config[@"authenticationTimeout"];
        if (timeout) {
            self.authenticationTimeout = [timeout doubleValue];
        }
        
        NSNumber *maxAttempts = config[@"maxFailedAttempts"];
        if (maxAttempts) {
            self.maxFailedAttempts = [maxAttempts unsignedIntegerValue];
        }
    });
}

- (NSDictionary *)currentConfiguration {
    return [self.configuration copy];
}

#pragma mark - Monitoring Control

- (BOOL)startMonitoringWithError:(NSError **)error {
    os_log_info(self.logger, "Starting voice unlock monitoring");
    
    // Check permissions
    if (![self.permissionManager checkAndRequestPermissions:error]) {
        return NO;
    }
    
    // Check if user is enrolled
    if (![self isUserEnrolled]) {
        if (error) {
            *error = [NSError errorWithDomain:JARVISVoiceUnlockErrorDomain
                                         code:1001
                                     userInfo:@{NSLocalizedDescriptionKey: @"No user enrolled for voice unlock"}];
        }
        return NO;
    }
    
    // Start Python bridge (optional for now)
    if (![self.pythonBridge startBridgeWithError:error]) {
        os_log_error(self.logger, "Python bridge failed to start: %@", *error);
        // Continue without Python bridge for testing
    }
    
    // Skip internal WebSocket server - using Python bridge instead
    /*
    NSError *wsError = nil;
    if (![self.webSocketServer startWithError:&wsError]) {
        os_log_error(self.logger, "Failed to start WebSocket server: %@", wsError);
        if (error) {
            *error = wsError;
        }
        return NO;
    }
    os_log_info(self.logger, "WebSocket server started on port 8765");
    */
    
    // Start WebSocket client bridge to connect to Python server
    [self.webSocketBridge startWithHost:@"localhost" port:8765];
    os_log_info(self.logger, "WebSocket bridge started, connecting to Python server on port 8765");
    
    // Start voice monitoring
    if (![self.voiceMonitor startMonitoring]) {
        if (error) {
            *error = [NSError errorWithDomain:JARVISVoiceUnlockErrorDomain
                                         code:1002
                                     userInfo:@{NSLocalizedDescriptionKey: @"Failed to start audio monitoring"}];
        }
        return NO;
    }
    
    self.isMonitoring = YES;
    self.state = JARVISVoiceUnlockStateMonitoring;
    
    // Start screen lock monitoring
    [self startScreenLockMonitoring];
    
    // Post notification
    [[NSNotificationCenter defaultCenter] postNotificationName:JARVISVoiceUnlockStatusChangedNotification
                                                        object:self
                                                      userInfo:@{@"status": @"started"}];
    
    os_log_info(self.logger, "Voice unlock monitoring started successfully");
    return YES;
}

- (void)stopMonitoring {
    os_log_info(self.logger, "Stopping voice unlock monitoring");
    
    self.isMonitoring = NO;
    self.state = JARVISVoiceUnlockStateInactive;
    
    // Stop components
    [self.voiceMonitor stopMonitoring];
    [self.pythonBridge stopBridge];
    [self.webSocketBridge stop];
    //[self.webSocketServer stop];
    
    // Stop timers
    [self.lockCheckTimer invalidate];
    self.lockCheckTimer = nil;
    
    // Post notification
    [[NSNotificationCenter defaultCenter] postNotificationName:JARVISVoiceUnlockStatusChangedNotification
                                                        object:self
                                                      userInfo:@{@"status": @"stopped"}];
}

#pragma mark - Screen Lock Monitoring

- (void)registerForScreenLockNotifications {
    NSDistributedNotificationCenter *center = [NSDistributedNotificationCenter defaultCenter];
    
    // Screen locked
    [center addObserver:self
               selector:@selector(screenDidLock:)
                   name:@"com.apple.screensaver.didstart"
                 object:nil];
    
    [center addObserver:self
               selector:@selector(screenDidLock:)
                   name:@"com.apple.screenIsLocked"
                 object:nil];
    
    // Screen unlocked
    [center addObserver:self
               selector:@selector(screenDidUnlock:)
                   name:@"com.apple.screensaver.didstop"
                 object:nil];
    
    [center addObserver:self
               selector:@selector(screenDidUnlock:)
                   name:@"com.apple.screenIsUnlocked"
                 object:nil];
}

- (void)startScreenLockMonitoring {
    // Check screen lock status periodically
    self.lockCheckTimer = [NSTimer scheduledTimerWithTimeInterval:5.0
                                                          target:self
                                                        selector:@selector(checkScreenLockStatus)
                                                        userInfo:nil
                                                         repeats:YES];
    
    // Initial check
    [self checkScreenLockStatus];
}

- (void)checkScreenLockStatus {
    BOOL locked = [self.unlockManager isScreenLocked];
    
    if (locked != self.isScreenLocked) {
        self.isScreenLocked = locked;
        os_log_info(self.logger, "Screen lock status changed: %@", locked ? @"LOCKED" : @"UNLOCKED");
        
        if (locked) {
            [self handleScreenLocked];
        } else {
            [self handleScreenUnlocked];
        }
    }
}

- (void)screenDidLock:(NSNotification *)notification {
    os_log_info(self.logger, "Screen lock notification received");
    self.isScreenLocked = YES;
    [self handleScreenLocked];
}

- (void)screenDidUnlock:(NSNotification *)notification {
    os_log_info(self.logger, "Screen unlock notification received");
    self.isScreenLocked = NO;
    [self handleScreenUnlocked];
}

- (void)handleScreenLocked {
    if (self.isMonitoring && self.options & JARVISVoiceUnlockOptionEnableContinuousAuthentication) {
        // Increase monitoring sensitivity when screen is locked
        [self.voiceMonitor setHighSensitivityMode:YES];
        
        // Notify Python backend
        [self.pythonBridge sendMessage:@{
            @"type": @"screen_locked",
            @"timestamp": @([[NSDate date] timeIntervalSince1970])
        }];
    }
}

- (void)handleScreenUnlocked {
    // Reduce monitoring sensitivity when screen is unlocked
    [self.voiceMonitor setHighSensitivityMode:NO];
    
    // Reset failed attempts on successful unlock
    self.failedAttemptCount = 0;
    
    // Notify Python backend
    [self.pythonBridge sendMessage:@{
        @"type": @"screen_unlocked",
        @"timestamp": @([[NSDate date] timeIntervalSince1970])
    }];
}

#pragma mark - Audio Processing

- (void)processAudioData:(NSData *)audioData {
    if (!self.isMonitoring || !self.isScreenLocked) {
        return;
    }
    
    // Check if in lockout period
    if (self.lockoutEndDate && [self.lockoutEndDate timeIntervalSinceNow] > 0) {
        os_log_info(self.logger, "In lockout period, ignoring audio");
        return;
    }
    
    dispatch_async(self.processingQueue, ^{
        self.state = JARVISVoiceUnlockStateProcessing;
        
        // Detect wake phrase
        NSString *detectedPhrase = [self detectWakePhraseInAudio:audioData];
        if (!detectedPhrase) {
            self.state = JARVISVoiceUnlockStateMonitoring;
            return;
        }
        
        os_log_info(self.logger, "Wake phrase detected: %@", detectedPhrase);
        
        // Perform voice authentication
        self.lastUnlockAttempt = [NSDate date];
        
        NSDictionary *authResult = [self.authenticator authenticateVoice:audioData
                                                           forUser:self.enrolledUserIdentifier];
        
        if ([authResult[@"success"] boolValue]) {
            [self handleSuccessfulAuthentication:authResult];
        } else {
            [self handleFailedAuthentication:authResult];
        }
    });
}

- (NSString *)detectWakePhraseInAudio:(NSData *)audioData {
    // Send audio to Python for wake phrase detection
    NSDictionary *response = [self.pythonBridge processAudioForWakePhrase:audioData];
    
    if ([response[@"detected"] boolValue]) {
        NSString *phrase = response[@"phrase"];
        
        // Check if it matches our unlock phrases
        for (NSString *unlockPhrase in self.dynamicUnlockPhrases) {
            if ([phrase localizedCaseInsensitiveContainsString:unlockPhrase]) {
                return phrase;
            }
        }
    }
    
    return nil;
}

#pragma mark - Authentication Handling

- (void)handleSuccessfulAuthentication:(NSDictionary *)authResult {
    os_log_info(self.logger, "Voice authentication successful");
    
    self.state = JARVISVoiceUnlockStateUnlocking;
    self.failedAttemptCount = 0;
    
    // Post success notification
    [[NSNotificationCenter defaultCenter] postNotificationName:JARVISVoiceUnlockAuthenticationSucceededNotification
                                                        object:self
                                                      userInfo:authResult];
    
    // Attempt to unlock screen
    NSError *unlockError = nil;
    BOOL unlocked = [self.unlockManager unlockScreenWithError:&unlockError];
    
    if (unlocked) {
        os_log_info(self.logger, "Screen unlocked successfully");
        
        // Log successful unlock
        [self logUnlockAttempt:YES details:@{
            @"user": self.enrolledUserIdentifier,
            @"confidence": authResult[@"confidence"],
            @"method": @"voice"
        }];
        
        // Send success to Python
        [self.pythonBridge sendMessage:@{
            @"type": @"unlock_success",
            @"user": self.enrolledUserIdentifier,
            @"timestamp": @([[NSDate date] timeIntervalSince1970])
        }];
        
    } else {
        os_log_error(self.logger, "Failed to unlock screen: %@", unlockError);
        
        // Send failure to Python
        [self.pythonBridge sendMessage:@{
            @"type": @"unlock_failure",
            @"error": unlockError.localizedDescription,
            @"timestamp": @([[NSDate date] timeIntervalSince1970])
        }];
    }
    
    self.state = self.isScreenLocked ? JARVISVoiceUnlockStateMonitoring : JARVISVoiceUnlockStateInactive;
}

- (void)handleFailedAuthentication:(NSDictionary *)authResult {
    os_log_info(self.logger, "Voice authentication failed: %@", authResult[@"reason"]);
    
    self.failedAttemptCount++;
    
    // Post failure notification
    [[NSNotificationCenter defaultCenter] postNotificationName:JARVISVoiceUnlockAuthenticationFailedNotification
                                                        object:self
                                                      userInfo:authResult];
    
    // Log failed attempt
    [self logUnlockAttempt:NO details:@{
        @"reason": authResult[@"reason"] ?: @"unknown",
        @"attemptCount": @(self.failedAttemptCount)
    }];
    
    // Check lockout
    if (self.failedAttemptCount >= self.maxFailedAttempts) {
        self.lockoutEndDate = [NSDate dateWithTimeIntervalSinceNow:self.lockoutDuration];
        os_log_info(self.logger, "Max failed attempts reached, entering lockout until %@", self.lockoutEndDate);
        
        // Send lockout notification to Python
        [self.pythonBridge sendMessage:@{
            @"type": @"lockout",
            @"duration": @(self.lockoutDuration),
            @"timestamp": @([[NSDate date] timeIntervalSince1970])
        }];
    }
    
    self.state = JARVISVoiceUnlockStateMonitoring;
}

#pragma mark - User Management

- (BOOL)isUserEnrolled {
    NSString *enrollmentPath = [@"~/.jarvis/voice_unlock/enrolled_users.json" stringByExpandingTildeInPath];
    
    if ([[NSFileManager defaultManager] fileExistsAtPath:enrollmentPath]) {
        NSData *data = [NSData dataWithContentsOfFile:enrollmentPath];
        if (data) {
            NSError *error = nil;
            NSDictionary *enrolledUsers = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            
            if (!error && enrolledUsers.count > 0) {
                // Get first enrolled user
                self.enrolledUserIdentifier = [[enrolledUsers allKeys] firstObject];
                return YES;
            }
        }
    }
    
    return NO;
}

#pragma mark - Status and Logging

- (NSDictionary *)getStatus {
    return @{
        @"state": @(self.state),
        @"isMonitoring": @(self.isMonitoring),
        @"isScreenLocked": @(self.isScreenLocked),
        @"enrolledUser": self.enrolledUserIdentifier ?: @"none",
        @"failedAttempts": @(self.failedAttemptCount),
        @"lastAttempt": self.lastUnlockAttempt ?: [NSNull null],
        @"inLockout": @(self.lockoutEndDate && [self.lockoutEndDate timeIntervalSinceNow] > 0),
        @"options": @(self.options)
    };
}

- (void)resetFailedAttempts {
    self.failedAttemptCount = 0;
    self.lockoutEndDate = nil;
    os_log_info(self.logger, "Failed attempts counter reset");
}

- (void)logUnlockAttempt:(BOOL)success details:(NSDictionary *)details {
    NSString *logPath = [@"~/.jarvis/voice_unlock/unlock_log.json" stringByExpandingTildeInPath];
    
    // Create log entry
    NSMutableDictionary *logEntry = [NSMutableDictionary dictionaryWithDictionary:@{
        @"timestamp": [NSDate date],
        @"success": @(success),
        @"details": details
    }];
    
    // Load existing log
    NSMutableArray *log = [NSMutableArray array];
    if ([[NSFileManager defaultManager] fileExistsAtPath:logPath]) {
        NSData *data = [NSData dataWithContentsOfFile:logPath];
        if (data) {
            NSArray *existingLog = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if (existingLog) {
                [log addObjectsFromArray:existingLog];
            }
        }
    }
    
    // Add new entry
    [log addObject:logEntry];
    
    // Keep only last 1000 entries
    if (log.count > 1000) {
        [log removeObjectsInRange:NSMakeRange(0, log.count - 1000)];
    }
    
    // Save log
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:log options:NSJSONWritingPrettyPrinted error:&error];
    if (jsonData && !error) {
        [jsonData writeToFile:logPath atomically:YES];
    }
}

#pragma mark - Test Methods

- (void)simulateVoiceUnlock:(NSString *)phrase {
    if (self.options & JARVISVoiceUnlockOptionEnableDebugLogging) {
        os_log_info(self.logger, "Simulating voice unlock with phrase: %@", phrase);
        
        // Simulate successful authentication
        [self handleSuccessfulAuthentication:@{
            @"success": @YES,
            @"confidence": @0.95,
            @"simulatxed": @YES
        }];
    }
}

- (void)simulateScreenLock {
    if (self.options & JARVISVoiceUnlockOptionEnableDebugLogging) {
        self.isScreenLocked = YES;
        [self handleScreenLocked];
    }
}

- (void)simulateScreenUnlock {
    if (self.options & JARVISVoiceUnlockOptionEnableDebugLogging) {
        self.isScreenLocked = NO;
        [self handleScreenUnlocked];
    }
}

#pragma mark - Getters

- (NSArray<NSString *> *)unlockPhrases {
    return [self.dynamicUnlockPhrases copy];
}

@end