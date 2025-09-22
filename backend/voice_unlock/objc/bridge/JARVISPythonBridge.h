/**
 * JARVISPythonBridge.h
 * JARVIS Voice Unlock System
 *
 * Bridge for direct Python integration using Python C API
 * and subprocess communication.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Python bridge state
typedef NS_ENUM(NSInteger, JARVISPythonBridgeState) {
    JARVISPythonBridgeStateInactive = 0,
    JARVISPythonBridgeStateInitializing,
    JARVISPythonBridgeStateActive,
    JARVISPythonBridgeStateError
};

// Python function result
@interface JARVISPythonResult : NSObject
@property (nonatomic, readonly) BOOL success;
@property (nonatomic, readonly, nullable) id result;
@property (nonatomic, readonly, nullable) NSError *error;
@property (nonatomic, readonly) NSTimeInterval executionTime;
@end

// Python bridge delegate
@protocol JARVISPythonBridgeDelegate <NSObject>
@optional
- (void)pythonBridgeDidInitialize;
- (void)pythonBridgeDidFailWithError:(NSError *)error;
- (void)pythonBridgeDidReceiveLog:(NSString *)message level:(NSString *)level;
@end

// Main Python bridge interface
@interface JARVISPythonBridge : NSObject

@property (nonatomic, weak, nullable) id<JARVISPythonBridgeDelegate> delegate;
@property (nonatomic, readonly) JARVISPythonBridgeState state;
@property (nonatomic, readonly) BOOL isActive;

// Configuration
@property (nonatomic, strong) NSString *pythonPath;
@property (nonatomic, strong) NSString *scriptsDirectory;
@property (nonatomic, strong) NSArray<NSString *> *pythonModulePaths;
@property (nonatomic, assign) BOOL enableDebugLogging;

// Initialization
- (BOOL)startBridgeWithError:(NSError **)error;
- (void)stopBridge;
- (BOOL)loadPythonModule:(NSString *)moduleName error:(NSError **)error;

// Message passing
- (BOOL)sendMessage:(NSDictionary *)message;
- (JARVISPythonResult *)sendMessageAndWait:(NSDictionary *)message timeout:(NSTimeInterval)timeout;

// Function calls
- (void)callPythonFunction:(NSString *)functionName
                arguments:(nullable NSArray *)arguments
               completion:(void (^)(JARVISPythonResult *result))completion;

- (id)callPythonFunctionSync:(NSString *)functionName
                   arguments:(nullable NSArray *)arguments
                       error:(NSError **)error;

// Voice processing
- (NSDictionary *)processAudioForWakePhrase:(NSData *)audioData;
- (NSDictionary *)extractVoiceFeatures:(NSData *)audioData;
- (float)compareVoiceprints:(NSArray<NSNumber *> *)features1 
                       with:(NSArray<NSNumber *> *)features2;

// Utility
- (NSString *)getPythonVersion;
- (NSArray<NSString *> *)getLoadedModules;
- (BOOL)isPythonFunctionAvailable:(NSString *)functionName;

@end

// Voice processing extension
@interface JARVISPythonBridge (VoiceProcessing)

// Wake phrase detection
- (void)detectWakePhrase:(NSData *)audioData
              completion:(void (^)(BOOL detected, NSString *phrase, float confidence))completion;

// Voice authentication
- (void)authenticateVoice:(NSData *)audioData
                 forUser:(NSString *)userID
              completion:(void (^)(BOOL authenticated, float confidence, NSDictionary *details))completion;

// Audio quality analysis
- (NSDictionary *)analyzeAudioQuality:(NSData *)audioData;

// Speech-to-text
- (void)transcribeAudio:(NSData *)audioData
             completion:(void (^)(NSString *transcript, float confidence, NSError *error))completion;

@end

NS_ASSUME_NONNULL_END