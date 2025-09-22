/**
 * JARVISPythonBridge.m
 * JARVIS Voice Unlock System
 *
 * Implementation of Python integration bridge.
 */

#import "JARVISPythonBridge.h"
#import <os/log.h>

// Python result implementation
@interface JARVISPythonResult ()
@property (nonatomic, readwrite) BOOL success;
@property (nonatomic, readwrite, nullable) id result;
@property (nonatomic, readwrite, nullable) NSError *error;
@property (nonatomic, readwrite) NSTimeInterval executionTime;
@end

@implementation JARVISPythonResult
@end

// Main implementation
@interface JARVISPythonBridge ()

@property (nonatomic, strong) NSTask *pythonProcess;
@property (nonatomic, strong) NSPipe *inputPipe;
@property (nonatomic, strong) NSPipe *outputPipe;
@property (nonatomic, strong) NSPipe *errorPipe;
@property (nonatomic, strong) dispatch_queue_t processingQueue;
@property (nonatomic, strong) NSMutableDictionary<NSString *, void (^)(JARVISPythonResult *)> *pendingCallbacks;
@property (nonatomic, strong) os_log_t logger;

@property (nonatomic, readwrite) JARVISPythonBridgeState state;
@property (nonatomic, strong) NSMutableData *outputBuffer;
@property (nonatomic, strong) NSMutableData *errorBuffer;

@end

@implementation JARVISPythonBridge

- (instancetype)init {
    self = [super init];
    if (self) {
        _processingQueue = dispatch_queue_create("com.jarvis.pythonbridge", DISPATCH_QUEUE_SERIAL);
        _pendingCallbacks = [NSMutableDictionary dictionary];
        _logger = os_log_create("com.jarvis.voiceunlock", "pythonbridge");
        
        _outputBuffer = [NSMutableData data];
        _errorBuffer = [NSMutableData data];
        
        // Default configuration
        _pythonPath = @"/Users/derekjrussell/miniforge3/bin/python3";
        _scriptsDirectory = [@"~/.jarvis/voice_unlock/scripts" stringByExpandingTildeInPath];
        _pythonModulePaths = @[];
        _enableDebugLogging = NO;
        
        _state = JARVISPythonBridgeStateInactive;
    }
    return self;
}

- (void)dealloc {
    [self stopBridge];
}

#pragma mark - Bridge Lifecycle

- (BOOL)startBridgeWithError:(NSError **)error {
    if (self.state == JARVISPythonBridgeStateActive) {
        return YES;
    }
    
    self.state = JARVISPythonBridgeStateInitializing;
    os_log_info(self.logger, "Starting Python bridge");
    
    // Create pipes
    self.inputPipe = [NSPipe pipe];
    self.outputPipe = [NSPipe pipe];
    self.errorPipe = [NSPipe pipe];
    
    // Create Python process
    self.pythonProcess = [[NSTask alloc] init];
    self.pythonProcess.launchPath = self.pythonPath;
    
    // Bridge script path
    NSString *bridgeScript = [self.scriptsDirectory stringByAppendingPathComponent:@"voice_unlock_bridge.py"];
    
    // Ensure bridge script exists
    if (![[NSFileManager defaultManager] fileExistsAtPath:bridgeScript]) {
        [self createBridgeScript:bridgeScript];
    }
    
    // Set arguments
    NSMutableArray *arguments = [NSMutableArray arrayWithObject:bridgeScript];
    if (self.enableDebugLogging) {
        [arguments addObject:@"--debug"];
    }
    self.pythonProcess.arguments = arguments;
    
    // Set environment
    NSMutableDictionary *environment = [[[NSProcessInfo processInfo] environment] mutableCopy];
    environment[@"PYTHONUNBUFFERED"] = @"1";
    
    // Add custom module paths
    if (self.pythonModulePaths.count > 0) {
        NSString *pythonPath = [self.pythonModulePaths componentsJoinedByString:@":"];
        if (environment[@"PYTHONPATH"]) {
            environment[@"PYTHONPATH"] = [NSString stringWithFormat:@"%@:%@", 
                                         environment[@"PYTHONPATH"], pythonPath];
        } else {
            environment[@"PYTHONPATH"] = pythonPath;
        }
    }
    self.pythonProcess.environment = environment;
    
    // Set pipes
    self.pythonProcess.standardInput = self.inputPipe;
    self.pythonProcess.standardOutput = self.outputPipe;
    self.pythonProcess.standardError = self.errorPipe;
    
    // Set up output handlers
    [self setupOutputHandlers];
    
    // Launch process
    @try {
        [self.pythonProcess launch];
    } @catch (NSException *exception) {
        os_log_error(self.logger, "Failed to launch Python process: %@", exception);
        self.state = JARVISPythonBridgeStateError;
        
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                         code:1001
                                     userInfo:@{NSLocalizedDescriptionKey: exception.reason ?: @"Failed to launch Python"}];
        }
        return NO;
    }
    
    // Wait for initialization
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block BOOL initialized = NO;
    
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), 
                   self.processingQueue, ^{
        // Send initialization command
        NSDictionary *initCommand = @{
            @"type": @"init",
            @"config": @{
                @"voice_unlock": @YES,
                @"debug": @(self.enableDebugLogging)
            }
        };
        
        if ([self sendMessage:initCommand]) {
            initialized = YES;
            self.state = JARVISPythonBridgeStateActive;
        }
        dispatch_semaphore_signal(semaphore);
    });
    
    dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
    
    if (initialized) {
        os_log_info(self.logger, "Python bridge initialized successfully");
        
        if ([self.delegate respondsToSelector:@selector(pythonBridgeDidInitialize)]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.delegate pythonBridgeDidInitialize];
            });
        }
        
        return YES;
    } else {
        self.state = JARVISPythonBridgeStateError;
        if (error) {
            *error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                         code:1002
                                     userInfo:@{NSLocalizedDescriptionKey: @"Failed to initialize Python bridge"}];
        }
        return NO;
    }
}

- (void)stopBridge {
    if (self.state == JARVISPythonBridgeStateInactive) {
        return;
    }
    
    os_log_info(self.logger, "Stopping Python bridge");
    self.state = JARVISPythonBridgeStateInactive;
    
    // Send shutdown command
    [self sendMessage:@{@"type": @"shutdown"}];
    
    // Close pipes
    [[self.inputPipe fileHandleForWriting] closeFile];
    [[self.outputPipe fileHandleForReading] closeFile];
    [[self.errorPipe fileHandleForReading] closeFile];
    
    // Terminate process
    if (self.pythonProcess.isRunning) {
        [self.pythonProcess terminate];
        
        // Wait for termination
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2.0 * NSEC_PER_SEC)), 
                       dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            if (self.pythonProcess.isRunning) {
                [self.pythonProcess interrupt];
            }
        });
    }
    
    [self.pendingCallbacks removeAllObjects];
}

#pragma mark - Output Handling

- (void)setupOutputHandlers {
    // Handle standard output
    NSFileHandle *outputHandle = [self.outputPipe fileHandleForReading];
    outputHandle.readabilityHandler = ^(NSFileHandle *handle) {
        NSData *data = handle.availableData;
        if (data.length > 0) {
            [self processOutputData:data];
        }
    };
    
    // Handle standard error
    NSFileHandle *errorHandle = [self.errorPipe fileHandleForReading];
    errorHandle.readabilityHandler = ^(NSFileHandle *handle) {
        NSData *data = handle.availableData;
        if (data.length > 0) {
            NSString *errorString = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            if (errorString) {
                os_log_error(self.logger, "Python error: %@", errorString);
                
                if ([self.delegate respondsToSelector:@selector(pythonBridgeDidReceiveLog:level:)]) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self.delegate pythonBridgeDidReceiveLog:errorString level:@"error"];
                    });
                }
            }
        }
    };
    
    // Handle process termination
    self.pythonProcess.terminationHandler = ^(NSTask *task) {
        os_log_info(self.logger, "Python process terminated with status %d", task.terminationStatus);
        
        dispatch_async(self.processingQueue, ^{
            self.state = JARVISPythonBridgeStateInactive;
            
            if (task.terminationStatus != 0 && 
                [self.delegate respondsToSelector:@selector(pythonBridgeDidFailWithError:)]) {
                NSError *error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                                     code:task.terminationStatus
                                                 userInfo:@{NSLocalizedDescriptionKey: @"Python process terminated unexpectedly"}];
                
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.delegate pythonBridgeDidFailWithError:error];
                });
            }
        });
    };
}

- (void)processOutputData:(NSData *)data {
    dispatch_async(self.processingQueue, ^{
        [self.outputBuffer appendData:data];
        
        // Look for complete JSON messages (newline delimited)
        NSData *newlineData = [@"\n" dataUsingEncoding:NSUTF8StringEncoding];
        NSRange range;
        
        while ((range = [self.outputBuffer rangeOfData:newlineData 
                                               options:0 
                                                 range:NSMakeRange(0, self.outputBuffer.length)]).location != NSNotFound) {
            
            NSData *messageData = [self.outputBuffer subdataWithRange:NSMakeRange(0, range.location)];
            [self.outputBuffer replaceBytesInRange:NSMakeRange(0, range.location + range.length) withBytes:NULL length:0];
            
            NSString *messageString = [[NSString alloc] initWithData:messageData encoding:NSUTF8StringEncoding];
            if (!messageString) continue;
            
            NSError *error = nil;
            NSDictionary *message = [NSJSONSerialization JSONObjectWithData:messageData options:0 error:&error];
            
            if (!error && [message isKindOfClass:[NSDictionary class]]) {
                [self handlePythonMessage:message];
            } else {
                os_log_error(self.logger, "Failed to parse Python message: %@", error ?: messageString);
            }
        }
    });
}

- (void)handlePythonMessage:(NSDictionary *)message {
    NSString *type = message[@"type"];
    NSString *messageId = message[@"id"];
    
    if ([type isEqualToString:@"log"]) {
        NSString *logMessage = message[@"message"];
        NSString *level = message[@"level"] ?: @"info";
        
        if (self.enableDebugLogging) {
            os_log_info(self.logger, "[Python %@] %@", level, logMessage);
        }
        
        if ([self.delegate respondsToSelector:@selector(pythonBridgeDidReceiveLog:level:)]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.delegate pythonBridgeDidReceiveLog:logMessage level:level];
            });
        }
        
    } else if ([type isEqualToString:@"response"] && messageId) {
        void (^callback)(JARVISPythonResult *) = self.pendingCallbacks[messageId];
        if (callback) {
            [self.pendingCallbacks removeObjectForKey:messageId];
            
            JARVISPythonResult *result = [[JARVISPythonResult alloc] init];
            result.success = [message[@"success"] boolValue];
            result.result = message[@"result"];
            
            if (!result.success && message[@"error"]) {
                NSDictionary *errorInfo = message[@"error"];
                result.error = [NSError errorWithDomain:@"JARVISPython"
                                                  code:[errorInfo[@"code"] integerValue]
                                              userInfo:@{NSLocalizedDescriptionKey: errorInfo[@"message"] ?: @"Unknown error"}];
            }
            
            NSNumber *executionTime = message[@"execution_time"];
            result.executionTime = executionTime ? [executionTime doubleValue] : 0.0;
            
            dispatch_async(dispatch_get_main_queue(), ^{
                callback(result);
            });
        }
    }
}

#pragma mark - Message Passing

- (BOOL)isActive {
    return self.state == JARVISPythonBridgeStateActive;
}

- (BOOL)sendMessage:(NSDictionary *)message {
    if (!self.isActive) {
        return NO;
    }
    
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:message options:0 error:&error];
    
    if (error) {
        os_log_error(self.logger, "Failed to serialize message: %@", error);
        return NO;
    }
    
    NSMutableData *data = [jsonData mutableCopy];
    [data appendData:[@"\n" dataUsingEncoding:NSUTF8StringEncoding]];
    
    @try {
        [[self.inputPipe fileHandleForWriting] writeData:data];
        return YES;
    } @catch (NSException *exception) {
        os_log_error(self.logger, "Failed to write to Python process: %@", exception);
        return NO;
    }
}

- (JARVISPythonResult *)sendMessageAndWait:(NSDictionary *)message timeout:(NSTimeInterval)timeout {
    if (!self.isActive) {
        JARVISPythonResult *result = [[JARVISPythonResult alloc] init];
        result.success = NO;
        result.error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                          code:1003
                                      userInfo:@{NSLocalizedDescriptionKey: @"Python bridge not active"}];
        return result;
    }
    
    __block JARVISPythonResult *result = nil;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    
    NSString *messageId = [[NSUUID UUID] UUIDString];
    NSMutableDictionary *fullMessage = [message mutableCopy];
    fullMessage[@"id"] = messageId;
    
    self.pendingCallbacks[messageId] = ^(JARVISPythonResult *pythonResult) {
        result = pythonResult;
        dispatch_semaphore_signal(semaphore);
    };
    
    if (![self sendMessage:fullMessage]) {
        [self.pendingCallbacks removeObjectForKey:messageId];
        
        result = [[JARVISPythonResult alloc] init];
        result.success = NO;
        result.error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                          code:1004
                                      userInfo:@{NSLocalizedDescriptionKey: @"Failed to send message"}];
        return result;
    }
    
    dispatch_time_t timeoutTime = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(semaphore, timeoutTime) != 0) {
        [self.pendingCallbacks removeObjectForKey:messageId];
        
        result = [[JARVISPythonResult alloc] init];
        result.success = NO;
        result.error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                          code:1005
                                      userInfo:@{NSLocalizedDescriptionKey: @"Request timeout"}];
    }
    
    return result;
}

#pragma mark - Function Calls

- (void)callPythonFunction:(NSString *)functionName
                arguments:(nullable NSArray *)arguments
               completion:(void (^)(JARVISPythonResult *))completion {
    
    NSString *messageId = [[NSUUID UUID] UUIDString];
    NSDictionary *message = @{
        @"id": messageId,
        @"type": @"function_call",
        @"function": functionName,
        @"args": arguments ?: @[]
    };
    
    self.pendingCallbacks[messageId] = completion;
    
    if (![self sendMessage:message]) {
        [self.pendingCallbacks removeObjectForKey:messageId];
        
        JARVISPythonResult *result = [[JARVISPythonResult alloc] init];
        result.success = NO;
        result.error = [NSError errorWithDomain:@"JARVISPythonBridge"
                                          code:1004
                                      userInfo:@{NSLocalizedDescriptionKey: @"Failed to send function call"}];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(result);
        });
    }
}

- (id)callPythonFunctionSync:(NSString *)functionName
                   arguments:(nullable NSArray *)arguments
                       error:(NSError **)error {
    
    JARVISPythonResult *result = [self sendMessageAndWait:@{
        @"type": @"function_call",
        @"function": functionName,
        @"args": arguments ?: @[]
    } timeout:30.0];
    
    if (error && result.error) {
        *error = result.error;
    }
    
    return result.success ? result.result : nil;
}

#pragma mark - Voice Processing

- (NSDictionary *)processAudioForWakePhrase:(NSData *)audioData {
    NSDictionary *message = @{
        @"type": @"wake_phrase_detection",
        @"audio_data": [audioData base64EncodedStringWithOptions:0]
    };
    
    JARVISPythonResult *result = [self sendMessageAndWait:message timeout:5.0];
    
    if (result.success && [result.result isKindOfClass:[NSDictionary class]]) {
        return result.result;
    }
    
    return @{@"detected": @NO};
}

- (NSDictionary *)extractVoiceFeatures:(NSData *)audioData {
    NSDictionary *message = @{
        @"type": @"feature_extraction",
        @"audio_data": [audioData base64EncodedStringWithOptions:0]
    };
    
    JARVISPythonResult *result = [self sendMessageAndWait:message timeout:10.0];
    
    if (result.success && [result.result isKindOfClass:[NSDictionary class]]) {
        return result.result;
    }
    
    return @{};
}

- (float)compareVoiceprints:(NSArray<NSNumber *> *)features1 
                       with:(NSArray<NSNumber *> *)features2 {
    
    NSDictionary *message = @{
        @"type": @"voiceprint_comparison",
        @"features1": features1,
        @"features2": features2
    };
    
    JARVISPythonResult *result = [self sendMessageAndWait:message timeout:2.0];
    
    if (result.success && [result.result isKindOfClass:[NSNumber class]]) {
        return [result.result floatValue];
    }
    
    return 0.0f;
}

#pragma mark - Utility

- (BOOL)loadPythonModule:(NSString *)moduleName error:(NSError **)error {
    NSDictionary *message = @{
        @"type": @"load_module",
        @"module": moduleName
    };
    
    JARVISPythonResult *result = [self sendMessageAndWait:message timeout:10.0];
    
    if (!result.success && error) {
        *error = result.error;
    }
    
    return result.success;
}

- (NSString *)getPythonVersion {
    id version = [self callPythonFunctionSync:@"sys.version" arguments:nil error:nil];
    return [version isKindOfClass:[NSString class]] ? version : @"Unknown";
}

- (NSArray<NSString *> *)getLoadedModules {
    id modules = [self callPythonFunctionSync:@"list_modules" arguments:nil error:nil];
    return [modules isKindOfClass:[NSArray class]] ? modules : @[];
}

- (BOOL)isPythonFunctionAvailable:(NSString *)functionName {
    id result = [self callPythonFunctionSync:@"function_exists" 
                                   arguments:@[functionName] 
                                       error:nil];
    return [result isKindOfClass:[NSNumber class]] ? [result boolValue] : NO;
}

#pragma mark - Bridge Script Creation

- (void)createBridgeScript:(NSString *)path {
    // Create directory if needed
    NSString *directory = [path stringByDeletingLastPathComponent];
    [[NSFileManager defaultManager] createDirectoryAtPath:directory
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    
    // Create basic bridge script
    NSString *bridgeScript = @"#!/usr/bin/env python3\n"
    @"\"\"\"JARVIS Voice Unlock Python Bridge\"\"\"\n"
    @"\n"
    @"import json\n"
    @"import sys\n"
    @"import logging\n"
    @"import base64\n"
    @"import numpy as np\n"
    @"from datetime import datetime\n"
    @"\n"
    @"# Configure logging\n"
    @"logging.basicConfig(level=logging.INFO)\n"
    @"logger = logging.getLogger('jarvis_bridge')\n"
    @"\n"
    @"def send_response(message_id, success, result=None, error=None, execution_time=0.0):\n"
    @"    response = {\n"
    @"        'type': 'response',\n"
    @"        'id': message_id,\n"
    @"        'success': success,\n"
    @"        'result': result,\n"
    @"        'execution_time': execution_time\n"
    @"    }\n"
    @"    if error:\n"
    @"        response['error'] = {'code': 500, 'message': str(error)}\n"
    @"    print(json.dumps(response), flush=True)\n"
    @"\n"
    @"def process_message(message):\n"
    @"    msg_type = message.get('type')\n"
    @"    msg_id = message.get('id')\n"
    @"    \n"
    @"    try:\n"
    @"        if msg_type == 'init':\n"
    @"            send_response(msg_id, True, {'initialized': True})\n"
    @"            \n"
    @"        elif msg_type == 'shutdown':\n"
    @"            sys.exit(0)\n"
    @"            \n"
    @"        elif msg_type == 'wake_phrase_detection':\n"
    @"            # Simplified wake phrase detection\n"
    @"            audio_data = base64.b64decode(message.get('audio_data', ''))\n"
    @"            detected = len(audio_data) > 1000  # Simplified check\n"
    @"            send_response(msg_id, True, {\n"
    @"                'detected': detected,\n"
    @"                'phrase': 'Hello JARVIS, unlock my Mac' if detected else None,\n"
    @"                'confidence': 0.95 if detected else 0.0\n"
    @"            })\n"
    @"            \n"
    @"        elif msg_type == 'feature_extraction':\n"
    @"            # Simplified feature extraction\n"
    @"            features = np.random.randn(128).tolist()\n"
    @"            send_response(msg_id, True, {'features': features})\n"
    @"            \n"
    @"        elif msg_type == 'function_call':\n"
    @"            func = message.get('function')\n"
    @"            args = message.get('args', [])\n"
    @"            \n"
    @"            if func == 'sys.version':\n"
    @"                send_response(msg_id, True, sys.version)\n"
    @"            else:\n"
    @"                send_response(msg_id, False, error=f'Unknown function: {func}')\n"
    @"                \n"
    @"        else:\n"
    @"            send_response(msg_id, False, error=f'Unknown message type: {msg_type}')\n"
    @"            \n"
    @"    except Exception as e:\n"
    @"        logger.error(f'Error processing message: {e}')\n"
    @"        send_response(msg_id, False, error=str(e))\n"
    @"\n"
    @"def main():\n"
    @"    logger.info('JARVIS Voice Unlock Python Bridge started')\n"
    @"    \n"
    @"    while True:\n"
    @"        try:\n"
    @"            line = sys.stdin.readline()\n"
    @"            if not line:\n"
    @"                break\n"
    @"                \n"
    @"            message = json.loads(line.strip())\n"
    @"            process_message(message)\n"
    @"            \n"
    @"        except json.JSONDecodeError as e:\n"
    @"            logger.error(f'Invalid JSON: {e}')\n"
    @"        except KeyboardInterrupt:\n"
    @"            break\n"
    @"        except Exception as e:\n"
    @"            logger.error(f'Unexpected error: {e}')\n"
    @"\n"
    @"if __name__ == '__main__':\n"
    @"    main()\n";
    
    [bridgeScript writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
    
    // Make executable
    [[NSFileManager defaultManager] setAttributes:@{NSFilePosixPermissions: @(0755)}
                                      ofItemAtPath:path
                                             error:nil];
}

@end

#pragma mark - Voice Processing Extension

@implementation JARVISPythonBridge (VoiceProcessing)

- (void)detectWakePhrase:(NSData *)audioData
              completion:(void (^)(BOOL, NSString *, float))completion {
    
    [self callPythonFunction:@"detect_wake_phrase"
                   arguments:@[[audioData base64EncodedStringWithOptions:0]]
                  completion:^(JARVISPythonResult *result) {
        if (result.success && [result.result isKindOfClass:[NSDictionary class]]) {
            NSDictionary *detection = result.result;
            BOOL detected = [detection[@"detected"] boolValue];
            NSString *phrase = detection[@"phrase"] ?: @"";
            float confidence = [detection[@"confidence"] floatValue];
            
            completion(detected, phrase, confidence);
        } else {
            completion(NO, @"", 0.0f);
        }
    }];
}

- (void)authenticateVoice:(NSData *)audioData
                 forUser:(NSString *)userID
              completion:(void (^)(BOOL, float, NSDictionary *))completion {
    
    [self callPythonFunction:@"authenticate_voice"
                   arguments:@[[audioData base64EncodedStringWithOptions:0], userID]
                  completion:^(JARVISPythonResult *result) {
        if (result.success && [result.result isKindOfClass:[NSDictionary class]]) {
            NSDictionary *authResult = result.result;
            BOOL authenticated = [authResult[@"authenticated"] boolValue];
            float confidence = [authResult[@"confidence"] floatValue];
            
            completion(authenticated, confidence, authResult);
        } else {
            completion(NO, 0.0f, @{});
        }
    }];
}

- (NSDictionary *)analyzeAudioQuality:(NSData *)audioData {
    return [self callPythonFunctionSync:@"analyze_audio_quality"
                             arguments:@[[audioData base64EncodedStringWithOptions:0]]
                                 error:nil] ?: @{};
}

- (void)transcribeAudio:(NSData *)audioData
             completion:(void (^)(NSString *, float, NSError *))completion {
    
    [self callPythonFunction:@"transcribe_audio"
                   arguments:@[[audioData base64EncodedStringWithOptions:0]]
                  completion:^(JARVISPythonResult *result) {
        if (result.success && [result.result isKindOfClass:[NSDictionary class]]) {
            NSDictionary *transcription = result.result;
            NSString *transcript = transcription[@"transcript"] ?: @"";
            float confidence = [transcription[@"confidence"] floatValue];
            
            completion(transcript, confidence, nil);
        } else {
            completion(@"", 0.0f, result.error);
        }
    }];
}

@end