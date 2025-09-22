/**
 * JARVISWebSocketBridge.m
 * JARVIS Voice Unlock System
 *
 * Implementation of WebSocket communication bridge.
 */

#import "JARVISWebSocketBridge.h"
#import <os/log.h>

// Define SRWebSocket interface for NSURLSession-based implementation
@interface SRWebSocket : NSObject
@property (nonatomic, weak) id delegate;
@property (nonatomic, readonly) NSURLSessionWebSocketTask *webSocketTask;
@property (nonatomic, strong) NSURLSession *session;

- (instancetype)initWithURLRequest:(NSURLRequest *)request;
- (void)open;
- (void)close;
- (void)sendString:(NSString *)string error:(NSError **)error;
- (void)sendData:(NSData *)data error:(NSError **)error;
@end

@protocol SRWebSocketDelegate <NSObject>
@optional
- (void)webSocketDidOpen:(SRWebSocket *)webSocket;
- (void)webSocket:(SRWebSocket *)webSocket didFailWithError:(NSError *)error;
- (void)webSocket:(SRWebSocket *)webSocket didCloseWithCode:(NSInteger)code reason:(NSString *)reason wasClean:(BOOL)wasClean;
- (void)webSocket:(SRWebSocket *)webSocket didReceiveMessageWithString:(NSString *)string;
- (void)webSocket:(SRWebSocket *)webSocket didReceiveMessageWithData:(NSData *)data;
@end

@implementation SRWebSocket

- (instancetype)initWithURLRequest:(NSURLRequest *)request {
    self = [super init];
    if (self) {
        NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
        _session = [NSURLSession sessionWithConfiguration:config];
        _webSocketTask = [_session webSocketTaskWithRequest:request];
    }
    return self;
}

- (void)open {
    [self.webSocketTask resume];
    [self receiveMessage];
    
    // Notify delegate after a short delay
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.1 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if ([self.delegate respondsToSelector:@selector(webSocketDidOpen:)]) {
            [self.delegate webSocketDidOpen:self];
        }
    });
}

- (void)close {
    [self.webSocketTask cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
}

- (void)sendString:(NSString *)string error:(NSError **)error {
    NSURLSessionWebSocketMessage *message = [[NSURLSessionWebSocketMessage alloc] initWithString:string];
    [self.webSocketTask sendMessage:message completionHandler:^(NSError * _Nullable err) {
        if (error && err) {
            *error = err;
        }
    }];
}

- (void)sendData:(NSData *)data error:(NSError **)error {
    NSURLSessionWebSocketMessage *message = [[NSURLSessionWebSocketMessage alloc] initWithData:data];
    [self.webSocketTask sendMessage:message completionHandler:^(NSError * _Nullable err) {
        if (error && err) {
            *error = err;
        }
    }];
}

- (void)receiveMessage {
    __weak typeof(self) weakSelf = self;
    [self.webSocketTask receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage * _Nullable message, NSError * _Nullable error) {
        if (error) {
            if ([weakSelf.delegate respondsToSelector:@selector(webSocket:didFailWithError:)]) {
                [weakSelf.delegate webSocket:weakSelf didFailWithError:error];
            }
            return;
        }
        
        if (message) {
            if (message.type == NSURLSessionWebSocketMessageTypeString) {
                if ([weakSelf.delegate respondsToSelector:@selector(webSocket:didReceiveMessageWithString:)]) {
                    [weakSelf.delegate webSocket:weakSelf didReceiveMessageWithString:message.string];
                }
            } else if (message.type == NSURLSessionWebSocketMessageTypeData) {
                if ([weakSelf.delegate respondsToSelector:@selector(webSocket:didReceiveMessageWithData:)]) {
                    [weakSelf.delegate webSocket:weakSelf didReceiveMessageWithData:message.data];
                }
            }
            
            // Continue receiving messages
            [weakSelf receiveMessage];
        }
    }];
}

@end

// Message implementation
@implementation JARVISWebSocketMessage

+ (instancetype)messageWithType:(JARVISMessageType)type payload:(NSDictionary *)payload {
    JARVISWebSocketMessage *message = [[JARVISWebSocketMessage alloc] init];
    message.type = type;
    message.payload = payload;
    message.timestamp = [NSDate date];
    message.identifier = [[NSUUID UUID] UUIDString];
    return message;
}

- (NSString *)typeString {
    switch (self.type) {
        case JARVISMessageTypeCommand: return @"command";
        case JARVISMessageTypeStatus: return @"status";
        case JARVISMessageTypeAudio: return @"audio";
        case JARVISMessageTypeAuthentication: return @"authentication";
        case JARVISMessageTypeScreenState: return @"screen_state";
        case JARVISMessageTypeConfiguration: return @"configuration";
        case JARVISMessageTypeHeartbeat: return @"heartbeat";
    }
}

- (NSString *)toJSONString {
    NSMutableDictionary *json = [NSMutableDictionary dictionary];
    json[@"id"] = self.identifier;
    json[@"type"] = [self typeString];
    json[@"timestamp"] = @([self.timestamp timeIntervalSince1970]);
    
    if (self.payload) {
        json[@"payload"] = self.payload;
    }
    
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:json options:0 error:&error];
    
    if (error || !jsonData) {
        return nil;
    }
    
    return [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
}

- (NSData *)toBinaryData {
    if (self.binaryData) {
        // Create header with message info
        NSMutableData *data = [NSMutableData data];
        
        // Header: type (1 byte) + id length (1 byte) + id
        uint8_t type = (uint8_t)self.type;
        [data appendBytes:&type length:1];
        
        NSData *idData = [self.identifier dataUsingEncoding:NSUTF8StringEncoding];
        uint8_t idLength = (uint8_t)idData.length;
        [data appendBytes:&idLength length:1];
        [data appendData:idData];
        
        // Append binary data
        [data appendData:self.binaryData];
        
        return data;
    }
    
    // Fall back to JSON
    return [[self toJSONString] dataUsingEncoding:NSUTF8StringEncoding];
}

@end

// Main bridge implementation
@interface JARVISWebSocketBridge () <SRWebSocketDelegate>

@property (nonatomic, strong) SRWebSocket *webSocket;
@property (nonatomic, strong) dispatch_queue_t socketQueue;
@property (nonatomic, strong) NSTimer *heartbeatTimer;
@property (nonatomic, strong) NSTimer *reconnectTimer;
@property (nonatomic, strong) os_log_t logger;

@property (nonatomic, readwrite) JARVISWebSocketState connectionState;
@property (nonatomic, readwrite) NSUInteger reconnectAttempts;

@property (nonatomic, strong) NSMutableDictionary<NSString *, void (^)(NSDictionary *, NSError *)> *pendingCallbacks;
@property (nonatomic, assign) BOOL isAudioStreaming;

@end

@implementation JARVISWebSocketBridge

- (instancetype)init {
    self = [super init];
    if (self) {
        _socketQueue = dispatch_queue_create("com.jarvis.websocket", DISPATCH_QUEUE_SERIAL);
        _logger = os_log_create("com.jarvis.voiceunlock", "websocket");
        _pendingCallbacks = [NSMutableDictionary dictionary];
        
        // Default configuration
        _serverHost = @"localhost";
        _serverPort = 8765;
        _enableAutoReconnect = YES;
        _reconnectDelay = 5.0;
        _maxReconnectAttempts = 10;
        _heartbeatInterval = 30.0;
        
        _connectionState = JARVISWebSocketStateDisconnected;
    }
    return self;
}

- (void)dealloc {
    [self stop];
}

#pragma mark - Connection Management

- (void)startWithPort:(NSUInteger)port {
    self.serverPort = port;
    [self connect];
}

- (void)startWithHost:(NSString *)host port:(NSUInteger)port {
    self.serverHost = host;
    self.serverPort = port;
    [self connect];
}

- (void)connect {
    if (self.connectionState == JARVISWebSocketStateConnected ||
        self.connectionState == JARVISWebSocketStateConnecting) {
        return;
    }
    
    self.connectionState = JARVISWebSocketStateConnecting;
    
    dispatch_async(self.socketQueue, ^{
        NSString *urlString = [NSString stringWithFormat:@"ws://%@:%lu/voice-unlock",
                              self.serverHost, (unsigned long)self.serverPort];
        NSURL *url = [NSURL URLWithString:urlString];
        
        os_log_info(self.logger, "Connecting to WebSocket at %@", urlString);
        
        NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
        request.timeoutInterval = 10.0;
        
        // Add headers
        [request setValue:@"JARVIS-VoiceUnlock/1.0" forHTTPHeaderField:@"User-Agent"];
        [request setValue:@"objc-daemon" forHTTPHeaderField:@"X-Client-Type"];
        
        self.webSocket = [[SRWebSocket alloc] initWithURLRequest:request];
        self.webSocket.delegate = self;
        
        [self.webSocket open];
    });
}

- (void)stop {
    os_log_info(self.logger, "Stopping WebSocket connection");
    
    [self.heartbeatTimer invalidate];
    self.heartbeatTimer = nil;
    
    [self.reconnectTimer invalidate];
    self.reconnectTimer = nil;
    
    self.connectionState = JARVISWebSocketStateDisconnecting;
    
    if (self.webSocket) {
        [self.webSocket close];
        self.webSocket = nil;
    }
    
    [self.pendingCallbacks removeAllObjects];
    self.isAudioStreaming = NO;
}

- (void)reconnect {
    if (!self.enableAutoReconnect) {
        return;
    }
    
    if (self.reconnectAttempts >= self.maxReconnectAttempts) {
        os_log_error(self.logger, "Max reconnection attempts reached");
        self.connectionState = JARVISWebSocketStateError;
        return;
    }
    
    self.reconnectAttempts++;
    
    NSTimeInterval delay = self.reconnectDelay * (1 + self.reconnectAttempts * 0.5);
    os_log_info(self.logger, "Reconnecting in %.1f seconds (attempt %lu)", 
                delay, (unsigned long)self.reconnectAttempts);
    
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)), 
                   dispatch_get_main_queue(), ^{
        [self connect];
    });
}

#pragma mark - Messaging

- (BOOL)sendMessage:(JARVISWebSocketMessage *)message {
    if (!self.isConnected) {
        os_log_error(self.logger, "Cannot send message - not connected");
        return NO;
    }
    
    if (message.type == JARVISMessageTypeAudio && message.binaryData) {
        // Send binary data
        NSData *data = [message toBinaryData];
        [self.webSocket sendData:data error:nil];
    } else {
        // Send JSON
        NSString *json = [message toJSONString];
        if (json) {
            [self.webSocket sendString:json error:nil];
        } else {
            os_log_error(self.logger, "Failed to serialize message");
            return NO;
        }
    }
    
    return YES;
}

- (BOOL)sendJSON:(NSDictionary *)json {
    JARVISWebSocketMessage *message = [JARVISWebSocketMessage messageWithType:JARVISMessageTypeCommand
                                                                       payload:json];
    return [self sendMessage:message];
}

- (BOOL)sendData:(NSData *)data {
    if (!self.isConnected) {
        return NO;
    }
    
    [self.webSocket sendData:data error:nil];
    return YES;
}

- (BOOL)sendCommand:(NSString *)command parameters:(nullable NSDictionary *)parameters {
    NSMutableDictionary *payload = [NSMutableDictionary dictionary];
    payload[@"command"] = command;
    
    if (parameters) {
        payload[@"parameters"] = parameters;
    }
    
    JARVISWebSocketMessage *message = [JARVISWebSocketMessage messageWithType:JARVISMessageTypeCommand
                                                                       payload:payload];
    return [self sendMessage:message];
}

#pragma mark - Audio Streaming

- (void)startAudioStream {
    if (self.isAudioStreaming) {
        return;
    }
    
    self.isAudioStreaming = YES;
    [self sendCommand:@"start_audio_stream" parameters:nil];
}

- (void)stopAudioStream {
    if (!self.isAudioStreaming) {
        return;
    }
    
    self.isAudioStreaming = NO;
    [self sendCommand:@"stop_audio_stream" parameters:nil];
}

- (BOOL)sendAudioData:(NSData *)audioData {
    if (!self.isAudioStreaming) {
        return NO;
    }
    
    JARVISWebSocketMessage *message = [[JARVISWebSocketMessage alloc] init];
    message.type = JARVISMessageTypeAudio;
    message.identifier = [[NSUUID UUID] UUIDString];
    message.timestamp = [NSDate date];
    message.binaryData = audioData;
    
    return [self sendMessage:message];
}

#pragma mark - Status

- (BOOL)isConnected {
    return self.connectionState == JARVISWebSocketStateConnected;
}

- (void)requestStatus {
    [self sendCommand:@"get_status" parameters:nil];
}

- (NSDictionary *)connectionInfo {
    return @{
        @"state": @(self.connectionState),
        @"connected": @(self.isConnected),
        @"reconnectAttempts": @(self.reconnectAttempts),
        @"serverHost": self.serverHost,
        @"serverPort": @(self.serverPort)
    };
}

#pragma mark - Heartbeat

- (void)startHeartbeat {
    [self.heartbeatTimer invalidate];
    
    self.heartbeatTimer = [NSTimer scheduledTimerWithTimeInterval:self.heartbeatInterval
                                                            target:self
                                                          selector:@selector(sendHeartbeat)
                                                          userInfo:nil
                                                           repeats:YES];
}

- (void)sendHeartbeat {
    JARVISWebSocketMessage *message = [JARVISWebSocketMessage messageWithType:JARVISMessageTypeHeartbeat
                                                                       payload:@{@"timestamp": @([[NSDate date] timeIntervalSince1970])}];
    [self sendMessage:message];
}

#pragma mark - SRWebSocketDelegate

- (void)webSocketDidOpen:(SRWebSocket *)webSocket {
    os_log_info(self.logger, "WebSocket connected");
    
    self.connectionState = JARVISWebSocketStateConnected;
    self.reconnectAttempts = 0;
    
    // Start heartbeat
    dispatch_async(dispatch_get_main_queue(), ^{
        [self startHeartbeat];
    });
    
    // Send initial handshake
    [self sendCommand:@"handshake" parameters:@{
        @"client": @"objc-daemon",
        @"version": @"1.0",
        @"capabilities": @[@"audio", @"authentication", @"screen_unlock"]
    }];
    
    // Notify delegate
    if ([self.delegate respondsToSelector:@selector(webSocketDidConnect)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate webSocketDidConnect];
        });
    }
    
    [self notifyConnectionStateChange];
}

- (void)webSocket:(SRWebSocket *)webSocket didFailWithError:(NSError *)error {
    os_log_error(self.logger, "WebSocket failed with error: %@", error);
    
    self.connectionState = JARVISWebSocketStateError;
    
    if ([self.delegate respondsToSelector:@selector(webSocketDidDisconnect:)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate webSocketDidDisconnect:error];
        });
    }
    
    [self notifyConnectionStateChange];
    
    // Attempt reconnection
    if (self.enableAutoReconnect) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self reconnect];
        });
    }
}

- (void)webSocket:(SRWebSocket *)webSocket didCloseWithCode:(NSInteger)code 
           reason:(nullable NSString *)reason 
         wasClean:(BOOL)wasClean {
    os_log_info(self.logger, "WebSocket closed - code: %ld, reason: %@, clean: %d",
                (long)code, reason ?: @"none", wasClean);
    
    self.connectionState = JARVISWebSocketStateDisconnected;
    
    [self.heartbeatTimer invalidate];
    self.heartbeatTimer = nil;
    
    if ([self.delegate respondsToSelector:@selector(webSocketDidDisconnect:)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate webSocketDidDisconnect:nil];
        });
    }
    
    [self notifyConnectionStateChange];
    
    // Attempt reconnection if not clean close
    if (!wasClean && self.enableAutoReconnect) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self reconnect];
        });
    }
}

- (void)webSocket:(SRWebSocket *)webSocket didReceiveMessageWithString:(NSString *)string {
    NSData *data = [string dataUsingEncoding:NSUTF8StringEncoding];
    if (!data) {
        return;
    }
    
    NSError *error = nil;
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    
    if (error || ![json isKindOfClass:[NSDictionary class]]) {
        os_log_error(self.logger, "Failed to parse WebSocket message: %@", error);
        return;
    }
    
    [self handleReceivedMessage:json];
}

- (void)webSocket:(SRWebSocket *)webSocket didReceiveMessageWithData:(NSData *)data {
    // Handle binary data (audio responses)
    if (data.length < 2) {
        return;
    }
    
    uint8_t type = ((uint8_t *)data.bytes)[0];
    
    if (type == JARVISMessageTypeAudio) {
        // Audio response
        JARVISWebSocketMessage *message = [[JARVISWebSocketMessage alloc] init];
        message.type = JARVISMessageTypeAudio;
        message.binaryData = [data subdataWithRange:NSMakeRange(1, data.length - 1)];
        message.timestamp = [NSDate date];
        
        if ([self.delegate respondsToSelector:@selector(webSocketDidReceiveMessage:)]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self.delegate webSocketDidReceiveMessage:message];
            });
        }
    }
}

- (void)handleReceivedMessage:(NSDictionary *)json {
    // Parse message
    NSString *messageId = json[@"id"];
    NSString *type = json[@"type"];
    NSDictionary *payload = json[@"payload"];
    
    // Check for callback
    if (messageId) {
        void (^callback)(NSDictionary *, NSError *) = self.pendingCallbacks[messageId];
        if (callback) {
            [self.pendingCallbacks removeObjectForKey:messageId];
            
            NSError *error = nil;
            if ([json[@"error"] isKindOfClass:[NSDictionary class]]) {
                NSDictionary *errorInfo = json[@"error"];
                error = [NSError errorWithDomain:@"JARVISWebSocket"
                                           code:[errorInfo[@"code"] integerValue]
                                       userInfo:@{NSLocalizedDescriptionKey: errorInfo[@"message"] ?: @"Unknown error"}];
            }
            
            dispatch_async(dispatch_get_main_queue(), ^{
                callback(payload, error);
            });
            
            return;
        }
    }
    
    // Create message object
    JARVISWebSocketMessage *message = [[JARVISWebSocketMessage alloc] init];
    message.identifier = messageId ?: [[NSUUID UUID] UUIDString];
    message.payload = payload;
    message.timestamp = [NSDate date];
    
    // Map type
    if ([type isEqualToString:@"command"]) {
        message.type = JARVISMessageTypeCommand;
    } else if ([type isEqualToString:@"status"]) {
        message.type = JARVISMessageTypeStatus;
    } else if ([type isEqualToString:@"authentication"]) {
        message.type = JARVISMessageTypeAuthentication;
    } else if ([type isEqualToString:@"screen_state"]) {
        message.type = JARVISMessageTypeScreenState;
    } else if ([type isEqualToString:@"configuration"]) {
        message.type = JARVISMessageTypeConfiguration;
    } else if ([type isEqualToString:@"heartbeat"]) {
        message.type = JARVISMessageTypeHeartbeat;
    }
    
    // Notify delegate
    if ([self.delegate respondsToSelector:@selector(webSocketDidReceiveMessage:)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate webSocketDidReceiveMessage:message];
        });
    }
}

- (void)notifyConnectionStateChange {
    if ([self.delegate respondsToSelector:@selector(webSocketConnectionStateChanged:)]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.delegate webSocketConnectionStateChanged:self.connectionState];
        });
    }
}

@end

#pragma mark - Python Bridge Extension

@implementation JARVISWebSocketBridge (PythonBridge)

- (void)sendPythonCommand:(NSString *)command
               arguments:(nullable NSArray *)arguments
              completion:(nullable void (^)(NSDictionary *, NSError *))completion {
    
    NSMutableDictionary *parameters = [NSMutableDictionary dictionary];
    parameters[@"function"] = command;
    
    if (arguments) {
        parameters[@"args"] = arguments;
    }
    
    JARVISWebSocketMessage *message = [JARVISWebSocketMessage messageWithType:JARVISMessageTypeCommand
                                                                       payload:parameters];
    
    if (completion) {
        self.pendingCallbacks[message.identifier] = completion;
    }
    
    [self sendMessage:message];
}

- (void)callPythonFunction:(NSString *)functionName
                parameters:(nullable NSDictionary *)parameters
                completion:(nullable void (^)(id, NSError *))completion {
    
    NSMutableDictionary *payload = [NSMutableDictionary dictionary];
    payload[@"function"] = functionName;
    payload[@"kwargs"] = parameters ?: @{};
    
    [self sendPythonCommand:@"call_function"
                  arguments:@[functionName, parameters ?: @{}]
                 completion:^(NSDictionary *response, NSError *error) {
        if (completion) {
            completion(response[@"result"], error);
        }
    }];
}

- (void)sendAuthenticationRequest:(NSData *)voiceData
                          userID:(NSString *)userID
                      completion:(void (^)(BOOL, float, NSError *))completion {
    
    // Convert voice data to base64
    NSString *audioBase64 = [voiceData base64EncodedStringWithOptions:0];
    
    NSDictionary *parameters = @{
        @"audio_data": audioBase64,
        @"user_id": userID,
        @"format": @"pcm16",
        @"sample_rate": @16000
    };
    
    [self callPythonFunction:@"authenticate_voice"
                  parameters:parameters
                  completion:^(id result, NSError *error) {
        if (error) {
            if (completion) {
                completion(NO, 0.0, error);
            }
            return;
        }
        
        NSDictionary *authResult = result;
        BOOL authenticated = [authResult[@"authenticated"] boolValue];
        float confidence = [authResult[@"confidence"] floatValue];
        
        if (completion) {
            completion(authenticated, confidence, nil);
        }
    }];
}

- (void)sendScreenStateUpdate:(BOOL)locked {
    [self sendCommand:@"screen_state_update" 
           parameters:@{@"locked": @(locked), 
                       @"timestamp": @([[NSDate date] timeIntervalSince1970])}];
}

- (void)sendUnlockAttempt:(BOOL)success method:(NSString *)method {
    [self sendCommand:@"unlock_attempt"
           parameters:@{@"success": @(success),
                       @"method": method,
                       @"timestamp": @([[NSDate date] timeIntervalSince1970])}];
}

@end