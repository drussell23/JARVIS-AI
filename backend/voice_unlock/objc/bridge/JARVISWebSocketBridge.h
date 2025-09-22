/**
 * JARVISWebSocketBridge.h
 * JARVIS Voice Unlock System
 *
 * WebSocket bridge for communication between Objective-C daemon
 * and Python backend services.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// WebSocket connection state
typedef NS_ENUM(NSInteger, JARVISWebSocketState) {
    JARVISWebSocketStateDisconnected = 0,
    JARVISWebSocketStateConnecting,
    JARVISWebSocketStateConnected,
    JARVISWebSocketStateDisconnecting,
    JARVISWebSocketStateError
};

// Message types
typedef NS_ENUM(NSInteger, JARVISMessageType) {
    JARVISMessageTypeCommand = 0,
    JARVISMessageTypeStatus,
    JARVISMessageTypeAudio,
    JARVISMessageTypeAuthentication,
    JARVISMessageTypeScreenState,
    JARVISMessageTypeConfiguration,
    JARVISMessageTypeHeartbeat
};

// WebSocket message
@interface JARVISWebSocketMessage : NSObject
@property (nonatomic, assign) JARVISMessageType type;
@property (nonatomic, strong) NSString *identifier;
@property (nonatomic, strong) NSDictionary *payload;
@property (nonatomic, strong) NSDate *timestamp;
@property (nonatomic, nullable) NSData *binaryData;

+ (instancetype)messageWithType:(JARVISMessageType)type payload:(NSDictionary *)payload;
- (NSString *)toJSONString;
- (NSData *)toBinaryData;
@end

// WebSocket delegate
@protocol JARVISWebSocketBridgeDelegate <NSObject>
@optional
- (void)webSocketDidConnect;
- (void)webSocketDidDisconnect:(NSError * _Nullable)error;
- (void)webSocketDidReceiveMessage:(JARVISWebSocketMessage *)message;
- (void)webSocketConnectionStateChanged:(JARVISWebSocketState)state;
@end

// Main WebSocket bridge interface
@interface JARVISWebSocketBridge : NSObject

@property (nonatomic, weak, nullable) id<JARVISWebSocketBridgeDelegate> delegate;
@property (nonatomic, readonly) JARVISWebSocketState connectionState;
@property (nonatomic, readonly) BOOL isConnected;
@property (nonatomic, readonly) NSUInteger reconnectAttempts;

// Configuration
@property (nonatomic, strong) NSString *serverHost;
@property (nonatomic, assign) NSUInteger serverPort;
@property (nonatomic, assign) BOOL enableAutoReconnect;
@property (nonatomic, assign) NSTimeInterval reconnectDelay;
@property (nonatomic, assign) NSUInteger maxReconnectAttempts;
@property (nonatomic, assign) NSTimeInterval heartbeatInterval;

// Connection management
- (void)startWithPort:(NSUInteger)port;
- (void)startWithHost:(NSString *)host port:(NSUInteger)port;
- (void)stop;
- (void)reconnect;

// Messaging
- (BOOL)sendMessage:(JARVISWebSocketMessage *)message;
- (BOOL)sendJSON:(NSDictionary *)json;
- (BOOL)sendData:(NSData *)data;
- (BOOL)sendCommand:(NSString *)command parameters:(nullable NSDictionary *)parameters;

// Audio streaming
- (void)startAudioStream;
- (void)stopAudioStream;
- (BOOL)sendAudioData:(NSData *)audioData;

// Status
- (void)requestStatus;
- (NSDictionary *)connectionInfo;

@end

// Python bridge extension
@interface JARVISWebSocketBridge (PythonBridge)

// High-level Python communication
- (void)sendPythonCommand:(NSString *)command 
               arguments:(nullable NSArray *)arguments
              completion:(nullable void (^)(NSDictionary *response, NSError *error))completion;

- (void)callPythonFunction:(NSString *)functionName
                parameters:(nullable NSDictionary *)parameters
                completion:(nullable void (^)(id result, NSError *error))completion;

// Voice unlock specific
- (void)sendAuthenticationRequest:(NSData *)voiceData
                          userID:(NSString *)userID
                      completion:(void (^)(BOOL authenticated, float confidence, NSError *error))completion;

- (void)sendScreenStateUpdate:(BOOL)locked;
- (void)sendUnlockAttempt:(BOOL)success method:(NSString *)method;

@end

NS_ASSUME_NONNULL_END