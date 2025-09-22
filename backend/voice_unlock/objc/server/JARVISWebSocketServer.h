/**
 * JARVISWebSocketServer.h
 * JARVIS Voice Unlock System
 *
 * WebSocket server for daemon communication with JARVIS API
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class JARVISWebSocketServer;
@class PSWebSocketServer;
@class PSWebSocket;

@protocol JARVISWebSocketServerDelegate <NSObject>
@optional
- (void)webSocketServer:(JARVISWebSocketServer *)server didStart:(NSUInteger)port;
- (void)webSocketServer:(JARVISWebSocketServer *)server didReceiveMessage:(NSDictionary *)message fromClient:(PSWebSocket *)client;
- (void)webSocketServer:(JARVISWebSocketServer *)server didConnectClient:(PSWebSocket *)client;
- (void)webSocketServer:(JARVISWebSocketServer *)server didDisconnectClient:(PSWebSocket *)client;
- (void)webSocketServer:(JARVISWebSocketServer *)server didFailWithError:(NSError *)error;
@end

@interface JARVISWebSocketServer : NSObject

@property (nonatomic, weak) id<JARVISWebSocketServerDelegate> delegate;
@property (nonatomic, readonly) NSUInteger port;
@property (nonatomic, readonly) BOOL isRunning;
@property (nonatomic, readonly) NSArray<PSWebSocket *> *connectedClients;

- (instancetype)initWithPort:(NSUInteger)port;
- (BOOL)startWithError:(NSError **)error;
- (void)stop;

// Send messages to clients
- (void)sendMessage:(NSDictionary *)message toClient:(PSWebSocket *)client;
- (void)broadcastMessage:(NSDictionary *)message;

// Handle daemon commands
- (void)handleCommand:(NSString *)command parameters:(NSDictionary *)parameters completion:(void (^)(NSDictionary *response))completion;

@end

NS_ASSUME_NONNULL_END