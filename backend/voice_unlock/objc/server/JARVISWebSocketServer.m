/**
 * JARVISWebSocketServer.m
 * JARVIS Voice Unlock System
 *
 * WebSocket server implementation for daemon communication
 */

#import "JARVISWebSocketServer.h"
#import "../daemon/JARVISVoiceUnlockDaemon.h"
#import <os/log.h>
#import <CommonCrypto/CommonDigest.h>

// Simple WebSocket server using NSURLSession
@interface SimpleWebSocketServer : NSObject
@property (nonatomic, strong) NSMutableArray<NSURLSessionWebSocketTask *> *clients;
@property (nonatomic, strong) NSURLSession *session;
@property (nonatomic, assign) NSUInteger port;
@property (nonatomic, strong) dispatch_queue_t serverQueue;
@property (nonatomic, copy) void (^messageHandler)(NSDictionary *message, NSURLSessionWebSocketTask *client);
@end

@implementation SimpleWebSocketServer

- (instancetype)initWithPort:(NSUInteger)port {
    self = [super init];
    if (self) {
        _port = port;
        _clients = [NSMutableArray array];
        _serverQueue = dispatch_queue_create("com.jarvis.websocket.server", DISPATCH_QUEUE_SERIAL);
        
        NSURLSessionConfiguration *config = [NSURLSessionConfiguration defaultSessionConfiguration];
        _session = [NSURLSession sessionWithConfiguration:config];
    }
    return self;
}

- (void)broadcastMessage:(NSDictionary *)message {
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:message options:0 error:&error];
    if (!error && jsonData) {
        NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        NSURLSessionWebSocketMessage *wsMessage = [[NSURLSessionWebSocketMessage alloc] initWithString:jsonString];
        
        dispatch_async(self.serverQueue, ^{
            for (NSURLSessionWebSocketTask *client in self.clients) {
                if (client.state == NSURLSessionTaskStateRunning) {
                    [client sendMessage:wsMessage completionHandler:^(NSError * _Nullable error) {
                        if (error) {
                            NSLog(@"Error sending message: %@", error);
                        }
                    }];
                }
            }
        });
    }
}

@end

// Use placeholder for PSWebSocket
@interface PSWebSocket : NSObject
@property (nonatomic, strong) NSString *identifier;
@property (nonatomic, strong) NSURLSessionWebSocketTask *task;
@end

@implementation PSWebSocket
@end

@interface PSWebSocketServer : NSObject
@property (nonatomic, strong) SimpleWebSocketServer *simpleServer;
@end

@implementation PSWebSocketServer
@end

@interface JARVISWebSocketServer ()
@property (nonatomic, strong) PSWebSocketServer *server;
@property (nonatomic, strong) SimpleWebSocketServer *simpleServer;
@property (nonatomic, readwrite) NSUInteger port;
@property (nonatomic, readwrite) BOOL isRunning;
@property (nonatomic, strong) NSMutableArray<PSWebSocket *> *clients;
@property (nonatomic, strong) os_log_t logger;
@property (nonatomic, strong) dispatch_queue_t messageQueue;
@property (nonatomic, strong) NSNetService *netService;
@end

@implementation JARVISWebSocketServer

- (instancetype)initWithPort:(NSUInteger)port {
    self = [super init];
    if (self) {
        _port = port;
        _clients = [NSMutableArray array];
        _logger = os_log_create("com.jarvis.voiceunlock", "websocket-server");
        _messageQueue = dispatch_queue_create("com.jarvis.websocket.messages", DISPATCH_QUEUE_SERIAL);
        
        // Initialize simple WebSocket server
        _simpleServer = [[SimpleWebSocketServer alloc] initWithPort:port];
        
        __weak typeof(self) weakSelf = self;
        _simpleServer.messageHandler = ^(NSDictionary *message, NSURLSessionWebSocketTask *client) {
            [weakSelf handleClientMessage:message fromTask:client];
        };
    }
    return self;
}

- (NSArray<PSWebSocket *> *)connectedClients {
    return [self.clients copy];
}

- (BOOL)startWithError:(NSError **)error {
    os_log_info(self.logger, "Starting WebSocket server on port %lu", (unsigned long)self.port);
    
    // For now, we'll use a simple HTTP server approach
    // In production, you would use a proper WebSocket library
    
    // Start listening using Bonjour for service discovery
    self.netService = [[NSNetService alloc] initWithDomain:@"local."
                                                      type:@"_jarvis-voice-unlock._tcp"
                                                      name:@"JARVIS Voice Unlock"
                                                      port:(int)self.port];
    [self.netService publish];
    
    self.isRunning = YES;
    
    // Notify delegate
    if ([self.delegate respondsToSelector:@selector(webSocketServer:didStart:)]) {
        [self.delegate webSocketServer:self didStart:self.port];
    }
    
    os_log_info(self.logger, "WebSocket server started successfully");
    
    // Start a simple TCP server using GCDAsyncSocket or similar
    // For now, we'll simulate the server being ready
    [self startSimpleHTTPServer];
    
    return YES;
}

- (void)startSimpleHTTPServer {
    // Create a simple HTTP server that upgrades to WebSocket
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        // Create socket
        int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            os_log_error(self.logger, "Failed to create socket");
            return;
        }
        
        // Allow socket reuse
        int yes = 1;
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        
        // Bind to port
        struct sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(self.port);
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
            os_log_error(self.logger, "Failed to bind to port %lu", (unsigned long)self.port);
            close(serverSocket);
            return;
        }
        
        // Listen
        if (listen(serverSocket, 5) < 0) {
            os_log_error(self.logger, "Failed to listen on socket");
            close(serverSocket);
            return;
        }
        
        os_log_info(self.logger, "HTTP server listening on port %lu", (unsigned long)self.port);
        
        // Accept connections
        while (self.isRunning) {
            struct sockaddr_in clientAddr;
            socklen_t clientLen = sizeof(clientAddr);
            int clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientLen);
            
            if (clientSocket < 0) {
                continue;
            }
            
            // Handle connection in background
            dispatch_async(self.messageQueue, ^{
                [self handleHTTPConnection:clientSocket];
            });
        }
        
        close(serverSocket);
    });
}

- (void)handleHTTPConnection:(int)clientSocket {
    char buffer[4096];
    ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    
    if (bytesRead <= 0) {
        close(clientSocket);
        return;
    }
    
    buffer[bytesRead] = '\0';
    NSString *request = [NSString stringWithUTF8String:buffer];
    
    // Check for WebSocket upgrade request
    if ([request containsString:@"Upgrade: websocket"]) {
        // Extract Sec-WebSocket-Key
        NSString *wsKey = nil;
        NSArray *lines = [request componentsSeparatedByString:@"\r\n"];
        for (NSString *line in lines) {
            if ([line hasPrefix:@"Sec-WebSocket-Key:"]) {
                wsKey = [[line substringFromIndex:18] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                break;
            }
        }
        
        // Calculate Sec-WebSocket-Accept
        NSString *acceptKey = @"dGhlIHNhbXBsZSBub25jZQ=="; // Default
        if (wsKey) {
            NSString *guid = @"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
            NSString *concat = [wsKey stringByAppendingString:guid];
            
            // SHA-1 hash
            unsigned char digest[CC_SHA1_DIGEST_LENGTH];
            CC_SHA1([concat UTF8String], (CC_LONG)[concat lengthOfBytesUsingEncoding:NSUTF8StringEncoding], digest);
            
            // Base64 encode
            NSData *digestData = [NSData dataWithBytes:digest length:CC_SHA1_DIGEST_LENGTH];
            acceptKey = [digestData base64EncodedStringWithOptions:0];
        }
        
        // Send WebSocket handshake response
        NSString *response = [NSString stringWithFormat:
                            @"HTTP/1.1 101 Switching Protocols\r\n"
                            @"Upgrade: websocket\r\n"
                            @"Connection: Upgrade\r\n"
                            @"Sec-WebSocket-Accept: %@\r\n"
                            @"\r\n", acceptKey];
        
        send(clientSocket, [response UTF8String], [response length], 0);
        
        // Create client wrapper
        PSWebSocket *client = [[PSWebSocket alloc] init];
        client.identifier = [[NSUUID UUID] UUIDString];
        
        [self.clients addObject:client];
        
        // Notify delegate
        if ([self.delegate respondsToSelector:@selector(webSocketServer:didConnectClient:)]) {
            [self.delegate webSocketServer:self didConnectClient:client];
        }
        
        // Send initial status
        [self sendStatusToClient:client];
        
        // Continue handling WebSocket frames
        [self handleWebSocketClient:clientSocket client:client];
    } else {
        // Regular HTTP request - send simple response
        NSString *response = @"HTTP/1.1 200 OK\r\n"
                            @"Content-Type: application/json\r\n"
                            @"\r\n"
                            @"{\"status\":\"JARVIS Voice Unlock WebSocket Server\",\"port\":8765}\r\n";
        send(clientSocket, [response UTF8String], [response length], 0);
        close(clientSocket);
    }
}

- (void)handleWebSocketClient:(int)clientSocket client:(PSWebSocket *)client {
    // Simple WebSocket frame handling
    while (self.isRunning) {
        unsigned char frame[4096];
        ssize_t bytesRead = recv(clientSocket, frame, sizeof(frame), 0);
        
        if (bytesRead <= 0) {
            break;
        }
        
        // Parse WebSocket frame (simplified)
        if (bytesRead > 2) {
            BOOL fin = (frame[0] & 0x80) != 0;
            int opcode = frame[0] & 0x0F;
            BOOL masked = (frame[1] & 0x80) != 0;
            uint64_t payloadLen = frame[1] & 0x7F;
            
            int offset = 2;
            if (payloadLen == 126) {
                payloadLen = (frame[2] << 8) | frame[3];
                offset = 4;
            } else if (payloadLen == 127) {
                // Skip extended length for simplicity
                continue;
            }
            
            unsigned char mask[4];
            if (masked) {
                memcpy(mask, &frame[offset], 4);
                offset += 4;
            }
            
            // Extract payload
            NSMutableData *payload = [NSMutableData dataWithLength:payloadLen];
            unsigned char *payloadBytes = (unsigned char *)[payload mutableBytes];
            
            for (uint64_t i = 0; i < payloadLen && offset + i < bytesRead; i++) {
                if (masked) {
                    payloadBytes[i] = frame[offset + i] ^ mask[i % 4];
                } else {
                    payloadBytes[i] = frame[offset + i];
                }
            }
            
            if (opcode == 1) { // Text frame
                NSString *text = [[NSString alloc] initWithData:payload encoding:NSUTF8StringEncoding];
                if (text) {
                    // Parse JSON
                    NSError *error = nil;
                    NSDictionary *message = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:&error];
                    
                    if (message && !error) {
                        [self handleClientMessage:message fromClient:client socket:clientSocket];
                    }
                }
            } else if (opcode == 8) { // Close frame
                break;
            }
        }
    }
    
    // Client disconnected
    [self.clients removeObject:client];
    close(clientSocket);
    
    if ([self.delegate respondsToSelector:@selector(webSocketServer:didDisconnectClient:)]) {
        [self.delegate webSocketServer:self didDisconnectClient:client];
    }
}

- (void)handleClientMessage:(NSDictionary *)message fromTask:(NSURLSessionWebSocketTask *)task {
    // Handle message from simplified server
    PSWebSocket *client = [[PSWebSocket alloc] init];
    client.task = task;
    [self handleClientMessage:message fromClient:client socket:0];
}

- (void)handleClientMessage:(NSDictionary *)message fromClient:(PSWebSocket *)client socket:(int)socket {
    os_log_info(self.logger, "Received message: %@", message);
    
    NSString *type = message[@"type"];
    NSString *command = message[@"command"];
    NSDictionary *parameters = message[@"parameters"];
    
    if ([type isEqualToString:@"command"]) {
        [self handleCommand:command parameters:parameters completion:^(NSDictionary *response) {
            if (socket > 0) {
                [self sendWebSocketMessage:response toSocket:socket];
            } else {
                [self sendMessage:response toClient:client];
            }
        }];
    } else {
        // Notify delegate
        if ([self.delegate respondsToSelector:@selector(webSocketServer:didReceiveMessage:fromClient:)]) {
            [self.delegate webSocketServer:self didReceiveMessage:message fromClient:client];
        }
    }
}

- (void)sendWebSocketMessage:(NSDictionary *)message toSocket:(int)socket {
    NSError *error = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:message options:0 error:&error];
    
    if (!error && jsonData) {
        NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        NSData *textData = [jsonString dataUsingEncoding:NSUTF8StringEncoding];
        
        // Create WebSocket text frame
        unsigned char frame[4096];
        int frameLen = 0;
        
        // FIN = 1, opcode = 1 (text)
        frame[0] = 0x81;
        
        NSUInteger dataLen = [textData length];
        if (dataLen <= 125) {
            frame[1] = (unsigned char)dataLen;
            frameLen = 2;
        } else if (dataLen <= 65535) {
            frame[1] = 126;
            frame[2] = (dataLen >> 8) & 0xFF;
            frame[3] = dataLen & 0xFF;
            frameLen = 4;
        }
        
        // Copy payload
        [textData getBytes:&frame[frameLen] length:dataLen];
        frameLen += dataLen;
        
        // Send frame
        send(socket, frame, frameLen, 0);
    }
}

- (void)stop {
    self.isRunning = NO;
    [self.netService stop];
    [self.clients removeAllObjects];
    
    os_log_info(self.logger, "WebSocket server stopped");
}

- (void)sendMessage:(NSDictionary *)message toClient:(PSWebSocket *)client {
    if (client.task) {
        NSError *error = nil;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:message options:0 error:&error];
        if (!error && jsonData) {
            NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
            NSURLSessionWebSocketMessage *wsMessage = [[NSURLSessionWebSocketMessage alloc] initWithString:jsonString];
            [client.task sendMessage:wsMessage completionHandler:nil];
        }
    }
}

- (void)broadcastMessage:(NSDictionary *)message {
    for (PSWebSocket *client in self.clients) {
        [self sendMessage:message toClient:client];
    }
    
    // Also broadcast via simple server
    [self.simpleServer broadcastMessage:message];
}

- (void)sendStatusToClient:(PSWebSocket *)client {
    JARVISVoiceUnlockDaemon *daemon = [JARVISVoiceUnlockDaemon sharedDaemon];
    NSDictionary *status = [daemon getStatus];
    
    NSDictionary *response = @{
        @"type": @"status",
        @"status": status,
        @"timestamp": @([[NSDate date] timeIntervalSince1970])
    };
    
    [self sendMessage:response toClient:client];
}

- (void)handleCommand:(NSString *)command parameters:(NSDictionary *)parameters completion:(void (^)(NSDictionary *))completion {
    JARVISVoiceUnlockDaemon *daemon = [JARVISVoiceUnlockDaemon sharedDaemon];
    
    dispatch_async(self.messageQueue, ^{
        NSDictionary *response = nil;
        
        if ([command isEqualToString:@"handshake"]) {
            response = @{
                @"type": @"handshake",
                @"success": @YES,
                @"version": @"1.0",
                @"daemon": @"JARVIS Voice Unlock"
            };
        }
        else if ([command isEqualToString:@"get_status"]) {
            NSDictionary *status = [daemon getStatus];
            response = @{
                @"type": @"status",
                @"status": status,
                @"success": @YES
            };
        }
        else if ([command isEqualToString:@"start_monitoring"]) {
            NSError *error = nil;
            BOOL success = [daemon startMonitoringWithError:&error];
            
            response = @{
                @"type": @"command_response",
                @"command": command,
                @"success": @(success),
                @"message": success ? @"Monitoring started" : @"Failed to start monitoring",
                @"error": error ? error.localizedDescription : [NSNull null]
            };
        }
        else if ([command isEqualToString:@"stop_monitoring"]) {
            [daemon stopMonitoring];
            response = @{
                @"type": @"command_response", 
                @"command": command,
                @"success": @YES,
                @"message": @"Monitoring stopped"
            };
        }
        else {
            response = @{
                @"type": @"error",
                @"message": [NSString stringWithFormat:@"Unknown command: %@", command],
                @"success": @NO
            };
        }
        
        if (completion) {
            completion(response);
        }
    });
}

@end