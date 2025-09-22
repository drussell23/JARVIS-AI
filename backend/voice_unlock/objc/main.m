/**
 * main.m
 * JARVIS Voice Unlock System
 *
 * Main entry point for the Voice Unlock daemon
 */

#import <Foundation/Foundation.h>
#import <signal.h>
#import "JARVISVoiceUnlockDaemon.h"

// Global daemon instance for signal handling
static JARVISVoiceUnlockDaemon *g_daemon = nil;

// Signal handler
void signalHandler(int signal) {
    NSLog(@"Received signal %d", signal);
    
    if (g_daemon) {
        [g_daemon stopMonitoring];
    }
    
    exit(0);
}

// Usage information
void printUsage(const char *programName) {
    printf("JARVIS Voice Unlock Daemon\n");
    printf("=========================\n\n");
    printf("Usage: %s [options]\n", programName);
    printf("\nOptions:\n");
    printf("  -h, --help          Show this help message\n");
    printf("  -v, --version       Show version information\n");
    printf("  -d, --debug         Enable debug logging\n");
    printf("  -c, --config PATH   Specify configuration file path\n");
    printf("  -t, --test          Run in test mode (foreground)\n");
    printf("  -s, --status        Check daemon status\n");
    printf("\n");
}

// Version information
void printVersion() {
    printf("JARVIS Voice Unlock Daemon v1.0.0\n");
    printf("Copyright (c) 2024 JARVIS AI\n");
    printf("Built with Objective-C for macOS\n");
}

// Check if already running
BOOL isDaemonRunning() {
    // Check if another instance is running by looking for lock file
    NSString *lockFile = [@"~/.jarvis/voice_unlock/daemon.lock" stringByExpandingTildeInPath];
    NSFileManager *fm = [NSFileManager defaultManager];
    
    if ([fm fileExistsAtPath:lockFile]) {
        // Read PID from lock file
        NSString *pidString = [NSString stringWithContentsOfFile:lockFile 
                                                       encoding:NSUTF8StringEncoding 
                                                          error:nil];
        if (pidString) {
            int pid = [pidString intValue];
            // Check if process is still running
            if (kill(pid, 0) == 0) {
                return YES;
            }
        }
        // Remove stale lock file
        [fm removeItemAtPath:lockFile error:nil];
    }
    
    return NO;
}

// Create lock file
void createLockFile() {
    NSString *lockFile = [@"~/.jarvis/voice_unlock/daemon.lock" stringByExpandingTildeInPath];
    NSString *lockDir = [lockFile stringByDeletingLastPathComponent];
    
    // Create directory if needed
    [[NSFileManager defaultManager] createDirectoryAtPath:lockDir 
                              withIntermediateDirectories:YES 
                                               attributes:nil 
                                                    error:nil];
    
    // Write PID to lock file
    NSString *pidString = [NSString stringWithFormat:@"%d", getpid()];
    [pidString writeToFile:lockFile 
                atomically:YES 
                  encoding:NSUTF8StringEncoding 
                     error:nil];
}

// Remove lock file
void removeLockFile() {
    NSString *lockFile = [@"~/.jarvis/voice_unlock/daemon.lock" stringByExpandingTildeInPath];
    [[NSFileManager defaultManager] removeItemAtPath:lockFile error:nil];
}

// Main function
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Parse command line arguments
        BOOL debugMode = NO;
        BOOL testMode = NO;
        BOOL showStatus = NO;
        NSString *configPath = nil;
        
        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            
            if ([arg isEqualToString:@"-h"] || [arg isEqualToString:@"--help"]) {
                printUsage(argv[0]);
                return 0;
            }
            else if ([arg isEqualToString:@"-v"] || [arg isEqualToString:@"--version"]) {
                printVersion();
                return 0;
            }
            else if ([arg isEqualToString:@"-d"] || [arg isEqualToString:@"--debug"]) {
                debugMode = YES;
            }
            else if ([arg isEqualToString:@"-t"] || [arg isEqualToString:@"--test"]) {
                testMode = YES;
            }
            else if ([arg isEqualToString:@"-s"] || [arg isEqualToString:@"--status"]) {
                showStatus = YES;
            }
            else if ([arg isEqualToString:@"-c"] || [arg isEqualToString:@"--config"]) {
                if (i + 1 < argc) {
                    configPath = [NSString stringWithUTF8String:argv[++i]];
                }
            }
        }
        
        // Check status
        if (showStatus) {
            if (isDaemonRunning()) {
                printf("JARVIS Voice Unlock Daemon is running\n");
                return 0;
            } else {
                printf("JARVIS Voice Unlock Daemon is not running\n");
                return 1;
            }
        }
        
        // Check if already running
        if (isDaemonRunning() && !testMode) {
            NSLog(@"JARVIS Voice Unlock Daemon is already running");
            return 1;
        }
        
        // Create lock file
        if (!testMode) {
            createLockFile();
        }
        
        // Setup signal handlers
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // Create and configure daemon
        g_daemon = [JARVISVoiceUnlockDaemon sharedDaemon];
        
        // Set debug mode
        if (debugMode) {
            g_daemon.options |= JARVISVoiceUnlockOptionEnableDebugLogging;
        }
        
        // Load configuration
        if (configPath) {
            [g_daemon loadConfigurationFromFile:configPath];
        }
        
        // Start daemon
        NSLog(@"Starting JARVIS Voice Unlock Daemon...");
        
        NSError *error = nil;
        if (![g_daemon startMonitoringWithError:&error]) {
            NSLog(@"Failed to start daemon: %@", error);
            removeLockFile();
            return 1;
        }
        
        NSLog(@"JARVIS Voice Unlock Daemon started successfully");
        
        // Get initial status
        NSDictionary *status = [g_daemon getStatus];
        NSLog(@"Daemon status: %@", status);
        
        // Run the run loop
        if (testMode) {
            NSLog(@"Running in test mode (foreground)");
            
            // In test mode, run for a limited time
            NSDate *endDate = [NSDate dateWithTimeIntervalSinceNow:300]; // 5 minutes
            while ([g_daemon isMonitoring] && [[NSDate date] compare:endDate] == NSOrderedAscending) {
                [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                                          beforeDate:[NSDate dateWithTimeIntervalSinceNow:1.0]];
            }
        } else {
            // In daemon mode, run indefinitely
            [[NSRunLoop currentRunLoop] run];
        }
        
        // Cleanup
        [g_daemon stopMonitoring];
        removeLockFile();
        
        NSLog(@"JARVIS Voice Unlock Daemon stopped");
    }
    
    return 0;
}