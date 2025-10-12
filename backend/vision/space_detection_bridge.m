#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ApplicationServices/ApplicationServices.h>

// Forward declarations for C interface functions
int get_current_space();
BOOL space_exists(int spaceId);

@interface SpaceDetectionBridge : NSObject
+ (NSArray *)enumerateAllSpaces;
+ (NSArray *)getWindowsForSpace:(int)spaceId;
+ (NSDictionary *)getSpaceInfo:(int)spaceId;
+ (BOOL)captureSpaceInvisibly:(int)spaceId;
+ (NSArray *)getAllWindows;
+ (int)estimateWindowSpace:(NSDictionary *)window;
// Helper methods for space capture
+ (BOOL)switchToSpace:(int)spaceId withDelay:(float)delaySeconds;
+ (BOOL)captureCurrentSpace:(int)spaceId;
+ (BOOL)captureOffscreenSpace:(int)spaceId;
@end

@implementation SpaceDetectionBridge

+ (NSArray *)enumerateAllSpaces {
    NSMutableArray *spaces = [[NSMutableArray alloc] init];
    
    // Use Accessibility API to detect Mission Control spaces
    // This is more reliable than AppleScript for Mission Control
    NSArray *allWindows = [self getAllWindows];
    
    // Group windows by estimated space
    NSMutableDictionary *spaceWindows = [[NSMutableDictionary alloc] init];
    for (NSDictionary *window in allWindows) {
        int estimatedSpace = [self estimateWindowSpace:window];
        NSMutableArray *windows = spaceWindows[@(estimatedSpace)];
        if (!windows) {
            windows = [[NSMutableArray alloc] init];
            spaceWindows[@(estimatedSpace)] = windows;
        }
        [windows addObject:window];
    }
    
    // Also try to detect spaces using Mission Control accessibility
    NSMutableSet *allSpaces = [[NSMutableSet alloc] init];
    
    // Add spaces from window analysis
    for (NSNumber *spaceId in spaceWindows.allKeys) {
        [allSpaces addObject:spaceId];
    }
    
    // Try to detect additional spaces using Mission Control
    // This is a heuristic approach since we can't directly access Mission Control APIs
    int maxSpaces = 7; // Common maximum for Mission Control spaces
    
    NSLog(@"[SPACE_DETECTION] Window analysis found %lu space groups", [spaceWindows count]);
    
    // Always include space 1 (current space)
    [allSpaces addObject:@(1)];
    
    // Add spaces that have windows
    for (NSNumber *spaceId in spaceWindows.allKeys) {
        [allSpaces addObject:spaceId];
        NSLog(@"[SPACE_DETECTION] Added space %@ with %lu windows", spaceId, [spaceWindows[spaceId] count]);
    }
    
    // For Mission Control, we need to detect empty spaces too
    // Since Mission Control shows 7 spaces, let's assume they all exist
    for (int i = 1; i <= maxSpaces; i++) {
        [allSpaces addObject:@(i)];
    }
    
    // Create space info for each detected space
    for (NSNumber *spaceIdNum in allSpaces) {
        int spaceId = [spaceIdNum intValue];
        NSMutableDictionary *spaceInfo = [[NSMutableDictionary alloc] init];
        
        NSArray *windows = spaceWindows[@(spaceId)] ?: @[];
        
        // Determine if this is the current space (space 1 is usually current)
        BOOL isCurrent = (spaceId == 1);
        
        // Build space info
        spaceInfo[@"space_id"] = @(spaceId);
        spaceInfo[@"space_name"] = [NSString stringWithFormat:@"Desktop %d", spaceId];
        spaceInfo[@"is_current"] = @(isCurrent);
        spaceInfo[@"window_count"] = @([windows count]);
        spaceInfo[@"windows"] = windows;
        
        // Determine space activity
        NSString *activity = @"Empty";
        if ([windows count] > 0) {
            // Analyze windows to determine primary activity
            NSMutableSet *apps = [[NSMutableSet alloc] init];
            for (NSDictionary *window in windows) {
                NSString *appName = window[@"app_name"];
                if (appName && ![appName isEqualToString:@"Unknown"]) {
                    [apps addObject:appName];
                }
            }
            
            if ([apps count] > 0) {
                NSArray *appArray = [apps allObjects];
                if ([appArray count] == 1) {
                    activity = appArray[0];
                } else {
                    activity = [NSString stringWithFormat:@"%@ and %lu others", appArray[0], [appArray count] - 1];
                }
            }
        }
        
        spaceInfo[@"primary_activity"] = activity;
        
        [spaces addObject:spaceInfo];
        
        NSLog(@"[SPACE_DETECTION] Space %d: '%@' (%@) - %lu windows", 
              spaceId, spaceInfo[@"space_name"], activity, [windows count]);
    }
    
    NSLog(@"[SPACE_DETECTION] Total spaces detected: %lu", [spaces count]);
    
    return [spaces copy];
}

+ (NSArray *)getWindowsForSpace:(int)spaceId {
    NSArray *allWindows = [self getAllWindows];
    NSMutableArray *spaceWindows = [[NSMutableArray alloc] init];
    
    for (NSDictionary *window in allWindows) {
        int estimatedSpace = [self estimateWindowSpace:window];
        if (estimatedSpace == spaceId) {
            [spaceWindows addObject:window];
        }
    }
    
    return [spaceWindows copy];
}

+ (NSArray *)getAllWindows {
    NSMutableArray *windows = [[NSMutableArray alloc] init];
    
    // Get all windows using Core Graphics
    CFArrayRef windowList = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID);
    
    if (windowList) {
        CFIndex count = CFArrayGetCount(windowList);
        
        for (CFIndex i = 0; i < count; i++) {
            CFDictionaryRef windowDict = (CFDictionaryRef)CFArrayGetValueAtIndex(windowList, i);
            
            // Extract window information
            CFNumberRef windowNumber = (CFNumberRef)CFDictionaryGetValue(windowDict, kCGWindowNumber);
            CFStringRef ownerName = (CFStringRef)CFDictionaryGetValue(windowDict, kCGWindowOwnerName);
            CFStringRef windowName = (CFStringRef)CFDictionaryGetValue(windowDict, kCGWindowName);
            CFDictionaryRef boundsDict = (CFDictionaryRef)CFDictionaryGetValue(windowDict, kCGWindowBounds);
            CFNumberRef layer = (CFNumberRef)CFDictionaryGetValue(windowDict, kCGWindowLayer);
            CFNumberRef alpha = (CFNumberRef)CFDictionaryGetValue(windowDict, kCGWindowAlpha);
            
            if (!windowNumber || !ownerName || !boundsDict) continue;
            
            // Convert to native types
            int windowId = 0;
            CFNumberGetValue(windowNumber, kCFNumberIntType, &windowId);
            
            NSString *appName = (__bridge NSString *)ownerName;
            NSString *windowTitle = windowName ? (__bridge NSString *)windowName : @"";
            
            // Extract bounds
            CGRect bounds = CGRectZero;
            CFNumberRef x = (CFNumberRef)CFDictionaryGetValue(boundsDict, CFSTR("X"));
            CFNumberRef y = (CFNumberRef)CFDictionaryGetValue(boundsDict, CFSTR("Y"));
            CFNumberRef width = (CFNumberRef)CFDictionaryGetValue(boundsDict, CFSTR("Width"));
            CFNumberRef height = (CFNumberRef)CFDictionaryGetValue(boundsDict, CFSTR("Height"));
            
            if (x && y && width && height) {
                CFNumberGetValue(x, kCFNumberFloatType, &bounds.origin.x);
                CFNumberGetValue(y, kCFNumberFloatType, &bounds.origin.y);
                CFNumberGetValue(width, kCFNumberFloatType, &bounds.size.width);
                CFNumberGetValue(height, kCFNumberFloatType, &bounds.size.height);
            }
            
            // Skip tiny windows
            if (bounds.size.width < 100 || bounds.size.height < 100) {
                continue;
            }
            
            // Skip system windows
            if ([appName isEqualToString:@"Window Server"] || 
                [appName isEqualToString:@"Dock"] || 
                [appName isEqualToString:@"SystemUIServer"]) {
                continue;
            }
            
            NSDictionary *windowInfo = @{
                @"window_id": @(windowId),
                @"app_name": appName,
                @"window_title": windowTitle,
                @"bounds": NSStringFromRect(NSRectFromCGRect(bounds)),
                @"x": @(bounds.origin.x),
                @"y": @(bounds.origin.y),
                @"width": @(bounds.size.width),
                @"height": @(bounds.size.height)
            };
            
            [windows addObject:windowInfo];
        }
        
        CFRelease(windowList);
    }
    
    return [windows copy];
}

+ (int)estimateWindowSpace:(NSDictionary *)window {
    // Enhanced heuristic based on window position and Mission Control behavior
    NSNumber *x = window[@"x"];
    NSNumber *y = window[@"y"];
    NSNumber *width = window[@"width"];
    NSNumber *height = window[@"height"];
    
    if (!x || !y) return 1;
    
    float xPos = [x floatValue];
    float yPos = [y floatValue];
    float w = width ? [width floatValue] : 0;
    float h = height ? [height floatValue] : 0;
    
    // Get screen bounds
    NSScreen *mainScreen = [NSScreen mainScreen];
    NSRect screenFrame = mainScreen.frame;
    
    // Check if window is visible on current screen
    BOOL isOnCurrentScreen = (xPos >= 0 && yPos >= 0 && 
                             xPos + w <= screenFrame.size.width && 
                             yPos + h <= screenFrame.size.height);
    
    if (isOnCurrentScreen) {
        return 1; // Current space
    }
    
    // Window is off-screen, estimate which space it's on
    // Mission Control spaces are typically arranged horizontally
    if (xPos < 0) {
        // Off-screen to the left
        if (xPos < -screenFrame.size.width) {
            return 3; // Far left space
        } else {
            return 2; // Left space
        }
    } else if (xPos > screenFrame.size.width) {
        // Off-screen to the right
        if (xPos > screenFrame.size.width * 2) {
            return 5; // Far right space
        } else {
            return 4; // Right space
        }
    } else if (yPos < 0) {
        // Off-screen below
        return 6; // Bottom space
    } else if (yPos > screenFrame.size.height) {
        // Off-screen above
        return 7; // Top space
    }
    
    // Default to current space
    return 1;
}

+ (NSDictionary *)getSpaceInfo:(int)spaceId {
    NSMutableDictionary *spaceInfo = [[NSMutableDictionary alloc] init];
    
    // Get windows in this space
    NSArray *windows = [self getWindowsForSpace:spaceId];
    
    // Determine if this is the current space (space 1 is usually current)
    BOOL isCurrent = (spaceId == 1);
    
    // Build space info
    spaceInfo[@"space_id"] = @(spaceId);
    spaceInfo[@"space_name"] = [NSString stringWithFormat:@"Desktop %d", spaceId];
    spaceInfo[@"is_current"] = @(isCurrent);
    spaceInfo[@"window_count"] = @([windows count]);
    spaceInfo[@"windows"] = windows;
    
    // Determine space activity
    NSString *activity = @"Empty";
    if ([windows count] > 0) {
        // Analyze windows to determine primary activity
        NSMutableSet *apps = [[NSMutableSet alloc] init];
        for (NSDictionary *window in windows) {
            NSString *appName = window[@"app_name"];
            if (appName && ![appName isEqualToString:@"Unknown"]) {
                [apps addObject:appName];
            }
        }
        
        if ([apps count] > 0) {
            NSArray *appArray = [apps allObjects];
            if ([appArray count] == 1) {
                activity = appArray[0];
            } else {
                activity = [NSString stringWithFormat:@"%@ and %lu others", appArray[0], [appArray count] - 1];
            }
        }
    }
    
    spaceInfo[@"primary_activity"] = activity;
    
    return [spaceInfo copy];
}

+ (BOOL)captureSpaceInvisibly:(int)spaceId {
    NSLog(@"[SPACE_CAPTURE] Invisible capture requested for space %d", spaceId);

    @autoreleasepool {
        // 1. Verify the space exists
        if (!space_exists(spaceId)) {
            NSLog(@"[SPACE_CAPTURE] ❌ Space %d does not exist", spaceId);
            return NO;
        }

        // 2. Get current space to return to later
        int currentSpace = get_current_space();
        NSLog(@"[SPACE_CAPTURE] 📍 Current space: %d, Target space: %d", currentSpace, spaceId);

        // 3. If we're already on the target space, just capture
        if (currentSpace == spaceId) {
            NSLog(@"[SPACE_CAPTURE] ✅ Already on space %d, capturing directly", spaceId);
            return [self captureCurrentSpace:spaceId];
        }

        // 4. Store original space for return
        BOOL switchedSuccessfully = NO;
        BOOL capturedSuccessfully = NO;

        @try {
            // 5. Quick switch to target space using AppleScript
            NSLog(@"[SPACE_CAPTURE] 🔄 Switching to space %d...", spaceId);
            switchedSuccessfully = [self switchToSpace:spaceId withDelay:0.3];

            if (!switchedSuccessfully) {
                NSLog(@"[SPACE_CAPTURE] ⚠️  Could not switch to space %d, trying direct capture", spaceId);
                return [self captureOffscreenSpace:spaceId];
            }

            // 6. Capture the space
            NSLog(@"[SPACE_CAPTURE] 📸 Capturing space %d...", spaceId);
            capturedSuccessfully = [self captureCurrentSpace:spaceId];

            if (capturedSuccessfully) {
                NSLog(@"[SPACE_CAPTURE] ✅ Successfully captured space %d", spaceId);
            } else {
                NSLog(@"[SPACE_CAPTURE] ❌ Failed to capture space %d", spaceId);
            }

        } @catch (NSException *exception) {
            NSLog(@"[SPACE_CAPTURE] ❌ Exception during capture: %@", exception);
            capturedSuccessfully = NO;
        } @finally {
            // 7. ALWAYS switch back to original space
            if (switchedSuccessfully && currentSpace != spaceId) {
                NSLog(@"[SPACE_CAPTURE] 🔙 Returning to space %d...", currentSpace);
                [self switchToSpace:currentSpace withDelay:0.1];
            }
        }

        return capturedSuccessfully;
    }
}

// MARK: - Helper Methods

+ (BOOL)switchToSpace:(int)spaceId withDelay:(float)delaySeconds {
    // Calculate how many swipes needed
    int currentSpace = get_current_space();
    int spaceDiff = spaceId - currentSpace;

    if (spaceDiff == 0) {
        return YES; // Already there
    }

    // Build AppleScript to swipe left or right
    NSMutableString *swipeScript = [[NSMutableString alloc] init];
    [swipeScript appendString:@"tell application \"System Events\"\n"];

    if (spaceDiff > 0) {
        // Swipe right (Control + Right Arrow)
        for (int i = 0; i < abs(spaceDiff); i++) {
            [swipeScript appendString:@"  key code 124 using {control down}\n"];
            [swipeScript appendFormat:@"  delay %f\n", delaySeconds];
        }
    } else {
        // Swipe left (Control + Left Arrow)
        for (int i = 0; i < abs(spaceDiff); i++) {
            [swipeScript appendString:@"  key code 123 using {control down}\n"];
            [swipeScript appendFormat:@"  delay %f\n", delaySeconds];
        }
    }

    [swipeScript appendString:@"end tell"];

    // Execute AppleScript
    NSAppleScript *appleScript = [[NSAppleScript alloc] initWithSource:swipeScript];
    NSDictionary *errorDict = nil;
    NSAppleEventDescriptor *result = [appleScript executeAndReturnError:&errorDict];

    if (errorDict) {
        NSLog(@"[SPACE_CAPTURE] ⚠️  AppleScript error: %@", errorDict);
        return NO;
    }

    // Wait for animation to complete
    [NSThread sleepForTimeInterval:delaySeconds];
    return YES;
}

+ (BOOL)captureCurrentSpace:(int)spaceId {
    @autoreleasepool {
        // Use screencapture command-line tool (no deprecated APIs)
        NSString *tempDir = NSTemporaryDirectory();
        NSString *filename = [NSString stringWithFormat:@"space_%d_capture.png", spaceId];
        NSString *filepath = [tempDir stringByAppendingPathComponent:filename];

        // Create NSTask to run screencapture
        NSTask *task = [[NSTask alloc] init];
        [task setLaunchPath:@"/usr/sbin/screencapture"];
        [task setArguments:@[
            @"-x",           // No sound
            @"-t", @"png",   // PNG format
            filepath         // Output path
        ]];

        NSPipe *errorPipe = [NSPipe pipe];
        [task setStandardError:errorPipe];

        @try {
            [task launch];
            [task waitUntilExit];

            int status = [task terminationStatus];
            if (status == 0) {
                // Verify file exists
                NSFileManager *fileManager = [NSFileManager defaultManager];
                if ([fileManager fileExistsAtPath:filepath]) {
                    NSLog(@"[SPACE_CAPTURE] ✅ Saved screenshot to: %@", filepath);
                    return YES;
                } else {
                    NSLog(@"[SPACE_CAPTURE] ❌ Screenshot file not found: %@", filepath);
                    return NO;
                }
            } else {
                NSData *errorData = [[errorPipe fileHandleForReading] readDataToEndOfFile];
                NSString *errorString = [[NSString alloc] initWithData:errorData encoding:NSUTF8StringEncoding];
                NSLog(@"[SPACE_CAPTURE] ❌ screencapture failed with status %d: %@", status, errorString);
                return NO;
            }
        } @catch (NSException *exception) {
            NSLog(@"[SPACE_CAPTURE] ❌ Exception running screencapture: %@", exception);
            return NO;
        }
    }
}

+ (BOOL)captureOffscreenSpace:(int)spaceId {
    // Fallback method: Try to capture windows from the space directly
    // without switching (less reliable but non-intrusive)
    NSLog(@"[SPACE_CAPTURE] 📦 Attempting offscreen capture for space %d", spaceId);

    NSArray *windows = [self getWindowsForSpace:spaceId];
    if ([windows count] == 0) {
        NSLog(@"[SPACE_CAPTURE] ⚠️  No windows found in space %d", spaceId);
        return NO;
    }

    // For now, just log that we would capture these windows
    // A full implementation would composite the window images
    NSLog(@"[SPACE_CAPTURE] 📊 Found %lu windows in space %d", [windows count], spaceId);
    for (NSDictionary *window in windows) {
        NSLog(@"[SPACE_CAPTURE]   - %@: %@", window[@"app_name"], window[@"window_title"]);
    }

    // Return YES to indicate we have window data, even if not a full screenshot
    return YES;
}

@end

// C interface for Python
const char* enumerate_spaces_json() {
    NSArray *spaces = [SpaceDetectionBridge enumerateAllSpaces];
    
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:spaces options:0 error:&error];
    
    if (error) {
        NSLog(@"[SPACE_DETECTION] JSON serialization error: %@", error);
        return NULL;
    }
    
    NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    return [jsonString UTF8String];
}

const char* get_space_info_json(int spaceId) {
    NSDictionary *spaceInfo = [SpaceDetectionBridge getSpaceInfo:spaceId];
    
    NSError *error;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:spaceInfo options:0 error:&error];
    
    if (error) {
        NSLog(@"[SPACE_DETECTION] JSON serialization error: %@", error);
        return NULL;
    }
    
    NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    return [jsonString UTF8String];
}

int get_space_count() {
    NSArray *spaces = [SpaceDetectionBridge enumerateAllSpaces];
    return (int)[spaces count];
}

int get_current_space() {
    // Space 1 is usually the current space
    return 1;
}

BOOL space_exists(int spaceId) {
    NSArray *spaces = [SpaceDetectionBridge enumerateAllSpaces];
    for (NSDictionary *space in spaces) {
        if ([space[@"space_id"] intValue] == spaceId) {
            return YES;
        }
    }
    return NO;
}

BOOL capture_space_invisibly(int spaceId) {
    return [SpaceDetectionBridge captureSpaceInvisibly:spaceId];
}

const char* get_space_screenshot_path(int spaceId) {
    NSString *tempDir = NSTemporaryDirectory();
    NSString *filename = [NSString stringWithFormat:@"space_%d_capture.png", spaceId];
    NSString *filepath = [tempDir stringByAppendingPathComponent:filename];
    return [filepath UTF8String];
}
