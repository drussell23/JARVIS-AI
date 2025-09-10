#import <Foundation/Foundation.h>
#import <CoreLocation/CoreLocation.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Requesting location for Terminal...");
        
        // Create location manager
        CLLocationManager *locationManager = [[CLLocationManager alloc] init];
        
        // Start location updates
        [locationManager startUpdatingLocation];
        
        // Try to get current location
        CLLocation *location = locationManager.location;
        if (location) {
            NSLog(@"Current location: %@", location);
        }
        
        // Keep running for a moment
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2.0]];
        
        NSLog(@"Done. Check Location Services for Terminal.");
    }
    return 0;
}
