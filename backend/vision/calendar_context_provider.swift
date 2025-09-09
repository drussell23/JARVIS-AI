#!/usr/bin/env swift

//
// Calendar Context Provider for Goal Inference System
// Provides native macOS calendar integration for goal inference
//

import Foundation
import EventKit
import Combine

/// Calendar event data for goal inference
struct CalendarEvent: Codable {
    let eventId: String
    let title: String
    let startDate: Date
    let endDate: Date
    let isAllDay: Bool
    let location: String?
    let notes: String?
    let attendees: [String]
    let eventType: EventType
    
    enum EventType: String, Codable {
        case meeting = "meeting"
        case appointment = "appointment"
        case deadline = "deadline"
        case reminder = "reminder"
        case other = "other"
    }
}

/// Time context for goal inference
struct TimeContext: Codable {
    let currentTime: Date
    let timeOfDay: TimeOfDay
    let dayOfWeek: DayOfWeek
    let isWorkingHours: Bool
    let upcomingEvents: [CalendarEvent]
    let currentEvent: CalendarEvent?
    let nextEvent: CalendarEvent?
    
    enum TimeOfDay: String, Codable {
        case earlyMorning = "early_morning"  // 5-8 AM
        case morning = "morning"             // 8-12 PM
        case afternoon = "afternoon"         // 12-5 PM
        case evening = "evening"             // 5-9 PM
        case night = "night"                 // 9 PM-5 AM
    }
    
    enum DayOfWeek: String, Codable {
        case weekday = "weekday"
        case weekend = "weekend"
    }
}

/// Calendar context provider for goal inference
class CalendarContextProvider: NSObject {
    private let eventStore = EKEventStore()
    var authorizationStatus: EKAuthorizationStatus = .notDetermined
    private let contextUpdateSubject = PassthroughSubject<TimeContext, Never>()
    private var updateTimer: Timer?
    
    // Configuration
    private let hoursAhead: Int
    private let maxEvents: Int
    
    init(hoursAhead: Int = 4, maxEvents: Int = 10) {
        self.hoursAhead = hoursAhead
        self.maxEvents = maxEvents
        super.init()
        requestCalendarAccess()
        startPeriodicUpdates()
    }
    
    deinit {
        updateTimer?.invalidate()
    }
    
    /// Request calendar access
    private func requestCalendarAccess() {
        if #available(macOS 14.0, *) {
            eventStore.requestFullAccessToEvents { [weak self] granted, error in
                self?.authorizationStatus = granted ? .fullAccess : .denied
                if let error = error {
                    print("[CalendarContext] Error requesting access: \(error)")
                }
            }
        } else {
            eventStore.requestAccess(to: .event) { [weak self] granted, error in
                self?.authorizationStatus = granted ? .authorized : .denied
                if let error = error {
                    print("[CalendarContext] Error requesting access: \(error)")
                }
            }
        }
    }
    
    /// Get current time context
    func getCurrentTimeContext() -> TimeContext? {
        // Check authorization status
        if #available(macOS 14.0, *) {
            if authorizationStatus != .fullAccess {
                print("[CalendarContext] Calendar access not authorized (status: \(authorizationStatus.rawValue))")
                return nil
            }
        } else {
            if authorizationStatus != .authorized {
                print("[CalendarContext] Calendar access not authorized (status: \(authorizationStatus.rawValue))")
                return nil
            }
        }
        
        let now = Date()
        let calendar = Calendar.current
        
        // Determine time of day
        let hour = calendar.component(.hour, from: now)
        let timeOfDay: TimeContext.TimeOfDay
        switch hour {
        case 5..<8:
            timeOfDay = .earlyMorning
        case 8..<12:
            timeOfDay = .morning
        case 12..<17:
            timeOfDay = .afternoon
        case 17..<21:
            timeOfDay = .evening
        default:
            timeOfDay = .night
        }
        
        // Determine day type
        let weekday = calendar.component(.weekday, from: now)
        let dayOfWeek: TimeContext.DayOfWeek = (weekday == 1 || weekday == 7) ? .weekend : .weekday
        
        // Check working hours (9 AM - 5 PM on weekdays)
        let isWorkingHours = dayOfWeek == .weekday && hour >= 9 && hour < 17
        
        // Get events
        let endDate = calendar.date(byAdding: .hour, value: hoursAhead, to: now) ?? now
        let predicate = eventStore.predicateForEvents(withStart: now, end: endDate, calendars: nil)
        let events = eventStore.events(matching: predicate)
        
        // Convert to CalendarEvent
        let calendarEvents = events.prefix(maxEvents).map { event in
            CalendarEvent(
                eventId: event.eventIdentifier,
                title: event.title ?? "",
                startDate: event.startDate,
                endDate: event.endDate,
                isAllDay: event.isAllDay,
                location: event.location,
                notes: event.notes,
                attendees: event.attendees?.compactMap { $0.name } ?? [],
                eventType: categorizeEvent(event)
            )
        }
        
        // Find current and next events
        let currentEvent = calendarEvents.first { event in
            event.startDate <= now && event.endDate > now
        }
        
        let nextEvent = calendarEvents.first { event in
            event.startDate > now
        }
        
        return TimeContext(
            currentTime: now,
            timeOfDay: timeOfDay,
            dayOfWeek: dayOfWeek,
            isWorkingHours: isWorkingHours,
            upcomingEvents: calendarEvents,
            currentEvent: currentEvent,
            nextEvent: nextEvent
        )
    }
    
    /// Categorize event type based on title and properties
    private func categorizeEvent(_ event: EKEvent) -> CalendarEvent.EventType {
        let title = event.title?.lowercased() ?? ""
        
        // Check for meeting patterns
        let meetingKeywords = ["meeting", "call", "sync", "standup", "1:1", "interview", "discussion"]
        if meetingKeywords.contains(where: { title.contains($0) }) || !event.attendees.isNilOrEmpty {
            return .meeting
        }
        
        // Check for appointment patterns
        let appointmentKeywords = ["appointment", "doctor", "dentist", "therapy", "consultation"]
        if appointmentKeywords.contains(where: { title.contains($0) }) {
            return .appointment
        }
        
        // Check for deadline patterns
        let deadlineKeywords = ["due", "deadline", "submission", "deliver", "release"]
        if deadlineKeywords.contains(where: { title.contains($0) }) {
            return .deadline
        }
        
        // Check for reminder patterns
        let reminderKeywords = ["reminder", "todo", "task", "check"]
        if reminderKeywords.contains(where: { title.contains($0) }) {
            return .reminder
        }
        
        return .other
    }
    
    /// Get calendar context as JSON for Python integration
    func getContextJSON() -> String? {
        guard let context = getCurrentTimeContext() else {
            return nil
        }
        
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        
        do {
            let data = try encoder.encode(context)
            return String(data: data, encoding: .utf8)
        } catch {
            print("[CalendarContext] Error encoding context: \(error)")
            return nil
        }
    }
    
    /// Start periodic context updates
    private func startPeriodicUpdates() {
        // Update every minute
        updateTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            if let context = self?.getCurrentTimeContext() {
                self?.contextUpdateSubject.send(context)
            }
        }
    }
    
    /// Get context update publisher
    var contextUpdates: AnyPublisher<TimeContext, Never> {
        contextUpdateSubject.eraseToAnyPublisher()
    }
}

// MARK: - Python Integration

/// C-compatible wrapper for Python FFI
@_cdecl("calendar_context_create")
public func calendar_context_create() -> UnsafeMutableRawPointer {
    let provider = CalendarContextProvider()
    return Unmanaged.passRetained(provider).toOpaque()
}

@_cdecl("calendar_context_get_json")
public func calendar_context_get_json(_ providerPtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
    let provider = Unmanaged<CalendarContextProvider>.fromOpaque(providerPtr).takeUnretainedValue()
    
    guard let json = provider.getContextJSON() else {
        return nil
    }
    
    // Create a mutable copy and return as immutable pointer
    let cString = strdup(json)
    return UnsafePointer(cString)
}

@_cdecl("calendar_context_free_string")
public func calendar_context_free_string(_ str: UnsafePointer<CChar>) {
    free(UnsafeMutablePointer(mutating: str))
}

@_cdecl("calendar_context_destroy")
public func calendar_context_destroy(_ providerPtr: UnsafeMutableRawPointer) {
    Unmanaged<CalendarContextProvider>.fromOpaque(providerPtr).release()
}

// MARK: - Extensions

extension Optional where Wrapped: Collection {
    var isNilOrEmpty: Bool {
        return self?.isEmpty ?? true
    }
}

// MARK: - Command Line Interface

// Main entry point for Python bridge
if CommandLine.argc > 0 {
    // Parse command line arguments
    var hoursAhead = 24
    var maxEvents = 50
    
    for i in 1..<Int(CommandLine.argc) {
        let arg = CommandLine.arguments[i]
        if arg == "--hours" && i + 1 < Int(CommandLine.argc) {
            hoursAhead = Int(CommandLine.arguments[i + 1]) ?? 24
        } else if arg == "--max-events" && i + 1 < Int(CommandLine.argc) {
            maxEvents = Int(CommandLine.arguments[i + 1]) ?? 50
        }
    }
    
    // Create provider and get context
    let provider = CalendarContextProvider(hoursAhead: hoursAhead, maxEvents: maxEvents)
    
    // Give a moment for authorization
    Thread.sleep(forTimeInterval: 0.5)
    
    if let json = provider.getContextJSON() {
        print(json)
        exit(0)
    } else {
        // Provide helpful error message
        let errorMessage: String
        if provider.authorizationStatus == .denied {
            errorMessage = "Calendar access denied. Please grant calendar permissions in System Settings > Privacy & Security > Calendar"
        } else if provider.authorizationStatus == .notDetermined {
            errorMessage = "Calendar access not yet requested. The system will prompt for permission."
        } else {
            errorMessage = "Failed to get calendar context"
        }
        print("{\"error\": \"\(errorMessage)\"}")
        exit(1)
    }
}

// Original test interface
if CommandLine.arguments.count > 1 && CommandLine.arguments[0].contains("test") {
    let command = CommandLine.arguments[1]
    
    switch command {
    case "test":
        // Test mode
        let provider = CalendarContextProvider()
        
        // Give time for authorization
        Thread.sleep(forTimeInterval: 2.0)
        
        if let json = provider.getContextJSON() {
            print(json)
        } else {
            print("{\"error\": \"Failed to get calendar context\"}")
        }
        
    case "monitor":
        // Monitor mode - continuous updates
        let provider = CalendarContextProvider()
        var cancellable: AnyCancellable?
        
        print("[CalendarContext] Starting monitor mode...")
        
        cancellable = provider.contextUpdates.sink { context in
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            
            if let data = try? encoder.encode(context),
               let json = String(data: data, encoding: .utf8) {
                print(json)
                fflush(stdout)
            }
        }
        
        // Run forever
        RunLoop.main.run()
        
    default:
        print("Usage: \(CommandLine.arguments[0]) [test|monitor]")
        exit(1)
    }
} else {
    print("Calendar Context Provider loaded")
}