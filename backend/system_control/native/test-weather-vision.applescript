-- Test AppleScript to extract Weather app data

tell application "Weather"
    activate
end tell

delay 1

tell application "System Events"
    tell process "Weather"
        set frontmost to true
        
        -- Get window properties
        set windowExists to exists window 1
        
        if windowExists then
            set windowTitle to title of window 1
            log "Window Title: " & windowTitle
            
            -- Try to get all UI elements
            tell window 1
                set allElements to entire contents
                
                -- Get all static text values
                set textValues to {}
                repeat with elem in allElements
                    try
                        if class of elem is static text then
                            set elemValue to value of elem
                            if elemValue is not missing value and length of elemValue > 0 then
                                set end of textValues to elemValue
                            end if
                        end if
                    on error
                        -- Skip elements that cause errors
                    end try
                end repeat
                
                -- Return combined text
                set AppleScript's text item delimiters to "|"
                return textValues as string
            end tell
        else
            return "Weather window not found"
        end if
    end tell
end tell