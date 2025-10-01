"""
Browser Controller
=================

Controls browser automation for Google Docs, web browsing, and other
browser-based tasks using AppleScript and JavaScript execution.
"""

import asyncio
import logging
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BrowserController:
    """
    Controls Safari/Chrome via AppleScript and JavaScript

    This provides a lightweight browser automation solution without
    requiring Selenium or other heavy dependencies.
    """

    def __init__(self, preferred_browser: str = "Chrome"):
        """
        Initialize browser controller

        Args:
            preferred_browser: "Safari" or "Chrome" (default: Chrome for Google Docs)
        """
        self.browser = preferred_browser
        self._current_url = None
        self._browser_app_name = "Google Chrome" if preferred_browser == "Chrome" else "Safari"

    async def navigate(self, url: str) -> bool:
        """
        Navigate to a URL

        Args:
            url: The URL to navigate to

        Returns:
            True if successful
        """
        try:
            script = self._build_navigation_script(url)
            result = await self._run_applescript(script)

            if result:
                self._current_url = url
                logger.info(f"Navigated to {url}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False

    async def get_current_url(self) -> Optional[str]:
        """Get the current URL of the active tab"""
        try:
            if self.browser == "Safari":
                script = '''
                tell application "Safari"
                    return URL of front document
                end tell
                '''
            else:  # Chrome
                script = '''
                tell application "Google Chrome"
                    return URL of active tab of front window
                end tell
                '''

            result = await self._run_applescript(script)
            if result:
                self._current_url = result.strip()
                return self._current_url

            return None

        except Exception as e:
            logger.error(f"Failed to get current URL: {e}")
            return None

    async def execute_javascript(self, js_code: str) -> Optional[str]:
        """
        Execute JavaScript in the current page

        Args:
            js_code: JavaScript code to execute

        Returns:
            Result of the JavaScript execution
        """
        try:
            if self.browser == "Safari":
                script = f'''
                tell application "Safari"
                    do JavaScript "{self._escape_js(js_code)}" in front document
                end tell
                '''
            else:  # Chrome
                script = f'''
                tell application "Google Chrome"
                    execute active tab of front window javascript "{self._escape_js(js_code)}"
                end tell
                '''

            result = await self._run_applescript(script)
            return result

        except Exception as e:
            logger.error(f"Failed to execute JavaScript: {e}")
            return None

    async def type_text(self, text: str, delay: float = 0.05) -> bool:
        """
        Type text into the current document

        Args:
            text: Text to type
            delay: Delay between characters (seconds)

        Returns:
            True if successful
        """
        try:
            # For Google Docs, we'll use JavaScript to insert text
            # This is more reliable than keystroke simulation
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            js_code = f'''
            (function() {{
                // Try to insert text into Google Docs editor
                var editor = document.querySelector('[contenteditable="true"]');
                if (editor) {{
                    // Create a text node
                    var textNode = document.createTextNode("{escaped_text}");

                    // Insert at cursor position
                    var selection = window.getSelection();
                    if (selection.rangeCount > 0) {{
                        var range = selection.getRangeAt(0);
                        range.deleteContents();
                        range.insertNode(textNode);

                        // Move cursor to end of inserted text
                        range.setStartAfter(textNode);
                        range.setEndAfter(textNode);
                        selection.removeAllRanges();
                        selection.addRange(range);
                    }} else {{
                        editor.appendChild(textNode);
                    }}

                    return "success";
                }}
                return "no editor found";
            }})();
            '''

            result = await self.execute_javascript(js_code)

            if result and "success" in result:
                logger.debug(f"Typed {len(text)} characters")
                return True

            logger.warning(f"Type result: {result}")
            return False

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    async def type_text_streaming(self, text: str, chunk_size: int = 100) -> bool:
        """
        Type text in chunks for real-time streaming effect

        Args:
            text: Text to type
            chunk_size: Number of characters per chunk

        Returns:
            True if successful
        """
        try:
            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            for chunk in chunks:
                success = await self.type_text(chunk, delay=0)
                if not success:
                    logger.warning("Failed to type chunk, continuing anyway")

                # Small delay between chunks for visual effect
                await asyncio.sleep(0.1)

            return True

        except Exception as e:
            logger.error(f"Failed to stream text: {e}")
            return False

    async def focus_browser(self) -> bool:
        """Bring browser to front"""
        try:
            script = f'''
            tell application "{self.browser}"
                activate
            end tell
            '''

            await self._run_applescript(script)
            return True

        except Exception as e:
            logger.error(f"Failed to focus browser: {e}")
            return False

    async def new_tab(self, url: Optional[str] = None) -> bool:
        """
        Open a new tab

        Args:
            url: Optional URL to open in the new tab

        Returns:
            True if successful
        """
        try:
            if self.browser == "Safari":
                if url:
                    script = f'''
                    tell application "Safari"
                        tell front window
                            set current tab to (make new tab with properties {{URL:"{url}"}})
                        end tell
                    end tell
                    '''
                else:
                    script = '''
                    tell application "Safari"
                        tell front window
                            set current tab to (make new tab)
                        end tell
                    end tell
                    '''
            else:  # Chrome
                if url:
                    script = f'''
                    tell application "Google Chrome"
                        tell front window
                            make new tab with properties {{URL:"{url}"}}
                        end tell
                    end tell
                    '''
                else:
                    script = '''
                    tell application "Google Chrome"
                        tell front window
                            make new tab
                        end tell
                    end tell
                    '''

            await self._run_applescript(script)
            if url:
                self._current_url = url

            return True

        except Exception as e:
            logger.error(f"Failed to open new tab: {e}")
            return False

    def _build_navigation_script(self, url: str) -> str:
        """Build AppleScript for navigation"""
        if self.browser == "Chrome":
            # Chrome: Open in normal (non-incognito) window
            return f'''
            tell application "Google Chrome"
                activate

                -- Check if Chrome is running
                if (count of windows) is 0 then
                    -- No windows, create new one
                    make new window
                    set URL of active tab of front window to "{url}"
                else
                    -- Use existing normal window or create new tab
                    set foundWindow to false
                    repeat with w in windows
                        if mode of w is not "incognito" then
                            set foundWindow to true
                            tell w
                                set URL of (make new tab) to "{url}"
                                set active tab index to (count of tabs)
                            end tell
                            exit repeat
                        end if
                    end repeat

                    if foundWindow is false then
                        -- All windows are incognito, create normal window
                        make new window
                        set URL of active tab of front window to "{url}"
                    end if
                end if
            end tell
            '''
        else:  # Safari
            return f'''
            tell application "Safari"
                activate
                if (count of windows) is 0 then
                    make new document
                end if
                set URL of front document to "{url}"
            end tell
            '''

    async def _run_applescript(self, script: str) -> Optional[str]:
        """
        Execute AppleScript and return result

        Args:
            script: AppleScript code to execute

        Returns:
            Script output or None if failed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return stdout.decode('utf-8').strip()
            else:
                error_msg = stderr.decode('utf-8').strip()
                logger.error(f"AppleScript error: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Failed to run AppleScript: {e}")
            return None

    def _escape_js(self, js_code: str) -> str:
        """Escape JavaScript code for AppleScript"""
        # Escape quotes and backslashes for AppleScript string
        escaped = js_code.replace('\\', '\\\\').replace('"', '\\"')
        return escaped


# Global instance
_browser_controller: Optional[BrowserController] = None


def get_browser_controller(preferred_browser: str = "Chrome") -> BrowserController:
    """Get or create global browser controller instance (defaults to Chrome for Google Docs)"""
    global _browser_controller
    if _browser_controller is None:
        _browser_controller = BrowserController(preferred_browser)
    return _browser_controller
