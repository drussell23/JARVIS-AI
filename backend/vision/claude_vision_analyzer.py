"""
Claude Vision Analyzer - Advanced screen understanding using Claude's vision capabilities
"""

import base64
import io
from typing import Dict, List, Optional, Any
from PIL import Image
import numpy as np
from anthropic import Anthropic
import json

class ClaudeVisionAnalyzer:
    """Use Claude's vision capabilities for advanced screen understanding"""
    
    def __init__(self, api_key: str):
        """Initialize Claude vision analyzer"""
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Current vision-capable model
        
    async def analyze_screenshot(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Send screenshot to Claude for analysis
        
        Args:
            image: Screenshot as PIL Image or numpy array
            prompt: What to analyze in the image
            
        Returns:
            Analysis results from Claude
        """
        # Handle both PIL Image and numpy array inputs
        if isinstance(image, np.ndarray):
            # Ensure it's the right dtype for PIL
            if image.dtype == object:
                raise ValueError("Invalid numpy array dtype. Expected uint8 array.")
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            # Already a PIL Image
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare message for Claude
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return self._parse_claude_response(message.content[0].text)
    
    def _parse_claude_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's response into structured data"""
        # Try to extract JSON if Claude returns structured data
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to text analysis
        return {
            "description": response,
            "has_updates": "update" in response.lower(),
            "applications_mentioned": self._extract_app_names(response),
            "actions_suggested": self._extract_actions(response)
        }
    
    def _extract_app_names(self, text: str) -> List[str]:
        """Extract application names from Claude's response"""
        common_apps = [
            "Chrome", "Safari", "Firefox", "Mail", "Messages", "Slack",
            "VS Code", "Xcode", "Terminal", "Finder", "System Preferences",
            "App Store", "Activity Monitor", "Spotify", "Discord"
        ]
        
        found_apps = []
        text_lower = text.lower()
        
        for app in common_apps:
            if app.lower() in text_lower:
                found_apps.append(app)
        
        return found_apps
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract suggested actions from Claude's response"""
        action_keywords = [
            "should update", "recommend updating", "needs to be updated",
            "click on", "open", "close", "restart", "install"
        ]
        
        actions = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in action_keywords:
                if keyword in sentence_lower:
                    actions.append(sentence.strip())
                    break
        
        return actions
    
    async def check_for_software_updates(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Specialized prompt for checking software updates"""
        prompt = """Analyze this screenshot and identify any software update notifications or indicators.
        
        Please look for:
        1. System update notifications (macOS updates)
        2. App update badges or notifications
        3. Browser update indicators
        4. Any text mentioning updates, new versions, or upgrades
        5. Red notification badges on app icons
        6. System preference or app store update sections
        
        Respond in JSON format:
        {
            "updates_found": true/false,
            "update_details": [
                {
                    "type": "system/app/browser",
                    "name": "application name",
                    "version": "version if visible",
                    "urgency": "critical/recommended/optional",
                    "location": "where on screen"
                }
            ],
            "recommended_action": "what the user should do"
        }"""
        
        return await self.analyze_screenshot(screenshot, prompt)
    
    async def understand_user_activity(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Understand what the user is currently doing"""
        prompt = """Analyze this screenshot and describe what the user appears to be doing.
        
        Please identify:
        1. Which applications are open and active
        2. What type of work or activity is happening
        3. Any potential distractions or interruptions
        4. Suggestions for productivity or assistance
        
        Be concise but informative in your response."""
        
        return await self.analyze_screenshot(screenshot, prompt)
    
    async def identify_ui_elements(self, screenshot: np.ndarray, target: str) -> Dict[str, Any]:
        """Identify specific UI elements on screen"""
        prompt = f"""Look at this screenshot and help me find: {target}
        
        Please describe:
        1. Where it is located on the screen (top/bottom/left/right/center)
        2. What it looks like (color, shape, text)
        3. Whether it appears clickable or interactive
        4. Any associated text or labels
        
        If you cannot find it, suggest where it might typically be located."""
        
        return await self.analyze_screenshot(screenshot, prompt)
    
    async def read_text_content(self, screenshot: np.ndarray, region_description: Optional[str] = None) -> str:
        """Read and extract text from screenshot"""
        if region_description:
            prompt = f"Please read and transcribe all text visible in the {region_description} area of this screenshot."
        else:
            prompt = "Please read and transcribe all significant text visible in this screenshot, organizing it by section or application."
        
        result = await self.analyze_screenshot(screenshot, prompt)
        return result.get("description", "")
    
    async def security_check(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Check for security concerns on screen"""
        prompt = """Analyze this screenshot for any security or privacy concerns.
        
        Look for:
        1. Exposed passwords or sensitive information
        2. Suspicious pop-ups or dialogs
        3. Security warnings or alerts
        4. Unusual system notifications
        5. Potentially malicious content
        
        Respond with any concerns found and recommended actions."""
        
        return await self.analyze_screenshot(screenshot, prompt)
    
    async def analyze_image_with_prompt(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """Analyze a base64 encoded image with a custom prompt
        
        Args:
            image_base64: Base64 encoded image string
            prompt: Custom prompt for analysis
            
        Returns:
            Analysis results from Claude
        """
        # Prepare message for Claude
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return self._parse_claude_response(message.content[0].text)
    
    async def analyze_workspace_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workspace context without screenshot
        
        Args:
            state: Dictionary containing workspace state information
            
        Returns:
            Analysis based on workspace metadata
        """
        # Create a text-based analysis when no screenshot is available
        analysis_text = f"""Based on the workspace information:
- Focused application: {state.get('focused_app', 'Unknown')}
- Number of windows open: {state.get('window_count', 0)}
- Active windows: {', '.join([w.get('app', 'Unknown') for w in state.get('windows', [])[:5]])}

The user appears to be working with multiple applications. Without seeing the actual screen content, I can provide general workspace information but cannot analyze specific content or provide detailed insights about what you're working on."""
        
        return {
            "description": analysis_text,
            "focused_app": state.get('focused_app'),
            "window_count": state.get('window_count'),
            "requires_screenshot": True,
            "confidence": 0.3
        }