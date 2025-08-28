#!/usr/bin/env python3
"""
Enhanced screen capture with optional Claude Vision intelligence
This is where your revolutionary insight comes to life!
"""

import subprocess
import tempfile
import os
import base64
import io
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any

def capture_screen_fallback():
    """
    Capture screen using macOS screencapture command as fallback
    This often works when Quartz fails due to permissions
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        # Use screencapture command (usually has better permissions)
        result = subprocess.run(
            ["screencapture", "-x", "-C", tmp_path], capture_output=True, text=True
        )

        if result.returncode == 0 and os.path.exists(tmp_path):
            # Load image
            image = Image.open(tmp_path)
            # Convert to numpy array
            img_array = np.array(image)

            # Clean up temp file
            os.unlink(tmp_path)

            # Convert RGBA to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            # Return PIL Image instead of numpy array for compatibility
            return image
        else:
            return None

    except Exception as e:
        print(f"Fallback capture failed: {e}")
        return None

def capture_with_intelligence(query: Optional[str] = None, 
                            use_claude: bool = True) -> Dict[str, Any]:
    """
    Revolutionary enhancement: Capture + Claude Intelligence
    
    This transforms basic screen capture into an intelligent vision system
    that understands context, answers questions, and provides insights.
    
    Args:
        query: Natural language question about the screen
        use_claude: Whether to use Claude Vision for analysis
        
    Returns:
        Dict with 'image', 'analysis', and 'intelligence_used'
    """
    # Step 1: Capture the screen
    screenshot = capture_screen_fallback()
    
    if screenshot is None:
        return {
            "success": False,
            "error": "Failed to capture screen",
            "suggestion": "Check screen recording permissions"
        }
    
    # Step 2: If no Claude analysis requested, return basic capture
    if not use_claude or not query:
        return {
            "success": True,
            "image": screenshot,
            "intelligence_used": False
        }
    
    # Step 3: Add Claude intelligence
    try:
        analysis = analyze_with_claude_vision(screenshot, query)
        return {
            "success": True,
            "image": screenshot,
            "analysis": analysis,
            "intelligence_used": True
        }
    except Exception as e:
        # Fallback to basic capture if Claude fails
        return {
            "success": True,
            "image": screenshot,
            "intelligence_used": False,
            "error": f"Claude analysis failed: {e}"
        }

def analyze_with_claude_vision(screenshot_array, 
                              query: str) -> str:
    """
    Send screenshot to Claude for intelligent analysis.
    
    This is the revolutionary part - transforming pixels into understanding!
    """
    try:
        # Import here to avoid dependency if not using Claude
        from anthropic import Anthropic
        import os
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "Claude Vision requires ANTHROPIC_API_KEY in environment"
        
        # Handle both PIL Image and numpy array
        if isinstance(screenshot_array, np.ndarray):
            image = Image.fromarray(screenshot_array)
        elif isinstance(screenshot_array, Image.Image):
            image = screenshot_array
        else:
            raise ValueError(f"Unsupported image type: {type(screenshot_array)}")
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Initialize Claude
        client = Anthropic(api_key=api_key)
        
        # Build intelligent prompt
        prompt = f"""You are JARVIS, an AI assistant with vision capabilities.
        Analyze this screenshot and answer: {query}
        
        Be specific, helpful, and focus on what's most relevant to the user's question.
        If you see any issues, errors, or things that need attention, point them out.
        """
        
        # Send to Claude Vision
        # Use Sonnet for best vision capabilities
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude Vision analysis failed: {e}")

# Example intelligent queries that showcase the power
EXAMPLE_QUERIES = {
    "general": "What do you see on my screen?",
    "errors": "Are there any error messages or problems visible?",
    "work": "What am I currently working on?",
    "help": "What should I do next based on what you see?",
    "updates": "Do you see any software updates or notifications?",
    "ui": "Help me find the save button or menu",
    "code": "Analyze the code on my screen and suggest improvements",
    "debug": "Help me debug the error I'm seeing"
}

if __name__ == "__main__":
    # Demo the enhanced capture
    print("🚀 Enhanced Screen Capture Demo")
    print("=" * 40)
    
    # Test basic capture
    print("\n1. Testing basic capture...")
    result = capture_with_intelligence(use_claude=False)
    if result["success"]:
        img = result['image']
        if isinstance(img, Image.Image):
            print(f"✅ Captured screen: {img.size}")
        else:
            print(f"✅ Captured screen: {img.shape}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Test intelligent capture (if API key available)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n2. Testing intelligent capture...")
        result = capture_with_intelligence(
            query="What applications are open and what is the user doing?",
            use_claude=True
        )
        if result["success"] and result.get("intelligence_used"):
            print("✅ Claude Analysis:")
            print(result["analysis"])
        else:
            print("❌ Intelligence not available")
    else:
        print("\n⚠️  Set ANTHROPIC_API_KEY for intelligent vision features")
