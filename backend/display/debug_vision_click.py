#!/usr/bin/env python3
"""
Debug Vision Navigator - See what JARVIS sees and where it clicks
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault('ANTHROPIC_API_KEY', os.getenv('ANTHROPIC_API_KEY', ''))

from PIL import Image, ImageDraw, ImageFont
import pyautogui
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_control_center_detection():
    """Test what JARVIS sees and where it clicks"""
    print("\n" + "="*70)
    print("🎯 Vision Navigator Debug - Control Center Detection")
    print("="*70)
    
    # Step 1: Capture screen
    print("\n1. Capturing screen...")
    temp_path = Path.home() / '.jarvis' / 'screenshots' / 'ui_navigation' / 'debug_control_center.png'
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    import subprocess
    subprocess.run(['screencapture', '-x', str(temp_path)], check=True)
    print(f"   ✅ Screenshot saved: {temp_path}")
    
    # Step 2: Show current heuristic click position
    screen_width, screen_height = pyautogui.size()
    heuristic_x = screen_width - 70
    heuristic_y = 12
    
    print(f"\n2. Current heuristic click position:")
    print(f"   Screen size: {screen_width}x{screen_height}")
    print(f"   Heuristic: ({heuristic_x}, {heuristic_y})")
    
    # Step 3: Analyze with Claude Vision
    print(f"\n3. Asking Claude Vision to find Control Center...")
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("   ❌ No ANTHROPIC_API_KEY found")
        return
    
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    image = Image.open(temp_path)
    
    prompt = """Look at this macOS menu bar screenshot.

I need you to find the Control Center icon. It's in the top-right corner of the screen.
The Control Center icon looks like two overlapping rounded rectangles or squares.

Please tell me:
1. Can you see the Control Center icon?
2. What is its EXACT pixel position from the LEFT edge of the screen?
3. What is its EXACT pixel position from the TOP edge of the screen?

Respond in this format:
FOUND: YES or NO
X_POSITION: [number from left edge]
Y_POSITION: [number from top edge]
DESCRIPTION: [what you see]"""
    
    response, metrics = await analyzer.analyze_screenshot(
        image=image,
        prompt=prompt,
        use_cache=False
    )
    
    print(f"\n4. Claude Vision Response:")
    print(f"   {response.get('response', response)}")
    
    # Step 4: Draw debug markers
    print(f"\n5. Creating debug visualization...")
    
    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)
    
    # Draw heuristic position (RED)
    marker_size = 20
    draw.ellipse(
        [heuristic_x - marker_size, heuristic_y - marker_size,
         heuristic_x + marker_size, heuristic_y + marker_size],
        outline='red', width=3
    )
    draw.text((heuristic_x - 50, heuristic_y + 30), "Heuristic", fill='red')
    
    # Draw screen dimensions
    draw.text((10, 10), f"Screen: {screen_width}x{screen_height}", fill='yellow')
    
    # Try to extract coordinates from Claude's response
    import re
    response_text = str(response.get('response', response))
    
    x_match = re.search(r'X_POSITION:\s*(\d+)', response_text, re.IGNORECASE)
    y_match = re.search(r'Y_POSITION:\s*(\d+)', response_text, re.IGNORECASE)
    
    if x_match and y_match:
        claude_x = int(x_match.group(1))
        claude_y = int(y_match.group(1))
        
        # Draw Claude's position (GREEN)
        draw.ellipse(
            [claude_x - marker_size, claude_y - marker_size,
             claude_x + marker_size, claude_y + marker_size],
            outline='green', width=3
        )
        draw.text((claude_x - 50, claude_y + 30), "Claude", fill='green')
        
        print(f"   ✅ Claude found Control Center at: ({claude_x}, {claude_y})")
    else:
        print(f"   ⚠️  Could not extract coordinates from Claude's response")
    
    # Save debug image
    debug_path = temp_path.parent / 'debug_annotated.png'
    debug_image.save(debug_path)
    print(f"\n6. Debug image saved: {debug_path}")
    print(f"   RED circle = Where JARVIS currently clicks (heuristic)")
    print(f"   GREEN circle = Where Claude Vision sees Control Center")
    
    # Step 5: Recommendations
    print(f"\n7. Recommendations:")
    if x_match and y_match:
        claude_x = int(x_match.group(1))
        diff_x = abs(claude_x - heuristic_x)
        
        if diff_x > 50:
            print(f"   ⚠️  Heuristic is OFF by {diff_x} pixels!")
            print(f"   📝 Update heuristic to: ({claude_x}, 12)")
        else:
            print(f"   ✅ Heuristic is close (within {diff_x}px)")
    
    print(f"\n" + "="*70)
    print("✅ Debug complete! Check the annotated screenshot.")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_control_center_detection())
