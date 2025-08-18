"""
Vision API endpoints for JARVIS screen comprehension
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import os
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.screen_vision import ScreenVisionSystem, JARVISVisionIntegration
from vision.claude_vision_analyzer import ClaudeVisionAnalyzer


router = APIRouter(prefix="/vision", tags=["vision"])

# Initialize vision systems
vision_system = ScreenVisionSystem()
claude_analyzer = None

# Initialize Claude analyzer if API key is available
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_key:
    claude_analyzer = ClaudeVisionAnalyzer(anthropic_key)

jarvis_vision = JARVISVisionIntegration(vision_system)


class VisionCommand(BaseModel):
    """Vision command request model"""
    command: str
    use_claude: bool = True
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)


class ScreenAnalysisRequest(BaseModel):
    """Screen analysis request model"""
    analysis_type: str  # "updates", "activity", "security", "text"
    prompt: Optional[str] = None
    region: Optional[Tuple[int, int, int, int]] = None


class UpdateMonitoringRequest(BaseModel):
    """Update monitoring configuration"""
    enabled: bool
    interval: int = 300  # seconds
    notify_critical_only: bool = False


# Global monitoring state
monitoring_config = {
    "active": False,
    "interval": 300,
    "last_check": None,
    "pending_updates": []
}


@router.get("/status")
async def get_vision_status() -> Dict[str, Any]:
    """Get current vision system status"""
    return {
        "vision_enabled": True,
        "claude_vision_available": claude_analyzer is not None,
        "monitoring_active": monitoring_config["active"],
        "last_scan": vision_system.last_scan_time.isoformat() if vision_system.last_scan_time else None,
        "detected_updates": len(vision_system.detected_updates),
        "capabilities": [
            "screen_capture",
            "text_extraction",
            "update_detection",
            "application_detection",
            "ui_element_detection",
            "claude_vision_analysis" if claude_analyzer else None
        ]
    }


@router.post("/command")
async def process_vision_command(request: VisionCommand) -> Dict[str, Any]:
    """Process a vision-related voice command"""
    try:
        # First try JARVIS integration
        response = await jarvis_vision.handle_vision_command(request.command)
        
        # If Claude is requested and available, enhance the response
        if request.use_claude and claude_analyzer and "what" in request.command.lower():
            # Capture screen for Claude analysis
            region = request.region
            screenshot = await vision_system.capture_screen(region)
            
            # Get Claude's analysis
            claude_result = await claude_analyzer.understand_user_activity(screenshot)
            
            # Combine responses
            response = f"{response} Additionally, {claude_result.get('description', '')}"
        
        return {
            "success": True,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_screen(request: ScreenAnalysisRequest) -> Dict[str, Any]:
    """Perform detailed screen analysis"""
    try:
        region = request.region
        screenshot = await vision_system.capture_screen(region)
        
        result = {}
        
        if request.analysis_type == "updates":
            # Check for software updates
            if claude_analyzer:
                result = await claude_analyzer.check_for_software_updates(screenshot)
            else:
                updates = await vision_system.scan_for_updates()
                result = {
                    "updates_found": len(updates) > 0,
                    "update_details": [
                        {
                            "type": u.update_type.value,
                            "name": u.application,
                            "version": u.version,
                            "urgency": u.urgency,
                            "description": u.description
                        }
                        for u in updates
                    ]
                }
        
        elif request.analysis_type == "activity" and claude_analyzer:
            # Understand user activity
            result = await claude_analyzer.understand_user_activity(screenshot)
        
        elif request.analysis_type == "security" and claude_analyzer:
            # Security check
            result = await claude_analyzer.security_check(screenshot)
        
        elif request.analysis_type == "text":
            # Extract text
            if claude_analyzer and request.prompt:
                text = await claude_analyzer.read_text_content(screenshot, request.prompt)
                result = {"extracted_text": text}
            else:
                elements = await vision_system.detect_text_regions(screenshot)
                result = {
                    "text_elements": [
                        {
                            "text": elem.text,
                            "location": elem.location,
                            "confidence": elem.confidence
                        }
                        for elem in elements
                    ]
                }
        
        else:
            # General context
            result = await vision_system.get_screen_context(region)
        
        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/updates")
async def configure_update_monitoring(
    request: UpdateMonitoringRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Configure automatic update monitoring"""
    try:
        monitoring_config["active"] = request.enabled
        monitoring_config["interval"] = request.interval
        
        if request.enabled and not jarvis_vision.monitoring_active:
            # Start monitoring in background
            jarvis_vision.monitoring_active = True
            background_tasks.add_task(monitor_updates_task)
            
            return {
                "success": True,
                "message": "Update monitoring activated",
                "config": monitoring_config
            }
        
        elif not request.enabled:
            jarvis_vision.monitoring_active = False
            monitoring_config["active"] = False
            
            return {
                "success": True,
                "message": "Update monitoring deactivated",
                "config": monitoring_config
            }
        
        else:
            return {
                "success": True,
                "message": "Monitoring already active",
                "config": monitoring_config
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def monitor_updates_task():
    """Background task for monitoring updates"""
    import asyncio
    
    while monitoring_config["active"]:
        try:
            # Scan for updates
            updates = await vision_system.scan_for_updates()
            
            if updates:
                # Store in config
                monitoring_config["pending_updates"] = [
                    {
                        "type": u.update_type.value,
                        "app": u.application,
                        "urgency": u.urgency,
                        "description": u.description,
                        "detected_at": u.detected_at.isoformat()
                    }
                    for u in updates
                ]
                monitoring_config["last_check"] = datetime.now().isoformat()
                
                # Here you would trigger JARVIS to speak about critical updates
                critical = [u for u in updates if u.urgency == "critical"]
                if critical:
                    print(f"JARVIS: Sir, {len(critical)} critical updates require your attention.")
            
        except Exception as e:
            print(f"Error in update monitoring: {e}")
        
        await asyncio.sleep(monitoring_config["interval"])


@router.get("/updates/pending")
async def get_pending_updates() -> Dict[str, Any]:
    """Get list of pending updates detected"""
    return {
        "pending_updates": monitoring_config["pending_updates"],
        "last_check": monitoring_config["last_check"],
        "monitoring_active": monitoring_config["active"]
    }


@router.post("/capture")
async def capture_screenshot() -> Dict[str, Any]:
    """Capture and describe current screen"""
    try:
        description = await vision_system.capture_and_describe()
        
        # If Claude is available, get more detailed analysis
        if claude_analyzer:
            screenshot = await vision_system.capture_screen()
            claude_result = await claude_analyzer.understand_user_activity(screenshot)
            description += f" {claude_result.get('description', '')}"
        
        return {
            "success": True,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_vision_capabilities() -> Dict[str, List[str]]:
    """Get detailed vision system capabilities"""
    return {
        "vision_commands": [
            "What's on my screen?",
            "Check for software updates",
            "Start monitoring for updates",
            "Stop monitoring",
            "Analyze my screen",
            "What applications are open?",
            "Read the text in [area]",
            "Is there anything I should update?"
        ],
        "analysis_types": [
            "updates - Check for software updates",
            "activity - Understand current user activity",
            "security - Check for security concerns",
            "text - Extract text from screen"
        ],
        "features": [
            "Real-time screen capture",
            "OCR text extraction",
            "Software update detection",
            "Application identification",
            "UI element detection",
            "Notification badge detection",
            "Claude vision integration" if claude_analyzer else "Claude vision (requires API key)"
        ]
    }