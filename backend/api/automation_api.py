from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

from engines.automation_engine import (
    AutomationEngine, ScheduledEvent, AutomationTask, 
    TaskStatus, TaskPriority, CalendarManager
)

class CalendarEventRequest(BaseModel):
    title: str
    start_time: str  # ISO format
    end_time: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    attendees: Optional[List[str]] = []
    reminders: Optional[List[int]] = [15, 60]  # Minutes before
    
    
class NaturalEventRequest(BaseModel):
    text: str
    
    
class WeatherRequest(BaseModel):
    location: str
    forecast_days: Optional[int] = 5
    
    
class InformationRequest(BaseModel):
    query_type: str  # news, stock, crypto, wikipedia
    query: str
    
    
class DeviceControlRequest(BaseModel):
    device_id: Optional[str] = None
    action: str  # turn_on, turn_off, set_brightness, etc.
    parameters: Optional[Dict[str, Any]] = {}
    
    
class SceneRequest(BaseModel):
    scene_name: str
    devices_config: Optional[List[Dict[str, Any]]] = []
    
    
class TaskRequest(BaseModel):
    name: str
    action: str
    parameters: Dict[str, Any]
    trigger: Optional[Dict[str, Any]] = None
    priority: Optional[str] = "medium"
    
    
class AutomationCommandRequest(BaseModel):
    command: str
    context: Optional[Dict[str, Any]] = {}

class AutomationAPI:
    """API for automation features"""
    
    def __init__(self, automation_engine: AutomationEngine):
        self.engine = automation_engine
        self.router = APIRouter()
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register API routes"""
        # Calendar endpoints
        self.router.add_api_route("/calendar/events", self.create_event, methods=["POST"])
        self.router.add_api_route("/calendar/events/natural", self.create_event_natural, methods=["POST"])
        self.router.add_api_route("/calendar/events", self.get_events, methods=["GET"])
        self.router.add_api_route("/calendar/events/{event_id}", self.delete_event, methods=["DELETE"])
        self.router.add_api_route("/calendar/today", self.get_today_events, methods=["GET"])
        self.router.add_api_route("/calendar/upcoming", self.get_upcoming_events, methods=["GET"])
        self.router.add_api_route("/calendar/export", self.export_calendar, methods=["GET"])
        
        # Weather endpoints
        self.router.add_api_route("/weather/current", self.get_current_weather, methods=["POST"])
        self.router.add_api_route("/weather/forecast", self.get_weather_forecast, methods=["POST"])
        
        # Information endpoints
        self.router.add_api_route("/info/query", self.get_information, methods=["POST"])
        self.router.add_api_route("/info/news", self.get_news, methods=["GET"])
        self.router.add_api_route("/info/stock/{symbol}", self.get_stock_info, methods=["GET"])
        self.router.add_api_route("/info/crypto/{symbol}", self.get_crypto_info, methods=["GET"])
        
        # Home automation endpoints
        self.router.add_api_route("/home/devices", self.list_devices, methods=["GET"])
        self.router.add_api_route("/home/devices/control", self.control_device, methods=["POST"])
        self.router.add_api_route("/home/scenes", self.create_scene, methods=["POST"])
        self.router.add_api_route("/home/scenes/{scene_name}", self.activate_scene, methods=["POST"])
        
        # Task automation endpoints
        self.router.add_api_route("/tasks", self.create_task, methods=["POST"])
        self.router.add_api_route("/tasks", self.list_tasks, methods=["GET"])
        self.router.add_api_route("/tasks/{task_id}", self.get_task, methods=["GET"])
        self.router.add_api_route("/tasks/{task_id}/execute", self.execute_task, methods=["POST"])
        self.router.add_api_route("/tasks/{task_id}/cancel", self.cancel_task, methods=["POST"])
        self.router.add_api_route("/tasks/plan", self.create_task_plan, methods=["POST"])
        
        # General automation endpoint
        self.router.add_api_route("/command", self.process_command, methods=["POST"])
        
    # Calendar endpoints
    async def create_event(self, request: CalendarEventRequest) -> Dict:
        """Create a calendar event"""
        try:
            event = ScheduledEvent(
                id=str(uuid.uuid4()),
                title=request.title,
                start_time=datetime.fromisoformat(request.start_time),
                end_time=datetime.fromisoformat(request.end_time) if request.end_time else None,
                location=request.location,
                description=request.description,
                attendees=request.attendees,
                reminders=request.reminders
            )
            
            event_id = self.engine.calendar.add_event(event)
            
            return {
                "status": "success",
                "event_id": event_id,
                "message": f"Event '{event.title}' scheduled for {event.start_time.strftime('%B %d at %I:%M %p')}"
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def create_event_natural(self, request: NaturalEventRequest) -> Dict:
        """Create event from natural language"""
        try:
            event = self.engine.calendar.create_event_from_text(request.text)
            
            if event:
                event_id = self.engine.calendar.add_event(event)
                return {
                    "status": "success",
                    "event_id": event_id,
                    "event": {
                        "title": event.title,
                        "start_time": event.start_time.isoformat(),
                        "reminders": event.reminders
                    },
                    "message": f"Event '{event.title}' scheduled for {event.start_time.strftime('%B %d at %I:%M %p')}"
                }
            else:
                raise ValueError("Could not parse event from text")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_events(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """Get calendar events"""
        try:
            if start_date and end_date:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                events = self.engine.calendar.get_events(start, end)
            else:
                # Default to next 7 days
                start = datetime.now()
                end = start + timedelta(days=7)
                events = self.engine.calendar.get_events(start, end)
                
            return {
                "events": [
                    {
                        "id": e.id,
                        "title": e.title,
                        "start_time": e.start_time.isoformat(),
                        "end_time": e.end_time.isoformat() if e.end_time else None,
                        "location": e.location,
                        "description": e.description
                    }
                    for e in events
                ],
                "count": len(events)
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def delete_event(self, event_id: str) -> Dict:
        """Delete a calendar event"""
        if self.engine.calendar.remove_event(event_id):
            return {"status": "success", "message": "Event deleted"}
        else:
            raise HTTPException(status_code=404, detail="Event not found")
            
    async def get_today_events(self) -> Dict:
        """Get today's events"""
        events = self.engine.calendar.get_today_events()
        return {
            "date": datetime.now().date().isoformat(),
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_time": e.start_time.strftime("%I:%M %p"),
                    "location": e.location
                }
                for e in events
            ],
            "count": len(events)
        }
        
    async def get_upcoming_events(self, hours: int = 24) -> Dict:
        """Get upcoming events"""
        events = self.engine.calendar.get_upcoming_events(hours)
        return {
            "hours": hours,
            "events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "start_time": e.start_time.isoformat(),
                    "time_until": str(e.start_time - datetime.now())
                }
                for e in events
            ],
            "count": len(events)
        }
        
    async def export_calendar(self) -> str:
        """Export calendar as iCal"""
        return self.engine.calendar.export_to_ical()
        
    # Weather endpoints
    async def get_current_weather(self, request: WeatherRequest) -> Dict:
        """Get current weather"""
        try:
            weather = await self.engine.weather.get_current_weather(request.location)
            return {
                "status": "success",
                "weather": weather,
                "formatted": self.engine.weather.format_weather_report(weather)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_weather_forecast(self, request: WeatherRequest) -> Dict:
        """Get weather forecast"""
        try:
            forecast = await self.engine.weather.get_forecast(request.location, request.forecast_days)
            return {
                "status": "success",
                "location": request.location,
                "days": request.forecast_days,
                "forecast": forecast
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Information endpoints
    async def get_information(self, request: InformationRequest) -> Dict:
        """Get information"""
        try:
            info = await self.engine.information.get_information(request.query_type, request.query)
            return {
                "status": "success",
                "query_type": request.query_type,
                "query": request.query,
                "data": info
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_news(self, topic: str = "technology") -> Dict:
        """Get news headlines"""
        try:
            news = await self.engine.information.get_information("news", topic)
            return news
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_stock_info(self, symbol: str) -> Dict:
        """Get stock information"""
        try:
            stock = await self.engine.information.get_information("stock", symbol)
            return stock
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_crypto_info(self, symbol: str) -> Dict:
        """Get cryptocurrency information"""
        try:
            crypto = await self.engine.information.get_information("crypto", symbol)
            return crypto
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Home automation endpoints
    async def list_devices(self) -> Dict:
        """List smart home devices"""
        devices = self.engine.home_automation.get_device_status()
        return {
            "devices": devices,
            "count": len(devices)
        }
        
    async def control_device(self, request: DeviceControlRequest) -> Dict:
        """Control a smart home device"""
        try:
            if request.device_id:
                result = await self.engine.home_automation.control_device(
                    request.device_id, 
                    request.action, 
                    request.parameters
                )
                return result
            else:
                # Control all devices of a type
                results = []
                for device_id, device in self.engine.home_automation.devices.items():
                    if request.action in ["turn_on", "turn_off"]:
                        result = await self.engine.home_automation.control_device(
                            device_id, 
                            request.action, 
                            request.parameters
                        )
                        results.append(result)
                        
                return {
                    "status": "success",
                    "action": request.action,
                    "results": results
                }
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def create_scene(self, request: SceneRequest) -> Dict:
        """Create a smart home scene"""
        try:
            self.engine.home_automation.create_scene(request.scene_name, request.devices_config)
            return {
                "status": "success",
                "scene_name": request.scene_name,
                "message": f"Scene '{request.scene_name}' created"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def activate_scene(self, scene_name: str) -> Dict:
        """Activate a smart home scene"""
        try:
            results = await self.engine.home_automation.activate_scene(scene_name)
            return {
                "status": "success",
                "scene_name": scene_name,
                "results": results
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Task automation endpoints
    async def create_task(self, request: TaskRequest) -> Dict:
        """Create an automation task"""
        try:
            task = AutomationTask(
                id=str(uuid.uuid4()),
                name=request.name,
                action=request.action,
                parameters=request.parameters,
                trigger=request.trigger,
                priority=TaskPriority[request.priority.upper()]
            )
            
            task_id = await self.engine.task_executor.create_task(task)
            
            return {
                "status": "success",
                "task_id": task_id,
                "message": f"Task '{task.name}' created"
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def list_tasks(self, status: Optional[str] = None) -> Dict:
        """List automation tasks"""
        try:
            task_status = TaskStatus[status.upper()] if status else None
            tasks = self.engine.task_executor.list_tasks(task_status)
            
            return {
                "tasks": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "action": t.action,
                        "status": t.status.value,
                        "priority": t.priority.value,
                        "created_at": t.created_at.isoformat(),
                        "executed_at": t.executed_at.isoformat() if t.executed_at else None
                    }
                    for t in tasks
                ],
                "count": len(tasks)
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    async def get_task(self, task_id: str) -> Dict:
        """Get task details"""
        task = self.engine.task_executor.get_task_status(task_id)
        if task:
            return {
                "id": task.id,
                "name": task.name,
                "action": task.action,
                "parameters": task.parameters,
                "status": task.status.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "executed_at": task.executed_at.isoformat() if task.executed_at else None,
                "result": task.result,
                "error": task.error
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    async def execute_task(self, task_id: str, background_tasks: BackgroundTasks) -> Dict:
        """Execute a task"""
        task = self.engine.task_executor.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
            
        # Execute in background
        background_tasks.add_task(self.engine.task_executor.execute_task, task_id)
        
        return {
            "status": "started",
            "task_id": task_id,
            "message": f"Task '{task.name}' execution started"
        }
        
    async def cancel_task(self, task_id: str) -> Dict:
        """Cancel a task"""
        if self.engine.task_executor.cancel_task(task_id):
            return {
                "status": "success",
                "task_id": task_id,
                "message": "Task cancelled"
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    async def create_task_plan(self, request: AutomationCommandRequest) -> Dict:
        """Create a task plan from a goal"""
        try:
            tasks = self.engine.task_executor.create_task_plan(request.command, request.context)
            
            # Create all tasks
            created_tasks = []
            for task in tasks:
                task_id = await self.engine.task_executor.create_task(task)
                created_tasks.append({
                    "id": task_id,
                    "name": task.name,
                    "action": task.action
                })
                
            return {
                "status": "success",
                "goal": request.command,
                "tasks": created_tasks,
                "count": len(created_tasks)
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # General automation endpoint
    async def process_command(self, request: AutomationCommandRequest) -> Dict:
        """Process a natural language automation command"""
        try:
            result = await self.engine.process_command(request.command, request.context)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))