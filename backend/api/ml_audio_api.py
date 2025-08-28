#!/usr/bin/env python3
"""
ML Audio API for JARVIS
Provides endpoints for ML-enhanced audio error handling and recovery
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from audio.ml_audio_manager import (
    AudioEvent,
    get_audio_manager,
    AudioPatternLearner
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/audio/ml", tags=["ML Audio"])

# WebSocket connection manager
class AudioWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_contexts: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_contexts[websocket] = {
            'connected_at': datetime.now(),
            'events': []
        }
        logger.info(f"New ML audio WebSocket connection: {len(self.active_connections)} total")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.client_contexts:
            del self.client_contexts[websocket]
        logger.info(f"ML audio WebSocket disconnected: {len(self.active_connections)} remaining")
    
    async def broadcast(self, message: dict):
        """Broadcast to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")

# Initialize WebSocket manager
ws_manager = AudioWebSocketManager()

# Request models
class AudioErrorRequest(BaseModel):
    error_code: str
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    timestamp: Optional[str] = None
    session_duration: Optional[int] = None
    retry_count: Optional[int] = 0
    permission_state: Optional[str] = None
    user_agent: Optional[str] = None
    audio_context_state: Optional[str] = None
    previous_errors: Optional[List[Dict[str, Any]]] = []

class AudioPredictionRequest(BaseModel):
    browser: str
    time_of_day: int
    day_of_week: int
    error_history: List[Dict[str, Any]]
    session_duration: int
    permission_state: str

class AudioTelemetryRequest(BaseModel):
    event: str
    data: Dict[str, Any]
    timestamp: str

class AudioConfigUpdate(BaseModel):
    enable_ml: Optional[bool] = None
    auto_recovery: Optional[bool] = None
    max_retries: Optional[int] = None
    retry_delays: Optional[List[int]] = None
    anomaly_threshold: Optional[float] = None
    prediction_threshold: Optional[float] = None

# API Endpoints
@router.get("/config")
async def get_ml_config():
    """Get ML audio configuration"""
    audio_manager = get_audio_manager()
    return JSONResponse(content=audio_manager.config)

@router.post("/config")
async def update_ml_config(config: AudioConfigUpdate):
    """Update ML audio configuration"""
    audio_manager = get_audio_manager()
    
    # Update configuration
    update_dict = config.dict(exclude_unset=True)
    audio_manager.config.update(update_dict)
    
    # Save to environment
    for key, value in update_dict.items():
        os.environ[f'JARVIS_AUDIO_{key.upper()}'] = str(value)
    
    logger.info(f"Updated ML audio config: {update_dict}")
    
    return JSONResponse(content={
        "success": True,
        "updated": update_dict,
        "config": audio_manager.config
    })

@router.post("/error")
async def handle_audio_error(request: AudioErrorRequest):
    """Handle audio error with ML-driven recovery strategy"""
    audio_manager = get_audio_manager()
    
    # Create context from request
    context = request.dict()
    
    # Handle the error
    result = await audio_manager.handle_error(request.error_code, context)
    
    # Broadcast to WebSocket clients
    await ws_manager.broadcast({
        "type": "error_handled",
        "error_code": request.error_code,
        "result": result
    })
    
    return JSONResponse(content={
        "success": result.get("success", False),
        "strategy": result,
        "ml_confidence": result.get("ml_confidence", 0)
    })

@router.post("/predict")
async def predict_audio_issue(request: AudioPredictionRequest):
    """Predict potential audio issues"""
    audio_manager = get_audio_manager()
    
    # Create context
    context = request.dict()
    
    # Get prediction
    probability = audio_manager.pattern_learner.predict_error_probability(context)
    
    # Determine recommended action based on probability
    recommended_action = None
    if probability > 0.8:
        recommended_action = "preemptive_permission_check"
    elif probability > 0.6:
        recommended_action = "show_permission_reminder"
    elif probability > 0.4:
        recommended_action = "prepare_fallback_mode"
    
    prediction_result = {
        "probability": probability,
        "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low",
        "recommended_action": recommended_action,
        "factors": _analyze_risk_factors(context, audio_manager)
    }
    
    # Broadcast prediction
    await ws_manager.broadcast({
        "type": "prediction",
        "prediction": prediction_result
    })
    
    return JSONResponse(content=prediction_result)

@router.post("/telemetry")
async def receive_telemetry(request: AudioTelemetryRequest):
    """Receive telemetry data from client"""
    logger.info(f"Audio telemetry: {request.event} - {request.data}")
    
    # Process telemetry
    if request.event == "recovery":
        # Learn from successful recovery
        audio_manager = get_audio_manager()
        event = AudioEvent(
            timestamp=datetime.fromisoformat(request.timestamp.replace('Z', '+00:00')),
            event_type="recovery",
            resolution=request.data.get("method"),
            browser=request.data.get("browser"),
            context=request.data
        )
        await audio_manager.pattern_learner.learn_from_event(event)
    
    return JSONResponse(content={"success": True})

@router.get("/metrics")
async def get_ml_metrics():
    """Get ML audio system metrics"""
    audio_manager = get_audio_manager()
    metrics = audio_manager.get_metrics()
    
    # Add additional insights
    metrics["insights"] = _generate_insights(metrics, audio_manager)
    
    return JSONResponse(content=metrics)

@router.get("/patterns")
async def get_audio_patterns():
    """Get learned audio error patterns"""
    audio_manager = get_audio_manager()
    pattern_learner = audio_manager.pattern_learner
    
    # Summarize patterns
    patterns = {
        "error_patterns": {},
        "success_patterns": {},
        "recovery_strategies": {},
        "anomalies": []
    }
    
    # Error patterns by type
    for error_code, events in pattern_learner.error_patterns.items():
        patterns["error_patterns"][error_code] = {
            "count": len(events),
            "browsers": _count_browsers(events),
            "peak_hours": _find_peak_hours(events),
            "avg_retry_count": _avg_retry_count(events)
        }
    
    # Success patterns
    for browser, events in pattern_learner.success_patterns.items():
        patterns["success_patterns"][browser] = {
            "count": len(events),
            "success_rate": len(events) / max(len(pattern_learner.event_history), 1),
            "avg_resolution_time": _avg_resolution_time(events)
        }
    
    # Top recovery strategies
    strategies = audio_manager.strategy_history
    strategy_counts = {}
    for entry in strategies:
        strategy = entry['strategy']
        if strategy not in strategy_counts:
            strategy_counts[strategy] = {'total': 0, 'success': 0}
        strategy_counts[strategy]['total'] += 1
        if entry['success']:
            strategy_counts[strategy]['success'] += 1
    
    patterns["recovery_strategies"] = {
        strategy: {
            "usage_count": counts['total'],
            "success_rate": counts['success'] / max(counts['total'], 1)
        }
        for strategy, counts in strategy_counts.items()
    }
    
    return JSONResponse(content=patterns)

@router.websocket("/stream")
async def ml_audio_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time ML audio updates"""
    await ws_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to ML Audio System",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send current metrics
        audio_manager = get_audio_manager()
        metrics = audio_manager.get_metrics()
        await websocket.send_json({
            "type": "metrics",
            "metrics": metrics
        })
        
        # Keep connection alive
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get("type") == "telemetry":
                # Process telemetry
                event_data = data.get("data", {})
                event = AudioEvent(
                    timestamp=datetime.now(),
                    event_type=data.get("event"),
                    browser=event_data.get("browser"),
                    context=event_data
                )
                await audio_manager.pattern_learner.learn_from_event(event)
                
            elif data.get("type") == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# Helper functions
def _analyze_risk_factors(context: Dict[str, Any], audio_manager) -> List[str]:
    """Analyze risk factors for audio issues"""
    factors = []
    
    # Check error history
    error_count = len(context.get('error_history', []))
    if error_count > 3:
        factors.append(f"High error frequency ({error_count} recent errors)")
    
    # Check time patterns
    hour = context.get('time_of_day', 0)
    if hour < 6 or hour > 22:
        factors.append("Off-peak hours (higher risk)")
    
    # Check permission state
    if context.get('permission_state') == 'denied':
        factors.append("Permission previously denied")
    
    # Check session duration
    session_duration = context.get('session_duration', 0)
    if session_duration < 5000:  # Less than 5 seconds
        factors.append("New session (permission not established)")
    
    return factors

def _generate_insights(metrics: Dict[str, Any], audio_manager) -> List[str]:
    """Generate insights from metrics"""
    insights = []
    
    # Success rate insight
    success_rate = metrics.get('success_rate', 0)
    if success_rate < 0.5:
        insights.append("Low recovery success rate - consider updating strategies")
    elif success_rate > 0.8:
        insights.append("High recovery success rate - ML strategies working well")
    
    # Strategy effectiveness
    strategy_rates = metrics.get('strategy_success_rates', {})
    if strategy_rates:
        best_strategy = max(strategy_rates.items(), key=lambda x: x[1])
        insights.append(f"Most effective strategy: {best_strategy[0]} ({best_strategy[1]:.0%} success)")
    
    # ML accuracy
    ml_accuracy = metrics.get('ml_model_accuracy', 0)
    if ml_accuracy > 0.7:
        insights.append(f"ML predictions are accurate ({ml_accuracy:.0%})")
    elif ml_accuracy < 0.3:
        insights.append("ML model needs more training data")
    
    return insights

def _count_browsers(events: List[AudioEvent]) -> Dict[str, int]:
    """Count events by browser"""
    browser_counts = {}
    for event in events:
        browser = event.browser or 'unknown'
        browser_counts[browser] = browser_counts.get(browser, 0) + 1
    return browser_counts

def _find_peak_hours(events: List[AudioEvent]) -> List[int]:
    """Find peak hours for events"""
    hour_counts = {}
    for event in events:
        hour = event.timestamp.hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    # Get top 3 hours
    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
    return [hour for hour, count in sorted_hours[:3]]

def _avg_retry_count(events: List[AudioEvent]) -> float:
    """Calculate average retry count"""
    retry_counts = []
    for event in events:
        if event.context and 'retry_count' in event.context:
            retry_counts.append(event.context['retry_count'])
    
    return np.mean(retry_counts) if retry_counts else 0

def _avg_resolution_time(events: List[AudioEvent]) -> float:
    """Calculate average resolution time"""
    times = []
    for event in events:
        if event.duration_ms:
            times.append(event.duration_ms)
    
    return np.mean(times) if times else 0

# Register router
__all__ = ['router']