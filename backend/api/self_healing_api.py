"""
Self-Healing API endpoints for managing Rust component recovery.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class SelfHealingStatus(BaseModel):
    """Self-healing system status"""
    running: bool
    total_fix_attempts: int
    success_rate: float
    last_successful_build: Optional[str]
    retry_counts: Dict[str, int]
    recent_fixes: List[Dict[str, Any]]

class DiagnoseResponse(BaseModel):
    """Diagnosis result"""
    issue_type: str
    details: Dict[str, Any]
    recommended_fix: str
    can_auto_fix: bool

class FixResponse(BaseModel):
    """Fix attempt result"""
    success: bool
    issue_type: str
    strategy_used: str
    error: Optional[str] = None

@router.get("/status", response_model=SelfHealingStatus)
async def get_self_healing_status():
    """Get the current status of the self-healing system."""
    try:
        from vision.rust_self_healer import get_self_healer
        healer = get_self_healer()
        health_report = healer.get_health_report()
        
        return SelfHealingStatus(
            running=health_report.get('running', False),
            total_fix_attempts=health_report.get('total_fix_attempts', 0),
            success_rate=health_report.get('success_rate', 0.0),
            last_successful_build=health_report.get('last_successful_build'),
            retry_counts=health_report.get('retry_counts', {}),
            recent_fixes=health_report.get('recent_fixes', [])
        )
    except Exception as e:
        logger.error(f"Error getting self-healing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_rust_issues():
    """Run a manual diagnosis of Rust component issues."""
    try:
        from vision.rust_self_healer import get_self_healer
        healer = get_self_healer()
        
        # Run diagnosis
        issue_type, details = await healer._diagnose_issue()
        
        # Determine fix strategy
        strategy = healer._determine_fix_strategy(issue_type, details)
        
        # Check if it can be auto-fixed
        can_auto_fix = strategy.value not in ['retry_later', 'install_rust']
        
        return DiagnoseResponse(
            issue_type=issue_type.value,
            details=details,
            recommended_fix=strategy.value,
            can_auto_fix=can_auto_fix
        )
    except Exception as e:
        logger.error(f"Error diagnosing Rust issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fix", response_model=FixResponse)
async def fix_rust_components():
    """Attempt to fix Rust component issues."""
    try:
        from vision.rust_self_healer import get_self_healer
        healer = get_self_healer()
        
        # Run diagnosis and fix
        success = await healer.diagnose_and_fix()
        
        # Get the last fix attempt details
        fix_history = healer._fix_history
        if fix_history:
            last_fix = fix_history[-1]
            return FixResponse(
                success=success,
                issue_type=last_fix.get('issue', 'unknown'),
                strategy_used=last_fix.get('strategy', 'unknown'),
                error=None if success else "Fix failed - check logs for details"
            )
        else:
            return FixResponse(
                success=success,
                issue_type="unknown",
                strategy_used="unknown",
                error="No fix history available"
            )
    except Exception as e:
        logger.error(f"Error fixing Rust components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/force-check")
async def force_component_check():
    """Force an immediate check of all components."""
    try:
        from vision.dynamic_component_loader import get_component_loader
        loader = get_component_loader()
        
        changes = await loader.force_check()
        
        return {
            "success": True,
            "changes": {k.value: v for k, v in changes.items()} if changes else {},
            "message": f"Found {len(changes)} component changes" if changes else "No changes detected"
        }
    except Exception as e:
        logger.error(f"Error forcing component check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/component-status")
async def get_component_status():
    """Get the status of all dynamic components."""
    try:
        from vision.dynamic_component_loader import get_component_loader
        loader = get_component_loader()
        
        status = loader.get_status()
        
        # Transform the status for better readability
        formatted_status = {
            "loader_running": status['running'],
            "check_interval_seconds": status['check_interval'],
            "components": {}
        }
        
        for comp_name, comp_info in status['components'].items():
            formatted_status['components'][comp_name] = {
                "active_implementation": comp_info['active']['type'] if comp_info['active'] else None,
                "available_implementations": {
                    impl_name: {
                        "available": impl_info['available'],
                        "performance_score": impl_info['performance_score'],
                        "error_count": impl_info['error_count']
                    }
                    for impl_name, impl_info in comp_info['implementations'].items()
                }
            }
        
        return formatted_status
    except Exception as e:
        logger.error(f"Error getting component status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-build")
async def clean_and_rebuild_rust():
    """Clean and rebuild Rust components."""
    try:
        from vision.rust_self_healer import get_self_healer
        healer = get_self_healer()
        
        # Clean build artifacts
        await healer._clean_build_artifacts()
        
        # Reset cargo cache
        await healer._reset_cargo_cache()
        
        # Rebuild
        success = await healer._build_rust_components()
        
        return {
            "success": success,
            "message": "Clean build completed successfully" if success else "Clean build failed - check logs"
        }
    except Exception as e:
        logger.error(f"Error during clean build: {e}")
        raise HTTPException(status_code=500, detail=str(e))