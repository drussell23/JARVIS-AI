"""
Vision System Integration Example
Shows how to use the optimized vision system with all Phase 0C features
"""

import asyncio
import os
from typing import Dict, Any

# Import the optimized components
from .optimized_vision_system import get_vision_system
from .realtime_vision_monitor import RealtimeVisionMonitor, TriggerCondition, TriggerType, MonitoringRegion
from .vision_query_optimizer import VisionQueryOptimizer, QueryType

# Example automation callbacks
async def handle_update_notification(region: str, analysis: Dict[str, Any]):
    """Handle software update notifications"""
    print(f"Update detected in {region}!")
    print(f"Details: {analysis.get('description', 'No details')}")
    
    # Could trigger:
    # - System notification
    # - Auto-update process
    # - Reminder to user
    # - Log to update tracking system

async def handle_app_switch(region: str, analysis: Dict[str, Any]):
    """Handle application switching"""
    apps = analysis.get('applications_mentioned', [])
    print(f"App switch detected: {apps}")
    
    # Could trigger:
    # - Workspace configuration change
    # - Tool palette updates
    # - Context-aware assistance

async def handle_security_alert(region: str, analysis: Dict[str, Any]):
    """Handle security concerns"""
    concerns = analysis.get('security_concerns', [])
    if concerns:
        print(f"SECURITY ALERT: {concerns}")
        # Could trigger immediate lockdown or notification

class VisionSystemExample:
    """Example implementation of the optimized vision system"""
    
    def __init__(self):
        # Get the singleton vision system
        self.vision_system = get_vision_system()
        
        # Initialize components
        self.monitor = RealtimeVisionMonitor(self.vision_system)
        self.query_optimizer = VisionQueryOptimizer()
        
        # Setup automation triggers
        self._setup_triggers()
    
    def _setup_triggers(self):
        """Setup automation triggers"""
        # Update detection trigger
        self.monitor.add_trigger(TriggerCondition(
            trigger_type=TriggerType.UPDATE_AVAILABLE,
            parameters={},
            callback=handle_update_notification,
            cooldown_seconds=300  # 5 minutes
        ))
        
        # App switching trigger
        self.monitor.add_trigger(TriggerCondition(
            trigger_type=TriggerType.APP_OPENS,
            parameters={"app": "chrome"},
            callback=handle_app_switch,
            cooldown_seconds=10
        ))
        
        # Security monitoring
        self.monitor.add_trigger(TriggerCondition(
            trigger_type=TriggerType.CUSTOM,
            parameters={
                "condition": lambda r, a: len(a.get('security_concerns', [])) > 0
            },
            callback=handle_security_alert,
            cooldown_seconds=60
        ))
    
    async def demonstrate_memory_efficient_analysis(self):
        """Demonstrate memory-efficient vision analysis"""
        print("\n=== Memory-Efficient Analysis Demo ===")
        
        # Analyze with different compression levels
        results = {}
        
        # High-quality text extraction
        print("1. Extracting text (high quality)...")
        results['text'] = await self.vision_system.analyze_screen(
            prompt="Extract all visible text",
            analysis_type="text",
            use_cache=True
        )
        
        # Medium-quality UI detection
        print("2. Detecting UI elements (medium quality)...")
        results['ui'] = await self.vision_system.analyze_screen(
            prompt="Identify clickable buttons and links",
            analysis_type="ui",
            use_cache=True
        )
        
        # Low-quality activity monitoring
        print("3. Monitoring activity (low quality, fast)...")
        results['activity'] = await self.vision_system.analyze_screen(
            prompt="What is the user doing?",
            analysis_type="activity",
            use_cache=True
        )
        
        # Show metrics
        metrics = self.vision_system.get_system_metrics()
        print(f"\nMemory usage: {metrics['cache']['memory_usage_mb']:.1f} MB")
        print(f"Cache hit rate: {metrics['cache']['cache_hit_rate']:.1%}")
        print(f"Compression savings: {metrics['cache']['compression_savings_mb']:.1f} MB")
        
        return results
    
    async def demonstrate_batch_processing(self):
        """Demonstrate batch processing for multiple regions"""
        print("\n=== Batch Processing Demo ===")
        
        # Define regions to analyze
        regions = [
            {"x": 0, "y": 0, "width": 500, "height": 100, "prompt": "What's in the top bar?"},
            {"x": 0, "y": 100, "width": 200, "height": 600, "prompt": "What's in the sidebar?"},
            {"x": 200, "y": 100, "width": 800, "height": 600, "prompt": "What's in the main content?"}
        ]
        
        print(f"Analyzing {len(regions)} regions in batch...")
        results = await self.vision_system.batch_analyze_regions(regions)
        
        for i, result in enumerate(results):
            print(f"\nRegion {i+1}: {result.get('description', 'No description')[:100]}...")
        
        return results
    
    async def demonstrate_query_optimization(self):
        """Demonstrate query optimization"""
        print("\n=== Query Optimization Demo ===")
        
        # Single optimized query
        optimized = self.query_optimizer.optimize_prompt(
            QueryType.UPDATE_DETECTION,
            max_tokens=30
        )
        print(f"Optimized update detection prompt: '{optimized.prompt}'")
        print(f"Expected tokens: {optimized.expected_tokens}")
        
        # Batch optimization
        queries = [
            (QueryType.TEXT_EXTRACTION, {"region": "menubar"}),
            (QueryType.TEXT_EXTRACTION, {"region": "statusbar"}),
            (QueryType.UI_NAVIGATION, {"target": "close button"}),
            (QueryType.UPDATE_DETECTION, {})
        ]
        
        batch_prompt = self.query_optimizer.batch_optimize_queries(queries)
        print(f"\nBatch optimized prompt:\n{batch_prompt}")
        
        # Token usage report
        optimized_queries = [
            self.query_optimizer.optimize_prompt(q[0], q[1]) for q in queries
        ]
        report = self.query_optimizer.get_token_usage_report(optimized_queries)
        print(f"\nToken usage report:")
        print(f"Total queries: {report['total_queries']}")
        print(f"Total estimated tokens: {report['total_estimated_tokens']}")
        print(f"Suggestions: {report['optimization_suggestions']}")
    
    async def demonstrate_realtime_monitoring(self):
        """Demonstrate real-time monitoring with automation"""
        print("\n=== Real-Time Monitoring Demo ===")
        
        # Add monitoring regions
        self.monitor.add_region(MonitoringRegion(
            name="notification_area",
            x=1000, y=0, width=400, height=100,
            sensitivity=0.03  # High sensitivity for notifications
        ))
        
        # Start monitoring
        print("Starting real-time monitoring for 30 seconds...")
        print("Try opening apps, showing notifications, or making changes!")
        
        await self.monitor.start_monitoring(interval=2.0)
        
        # Monitor for 30 seconds
        await asyncio.sleep(30)
        
        # Stop and show stats
        await self.monitor.stop_monitoring()
        
        stats = self.monitor.get_monitoring_stats()
        print(f"\nMonitoring stats:")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Changes detected: {stats['changes_detected']}")
        print(f"Triggers fired: {stats['triggers_fired']}")
        print(f"Average FPS: {stats['fps']:.1f}")
        
        # Show trigger stats
        trigger_stats = self.monitor.get_trigger_stats()
        for trigger in trigger_stats:
            if trigger['trigger_count'] > 0:
                print(f"\nTrigger '{trigger['type']}' fired {trigger['trigger_count']} times")
    
    async def demonstrate_multi_monitor(self):
        """Demonstrate multi-monitor support"""
        print("\n=== Multi-Monitor Support Demo ===")
        
        results = await self.vision_system.analyze_multi_monitor()
        
        for monitor_result in results:
            monitor_info = monitor_result.get('monitor_info', {})
            print(f"\nMonitor {monitor_info.get('index', 0) + 1}:")
            print(f"  Resolution: {monitor_info.get('resolution', 'Unknown')}")
            print(f"  Primary: {monitor_info.get('primary', False)}")
            print(f"  Content: {monitor_result.get('description', 'No description')[:100]}...")
    
    async def demonstrate_circuit_breaker(self):
        """Demonstrate circuit breaker protection"""
        print("\n=== Circuit Breaker Demo ===")
        
        if not self.vision_system.circuit_breaker:
            print("No API key configured - circuit breaker not available")
            return
        
        # Get current health
        health = self.vision_system.circuit_breaker.get_health_status()
        print(f"Circuit breaker state: {health['state']}")
        print(f"Is healthy: {health['is_healthy']}")
        
        # Simulate API failures (example only)
        print("\nNote: In production, the circuit breaker automatically")
        print("protects against API failures and switches to fallback mode")
    
    async def run_all_demos(self):
        """Run all demonstration features"""
        print("=== JARVIS Vision System Phase 0C Demo ===")
        print("Demonstrating optimized vision capabilities for 16GB RAM systems")
        
        # Run each demo
        await self.demonstrate_memory_efficient_analysis()
        await self.demonstrate_batch_processing()
        await self.demonstrate_query_optimization()
        await self.demonstrate_multi_monitor()
        await self.demonstrate_circuit_breaker()
        
        # Real-time monitoring last (interactive)
        await self.demonstrate_realtime_monitoring()
        
        # Cleanup
        print("\nCleaning up...")
        await self.vision_system.cleanup()
        
        # Final metrics
        final_metrics = self.vision_system.get_system_metrics()
        print("\nFinal system metrics:")
        print(f"Total analyses: {final_metrics['performance']['total_analyses']}")
        print(f"API calls: {final_metrics['performance']['api_analyses']}")
        print(f"Cached responses: {final_metrics['performance']['cached_analyses']}")
        print(f"Fallback analyses: {final_metrics['performance']['fallback_analyses']}")
        print(f"Average response time: {final_metrics['performance']['average_response_time']:.2f}s")

# Main execution
async def main():
    """Main entry point"""
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("WARNING: No ANTHROPIC_API_KEY found in environment")
        print("Vision features will run in fallback mode")
        print("Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print()
    
    # Create and run example
    example = VisionSystemExample()
    await example.run_all_demos()

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())