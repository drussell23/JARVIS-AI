#!/usr/bin/env python3
"""
JARVIS Voice Unlock Integration
==============================

Main entry point for JARVIS voice unlock system with ML optimization.

Enhanced Features (v2.0):
- Multi-factor authentication fusion
- LangGraph adaptive retry reasoning
- Anti-spoofing detection
- Progressive voice feedback
- Full authentication audit trail
"""

import os
import sys
import asyncio
import logging
import signal
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import click

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.voice_unlock.voice_unlock_integration import VoiceUnlockSystem, create_voice_unlock_system
from backend.voice_unlock.config import get_config, reset_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JARVISVoiceUnlock:
    """
    JARVIS Voice Unlock service with ML optimization
    """
    
    def __init__(self):
        self.system: Optional[VoiceUnlockSystem] = None
        self.config = get_config()
        self.running = False
        
    async def start(self):
        """Start JARVIS voice unlock service"""
        logger.info("🚀 Starting JARVIS Voice Unlock System...")
        
        # Show configuration
        logger.info(f"Configuration:")
        logger.info(f"  - Max Memory: {self.config.performance.max_memory_mb}MB")
        logger.info(f"  - Cache Size: {self.config.performance.cache_size_mb}MB")
        logger.info(f"  - Integration Mode: {self.config.system.integration_mode}")
        logger.info(f"  - Anti-spoofing: {self.config.security.anti_spoofing_level}")
        
        # Create and start system
        self.system = await create_voice_unlock_system()
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ JARVIS Voice Unlock is active")
        
        # Show initial status
        status = self.system.get_status()
        logger.info(f"System Status: {json.dumps(status, indent=2)}")
        
        # Start main loop
        await self._main_loop()
        
    async def _main_loop(self):
        """Main service loop"""
        while self.running:
            try:
                # Sleep for a bit
                await asyncio.sleep(1)
                
                # Periodic health check
                if hasattr(self.system, 'ml_system'):
                    health = self.system.ml_system._get_system_health_status()
                    
                    # Log warnings if needed
                    if health['memory_percent'] > 80:
                        logger.warning(f"High memory usage: {health['memory_percent']:.1f}%")
                    
                    if health['degraded_mode']:
                        logger.warning("System running in degraded mode")
                        
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
                
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.running = False
        
    async def stop(self):
        """Stop JARVIS voice unlock service"""
        logger.info("🛑 Stopping JARVIS Voice Unlock System...")
        
        if self.system:
            # Get final report
            try:
                report = self.system.ml_system.get_performance_report()
                logger.info(f"Final ML Performance Report:")
                logger.info(f"  - Models Loaded: {report['ml_performance']['models']['loaded']}")
                logger.info(f"  - Cache Hit Rate: {report['ml_performance']['cache']['hit_rate']:.1f}%")
                logger.info(f"  - Avg Load Time: {report['ml_performance']['models']['avg_load_time']:.2f}s")
                
                # Export diagnostics
                self.system.ml_system.export_diagnostics("jarvis_voice_unlock_diagnostics.json")
                
            except Exception as e:
                logger.error(f"Failed to generate final report: {e}")
                
            # Stop system
            await self.system.stop()
            
        self.running = False
        logger.info("✅ JARVIS Voice Unlock stopped")
        
    async def enroll_user(self, user_id: str):
        """Interactive user enrollment"""
        if not self.system:
            self.system = await create_voice_unlock_system()
            
        print(f"\n🎤 Voice Enrollment for {user_id}")
        print("=" * 50)
        print("You will be asked to speak 3-5 times for enrollment.")
        print("Please speak clearly and naturally.")
        print("\nSuggested phrases:")
        for phrase in self.config.enrollment.default_phrases:
            print(f"  - {phrase.replace('{user}', user_id)}")
        print()
        
        import sounddevice as sd
        
        samples = []
        for i in range(self.config.enrollment.min_samples):
            input(f"\nPress Enter to start recording sample {i+1}/{self.config.enrollment.min_samples}...")
            
            duration = 3.0
            print(f"🔴 Recording for {duration} seconds... Speak now!")
            
            audio = sd.rec(
                int(duration * self.config.audio.sample_rate), 
                samplerate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
                dtype='float32'
            )
            sd.wait()
            
            print("✅ Recording complete")
            samples.append(audio.flatten())
            
        # Enroll user
        print("\n⏳ Processing enrollment...")
        result = await self.system.enroll_user(user_id, samples)
        
        if result['success']:
            print(f"\n✅ Successfully enrolled {user_id}!")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Memory used: {result['memory_used_mb']:.1f}MB")
        else:
            print(f"\n❌ Enrollment failed: {result.get('error', 'Unknown error')}")
            
    async def test_authentication(self, user_id: Optional[str] = None):
        """Test voice authentication"""
        if not self.system:
            self.system = await create_voice_unlock_system()

        print("\n🔐 Voice Authentication Test")
        print("=" * 50)

        if user_id:
            print(f"Testing authentication for user: {user_id}")
        else:
            print("Testing authentication (user will be identified from voice)")

        print("\nSpeak your authentication phrase when ready...")
        print("(10 second timeout)")

        # Test authentication
        result = await self.system.authenticate_with_voice(timeout=10.0)

        print(f"\n{'✅' if result['authenticated'] else '❌'} Authentication Result:")
        print(f"   Authenticated: {result['authenticated']}")
        print(f"   User: {result.get('user_id', 'Unknown')}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Processing time: {result.get('processing_time', 0):.3f}s")

        if 'error' in result:
            print(f"   Error: {result['error']}")

        # Show system health
        if hasattr(self.system, 'ml_system'):
            health = self.system.ml_system._get_system_health_status()
            print(f"\n📊 System Health:")
            print(f"   Memory: {health['memory_percent']:.1f}%")
            print(f"   ML Memory: {health['ml_memory_mb']:.1f}MB")
            print(f"   Cache Size: {health['cache_size_mb']:.1f}MB")

    async def test_enhanced_authentication(
        self,
        user_id: Optional[str] = None,
        require_watch: bool = False,
        max_attempts: int = 3,
        use_adaptive: bool = True
    ):
        """Test enhanced voice authentication with multi-factor fusion."""
        if not self.system:
            self.system = await create_voice_unlock_system()

        print("\n🔐 Enhanced Voice Authentication Test (v2.0)")
        print("=" * 60)

        if user_id:
            self.system.authorized_user = user_id
            print(f"Testing authentication for user: {user_id}")
        else:
            print(f"Testing authentication for: {self.system.authorized_user}")

        print(f"\nEnhanced Features:")
        print(f"   • Multi-factor fusion: ✅")
        print(f"   • Adaptive retry: {'✅' if use_adaptive else '❌'}")
        print(f"   • Anti-spoofing: ✅")
        print(f"   • Apple Watch required: {'✅' if require_watch else '❌'}")
        print(f"   • Max attempts: {max_attempts}")

        print("\n🎤 Speak your authentication phrase when ready...")
        print("(10 second timeout)")

        # Test enhanced authentication
        result = await self.system.authenticate_enhanced(
            timeout=10.0,
            require_watch=require_watch,
            max_attempts=max_attempts,
            use_adaptive=use_adaptive
        )

        print(f"\n{'✅' if result['authenticated'] else '❌'} Enhanced Authentication Result:")
        print(f"   Authenticated: {result['authenticated']}")
        print(f"   User: {result.get('user_id', 'Unknown')}")
        print(f"   Overall Confidence: {result['confidence']:.1%}")
        print(f"   Voice Confidence: {result.get('voice_confidence', 0):.1%}")
        print(f"   Behavioral Confidence: {result.get('behavioral_confidence', 0):.1%}")
        print(f"   Context Confidence: {result.get('context_confidence', 0):.1%}")
        print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
        print(f"   Attempts used: {result.get('attempts', 1)}")

        if result.get('feedback'):
            print(f"\n💬 Voice Feedback: {result['feedback']}")

        if result.get('trace_id'):
            print(f"\n🔍 Trace ID: {result['trace_id']}")

        if result.get('threat_detected'):
            print(f"\n⚠️ Security Alert: {result['threat_detected']}")

        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")

        # Show detailed trace if available
        if result.get('trace_id') and hasattr(self.system, 'get_authentication_trace'):
            trace = self.system.get_authentication_trace(result['trace_id'])
            if trace:
                print(f"\n📋 Authentication Trace:")
                print(f"   Phases completed: {len(trace.get('phases', []))}")
                for phase in trace.get('phases', []):
                    status = '✅' if phase.get('success', False) else '❌'
                    print(f"      {status} {phase.get('phase')}: {phase.get('duration_ms', 0):.1f}ms")

        # Show cache statistics
        if hasattr(self.system, 'get_cache_stats'):
            cache_stats = self.system.get_cache_stats()
            if cache_stats.get('available', True):
                print(f"\n💾 Cache Statistics:")
                print(f"   Hit Rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
                print(f"   Cost Saved: ${cache_stats.get('cost_saved_usd', 0):.4f}")

        # Show system health
        if hasattr(self.system, 'ml_system') and self.system.ml_system:
            try:
                health = self.system.ml_system._get_system_health_status()
                print(f"\n📊 System Health:")
                print(f"   Memory: {health['memory_percent']:.1f}%")
                print(f"   ML Memory: {health['ml_memory_mb']:.1f}MB")
                print(f"   Cache Size: {health['cache_size_mb']:.1f}MB")
            except Exception:
                pass

    async def show_recent_authentications(self, limit: int = 10):
        """Show recent authentication attempts."""
        if not self.system:
            self.system = await create_voice_unlock_system()

        print(f"\n📜 Recent Authentication Attempts (last {limit})")
        print("=" * 60)

        if hasattr(self.system, 'get_recent_authentications'):
            auths = self.system.get_recent_authentications(limit=limit)

            if not auths:
                print("No recent authentication attempts found.")
                return

            for i, auth in enumerate(auths, 1):
                decision = auth.get('decision', 'unknown')
                emoji = '✅' if decision == 'authenticated' else '❌'
                timestamp = auth.get('timestamp', 'unknown')
                confidence = auth.get('verification', {}).get('fused_confidence', 0)
                trace_id = auth.get('trace_id', 'N/A')

                print(f"\n{i}. {emoji} {decision.upper()}")
                print(f"   Time: {timestamp}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Trace: {trace_id[:16]}...")
                if auth.get('security', {}).get('threat_detected') != 'none':
                    print(f"   ⚠️ Threat: {auth['security']['threat_detected']}")
        else:
            print("Authentication history not available.")
            
    async def show_status(self):
        """Show system status and statistics"""
        if not self.system:
            print("❌ System not running")
            return
            
        status = self.system.get_status()
        ml_report = self.system.ml_system.get_performance_report()
        
        print("\n📊 JARVIS Voice Unlock Status")
        print("=" * 50)
        print(f"System State:")
        print(f"   Active: {status['is_active']}")
        print(f"   Locked: {status['is_locked']}")
        print(f"   Current User: {status['current_user'] or 'None'}")
        print(f"   Last Auth: {status['last_auth_time'] or 'Never'}")
        
        print(f"\nML Performance:")
        print(f"   Models Loaded: {ml_report['ml_performance']['models']['loaded']}")
        print(f"   Total Loaded: {ml_report['ml_performance']['models']['total_loaded']}")
        print(f"   Cache Hit Rate: {ml_report['ml_performance']['cache']['hit_rate']:.1f}%")
        print(f"   Avg Load Time: {ml_report['ml_performance']['models']['avg_load_time']:.3f}s")
        
        print(f"\nSystem Health:")
        health = ml_report['system_health']
        print(f"   Healthy: {health['healthy']}")
        print(f"   Memory: {health['memory_percent']:.1f}%")
        print(f"   CPU: {health['cpu_percent']:.1f}%")
        print(f"   ML Memory: {health['ml_memory_mb']:.1f}MB")
        
        print(f"\nRecommendations:")
        for rec in ml_report['recommendations']:
            print(f"   - {rec}")


# CLI Commands
@click.group()
def cli():
    """JARVIS Voice Unlock System CLI"""
    pass


@cli.command()
def start():
    """Start JARVIS voice unlock service"""
    service = JARVISVoiceUnlock()
    
    async def run():
        try:
            await service.start()
        except KeyboardInterrupt:
            pass
        finally:
            await service.stop()
            
    asyncio.run(run())


@cli.command()
@click.argument('user_id')
def enroll(user_id: str):
    """Enroll a new user"""
    service = JARVISVoiceUnlock()
    
    async def run():
        try:
            await service.enroll_user(user_id)
        finally:
            if service.system:
                await service.system.stop()
                
    asyncio.run(run())


@cli.command()
@click.option('--user', '-u', help='User ID to test (optional)')
def test(user: Optional[str]):
    """Test voice authentication (basic)"""
    service = JARVISVoiceUnlock()

    async def run():
        try:
            await service.test_authentication(user)
        finally:
            if service.system:
                await service.system.stop()

    asyncio.run(run())


@cli.command()
@click.option('--user', '-u', help='User ID to test (optional)')
@click.option('--watch', '-w', is_flag=True, help='Require Apple Watch proximity')
@click.option('--attempts', '-a', default=3, help='Max retry attempts')
@click.option('--no-adaptive', is_flag=True, help='Disable adaptive retry reasoning')
def test_enhanced(user: Optional[str], watch: bool, attempts: int, no_adaptive: bool):
    """Test enhanced voice authentication (v2.0 with multi-factor fusion)"""
    service = JARVISVoiceUnlock()

    async def run():
        try:
            await service.test_enhanced_authentication(
                user_id=user,
                require_watch=watch,
                max_attempts=attempts,
                use_adaptive=not no_adaptive
            )
        finally:
            if service.system:
                await service.system.stop()

    asyncio.run(run())


@cli.command()
@click.option('--limit', '-l', default=10, help='Number of recent attempts to show')
def history(limit: int):
    """Show recent authentication attempts"""
    service = JARVISVoiceUnlock()

    async def run():
        try:
            await service.show_recent_authentications(limit=limit)
        finally:
            if service.system:
                await service.system.stop()

    asyncio.run(run())


@cli.command()
def status():
    """Show system status"""
    service = JARVISVoiceUnlock()
    
    async def run():
        service.system = await create_voice_unlock_system()
        await service.show_status()
        await service.system.stop()
        
    asyncio.run(run())


@cli.command()
def configure():
    """Interactive configuration"""
    config = get_config()
    
    print("\n⚙️  JARVIS Voice Unlock Configuration")
    print("=" * 50)
    
    # Memory settings
    print("\nMemory Settings:")
    max_memory = click.prompt(
        "Maximum memory for ML models (MB)", 
        default=config.performance.max_memory_mb, 
        type=int
    )
    cache_size = click.prompt(
        "Cache size (MB)", 
        default=config.performance.cache_size_mb, 
        type=int
    )
    
    # Security settings
    print("\nSecurity Settings:")
    anti_spoofing = click.prompt(
        "Anti-spoofing level",
        default=config.security.anti_spoofing_level,
        type=click.Choice(['low', 'medium', 'high'])
    )
    
    # Integration settings
    print("\nIntegration Settings:")
    integration_mode = click.prompt(
        "Integration mode",
        default=config.system.integration_mode,
        type=click.Choice(['screensaver', 'pam', 'both'])
    )
    
    # Update configuration
    updates = {
        'performance': {
            'max_memory_mb': max_memory,
            'cache_size_mb': cache_size
        },
        'security': {
            'anti_spoofing_level': anti_spoofing
        },
        'system': {
            'integration_mode': integration_mode
        }
    }
    
    config.update_from_dict(updates)
    config.save_to_file()
    
    print("\n✅ Configuration saved!")


if __name__ == '__main__':
    cli()