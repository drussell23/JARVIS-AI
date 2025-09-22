#!/usr/bin/env python3
"""
JARVIS Voice Unlock Integration
==============================

Main entry point for JARVIS voice unlock system with ML optimization.
"""

import os
import sys
import asyncio
import logging
import signal
import json
from pathlib import Path
from typing import Optional, Dict, Any
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
    """Test voice authentication"""
    service = JARVISVoiceUnlock()
    
    async def run():
        try:
            await service.test_authentication(user)
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