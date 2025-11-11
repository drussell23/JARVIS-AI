#!/usr/bin/env python3
"""
🎤🧠 Voice Memory Agent - JARVIS's Persistent Voice Recognition Memory System

A self-aware, memory-persistent AI agent that ensures JARVIS never forgets your voice.

Features:
- Persistent voice memory across restarts
- Automatic freshness monitoring on startup
- Integrates with JARVISLearningDatabase for long-term storage
- ML-based voice pattern recognition and recall
- Memory-aware voice profile management
- Adaptive learning from every interaction
- Predictive voice degradation detection

This agent runs automatically when JARVIS starts and maintains voice recognition accuracy
by managing sample freshness, triggering automatic updates, and ensuring voice profiles
remain current and accurate.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.learning_database import get_learning_database
from voice.speaker_verification_service import get_speaker_verification_service

logger = logging.getLogger(__name__)


class VoiceMemoryAgent:
    """
    Intelligent agent for persistent voice memory management

    This agent ensures JARVIS maintains accurate voice recognition by:
    1. Loading voice profiles on startup
    2. Checking sample freshness automatically
    3. Triggering refresh recommendations when needed
    4. Storing all voice interactions for continuous learning
    5. Maintaining voice pattern memory across sessions
    """

    def __init__(self):
        self.learning_db = None
        self.speaker_service = None

        # Memory state
        self.voice_memory = {}  # In-memory cache of voice characteristics
        self.last_interaction = {}  # Last interaction time per speaker
        self.interaction_count = {}  # Interaction counter per speaker

        # Freshness management
        self.freshness_check_interval = 24 * 3600  # Check every 24 hours
        self.last_freshness_check = None
        self.auto_refresh_threshold = 0.6  # Trigger refresh below 60% freshness

        # Learning state
        self.learning_enabled = True
        self.samples_since_update = {}  # Track samples since last profile update
        self.update_threshold = 10  # Update profile every 10 samples

        # Memory persistence
        self.memory_file = Path.home() / '.jarvis' / 'voice_memory.json'

        # Configuration
        self.config = {
            'auto_freshness_check': True,
            'auto_profile_update': True,
            'continuous_learning': True,
            'memory_persistence': True,
            'proactive_notifications': True
        }

    async def initialize(self):
        """Initialize the voice memory agent"""
        logger.info("🧠 Initializing Voice Memory Agent...")

        # Initialize database connection
        self.learning_db = await get_learning_database()

        # Initialize speaker verification service
        self.speaker_service = await get_speaker_verification_service()

        # Load persistent memory
        await self._load_persistent_memory()

        # Load voice profiles into memory
        await self._load_voice_profiles()

        logger.info("✅ Voice Memory Agent initialized")

    async def _load_persistent_memory(self):
        """Load persistent voice memory from disk"""
        try:
            if self.memory_file.exists():
                import json
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)

                self.voice_memory = data.get('voice_memory', {})
                self.last_interaction = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get('last_interaction', {}).items()
                }
                self.interaction_count = data.get('interaction_count', {})
                self.last_freshness_check = (
                    datetime.fromisoformat(data['last_freshness_check'])
                    if data.get('last_freshness_check') else None
                )

                logger.info(f"📂 Loaded voice memory for {len(self.voice_memory)} speakers")
        except Exception as e:
            logger.warning(f"Could not load persistent memory: {e}")

    async def _save_persistent_memory(self):
        """Save voice memory to disk"""
        try:
            import json
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'voice_memory': self.voice_memory,
                'last_interaction': {
                    k: v.isoformat()
                    for k, v in self.last_interaction.items()
                },
                'interaction_count': self.interaction_count,
                'last_freshness_check': (
                    self.last_freshness_check.isoformat()
                    if self.last_freshness_check else None
                ),
                'timestamp': datetime.now().isoformat()
            }

            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("💾 Voice memory saved to disk")
        except Exception as e:
            logger.error(f"Failed to save persistent memory: {e}")

    async def _load_voice_profiles(self):
        """Load all voice profiles into memory"""
        try:
            profiles = await self.learning_db.get_all_speaker_profiles()

            for profile in profiles:
                speaker_name = profile.get('speaker_name')
                if speaker_name:
                    # Cache voice characteristics in memory
                    self.voice_memory[speaker_name] = {
                        'speaker_id': profile.get('speaker_id'),
                        'total_samples': profile.get('total_samples', 0),
                        'last_trained': profile.get('last_trained'),
                        'confidence': profile.get('avg_confidence', 0.0),
                        'loaded_at': datetime.now().isoformat()
                    }

                    # Initialize interaction tracking
                    if speaker_name not in self.interaction_count:
                        self.interaction_count[speaker_name] = 0

            logger.info(f"🎤 Loaded {len(self.voice_memory)} voice profiles into memory")

        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")

    async def startup_check(self) -> Dict:
        """
        Perform startup check - runs automatically when JARVIS starts

        Returns:
            Dict with status and recommendations
        """
        logger.info("🔍 Performing voice memory startup check...")

        results = {
            'status': 'healthy',
            'checks_performed': [],
            'recommendations': [],
            'actions_taken': [],
            'freshness': {}
        }

        # Check 1: Voice profiles loaded
        if not self.voice_memory:
            results['status'] = 'warning'
            results['recommendations'].append({
                'priority': 'HIGH',
                'action': 'No voice profiles found',
                'suggestion': 'Run voice enrollment: python backend/voice/enroll_voice.py'
            })
        else:
            results['checks_performed'].append('Voice profiles loaded')

        # Check 2: Sample freshness
        if self.config['auto_freshness_check']:
            should_check = (
                self.last_freshness_check is None or
                (datetime.now() - self.last_freshness_check).total_seconds() > self.freshness_check_interval
            )

            if should_check:
                freshness_results = await self._check_voice_freshness()
                results['freshness'] = freshness_results
                results['checks_performed'].append('Sample freshness checked')

                # Trigger actions based on freshness
                for speaker, metrics in freshness_results.items():
                    if metrics.get('freshness_score', 1.0) < self.auto_refresh_threshold:
                        results['status'] = 'needs_attention'
                        results['recommendations'].append({
                            'priority': 'HIGH',
                            'action': f'Voice samples for {speaker} need refresh',
                            'freshness': f"{metrics['freshness_score']:.1%}",
                            'suggestion': f'Record 10-30 new samples for {speaker}'
                        })

                self.last_freshness_check = datetime.now()

        # Check 3: Memory-profile sync
        sync_result = await self._sync_memory_with_profiles()
        results['checks_performed'].append(f'Memory sync: {sync_result}')

        # Save state
        await self._save_persistent_memory()

        # Log summary
        logger.info(f"✅ Startup check complete: {results['status']}")
        if results['recommendations']:
            logger.info(f"💡 {len(results['recommendations'])} recommendations")

        return results

    async def _check_voice_freshness(self) -> Dict:
        """Check voice sample freshness for all speakers"""
        freshness_results = {}

        try:
            for speaker_name in self.voice_memory.keys():
                # Get freshness report from database
                report = await self.learning_db.get_sample_freshness_report(speaker_name)

                if 'error' not in report:
                    # Calculate overall freshness score
                    age_dist = report.get('age_distribution', {})
                    total_samples = sum(d['count'] for d in age_dist.values())

                    # Simple freshness scoring
                    weights = {
                        '0-7 days': 1.0,
                        '8-14 days': 0.8,
                        '15-30 days': 0.6,
                        '31-60 days': 0.3,
                        '60+ days': 0.1
                    }

                    if total_samples > 0:
                        weighted_sum = sum(
                            age_dist.get(bracket, {}).get('count', 0) * weight
                            for bracket, weight in weights.items()
                        )
                        freshness_score = weighted_sum / total_samples
                    else:
                        freshness_score = 0.0

                    freshness_results[speaker_name] = {
                        'total_samples': total_samples,
                        'freshness_score': freshness_score,
                        'recommendations': report.get('recommendations', [])
                    }

                    # Update memory
                    self.voice_memory[speaker_name]['freshness'] = freshness_score
                    self.voice_memory[speaker_name]['last_checked'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Freshness check failed: {e}")

        return freshness_results

    async def _sync_memory_with_profiles(self) -> str:
        """Synchronize in-memory state with database profiles"""
        try:
            profiles = await self.learning_db.get_all_speaker_profiles()

            synced = 0
            for profile in profiles:
                speaker_name = profile.get('speaker_name')
                if speaker_name:
                    # Update memory with latest profile data
                    if speaker_name in self.voice_memory:
                        self.voice_memory[speaker_name].update({
                            'total_samples': profile.get('total_samples', 0),
                            'last_trained': profile.get('last_trained'),
                            'confidence': profile.get('avg_confidence', 0.0),
                            'synced_at': datetime.now().isoformat()
                        })
                        synced += 1

            return f"Synced {synced} profiles"

        except Exception as e:
            logger.error(f"Memory sync failed: {e}")
            return "Sync failed"

    async def record_interaction(self, speaker_name: str, confidence: float, verified: bool):
        """
        Record a voice interaction for memory and learning

        Args:
            speaker_name: Speaker who interacted
            confidence: Verification confidence
            verified: Whether verification succeeded
        """
        # Update interaction tracking
        self.last_interaction[speaker_name] = datetime.now()
        self.interaction_count[speaker_name] = self.interaction_count.get(speaker_name, 0) + 1

        # Update memory
        if speaker_name in self.voice_memory:
            self.voice_memory[speaker_name]['last_interaction'] = datetime.now().isoformat()
            self.voice_memory[speaker_name]['interaction_count'] = self.interaction_count[speaker_name]
            self.voice_memory[speaker_name]['recent_confidence'] = confidence

        # Track samples for profile update
        if verified and self.config['auto_profile_update']:
            self.samples_since_update[speaker_name] = self.samples_since_update.get(speaker_name, 0) + 1

            # Trigger profile update if threshold reached
            if self.samples_since_update[speaker_name] >= self.update_threshold:
                await self._trigger_profile_update(speaker_name)
                self.samples_since_update[speaker_name] = 0

        # Save state periodically
        if self.interaction_count[speaker_name] % 5 == 0:
            await self._save_persistent_memory()

    async def _trigger_profile_update(self, speaker_name: str):
        """Trigger automatic profile update from recent samples"""
        logger.info(f"🔄 Triggering automatic profile update for {speaker_name}")

        try:
            # Get recent samples for training
            samples = await self.learning_db.get_voice_samples_for_training(
                speaker_name=speaker_name,
                limit=10,
                min_confidence=0.1
            )

            if len(samples) >= 5:
                # Perform incremental learning
                result = await self.learning_db.perform_incremental_learning(
                    speaker_name=speaker_name,
                    new_samples=samples
                )

                if result.get('success'):
                    logger.info(f"✅ Profile updated for {speaker_name}")

                    # Update memory
                    if speaker_name in self.voice_memory:
                        self.voice_memory[speaker_name]['last_updated'] = datetime.now().isoformat()
                        self.voice_memory[speaker_name]['auto_updates'] = \
                            self.voice_memory[speaker_name].get('auto_updates', 0) + 1

        except Exception as e:
            logger.error(f"Profile update failed: {e}")

    async def get_memory_summary(self, speaker_name: str) -> Dict:
        """
        Get comprehensive memory summary for a speaker

        Args:
            speaker_name: Speaker to get summary for

        Returns:
            Dict with memory information
        """
        if speaker_name not in self.voice_memory:
            return {'error': 'Speaker not found in memory'}

        memory = self.voice_memory[speaker_name]

        return {
            'speaker_name': speaker_name,
            'memory_loaded': True,
            'total_interactions': self.interaction_count.get(speaker_name, 0),
            'last_interaction': self.last_interaction.get(speaker_name),
            'voice_characteristics': memory,
            'freshness_score': memory.get('freshness', 'unknown'),
            'last_profile_update': memory.get('last_updated'),
            'auto_updates_count': memory.get('auto_updates', 0),
            'memory_age_hours': (
                (datetime.now() - datetime.fromisoformat(memory.get('loaded_at', datetime.now().isoformat()))).total_seconds() / 3600
            )
        }

    async def get_all_memories(self) -> Dict:
        """Get summary of all voice memories"""
        return {
            'total_speakers': len(self.voice_memory),
            'total_interactions': sum(self.interaction_count.values()),
            'last_freshness_check': self.last_freshness_check,
            'memory_file': str(self.memory_file),
            'speakers': {
                name: {
                    'interactions': self.interaction_count.get(name, 0),
                    'last_seen': self.last_interaction.get(name),
                    'freshness': memory.get('freshness', 'unknown')
                }
                for name, memory in self.voice_memory.items()
            }
        }

    async def cleanup(self):
        """Cleanup resources"""
        # Save final state
        await self._save_persistent_memory()

        # Close connections
        if self.learning_db:
            await self.learning_db.close()

        logger.info("🧹 Voice Memory Agent cleaned up")


# Global instance
_voice_memory_agent = None


async def get_voice_memory_agent() -> VoiceMemoryAgent:
    """Get or create the global voice memory agent"""
    global _voice_memory_agent

    if _voice_memory_agent is None:
        _voice_memory_agent = VoiceMemoryAgent()
        await _voice_memory_agent.initialize()

    return _voice_memory_agent


async def startup_voice_memory_check() -> Dict:
    """
    Convenience function to run voice memory startup check
    Called automatically by start_system.py
    """
    agent = await get_voice_memory_agent()
    return await agent.startup_check()


# For testing
if __name__ == "__main__":
    async def test():
        agent = await get_voice_memory_agent()
        result = await agent.startup_check()

        print("\n" + "="*80)
        print("🧠 VOICE MEMORY AGENT TEST")
        print("="*80)
        print(f"\nStatus: {result['status']}")
        print(f"Checks: {result['checks_performed']}")

        if result['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"   [{rec['priority']}] {rec['action']}")
                print(f"   → {rec['suggestion']}")

        if result['freshness']:
            print(f"\n📊 Freshness:")
            for speaker, metrics in result['freshness'].items():
                print(f"   {speaker}: {metrics['freshness_score']:.1%} ({metrics['total_samples']} samples)")

        # Get memory summary
        all_mem = await agent.get_all_memories()
        print(f"\n🧠 Total speakers in memory: {all_mem['total_speakers']}")
        print(f"🎤 Total interactions: {all_mem['total_interactions']}")

        await agent.cleanup()
        print("\n✅ Test complete!")

    asyncio.run(test())
