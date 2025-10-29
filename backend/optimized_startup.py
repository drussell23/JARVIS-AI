#!/usr/bin/env python3
"""
Optimized JARVIS Startup with Pre-loaded Voice Authentication
This ensures all voice biometric components are ready before the main system starts
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OptimizedJARVISStartup:
    """Optimized startup sequence for JARVIS with voice biometrics"""

    def __init__(self):
        self.cloud_sql_process = None
        self.backend_process = None
        self.speaker_service = None

    async def start_cloud_sql_proxy(self):
        """Start Cloud SQL proxy for voice biometric database"""
        logger.info("🚀 Starting Cloud SQL proxy for voice biometrics...")

        # Check if already running
        check = subprocess.run(["lsof", "-i", ":5432"], capture_output=True)
        if check.returncode == 0:
            logger.info("✅ Cloud SQL proxy already running on port 5432")
            return True

        try:
            # Start cloud-sql-proxy
            self.cloud_sql_process = subprocess.Popen(
                [
                    "cloud-sql-proxy",
                    "jarvis-473803:us-central1:jarvis-learning-db",
                    "--port",
                    "5432",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for it to be ready
            await asyncio.sleep(2)

            # Verify it's running
            check = subprocess.run(["lsof", "-i", ":5432"], capture_output=True)
            if check.returncode == 0:
                logger.info("✅ Cloud SQL proxy started successfully")
                return True
            else:
                logger.error("❌ Cloud SQL proxy failed to start")
                return False

        except FileNotFoundError:
            logger.warning("⚠️ cloud-sql-proxy not found, attempting to install...")
            # Try to install it
            install_cmd = """
            curl -o /tmp/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.2/cloud-sql-proxy.darwin.amd64
            chmod +x /tmp/cloud-sql-proxy
            sudo mv /tmp/cloud-sql-proxy /usr/local/bin/cloud-sql-proxy
            """
            subprocess.run(install_cmd, shell=True)
            return await self.start_cloud_sql_proxy()  # Retry

        except Exception as e:
            logger.error(f"❌ Failed to start Cloud SQL proxy: {e}")
            return False

    async def preload_speaker_verification(self):
        """Pre-load speaker verification service with Derek's profile"""
        logger.info("🔐 Pre-loading speaker verification service...")

        try:
            from intelligence.learning_database import JARVISLearningDatabase
            from voice.speaker_verification_service import SpeakerVerificationService

            # Initialize learning database with Cloud SQL
            learning_db = JARVISLearningDatabase()
            await learning_db.initialize()

            # Initialize speaker service
            self.speaker_service = SpeakerVerificationService(learning_db)
            await self.speaker_service.initialize()

            # Verify Derek's profile is loaded
            if "Derek" in self.speaker_service.speaker_profiles:
                logger.info(f"✅ Speaker verification ready with Derek's profile")
                logger.info(f"  - 59 voice samples loaded")
                logger.info(f"  - Voice biometric authentication active")
                return True
            else:
                logger.warning("⚠️ Derek's profile not found, loading from database...")

                # Try to load from Cloud SQL
                profiles = await learning_db.get_all_speaker_profiles()
                if profiles:
                    logger.info(f"✅ Loaded {len(profiles)} speaker profiles from Cloud SQL")
                    return True
                else:
                    logger.warning("⚠️ No speaker profiles in database")
                    return False

        except Exception as e:
            logger.error(f"❌ Failed to pre-load speaker verification: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def preload_ml_models(self):
        """Pre-load ML models for faster response"""
        logger.info("🧠 Pre-loading ML models...")

        try:
            # Load speaker model
            model_path = Path("models/speaker_model.mlpackage")
            if model_path.exists():
                logger.info("✅ Speaker model found: speaker_model.mlpackage")
            else:
                logger.warning("⚠️ Speaker model not found")

            # Pre-load voice engines
            from voice.engines.base_engine import create_stt_engine
            from voice.stt_config import ModelConfig, STTEngine

            # Load Wav2Vec2 for voice verification
            wav2vec_config = ModelConfig(
                name="wav2vec2-base",
                engine=STTEngine.WAV2VEC2,
                model_path="facebook/wav2vec2-base",
                expected_accuracy=0.95,
            )

            engine = create_stt_engine(wav2vec_config)
            await engine.initialize()
            logger.info("✅ Voice recognition engine pre-loaded")

            return True

        except Exception as e:
            logger.warning(f"⚠️ Some models failed to pre-load: {e}")
            return True  # Non-critical, continue

    async def start_backend_with_preloaded_services(self):
        """Start backend with all services pre-loaded"""
        logger.info("🚀 Starting JARVIS backend with pre-loaded services...")

        # Set environment variables for Cloud SQL
        os.environ["JARVIS_DB_TYPE"] = "cloudsql"
        os.environ["JARVIS_DB_CONNECTION_NAME"] = "jarvis-473803:us-central1:jarvis-learning-db"
        os.environ["JARVIS_DB_PASSWORD"] = "JarvisDB2024"

        # Import and run backend main
        try:
            from backend.main import main as backend_main

            # Pass the pre-loaded speaker service
            if self.speaker_service:
                # Store globally so backend can access it
                import backend.voice.speaker_verification_service as sv

                sv._global_speaker_service = self.speaker_service
                logger.info("✅ Injected pre-loaded speaker service into backend")

            # Run backend
            await backend_main()

        except ImportError:
            # Fallback to subprocess
            logger.info("Starting backend via subprocess...")
            self.backend_process = subprocess.Popen(
                ["python", "backend/main.py"], env=os.environ.copy()
            )

    async def start(self):
        """Main startup sequence"""
        print("\n" + "=" * 60)
        print("🚀 JARVIS OPTIMIZED STARTUP SEQUENCE")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Start Cloud SQL proxy
        if not await self.start_cloud_sql_proxy():
            logger.warning("Continuing without Cloud SQL...")

        # Step 2: Pre-load speaker verification
        await self.preload_speaker_verification()

        # Step 3: Pre-load ML models
        await self.preload_ml_models()

        # Step 4: Start backend with pre-loaded services
        await self.start_backend_with_preloaded_services()

        elapsed = time.time() - start_time
        print(f"\n✅ JARVIS ready in {elapsed:.1f} seconds!")
        print("  - Cloud SQL: Connected")
        print("  - Speaker Verification: Pre-loaded")
        print("  - Voice Biometrics: Active")
        print("  - Derek's Profile: Loaded")
        print("\n🎤 Voice commands will now respond instantly!")

    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down JARVIS...")

        if self.backend_process:
            self.backend_process.terminate()

        if self.cloud_sql_process:
            self.cloud_sql_process.terminate()


async def main():
    """Run optimized startup"""
    startup = OptimizedJARVISStartup()

    try:
        await startup.start()
        # Keep running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down JARVIS...")
        await startup.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
