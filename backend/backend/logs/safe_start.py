#!/usr/bin/env python3
import sys
import os
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Set safe mode flag
    os.environ['JARVIS_SAFE_MODE'] = '1'
    
    logger.info("Starting JARVIS in Safe Mode...")
    
    # Import with error handling
    try:
        import main
        import uvicorn
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Installing missing dependencies...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import main
        import uvicorn
    
    # Start server with custom error handler
    class SafeServer:
        def __init__(self):
            self.app = main.app
            
        def run(self):
            try:
                uvicorn.run(
                    self.app, 
                    host="127.0.0.1", 
                    port=8010,
                    log_level="info",
                    access_log=False,  # Reduce logging overhead
                    loop="asyncio",
                    limit_concurrency=100,  # Limit concurrent connections
                    limit_max_requests=1000,  # Restart worker after 1000 requests
                )
            except Exception as e:
                logger.error(f"Server error: {e}")
                logger.info("Starting minimal server...")
                # Start with minimal functionality
                from fastapi import FastAPI
                minimal_app = FastAPI()
                
                @minimal_app.get("/")
                async def root():
                    return {"status": "safe_mode", "message": "JARVIS running in safe mode"}
                
                @minimal_app.get("/health")
                async def health():
                    return {"status": "degraded", "mode": "safe"}
                
                uvicorn.run(minimal_app, host="127.0.0.1", port=8010)
    
    server = SafeServer()
    server.run()
    
except Exception as e:
    logger.error(f"Fatal error: {e}")
    traceback.print_exc()
    sys.exit(1)
