#!/usr/bin/env python3
"""
Test script to verify all endpoints handle errors gracefully
"""

import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_endpoint(session, url, method="GET", data=None):
    """Test a single endpoint"""
    try:
        async with session.request(method, url, json=data) as response:
            status = response.status
            text = await response.text()
            
            if status >= 500:
                logger.error(f"❌ {method} {url} returned {status}: {text[:100]}")
                return False
            else:
                logger.info(f"✅ {method} {url} returned {status}")
                return True
    except Exception as e:
        logger.error(f"❌ {method} {url} failed: {e}")
        return False


async def main():
    """Test all endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("/voice/jarvis/activate", "POST"),
        ("/voice/jarvis/status", "GET"),
        ("/voice/jarvis/command", "POST", {"text": "test command"}),
        ("/vision/analyze_now", "POST"),
        ("/navigation/search", "POST", {"query": "test"}),
        ("/notifications/vision/subscribe", "POST", {"client_id": "test"})
    ]
    
    async with aiohttp.ClientSession() as session:
        logger.info("Testing endpoints for graceful error handling...")
        logger.info("=" * 60)
        
        results = []
        for endpoint in endpoints:
            url = base_url + endpoint[0]
            method = endpoint[1]
            data = endpoint[2] if len(endpoint) > 2 else None
            
            result = await test_endpoint(session, url, method, data)
            results.append(result)
        
        logger.info("=" * 60)
        passed = sum(results)
        total = len(results)
        logger.info(f"Results: {passed}/{total} endpoints handle errors gracefully")
        
        if passed == total:
            logger.info("✅ All endpoints are protected against 50x errors!")
        else:
            logger.warning("⚠️  Some endpoints still need attention")


if __name__ == "__main__":
    asyncio.run(main())
