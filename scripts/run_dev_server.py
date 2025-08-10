#!/usr/bin/env python3
"""
Development server runner for LUKi Core Agent

Starts the FastAPI development server with proper configuration.
"""

import os
import sys
import uvicorn

# Add the parent directory to Python path so we can import luki_agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from luki_agent.config import settings

def main():
    """Start the development server"""
    print(f"🚀 Starting LUKi Core Agent Development Server")
    print(f"📍 Host: {settings.host}:{settings.port}")
    print(f"🤖 Model Backend: {settings.model_backend}")
    print(f"🔧 Environment: {settings.environment}")
    print(f"📊 Debug Mode: {settings.debug}")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "luki_agent.dev_api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )

if __name__ == "__main__":
    main()
