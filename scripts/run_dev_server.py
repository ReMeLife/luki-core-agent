#!/usr/bin/env python3
"""
Development server launcher for LUKi Core Agent

Starts the FastAPI development server with proper configuration.
Supports OpenAI GPT models and other configurable LLM backends.
"""

import os
import sys
import uvicorn

# Add parent directory to Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from luki_agent.config import settings

def main():
    """Launch the development server"""
    print(f"🚀 Starting LUKi Core Agent Development Server")
    print(f"📍 Host: {settings.host}:{settings.port}")
    print(f"🤖 Model Backend: {settings.model_backend}")
    print(f"🔧 Environment: {settings.environment}")
    print(f"📊 Debug Mode: {settings.debug}")
    print("-" * 50)
    
    # Launch the FastAPI server
    uvicorn.run(
        "luki_agent.dev_api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )

if __name__ == "__main__":
    main()
