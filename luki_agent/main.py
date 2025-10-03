"""
Main entry point for LUKi Core Agent - Railway deployment
"""

try:
    from .dev_api import app
except ImportError:
    # Fallback to minimal API for Railway deployment
    from .minimal_api import app

# Export the FastAPI app for deployment
__all__ = ["app"]
