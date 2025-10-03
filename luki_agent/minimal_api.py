"""
Minimal API for Railway deployment - bypasses complex imports
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LUKi Core Agent API",
    description="Minimal API for Railway deployment",
    version="0.1.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo_user"
    session_id: str = "demo_session"

class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    response: str
    session_id: str
    user_id: str
    model_used: str = "minimal"

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": "luki-core-agent",
        "version": "0.1.0",
        "status": "running",
        "port": os.getenv("PORT", "8080"),
        "deployment": "railway-minimal"
    }

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "service": "luki-core-agent",
        "version": "0.1.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Minimal chat endpoint for testing"""
    try:
        # Simple response for testing
        response_text = f"Hello! I'm LUKi, your AI assistant. You said: '{request.message}'. This is a minimal deployment version for testing Railway connectivity."
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            user_id=request.user_id,
            model_used="minimal-fallback"
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    """Service status endpoint"""
    return {
        "service": "luki-core-agent",
        "status": "operational",
        "mode": "minimal-deployment",
        "features": ["basic-chat", "health-check"],
        "environment": {
            "port": os.getenv("PORT", "8080"),
            "python_version": "3.11.13"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
