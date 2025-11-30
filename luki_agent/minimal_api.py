"""
Minimal API for Railway deployment - bypasses complex imports
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
import logging
import os
import json

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
        response_text = (
            f"Hello! I'm LUKi, your AI assistant. You said: '{request.message}'. "
            "This is a minimal deployment version for testing Railway connectivity."
        )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            user_id=request.user_id,
            model_used="minimal-fallback",
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_v1(request: ChatRequest):
    """Compatibility alias for core agent /v1/chat expected by the API Gateway."""
    return await chat(request)


@app.post("/v1/chat/stream")
async def chat_stream_v1(request: ChatRequest):
    """Minimal streaming alias for /v1/chat/stream.

    This provides a very simple Server-Sent Events style stream so that the
    gateway does not receive a 404 when requesting streaming responses.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Reuse the minimal chat implementation
            result = await chat(request)

            # Stream the response back as a single chunk followed by done marker
            payload = {"token": result.response}
            yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Streaming chat error: {e}")
            error_payload = {"error": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
