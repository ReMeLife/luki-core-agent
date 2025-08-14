"""
Development API for LUKi Core Agent

FastAPI server providing HTTP endpoints for testing and development.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import traceback

from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LUKi Core Agent API",
    description="Development API for LUKi Core Agent",
    version="0.1.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str
    model_used: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": "luki-core-agent",
        "version": "0.1.0",
        "status": "running",
        "model_backend": settings.model_backend,
        "environment": settings.environment
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_backend": settings.model_backend,
        "debug": settings.debug
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for conversing with LUKi
    
    This is a simplified endpoint for testing the LLM integration.
    In production, this would route through the full conversation chain.
    """
    try:
        logger.info(f"Chat request from user {request.user_id}, session {request.session_id}")
        logger.info(f"Message: {request.message}")
        logger.info(f"Model backend: {settings.model_backend}")
        
        # Import here to avoid circular imports
        from .llm_backends import LLMManager
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        
        # Generate response directly through LLMManager
        logger.info(f"Generating response using {settings.model_backend} backend...")
        response = await llm_manager.generate(
            prompt=f"You are LUKi, a helpful AI assistant. User message: {request.message}",
            max_tokens=settings.max_tokens,
            temperature=settings.model_temperature
        )
        
        logger.info(f"Generated response: {response.content[:100]}...")
        
        return ChatResponse(
            response=response.content,
            session_id=request.session_id,
            user_id=request.user_id,
            model_used=settings.model_backend,
            metadata={
                "model_name": settings.model_name,
                "usage": response.usage,
                "finish_reason": response.finish_reason
            }
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return fallback response for debugging
        return ChatResponse(
            response=f"I'm LUKi, but I'm having technical difficulties right now. Error: {str(e)}",
            session_id=request.session_id,
            user_id=request.user_id,
            model_used="fallback",
            metadata={"error": str(e)}
        )

@app.get("/v1/config")
async def get_config():
    """Get current configuration (for debugging)"""
    return {
        "model_backend": settings.model_backend,
        "model_name": settings.model_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
