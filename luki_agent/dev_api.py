"""
Development API for LUKi Core Agent

FastAPI server providing HTTP endpoints for testing and development.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
import logging
import traceback
import json
import asyncio

from .config import settings
from .prompts_enhanced import get_enhanced_system_prompt, format_llama_prompt, get_context_strategy

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
    Chat endpoint for conversing with LUKi with full ELR integration
    """
    try:
        logger.info(f"Chat request from user {request.user_id}, session {request.session_id}")
        logger.info(f"Message: {request.message}")
        logger.info(f"Model backend: {settings.model_backend}")
        
        # Import here to avoid circular imports
        from .llm_backends import LLMManager
        from .memory.memory_service_client import MemoryServiceClient
        
        # Initialize components
        llm_manager = LLMManager()
        memory_client = MemoryServiceClient()
        
        # Retrieve ELR memories for the user
        try:
            memories = await memory_client.search_memories(
                user_id=request.user_id,
                query=request.message,
                k=5
            )
            logger.info(f"Retrieved {len(memories)} memories for user {request.user_id}")
        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")
            memories = []
        
        # Determine context strategy using enhanced analysis
        context_strategy = get_context_strategy(
            user_message=request.message,
            user_id=request.user_id,
            memories=memories
        )
        
        # Get enhanced system prompt with appropriate personality
        personality_mode = "empathetic" if context_strategy["needs_empathetic_response"] else "default"
        system_prompt = get_enhanced_system_prompt(
            user_id=request.user_id,
            personality_mode=personality_mode,
            project_knowledge=context_strategy["use_project_knowledge"]
        )
        
        # Build memory context if strategy indicates it's needed
        memory_context = ""
        if context_strategy["use_memory_context"]:
            memory_context = "User Context:\n"
            for memory in memories[:3]:  # Top 3 most relevant
                memory_context += f"- {memory.content}\n"
        
        # Get conversation history from context
        conversation_history = []
        if request.context and "conversation_history" in request.context:
            conversation_history = request.context["conversation_history"]
        
        # Format using proper LLaMA 3.3 structure
        full_prompt = format_llama_prompt(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_message=request.message,
            memory_context=memory_context
        )
        
        # Generate response using full context
        logger.info(f"Generating response using {settings.model_backend} backend...")
        response = await llm_manager.generate(
            prompt=full_prompt,
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

async def generate_streaming_response(prompt: str, user_id: str, session_id: str) -> AsyncGenerator[str, None]:
    """Generate streaming response from LLM"""
    try:
        # Import here to avoid circular imports
        from .llm_backends import LLMManager
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        
        # For now, simulate streaming by chunking the response
        # In a full implementation, this would use the LLM's streaming capability
        logger.info(f"Generating streaming response using {settings.model_backend} backend...")
        response = await llm_manager.generate(
            prompt=prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.model_temperature
        )
        
        # Simulate streaming by sending response in chunks
        content = response.content
        words = content.split()
        
        for i, word in enumerate(words):
            chunk_data = {
                "token": word + (" " if i < len(words) - 1 else ""),
                "session_id": session_id,
                "user_id": user_id
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
        
        # Send completion signal
        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        error_data = {
            "error": str(e),
            "session_id": session_id,
            "user_id": user_id
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for conversing with LUKi with full ELR integration
    
    Returns server-sent events with streaming response tokens.
    """
    try:
        logger.info(f"Streaming chat request from user {request.user_id}, session {request.session_id}")
        logger.info(f"Message: {request.message}")
        
        # Import here to avoid circular imports
        from .memory.memory_service_client import MemoryServiceClient
        
        # Initialize memory client
        memory_client = MemoryServiceClient()
        
        # Retrieve ELR memories for the user
        try:
            memories = await memory_client.search_memories(
                user_id=request.user_id,
                query=request.message,
                k=5
            )
            logger.info(f"Retrieved {len(memories)} memories for streaming user {request.user_id}")
        except Exception as e:
            logger.warning(f"Failed to retrieve memories for streaming: {e}")
            memories = []
        
        # Determine context strategy using enhanced analysis
        context_strategy = get_context_strategy(
            user_message=request.message,
            user_id=request.user_id,
            memories=memories
        )
        
        # Get enhanced system prompt with appropriate personality
        personality_mode = "empathetic" if context_strategy["needs_empathetic_response"] else "default"
        system_prompt = get_enhanced_system_prompt(
            user_id=request.user_id,
            personality_mode=personality_mode,
            project_knowledge=context_strategy["use_project_knowledge"]
        )
        
        # Build memory context if strategy indicates it's needed
        memory_context = ""
        if context_strategy["use_memory_context"]:
            memory_context = "User Context:\n"
            for memory in memories[:3]:  # Top 3 most relevant
                memory_context += f"- {memory.content}\n"
        
        # Get conversation history from context
        conversation_history = []
        if request.context and "conversation_history" in request.context:
            conversation_history = request.context["conversation_history"]
        
        # Format using proper LLaMA 3.3 structure
        full_prompt = format_llama_prompt(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_message=request.message,
            memory_context=memory_context
        )
        
        # Return streaming response with full context
        return StreamingResponse(
            generate_streaming_response(full_prompt, request.user_id, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming chat endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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
