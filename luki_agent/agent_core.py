"""
LUKi Agent Core

The main orchestrator class that brings together all components:
- Context building and memory retrieval
- LLM integration and response generation
- Tool orchestration and safety filtering
- Session management and conversation flow
"""

import uuid
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from .config import settings
from .context_builder import ContextBuilder, ContextBuildResult
from .llm_backends import LLMManager, ModelResponse
from .memory.retriever import MemoryRetriever
from .memory.session_store import SessionStore, SessionState
from .prompts_system import get_system_prompt, get_instruction_template
from .safety_chain import SafetyChain
from .tools.registry import ToolRegistry


@dataclass
class AgentResponse:
    """Response from LUKi agent"""
    content: str
    session_id: str
    user_id: str
    timestamp: datetime
    context_used: Optional[ContextBuildResult] = None
    model_response: Optional[ModelResponse] = None
    tools_used: Optional[List[str]] = None
    safety_filtered: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationRequest:
    """Request for agent conversation"""
    user_input: str
    user_id: str
    session_id: Optional[str] = None
    handler_type: str = "general_chat"
    streaming: bool = False
    context_override: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LukiAgent:
    """
    Main LUKi Agent class - the brain of the system
    
    Orchestrates conversation flow, memory retrieval, context building,
    LLM generation, tool use, and safety filtering.
    """
    
    def __init__(
        self,
        memory_service_url: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        # Initialize components
        self.memory_retriever = MemoryRetriever(memory_service_url)
        self.session_store = SessionStore(redis_url)
        self.context_builder = ContextBuilder(self.memory_retriever)
        self.llm_manager = LLMManager()
        self.safety_chain = SafetyChain()
        self.tool_registry = ToolRegistry()
        
        # Register tools
        self._register_tools()
        
        # Agent state
        self.agent_id = str(uuid.uuid4())
        self.started_at = datetime.utcnow()
        self.conversation_count = 0
        
        print(f"LUKi Agent initialized (ID: {self.agent_id})")
    
    def _register_tools(self):
        """Register all available tools"""
        # Register memory tools
        self.tool_registry.register_memory_tools(self.memory_retriever)
        
        # Register cognitive tools (Phase 1C integration)
        self.tool_registry.register_cognitive_tools()
        
        # Log registered tools
        tools = self.tool_registry.list_tools()
        print(f"📋 Registered {len(tools)} tools: {[t['name'] for t in tools]}")
    
    async def chat(self, request: ConversationRequest) -> AgentResponse:
        """
        Main chat interface - handles a single conversation turn
        
        Args:
            request: Conversation request with user input and metadata
            
        Returns:
            Agent response with generated content and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Get or create session
            session = await self._get_or_create_session(request.user_id, request.session_id)
            
            # Store user message in session
            self.session_store.add_conversation_turn(
                session.session_id,
                "user",
                request.user_input,
                request.metadata
            )
            
            # Build context using the 5-layer model
            context_result = await self._build_context(request, session)
            
            # Apply safety filtering to user input
            filtered_input, safety_filtered = await self.safety_chain.filter_input(
                request.user_input,
                request.user_id
            )
            
            if safety_filtered:
                # Return safety message if input was filtered
                safety_response = await self._create_safety_response(request, session)
                return safety_response
            
            # Generate response using LLM
            if request.streaming:
                # For streaming, we'll return the first chunk and handle streaming separately
                response_content = ""
                async for chunk in self._generate_streaming_response(context_result.final_prompt):
                    response_content += chunk
                    break  # Just get the first chunk for now
            else:
                model_response = await self.llm_manager.generate(
                    prompt=context_result.final_prompt,
                    max_tokens=settings.max_tokens,
                    temperature=settings.model_temperature,
                    stop_sequences=["User:", "Human:"]
                )
                response_content = model_response.content
            
            # Apply safety filtering to response
            filtered_response, response_filtered = await self.safety_chain.filter_output(
                response_content,
                request.user_id
            )
            
            # Store assistant message in session
            self.session_store.add_conversation_turn(
                session.session_id,
                "assistant",
                filtered_response,
                {"safety_filtered": response_filtered}
            )
            
            # Update conversation count
            self.conversation_count += 1
            
            # Create response
            agent_response = AgentResponse(
                content=filtered_response,
                session_id=session.session_id,
                user_id=request.user_id,
                timestamp=start_time,
                context_used=context_result,
                model_response=model_response if not request.streaming else None,
                safety_filtered=response_filtered,
                metadata={
                    "handler_type": request.handler_type,
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "conversation_count": self.conversation_count
                }
            )
            
            # Log conversation for telemetry
            await self._log_conversation(request, agent_response)
            
            return agent_response
            
        except Exception as e:
            print(f"Agent error: {e}")
            return await self._create_error_response(request, str(e))
    
    async def chat_stream(self, request: ConversationRequest) -> AsyncGenerator[str, None]:
        """
        Streaming chat interface
        
        Args:
            request: Conversation request with streaming=True
            
        Yields:
            Response chunks as they are generated
        """
        try:
            # Get or create session
            session = await self._get_or_create_session(request.user_id, request.session_id)
            
            # Store user message
            self.session_store.add_conversation_turn(
                session.session_id,
                "user",
                request.user_input,
                request.metadata
            )
            
            # Build context
            context_result = await self._build_context(request, session)
            
            # Apply safety filtering
            filtered_input, safety_filtered = await self.safety_chain.filter_input(
                request.user_input,
                request.user_id
            )
            
            if safety_filtered:
                yield "I apologize, but I cannot respond to that request for safety reasons."
                return
            
            # Generate streaming response
            full_response = ""
            async for chunk in self._generate_streaming_response(context_result.final_prompt):
                # Apply safety filtering to each chunk
                filtered_chunk, _ = await self.safety_chain.filter_output(chunk, request.user_id)
                full_response += filtered_chunk
                yield filtered_chunk
            
            # Store complete response
            self.session_store.add_conversation_turn(
                session.session_id,
                "assistant",
                full_response
            )
            
        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    async def _get_or_create_session(self, user_id: str, session_id: Optional[str]) -> SessionState:
        """Get existing session or create new one"""
        if session_id:
            session = self.session_store.get_session(session_id)
            if session:
                return session
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        return self.session_store.create_session(user_id, new_session_id)
    
    async def _build_context(
        self, 
        request: ConversationRequest, 
        session: SessionState
    ) -> ContextBuildResult:
        """Build context using the 5-layer model"""
        conversation_history = self.session_store.get_conversation_history(
            session.session_id,
            limit=settings.conversation_buffer_size
        )
        
        context_result = await self.context_builder.build(
            user_input=request.user_input,
            user_id=request.user_id,
            conversation_history=conversation_history,
            handler_type=request.handler_type,
            **(request.context_override or {})
        )
        
        return context_result
    
    async def _generate_streaming_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        async for chunk in self.llm_manager.generate_stream(
            prompt=prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.model_temperature,
            stop_sequences=["User:", "Human:"]
        ):
            yield chunk
    
    async def _create_safety_response(
        self, 
        request: ConversationRequest, 
        session: SessionState
    ) -> AgentResponse:
        """Create response when input is safety filtered"""
        safety_message = "I understand you'd like to chat, but I need to be careful about the topics we discuss. Could you please rephrase your message or ask about something else?"
        
        self.session_store.add_conversation_turn(
            session.session_id,
            "assistant",
            safety_message,
            {"safety_filtered": True, "reason": "input_filtered"}
        )
        
        return AgentResponse(
            content=safety_message,
            session_id=session.session_id,
            user_id=request.user_id,
            timestamp=datetime.utcnow(),
            safety_filtered=True,
            metadata={"reason": "input_safety_filter"}
        )
    
    async def _create_error_response(
        self, 
        request: ConversationRequest, 
        error_message: str
    ) -> AgentResponse:
        """Create response when an error occurs"""
        error_response = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
        
        return AgentResponse(
            content=error_response,
            session_id=request.session_id or "error",
            user_id=request.user_id,
            timestamp=datetime.utcnow(),
            metadata={"error": error_message}
        )
    
    async def _log_conversation(
        self, 
        request: ConversationRequest, 
        response: AgentResponse
    ):
        """Log conversation for telemetry and evaluation"""
        try:
            # Store conversation summary in memory service
            if response.context_used and response.context_used.retrieval_results:
                await self.memory_retriever.update_conversation_summary(
                    user_id=request.user_id,
                    session_id=response.session_id,
                    summary=f"User: {request.user_input[:100]}... | LUKi: {response.content[:100]}...",
                    metadata={
                        "handler_type": request.handler_type,
                        "retrieval_count": len(response.context_used.retrieval_results) if response.context_used and response.context_used.retrieval_results else 0,
                        "context_tokens": response.context_used.total_tokens if response.context_used else 0,
                        "processing_time": response.metadata.get("processing_time", 0) if response.metadata else 0
                    }
                )
        except Exception as e:
            print(f"Conversation logging error: {e}")
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data"""
        return await self.memory_retriever.get_user_profile(user_id)
    
    async def get_recent_activities(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent user activities"""
        return await self.memory_retriever.get_recent_activities(user_id, limit)
    
    async def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.session_store.get_conversation_history(session_id, limit)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session"""
        return self.session_store.delete_session(session_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the agent"""
        try:
            # Test LLM connection
            test_response = await self.llm_manager.generate(
                "Hello", max_tokens=10, temperature=0.1
            )
            llm_healthy = bool(test_response.content)
        except:
            llm_healthy = False
        
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if llm_healthy else "degraded",
            "uptime": (datetime.utcnow() - self.started_at).total_seconds(),
            "conversation_count": self.conversation_count,
            "components": {
                "llm": "healthy" if llm_healthy else "error",
                "memory": "healthy",  # TODO: Add actual health check
                "session_store": "healthy",  # TODO: Add actual health check
                "safety": "healthy"  # TODO: Add actual health check
            }
        }
    
    async def close(self):
        """Clean up agent resources"""
        await self.llm_manager.close()
        await self.memory_retriever.close()
        print(f"LUKi Agent closed (ID: {self.agent_id})")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass
