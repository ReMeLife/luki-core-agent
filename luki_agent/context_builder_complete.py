"""
Context Builder for LUKi Agent - Complete Implementation

Implements the 5-layer context model as defined in 06-Context-Engineering-Playbook.md:
- L0 System: Role, tone, safety rails (128 tokens)
- L1 Instructions: Handler-specific prompts (256 tokens)
- L2 Retrieved Facts: ELR snippets, knowledge (1000 tokens)
- L3 Short-term Memory: Conversation summary (512 tokens)
- L4 Scratchpad: Tool results, reasoning (256 tokens)
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Use a simple token counter for now to avoid tiktoken dependency
def count_tokens_simple(text: str) -> int:
    """Simple token counter approximation"""
    return len(text.split()) + len(text) // 4

from .memory.retriever import MemoryRetriever
from .prompts_system import get_system_prompt, get_instruction_template
from .config import get_context_config


class ContextLayer(str, Enum):
    """Context layer enumeration"""
    SYSTEM = "L0_system"
    INSTRUCTIONS = "L1_instructions"
    RETRIEVED_FACTS = "L2_retrieved_facts"
    SHORT_TERM_MEMORY = "L3_short_term_memory"
    SCRATCHPAD = "L4_scratchpad"


@dataclass
class ContextChunk:
    """A chunk of context with metadata"""
    layer: str
    content: str
    token_count: int
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContextBuildResult:
    """Result of context building process"""
    final_prompt: str
    total_tokens: int
    layer_breakdown: Dict[str, int]
    chunks_used: List[str]
    chunks_trimmed: List[str]
    retrieval_results: Optional[List[Dict[str, Any]]] = None


class ContextBuilder:
    """Builds context for LLM prompts following the 5-layer model"""
    
    def __init__(self, memory_retriever: Optional[MemoryRetriever] = None):
        self.memory_retriever = memory_retriever
        self.config = get_context_config()
        self.max_context_tokens = self.config.get("max_context_tokens", 2048)
        
        # Token budgets per layer (can be adjusted dynamically)
        self.layer_budgets = {
            "L0_system": 128,
            "L1_instructions": 256,
            "L2_retrieved_facts": 1000,
            "L3_short_term_memory": 512,
            "L4_scratchpad": 256,
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using simple approximation"""
        return count_tokens_simple(text)
    
    async def build(
        self,
        user_input: str,
        user_id: str,
        conversation_history: List[Dict[str, str]],
        handler_type: str = "general_chat",
        scratchpad_content: Optional[str] = None,
        force_retrieval: bool = False,
        **kwargs
    ) -> ContextBuildResult:
        """
        Build complete context following the 5-layer model
        
        Args:
            user_input: Current user message
            user_id: User identifier for personalization
            conversation_history: Recent conversation turns
            handler_type: Type of handler (chat, activity_rec, etc.)
            scratchpad_content: Tool results or chain-of-thought
            force_retrieval: Force memory retrieval even if not needed
            **kwargs: Additional context parameters
            
        Returns:
            ContextBuildResult with assembled prompt and metadata
        """
        chunks = []
        
        # L0: System layer (persona, safety, role)
        system_chunk = self._build_system_layer(user_id, **kwargs)
        chunks.append(system_chunk)
        
        # L1: Instructions layer (handler-specific prompts)
        instruction_chunk = self._build_instruction_layer(handler_type, user_input, **kwargs)
        chunks.append(instruction_chunk)
        
        # L2: Retrieved facts (ELR snippets, knowledge)
        retrieval_results = None
        if self.memory_retriever and (force_retrieval or self._should_retrieve(user_input, handler_type)):
            retrieved_chunk, retrieval_results = await self._build_retrieval_layer(
                user_input, user_id, conversation_history, **kwargs
            )
            if retrieved_chunk:
                chunks.append(retrieved_chunk)
        
        # L3: Short-term memory (conversation summary)
        if conversation_history:
            memory_chunk = self._build_memory_layer(conversation_history, user_id, **kwargs)
            if memory_chunk:
                chunks.append(memory_chunk)
        
        # L4: Scratchpad (tool results, reasoning)
        if scratchpad_content:
            scratchpad_chunk = self._build_scratchpad_layer(scratchpad_content, **kwargs)
            chunks.append(scratchpad_chunk)
        
        # Assemble and optimize context
        return self._assemble_context(chunks, user_input, retrieval_results)
    
    def _build_system_layer(self, user_id: str, **kwargs) -> ContextChunk:
        """Build L0 system layer with persona and safety"""
        system_prompt = get_system_prompt(
            user_id=user_id,
            personality_mode=kwargs.get("personality_mode", "default"),
            safety_level=kwargs.get("safety_level", "standard")
        )
        
        return ContextChunk(
            layer="L0_system",
            content=system_prompt,
            token_count=self._count_tokens(system_prompt),
            priority=0,
            metadata={"user_id": user_id}
        )
    
    def _build_instruction_layer(self, handler_type: str, user_input: str, **kwargs) -> ContextChunk:
        """Build L1 instruction layer with handler-specific template"""
        instruction_template = get_instruction_template(
            handler_type=handler_type
        )
        
        return ContextChunk(
            layer="L1_instructions",
            content=instruction_template,
            token_count=self._count_tokens(instruction_template),
            priority=1,
            metadata={"handler_type": handler_type}
        )
    
    async def _build_retrieval_layer(
        self, 
        user_input: str, 
        user_id: str, 
        conversation_history: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[Optional[ContextChunk], Optional[List[Dict[str, Any]]]]:
        """Build L2 retrieval layer with relevant ELR snippets"""
        if not self.memory_retriever:
            return None, None
        
        try:
            # Build search query from user input and recent context
            search_query = self._build_search_query(user_input, conversation_history)
            
            # Get retrieval results
            retrieval_results = await self.memory_retriever.search_memories(
                query=search_query,
                user_id=user_id,
                top_k=self.config.get("retrieval_top_k", 6),
                min_score=0.7
            )
            
            if not retrieval_results or len(retrieval_results) == 0:
                return None, None
            
            # Format retrieved content
            retrieved_content = self._format_retrieval_results(retrieval_results)
            
            return ContextChunk(
                layer="L2_retrieved_facts",
                content=retrieved_content,
                token_count=self._count_tokens(retrieved_content),
                priority=2,
                metadata={
                    "retrieval_count": len(retrieval_results),
                    "search_query": search_query
                }
            ), retrieval_results
            
        except Exception as e:
            # Log error but don't fail context building
            print(f"Retrieval error: {e}")
            return None, None
    
    def _should_retrieve(self, user_input: str, handler_type: str) -> bool:
        """Determine if memory retrieval is needed for this input"""
        # Always retrieve for general chat and questions
        if handler_type in ["general_chat", "question_answering"]:
            return True
        
        # Skip retrieval for simple greetings
        simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if user_input.lower().strip() in simple_greetings:
            return False
        
        # Retrieve if input contains question words or personal references
        question_words = ["what", "how", "why", "when", "where", "who", "tell me", "explain", "remember"]
        input_lower = user_input.lower()
        
        return any(word in input_lower for word in question_words)
    
    def _build_memory_layer(self, conversation_history: List[Dict[str, str]], user_id: str, **kwargs) -> Optional[ContextChunk]:
        """Build L3 memory layer with conversation summary"""
        if not conversation_history or len(conversation_history) < 2:
            return None
        
        # Take recent conversation turns for context
        recent_turns = conversation_history[-6:]  # Last 3 exchanges
        
        # Build conversation summary
        summary_lines = []
        for turn in recent_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if content:
                summary_lines.append(f"{role.title()}: {content[:200]}{'...' if len(content) > 200 else ''}")
        
        if not summary_lines:
            return None
        
        summary_content = "\n".join(summary_lines)
        
        return ContextChunk(
            layer="L3_short_term_memory",
            content=f"Recent conversation context:\n{summary_content}",
            token_count=self._count_tokens(summary_content),
            priority=3,
            metadata={"conversation_turns": len(recent_turns)}
        )
    
    def _build_scratchpad_layer(self, scratchpad_content: str, **kwargs) -> ContextChunk:
        """Build L4 scratchpad layer with tool results and reasoning"""
        return ContextChunk(
            layer="L4_scratchpad",
            content=f"Working memory and tool results:\n{scratchpad_content}",
            token_count=self._count_tokens(scratchpad_content),
            priority=4,
            metadata={"type": "scratchpad"}
        )
    
    def _assemble_context(self, chunks: List[ContextChunk], user_input: str, retrieval_results: Optional[List[Dict[str, Any]]]) -> ContextBuildResult:
        """Assemble final context from chunks with token budget management"""
        # Sort chunks by priority (lower number = higher priority)
        chunks.sort(key=lambda x: x.priority)
        
        # Calculate total tokens and manage budget
        total_tokens = sum(chunk.token_count for chunk in chunks)
        user_input_tokens = self._count_tokens(user_input)
        
        # Reserve tokens for user input and response
        available_tokens = self.max_context_tokens - user_input_tokens - 500  # Reserve 500 for response
        
        final_chunks = []
        chunks_trimmed = []
        current_tokens = 0
        
        # Add chunks in priority order until budget is exhausted
        for chunk in chunks:
            if current_tokens + chunk.token_count <= available_tokens:
                final_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                # Try to trim the chunk if it's not critical
                if chunk.layer not in ["L0_system", "L1_instructions"]:
                    # Trim content to fit remaining budget
                    remaining_tokens = available_tokens - current_tokens
                    if remaining_tokens > 100:  # Only trim if we have reasonable space
                        trimmed_content = self._trim_content_to_tokens(chunk.content, remaining_tokens - 50)
                        if trimmed_content:
                            trimmed_chunk = ContextChunk(
                                layer=chunk.layer,
                                content=trimmed_content,
                                token_count=self._count_tokens(trimmed_content),
                                priority=chunk.priority,
                                metadata={**(chunk.metadata or {}), "trimmed": True}
                            )
                            final_chunks.append(trimmed_chunk)
                            current_tokens += trimmed_chunk.token_count
                            chunks_trimmed.append(chunk.layer)
                break
        
        # Assemble final prompt
        prompt_parts = []
        layer_breakdown = {}
        
        for chunk in final_chunks:
            prompt_parts.append(chunk.content)
            layer_breakdown[chunk.layer] = {
                "tokens": chunk.token_count,
                "priority": chunk.priority,
                "metadata": chunk.metadata
            }
        
        # Add user input at the end
        prompt_parts.append(f"\nUser: {user_input}\nAssistant:")
        
        final_prompt = "\n\n".join(prompt_parts)
        final_token_count = self._count_tokens(final_prompt)
        
        return ContextBuildResult(
            final_prompt=final_prompt,
            total_tokens=final_token_count,
            layer_breakdown=layer_breakdown,
            chunks_used=[chunk.layer for chunk in final_chunks],
            chunks_trimmed=chunks_trimmed,
            retrieval_results=retrieval_results
        )
    
    def _build_search_query(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build search query from user input and conversation context"""
        # Start with user input
        query_parts = [user_input]
        
        # Add recent context if available
        if conversation_history:
            recent_user_messages = [
                turn.get("content", "") 
                for turn in conversation_history[-3:] 
                if turn.get("role") == "user"
            ]
            query_parts.extend(recent_user_messages)
        
        return " ".join(query_parts)
    
    def _format_retrieval_results(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results for context inclusion"""
        formatted_parts = []
        
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            content = result.get("content", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # Format each result
            formatted_parts.append(f"[Memory {i+1}] (relevance: {score:.2f})\n{content}")
        
        return "\n\n".join(formatted_parts)
    
    def _trim_content_to_tokens(self, content: str, max_tokens: int) -> str:
        """Trim content to fit within token budget"""
        if self._count_tokens(content) <= max_tokens:
            return content
        
        # Simple trimming by character count (rough approximation)
        chars_per_token = len(content) / max(self._count_tokens(content), 1)
        target_chars = int(max_tokens * chars_per_token * 0.9)  # 90% safety margin
        
        if target_chars < len(content):
            return content[:target_chars] + "..."
        
        return content
