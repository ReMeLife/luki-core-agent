"""
Context Builder for LUKi Agent

Implements the 5-layer context model as defined in 06-Context-Engineering-Playbook.md:
- L0 System: Role, tone, safety rails (128 tokens)
- L1 Instructions: Handler-specific prompt template (256 tokens)  
- L2 Retrieved Facts: Personal ELR snippets, knowledge docs (1000 tokens)
- L3 Short-term Memory: Last ~20 messages summarised (512 tokens)
- L4 Scratchpad: Tool results, chain-of-thought (256 tokens)

Total budget: ≤2k tokens, keeping ≥1k tokens for generation.
"""

import tiktoken
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .config import settings, get_context_config
from .memory.retriever import MemoryRetriever
from .prompts_system import get_system_prompt, get_instruction_template


class ContextLayer(Enum):
    """Context layer enumeration"""
    SYSTEM = "L0_system"
    INSTRUCTIONS = "L1_instructions"
    RETRIEVED_FACTS = "L2_retrieved_facts"
    SHORT_TERM_MEMORY = "L3_short_term_memory"
    SCRATCHPAD = "L4_scratchpad"


@dataclass
class ContextChunk:
    """A chunk of context with metadata"""
    layer: ContextLayer
    content: str
    token_count: int
    priority: int = 1  # Higher priority = more important
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContextBuildResult:
    """Result of context building process"""
    final_prompt: str
    total_tokens: int
    layer_breakdown: Dict[ContextLayer, int]
    chunks_used: List[ContextChunk]
    chunks_trimmed: List[ContextChunk]
    retrieval_results: Optional[List[Dict[str, Any]]] = None


class ContextBuilder:
    """
    Builds context for LLM prompts following the 5-layer model
    """
    
    def __init__(self, memory_retriever: Optional[MemoryRetriever] = None):
        self.memory_retriever = memory_retriever
        self.config = get_context_config()
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
        # Token budgets per layer (can be adjusted dynamically)
        self.layer_budgets = {
            ContextLayer.SYSTEM: 128,
            ContextLayer.INSTRUCTIONS: 256,
            ContextLayer.RETRIEVED_FACTS: 1000,
            ContextLayer.SHORT_TERM_MEMORY: 512,
            ContextLayer.SCRATCHPAD: 256,
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
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
        
        # L0: System prompt (persona, safety rails)
        system_chunk = self._build_system_layer(user_id, **kwargs)
        chunks.append(system_chunk)
        
        # L1: Instructions (handler-specific template)
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
        
        token_count = self.count_tokens(system_prompt)
        
        return ContextChunk(
            layer=ContextLayer.SYSTEM,
            content=system_prompt,
            token_count=token_count,
            priority=5,  # Highest priority - never trim
            metadata={"user_id": user_id}
        )
    
    def _build_instruction_layer(self, handler_type: str, user_input: str, **kwargs) -> ContextChunk:
        """Build L1 instruction layer with handler-specific template"""
        instruction_template = get_instruction_template(handler_type)
        
        # Fill template with current context
        instructions = instruction_template.format(
            user_input=user_input,
            current_time=kwargs.get("current_time", ""),
            user_mood=kwargs.get("user_mood", ""),
            **kwargs
        )
        
        token_count = self.count_tokens(instructions)
        
        return ContextChunk(
            layer=ContextLayer.INSTRUCTIONS,
            content=instructions,
            token_count=token_count,
            priority=4,  # High priority
            metadata={"handler_type": handler_type}
        )
    
    async def _build_retrieval_layer(
        self, 
        user_input: str, 
        user_id: str, 
        conversation_history: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[Optional[ContextChunk], Optional[List[Dict[str, Any]]]:
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
                top_k=self.config["retrieval_top_k"],
                similarity_threshold=0.25,
                **kwargs
            )
            
            if not retrieval_results or len(retrieval_results) == 0:
                return None, None
            
            # Format retrieved content
            retrieved_content = self._format_retrieval_results(retrieval_results)
            token_count = self.count_tokens(retrieved_content)
            
            return ContextChunk(
                layer=ContextLayer.RETRIEVED_FACTS,
                content=retrieved_content,
                token_count=token_count,
                priority=3,
                metadata={"retrieval_count": len(retrieval_results)}
            ), retrieval_results
            
        except Exception as e:
            # Log error but don't fail context building
            print(f"Retrieval error: {e}")
            return None, None
    
    def _build_memory_layer(self, conversation_history: List[Dict[str, str]], user_id: str, **kwargs) -> Optional[ContextChunk]:
        """Build L3 short-term memory layer with conversation summary"""
        if not conversation_history:
            return None
        
        # Take recent conversation turns (last N messages)
        recent_history = conversation_history[-self.config["conversation_buffer_size"]:]
        
        # Format conversation history
        formatted_history = self._format_conversation_history(recent_history)
        token_count = self.count_tokens(formatted_history)
        
        # If too long, summarize older parts
        if token_count > self.layer_budgets[ContextLayer.SHORT_TERM_MEMORY]:
            formatted_history = self._summarize_conversation(recent_history, user_id)
            token_count = self.count_tokens(formatted_history)
        
        return ContextChunk(
            layer=ContextLayer.SHORT_TERM_MEMORY,
            content=formatted_history,
            token_count=token_count,
            priority=2,
            metadata={"message_count": len(recent_history)}
        )
    
    def _build_scratchpad_layer(self, scratchpad_content: str, **kwargs) -> ContextChunk:
        """Build L4 scratchpad layer with tool results"""
        token_count = self.count_tokens(scratchpad_content)
        
        return ContextChunk(
            layer=ContextLayer.SCRATCHPAD,
            content=scratchpad_content,
            token_count=token_count,
            priority=1,  # Lowest priority - trim first
            metadata={"type": "scratchpad"}
        )
    
    def _should_retrieve(self, user_input: str, handler_type: str) -> bool:
        """Determine if memory retrieval is needed"""
        # Always retrieve for certain handler types
        if handler_type in ["activity_recommendation", "wellbeing_report"]:
            return True
        
        # Check for personal references in user input
        personal_indicators = [
            "my", "i", "me", "remember", "recall", "told you", "mentioned",
            "family", "friend", "hobby", "like", "enjoy", "prefer"
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in personal_indicators)
    
    def _build_search_query(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build effective search query from user input and context"""
        # Start with user input
        query_parts = [user_input]
        
        # Add recent context if relevant
        if conversation_history:
            last_turn = conversation_history[-1]
            if last_turn.get("role") == "assistant":
                # Include last assistant response for context continuity
                query_parts.append(last_turn.get("content", "")[:100])
        
        return " ".join(query_parts)
    
    def _format_retrieval_results(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieval results for context inclusion"""
        if not results:
            return ""
        
        formatted_parts = ["Relevant background information:"]
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "memory")
            
            formatted_parts.append(f"{i}. [{source}] {content}")
        
        return "\n".join(formatted_parts)
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for context"""
        if not history:
            return ""
        
        formatted_parts = ["Recent conversation:"]
        
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            if role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"LUKi: {content}")
        
        return "\n".join(formatted_parts)
    
    def _summarize_conversation(self, history: List[Dict[str, str]], user_id: str) -> str:
        """Summarize conversation history to fit token budget"""
        # For now, just truncate. In future, use LLM to summarize
        formatted = self._format_conversation_history(history)
        
        # Simple truncation strategy
        max_chars = self.layer_budgets[ContextLayer.SHORT_TERM_MEMORY] * 4  # Rough char estimate
        if len(formatted) > max_chars:
            formatted = formatted[:max_chars] + "...\n[Earlier conversation summarized]"
        
        return formatted
    
    def _assemble_context(
        self, 
        chunks: List[ContextChunk], 
        user_input: str,
        retrieval_results: Optional[List[Dict[str, Any]]] = None
    ) -> ContextBuildResult:
        """Assemble final context with token budget management"""
        # Calculate total tokens
        total_tokens = sum(chunk.token_count for chunk in chunks)
        
        # Track what we use vs trim
        chunks_used = []
        chunks_trimmed = []
        
        # If over budget, trim lower priority chunks first
        if total_tokens > self.config["max_context_tokens"]:
            chunks = sorted(chunks, key=lambda x: x.priority, reverse=True)
            
            running_total = 0
            for chunk in chunks:
                if running_total + chunk.token_count <= self.config["max_context_tokens"]:
                    chunks_used.append(chunk)
                    running_total += chunk.token_count
                else:
                    chunks_trimmed.append(chunk)
        else:
            chunks_used = chunks
        
        # Assemble final prompt in layer order
        final_chunks = sorted(chunks_used, key=lambda x: list(ContextLayer).index(x.layer))
        
        prompt_parts = []
        for chunk in final_chunks:
            if chunk.content.strip():
                prompt_parts.append(chunk.content)
        
        # Add current user input at the end
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("LUKi:")  # Prompt for assistant response
        
        final_prompt = "\n\n".join(prompt_parts)
        final_token_count = self.count_tokens(final_prompt)
