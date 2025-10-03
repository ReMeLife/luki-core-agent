"""
Context Builder for LUKi Agent - Hardened Version

Builds context for LLM prompts with strict slot separation, sanitization, and token budgets.
"""

from typing import Dict, List, Optional, Any
import logging
import re
from datetime import datetime

from .prompt_registry import prompt_registry

logger = logging.getLogger(__name__)

def count_tokens_simple(text: str) -> int:
    """Simple token counter approximation (4 chars ≈ 1 token)"""
    return len(text.split()) + len(text) // 4

def sanitize_retrieval_context(text: str) -> str:
    """Sanitize retrieved context to prevent prompt injection"""
    if not text:
        return ""
    
    # Check for dangerous patterns that should be completely removed
    dangerous_patterns = [
        r'^\s*System:',
        r'^\s*#\s*Instructions?',
        r'Ignore\s+previous\s+instructions',
        r'This\s+is\s+a\s+system\s+message',
        r'```json.*?```',
        r'\{\s*"role"\s*:',
    ]
    
    # Check if text contains dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL):
            return ""  # Remove entirely if dangerous
    
    # Clean up the text but preserve content
    sanitized = text
    
    # Remove special tokens but preserve normal content
    sanitized = re.sub(r'<\|[^|]*\|>', '', sanitized)
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\n\s*\n\s*\n+', '\n\n', sanitized)
    sanitized = sanitized.strip()
    
    return sanitized

class ContextBuilder:
    """Hardened context builder with slot separation and sanitization"""
    
    def __init__(self, memory_retriever=None):
        self.memory_retriever = memory_retriever
        self.max_context_tokens = 2048
        
        # Token budget allocation per slot
        self.slot_budgets = {
            'system_core': 800,      # Core system prompt
            'persona': 300,          # Personality traits
            'user_guidance': 200,    # User-specific guidance
            'retrieval_context': 700, # ELR/memory context (increased for broader coverage)
            'conversation_history': 300, # Recent conversation
            'safety_rules': 200      # Safety guidelines
        }
    
    async def build(
        self,
        user_input: str,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        handler_type: str = "general_chat",
        memory_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build context with strict slot separation and sanitization"""
        
        # Build system prompt using registry with compact routing for short inputs
        personality_mode = kwargs.get("personality_mode", "default")
        include_safety = True
        compact = len(user_input.strip()) <= 40
        if compact:
            # Use minimal core to reduce token overhead
            core = prompt_registry.load_prompt("system_core_min", "v1")
            persona = prompt_registry.load_prompt("persona_luki", "v1")
            # Call private helper for user guidance to maintain consistency
            user_guidance = prompt_registry._get_user_guidance(user_id)
            safety = prompt_registry.load_prompt("safety_rules", "v1") if include_safety else ""
            components = [core, persona, user_guidance]
            if safety:
                components.append(safety)
            system_prompt = "\n\n".join(components)
        else:
            system_prompt = prompt_registry.build_system_prompt(
                user_id=user_id,
                personality_mode=personality_mode,
                include_safety=include_safety
            )
        
        # Build sanitized retrieval context
        retrieval_context = ""
        
        if memory_context:
            # Sanitize and filter memory context
            sanitized_memories = []
            for item in memory_context[:5]:  # Limit to top 5 for expanded knowledge
                content = item.get("content", "")
                sanitized_content = sanitize_retrieval_context(content)
                
                if sanitized_content and len(sanitized_content) > 10:
                    # Truncate to fit budget
                    max_length = self.slot_budgets['retrieval_context'] // 3
                    if len(sanitized_content) > max_length:
                        # Truncate at sentence boundary
                        truncated = sanitized_content[:max_length]
                        last_period = truncated.rfind('.')
                        if last_period > max_length // 2:
                            sanitized_content = truncated[:last_period + 1]
                        else:
                            sanitized_content = truncated + "..."
                    
                    sanitized_memories.append(sanitized_content)
            
            if sanitized_memories:
                retrieval_context = "\n\nRelevant Context:\n" + "\n".join(f"- {mem}" for mem in sanitized_memories)
        
        # Build conversation context with token budget
        conversation_context = ""
        if conversation_history:
            # Include last 2-3 exchanges within token budget
            recent_messages = conversation_history[-4:]  # Last 4 messages (2 exchanges)
            
            context_parts = []
            token_count = 0
            max_tokens = self.slot_budgets['conversation_history']
            
            for msg in reversed(recent_messages):
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                # Sanitize message content
                sanitized_content = sanitize_retrieval_context(content)
                
                if sanitized_content:
                    msg_tokens = count_tokens_simple(sanitized_content)
                    if token_count + msg_tokens <= max_tokens:
                        context_parts.insert(0, f"{role.title()}: {sanitized_content}")
                        token_count += msg_tokens
                    else:
                        break
            
            if context_parts:
                conversation_context = "\n\nRecent Conversation:\n" + "\n".join(context_parts)
        
        # Assemble final context with slot separation
        context_slots = {
            'system_prompt': system_prompt,
            'retrieval_context': retrieval_context,
            'conversation_context': conversation_context,
            'current_input': f"\n\nUser: {user_input}\nAssistant:"
        }
        
        # Calculate token usage per slot
        slot_tokens = {}
        total_tokens = 0
        
        for slot_name, content in context_slots.items():
            if content:
                tokens = count_tokens_simple(content)
                slot_tokens[slot_name] = tokens
                total_tokens += tokens
        
        # Truncate if over budget (preserve system prompt priority)
        if total_tokens > self.max_context_tokens:
            # Trim retrieval context first
            if 'retrieval_context' in context_slots and context_slots['retrieval_context']:
                excess = total_tokens - self.max_context_tokens
                current_length = len(context_slots['retrieval_context'])
                new_length = max(100, current_length - (excess * 4))  # Rough char estimate
                context_slots['retrieval_context'] = context_slots['retrieval_context'][:new_length] + "..."
        
        # Build the final prompt as a structured dictionary for the instructor library
        final_prompt = {
            "system_prompt": context_slots.get('system_prompt', ''),
            "retrieval_context": context_slots.get('retrieval_context', ''),
            "conversation_history": context_slots.get('conversation_context', ''),
            "user_input": user_input  # Pass the raw user input separately
        }

        # Calculate final token count based on a representative string version
        final_token_count = sum(slot_tokens.values()) + count_tokens_simple(user_input)

        return {
            "final_prompt": final_prompt, # This is now a dictionary
            "total_tokens": final_token_count,
            "slot_tokens": slot_tokens,
            "slots_used": list(context_slots.keys()),
            "prompt_hash": prompt_registry._hashes.get("system_core_v1", "unknown")
        }
    

# For compatibility
ContextBuildResult = Dict[str, Any]
