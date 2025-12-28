"""
Context Builder for LUKi Agent - Hardened Version

Builds context for LLM prompts with strict slot separation, sanitization, and token budgets.
"""

from typing import Dict, List, Optional, Any
import logging
import re
from datetime import datetime

from .prompt_registry import prompt_registry
from .features.tiers import infer_tier_from_balance

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
        # CRITICAL: Increased context limit - modern LLMs support 128k+ tokens
        # Previous limit of 2048 was causing unnecessary truncation
        # Together AI models support much larger contexts
        self.max_context_tokens = 16384  # 16k tokens for context (out of 128k+ model capacity)
        
        # Token budget allocation per slot (generous budgets for accuracy)
        self.slot_budgets = {
            'system_prompt': 4096,      # Full system prompt without truncation
            'retrieval_context': 2048,  # User memories
            'knowledge_context': 4096,  # Platform knowledge (NO truncation)
            'conversation_history': 2048, # Recent conversation
            'current_input': 512        # User query
        }
    
    async def build(
        self,
        user_input: str,
        user_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        knowledge_context: Optional[List[Dict[str, Any]]] = None,  # NEW: Separate knowledge
        wallet_context: Optional[Dict[str, Any]] = None,
        personality_mode: str = "default",
        include_safety: bool = True,
        world_day_context: Optional[Dict[str, Any]] = None,  # World day awareness
        client_tag: Optional[str] = None  # Widget mode detection
    ) -> Dict[str, Any]:
        """Build context with strict slot separation and sanitization"""
        
        # ALWAYS use FULL prompt for ALL users - no quality degradation for anonymous users
        # The prompt_registry already handles auth status via _get_user_guidance()
        # which adds appropriate messaging about ELR/memory access based on user_id
        system_prompt = prompt_registry.build_system_prompt(
            user_id=user_id,
            personality_mode=personality_mode,
            include_safety=include_safety,
            client_tag=client_tag
        )

        if wallet_context:
            try:
                wallet_connected = bool(
                    wallet_context.get("connected")
                    or wallet_context.get("wallet_address")
                )
                balance = wallet_context.get("luki_balance")
                inferred_tier = None
                if isinstance(balance, (int, float)):
                    inferred_tier = infer_tier_from_balance(balance)
                tier = (
                    wallet_context.get("tier")
                    or wallet_context.get("user_tier")
                    or inferred_tier
                    or "basic"
                )
                has_genesis = bool(
                    wallet_context.get("has_genesis_nft")
                    or wallet_context.get("genesis_holder")
                )

                lines = [
                    "## Wallet & On-Chain Status:",
                    f"- Wallet connected: {'yes' if wallet_connected else 'no'}",
                    f"- On-chain tier: {tier}",
                    f"- Genesis NFT holder: {'yes' if has_genesis else 'no'}",
                ]
                if isinstance(balance, (int, float)):
                    lines.append(f"- Approx. $LUKI balance: {balance}")

                system_prompt = system_prompt + "\n\n" + "\n".join(lines)
            except Exception:
                pass

        # Add world day context if provided (for "today's world day" awareness)
        if world_day_context:
            try:
                world_day_name = world_day_context.get("name", "")
                world_day_desc = world_day_context.get("description", "")
                world_day_fact = world_day_context.get("fun_fact", "")
                world_day_emoji = world_day_context.get("emoji", "🌍")
                
                if world_day_name:
                    today_str = datetime.now().strftime("%B %d")
                    world_day_lines = [
                        f"## Today's World Day ({today_str}):",
                        f"- {world_day_emoji} **{world_day_name}**",
                        f"- {world_day_desc}" if world_day_desc else "",
                        f"- Fun fact: {world_day_fact}" if world_day_fact else "",
                        "",
                        "If the user asks about today's world day, special day, or what day it is, share this information in your signature LUKi style. Don't force it into unrelated conversations."
                    ]
                    system_prompt = system_prompt + "\n\n" + "\n".join(line for line in world_day_lines if line)
                    print(f"🌍 Added world day context: {world_day_name}")
            except Exception as e:
                print(f"⚠️ Failed to add world day context: {e}")

        # Determine auth status for logging
        is_authenticated = (
            user_id and 
            user_id != 'anonymous_base_user' and
            user_id.lower() not in ('anonymous', 'guest') and
            not user_id.startswith('anonymous_')
        )
        auth_status = "authenticated" if is_authenticated else "anonymous"
        print(f"✅ Using FULL prompt for {auth_status} user (query: {len(user_input)} chars)")
        
        # Build sanitized retrieval context
        retrieval_context = ""
        
        if memory_context:
            # Log what we received
            print(f"📦 ContextBuilder: Received {len(memory_context)} memory items")
            
            # Sanitize and filter memory context with temporal awareness
            sanitized_memories = []
            for item in memory_context[:10]:  # Increase limit to show more memories
                content = item.get("content", "")
                sanitized_content = sanitize_retrieval_context(content)
                
                # Extract timestamp if available
                created_at = None
                if "created_at" in item:
                    try:
                        # Handle both datetime objects and ISO strings
                        if isinstance(item["created_at"], str):
                            created_at = datetime.fromisoformat(item["created_at"].replace('Z', '+00:00'))
                        elif isinstance(item["created_at"], datetime):
                            created_at = item["created_at"]
                    except (ValueError, TypeError) as e:
                        print(f"  Warning: Failed to parse created_at timestamp: {e}")
                
                # Log each memory being processed
                if sanitized_content:
                    date_str = f" (saved {created_at.strftime('%Y-%m-%d')})" if created_at else ""
                    print(f"  Processing memory: {sanitized_content[:50]}...{date_str}")
                
                if sanitized_content and len(sanitized_content) > 10:
                    # Don't truncate memories too aggressively - allow fuller content
                    # With retrieval_context budget of 1200, this gives us 400 chars per memory
                    max_length = min(600, self.slot_budgets['retrieval_context'] // 3)
                    if len(sanitized_content) > max_length:
                        # Truncate at sentence boundary
                        truncated = sanitized_content[:max_length]
                        last_period = truncated.rfind('.')
                        if last_period > max_length // 2:
                            sanitized_content = truncated[:last_period + 1]
                        else:
                            sanitized_content = truncated + "..."
                    
                    # Format memory with timestamp if available
                    if created_at:
                        # Calculate how long ago
                        now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.utcnow()
                        days_ago = (now - created_at).days
                        
                        if days_ago == 0:
                            time_context = "(saved today)"
                        elif days_ago == 1:
                            time_context = "(saved yesterday)"
                        elif days_ago < 7:
                            time_context = f"(saved {days_ago} days ago)"
                        elif days_ago < 30:
                            weeks_ago = days_ago // 7
                            time_context = f"(saved {weeks_ago} week{'s' if weeks_ago > 1 else ''} ago)"
                        elif days_ago < 365:
                            months_ago = days_ago // 30
                            time_context = f"(saved {months_ago} month{'s' if months_ago > 1 else ''} ago)"
                        else:
                            years_ago = days_ago // 365
                            time_context = f"(saved {years_ago} year{'s' if years_ago > 1 else ''} ago)"
                        
                        formatted_memory = f"{sanitized_content} {time_context}"
                    else:
                        formatted_memory = sanitized_content
                    
                    sanitized_memories.append(formatted_memory)
            
            if sanitized_memories:
                retrieval_context = "\n\nRelevant Context:\n" + "\n".join(f"- {mem}" for mem in sanitized_memories)
                print(f"🎉 ContextBuilder: Built retrieval context with {len(sanitized_memories)} memories")
            else:
                print(f"⚠️ ContextBuilder: No valid memories after sanitization")
        
        # Build knowledge context SEPARATELY from memories
        knowledge_text = ""
        if knowledge_context:
            print(f"📚 ContextBuilder: Processing {len(knowledge_context)} knowledge items")
            knowledge_items = []
            for item in knowledge_context[:8]:  # Use all retrieved chunks (increased from 5)
                content = item.get("content", "")
                if content:
                    # NO TRUNCATION - Use complete semantic chunks from ProjectKB
                    # ProjectKB already returns semantically meaningful chunks
                    # Truncating defeats the purpose of semantic chunking
                    knowledge_items.append(content)  # FULL content, no character limits
            
            if knowledge_items:
                knowledge_text = "\n\nPlatform Knowledge:\n" + "\n".join(f"- {k}" for k in knowledge_items)
                total_chars = sum(len(k) for k in knowledge_items)
                print(f"🎯 ContextBuilder: Built knowledge context with {len(knowledge_items)} items ({total_chars} chars total)")
        
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
                # Calculate conversation depth
                total_messages = len(conversation_history) if conversation_history else 0
                conversation_context = (
                    f"\n\n## RECENT CONVERSATION - Message #{total_messages} (ONGOING - NO GREETINGS!):\n" + 
                    "\n".join(context_parts) +
                    "\n\n## CRITICAL CONVERSATION RULES:\n"
                    f"1. **CONVERSATION STATE**: You are at message #{total_messages} in an ongoing conversation\n"
                    "2. **NEVER GREET**: Do not say 'Hey!', 'Hello!', 'Hi!' or introduce yourself mid-conversation\n"
                    "3. **FOLLOW-UP DETECTION**: If user's message is clearly answering your previous question, treat it as such\n"
                    "4. **SHORT RESPONSES**: Single words/locations are often answers, not new queries\n"
                    "5. **CONTINUITY**: Reference earlier discussion when relevant, maintain natural flow\n"
                    "6. **Examples:**\n"
                    "   - You ask: 'Where are you checking weather?' → User says: 'london' → This is an ANSWER\n"
                    "   - You ask: 'What type of help?' → User says: 'avatar' → This is an ANSWER\n"
                    "   - Message #15: User asks question → DO NOT say 'Hey!' before answering\n"
                )
        
        # Assemble final context with slot separation
        context_slots = {
            'system_prompt': system_prompt,
            'retrieval_context': retrieval_context,  # User memories ONLY
            'knowledge_context': knowledge_text,      # Platform knowledge ONLY
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
        
        # Only truncate if DRASTICALLY over budget (preserve accuracy)
        if total_tokens > self.max_context_tokens:
            print(f"⚠️ Context exceeds budget: {total_tokens} > {self.max_context_tokens}")
            # Trim conversation history first (least critical for accuracy)
            if 'conversation_context' in context_slots and context_slots['conversation_context']:
                excess = total_tokens - self.max_context_tokens
                current_length = len(context_slots['conversation_context'])
                new_length = max(200, current_length - (excess * 4))
                context_slots['conversation_context'] = context_slots['conversation_context'][:new_length] + "..."
                print(f"📉 Trimmed conversation history to fit budget")
        
        # Build the final prompt as a structured dictionary for the instructor library
        final_prompt = {
            "system_prompt": context_slots.get('system_prompt', ''),
            "retrieval_context": context_slots.get('retrieval_context', ''),  # User memories
            "knowledge_context": context_slots.get('knowledge_context', ''),  # Platform knowledge
            "conversation_history": context_slots.get('conversation_context', ''),
            "user_input": user_input,  # Pass the raw user input separately
            "user_id": user_id,  # Include user_id for function calling decisions
            "raw_conversation_history": conversation_history or [],
            "wallet_context": wallet_context or {},
            "personality_mode": personality_mode,
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
