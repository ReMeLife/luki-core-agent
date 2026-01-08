"""
Token usage optimization for LUKi Core Agent
Provides intelligent token counting, budget management, and context optimization
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget allocation for different context components"""
    system_prompt: int = 4096
    retrieval_context: int = 2048
    knowledge_context: int = 4096
    conversation_history: int = 2048
    current_input: int = 512
    response_reserve: int = 4096
    
    @property
    def total_input_budget(self) -> int:
        """Total tokens allocated for input"""
        return (
            self.system_prompt +
            self.retrieval_context +
            self.knowledge_context +
            self.conversation_history +
            self.current_input
        )
    
    @property
    def total_budget(self) -> int:
        """Total tokens including response reserve"""
        return self.total_input_budget + self.response_reserve


class TokenCounter:
    """Accurate token counting using tiktoken"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token counter
        
        Args:
            model_name: Model name for encoding (default: gpt-3.5-turbo for general use)
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in message list (ChatML format)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Total token count including message formatting overhead
        """
        tokens = 0
        for message in messages:
            # Add tokens for message formatting
            tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            
            for key, value in message.items():
                tokens += self.count_tokens(str(value))
                if key == "name":
                    tokens += -1  # Role is omitted if name is present
        
        tokens += 2  # Every reply is primed with <im_start>assistant
        return tokens
    
    def truncate_to_budget(self, text: str, max_tokens: int, strategy: str = "end") -> str:
        """
        Truncate text to fit within token budget
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            strategy: 'start', 'end', or 'middle' - where to truncate
        
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if strategy == "start":
            # Keep the end
            truncated_tokens = tokens[-max_tokens:]
            return "..." + self.encoding.decode(truncated_tokens)
        
        elif strategy == "end":
            # Keep the start (most common)
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens) + "..."
        
        elif strategy == "middle":
            # Keep start and end, remove middle
            keep_each = max_tokens // 2
            start_tokens = tokens[:keep_each]
            end_tokens = tokens[-keep_each:]
            return self.encoding.decode(start_tokens) + "\n...[truncated]...\n" + self.encoding.decode(end_tokens)
        
        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")


class ContextOptimizer:
    """Optimize context to fit within token budgets"""
    
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.token_counter = token_counter or TokenCounter()
    
    def optimize_context(
        self,
        system_prompt: str,
        retrieval_context: List[str],
        knowledge_context: List[str],
        conversation_history: List[Dict[str, str]],
        current_input: str,
        budget: TokenBudget
    ) -> Dict[str, Any]:
        """
        Optimize all context components to fit within budget
        
        Returns:
            Dict with optimized components and token usage stats
        """
        result = {
            "system_prompt": system_prompt,
            "retrieval_context": [],
            "knowledge_context": [],
            "conversation_history": [],
            "current_input": current_input,
            "token_usage": {},
            "truncated": {}
        }
        
        # 1. System prompt (highest priority, minimal truncation)
        system_tokens = self.token_counter.count_tokens(system_prompt)
        if system_tokens > budget.system_prompt:
            logger.warning(f"System prompt exceeds budget: {system_tokens} > {budget.system_prompt}")
            result["system_prompt"] = self.token_counter.truncate_to_budget(
                system_prompt, budget.system_prompt, strategy="end"
            )
            result["truncated"]["system_prompt"] = True
        result["token_usage"]["system_prompt"] = min(system_tokens, budget.system_prompt)
        
        # 2. Current input (never truncate user input)
        input_tokens = self.token_counter.count_tokens(current_input)
        result["token_usage"]["current_input"] = input_tokens
        
        # 3. Knowledge context (high priority, preserve completeness)
        knowledge_tokens = 0
        for item in knowledge_context:
            item_tokens = self.token_counter.count_tokens(item)
            if knowledge_tokens + item_tokens <= budget.knowledge_context:
                result["knowledge_context"].append(item)
                knowledge_tokens += item_tokens
            else:
                # Try to fit partial item if space remains
                remaining = budget.knowledge_context - knowledge_tokens
                if remaining > 100:  # Only if meaningful space remains
                    truncated = self.token_counter.truncate_to_budget(item, remaining, strategy="end")
                    result["knowledge_context"].append(truncated)
                    knowledge_tokens += remaining
                    result["truncated"]["knowledge_context"] = True
                break
        result["token_usage"]["knowledge_context"] = knowledge_tokens
        
        # 4. Retrieval context (user memories, important but can be selective)
        retrieval_tokens = 0
        for item in retrieval_context:
            item_tokens = self.token_counter.count_tokens(item)
            if retrieval_tokens + item_tokens <= budget.retrieval_context:
                result["retrieval_context"].append(item)
                retrieval_tokens += item_tokens
            else:
                break
        result["token_usage"]["retrieval_context"] = retrieval_tokens
        
        # 5. Conversation history (keep most recent, can truncate older)
        history_tokens = 0
        # Process in reverse to keep most recent messages
        for message in reversed(conversation_history):
            message_tokens = self.token_counter.count_tokens(
                f"{message.get('role', '')}: {message.get('content', '')}"
            )
            if history_tokens + message_tokens <= budget.conversation_history:
                result["conversation_history"].insert(0, message)
                history_tokens += message_tokens
            else:
                result["truncated"]["conversation_history"] = True
                break
        result["token_usage"]["conversation_history"] = history_tokens
        
        # Calculate totals
        total_input_tokens = sum(result["token_usage"].values())
        result["token_usage"]["total_input"] = total_input_tokens
        result["token_usage"]["budget_remaining"] = budget.total_input_budget - total_input_tokens
        result["token_usage"]["budget_utilization"] = total_input_tokens / budget.total_input_budget
        
        # Log if over budget
        if total_input_tokens > budget.total_input_budget:
            logger.error(
                f"Context exceeds budget: {total_input_tokens} > {budget.total_input_budget}",
                extra={"token_usage": result["token_usage"]}
            )
        
        return result
    
    def prioritize_retrieval_items(
        self,
        items: List[Dict[str, Any]],
        max_tokens: int,
        score_key: str = "score"
    ) -> List[str]:
        """
        Select highest-priority retrieval items within token budget
        
        Args:
            items: List of retrieval items with scores
            max_tokens: Maximum tokens to use
            score_key: Key for relevance score
        
        Returns:
            List of selected item texts
        """
        # Sort by score (descending)
        sorted_items = sorted(items, key=lambda x: x.get(score_key, 0), reverse=True)
        
        selected = []
        total_tokens = 0
        
        for item in sorted_items:
            text = item.get("text", "") or item.get("content", "")
            tokens = self.token_counter.count_tokens(text)
            
            if total_tokens + tokens <= max_tokens:
                selected.append(text)
                total_tokens += tokens
            else:
                break
        
        logger.debug(
            f"Selected {len(selected)}/{len(items)} retrieval items using {total_tokens}/{max_tokens} tokens"
        )
        
        return selected


# Global instances
default_token_counter = TokenCounter()
default_optimizer = ContextOptimizer(default_token_counter)
