"""
Generic System Prompts Stub
Replace with your own system prompt definitions.
"""

from typing import Dict, Any

# Generic system prompts - customize for your application
SYSTEM_PROMPTS = {
    "default": {
        "role": "system",
        "content": """You are a helpful AI assistant. Provide accurate, helpful, and safe responses to user queries.

Key guidelines:
- Be helpful and informative
- Maintain a professional tone
- Respect user privacy and safety
- Provide accurate information based on your training
- Ask for clarification when needed"""
    },
    
    "conversational": {
        "role": "system", 
        "content": """You are a friendly conversational AI assistant. Engage naturally with users while being helpful and informative.

Guidelines:
- Use a warm, approachable tone
- Show empathy and understanding
- Provide thoughtful responses
- Ask follow-up questions when appropriate
- Maintain conversation flow"""
    },
    
    "analytical": {
        "role": "system",
        "content": """You are an analytical AI assistant focused on providing detailed, structured responses.

Guidelines:
- Break down complex topics systematically
- Provide evidence-based information
- Use clear reasoning and logic
- Organize responses with clear structure
- Cite sources when relevant"""
    }
}

def get_system_prompt(prompt_type: str = "default") -> Dict[str, Any]:
    """
    Get system prompt by type.
    
    Args:
        prompt_type: Type of system prompt to retrieve
        
    Returns:
        System prompt dictionary with role and content
    """
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])

def get_available_prompts() -> list:
    """Get list of available system prompt types"""
    return list(SYSTEM_PROMPTS.keys())

def add_custom_prompt(name: str, content: str) -> None:
    """
    Add a custom system prompt.
    
    Args:
        name: Name/key for the prompt
        content: System prompt content
    """
    SYSTEM_PROMPTS[name] = {
        "role": "system",
        "content": content
    }
