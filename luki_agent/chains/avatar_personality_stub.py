"""
Generic Avatar Personality Stub
Replace with your own personality framework and conversation logic.
"""

from typing import Dict, Any, List
from enum import Enum

class PersonalityTrait(Enum):
    """Generic personality traits - customize for your use case"""
    HELPFUL = "helpful"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"

class CommunicationStyle(Enum):
    """Communication styles - customize for your domain"""
    SUPPORTIVE = "supportive"
    INFORMATIVE = "informative"
    CASUAL = "casual"
    FORMAL = "formal"

class PersonalityContext:
    """Context for personality decisions"""
    def __init__(self, user_message: str, conversation_history: List[Dict] = None):
        self.user_message = user_message
        self.conversation_history = conversation_history or []
        self.detected_mood = "neutral"
        self.context_type = "general"

class PersonalityResponse:
    """Response configuration from personality system"""
    def __init__(self):
        self.core_traits = [PersonalityTrait.HELPFUL, PersonalityTrait.PROFESSIONAL]
        self.communication_style = CommunicationStyle.SUPPORTIVE
        self.behavioral_guidelines = ["Be helpful", "Stay professional", "Provide accurate information"]

class LukiAvatarPersonality:
    """
    Generic Avatar Personality System - Stub Implementation
    Replace with your own personality framework.
    """
    
    def __init__(self):
        # Generic personality traits mapping
        self.personality_traits = {
            PersonalityTrait.HELPFUL: {
                "description": "Provides useful assistance",
                "weight": 0.8
            },
            PersonalityTrait.PROFESSIONAL: {
                "description": "Maintains professional demeanor", 
                "weight": 0.7
            },
            PersonalityTrait.FRIENDLY: {
                "description": "Warm and approachable",
                "weight": 0.6
            },
            PersonalityTrait.EMPATHETIC: {
                "description": "Understanding and supportive",
                "weight": 0.9
            }
        }
        
        # Generic communication patterns
        self.communication_patterns = {
            CommunicationStyle.SUPPORTIVE: {
                "tone": "warm and encouraging",
                "language": "accessible and clear",
                "formality": "moderate"
            },
            CommunicationStyle.INFORMATIVE: {
                "tone": "clear and factual",
                "language": "precise and detailed", 
                "formality": "professional"
            },
            CommunicationStyle.CASUAL: {
                "tone": "relaxed and friendly",
                "language": "conversational",
                "formality": "low"
            },
            CommunicationStyle.FORMAL: {
                "tone": "respectful and structured",
                "language": "formal and precise",
                "formality": "high"
            }
        }
    
    def generate_personality_response(
        self, 
        context: PersonalityContext,
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> PersonalityResponse:
        """
        Generate personality response configuration.
        
        This is a stub implementation - replace with your own personality logic.
        """
        response = PersonalityResponse()
        
        # Simple keyword-based personality selection
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["help", "support", "assistance"]):
            response.core_traits = [PersonalityTrait.HELPFUL, PersonalityTrait.EMPATHETIC]
            response.communication_style = CommunicationStyle.SUPPORTIVE
        elif any(word in message_lower for word in ["information", "explain", "what is"]):
            response.core_traits = [PersonalityTrait.PROFESSIONAL, PersonalityTrait.HELPFUL]
            response.communication_style = CommunicationStyle.INFORMATIVE
        else:
            response.core_traits = [PersonalityTrait.FRIENDLY, PersonalityTrait.HELPFUL]
            response.communication_style = CommunicationStyle.CASUAL
            
        return response
