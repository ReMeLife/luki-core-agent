"""
User Context Detection for LUKi
Determines if a user is new or existing and adapts greeting/behavior accordingly.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class UserContextDetector:
    """Detects user context to adapt LUKi's behavior for new vs existing users."""
    
    def __init__(self, memory_client=None):
        """
        Initialize user context detector.
        
        Args:
            memory_client: Memory service client for ELR data access
        """
        self.memory_client = memory_client
        
    async def analyze_user_context(self, user_id: str, session_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze user context to determine appropriate interaction approach.
        
        Args:
            user_id: User identifier
            session_history: Current session conversation history
            
        Returns:
            Context analysis with user status and recommended approach
        """
        try:
            context = {
                "user_id": user_id,
                "user_status": "unknown",
                "has_elr_data": False,
                "interaction_count": 0,
                "relationship_stage": "first_contact",
                "recommended_greeting": "new_user",
                "personalization_level": "minimal",
                "available_data_types": [],
                "confidence_score": 0.0
            }
            
            # Check for existing ELR data
            if self.memory_client is not None:
                elr_data = await self._check_elr_data(user_id)
                context.update(elr_data)
            
            # Analyze session history
            if session_history:
                session_analysis = self._analyze_session_history(session_history)
                context.update(session_analysis)
            
            # Determine user status and recommendations
            context = self._determine_user_status(context)
            context = self._generate_recommendations(context)
            
            logger.info(f"User context analysis complete for {user_id}: {context['user_status']}")
            return context
            
        except Exception as e:
            logger.error(f"User context analysis failed for {user_id}: {e}")
            return self._get_fallback_context(user_id)
    
    async def _check_elr_data(self, user_id: str) -> Dict[str, Any]:
        """Check for existing ELR data for the user."""
        try:
            # Query memory service for user's ELR data
            if self.memory_client is None:
                return {
                    "has_elr_data": False,
                    "available_data_types": [],
                    "elr_data_count": 0
                }
                
            search_results = await self.memory_client.search_memories(
                query="",  # Empty query to get all user data
                user_id=user_id,
                limit=10
            )
            
            has_data = len(search_results.get('results', [])) > 0
            data_types = []
            
            if has_data:
                # Analyze types of data available
                for result in search_results.get('results', []):
                    metadata = result.get('metadata', {})
                    section = metadata.get('section', '')
                    if section and section not in data_types:
                        data_types.append(section)
            
            return {
                "has_elr_data": has_data,
                "available_data_types": data_types,
                "elr_data_count": len(search_results.get('results', []))
            }
            
        except Exception as e:
            logger.warning(f"Could not check ELR data for {user_id}: {e}")
            return {
                "has_elr_data": False,
                "available_data_types": [],
                "elr_data_count": 0
            }
    
    def _analyze_session_history(self, session_history: List[Dict]) -> Dict[str, Any]:
        """Analyze current session history for interaction patterns."""
        interaction_count = len([turn for turn in session_history if turn.get('role') == 'user'])
        
        # Check if user has introduced themselves
        user_introduced = False
        for turn in session_history:
            if turn.get('role') == 'user':
                content = turn.get('content', '').lower()
                if any(phrase in content for phrase in ['my name is', 'i am', "i'm", 'call me']):
                    user_introduced = True
                    break
        
        return {
            "interaction_count": interaction_count,
            "user_introduced": user_introduced,
            "session_length": len(session_history)
        }
    
    def _determine_user_status(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine user status based on available data."""
        has_elr = context.get("has_elr_data", False)
        interaction_count = context.get("interaction_count", 0)
        
        if has_elr and interaction_count > 0:
            context["user_status"] = "returning_with_data"
            context["relationship_stage"] = "established"
            context["confidence_score"] = 0.9
        elif has_elr and interaction_count == 0:
            context["user_status"] = "first_session_with_data"
            context["relationship_stage"] = "reconnecting"
            context["confidence_score"] = 0.8
        elif not has_elr and interaction_count > 3:
            context["user_status"] = "new_user_engaging"
            context["relationship_stage"] = "building"
            context["confidence_score"] = 0.7
        elif not has_elr and interaction_count > 0:
            context["user_status"] = "new_user_exploring"
            context["relationship_stage"] = "introduction"
            context["confidence_score"] = 0.6
        else:
            context["user_status"] = "completely_new"
            context["relationship_stage"] = "first_contact"
            context["confidence_score"] = 0.5
        
        return context
    
    def _generate_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for interaction approach."""
        user_status = context.get("user_status", "unknown")
        
        greeting_map = {
            "completely_new": "introduce_and_explain",
            "new_user_exploring": "warm_continuation",
            "new_user_engaging": "relationship_building",
            "first_session_with_data": "reconnect_with_context",
            "returning_with_data": "personalized_welcome"
        }
        
        personalization_map = {
            "completely_new": "minimal",
            "new_user_exploring": "basic",
            "new_user_engaging": "moderate",
            "first_session_with_data": "high",
            "returning_with_data": "maximum"
        }
        
        context["recommended_greeting"] = greeting_map.get(user_status, "introduce_and_explain")
        context["personalization_level"] = personalization_map.get(user_status, "minimal")
        
        return context
    
    def _get_fallback_context(self, user_id: str) -> Dict[str, Any]:
        """Fallback context if analysis fails."""
        return {
            "user_id": user_id,
            "user_status": "unknown",
            "has_elr_data": False,
            "interaction_count": 0,
            "relationship_stage": "first_contact",
            "recommended_greeting": "introduce_and_explain",
            "personalization_level": "minimal",
            "available_data_types": [],
            "confidence_score": 0.0
        }
    
    def get_adaptive_greeting(self, context: Dict[str, Any]) -> str:
        """Generate appropriate greeting based on user context."""
        greeting_type = context.get("recommended_greeting", "introduce_and_explain")
        user_status = context.get("user_status", "unknown")
        has_elr = context.get("has_elr_data", False)
        data_types = context.get("available_data_types", [])
        
        greetings = {
            "introduce_and_explain": f"""Hello! I'm LUKi, your AI companion in the ReMeLife ecosystem. 

I'm here to support you with personalized care and conversation. I can learn about your preferences, interests, and life story to provide more meaningful interactions over time.

How would you like to get started today?""",
            
            "warm_continuation": f"""Hi there! Great to continue our conversation. 

I'm still getting to know you and learning about your preferences. Feel free to share anything about yourself that might help me provide better support.

What's on your mind today?""",
            
            "relationship_building": f"""Hello again! I'm enjoying getting to know you better. 

As we continue talking, I'm learning more about what matters to you. This helps me provide more personalized support and suggestions.

What would you like to explore today?""",
            
            "reconnect_with_context": f"""Welcome back! I'm LUKi, and I have access to some information about your preferences and interests from your RemindMeCare activities.

I can use these insights to provide personalized recommendations and support. Would you like me to suggest some activities based on what I know about you, or is there something specific you'd like to discuss?""",
            
            "personalized_welcome": f"""Hello! It's wonderful to see you again. Based on your Electronic Life Record, I can see you've been engaging with {', '.join(data_types[:2]) if data_types else 'various activities'}.

I'm here to provide personalized support using your life patterns and preferences. How can I help you today?"""
        }
        
        return greetings.get(greeting_type, greetings["introduce_and_explain"])


def create_user_context_detector(memory_client=None) -> UserContextDetector:
    """Factory function to create user context detector."""
    return UserContextDetector(memory_client)
