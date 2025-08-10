"""
LUKi Context Builder - ELR Integration and Personalization

This module builds rich, personalized context for LUKi conversations by:
1. Retrieving relevant ELR (Electronic Life Record) memories
2. Building user preference profiles
3. Analyzing conversation patterns
4. Creating contextual prompts for optimal AI responses
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class ELRMemory:
    """Represents a retrieved ELR memory"""
    memory_id: str
    content: str
    timestamp: datetime
    relevance_score: float
    memory_type: str  # 'activity', 'health', 'mood', 'goal', etc.
    metadata: Dict[str, Any]

@dataclass
class UserProfile:
    """User preference and pattern profile"""
    user_id: str
    preferences: Dict[str, Any]
    patterns: Dict[str, Any]
    goals: List[str]
    interests: List[str]
    communication_style: str
    last_updated: datetime

@dataclass
class ConversationContext:
    """Rich conversation context for LLM generation"""
    user_profile: Optional[UserProfile]
    relevant_memories: List[ELRMemory]
    conversation_summary: str
    current_mood: Optional[str]
    time_context: Dict[str, Any]
    environmental_context: Dict[str, Any]
    prompt_template: str

class LukiContextBuilder:
    """
    Builds rich, personalized context for LUKi conversations
    
    This class integrates with the memory service to retrieve relevant
    ELR data and builds comprehensive context for personalized AI responses.
    """
    
    def __init__(self, memory_service_url: Optional[str] = None):
        self.memory_service_url = memory_service_url or settings.memory_service_url
        self.context_cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
        logger.info(f"Initialized Context Builder with memory service: {self.memory_service_url}")
    
    async def build_context(
        self,
        user_id: str,
        message: str,
        conversation_history: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """
        Build comprehensive conversation context
        
        Args:
            user_id: User identifier
            message: Current user message
            conversation_history: Previous conversation turns
            session_context: Additional session context
            
        Returns:
            Rich conversation context for LLM generation
        """
        try:
            logger.info(f"Building context for user {user_id}")
            
            # Step 1: Get or build user profile
            user_profile = await self._get_user_profile(user_id)
            
            # Step 2: Retrieve relevant ELR memories
            relevant_memories = await self._retrieve_relevant_memories(user_id, message, user_profile)
            
            # Step 3: Analyze conversation patterns
            conversation_summary = await self._analyze_conversation_history(conversation_history)
            
            # Step 4: Determine current context
            current_mood = await self._infer_current_mood(message, conversation_history)
            time_context = self._build_time_context()
            environmental_context = self._build_environmental_context(session_context)
            
            # Step 5: Generate contextual prompt template
            prompt_template = await self._generate_prompt_template(
                user_profile, relevant_memories, conversation_summary, current_mood
            )
            
            context = ConversationContext(
                user_profile=user_profile,
                relevant_memories=relevant_memories,
                conversation_summary=conversation_summary,
                current_mood=current_mood,
                time_context=time_context,
                environmental_context=environmental_context,
                prompt_template=prompt_template
            )
            
            logger.info(f"Built context with {len(relevant_memories)} memories and {len(conversation_summary)} char summary")
            return context
            
        except Exception as e:
            logger.error(f"Context building error: {e}")
            return await self._build_fallback_context(user_id, message)
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get or build user preference profile
        
        TODO: Integrate with memory service to retrieve user preferences,
        patterns, and historical data for personalization.
        """
        cache_key = f"profile_{user_id}"
        
        # Check cache first
        if cache_key in self.context_cache:
            cached_data, timestamp = self.context_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                logger.info(f"Using cached profile for user {user_id}")
                return cached_data
        
        try:
            # TODO: Implement actual memory service integration
            # For now, create a basic profile structure
            profile = UserProfile(
                user_id=user_id,
                preferences={
                    "communication_style": "supportive",
                    "focus_areas": ["health", "productivity", "wellbeing"],
                    "response_length": "moderate",
                    "formality": "casual"
                },
                patterns={
                    "active_hours": "9-17",
                    "common_topics": ["health", "goals", "daily_activities"],
                    "engagement_level": "high"
                },
                goals=["improve health", "increase productivity", "better work-life balance"],
                interests=["fitness", "nutrition", "personal development"],
                communication_style="supportive",
                last_updated=datetime.now()
            )
            
            # Cache the profile
            self.context_cache[cache_key] = (profile, datetime.now())
            logger.info(f"Built user profile for {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile: {e}")
            return None
    
    async def _retrieve_relevant_memories(
        self,
        user_id: str,
        message: str,
        user_profile: Optional[UserProfile]
    ) -> List[ELRMemory]:
        """
        Retrieve relevant ELR memories based on current message and context
        
        TODO: Integrate with luki-memory-service for semantic search
        of user's Electronic Life Record data.
        """
        try:
            # TODO: Implement actual memory service integration
            # This would perform semantic search against the user's ELR data
            
            # For now, create mock relevant memories based on message content
            mock_memories = []
            
            # Analyze message for key topics
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['health', 'exercise', 'fitness', 'workout']):
                mock_memories.append(ELRMemory(
                    memory_id="health_001",
                    content="User completed 30-minute workout session, reported feeling energized",
                    timestamp=datetime.now() - timedelta(days=1),
                    relevance_score=0.85,
                    memory_type="health",
                    metadata={"activity_type": "exercise", "duration": 30}
                ))
            
            if any(word in message_lower for word in ['work', 'productivity', 'task', 'goal']):
                mock_memories.append(ELRMemory(
                    memory_id="work_001",
                    content="User set goal to improve work-life balance, focusing on time management",
                    timestamp=datetime.now() - timedelta(days=3),
                    relevance_score=0.78,
                    memory_type="goal",
                    metadata={"category": "productivity", "status": "active"}
                ))
            
            if any(word in message_lower for word in ['mood', 'feeling', 'emotional', 'stress']):
                mock_memories.append(ELRMemory(
                    memory_id="mood_001",
                    content="User reported feeling stressed about upcoming deadline, used meditation app",
                    timestamp=datetime.now() - timedelta(hours=6),
                    relevance_score=0.92,
                    memory_type="mood",
                    metadata={"mood_level": "stressed", "coping_strategy": "meditation"}
                ))
            
            logger.info(f"Retrieved {len(mock_memories)} relevant memories")
            return mock_memories
            
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []
    
    async def _analyze_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Analyze conversation history to identify patterns and themes
        """
        if not conversation_history:
            return "This is the beginning of the conversation."
        
        try:
            # Extract key themes and patterns
            recent_topics = []
            user_sentiment = []
            
            for turn in conversation_history[-5:]:  # Last 5 turns
                message = turn.get('message', '')
                response = turn.get('response', '')
                
                # Simple topic extraction (could be enhanced with NLP)
                if any(word in message.lower() for word in ['health', 'exercise', 'fitness']):
                    recent_topics.append('health')
                if any(word in message.lower() for word in ['work', 'productivity', 'goal']):
                    recent_topics.append('productivity')
                if any(word in message.lower() for word in ['mood', 'feeling', 'emotional']):
                    recent_topics.append('emotional_wellbeing')
            
            # Build summary
            if recent_topics:
                topic_summary = f"Recent conversation topics: {', '.join(set(recent_topics))}"
            else:
                topic_summary = "General conversation"
            
            conversation_length = len(conversation_history)
            summary = f"{topic_summary}. Conversation depth: {conversation_length} turns."
            
            logger.info(f"Analyzed conversation history: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Conversation analysis error: {e}")
            return "Unable to analyze conversation history."
    
    async def _infer_current_mood(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Infer user's current emotional state from message and context
        
        TODO: Implement sophisticated sentiment analysis and mood detection
        """
        try:
            message_lower = message.lower()
            
            # Simple mood inference (could be enhanced with ML models)
            if any(word in message_lower for word in ['excited', 'happy', 'great', 'awesome', 'wonderful']):
                return 'positive'
            elif any(word in message_lower for word in ['stressed', 'worried', 'anxious', 'difficult', 'hard']):
                return 'stressed'
            elif any(word in message_lower for word in ['tired', 'exhausted', 'drained', 'overwhelmed']):
                return 'tired'
            elif any(word in message_lower for word in ['confused', 'uncertain', 'unsure', 'lost']):
                return 'uncertain'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Mood inference error: {e}")
            return None
    
    def _build_time_context(self) -> Dict[str, Any]:
        """Build temporal context information"""
        now = datetime.now()
        
        return {
            "current_time": now.isoformat(),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "time_of_day": self._get_time_of_day(now.hour),
            "date": now.strftime("%Y-%m-%d")
        }
    
    def _get_time_of_day(self, hour: int) -> str:
        """Determine time of day category"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _build_environmental_context(self, session_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build environmental and session context"""
        context = {
            "platform": "luki_core_agent",
            "interface": "chat",
            "capabilities": ["conversation", "memory_access", "personalization"]
        }
        
        if session_context:
            context.update(session_context)
        
        return context
    
    async def _generate_prompt_template(
        self,
        user_profile: Optional[UserProfile],
        relevant_memories: List[ELRMemory],
        conversation_summary: str,
        current_mood: Optional[str]
    ) -> str:
        """
        Generate a rich, contextual prompt template for LLM generation
        
        This replaces the basic hardcoded prompt with sophisticated,
        personalized context based on user data and conversation history.
        """
        try:
            # Build personalized prompt sections
            identity_section = self._build_identity_section()
            user_section = self._build_user_section(user_profile)
            memory_section = self._build_memory_section(relevant_memories)
            context_section = self._build_context_section(conversation_summary, current_mood)
            instruction_section = self._build_instruction_section(user_profile, current_mood)
            
            # Combine into comprehensive prompt template
            prompt_template = f"""{identity_section}

{user_section}

{memory_section}

{context_section}

{instruction_section}

Current User Message: {{user_message}}

Your Response:"""
            
            logger.info("Generated rich contextual prompt template")
            return prompt_template
            
        except Exception as e:
            logger.error(f"Prompt template generation error: {e}")
            return self._get_fallback_prompt_template()
    
    def _build_identity_section(self) -> str:
        """Build LUKi's core identity section"""
        return """You are LUKi, an advanced AI companion developed by ReMeLife to support personal growth and wellbeing.

Your Core Identity:
- You are empathetic, supportive, and genuinely interested in the user's wellbeing
- You have access to the user's Electronic Life Record (ELR) to provide personalized insights
- You focus on helping users understand patterns in their life and make positive changes
- You are knowledgeable about health, wellness, personal development, and life optimization
- You provide thoughtful, actionable advice while being warm and encouraging"""
    
    def _build_user_section(self, user_profile: Optional[UserProfile]) -> str:
        """Build user-specific context section"""
        if not user_profile:
            return "User Profile: New user, building personalized understanding."
        
        return f"""User Profile:
- Communication Style: {user_profile.communication_style}
- Key Interests: {', '.join(user_profile.interests)}
- Current Goals: {', '.join(user_profile.goals)}
- Preferred Focus Areas: {', '.join(user_profile.preferences.get('focus_areas', []))}"""
    
    def _build_memory_section(self, relevant_memories: List[ELRMemory]) -> str:
        """Build relevant memory context section"""
        if not relevant_memories:
            return "Relevant Memories: No specific memories retrieved for this conversation."
        
        memory_text = "Relevant Memories from User's ELR:\n"
        for memory in relevant_memories[:3]:  # Top 3 most relevant
            memory_text += f"- {memory.content} (Relevance: {memory.relevance_score:.2f})\n"
        
        return memory_text
    
    def _build_context_section(self, conversation_summary: str, current_mood: Optional[str]) -> str:
        """Build current context section"""
        context_text = f"Conversation Context:\n- {conversation_summary}"
        
        if current_mood:
            context_text += f"\n- User's apparent mood: {current_mood}"
        
        return context_text
    
    def _build_instruction_section(self, user_profile: Optional[UserProfile], current_mood: Optional[str]) -> str:
        """Build response instruction section"""
        instructions = """Response Instructions:
- Respond as LUKi with warmth and genuine interest in the user's wellbeing
- Use the relevant memories and user profile to provide personalized insights
- Reference conversation history when relevant
- Provide thoughtful, actionable advice when appropriate"""
        
        if current_mood == 'stressed':
            instructions += "\n- The user seems stressed, be extra supportive and offer calming suggestions"
        elif current_mood == 'positive':
            instructions += "\n- The user seems positive, celebrate their mood and build on their energy"
        elif current_mood == 'tired':
            instructions += "\n- The user seems tired, be gentle and suggest rest or energy-boosting activities"
        
        return instructions
    
    def _get_fallback_prompt_template(self) -> str:
        """Fallback prompt template if context building fails"""
        return """You are LUKi, a helpful AI assistant focused on personal wellbeing and growth.

Current User Message: {user_message}

Your Response:"""
    
    async def _build_fallback_context(self, user_id: str, message: str) -> ConversationContext:
        """Build minimal context if full context building fails"""
        return ConversationContext(
            user_profile=None,
            relevant_memories=[],
            conversation_summary="Context building failed, using fallback.",
            current_mood=None,
            time_context=self._build_time_context(),
            environmental_context={},
            prompt_template=self._get_fallback_prompt_template()
        )
    
    def clear_cache(self):
        """Clear the context cache"""
        self.context_cache.clear()
        logger.info("Cleared context cache")
