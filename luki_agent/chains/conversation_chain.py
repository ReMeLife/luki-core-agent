"""
LUKi Conversation Chain - Core Agent Orchestration

This module orchestrates the complete conversation flow for LUKi:
1. Session management and conversation history
2. Context building from ELR and user data
3. Avatar personality integration
4. Safety and compliance filtering
5. LLM generation with rich context
6. Response assembly and tool integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..config import settings
from ..llm_backends import LLMManager
from ..memory.session_store import SessionStore, SessionState
from ..memory.session_store import ConversationTurn as SessionConversationTurn
from .avatar_personality import LukiAvatarPersonality, PersonalityContext
from .personality_templates import PersonalityPromptTemplates
from ..knowledge.project_glossary import ProjectKnowledgeBase

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Conversation turn compatible with chain logic"""
    user_id: str
    session_id: str
    message: str
    response: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_session_turn(self) -> SessionConversationTurn:
        """Convert to session store format"""
        return SessionConversationTurn(
            role="user",  # Will be handled by session store
            content=self.message,
            timestamp=self.timestamp,
            metadata=self.metadata
        )

class LukiConversationChain:
    """
    Core conversation orchestration for LUKi Agent
    
    This class manages the complete conversation flow:
    - Session management and history
    - Context building and personalization
    - Avatar personality integration
    - Safety filtering and compliance
    - LLM generation and response assembly
    """
    
    def __init__(self, memory_service_client=None):
        self.llm_manager = LLMManager()
        self.session_store = SessionStore()  # Redis-based session storage
        self.avatar_personality = LukiAvatarPersonality()
        self.prompt_templates = PersonalityPromptTemplates()
        self.memory_service_client = memory_service_client  # Optional memory service integration
        self.knowledge_base = ProjectKnowledgeBase()  # Comprehensive glossary and platform knowledge
        self.chain_id = f"luki_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_count = 0
        
        logger.info(f"Initialized LUKi Conversation Chain with Avatar Personality: {self.chain_id}")
    
    async def process_conversation(
        self,
        user_id: str,
        session_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete conversation turn
        
        Args:
            user_id: Unique user identifier
            session_id: Session identifier
            message: User's message
            context: Additional context data
            
        Returns:
            Complete response with metadata
        """
        try:
            self.conversation_count += 1
            logger.info(f"Processing conversation {self.conversation_count} for user {user_id}")
            
            # Step 1: Load or create session
            session = await self._get_or_create_session(user_id, session_id)
            
            # Step 2: Build conversation context
            conversation_context = await self._build_conversation_context(session, message, context)
            
            # Step 3: Apply avatar personality
            personality_prompt = await self._apply_avatar_personality(conversation_context)
            
            # Step 4: Safety and compliance check
            safe_prompt = await self._apply_safety_filters(personality_prompt, message)
            
            # Step 5: Generate LLM response
            llm_response = await self._generate_llm_response(safe_prompt)
            
            # Step 6: Post-process and filter response
            final_response = await self._post_process_response(llm_response)
            
            # Step 7: Update session and save conversation
            await self._update_session(session, message, final_response, conversation_context)
            
            # Step 8: Assemble final response
            return await self._assemble_response(session, final_response, conversation_context)
            
        except Exception as e:
            logger.error(f"Conversation processing error: {e}")
            return await self._handle_error(user_id, session_id, message, str(e))
    
    async def _get_or_create_session(self, user_id: str, session_id: str) -> SessionState:
        """Load existing session or create new one"""
        try:
            session_state = self.session_store.get_session(session_id)
            if session_state:
                logger.info(f"Loaded existing session {session_id}")
                return session_state
        except Exception as e:
            logger.warning(f"Could not load session {session_id}: {e}")
        
        # Create new session
        logger.info(f"Creating new session {session_id} for user {user_id}")
        now = datetime.now()
        session = SessionState(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_activity=now,
            conversation_history=[],
            user_context={},
            agent_context={}
        )
        
        return session
    
    async def _build_conversation_context(
        self,
        session: SessionState,
        message: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build rich conversation context with Context Block Protocol (CBP)
        
        Integrates:
        - ELR memory retrieval with health checks
        - Glossary for token definitions
        - Platform KB for documentation
        - Gap notices for missing context
        """
        context = {
            "user_id": session.user_id,
            "session_id": session.session_id,
            "current_message": message,
            "conversation_history": self._format_conversation_history(session.conversation_history),
            "user_context": session.user_context,
            "agent_context": session.agent_context,
            "session_metadata": {
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "turn_count": len(session.conversation_history)
            }
        }
        
        # Add any additional context
        if additional_context:
            context.update(additional_context)
        
        # Retrieve ELR memories (mandatory for personal queries)
        elr_data = None
        is_personal = any(term in message.lower() for term in ['my', 'me', 'i\'m', 'elizabeth'])
        if is_personal and self.memory_service_client:
            try:
                elr_results = await self.memory_service_client.search_memories(
                    user_id=session.user_id,
                    query=message,
                    top_k=8
                )
                if elr_results:
                    elr_data = [{
                        'title': mem.get('content', '')[:50],
                        'synopsis': mem.get('content', '')[:120],
                        'date': mem.get('metadata', {}).get('timestamp', ''),
                        'confidence': mem.get('score', 0.0)
                    } for mem in elr_results]
                    context["elr_memories"] = elr_data
                else:
                    context["elr_gap"] = True
            except Exception as e:
                logger.warning(f"ELR retrieval failed: {e}")
                context["elr_gap"] = True
        elif is_personal and not self.memory_service_client:
            # Personal query but no memory service available
            context["elr_gap"] = True
            logger.warning("Personal query detected but memory service not available")
        
        # Always load comprehensive glossary and platform knowledge
        glossary = self.knowledge_base.get_glossary()
        platform_kb = self.knowledge_base.get_platform_knowledge()
        
        # Check for specific token/platform queries
        is_token_query = any(term in message.lower() for term in ['cap', 'reme', 'luki', 'token', 'remelife', 'remecare'])
        if is_token_query:
            # Extract relevant glossary terms
            relevant_terms = self._extract_relevant_glossary_terms(message, glossary)
            context["glossary"] = relevant_terms
        else:
            # Include essential glossary terms for context
            context["glossary"] = {k: glossary[k] for k in ['LUKi', 'ReMeLife', 'ELR', 'CAP', 'REME'] if k in glossary}
        
        # Always include platform knowledge
        context["platform_kb"] = platform_kb
        
        # Build Context Blocks using CBP
        context["context_blocks"] = self.prompt_templates.build_context_blocks(
            elr_data=elr_data,
            glossary_data=context.get("glossary"),
            platform_kb=[platform_kb] if platform_kb else None,
            query_type=message
        )
        
        logger.info(f"Built conversation context with CBP: ELR={bool(elr_data)}, Glossary={bool(context.get('glossary'))}, KB={bool(platform_kb)}")
        return context
    
    async def _apply_avatar_personality(self, context: Dict[str, Any]) -> str:
        """
        Apply LUKi's sophisticated avatar personality system
        
        This method uses the avatar personality system to create rich, contextual,
        and dynamically adaptive prompts based on user context, mood, and relationship stage.
        """
        try:
            user_message = context["current_message"]
            conversation_history = context.get("conversation_history", [])
            session_metadata = context.get("session_metadata", {})
            user_context = context.get("user_context", {})
            
            # Create personality context
            personality_context = PersonalityContext(
                user_mood=self._infer_user_mood(user_message),
                conversation_tone=self._analyze_conversation_tone(conversation_history),
                user_preferences=user_context.get("preferences", {}),
                situation_type=self._determine_situation_type(user_message),
                relationship_stage=user_context.get("relationship_stage", "developing")
            )
            
            # Generate personality response configuration
            personality_response = self.avatar_personality.generate_personality_response(
                context=personality_context,
                user_message=user_message,
                conversation_history=conversation_history
            )
            
            # Build comprehensive system prompt using templates
            personality_config = {
                'core_traits': [(trait.value, self.avatar_personality.personality_traits[trait]) 
                               for trait in personality_response.core_traits],
                'communication_style': {
                    'name': personality_response.communication_style.value,
                    'tone': self.avatar_personality.communication_patterns[personality_response.communication_style]['tone'],
                    'language': self.avatar_personality.communication_patterns[personality_response.communication_style]['language'],
                    'formality': self.avatar_personality.communication_patterns[personality_response.communication_style]['formality'],
                    'approach': 'thoughtful and personalized'
                },
                'behavioral_guidelines': personality_response.behavioral_guidelines
            }
            
            # Build system prompt with personality
            system_prompt = self.prompt_templates.build_system_prompt(
                personality_config=personality_config,
                user_context=user_context
            )
            
            # Build conversation context
            conversation_context = self.prompt_templates.build_conversation_context(
                conversation_history=conversation_history,
                user_memories=context.get("user_memories", []),
                current_context={
                    'user_mood': personality_context.user_mood,
                    'session_focus': session_metadata.get('focus', 'general conversation'),
                    'time_context': datetime.now().strftime('%A %I:%M %p'),
                    'immediate_concerns': 'none noted'
                }
            )
            
            # Build context blocks from retrieved data
            context_blocks = self.prompt_templates.build_context_blocks(
                elr_data=context.get("elr_memories"),
                glossary_data=context.get("glossary"),
                platform_kb=context.get("platform_kb"),
                query_type=user_message
            )
            
            # Combine system prompt with context blocks
            full_prompt = f"""{system_prompt}

{context_blocks}

User: {user_message}

Assistant:"""
            
            logger.info(f"Applied sophisticated avatar personality with {len(personality_response.core_traits)} traits and {personality_response.response_tone.value} tone")
            return full_prompt
            
        except Exception as e:
            logger.error(f"Avatar personality application error: {e}")
            # Fallback to basic personality prompt
            return await self._apply_basic_personality(context)
    
    async def _apply_safety_filters(self, prompt: str, user_message: str) -> str:
        """
        Apply safety and compliance filtering
        
        TODO: Implement comprehensive safety filtering:
        - PII detection and redaction
        - Content moderation
        - Compliance checking
        - Harmful content filtering
        """
        # For now, basic safety check
        if len(user_message) > 10000:
            logger.warning("User message too long, truncating")
            # Could implement truncation logic here
        
        # TODO: Add PII redaction
        # TODO: Add content moderation
        # TODO: Add compliance checks
        
        logger.info("Applied safety filters (basic implementation)")
        return prompt
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using LLM with strict output validation"""
        try:
            # Generate response using LLMManager's generate method
            model_response = await self.llm_manager.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            # Extract content from ModelResponse
            response_text = model_response.content if hasattr(model_response, 'content') else str(model_response)
            
            # Validate response doesn't contain self-referential conversation patterns
            if self._contains_conversation_loop(response_text):
                logger.warning("Detected conversation loop pattern in response, filtering...")
                response_text = self._filter_conversation_loops(response_text)
            
            logger.info(f"LLM response generated: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
    
    def _extract_relevant_glossary_terms(self, message: str, glossary: Dict[str, str]) -> Dict[str, str]:
        """Extract glossary terms relevant to the user's message."""
        relevant_terms = {}
        message_lower = message.lower()
        
        # Check each glossary term for relevance
        for term, definition in glossary.items():
            term_lower = term.lower()
            # Check if term appears in message or if it's a related concept
            if (term_lower in message_lower or 
                (term_lower == 'cap' and any(w in message_lower for w in ['care', 'action', 'points', 'earn'])) or
                (term_lower == 'reme' and any(w in message_lower for w in ['token', 'reward', 'convert'])) or
                (term_lower == 'elr' and any(w in message_lower for w in ['memory', 'record', 'life', 'personal'])) or
                (term_lower == 'remecare' and any(w in message_lower for w in ['activity', 'dementia', 'care']))):
                relevant_terms[term] = definition
        
        # Always include core terms if nothing specific found
        if not relevant_terms:
            for key in ['LUKi', 'ReMeLife', 'ELR']:
                if key in glossary:
                    relevant_terms[key] = glossary[key]
        
        return relevant_terms
    
    async def _post_process_response(self, response: str) -> str:
        """
        Post-process and validate LLM response
        
        Ensures:
        - No role confusion or inner monologue
        - Proper response structure
        - No fictional user messages
        """
        try:
            # Remove any accidental role reversals or inner monologue
            lines_to_remove = []
            lines = response.split('\n')
            
            for i, line in enumerate(lines):
                # Detect and remove internal dialogue patterns
                if any(pattern in line.lower() for pattern in [
                    "thinking:", "thought:", "internal:", "note to self:",
                    "as an ai", "i am thinking", "let me think"
                ]):
                    lines_to_remove.append(i)
                    
                # Detect and remove user impersonation
                if any(pattern in line for pattern in [
                    "User:", "Human:", "You:",
                    "Elizabeth:", "Patient:"
                ]) and not line.startswith("From your ELR"):
                    lines_to_remove.append(i)
            
            # Remove problematic lines
            filtered_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
            cleaned_response = '\n'.join(filtered_lines)
            
            # Ensure response follows the structure if not already formatted
            if "**Main Response**" not in cleaned_response:
                # Apply default structure
                cleaned_response = self._apply_response_structure(cleaned_response)
            
            logger.info(f"Post-processed response: removed {len(lines_to_remove)} problematic lines")
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Response post-processing error: {e}")
            return response
    
    def _apply_response_structure(self, raw_response: str) -> str:
        """Apply the default response structure to unformatted responses"""
        # Parse raw response to extract main points
        sentences = raw_response.split('. ')
        main_response = '. '.join(sentences[:3]) if len(sentences) >= 3 else raw_response
        
        # Create structured response
        structured = f"""**Main Response**
{main_response}

**Actionable Steps**
• Review your recent ELR data for patterns
• Consider the context provided above
• Let me know if you need specific guidance

**Context Anchors**
• Response based on current conversation context

**Conclusion**
How can I assist you further with your request?"""
        
        return structured
        """
        Post-process the LLM response
        
        TODO: Implement response post-processing:
        - Format cleanup
        - Safety filtering of output
        - Tool integration (recommendations, reports)
        - Response enhancement
        """
        # Basic post-processing for now
        processed_response = response.strip()
        
        # TODO: Add output safety filtering
        # TODO: Add tool integration
        # TODO: Add response enhancement
        
        logger.info("Applied response post-processing")
        return processed_response
    
    async def _update_session(
        self,
        session: SessionState,
        user_message: str,
        response: str,
        context: Dict[str, Any]
    ):
        """Update session with new conversation turn"""
        try:
            # Create conversation turn
            turn = ConversationTurn(
                user_id=session.user_id,
                session_id=session.session_id,
                message=user_message,
                response=response,
                timestamp=datetime.now(),
                context=context,
                metadata={"model_backend": settings.model_backend}
            )
            
            # Convert to session store format and add to history
            session_turn = turn.to_session_turn()
            session.conversation_history.append(session_turn)
            session.last_activity = datetime.now()
            
            # Keep only recent conversation history (configurable)
            max_history = settings.conversation_buffer_size
            if len(session.conversation_history) > max_history:
                session.conversation_history = session.conversation_history[-max_history:]
            
            # Save session directly (session store handles serialization)
            self.session_store.save_session(session)
            logger.info(f"Updated session {session.session_id} with new conversation turn")
            
        except Exception as e:
            logger.error(f"Session update error: {e}")
    
    async def _assemble_response(
        self,
        session: SessionState,
        response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble final response with metadata"""
        return {
            "response": response,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "model_used": settings.model_backend,
            "metadata": {
                "model_name": settings.model_name,
                "conversation_turn": len(session.conversation_history),
                "session_created": session.created_at.isoformat(),
                "processing_chain": "full_orchestration",
                "chain_id": self.chain_id,
                "elr_retrieved": bool(context.get("elr_memories")),
                "elr_gap": context.get("elr_gap", False),
                "glossary_terms_used": len(context.get("glossary", {})),
                "context_blocks_generated": bool(context.get("context_blocks"))
            }
        }
    
    async def _handle_error(
        self,
        user_id: str,
        session_id: str,
        message: str,
        error: str
    ) -> Dict[str, Any]:
        """Handle conversation processing errors"""
        logger.error(f"Conversation error for user {user_id}: {error}")
        
        return {
            "response": "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment.",
            "session_id": session_id,
            "user_id": user_id,
            "model_used": "error_fallback",
            "metadata": {
                "error": error,
                "processing_chain": "error_handler"
            }
        }
    
    def _load_glossary(self) -> Dict[str, str]:
        """Load glossary definitions for tokens and terms"""
        return {
            "CAP": "Care Action Points - Earned for performing care actions, unlimited supply, non-transferable",
            "REME": "Primary utility token for purchases, fixed supply, tradeable on exchanges",
            "LUKi": "AI participation rewards in federated learning system",
            "ELR": "Electronic Life Records - Proprietary data architecture capturing activities, preferences, habits, life stories",
            "ReMeLife": "Web3-based care ecosystem rewarding digital care actions through tokens",
            "ReMeCare": "B2B care provider app with activity-based digital care (formerly RemindMeCare)"
        }
    
    def _load_platform_kb(self, query: str) -> List[Dict[str, Any]]:
        """Load platform knowledge base items relevant to query"""
        # This would normally do semantic search; for now return static docs
        return [
            {
                "doc_title": "ReMeLife Whitepaper",
                "snippet": "ReMeLife uses a tri-token model: CAPs for care actions, REME for utility, and LUKi for AI participation",
                "doc_id": "whitepaper_v3"
            },
            {
                "doc_title": "Token Economics",
                "snippet": "The tri-token configuration ensures sustainable rewards for care providers while maintaining economic stability",
                "doc_id": "tokenomics_v2"
            }
        ]
    
    def _format_conversation_history(self, history: List[SessionConversationTurn]) -> List[Dict[str, Any]]:
        """Format conversation history for prompt inclusion"""
        if not history:
            return [{"text": "This is the start of your conversation with this user."}]
        
        formatted_history = []
        for i, turn in enumerate(history[-5:], 1):  # Last 5 turns
            # Handle both object and dict formats
            if isinstance(turn, dict):
                formatted_history.append({
                    "turn": i,
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "timestamp": turn.get("timestamp")
                })
            else:
                formatted_history.append({
                    "turn": i,
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat() if hasattr(turn, 'timestamp') else None
                })
        
        return formatted_history
    
    def _infer_user_mood(self, user_message: str) -> Optional[str]:
        """Infer user mood from their message"""
        message_lower = user_message.lower()
        
        # Positive mood indicators
        if any(word in message_lower for word in ['excited', 'happy', 'great', 'awesome', 'amazing', 'wonderful']):
            return 'positive'
        
        # Stressed mood indicators
        if any(word in message_lower for word in ['stressed', 'overwhelmed', 'anxious', 'worried', 'difficult']):
            return 'stressed'
        
        # Uncertain mood indicators
        if any(word in message_lower for word in ['confused', 'unsure', 'don\'t know', 'uncertain', 'maybe']):
            return 'uncertain'
        
        # Sad mood indicators
        if any(word in message_lower for word in ['sad', 'down', 'depressed', 'disappointed', 'frustrated']):
            return 'sad'
        
        return None  # Neutral/unknown
    
    def _analyze_conversation_tone(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Analyze the overall tone of the conversation"""
        if not conversation_history:
            return 'neutral'
        
        # Simple tone analysis based on recent messages
        recent_messages = conversation_history[-3:]  # Last 3 turns
        
        positive_indicators = 0
        negative_indicators = 0
        
        for turn in recent_messages:
            message = turn.get('message', '').lower()
            if any(word in message for word in ['good', 'great', 'thanks', 'helpful', 'appreciate']):
                positive_indicators += 1
            elif any(word in message for word in ['problem', 'issue', 'difficult', 'wrong', 'bad']):
                negative_indicators += 1
        
        if positive_indicators > negative_indicators:
            return 'positive'
        elif negative_indicators > positive_indicators:
            return 'concerned'
        else:
            return 'neutral'
    
    def _determine_situation_type(self, user_message: str) -> str:
        """Determine the type of situation/conversation"""
        message_lower = user_message.lower()
        
        # Goal setting
        if any(word in message_lower for word in ['goal', 'want to', 'plan to', 'hoping to', 'trying to']):
            return 'goal_setting'
        
        # Problem solving
        if any(word in message_lower for word in ['problem', 'issue', 'help with', 'stuck', 'challenge']):
            return 'problem_solving'
        
        # Celebration/achievement
        if any(word in message_lower for word in ['achieved', 'accomplished', 'succeeded', 'completed', 'finished']):
            return 'celebration'
        
        # Information seeking
        if any(word in message_lower for word in ['what is', 'how do', 'can you explain', 'tell me about']):
            return 'information_seeking'
        
        # Reflection/analysis
        if any(word in message_lower for word in ['think about', 'reflect', 'analyze', 'understand', 'why']):
            return 'reflection'
        
        return 'general_conversation'
    
    def _contains_conversation_loop(self, response: str) -> bool:
        """Detect if response contains self-referential conversation patterns"""
        loop_indicators = [
            "User:",
            "LUKi:",
            "Human:",
            "Assistant:",
            "You said:",
            "I replied:",
            "Then you",
            "Then I",
            "Our conversation",
            "In our chat",
            "Earlier you mentioned",
            "As we discussed"
        ]
        
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in loop_indicators)
    
    def _filter_conversation_loops(self, response: str) -> str:
        """Remove conversation loop patterns from response"""
        lines = response.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that look like conversation transcripts
            if any(pattern in line_lower for pattern in ['user:', 'luki:', 'human:', 'assistant:', 'you said:', 'i replied:']):
                continue
            # Skip empty lines after filtering
            if line.strip():
                filtered_lines.append(line)
        
        filtered_response = '\n'.join(filtered_lines)
        
        # If we filtered too much, provide a safe fallback
        if len(filtered_response.strip()) < 20:
            return "I understand what you're asking about. Let me help you with that in a clear and direct way."
        
        return filtered_response
    
    async def _apply_basic_personality(self, context: Dict[str, Any]) -> str:
        """Fallback basic personality prompt if sophisticated system fails"""
        user_message = context["current_message"]
        conversation_history = context.get("conversation_history", [])
        turn_count = context.get("session_metadata", {}).get("turn_count", 0)
        
        return f"""You are LUKi, an empathetic AI companion focused on personal growth and wellbeing.

Your Core Identity:
- You are supportive, caring, and genuinely interested in the user's wellbeing
- You provide personalized guidance based on understanding life patterns
- You focus on helping users make positive changes and achieve their goals
- You maintain a warm, encouraging, and optimistic communication style

Current Context:
- This is conversation turn #{turn_count + 1}
- User ID: {context.get('user_id', 'unknown')}
- Session: {context.get('session_id', 'unknown')}

{self._format_conversation_history(conversation_history)}

Current User Message: {user_message}

Please respond as LUKi with warmth, empathy, and helpful guidance."""

    async def close(self):
        """Clean up resources"""
        if hasattr(self.llm_manager, 'close'):
            await self.llm_manager.close()
        logger.info(f"Closed conversation chain {self.chain_id}")
