"""
Session Store for LUKi Agent

Manages ephemeral session memory including conversation history,
user state, and temporary context data.
"""

import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..config import settings


@dataclass
class ConversationTurn:
    """A single conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SessionState:
    """Session state data"""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[ConversationTurn]
    user_context: Dict[str, Any]
    agent_context: Dict[str, Any]


class SessionStore:
    """
    Manages session data using Redis for fast access
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.session_ttl = 24 * 60 * 60  # 24 hours
        self.key_prefix = "luki:session:"
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionState or None if not found
        """
        try:
            key = f"{self.key_prefix}{session_id}"
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            session_data = json.loads(data)
            
            # Convert conversation history back to objects
            conversation_history = []
            for turn_data in session_data.get("conversation_history", []):
                turn = ConversationTurn(
                    role=turn_data["role"],
                    content=turn_data["content"],
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    metadata=turn_data.get("metadata")
                )
                conversation_history.append(turn)
            
            return SessionState(
                user_id=session_data["user_id"],
                session_id=session_data["session_id"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                conversation_history=conversation_history,
                user_context=session_data.get("user_context", {}),
                agent_context=session_data.get("agent_context", {})
            )
            
        except Exception as e:
            print(f"Session retrieval error: {e}")
            return None
    
    def save_session(self, session: SessionState) -> bool:
        """
        Save session state
        
        Args:
            session: SessionState to save
            
        Returns:
            Success status
        """
        try:
            key = f"{self.key_prefix}{session.session_id}"
            
            # Convert to serializable format
            session_data = {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "conversation_history": [
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "timestamp": turn.timestamp.isoformat(),
                        "metadata": turn.metadata
                    }
                    for turn in session.conversation_history
                ],
                "user_context": session.user_context,
                "agent_context": session.agent_context
            }
            
            self.redis_client.setex(
                key,
                self.session_ttl,
                json.dumps(session_data)
            )
            
            return True
            
        except Exception as e:
            print(f"Session save error: {e}")
            return False
    
    def create_session(self, user_id: str, session_id: str) -> SessionState:
        """
        Create a new session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            New SessionState
        """
        now = datetime.utcnow()
        
        session = SessionState(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_activity=now,
            conversation_history=[],
            user_context={},
            agent_context={}
        )
        
        self.save_session(session)
        return session
    
    def add_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation turn to session
        
        Args:
            session_id: Session identifier
            role: Role (user/assistant)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        session.conversation_history.append(turn)
        session.last_activity = datetime.utcnow()
        
        # Keep only recent history (configurable limit)
        max_history = settings.conversation_buffer_size * 2  # User + assistant pairs
        if len(session.conversation_history) > max_history:
            session.conversation_history = session.conversation_history[-max_history:]
        
        return self.save_session(session)
    
    def update_user_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update user context in session
        
        Args:
            session_id: Session identifier
            context_updates: Context updates to apply
            
        Returns:
            Success status
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.user_context.update(context_updates)
        session.last_activity = datetime.utcnow()
        
        return self.save_session(session)
    
    def update_agent_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update agent context in session
        
        Args:
            session_id: Session identifier
            context_updates: Context updates to apply
            
        Returns:
            Success status
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.agent_context.update(context_updates)
        session.last_activity = datetime.utcnow()
        
        return self.save_session(session)
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in simple format
        
        Args:
            session_id: Session identifier
            limit: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        history = session.conversation_history
        if limit:
            history = history[-limit:]
        
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat()
            }
            for turn in history
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        try:
            key = f"{self.key_prefix}{session_id}"
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Session deletion error: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            cleaned = 0
            cutoff = datetime.utcnow() - timedelta(seconds=self.session_ttl)
            
            for key in keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        session_data = json.loads(data)
                        last_activity = datetime.fromisoformat(session_data["last_activity"])
                        
                        if last_activity < cutoff:
                            self.redis_client.delete(key)
                            cleaned += 1
                except:
                    # If we can't parse, delete it
                    self.redis_client.delete(key)
                    cleaned += 1
            
            return cleaned
            
        except Exception as e:
            print(f"Session cleanup error: {e}")
            return 0
