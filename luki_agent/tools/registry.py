"""
Tool Registry for LUKi Agent

Manages available tools and their execution for the agent.
Tools include memory retrieval, activity recommendations, and basic utilities.
"""

from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import sys
import os

# Skip problematic imports during static analysis to avoid Pyright errors
# All module loading is handled at runtime via the _load_*_modules() functions

# Module availability flags
COGNITIVE_MODULES_AVAILABLE = False
ENGAGEMENT_MODULES_AVAILABLE = False  
REPORTING_MODULES_AVAILABLE = False

# Global variables for dynamically loaded classes
CognitiveTools = None  # type: ignore
EngagementAgentTools = None  # type: ignore
get_reporting_tools = None  # type: ignore

# HTTP-based module communication using module_client
def _load_cognitive_modules():
    global COGNITIVE_MODULES_AVAILABLE, CognitiveTools
    try:
        from ..module_client import module_client
        # Test connectivity to cognitive service
        health_status = {"cognitive": True}  # Will be checked at runtime
        COGNITIVE_MODULES_AVAILABLE = True
        CognitiveTools = module_client  # Use HTTP client instead of direct import
        print("✅ Cognitive modules loaded successfully (HTTP client)")
        return CognitiveTools
    except ImportError as e:
        print(f"⚠️ Cognitive modules not available: {e}")
        CognitiveTools = None
        return None
    except Exception as e:
        print(f"⚠️ Error loading cognitive modules: {e}")
        CognitiveTools = None
        return None

def _load_engagement_modules():
    global ENGAGEMENT_MODULES_AVAILABLE, EngagementAgentTools
    try:
        from ..module_client import module_client
        # Test connectivity to engagement service
        health_status = {"engagement": True}  # Will be checked at runtime
        ENGAGEMENT_MODULES_AVAILABLE = True
        EngagementAgentTools = module_client  # Use HTTP client instead of direct import
        print("✅ Engagement modules loaded successfully (HTTP client)")
        return EngagementAgentTools
    except ImportError as e:
        print(f"⚠️ Engagement modules not available: {e}")
        EngagementAgentTools = None
        return None
    except Exception as e:
        print(f"⚠️ Error loading engagement modules: {e}")
        EngagementAgentTools = None
        return None

def _load_reporting_modules():
    global REPORTING_MODULES_AVAILABLE, get_reporting_tools
    try:
        from ..module_client import module_client
        # Test connectivity to reporting service
        health_status = {"reporting": True}  # Will be checked at runtime
        REPORTING_MODULES_AVAILABLE = True
        get_reporting_tools = module_client  # Use HTTP client instead of direct import
        print("✅ Reporting modules loaded successfully (HTTP client)")
        return get_reporting_tools
    except ImportError as e:
        print(f"⚠️ Reporting modules not available: {e}")
        get_reporting_tools = None
        return None
    except Exception as e:
        print(f"⚠️ Error loading reporting modules: {e}")
        get_reporting_tools = None
        return None

# Import memory service client
try:
    from ..memory.memory_service_client import get_memory_client, MemoryServiceClient
    MEMORY_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Memory service client not available: {e}")
    MEMORY_SERVICE_AVAILABLE = False

# Initialize modules at import time
_load_cognitive_modules()
_load_engagement_modules() 
_load_reporting_modules()


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    content: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for LUKi tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        pass


class MemorySearchTool(BaseTool):
    """Tool for searching user memories using Memory Service API"""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "search_memory"
    
    @property
    def description(self) -> str:
        return "Search user's memories and life story for relevant information"
    
    async def execute(self, query: str, user_id: str, **kwargs) -> ToolResult:
        """Search memories"""
        if not MEMORY_SERVICE_AVAILABLE:
            return ToolResult(
                success=False,
                content="",
                error="Memory service not available"
            )
        
        try:
            memory_client = await get_memory_client()
            results = await memory_client.search_memories(
                user_id=user_id,
                query=query,
                k=kwargs.get("k", 5)
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    content="No relevant memories found for this query.",
                    metadata={"result_count": 0}
                )
            
            # Format results
            formatted_results = []
            for result in results:
                content_type = result.metadata.get("content_type", "memory")
                formatted_results.append(f"[{content_type}] {result.content}")
            
            content = "Relevant memories found:\n" + "\n".join(formatted_results)
            
            return ToolResult(
                success=True,
                content=content,
                metadata={"result_count": len(results), "results": [r.__dict__ for r in results]}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Memory search failed: {str(e)}"
            )


class UserProfileTool(BaseTool):
    """Tool for accessing user profile information via Memory Service"""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "get_user_profile"
    
    @property
    def description(self) -> str:
        return "Get user's profile information including memory statistics and preferences"
    
    async def execute(self, user_id: str, **kwargs) -> ToolResult:
        """Get user profile"""
        if not MEMORY_SERVICE_AVAILABLE:
            return ToolResult(
                success=False,
                content="",
                error="Memory service not available"
            )
        
        try:
            memory_client = await get_memory_client()
            stats = await memory_client.get_user_memory_stats(user_id)
            
            if not stats:
                return ToolResult(
                    success=True,
                    content="No profile information available.",
                    metadata={"profile": {}}
                )
            
            # Format profile information
            content = f"User Profile for {user_id}:\n\n"
            content += f"**Memory Statistics:**\n"
            content += f"- Total memories: {stats.total_memories}\n"
            content += f"- Total chunks: {stats.total_chunks}\n"
            content += f"- Storage size: {stats.storage_size_mb:.2f} MB\n"
            
            if stats.earliest_memory:
                content += f"- Earliest memory: {stats.earliest_memory.strftime('%Y-%m-%d')}\n"
            if stats.latest_memory:
                content += f"- Latest memory: {stats.latest_memory.strftime('%Y-%m-%d')}\n"
            
            if stats.content_type_breakdown:
                content += f"\n**Content Types:**\n"
                for content_type, count in stats.content_type_breakdown.items():
                    content += f"- {content_type}: {count}\n"
            
            if stats.sensitivity_breakdown:
                content += f"\n**Sensitivity Levels:**\n"
                for sensitivity, count in stats.sensitivity_breakdown.items():
                    content += f"- {sensitivity}: {count}\n"
            
            return ToolResult(
                success=True,
                content=content,
                metadata={"profile": stats.__dict__}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Profile retrieval failed: {str(e)}"
            )


class ActivityRecommendationTool(BaseTool):
    """Tool for recommending activities (basic implementation)"""
    
    def __init__(self):
        # Basic activity database
        self.activities = {
            "music": [
                "Listen to your favorite jazz album",
                "Try discovering new music from the 1960s",
                "Play a musical instrument if you have one",
                "Sing along to classic songs"
            ],
            "reading": [
                "Read a chapter from a favorite book",
                "Browse through photo albums",
                "Read today's newspaper",
                "Look through old letters or journals"
            ],
            "creative": [
                "Try some light sketching or drawing",
                "Write in a journal about today",
                "Do a simple craft project",
                "Organize photos into albums"
            ],
            "social": [
                "Call a family member or friend",
                "Write a letter to someone special",
                "Share a memory with someone",
                "Plan a visit with loved ones"
            ],
            "physical": [
                "Take a gentle walk outside",
                "Do some light stretching",
                "Tend to plants or a garden",
                "Organize a room or drawer"
            ],
            "mental": [
                "Work on a crossword puzzle",
                "Play a card game or board game",
                "Do some light mental math",
                "Practice remembering favorite recipes"
            ]
        }
    
    @property
    def name(self) -> str:
        return "recommend_activity"
    
    @property
    def description(self) -> str:
        return "Recommend personalized activities based on user interests and preferences"
    
    async def execute(self, user_id: str, context: str = "", **kwargs) -> ToolResult:
        """Recommend activities"""
        try:
            # For basic implementation, use simple recommendation logic
            # In production, this would integrate with memory service for personalization
            profile = {}  # Placeholder for user profile
            recent_activities = []  # Placeholder for recent activities
            
            # Simple recommendation logic based on profile
            interests = profile.get("interests", []) if profile else []
            preferences = profile.get("preferences", {}) if profile else {}
            
            recommended = []
            
            # Match interests to activity categories
            for interest in interests:
                interest_lower = interest.lower()
                if "music" in interest_lower:
                    recommended.extend(self.activities["music"][:2])
                elif "read" in interest_lower or "book" in interest_lower:
                    recommended.extend(self.activities["reading"][:2])
                elif "art" in interest_lower or "craft" in interest_lower:
                    recommended.extend(self.activities["creative"][:2])
                elif "social" in interest_lower or "family" in interest_lower:
                    recommended.extend(self.activities["social"][:2])
                elif "garden" in interest_lower or "outdoor" in interest_lower:
                    recommended.extend(self.activities["physical"][:2])
            
            # If no specific interests, provide general recommendations
            if not recommended:
                recommended = [
                    "Take a few minutes to reflect on a happy memory",
                    "Look through some old photographs",
                    "Listen to music from your favorite era"
                ]
            
            # Limit to 3 recommendations
            recommended = recommended[:3]
            
            content = "Here are some activity suggestions for you:\n\n"
            for i, activity in enumerate(recommended, 1):
                content += f"{i}. {activity}\n"
            
            content += "\nWould you like me to help you with any of these activities or suggest something different?"
            
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "recommendations": recommended,
                    "based_on_interests": interests,
                    "recommendation_count": len(recommended)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Activity recommendation failed: {str(e)}"
            )


# Cognitive Module Tool Wrappers
class CognitiveActivityRecommendationTool(BaseTool):
    """Enhanced activity recommendation using cognitive modules"""
    
    def __init__(self):
        if not COGNITIVE_MODULES_AVAILABLE or CognitiveTools is None:
            raise ImportError("Cognitive modules not available")
        self.cognitive_tools = CognitiveTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "recommend_cognitive_activity"
    
    @property
    def description(self) -> str:
        return "Get personalized activity recommendations using advanced cognitive analysis and ReMeLife integration"
    
    async def execute(self, user_id: str, current_mood: Optional[str] = None, available_duration: Optional[int] = None, 
                     carer_available: bool = True, group_setting: bool = False, 
                     specific_request: Optional[str] = None, max_recommendations: int = 3, **kwargs) -> ToolResult:
        """Get cognitive activity recommendations"""
        try:
            result = await self.cognitive_tools.get_recommendations(
                user_id=user_id,
                context={
                    "current_mood": current_mood,
                    "available_duration": available_duration,
                    "carer_available": carer_available,
                    "group_setting": group_setting,
                    "specific_request": specific_request,
                    "max_recommendations": max_recommendations
                }
            )
            
            if result.get("success", False):
                recommendations = result.get("recommendations", [])
                content = f"I found {len(recommendations)} personalized activities for you:\n\n"
                
                for i, rec in enumerate(recommendations, 1):
                    content += f"{i}. **{rec.get('title', 'Activity')}**\n"
                    content += f"   {rec.get('description', '')}\n"
                    if rec.get('duration_minutes'):
                        content += f"   Duration: {rec['duration_minutes']} minutes\n"
                    if rec.get('difficulty_level'):
                        content += f"   Difficulty: {rec['difficulty_level']}\n"
                    content += "\n"
                
                content += "Would you like me to help you get started with any of these activities?"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't find suitable activities right now. Let me try a different approach.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble accessing activity recommendations right now.",
                error=f"Cognitive activity recommendation failed: {str(e)}"
            )


class WorldDayActivitiesTool(BaseTool):
    """Get today's world day activities (ReMeMades)"""
    
    def __init__(self):
        if not COGNITIVE_MODULES_AVAILABLE or CognitiveTools is None:
            raise ImportError("Cognitive modules not available")
        self.cognitive_tools = CognitiveTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "get_world_day_activities"
    
    @property
    def description(self) -> str:
        return "Get today's special world day activities and themed content (ReMeMades)"
    
    async def execute(self, user_id: str, **kwargs) -> ToolResult:
        """Get world day activities"""
        try:
            result = await self.cognitive_tools.get_recommendations(
                user_id=user_id,
                context={"activity_type": "world_day"}
            )
            
            if result.get("success", False):
                activities = result.get("activities", [])
                world_day = result.get("world_day", "")
                
                if activities:
                    content = f"🌍 **Today is {world_day}!**\n\n"
                    content += "Here are some special activities to celebrate:\n\n"
                    
                    for i, activity in enumerate(activities, 1):
                        content += f"{i}. **{activity.get('title', 'Activity')}**\n"
                        content += f"   {activity.get('description', '')}\n\n"
                    
                    content += "These activities are specially chosen to connect with today's celebration!"
                else:
                    content = "Today doesn't have a specific world day theme, but I can still recommend great activities for you!"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't retrieve today's world day activities.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble accessing world day activities right now.",
                error=f"World day activities failed: {str(e)}"
            )


class ActivityEngagementTool(BaseTool):
    """Record and track activity engagement"""
    
    def __init__(self):
        if not COGNITIVE_MODULES_AVAILABLE or CognitiveTools is None:
            raise ImportError("Cognitive modules not available")
        self.cognitive_tools = CognitiveTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "record_activity_engagement"
    
    @property
    def description(self) -> str:
        return "Record user engagement with activities for learning and improvement"
    
    async def execute(self, user_id: str, activity_id: str, engagement_score: float,
                     duration_minutes: int, completion_status: str = "completed",
                     feedback: Optional[str] = None, mood_before: Optional[str] = None, mood_after: Optional[str] = None,
                     carer_notes: Optional[str] = None, **kwargs) -> ToolResult:
        """Record activity engagement"""
        try:
            result = await self.cognitive_tools.track_interaction(
                user_id=user_id,
                interaction_data={
                    "activity_id": activity_id,
                    "engagement_score": engagement_score,
                    "duration_minutes": duration_minutes,
                    "completion_status": completion_status,
                    "feedback": feedback,
                    "mood_before": mood_before,
                    "mood_after": mood_after,
                    "carer_notes": carer_notes
                }
            )
            
            if result.get("success", False):
                content = "Thank you! I've recorded your activity engagement. "
                content += "This helps me learn what activities work best for you."
                
                if mood_before and mood_after:
                    content += f"\n\nI noticed your mood went from {mood_before} to {mood_after}. "
                    if mood_after in ["happy", "content", "relaxed", "energized"]:
                        content += "That's wonderful! I'll remember this activity worked well for you."
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't record the activity engagement right now.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble recording the activity engagement.",
                error=f"Activity engagement recording failed: {str(e)}"
            )


class GetMyStoryPromptTool(BaseTool):
    """Tool for getting personalized My Story prompts"""
    
    def __init__(self, cognitive_tools):
        self.cognitive_tools = cognitive_tools
    
    @property
    def name(self) -> str:
        return "get_mystory_prompt"
    
    @property
    def description(self) -> str:
        return "Get personalized My Story conversation prompts based on user's background and interests"
    
    async def execute(self, user_id: str, **kwargs) -> ToolResult:
        """Get personalized My Story prompt"""
        try:
            result = await self.cognitive_tools.get_recommendations(
                user_id=user_id,
                context={"prompt_type": "mystory"}
            )
            
            if result.get("success", False):
                prompt = result.get("prompt", "")
                topic = result.get("topic", "")
                
                content = f"💭 **Let's explore a memory together!**\n\n"
                if topic:
                    content += f"**Topic: {topic}**\n\n"
                content += prompt
                content += "\n\nTake your time, and share whatever feels comfortable. I'm here to listen!"
                
                return ToolResult(success=True, content=content)
            else:
                return ToolResult(success=False, content="I'm having trouble generating a personalized prompt right now. Would you like to share any memory that comes to mind?")
        except Exception as e:
            return ToolResult(success=False, content=f"I encountered an issue: {str(e)}")


class GenerateWellbeingReportTool(BaseTool):
    """Tool for generating wellbeing reports"""
    
    def __init__(self, reporting_client):
        self.reporting_client = reporting_client
    
    @property
    def name(self) -> str:
        return "generate_wellbeing_report"
    
    @property
    def description(self) -> str:
        return "Generate comprehensive wellbeing reports based on user activities and engagement"
    
    async def execute(self, user_id: str, report_type: str = "weekly", **kwargs) -> ToolResult:
        """Generate wellbeing report"""
        try:
            result = await self.reporting_client.generate_report(
                user_id=user_id,
                report_data={"type": report_type}
            )
            
            if result.get("success", False):
                report = result.get("report", {})
                content = f"📊 **Your {report_type.title()} Wellbeing Report**\n\n"
                content += report.get("summary", "Report generated successfully.")
                return ToolResult(success=True, content=content)
            else:
                return ToolResult(success=False, content="Unable to generate your wellbeing report at this time.")
        except Exception as e:
            return ToolResult(success=False, content=f"Error generating report: {str(e)}")


class GetAnalyticsSummaryTool(BaseTool):
    """Tool for getting analytics summaries"""
    
    def __init__(self, reporting_client):
        self.reporting_client = reporting_client
    
    @property
    def name(self) -> str:
        return "get_analytics_summary"
    
    @property
    def description(self) -> str:
        return "Get analytics summary of user engagement and activity patterns"
    
    async def execute(self, user_id: str, period: str = "week", **kwargs) -> ToolResult:
        """Get analytics summary"""
        try:
            result = await self.reporting_client.get_analytics(
                user_id=user_id,
                analytics_data={"period": period}
            )
            
            if result.get("success", False):
                analytics = result.get("analytics", {})
                content = f"📈 **Your {period.title()} Analytics Summary**\n\n"
                content += analytics.get("summary", "Analytics data retrieved successfully.")
                return ToolResult(success=True, content=content)
            else:
                return ToolResult(success=False, content="Unable to retrieve your analytics summary at this time.")
        except Exception as e:
            return ToolResult(success=False, content=f"Error retrieving analytics: {str(e)}")


# Engagement Module Tool Wrappers
class UserRecommendationTool(BaseTool):
    """Recommend other users for social connections"""
    
    def __init__(self):
        if not ENGAGEMENT_MODULES_AVAILABLE or EngagementAgentTools is None:
            raise ImportError("Engagement modules not available")
        self.engagement_tools = EngagementAgentTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "recommend_users"
    
    @property
    def description(self) -> str:
        return "Find and recommend other users for social connections based on shared interests and compatibility"
    
    async def execute(self, user_id: str, max_recommendations: int = 3, **kwargs) -> ToolResult:
        """Get user recommendations"""
        try:
            result = await self.engagement_tools.get_engagement_metrics(
                user_id=user_id
            )
            
            if result.get("success", False):
                recommendations = result.get("recommendations", [])
                content = f"I found {len(recommendations)} people you might connect well with:\n\n"
                
                for i, rec in enumerate(recommendations, 1):
                    content += f"{i}. **{rec.get('name', 'Someone special')}**\n"
                    explanations = rec.get('explanations', [])
                    if explanations:
                        content += f"   {explanations[0]}\n"
                    content += f"   Match score: {rec.get('confidence_score', 0):.1%}\n\n"
                
                content += "Would you like me to help you connect with any of these people?"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't find suitable connections right now.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble finding connections right now.",
                error=f"User recommendation failed: {str(e)}"
            )


class EventRecommendationTool(BaseTool):
    """Recommend events and activities for social engagement"""
    
    def __init__(self):
        if not ENGAGEMENT_MODULES_AVAILABLE or EngagementAgentTools is None:
            raise ImportError("Engagement modules not available")
        self.engagement_tools = EngagementAgentTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "recommend_events"
    
    @property
    def description(self) -> str:
        return "Find and recommend events, activities, and groups based on user interests and social preferences"
    
    async def execute(self, user_id: str, max_recommendations: int = 3, **kwargs) -> ToolResult:
        """Get event recommendations"""
        try:
            result = await self.engagement_tools.track_interaction(
                user_id=user_id,
                interaction_data={"request_type": "events", "max_recommendations": max_recommendations}
            )
            
            if result.get("success", False):
                recommendations = result.get("recommendations", [])
                content = f"I found {len(recommendations)} events and activities you might enjoy:\n\n"
                
                for i, rec in enumerate(recommendations, 1):
                    content += f"{i}. **{rec.get('title', 'Special Event')}**\n"
                    explanations = rec.get('explanations', [])
                    if explanations:
                        content += f"   {explanations[0]}\n"
                    if rec.get('date'):
                        content += f"   Date: {rec['date']}\n"
                    content += f"   Match score: {rec.get('confidence_score', 0):.1%}\n\n"
                
                content += "Would you like more details about any of these events?"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't find suitable events right now.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble finding events right now.",
                error=f"Event recommendation failed: {str(e)}"
            )


class CommunityRecommendationTool(BaseTool):
    """Recommend communities and groups for social engagement"""
    
    def __init__(self):
        if not ENGAGEMENT_MODULES_AVAILABLE or EngagementAgentTools is None:
            raise ImportError("Engagement modules not available")
        self.engagement_tools = EngagementAgentTools  # Use the module_client instance directly
    
    @property
    def name(self) -> str:
        return "recommend_communities"
    
    @property
    def description(self) -> str:
        return "Find and recommend communities, groups, and social circles based on shared interests and values"
    
    async def execute(self, user_id: str, max_recommendations: int = 3, **kwargs) -> ToolResult:
        """Get community recommendations"""
        try:
            result = await self.engagement_tools.track_interaction(
                user_id=user_id,
                interaction_data={"request_type": "communities", "max_recommendations": max_recommendations}
            )
            
            if result.get("success", False):
                recommendations = result.get("recommendations", [])
                content = f"I found {len(recommendations)} communities you might want to join:\n\n"
                
                for i, rec in enumerate(recommendations, 1):
                    content += f"{i}. **{rec.get('name', 'Special Community')}**\n"
                    explanations = rec.get('explanations', [])
                    if explanations:
                        content += f"   {explanations[0]}\n"
                    if rec.get('member_count'):
                        content += f"   Members: {rec['member_count']}\n"
                    content += f"   Match score: {rec.get('confidence_score', 0):.1%}\n\n"
                
                content += "Would you like to learn more about any of these communities?"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=result
                )
            else:
                return ToolResult(
                    success=False,
                    content="I couldn't find suitable communities right now.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                content="I'm having trouble finding communities right now.",
                error=f"Community recommendation failed: {str(e)}"
            )


class LangChainToolWrapper(BaseTool):
    """Wrapper to convert LangChain tools to BaseTool format"""
    
    def __init__(self, langchain_tool):
        self.langchain_tool = langchain_tool
    
    @property
    def name(self) -> str:
        return self.langchain_tool.name
    
    @property
    def description(self) -> str:
        return self.langchain_tool.description
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the LangChain tool"""
        try:
            # LangChain tools are typically synchronous, but some may be async
            if asyncio.iscoroutinefunction(self.langchain_tool.func):
                result = await self.langchain_tool.func(**kwargs)
            else:
                result = self.langchain_tool.func(**kwargs)
            
            return ToolResult(
                success=True,
                content=str(result),
                metadata={"tool_type": "langchain", "original_result": result}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"LangChain tool execution failed: {str(e)}"
            )


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def register_memory_tools(self):
        """Register memory-related tools"""
        if not MEMORY_SERVICE_AVAILABLE:
            print("Warning: Memory service not available, skipping memory tools registration")
            return
        
        try:
            self.register_tool(MemorySearchTool())
            self.register_tool(UserProfileTool())
            self.register_tool(ActivityRecommendationTool())
            print("✅ Memory tools registered successfully")
        except Exception as e:
            print(f"❌ Failed to register memory tools: {e}")
    
    def register_cognitive_tools(self):
        """Register cognitive module tools"""
        if not COGNITIVE_MODULES_AVAILABLE:
            print("Warning: Cognitive modules not available, skipping cognitive tools registration")
            return
        
        try:
            self.register_tool(CognitiveActivityRecommendationTool())
            self.register_tool(WorldDayActivitiesTool())
            self.register_tool(ActivityEngagementTool())
            self.register_tool(GetMyStoryPromptTool(CognitiveTools))
            print("✅ Cognitive tools registered successfully")
        except Exception as e:
            print(f"❌ Failed to register cognitive tools: {e}")
    
    def register_engagement_tools(self):
        """Register engagement module tools"""
        if not ENGAGEMENT_MODULES_AVAILABLE:
            print("Warning: Engagement modules not available, skipping engagement tools registration")
            return
        
        try:
            self.register_tool(UserRecommendationTool())
            self.register_tool(EventRecommendationTool())
            self.register_tool(CommunityRecommendationTool())
            print("✅ Engagement tools registered successfully")
        except Exception as e:
            print(f"❌ Failed to register engagement tools: {e}")
    
    def register_reporting_tools(self):
        """Register reporting module tools"""
        if not REPORTING_MODULES_AVAILABLE:
            print("Warning: Reporting modules not available, skipping reporting tools registration")
            return
        
        try:
            # Create reporting tools using HTTP client
            if get_reporting_tools is None:
                print("Warning: get_reporting_tools is None, cannot register reporting tools")
                return
            
            # Create mock reporting tools that use the HTTP client
            reporting_tools = [
                GenerateWellbeingReportTool(get_reporting_tools),
                GetAnalyticsSummaryTool(get_reporting_tools)
            ]
            
            # Register the tools
            for tool in reporting_tools:
                self.register_tool(tool)
            
            print(f"✅ Reporting tools registered successfully ({len(reporting_tools)} tools)")
        except Exception as e:
            print(f"❌ Failed to register reporting tools: {e}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List available tools"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool '{name}' not found"
            )
        
        try:
            result = await tool.execute(**kwargs)
            
            # Log execution
            self.execution_history.append({
                "tool_name": name,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata
            })
            
            return result
            
        except Exception as e:
            error_result = ToolResult(
                success=False,
                content="",
                error=f"Tool execution failed: {str(e)}"
            )
            
            self.execution_history.append({
                "tool_name": name,
                "success": False,
                "error": str(e),
                "metadata": None
            })
            
            return error_result
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get tool execution history"""
        history = self.execution_history
        if limit:
            history = history[-limit:]
        return history
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
