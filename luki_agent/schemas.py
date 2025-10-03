"""
Data Schemas for LUKi Agent

This module defines the Pydantic models used for structured input and output
with the LLM, ensuring reliable, machine-readable data exchange.
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class LUKiThoughtProcess(BaseModel):
    """Schema for the AI's internal monologue and reasoning."""
    internal_analysis: str = Field(
        description="Brief reasoning (max 20 words)."
    )
    confidence_score: float = Field(
        description="Score 0.0-1.0.",
        ge=0.0,
        le=1.0
    )
    knowledge_source: Optional[str] = Field(
        default=None,
        description="Source ID."
    )

class LUKiResponse(BaseModel):
    """
    The primary structured response model for the LUKi agent.
    This ensures that the AI's output is always clean, predictable, and separated
    into internal thoughts and the final, user-facing message.
    """
    thought: LUKiThoughtProcess = Field(
        description="INTERNAL REASONING - This field is MANDATORY and contains your step-by-step thinking process. This is for system logging, debugging, and quality assurance. The user will NEVER see this content - it's completely internal to the system."
    )
    final_response: str = Field(
        description="USER-FACING RESPONSE - The final, polished response written in the LUKi persona. Be sharp, witty, and competent. Use subtle expressions like *chuckles*, *grins*, *nods* occasionally when they enhance the response - not in every message. Focus on being impressively helpful with natural personality, not forced cuteness. This is the ONLY part shown to the user."
    )
    web_search_used: bool = Field(
        default=False,
        description="METADATA - Set to true if web search results were used to answer this question. Used for UI indicators."
    )

class LUKiMinimalResponse(BaseModel):
    """
    Minimal fallback schema when structured reasoning times out or fails.
    """
    final_response: str = Field(
        description="User-facing response with natural, subtle personality (fallback mode)."
    )

class MemoryToWrite(BaseModel):
    """Schema for a new memory the AI decides to write to the ELR."""
    content: str = Field(description="The specific fact, preference, or detail to remember about the user.")
    importance: float = Field(
        description="The importance of this memory for future interactions, from 0.0 (trivial) to 1.0 (critical).",
        ge=0.0,
        le=1.0
    )

class LUKiAgentDecision(BaseModel):
    """
    A more advanced schema that allows the agent to decide on actions beyond just talking.
    This will be used to integrate tool use and memory writing directly into the generation step.
    """
    thought: LUKiThoughtProcess = Field(
        description="The internal thinking process of the AI, including its decision-making process."
    )
    final_response: str = Field(
        description="The final, user-facing response to be delivered to the user."
    )
    memories_to_write: Optional[List[MemoryToWrite]] = Field(
        default=None,
        description="A list of new memories to be written to the user's Electronic Life Record (ELR)."
    )
