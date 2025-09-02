"""
Generic Personality Templates Stub
Replace with your own conversation templates and prompt engineering.
"""

from typing import Dict, Any, List, Optional

class PersonalityPromptTemplates:
    """
    Generic Prompt Template System - Stub Implementation
    Replace with your own prompt engineering and conversation templates.
    """
    
    def __init__(self):
        # Generic system prompt template
        self.base_system_template = """You are a helpful AI assistant. 

Your core traits:
{core_traits}

Communication style: {communication_style}

Guidelines:
{behavioral_guidelines}

Respond in a {tone} manner using {language} language."""

    def build_system_prompt(self, personality_config: Dict[str, Any]) -> str:
        """
        Build system prompt from personality configuration.
        
        This is a stub implementation - replace with your own prompt engineering.
        """
        # Extract configuration with defaults
        core_traits = personality_config.get('core_traits', [])
        comm_style = personality_config.get('communication_style', {})
        guidelines = personality_config.get('behavioral_guidelines', [])
        
        # Format traits
        traits_text = ", ".join([f"{trait[0]} ({trait[1]})" for trait in core_traits]) if core_traits else "helpful, professional"
        
        # Format communication style
        tone = comm_style.get('tone', 'professional')
        language = comm_style.get('language', 'clear and accessible')
        
        # Format guidelines
        guidelines_text = "\n".join([f"- {guideline}" for guideline in guidelines]) if guidelines else "- Be helpful and accurate"
        
        # Build prompt
        return self.base_system_template.format(
            core_traits=traits_text,
            communication_style=comm_style.get('name', 'professional'),
            behavioral_guidelines=guidelines_text,
            tone=tone,
            language=language
        )
    
    def get_conversation_template(self, template_type: str = "default") -> str:
        """
        Get conversation template by type.
        
        This is a stub implementation - replace with your own templates.
        """
        templates = {
            "default": "Respond helpfully to the user's question: {user_message}",
            "supportive": "Provide supportive guidance for: {user_message}",
            "informative": "Provide clear information about: {user_message}",
            "casual": "Have a friendly conversation about: {user_message}"
        }
        
        return templates.get(template_type, templates["default"])
    
    def format_prompt(self, template_name: str, context: Optional[Dict[str, str]] = None) -> str:
        """
        Format prompt with context.
        
        This is a stub implementation - customize for your prompt format.
        """
        template = self.get_conversation_template(template_name)
        if not context:
            return template
            
        try:
            return template.format(**context)
        except KeyError as e:
            # If template has placeholders not in context, return template as-is
            return template
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template types"""
        return ["default", "supportive", "informative", "casual"]
