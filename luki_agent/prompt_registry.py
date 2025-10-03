"""
Prompt Registry for LUKi Agent - Single Source of Truth for Prompts

Manages versioned prompt templates and provides clean loading interface.
"""

import os
from typing import Dict, Optional
from pathlib import Path
import hashlib

class PromptRegistry:
    """Registry for managing versioned prompt templates"""
    
    def __init__(self):
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self._cache: Dict[str, str] = {}
        self._hashes: Dict[str, str] = {}
    
    def load_prompt(self, template_name: str, version: str = "v1") -> str:
        """Load a prompt template by name and version"""
        cache_key = f"{template_name}_{version}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        template_file = self.prompts_dir / f"{template_name}_{version}.j2"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_file}")
        
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cache the content and its hash
        self._cache[cache_key] = content
        self._hashes[cache_key] = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return content
    
    def get_prompt_hash(self, template_name: str, version: str = "v1") -> str:
        """Get hash of prompt template for versioning/telemetry"""
        cache_key = f"{template_name}_{version}"
        
        if cache_key not in self._hashes:
            # Load to generate hash
            self.load_prompt(template_name, version)
        
        return self._hashes[cache_key]
    
    def build_system_prompt(
        self,
        user_id: str,
        personality_mode: str = "default",
        include_safety: bool = True
    ) -> str:
        """Build complete system prompt from components"""
        
        # Load core system prompt
        system_core = self.load_prompt("system_core", "v1")
        
        # Load personality
        persona = self.load_prompt("persona_luki", "v1")
        
        # Add user-specific guidance
        user_guidance = self._get_user_guidance(user_id)
        
        # Load safety rules if requested
        safety_rules = ""
        if include_safety:
            safety_rules = self.load_prompt("safety_rules", "v1")
        
        # Combine components with clear separation
        components = [system_core, persona, user_guidance]
        
        if safety_rules:
            components.append(safety_rules)
        
        return "\n\n".join(components)
    
    def _get_user_guidance(self, user_id: str) -> str:
        """Generate user-specific guidance based on authentication status"""
        # Treat any missing or anonymous-prefixed ID as anonymous
        if (not user_id or 
            user_id == 'anonymous_base_user' or 
            user_id.lower() in ('anonymous', 'guest') or 
            user_id.startswith('anonymous_')):
            
            return """## User Status: Anonymous
- You're currently using LUKi in guest mode with access to general knowledge
- For personalized experiences with Electronic Life Records (ELR), consider creating an account
- ELR enables personalized memory recall, care recommendations, and family context
- Gently encourage account creation when relevant, but don't be pushy"""
        else:
            return """## User Status: Authenticated
- You have access to personalized Electronic Life Records (ELR) and memory context
- Your personal memories, preferences, and family context inform responses
- ELR data is private and secure, used only to enhance your experience
- Feel free to share personal information for better, more relevant assistance"""

# Global registry instance
prompt_registry = PromptRegistry()
