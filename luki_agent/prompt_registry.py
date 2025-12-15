"""
Prompt Registry for LUKi Agent - Single Source of Truth for Prompts

Manages versioned prompt templates and provides clean loading interface.
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path
import hashlib

class PromptRegistry:
    """Registry for managing versioned prompt templates"""
    
    def __init__(self):
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self._cache: Dict[str, str] = {}
        self._hashes: Dict[str, str] = {}
        # Optional persona catalog for overlay prompts
        self.personas_catalog_path = self.prompts_dir / "personas_catalog.json"
        self._persona_catalog: Dict[str, dict] = {}
    
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

    def _load_personas_catalog(self) -> Dict[str, dict]:
        """Load persona catalog JSON if present.

        The catalog maps persona IDs (e.g. "default", "lukicool") to
        configuration objects that specify which prompt files should be
        concatenated for that persona.
        """
        if self._persona_catalog:
            return self._persona_catalog

        if self.personas_catalog_path.exists():
            try:
                with open(self.personas_catalog_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._persona_catalog = data
            except Exception:
                self._persona_catalog = {}
        else:
            self._persona_catalog = {}

        return self._persona_catalog

    def load_persona_stack(self, personality_mode: str) -> str:
        """Load persona overlay(s) for the given personality mode.

        This uses prompts/personas_catalog.json when available. Each persona
        entry provides a list of prompt_files (relative to the prompts dir).
        If anything fails, we fall back to the legacy persona_luki_v1.j2.
        """
        catalog = self._load_personas_catalog()
        persona_id = personality_mode or "default"
        cfg = catalog.get(persona_id) or catalog.get("default")

        # Legacy fallback: no catalog or invalid entry
        if not cfg or not isinstance(cfg, dict):
            return self.load_prompt("persona_luki", "v1")

        cache_key = f"persona_stack::{persona_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        files = cfg.get("prompt_files") or []
        texts = []
        for rel_path in files:
            try:
                path = self.prompts_dir / rel_path
                if path.exists():
                    texts.append(path.read_text(encoding="utf-8"))
            except Exception:
                # Ignore bad entries and continue
                continue

        if not texts:
            persona_text = self.load_prompt("persona_luki", "v1")
        else:
            persona_text = "\n\n".join(texts)

        self._cache[cache_key] = persona_text
        self._hashes[cache_key] = hashlib.md5(persona_text.encode()).hexdigest()[:8]
        return persona_text
    
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
        
        # Normalize personality mode for downstream logic
        persona_id = personality_mode or "default"

        # Load core system prompt
        system_core = self.load_prompt("system_core", "v1")
        
        # Load personality overlay stack based on requested mode
        try:
            persona = self.load_persona_stack(persona_id)
        except Exception:
            # Hard fallback to legacy default persona
            persona = self.load_prompt("persona_luki", "v1")
        
        # Add a short, explicit header so the model knows which persona
        # is active BEFORE reading the long core prompt. This helps the
        # persona instructions "stick" even for short replies.
        persona_summaries = {
            "default": "Base LUKi personality: sharp, witty, balanced between care and playful intelligence.",
            "lukicool": "LUKiCool: high-confidence, slightly chaotic, dry-humored guide who uses modern phrasing and occasional 😎-style energy.",
            "lukia": "LUKia: softer, feminine, emotionally attuned presence with grounded, gentle wording and occasional 🌱/✨ energy.",
            "lukiquant": "LUKiQuant: analytical, market-obsessed quant who speaks in trading terminology, loves crypto/stocks/math, and approaches everything with data-driven precision and 📊/📈 energy.",
        }
        summary = persona_summaries.get(
            persona_id,
            "Use the specific style, tone, actions/ticks, and emojis defined in the persona section for this mode."
        )
        persona_header = (
            f"## ACTIVE PERSONA MODE: {persona_id}\n"
            f"- {summary}\n"
            f"- All wording, tone, and persona actions MUST reflect this persona (not any other variant).\n"
        )

        # Add user-specific guidance
        user_guidance = self._get_user_guidance(user_id)
        
        # Load safety rules if requested
        safety_rules = ""
        if include_safety:
            safety_rules = self.load_prompt("safety_rules", "v1")
        
        # Combine components with clear separation. Persona header goes
        # first so the model sees which skin is active, then the core
        # system prompt, then the detailed persona overlay.
        components = [persona_header, system_core, persona, user_guidance]
        
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
