"""LLM Backend Implementations for LUKi Agent

This module provides concrete implementations of LLM backends.
Deploy: 2025-10-05T18:51:00"""
import json
import re
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .config import settings, get_model_config
from .schemas import LUKiResponse, LUKiMinimalResponse, MemoryDetectionResponse
from .prompt_registry import prompt_registry
from .tools.web_search import WebSearchTool

@dataclass
class ModelResponse:
    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, prompt: Dict[str, Any], **kwargs) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, prompt: Dict[str, Any], **kwargs) -> AsyncGenerator[str, None]:
        yield ""
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        pass
class TogetherAIBackend(LLMBackend):
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
  
        try:
            import openai  # type: ignore[reportMissingImports]
            import instructor  # type: ignore[reportMissingImports]
            self._openai_module = openai
            self._instructor_module = instructor
        except ImportError as e:
            print(f"❌ CRITICAL: Missing dependencies for TogetherAIBackend: {e}")
            print("Please ensure 'openai' and 'instructor' are installed in requirements.txt")
            raise ImportError(f"Missing dependencies for TogetherAIBackend: {e}. Please run 'pip install openai instructor'.")
  
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("❌ CRITICAL: Together AI API key not found in config!")
        
        print("🔍 Initializing Together AI client (API key masked)")
        
        # Base OpenAI-compatible client (Together AI)
        openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            timeout=120.0,  # provider-level timeout (we still apply our own soft timeout)
            max_retries=1   # keep provider retries low; we'll handle fallback and soft timeout
        )

        # Save raw client for fallbacks
        self._openai_client = openai_client

        # Configure for structured output with instructor
        self.client = instructor.patch(
            openai_client,
            mode=instructor.Mode.JSON
        )

        # Tunables
        self.structured_timeout = int(os.getenv("LUKI_STRUCTURED_TIMEOUT", "20"))
        self.fast_fallback_tokens = int(os.getenv("LUKI_FALLBACK_MAX_TOKENS", "256"))
        self.fallback_model = os.getenv("LUKI_FALLBACK_MODEL")  # optional override model for fallback

        # Sampling preferences for all chat calls (can be overridden via env)
        self.top_p = float(os.getenv("LUKI_TOP_P", "0.9"))
        self.presence_penalty = float(os.getenv("LUKI_PRESENCE_PENALTY", "0.4"))
        self.frequency_penalty = float(os.getenv("LUKI_FREQUENCY_PENALTY", "0.2"))
        # Optional: allow re-enabling persona ticks/actions via env. By default in
        # this develop build we run in tickless mode so responses contain no
        # *stage directions* unless LUKI_DISABLE_TICKS is explicitly set false.
        self.disable_ticks = os.getenv("LUKI_DISABLE_TICKS", "true").lower() == "true"
        
        # Initialize web search tool (gracefully handle missing API key)
        try:
            self.web_search_tool = WebSearchTool()
            print("✅ Web search tool enabled")
        except (ValueError, ImportError) as e:
            print(f"⚠️  Web search tool disabled: {e}")
            self.web_search_tool = None
        
        print(f"✅ Together AI backend initialized: {self.model_name} with max_tokens={self.max_tokens}")

    async def _check_and_execute_web_search(
        self,
        messages: list[Dict[str, str]],
        user_input: str,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Check if user's question needs web search and execute if needed.
        Includes follow-up detection for multi-turn queries.
        
        Returns:
            Search results as formatted string if search was needed, None otherwise.
        """
        if not self.web_search_tool:
            return None
        
        # CRITICAL: Check if user is answering AI's previous location question
        # Example: AI: "Where would you like to check weather?" → User: "london"
        follow_up_search = self._detect_location_followup(messages, user_input)
        if follow_up_search:
            # User is answering a location question - trigger search with their answer
            needs_search = True
            print(f"🔍 Follow-up detected: User answering location question with '{user_input}'")
        else:
            # Quick heuristic check: does the question likely need current info?
            needs_search = await self._needs_web_search(user_input)
            if not needs_search:
                return None
        
        # Prepare optional trace entry for structured metadata
        trace_entry: Optional[Dict[str, Any]] = None
        if trace is not None:
            trace_entry = {
                "used": False,
                "query": None,
                "answer": "",
                "sources": [],
                "search_depth": None,
                "error": None,
            }
        try:
            print("🔍 Web search triggered - analyzing query...")
            
            # Generate search query using simple keyword extraction
            # Fallback to original question if extraction fails
            search_query = self._generate_search_query(user_input)
            print(f"🔍 Search query generated: {search_query}")
            if trace_entry is not None:
                trace_entry["query"] = search_query
            
            # Execute search
            search_results = self.web_search_tool.search(search_query, max_results=3)
            
            if search_results.get("success"):
                formatted_results = self.web_search_tool.format_results_for_llm(search_results)
                print(f"✅ Search completed: {len(search_results.get('results', []))} results")
                if trace_entry is not None:
                    trace_entry["used"] = True
                    trace_entry["answer"] = search_results.get("answer", "")
                    trace_entry["search_depth"] = search_results.get("search_depth")
                    # Capture up to 3 top sources with title + URL
                    top_results = search_results.get("results", [])[:3]
                    trace_entry["sources"] = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                        }
                        for r in top_results
                    ]
                if trace is not None and trace_entry is not None:
                    trace["web_search"] = trace_entry
                return formatted_results
            else:
                print(f"⚠️  Search failed: {search_results.get('error')}")
                if trace_entry is not None:
                    trace_entry["error"] = search_results.get("error") or "unknown_error"
                if trace is not None and trace_entry is not None:
                    trace["web_search"] = trace_entry
                return None
                
        except Exception as e:
            print(f"❌ Web search error: {e}")
            if trace_entry is not None:
                trace_entry["error"] = str(e)
            if trace is not None and trace_entry is not None:
                trace["web_search"] = trace_entry
            return None
    
    def _detect_location_followup(self, messages: list[Dict[str, str]], user_input: str) -> bool:
        """
        Detect if user is answering AI's previous question about location.
        
        Returns:
            True if this appears to be a location answer to AI's question, False otherwise.
        """
        # Check if user input is short (likely a location name)
        words = user_input.strip().split()
        if len(words) > 3:  # Too long to be a simple location answer
            return False
        
        # Get last AI message from conversation
        last_ai_message = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_ai_message = msg.get("content", "").lower()
                break
        
        if not last_ai_message:
            return False
        
        # Check if AI asked about location in previous message
        location_patterns = [
            "where", "which city", "which location", "what city",
            "where are you", "where do you want", "which place"
        ]
        
        # Also check for weather context
        weather_context = any(word in last_ai_message for word in ["weather", "temperature", "forecast", "climate"])
        
        asked_location = any(pattern in last_ai_message for pattern in location_patterns)
        
        return asked_location and weather_context
    
    def _generate_search_query(self, user_input: str) -> str:
        """
        Generate optimized search query from user input using simple heuristics.
        Falls back to original question if extraction fails.
        """
        import re
        from datetime import datetime
        
        user_lower = user_input.lower()
        current_year = datetime.now().year
        
        # CRITICAL: If input is very short (1-3 words), assume it's a location for weather
        words = user_input.strip().split()
        if len(words) <= 3 and not any(word in user_lower for word in ["is", "are", "was", "were", "the", "what", "who", "when"]):
            # Likely a simple location answer - add "weather" context
            return f"{user_input.strip()} weather"
        
        # Pattern: "who is the [role] of [location]" → "[role] of [location] {year}"
        role_match = re.search(r'who (?:is|\'s) (?:the )?(?:current )?(prime minister|pm|president|ceo|leader|king|queen)(?: of)? (?:the )?(\w+)', user_lower, re.IGNORECASE)
        if role_match:
            role = "Prime Minister" if role_match.group(1).lower() in ["pm", "prime minister"] else role_match.group(1).title()
            location = role_match.group(2).upper() if len(role_match.group(2)) <= 3 else role_match.group(2).title()
            return f"{location} {role} {current_year}"
        
        # Pattern: "what is the latest/newest [thing]" → "latest [thing] {year}"
        latest_match = re.search(r'(?:what is |what\'s )?(?:the )?(latest|newest|current|new) (.+?)[\?\.!]?$', user_lower, re.IGNORECASE)
        if latest_match:
            thing = latest_match.group(2).strip()
            return f"latest {thing} {current_year}"
        
        # If user explicitly says "search" - extract the search terms
        search_match = re.search(r'search(?: for| online| the (?:internet|web))?(?: for)?\s+(.+?)[\?\.!]?$', user_lower, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            if query:
                return f"{query} {current_year}"
        
        # Default: use the original question with current year appended
        clean_query = user_input.strip().rstrip('?.!')
        return f"{clean_query} {current_year}"
    
    def _clean_response(self, text: str) -> str:
        """Remove HTML/metadata noise and normalize markdown structure for rendering."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove numbered citations like (1), [1], etc.
        text = re.sub(r'\([0-9]+\)', '', text)
        text = re.sub(r'\[[0-9]+\]', '', text)
        
        # CRITICAL: Remove ALL metadata leakage patterns
        # Remove web_search_used in any form (tags, plain text, key=value format)
        text = re.sub(r'<web_search_used>.*?</web_search_used>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\bweb_search_used\s*=\s*(true|false)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bweb_search_used:\s*(true|false)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\"web_search_used\"\s*:\s*(true|false)', '', text, flags=re.IGNORECASE)
        
        # Remove other potential metadata leakage patterns
        text = re.sub(r'\bconfidence_score\s*=\s*[\d.]+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bknowledge_source\s*=\s*\w+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\binternal_analysis\s*[:=].*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bthought\s*[:=].*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmetadata\s*[:=]', '', text, flags=re.IGNORECASE)
        
        # CRITICAL: Convert escaped markdown back to proper formatting
        # The model sometimes generates literal \n instead of actual newlines
        text = text.replace('\\n', '\n')
        # Convert markdown list markers from escaped to actual
        text = re.sub(r'\\([*\-])', r'\1', text)

        # Normalize common markdown separators that often appear inline when
        # generated via streaming or from documentation chunks. This helps
        # front-ends like ReactMarkdown render consistent structure instead of
        # showing raw `*` or `---` inside long paragraphs.

        # Ensure horizontal rules like "---" sit on their own line.
        text = re.sub(r"\s+(-{3,})\s+", r"\n\n\\1\n\n", text)

        # Ensure headings starting with # appear at the beginning of a line.
        text = re.sub(r"\s+(#{1,6}\s+)", r"\n\n\\1", text)

        # Ensure bullet markers "* " or "- " start on their own line when
        # they are acting as list items rather than inline characters.
        text = re.sub(r"([^\n])\s+([*\-]\s+)", r"\\1\n\\2", text)
        
        # TARGETED FIX: Remove stray asterisks at the very end of response
        # Only fix if asterisks appear after punctuation (likely unintentional)
        text = re.sub(r'([.!?])\*{1,2}$', r'\1', text.rstrip())
        
        # DEFENSIVE FIX: Close unclosed italic markdown for action expressions
        # Pattern: *word (where word looks like an action: grins, chuckles, nods, etc.)
        # followed by text without closing * within reasonable distance
        # This prevents "*grins and here's more text..." from making everything italic
        action_words = r'(?:grins|chuckles|nods|smiles|laughs|winks|shrugs|thinks|pauses|sighs)'
        # Find "*action " without a closing * within the next 10 characters
        text = re.sub(
            rf'\*({action_words})(\s+)(?!\*)',
            r'*\1*\2',
            text,
            flags=re.IGNORECASE
        )
        
        # Clean up extra spaces on same line (but preserve newlines for formatting)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove excessive blank lines (max 2 newlines in a row)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # DEFENSIVE FIX: Collapse duplicated trailing stage ticks like
        # "*nods slowly* slowly*" → "*nods slowly*".
        # This can happen when we auto-close '*nods' and later rewrite it to a
        # persona-specific action, leaving a stray "word*" segment at the end.
        def _dedupe_trailing_tick(match: re.Match) -> str:
            full = match.group(1)  # e.g. "*nods slowly*"
            trailing = match.group(2)  # e.g. "slowly"
            inner = full.strip('*').strip()
            parts = inner.split()
            last_word = parts[-1] if parts else ""
            if last_word and trailing.lower() == last_word.lower():
                # Drop the duplicated trailing "word*" segment
                return full
            return match.group(0)

        text = re.sub(r'(\*[^\*]+\*)\s+([A-Za-z]+)\*', _dedupe_trailing_tick, text)
        return text.strip()

    def _enforce_persona_actions(self, text: str, persona_mode: Optional[str]) -> str:
        """Rewrite generic base ticks into persona-specific actions.

        This is a lightweight, defensive post-processing step to guarantee that
        LUKiCool and LUKia do not accidentally reuse base LUKi ticks like
        *smiles*, *chuckles*, or *nods* even if the model reaches for them
        from general training data.
        """
        if not text or not persona_mode:
            return text

        mode = persona_mode.lower()
        if mode not in ("lukicool", "lukia"):
            return text

        # Only match single-word stage directions like *smiles*, not
        # multi-word actions such as *softly smiles* which belong to LUKia.
        pattern = re.compile(r"\*(smiles?|chuckles?|nods?)\*", re.IGNORECASE)

        def repl_lukicool(match: re.Match) -> str:
            word = match.group(1).lower()
            # Map base LUKi ticks to a small palette so we don't overuse *smirks*.
            if word.startswith("smile"):
                return "*leans back*"
            if word.startswith("chuckle"):
                return "*snorts softly*"
            if word.startswith("nod"):
                return "*raises an eyebrow*"
            return match.group(0)

        def repl_lukia(match: re.Match) -> str:
            word = match.group(1).lower()
            if word.startswith("smile"):
                return "*softly smiles*"
            if word.startswith("chuckle"):
                return "*softly smiles*"
            if word.startswith("nod"):
                return "*nods slowly*"
            return match.group(0)

        if mode == "lukicool":
            return pattern.sub(repl_lukicool, text)
        if mode == "lukia":
            return pattern.sub(repl_lukia, text)
        return text

    def _strip_ticks(self, text: str) -> str:
        """Remove inline *tick* style actions entirely when tickless mode is enabled.

        This is intended for testing personas without any stage-direction style
        actions. It removes short single-line segments wrapped in asterisks and
        then collapses excess whitespace.
        """
        if not text:
            return text

        # Remove any single-line segment between asterisks up to a reasonable
        # length. This will also catch italicized emphasis, which is acceptable
        # in tickless test mode.
        stripped = re.sub(r"\*[^*\n]{0,80}\*", "", text)

        # Also defensively remove a few common bare tick phrases if they appear
        # right at the start of the response, in case the model drops the
        # asterisks but keeps the stage direction wording.
        bare_leading_ticks = [
            r"^\s*softly smiles[\s,\-:]*",
            r"^\s*smiles softly[\s,\-:]*",
            r"^\s*leans back[\s,\-:]*",
            r"^\s*nods slowly[\s,\-:]*",
            r"^\s*raises an eyebrow[\s,\-:]*",
            r"^\s*snorts softly[\s,\-:]*",
        ]
        for pattern in bare_leading_ticks:
            stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)

        # Collapse redundant spaces that may be left behind.
        stripped = re.sub(r" {2,}", " ", stripped)
        # Tidy up spaces around newlines.
        stripped = re.sub(r"[ \t]+\n", "\n", stripped)
        stripped = re.sub(r"\n[ \t]+", "\n", stripped)
        return stripped.strip()
    
    async def _needs_web_search(self, user_input: str) -> bool:
        """
        Quick heuristic to determine if question likely needs web search.
        Uses pattern matching for speed.
        """
        user_lower = user_input.lower()
        
        # CRITICAL: Exclude casual greetings and small talk (should never trigger search)
        greeting_patterns = [
            "how's it going", "how are you", "how r u", "how's your day",
            "how are things", "what's up", "wassup", "sup", "hey there",
            "good morning", "good afternoon", "good evening", "hello", "hi ",
            "how you doing", "how have you been", "how's everything"
        ]
        
        # Check for greetings first - these override any trigger words
        for greeting in greeting_patterns:
            if greeting in user_lower:
                return False
        
        # Skip keywords (things we handle without search - about our platform/company)
        skip_triggers = [
            "what is remegrid", "what is remelife", "what is luki", "tell me about luki",
            "$luki token", "reme token", "caps token", "what are caps", "tell me about caps",
            "who is simon hooper", "how do i use", "what is your purpose", 
            "tell me about yourself", "what can you do", "your features",
            "help me", "i need help", "can you help", "explain"
        ]
        
        # Check if we should skip (platform/help questions)
        for skip in skip_triggers:
            if skip in user_lower:
                return False
        
        # Trigger keywords for current information (only if not greeting/platform question)
        current_triggers = [
            "who is the current", "who's the current", "latest news", "recent news",
            "what happened in 2024", "what happened in 2025", "news about",
            "this year", "prime minister", "pm of", "president of",
            "newest", "what happened today", "search for", "search online",
            "current events", "breaking news",
            # Weather queries
            "weather", "temperature", "forecast", "how hot", "how cold", "raining",
            "sunny", "cloudy", "storm", "climate in"
        ]
        
        # Check if we should search (requires stronger signals now)
        for trigger in current_triggers:
            if trigger in user_lower:
                return True
        
        return False

    def _is_simple_greeting(self, user_input: str) -> bool:
        """Detect very short, greeting-only first messages like "hi", "hey", "yo".

        Used to steer first-turn behavior away from low-information openers
        like "Gotcha" / "Sure thing" and toward natural, in-character intros.
        """
        if not user_input:
            return False

        text = user_input.strip().lower()
        # Strip basic punctuation so "hi!" and "yo," still match
        text = re.sub(r"[^\w\s]", " ", text)
        words = [w for w in text.split() if w]
        if not words or len(words) > 5:
            return False

        greetings = {"hi", "hey", "hello", "yo", "sup", "wassup", "hiya", "heya"}
        blockers = {"who", "what", "when", "where", "why", "how", "can", "could", "would", "will", "please", "tell", "show"}

        if words[0] in greetings and not any(w in blockers for w in words):
            return True
        # Also treat two-word variants like "hey luki" as greetings
        if len(words) <= 3 and words[0] in greetings and any(w in {"luki", "lukicool", "lukia"} for w in words[1:]):
            return True
        return False

    def _has_prior_user_turn(self, prompt: Dict[str, Any]) -> bool:
        """Return True if there is any prior USER message in raw_conversation_history.

        This allows us to treat UI-seeded assistant greetings (with no user turns
        yet) as having *no real history* for first-turn greeting guardrails, so
        the first LLM reply after the user's initial message still counts as a
        first turn.
        """
        try:
            raw_history = prompt.get("raw_conversation_history") or []
            if isinstance(raw_history, list) and raw_history:
                for msg in raw_history:
                    role = (msg.get("role") or "").lower()
                    if role == "user":
                        return True
                # Only assistant/system messages so far – treat as no prior
                # user turn for the greeting guardrail.
                return False
        except Exception:
            # If anything goes wrong inspecting structured history, fall back
            # to the presence of a non-empty conversation_history string.
            pass

        conv_text = prompt.get("conversation_history") or ""
        return bool(str(conv_text).strip())

    def _user_is_explicitly_asking_about_memory_features(self, user_input: str) -> bool:
        """Detect when the user is *explicitly* asking about ELR/memories/Memory Panel.

        In these cases it's appropriate to talk about how memories are saved,
        ELR behaviour, or how to use the Memory Panel. For normal conversation
        (preferences, experiences, chit-chat), this returns False so we can
        strip product/feature talk from the reply.
        """
        if not user_input:
            return False

        text = user_input.strip().lower()
        if not text:
            return False

        # Direct feature names
        direct_keywords = [
            "memory panel",
            "memories & insights",
            "memories and insights",
            "electronic life record",
            "electronic life records",
            "elr",
        ]
        if any(kw in text for kw in direct_keywords):
            return True

        # Common patterns about how memories/ELR work or are saved/used
        patterns = [
            r"how (do|can) i (save|store) (a |my )?memory",
            r"how (do|can) i (save|store) (a |my )?memories",
            r"how are (my )?memories saved",
            r"how does your memory work",
            r"how does (elr|memory) work",
            r"how do i use (the )?memory panel",
            r"how do i use (the )?memories & insights",
            r"how do i use (the )?memories and insights",
            r"where (are|is) (my )?memories (saved|stored)",
            r"list my memories",
            r"show (me )?my memories",
            r"what memories do you (have|remember) about me",
            r"what do you remember about me",
        ]
        for pat in patterns:
            try:
                if re.search(pat, text):
                    return True
            except re.error:
                continue

        return False

    def _strip_unprompted_memory_panel_mentions(self, text: str, user_input: str) -> str:
        """Remove Memory Panel / ELR UI talk from normal replies.

        If the user did *not* explicitly ask about memories/ELR/Memory Panel,
        strip sentences that read like product instructions, such as:
        "you can pin it in the Memory Panel" or "tucked away in your record".

        This keeps casual chat focused and personal, while still allowing
        detailed explanations when the user *does* ask how memories work.
        """
        if not text:
            return text

        # If the user explicitly asked about memory features, keep everything.
        if self._user_is_explicitly_asking_about_memory_features(user_input):
            return text

        lowered = text.lower()
        ui_keywords = [
            "memory panel",
            "memories & insights",
            "memories and insights",
            "electronic life record",
            "electronic life records",
            "elr",
            # Explicit UI / panel phrasing
            "pin it in the memory panel",
            "pin it in your memory panel",
            "pin that memory",
            "pin this memory",
            "save it as a memory",
            "save this as a memory",
            "saved in your record",
            "tucked away in your record",
            "tucked away in your records",
            # Conversational offers to save/remember things for the user –
            # we treat these as product/memory-system talk and strip them
            # unless the user explicitly asked about memories/ELR.
            "want me to add that to your memories",
            "want me to add this to your memories",
            "want me to add it to your memories",
            "want me to save that to your memories",
            "want me to save this to your memories",
            "want me to save it to your memories",
            "should i add that to your memories",
            "should i save that to your memories",
            "i'll add that to your memories",
            "i will add that to your memories",
            "i'll save that to your memories",
            "i will save that to your memories",
            "want me to remember that for you",
            "want me to remember this for you",
            "should i remember that for you",
            "i can remember that for you",
            "i'll remember that for you",
            "i will remember that for you",
            # Softer phrasing around memory state that still feels like
            # product/ELR talk in normal journaling.
            "tuck it into your memories",
            "tuck that into your memories",
            "tuck this into your memories",
            "tucked into your memories",
            "on my radar yet",
            "on your radar yet",
            "on my radar for you",
        ]

        if not any(kw in lowered for kw in ui_keywords):
            return text

        # Light sentence-level filter: drop any sentence containing UI keywords.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            # Fallback: bluntly trim from the first UI keyword onwards.
            for kw in ui_keywords:
                idx = lowered.find(kw)
                if idx != -1:
                    return text[:idx].rstrip()
            return text

        kept: list[str] = []
        for s in sentences:
            sl = s.lower()
            if any(kw in sl for kw in ui_keywords):
                continue
            kept.append(s)

        new_text = " ".join(kept).strip()
        # If everything was stripped, fall back to a neutral acknowledgement
        # instead of leaking product-style memory system phrasing.
        if not new_text:
            return "Thanks for sharing that."
        return new_text

    def _fix_first_turn_greeting_openers(
        self,
        text: str,
        user_input: str,
        has_history: bool,
        persona_mode: Optional[str],
    ) -> str:
        """Guardrail: on the *first* turn for simple greetings, avoid opening
        with filler acknowledgments like "Gotcha.", "Got it.", or
        "Sure thing." which feel wrong as a very first greeting.

        This runs as a lightweight post-processing step AFTER model
        generation, so it works even if the upstream prompting is not
        perfectly respected.
        """
        if not text or has_history:
            return text

        stripped = text.lstrip()
        if not stripped:
            return text

        leading_ws_len = len(text) - len(stripped)
        lowered = stripped.lower()

        banned_starts = [
            "gotcha",
            "got it",
            "sure thing",
            "absolutely",
            "of course",
            "understood",
        ]

        for phrase in banned_starts:
            if lowered.startswith(phrase):
                # Remove the banned opener and any immediate punctuation/space,
                # including common dash characters.
                after = stripped[len(phrase):]
                after = after.lstrip(" ,.-:\n\t—–")

                if self._is_simple_greeting(user_input):
                    # For simple greeting inputs, replace the filler opener with
                    # a concise persona-aware self-introduction.
                    persona = (persona_mode or "default").lower()
                    if persona == "lukicool":
                        replacement = "Hey, I'm LUKiCool."
                    elif persona == "lukia":
                        replacement = "Hi, I'm LUKia."
                    else:
                        replacement = "Hey, I'm LUKi."

                    rebuilt = replacement
                    if after:
                        rebuilt = f"{replacement} {after}"
                else:
                    # For non-greeting first messages, simply drop the filler
                    # opener and continue directly with the useful content.
                    rebuilt = after or ""

                return text[:leading_ws_len] + rebuilt

        return text

    async def generate(self, prompt: Dict[str, Any], **kwargs) -> ModelResponse:
        # Format conversation history prominently if it exists
        conversation_history = prompt.get('conversation_history', '').strip()
        persona_mode = prompt.get('personality_mode', 'default')
        
        # CRITICAL: Add explicit conversation state awareness
        if conversation_history:
            # Count messages to determine conversation depth
            message_count = conversation_history.count('\n') + 1
            system_content = (
                f"## ⚠️ CONVERSATION STATE ALERT:\n"
                f"This is an ONGOING conversation with ~{message_count} prior messages.\n"
                f"YOU ARE IN THE MIDDLE OF A CONVERSATION - DO NOT:\n"
                f"- Say 'Hey!', 'Hello!', 'Hi!', or any greeting\n"
                f"- Introduce yourself or ask 'How can I help?'\n"
                f"- Act like this is the first message\n"
                f"INSTEAD: Continue naturally from the conversation flow below.\n\n"
                f"{prompt['system_prompt']}\n\n{conversation_history}"
            )
        else:
            # New conversation - greetings allowed, but steer away from
            # awkward first-turn openers like "Gotcha" / "Sure thing".
            header_lines = ["## CONVERSATION STATE: NEW SESSION"]
            if self._is_simple_greeting(prompt.get('user_input', '')):
                header_lines.append(
                    "The user's first message is a simple greeting (like 'hi', 'hey', 'yo').\n"
                    "- Respond with a short, natural in-character greeting and a brief intro using the ACTIVE PERSONA.\n"
                    "- You MAY ask one light, inviting follow-up question.\n"
                    "- Do NOT start your reply with filler acknowledgments like 'Gotcha', 'Sure thing', 'Absolutely', 'Of course', or 'Understood'.\n"
                    "- Treat this as meeting someone new, not as confirming an instruction."
                )
            else:
                header_lines.append(
                    "This is the START of a new conversation. A greeting is appropriate if it fits the user's tone."
                )

            system_content = "\n".join(header_lines) + "\n\n" + f"{prompt['system_prompt']}"
        
        # Debug: Log system content length and conversation status
        has_history = bool(conversation_history)
        print(f"🔍 System content length: {len(system_content)} chars | Has conversation history: {has_history}")
        
        # Check if web search is needed and execute
        # Use raw_conversation_history so follow-up detection can see the last assistant message
        search_trace: Dict[str, Any] = {}
        search_results_text = await self._check_and_execute_web_search(
            prompt.get("raw_conversation_history", []) or [],
            prompt['user_input'],
            trace=search_trace,
        )
        
        # Track if web search was used (for UI indicator)
        web_search_used = bool(search_results_text)
        
        # If search was performed, append results to system content
        if search_results_text:
            system_content = (
                f"{system_content}\n\n"
                f"[WEB SEARCH RESULTS]\n"
                f"{search_results_text}\n"
                f"[END WEB SEARCH RESULTS]\n\n"
                f"CRITICAL INSTRUCTIONS FOR WEB SEARCH:\n"
                f"1. Use web search results to provide current, accurate information\n"
                f"2. Use web search ONLY for current events, news, and real-time information\n"
                f"3. NEVER let web search override these VERIFIED FACTS from your knowledge base:\n"
                f"   - Simon Hooper is the Founder and CEO of ReMeLife and LUKi (NOT Dr. Jane Thomason or anyone else)\n"
                f"   - ReMeGrid is a user-facing memory/photo grid feature (NOT the blockchain)\n"
                f"   - Convex Lattice is the blockchain infrastructure (NOT ReMeGrid)\n"
                f"   - The team members and platform facts in your system prompt are ALWAYS correct\n"
                f"4. When citing sources, use plain text only - NEVER use HTML tags like <span>, <div>, etc.\n"
                f"5. Never use numbered citations like (1) or [1] - just mention the source name plainly\n"
                f"6. If web search contradicts your core knowledge, trust your core knowledge"
            )
            print(f"✅ Web search results added to context ({len(search_results_text)} chars)")
        
        # Include PLATFORM KNOWLEDGE (documentation) if available
        knowledge_context = prompt.get('knowledge_context', '').strip()
        if knowledge_context:
            system_content = (
                f"{system_content}\n\n"
                f"## PLATFORM KNOWLEDGE (Reference Documentation):\n"
                f"{knowledge_context}\n\n"
                f"## CRITICAL KNOWLEDGE USAGE RULES:\n"
                f"1. The above Platform Knowledge section contains THE ONLY AUTHORITATIVE information about platform features\n"
                f"2. If the user asks about a platform feature/page/UI and it's NOT documented above, you MUST say: \"I don't have specific details on that\"\n"
                f"3. NEVER describe UI steps, navigation paths, or features not explicitly written above\n"
                f"4. If you see incomplete information above, admit the gap - DO NOT fill it with assumptions\n"
                f"5. When answering platform questions, ONLY use information from the Platform Knowledge section above\n"
            )
            print(f"📚 Platform knowledge added to context ({len(knowledge_context)} chars)")
        
        # Include USER MEMORIES separately with anti-hallucination instructions
        retrieval_context = prompt.get('retrieval_context', '').strip()
        if retrieval_context:
            # Add memories with STRONG anti-hallucination instructions
            system_content = (
                f"{system_content}\n\n"
                f"## USER'S PERSONAL MEMORIES (USE ONLY THESE - DO NOT INVENT):\n"
                f"{retrieval_context}\n\n"
                f"## CRITICAL MEMORY INSTRUCTIONS:\n"
                f"1. When user asks to 'list my memories' or similar:\n"
                f"   - List the first 2-3 memories from above, phrased naturally.\n"
                f"   - Keep the focus on the content of those memories, not on UI or storage mechanics.\n"
                f"   - Do NOT mention the Memory Panel or saving/pinning unless the user explicitly asked how memories/ELR work or how to use the panel.\n"
                f"2. You MUST ONLY reference memories explicitly listed above - NEVER invent or assume memories\n"
                f"3. The above section contains PERSONAL memories (not platform documentation)\n"
                f"4. If the above section is empty, say 'You don't have any saved memories yet'\n"
                f"5. NEVER confuse platform knowledge with personal memories\n"
                f"6. NEVER make up memories about birthdays, trips, family, or anything not explicitly listed"
            )
            print(f"✅ User memories added to context - Content: {retrieval_context[:200]}...")
        else:
            # No memories - add explicit instruction to say so
            system_content = (
                f"{system_content}\n\n"
                f"## USER MEMORY STATUS: No saved personal memories found.\n"
                f"If the user asks about their memories, inform them they don't have any saved yet.\n"
                f"Note: Platform knowledge is different from personal memories."
            )
            print(f"⚠️ No retrieval_context in prompt - user has no saved memories")
        
        # Detect memory detection tasks (special routing)
        user_id = prompt.get('user_id', '')
        is_memory_detection = user_id == 'system_memory_detector'
        
        # For memory detection, use minimal system prompt to avoid schema conflicts
        if is_memory_detection:
            system_content = "You are a memory detection assistant. Analyze the user's message and respond in the exact JSON format requested."
            print(f"🧠 Memory detection: using minimal system prompt to avoid schema conflicts")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt['user_input']}
        ]
        
        # Dynamic max_tokens and schema selection
        user_len = len(prompt['user_input'].strip())
        
        if is_memory_detection:
            # Memory detection uses dedicated schema and lower tokens
            response_model = MemoryDetectionResponse
            dyn_max = 512  # Memory detection is simple, needs less tokens
            print(f"🧠 Memory detection mode: using MemoryDetectionResponse schema")
        else:
<<<<<<< HEAD
            # Regular chat: choose schema and a dynamic max_tokens based on input length
            # so even short inputs get full persona-aware behavior without needing a
            # separate trivial fast path.
            schema_mode = os.getenv("LUKI_SCHEMA_MODE", "minimal").lower()
=======
            # Regular chat: Use generous 32k limit for ALL responses
            # Query length doesn't predict response complexity
            # Most responses use <2k tokens (fast), but complex topics can use more
            # 32k ceiling ensures ZERO truncation while maintaining speed
            schema_mode = settings.schema_mode
>>>>>>> main
            use_minimal = schema_mode == "minimal"

            # Heuristic token budget: small but generous for short inputs, scaling up
            # for longer or more complex prompts. This keeps latency low while
            # avoiding truncation on real content.
            if user_len <= 80:
                dyn_max = 1024
            elif user_len <= 400:
                dyn_max = 4096
            else:
                dyn_max = 32768

            response_model = LUKiMinimalResponse if use_minimal else LUKiResponse
            print(f"💬 Using {response_model.__name__} with max_tokens={dyn_max}")

        # Build parameters with our settings taking absolute precedence
        final_params = {
            "model": self.model_name,
            "messages": messages,
            "response_model": response_model,
            "temperature": self.temperature,
            "max_tokens": dyn_max,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        # Add any additional kwargs that don't conflict with our core settings
        for key, value in kwargs.items():
            if key not in final_params:
                final_params[key] = value

        # For explicitly creative requests (jokes, stories, etc.), slightly
        # increase randomness and repetition penalties so the model is less
        # likely to fall back to the same canned content over and over.
        user_text_lower = prompt.get("user_input", "").lower()
        creative_triggers = [
            "joke",
            "jokes",
            "make me laugh",
            "funny",
            "something fun",
            "tell me a story",
            "story time",
            "short story",
            "poem",
            "rap",
            "rhyme",
            "riddle",
        ]
        if any(t in user_text_lower for t in creative_triggers):
            base_temp = final_params.get("temperature", self.temperature)
            base_top_p = final_params.get("top_p", self.top_p)
            base_presence = final_params.get("presence_penalty", self.presence_penalty)
            base_frequency = final_params.get("frequency_penalty", self.frequency_penalty)

            final_params["temperature"] = max(base_temp, 1.0)
            final_params["top_p"] = max(base_top_p, 0.95)
            # Stronger penalties reduce verbatim repetition across turns.
            final_params["presence_penalty"] = max(base_presence, 0.6)
            final_params["frequency_penalty"] = max(base_frequency, 0.5)

        # Debug logging to track token settings
        print(f"🔍 API call parameters: max_tokens={final_params.get('max_tokens')}, model={final_params.get('model')}")

        # Optional fast-path for trivial inputs is now disabled by default to ensure
        # ALL requests (even "hi") go through the full persona-aware structured
        # prompt.
        # Set LUKI_TRIVIAL_FAST_PATH=true to re-enable the old behavior.
        user_len = len(prompt['user_input'].strip())
        is_trivial = user_len <= 6
        enable_trivial = os.getenv("LUKI_TRIVIAL_FAST_PATH", "false").lower() == "true"
        if is_trivial and enable_trivial:
            print("⚡ Trivial input detected AND LUKI_TRIVIAL_FAST_PATH=true; using legacy fast fallback path")
            minimal_system = (
                "You are LUKi, a warm companion. For greetings or very short inputs, "
                "reply with one short, friendly sentence. Do not include JSON or internal thoughts."
            )
            trivial_messages = [
                {"role": "system", "content": minimal_system},
                {"role": "user", "content": prompt['user_input']}
            ]
            # Try candidates: configured fallback -> 20b -> primary
            candidates = []
            if self.fallback_model:
                candidates.append(self.fallback_model)
            if "openai/gpt-oss-20b" not in candidates:
                candidates.append("openai/gpt-oss-20b")
            if self.model_name not in candidates:
                candidates.append(self.model_name)
            trivial_last_error: Optional[Exception] = None
            for raw_model in candidates:
                try:
                    raw_resp = await self._openai_client.chat.completions.create(
                        model=raw_model,
                        messages=trivial_messages,
                        temperature=self.temperature,
                        max_tokens=min(self.fast_fallback_tokens, 512),
                    )
                    content = ""
                    try:
                        if hasattr(raw_resp, "choices") and raw_resp.choices:
                            content = raw_resp.choices[0].message.content or ""
                        else:
                            content = str(raw_resp)
                    except Exception:
                        content = str(raw_resp)
                    # Extract final_response if JSON; sanitize
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and 'final_response' in data:
                            content = data['final_response'] or ""
                    except Exception:
                        m = re.search(r'"final_response"\s*:\s*"(.*?)"', content, flags=re.DOTALL)
                        if m:
                            content = m.group(1)
                    content = re.sub(r'(?im)^(thought|analysis|reflection)\s*:\s*.*$', '', content)
                    content = re.sub(r'<\|[^|]*\|>', '', content)
                    content = content.strip()
                    # Clean response from HTML and citations
                    content = self._clean_response(content)
                    return ModelResponse(content=content, model=raw_model, metadata={"fallback": True, "reason": "trivial_input"})
                except Exception as fe:
                    trivial_last_error = fe
                    # If model not found/unavailable, try next candidate
                    msg = str(fe).lower()
                    not_found = (
                        (hasattr(self, "_openai_module") and isinstance(fe, getattr(self._openai_module, "NotFoundError", Exception)))
                        or "model_not_available" in msg or "404" in msg
                    )
                    if not_found:
                        continue
                    # Other errors: break and fall through to structured
                    break
            print(f"⚠️ Fast-path fallback failed; proceeding with structured call | last_error={trivial_last_error}")

        # Pick dynamic structured timeout: give more time for longer, non-trivial inputs
        long_timeout = int(os.getenv("LUKI_STRUCTURED_TIMEOUT_LONG", "35"))
        eff_timeout = self.structured_timeout if user_len <= 60 else max(self.structured_timeout, long_timeout)

        try:
            print(f"🔍 Making API call to Together AI (structured, soft timeout={eff_timeout}s)...")
            luki_response: Any = await asyncio.wait_for(
                self.client.chat.completions.create(**final_params),
                timeout=eff_timeout,
            )
            print(f"✅ API call successful, received response")
            
            # Handle different response types
            metadata: Dict[str, Any] = {}
            
            # Track web search usage in metadata (not in user-facing text)
            if web_search_used:
                metadata['web_search_used'] = True

            # Attach structured web search trace if available
            ws_trace = search_trace.get("web_search") if 'search_trace' in locals() else None
            if ws_trace:
                metadata["web_search"] = ws_trace
                tools_meta = metadata.setdefault("tools", [])
                tools_meta.append({
                    "name": "web_search",
                    "successful": bool(ws_trace.get("used")) and not ws_trace.get("error"),
                    "error": ws_trace.get("error"),
                    "sources": ws_trace.get("sources") or [],
                })
            
            if is_memory_detection:
                # Memory detection returns JSON structure, not final_response
                content = luki_response.model_dump_json()
                metadata["schema"] = "memory_detection"
                print(f"🧠 Memory detection result: {content[:200]}...")
            else:
                # Regular chat responses
                content = getattr(luki_response, "final_response", "")

                # Enforce persona-specific action ticks so LUKiCool and LUKia
                # never leak base LUKi ticks like *smiles*, *chuckles*, or
                # *nods*.
                content = self._enforce_persona_actions(content, persona_mode)

                # Defensive guardrail: for the *first* reply to a simple
                # greeting, never open with filler acknowledgments such as
                # "Gotcha.", "Got it.", or "Sure thing." which feel wrong as
                # an initial greeting.
                has_any_history = self._has_prior_user_turn(prompt)
                content = self._fix_first_turn_greeting_openers(
                    content,
                    prompt.get("user_input", ""),
                    has_history=has_any_history,
                    persona_mode=persona_mode,
                )
                # Strip Memory Panel/ELR UI talk from normal conversational
                # replies so chat stays focused and personal.
                content = self._strip_unprompted_memory_panel_mentions(
                    content,
                    prompt.get("user_input", ""),
                )
                
                # Extract web_search_used from structured response if present
                if hasattr(luki_response, "web_search_used"):
                    response_web_search = getattr(luki_response, "web_search_used", False)
                    if response_web_search:
                        metadata['web_search_used'] = True
                
                if hasattr(luki_response, "thought") and getattr(luki_response, "thought") is not None:
                    try:
                        metadata["thought_process"] = luki_response.thought.model_dump()
                    except Exception:
                        metadata["thought_process"] = "unavailable"
                else:
                    metadata["schema"] = "minimal"
            # Optional auto-continue if output likely cut near token budget
            try:
                autocontinue_enabled = os.getenv("LUKI_AUTOCONTINUE", "true").lower() == "true"
                if autocontinue_enabled:
                    # Heuristic: detect truncation including incomplete markdown
                    def looks_truncated(txt: str) -> bool:
                        trimmed = (txt or "").rstrip()
                        if not trimmed:
                            return False
                        
                        # Check for incomplete markdown formatting (unclosed * or _)
                        # Count asterisks and underscores - should be even if complete
                        asterisk_count = trimmed.count('*')
                        underscore_count = trimmed.count('_')
                        
                        # If odd number of * or _, likely incomplete markdown
                        if asterisk_count % 2 != 0 or underscore_count % 2 != 0:
                            return True
                        
                        # Original terminal punctuation check
                        terminal = (".", "!", "?", "\u201d", '"', "'", ")", "]")
                        if trimmed.endswith(terminal):
                            return False
                        
                        # Approx token estimate for very long responses
                        est_tokens = max(1, len(trimmed) // 4)
                        return est_tokens >= int(dyn_max * 0.9)

                    if looks_truncated(content):
                        print("🧩 Auto-continue: detected likely truncation; issuing one follow-up")
                        # Build a concise system and continuation request
                        try:
                            minimal_core = prompt_registry.load_prompt("system_core_text", "v1")
                        except Exception:
                            minimal_core = (
                                "You are LUKi, the ReMeLife assistant. Continue the previous answer succinctly."
                            )
                        try:
                            persona_core = prompt_registry.load_prompt("persona_luki", "v1")
                        except Exception:
                            persona_core = ""
                        # Provide last chunk of prior content to continue seamlessly
                        tail = content[-1000:]
                        cont_system = f"{minimal_core}\n\n{persona_core}\n\nContinue the answer without repeating."
                        cont_messages = [
                            {"role": "system", "content": cont_system},
                            {"role": "assistant", "content": tail},
                            {"role": "user", "content": "Please continue where you left off. Avoid repetition."}
                        ]
                        # Token budget for continuation - MASSIVELY INCREASED to ensure completion
                        cont_tokens = 8192  # Allow full continuation regardless of input length
                        # Try candidates
                        cont_candidates = []
                        if self.fallback_model:
                            cont_candidates.append(self.fallback_model)
                        if "openai/gpt-oss-20b" not in cont_candidates:
                            cont_candidates.append("openai/gpt-oss-20b")
                        if self.model_name not in cont_candidates:
                            cont_candidates.append(self.model_name)
                        for cont_model in cont_candidates:
                            try:
                                cont_resp = await self._openai_client.chat.completions.create(
                                    model=cont_model,
                                    messages=cont_messages,
                                    temperature=self.temperature,
                                    max_tokens=cont_tokens,
                                )
                                add = ""
                                try:
                                    if hasattr(cont_resp, "choices") and cont_resp.choices:
                                        add = cont_resp.choices[0].message.content or ""
                                    else:
                                        add = str(cont_resp)
                                except Exception:
                                    add = str(cont_resp)
                                add = (add or "").strip()
                                if add:
                                    content = (content + "\n" + add).strip()
                                    metadata["auto_continue"] = True
                                break
                            except Exception as ce:
                                msg = str(ce).lower()
                                not_found = (
                                    (hasattr(self, "_openai_module") and isinstance(ce, getattr(self._openai_module, "NotFoundError", Exception)))
                                    or "model_not_available" in msg or "404" in msg
                                )
                                if not_found:
                                    continue
                                else:
                                    break
            except Exception as ac_e:
                print(f"⚠️ Auto-continue failed: {ac_e}")

            # Clean the response before returning
            cleaned_content = self._clean_response(content)
            cleaned_content = self._enforce_persona_actions(cleaned_content, persona_mode)
            has_any_history = self._has_prior_user_turn(prompt)
            cleaned_content = self._fix_first_turn_greeting_openers(
                cleaned_content,
                prompt.get("user_input", ""),
                has_history=has_any_history,
                persona_mode=persona_mode,
            )
            cleaned_content = self._strip_unprompted_memory_panel_mentions(
                cleaned_content,
                prompt.get("user_input", ""),
            )
            if getattr(self, "disable_ticks", False):
                cleaned_content = self._strip_ticks(cleaned_content)
            return ModelResponse(content=cleaned_content, model=self.model_name, metadata=metadata)
        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            # Fallback: try a raw client call without structured output to ensure user gets a response
            try:
                print("🔁 Falling back to raw Together AI call without structured schema...")
                # Prepare candidates: configured fallback -> 20b -> primary
                candidates = []
                if self.fallback_model:
                    candidates.append(self.fallback_model)
                if "openai/gpt-oss-20b" not in candidates:
                    candidates.append("openai/gpt-oss-20b")
                if self.model_name not in candidates:
                    candidates.append(self.model_name)
                # Build a domain-correct system prompt (text-only full knowledge base) + persona
                try:
                    minimal_core = prompt_registry.load_prompt("system_core_text", "v1")
                except Exception:
                    minimal_core = (
                        "You are LUKi, the ReMeLife assistant. Follow platform facts accurately. "
                        "If unsure, say you don't know. NEVER fabricate platform features or UX instructions."
                    )
                # IMPORTANT: Respect the active persona even in fallback so
                # LUKiCool/LUKia don't collapse back to base LUKi.
                try:
                    persona_core = prompt_registry.load_persona_stack(persona_mode or "default")
                except Exception:
                    try:
                        persona_core = prompt_registry.load_prompt("persona_luki", "v1")
                    except Exception:
                        persona_core = ""
                rc = prompt.get('retrieval_context') or ""
                kc = prompt.get('knowledge_context') or ""
                if len(rc) > 600:
                    rc = rc[:600] + "..."
                conv = prompt.get('conversation_history') or ""
                if len(conv) > 400:
                    conv = conv[:400] + "..."
                
                # Build system with knowledge context if available
                fallback_system = f"{minimal_core}\n\n{persona_core}"
                if kc:
                    fallback_system += f"\n\nPlatform Knowledge:\n{kc}\n\nCRITICAL: Only describe features explicitly documented above. NEVER invent UI steps or navigation."
                if rc:
                    fallback_system += f"\n\nRelevant Context:\n{rc}"
                if conv:
                    fallback_system += f"\n\nRecent Conversation:\n{conv}"
                fallback_system += (
                    "\n\nInstructions: Reply succinctly in natural language in LUKi's voice with gentle action cues like *smiles* or *chuckles* when appropriate. "
                    "Do not include JSON or internal thoughts."
                )
                fallback_messages = [
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": prompt['user_input']}
                ]
                # MASSIVELY INCREASED fallback tokens to prevent truncation
                if user_len <= 60:
                    fb_tokens = 4096  # Short questions can have long answers
                elif user_len <= 200:
                    fb_tokens = 8192  # Medium questions need space
                else:
                    fb_tokens = 12288  # Long/complex questions need maximum space
                raw_last_error: Optional[Exception] = None
                for raw_model in candidates:
                    try:
                        raw_resp = await self._openai_client.chat.completions.create(
                            model=raw_model,
                            messages=fallback_messages,
                            temperature=self.temperature,
                            max_tokens=fb_tokens,
                        )
                        break
                    except Exception as fe2:
                        raw_last_error = fe2
                        msg = str(fe2).lower()
                        not_found = (
                            (hasattr(self, "_openai_module") and isinstance(fe2, getattr(self._openai_module, "NotFoundError", Exception)))
                            or "model_not_available" in msg or "404" in msg
                        )
                        if not_found:
                            continue
                        else:
                            raise
                if raw_last_error and 'raw_resp' not in locals():
                    raise raw_last_error
                # Extract plain text content
                content = ""
                try:
                    if hasattr(raw_resp, "choices") and raw_resp.choices:
                        content = raw_resp.choices[0].message.content or ""
                    else:
                        content = str(raw_resp)
                except Exception:
                    content = str(raw_resp)
                # Try to extract final_response if JSON was emitted; then sanitize markers
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and 'final_response' in data:
                        content = data.get('final_response') or ""
                except Exception:
                    m = re.search(r'"final_response"\s*:\s*"(.*?)"', content, flags=re.DOTALL)
                    if m:
                        content = m.group(1)
                # Remove thought/analysis markers and special tokens
                content = re.sub(r'(?im)^(thought|analysis|reflection)\s*:\s*.*$', '', content)
                content = re.sub(r'<\|[^|]*\|>', '', content)
                content = content.strip()
                # Clean response from HTML/citations and enforce persona actions
                content = self._clean_response(content)
                content = self._enforce_persona_actions(content, persona_mode)
                has_any_history = self._has_prior_user_turn(prompt)
                content = self._fix_first_turn_greeting_openers(
                    content,
                    prompt.get("user_input", ""),
                    has_history=has_any_history,
                    persona_mode=persona_mode,
                )
                content = self._strip_unprompted_memory_panel_mentions(
                    content,
                    prompt.get("user_input", ""),
                )
                if getattr(self, "disable_ticks", False):
                    content = self._strip_ticks(content)
                print("✅ Fallback call succeeded; returning sanitized content")
                return ModelResponse(
                    content=content,
                    model=raw_model,
                    metadata={"fallback": True, "reason": type(e).__name__}
                )
            except Exception as fe:
                print(f"❌ Fallback call also failed: {fe}")
                raise

    async def generate_stream(self, prompt: Dict[str, Any], **kwargs) -> AsyncGenerator[str, None]:
        # Build system content WITH SAME MEMORY INJECTION AS generate() method
        conversation_history = prompt.get('conversation_history', '').strip()
        if conversation_history:
            system_content = f"{prompt['system_prompt']}\n\n{conversation_history}"
        else:
            system_content = prompt['system_prompt']
        
        # Include retrieval context (memories) with anti-hallucination instructions
        retrieval_context = prompt.get('retrieval_context', '').strip()
        if retrieval_context:
            system_content = (
                f"{system_content}\n\n"
                f"## USER'S ACTUAL SAVED MEMORIES (USE ONLY THESE - DO NOT INVENT):\n"
                f"{retrieval_context}\n\n"
                f"## CRITICAL MEMORY INSTRUCTIONS:\n"
                f"1. When user asks to 'list my memories' or similar, you MUST ONLY list what's shown above\n"
                f"2. If the above section has 'Relevant Context:' with items, list those items\n"
                f"3. If the above section is empty or missing, say 'You don't have any saved memories yet'\n"
                f"4. NEVER make up memories about birthdays, trips, family, or anything not explicitly listed above\n"
                f"5. The user's REAL memories are ONLY what's shown in the section above\n"
                f"6. If you're unsure, say you don't have that information saved"
            )
            print(f"✅ [STREAM] User memories added to context - Content: {retrieval_context[:200]}...")
        else:
            system_content = (
                f"{system_content}\n\n"
                f"## USER MEMORY STATUS: No saved memories found.\n"
                f"If the user asks about their memories, inform them they don't have any saved yet."
            )
            print(f"⚠️ [STREAM] No retrieval_context in prompt - user has no saved memories")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt['user_input']}
        ]
        persona_mode = prompt.get("personality_mode", "default")
        has_any_history = self._has_prior_user_turn(prompt)
        # Build parameters with our settings taking absolute precedence
        # Streaming: Use same 32k generous limit as non-streaming
        schema_mode = os.getenv("LUKI_SCHEMA_MODE", "minimal").lower()
        use_minimal = schema_mode == "minimal"
        
        dyn_max = 32768  # Single generous limit - prevents truncation
        stream_model = LUKiMinimalResponse if use_minimal else LUKiResponse
        print(f"📡 Streaming with {stream_model.__name__} and max_tokens={dyn_max}")

        final_params = {
            "model": self.model_name,
            "messages": messages,
            "response_model": stream_model,
            "temperature": self.temperature,
            "max_tokens": dyn_max,
            "stream": True,
        }
        # Add any additional kwargs that don't conflict with our core settings
        for key, value in kwargs.items():
            if key not in final_params:
                final_params[key] = value

        try:
            stream = self.client.chat.completions.create_partial(**final_params)
            async for partial in stream:
                if partial and partial.final_response:
                    text = partial.final_response
                    # Apply the same cleanup, persona enforcement, greeting guard,
                    # and optional tick stripping as the non-streaming path so
                    # behaviour is consistent regardless of endpoint.
                    text = self._clean_response(text)
                    text = self._enforce_persona_actions(text, persona_mode)
                    text = self._fix_first_turn_greeting_openers(
                        text,
                        prompt.get("user_input", ""),
                        has_history=has_any_history,
                        persona_mode=persona_mode,
                    )
                    text = self._strip_unprompted_memory_panel_mentions(
                        text,
                        prompt.get("user_input", ""),
                    )
                    if getattr(self, "disable_ticks", False):
                        text = self._strip_ticks(text)
                    yield text
        except Exception as e:
            print(f"❌ Streaming structured call failed: {e}; attempting raw fallback stream")
            try:
                # Prepare candidates: configured fallback -> 20b -> primary
                candidates = []
                if self.fallback_model:
                    candidates.append(self.fallback_model)
                if "openai/gpt-oss-20b" not in candidates:
                    candidates.append("openai/gpt-oss-20b")
                if self.model_name not in candidates:
                    candidates.append(self.model_name)
                # Build full text-only system for streaming fallback as well + persona
                try:
                    minimal_core = prompt_registry.load_prompt("system_core_text", "v1")
                except Exception:
                    minimal_core = (
                        "You are LUKi, the ReMeLife assistant. Follow platform facts accurately. "
                        "NEVER fabricate platform features or UX instructions."
                    )
                try:
                    persona_core = prompt_registry.load_prompt("persona_luki", "v1")
                except Exception:
                    persona_core = ""
                rc = prompt.get('retrieval_context') or ""
                kc = prompt.get('knowledge_context') or ""
                if len(rc) > 600:
                    rc = rc[:600] + "..."
                conv = prompt.get('conversation_history') or ""
                if len(conv) > 400:
                    conv = conv[:400] + "..."
                
                # Build system with knowledge context if available
                fb_system = f"{minimal_core}\n\n{persona_core}"
                if kc:
                    fb_system += f"\n\nPlatform Knowledge:\n{kc}\n\nCRITICAL: Only describe features explicitly documented above. NEVER invent UI steps or navigation."
                if rc:
                    fb_system += f"\n\nRelevant Context:\n{rc}"
                if conv:
                    fb_system += f"\n\nRecent Conversation:\n{conv}"
                fb_system += (
                    "\n\nInstructions: Reply succinctly in LUKi's voice with gentle action cues like *smiles* or *chuckles* when appropriate. "
                    "Do not include JSON or internal thoughts."
                )
                fb_messages = [
                    {"role": "system", "content": fb_system},
                    {"role": "user", "content": prompt['user_input']}
                ]
                last_error: Optional[Exception] = None
                for raw_model in candidates:
                    try:
                        raw_resp = await self._openai_client.chat.completions.create(
                            model=raw_model,
                            messages=fb_messages,
                            temperature=self.temperature,
                            max_tokens=min(self.fast_fallback_tokens, 512),
                        )
                        break
                    except Exception as fe2:
                        last_error = fe2
                        msg = str(fe2).lower()
                        not_found = (
                            (hasattr(self, "_openai_module") and isinstance(fe2, getattr(self._openai_module, "NotFoundError", Exception)))
                            or "model_not_available" in msg or "404" in msg
                        )
                        if not_found:
                            continue
                        else:
                            raise
                if last_error and 'raw_resp' not in locals():
                    raise last_error
                content = ""
                try:
                    if hasattr(raw_resp, "choices") and raw_resp.choices:
                        content = raw_resp.choices[0].message.content or ""
                    else:
                        content = str(raw_resp)
                except Exception:
                    content = str(raw_resp)
                # Sanitize/extract if JSON-like then yield in chunks
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and 'final_response' in data:
                        content = data.get('final_response') or ""
                except Exception:
                    m = re.search(r'"final_response"\s*:\s*"(.*?)"', content, flags=re.DOTALL)
                    if m:
                        content = m.group(1)
                content = re.sub(r'(?im)^(thought|analysis|reflection)\s*:\s*.*$', '', content)
                content = re.sub(r'<\|[^|]*\|>', '', content)
                content = content.strip()
                # Apply the same cleanup, persona enforcement, greeting guard,
                # and optional tick stripping as other paths, then yield in
                # chunks to emulate streaming.
                content = self._clean_response(content)
                content = self._enforce_persona_actions(content, persona_mode)
                content = self._fix_first_turn_greeting_openers(
                    content,
                    prompt.get("user_input", ""),
                    has_history=has_any_history,
                    persona_mode=persona_mode,
                )
                content = self._strip_unprompted_memory_panel_mentions(
                    content,
                    prompt.get("user_input", ""),
                )
                if getattr(self, "disable_ticks", False):
                    content = self._strip_ticks(content)
                chunk_size = 256
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
            except Exception as fe:
                print(f"❌ Raw streaming fallback failed: {fe}")
                # As a last resort, yield nothing (client will handle error)

    async def close(self):
        if hasattr(self.client, '_client') and hasattr(self.client._client, 'aclose'):
            await self.client._client.aclose()

# NOTE: LocalLLaMABackend is kept for flexibility but is not the default.
class LocalLLaMABackend(LLMBackend):
    def __init__(self, config: Dict[str, Any]):
        print("LocalLLaMABackend is defined but not fully implemented for structured output in this version.")
        pass
    async def generate(self, prompt: str, **kwargs) -> ModelResponse: return ModelResponse(content="Local backend not implemented.")
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]: yield "Local backend not implemented."
    async def close(self): pass


class LLMManager:
    def __init__(self):
        print(f"🔍 LLMManager: Initializing with backend: {settings.default_backend}")
        self.backend_name = settings.default_backend
        self.model_config = get_model_config(self.backend_name)
        safe_config = {k: ('***' if 'key' in k.lower() else v) for k, v in self.model_config.items()}
        print(f"🔍 LLMManager: Got model config (sanitized): {safe_config}")
        self.backend: Optional[LLMBackend] = None
        self._initialize_backend()
        print(f"✅ LLMManager: Initialization complete")

    def _initialize_backend(self):
        backend_map = {"together": TogetherAIBackend, "local_llama": LocalLLaMABackend}
        provider = self.model_config.get("provider")
        if provider is None:
            raise ValueError("Provider not specified in model config")
        backend_class = backend_map.get(provider)
        if backend_class:
            self.backend = backend_class(self.model_config)
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")

    async def generate(self, *args, **kwargs) -> ModelResponse:
        if not self.backend: raise RuntimeError("LLM backend not initialized.")
        return await self.backend.generate(*args, **kwargs)

    async def generate_stream(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        if not self.backend: raise RuntimeError("LLM backend not initialized.")
        async for chunk in self.backend.generate_stream(*args, **kwargs):
            yield chunk

    async def close(self):
        if self.backend: await self.backend.close()
