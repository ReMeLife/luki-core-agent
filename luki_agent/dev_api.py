"""Development API for LUKi Core Agent."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import json
import logging
import traceback
from dataclasses import asdict
import os
import base64
import io
import zipfile
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_identity_question(text: str) -> bool:
    """Detect simple identity questions like 'who are you' that should be
    answered as the active persona, not via platform/RAG docs.

    This is intentionally lightweight and string-based so it can run in both
    the normal chat and streaming handlers without extra model calls.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    patterns = [
        "who are you",
        "who are you really",
        "what is your name",
        "what's your name",
        "whats your name",
        "who am i talking to",
        "who am i chatting with",
        "tell me about yourself",
    ]
    return any(p in t for p in patterns)


try:
    from .config import settings
    from .llm_backends import LLMManager
    from .context_builder import ContextBuilder
    from .memory.memory_service_client import MemoryServiceClient
    from .project_kb import ProjectKB
    from .safety_chain import SafetyChain
    from .tools import ToolRegistry
    from .module_client import ModuleClient
    logger.info("✅ All core imports successful")
except ImportError as e:
    logger.error(f"❌ CRITICAL IMPORT ERROR: {e}")
    raise

app = FastAPI(title="LUKi Core Agent API", version="2.0.0")

def _bootstrap_project_kb(source_dirs: List[str]) -> List[str]:
    """
    Ensure project KB sources exist in container by optionally bootstrapping
    from an archive provided via env vars. This is needed when using a Dockerfile
    build on Railway because gitignored directories (e.g. "./_context") are not
    part of the build context.

    Env vars:
    - LUKI_PROJECT_KB_ARCHIVE_B64: base64-encoded ZIP archive of project context
    - LUKI_PROJECT_KB_ARCHIVE_URL: HTTPS URL to a ZIP archive of project context
    - LUKI_PROJECT_KB_TARGET_DIR: where to extract (default: first source dir or './_context')
    """
    try:
        # Determine target dir
        default_target = source_dirs[0] if (source_dirs and source_dirs[0]) else "./_context"
        target_dir = os.getenv("LUKI_PROJECT_KB_TARGET_DIR", default_target).strip() or default_target

        # If target dir already exists and is non-empty, do nothing
        if os.path.isdir(target_dir):
            try:
                # non-empty check
                if any(True for _ in os.scandir(target_dir)):
                    logger.info(f"ProjectKB bootstrap: '{target_dir}' exists and is non-empty; skipping bootstrap")
                    return [target_dir]
            except Exception:
                pass

        # Try Base64 first
        b64 = os.getenv("LUKI_PROJECT_KB_ARCHIVE_B64", "").strip()
        if b64:
            try:
                data = base64.b64decode(b64)
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    zf.extractall(target_dir)
                logger.info(f"ProjectKB bootstrap: extracted Base64 archive to '{target_dir}'")
                return [target_dir]
            except Exception as e:
                logger.warning(f"ProjectKB bootstrap (b64) failed: {e}")

        # Try URL next
        url = os.getenv("LUKI_PROJECT_KB_ARCHIVE_URL", "").strip()
        if url:
            try:
                import requests  # local import to avoid hard dep at import-time
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(target_dir)
                logger.info(f"ProjectKB bootstrap: downloaded and extracted archive to '{target_dir}'")
                return [target_dir]
            except Exception as e:
                logger.warning(f"ProjectKB bootstrap (url) failed: {e}")

        # If we reach here, no archive provided or extraction failed
        logger.info(f"ProjectKB bootstrap: no archive provided or extraction failed; using '{target_dir}' as-is")
        return [target_dir]
    except Exception as e:
        logger.warning(f"ProjectKB bootstrap encountered an unexpected error: {e}")
        return source_dirs or ["./_context"]

def _bootstrap_prompts_dir():
    """Bootstrap prompts directory from Base64 archive if needed"""
    try:
        prompts_archive_b64 = os.getenv("LUKI_PROMPTS_ARCHIVE_B64", "").strip()
        target_dir = os.getenv("LUKI_PROMPTS_TARGET_DIR", "/app/prompts").strip()
        
        if not prompts_archive_b64:
            logger.info("Prompts bootstrap: no archive provided; using existing prompts/ if available")
            return
            
        if os.path.exists(target_dir) and os.listdir(target_dir):
            logger.info(f"Prompts bootstrap: {target_dir} already exists with files; skipping extraction")
            return
            
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Decode and extract
        import base64
        import zipfile
        import tempfile
        
        archive_data = base64.b64decode(prompts_archive_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            temp_file.write(archive_data)
            temp_file.flush()
            
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                
        os.unlink(temp_file.name)
        
        extracted_files = os.listdir(target_dir) if os.path.exists(target_dir) else []
        logger.info(f"Prompts bootstrap: extracted {len(extracted_files)} files to '{target_dir}'")
        
    except Exception as e:
        logger.warning(f"Prompts bootstrap encountered an error: {e}")

@app.on_event("startup")
async def on_startup():
    # Initialize and cache the LLM manager once to avoid per-request cold starts
    try:
        logger.info("🔧 Initializing global LLMManager on startup...")
        app.state.llm_manager = LLMManager()
        logger.info("✅ Global LLMManager ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLMManager on startup: {e}")
        # Do not raise, allow lazy init per-request as fallback
        app.state.llm_manager = None

    try:
        logger.info("🔧 Initializing global SafetyChain on startup...")
        app.state.safety_chain = SafetyChain()
        logger.info("✅ SafetyChain ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SafetyChain on startup: {e}")
        app.state.safety_chain = None

    # Bootstrap prompts directory if needed
    _bootstrap_prompts_dir()

    # Initialize Project Knowledge Base (separate from ELR)
    try:
        paths_env = os.getenv("LUKI_PROJECT_KB_PATHS", "").strip()
        source_dirs: List[str] = []
        if paths_env:
            # Support ',', ';', or whitespace separators
            for p in re.split(r"[;,\s]+", paths_env):
                p = p.strip()
                if p:
                    source_dirs.append(p)
        logger.info(f"🔧 ProjectKB parsed source dirs: {source_dirs}")
        # Bootstrap if directories are missing in container due to Docker build context
        source_dirs = _bootstrap_project_kb(source_dirs or ["./_context"])
        # Resolve common absolute/relative variants so local Docker builds work regardless of env value
        resolved_dirs: List[str] = []
        for p in source_dirs:
            candidates = [p]
            if p.startswith("/app/"):
                # Also try relative variant (WORKDIR is /app)
                candidates.append("." + p[len("/app"):])  # '/app/_context' -> './_context'
            elif p.startswith("./"):
                # Also try absolute variant
                candidates.append("/app/" + p[2:])
            chosen = None
            for cand in candidates:
                if os.path.isdir(cand):
                    chosen = cand
                    break
            resolved_dirs.append(chosen or p)
        try:
            # Light debug: show top-level /app entries to aid diagnosis
            top = []
            for name in os.listdir("/app"):
                top.append(name)
                if len(top) >= 15:  # avoid noisy logs
                    break
            logger.info(f"🔎 /app contains (first 15): {top}")
        except Exception:
            pass
        logger.info(f"🔧 ProjectKB effective source dirs (resolved): {resolved_dirs}")
        app.state.project_kb = ProjectKB(source_dirs=resolved_dirs)
        logger.info(f"🔧 ProjectKB initializing with {len(resolved_dirs)} source dirs")
        app.state.project_kb.ingest()
        logger.info("✅ ProjectKB ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ProjectKB: {e}")
        app.state.project_kb = None

    # Initialize ToolRegistry for module tools (cognitive + reporting + memory)
    try:
        logger.info("🔧 Initializing ToolRegistry on startup...")
        tool_registry = ToolRegistry()
        tool_registry.register_cognitive_tools()
        tool_registry.register_reporting_tools()
        tool_registry.register_memory_tools()  # Includes UploadSearchTool
        app.state.tool_registry = tool_registry
        logger.info("✅ ToolRegistry ready (cognitive + reporting + memory tools)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ToolRegistry: {e}")
        app.state.tool_registry = None

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    persona_id: Optional[str] = None
    file_search_mode: Optional[bool] = False  # Explicit file search intent from UI toggle
    client_tag: Optional[str] = None  # Widget mode detection (e.g., "remelife_widget")


class PhotoReminiscenceImageRequest(BaseModel):
    user_id: str
    activity_title: Optional[str] = None
    answers: List[str]
    n: Optional[int] = 1
    account_tier: Optional[str] = "free"  # free, plus, pro - determines image generation limits


async def _maybe_handle_with_tools(request: ChatRequest, safety_chain=None) -> Optional[Dict[str, Any]]:
    """Optionally handle specific journeys via module tools (Epic 1).

    This provides a fast path for:
    - Activity suggestions via cognitive module tools.
    - Simple wellbeing summaries via reporting module tools.
    - File search when file_search_mode is enabled (explicit user toggle).

    For all other inputs, it returns None and the normal LLM pipeline runs.
    
    NOTE: Module tools (cognitive, reporting, engagement) are DISABLED for
    remelife_widget to keep it as a simple stateless page guide.
    """
    # Skip ALL module tools for remelife_widget - it's a stateless page guide
    if request.client_tag == "remelife_widget":
        logger.info("🚫 Skipping module tools for remelife_widget (stateless guide mode)")
        return None
    
    tool_registry = getattr(app.state, "tool_registry", None)
    if tool_registry is None:
        return None

    text = (request.message or "").strip().lower()
    if not text:
        return None

    async def _run_tool(name: str, **kwargs) -> Optional[Dict[str, Any]]:
        logger.info(f"🔧 Executing tool '{name}' with kwargs: {list(kwargs.keys())}")
        
        # Check if tool exists
        tool = tool_registry.get_tool(name)
        if not tool:
            logger.error(f"❌ Tool '{name}' not found in registry. Available tools: {[t['name'] for t in tool_registry.list_tools()]}")
            return None
        
        result = await tool_registry.execute_tool(name, **kwargs)
        
        if not result.success:
            logger.warning(f"⚠️ Tool '{name}' failed: {result.error}")
            return None
        if not result.content:
            logger.warning(f"⚠️ Tool '{name}' returned empty content")
            return None
            
        logger.info(f"✅ Tool '{name}' succeeded with {len(result.content)} chars")
        response_text = result.content
        # Apply output safety filter if available
        if safety_chain is not None:
            try:
                await safety_chain.filter_output(response_text, request.user_id)
            except Exception as e:
                logger.warning(f"Safety output filter (tool '{name}') failed: {e}")
        metadata = result.metadata or {}
        metadata["tool_name"] = name
        return {"response": response_text, "metadata": metadata}
    
    # PRIORITY: If file_search_mode is explicitly enabled, handle file search
    if request.file_search_mode:
        logger.info(f"🔍 File Search Mode ENABLED - processing: '{request.message}'")
        raw_message = request.message.strip()
        msg_lower = raw_message.lower()
        
        # Extract semantic search keywords using LLM (handles "beautiful sun" → "sun sunset", "my mom" → "mom mother")
        extracted_query = await _extract_upload_search_query(raw_message)
        logger.info(f"🔍 [File Search Mode] Extracted query: '{extracted_query}'")
        
        # If LLM couldn't extract anything meaningful, use the raw message as fallback
        # This allows searches for files literally named "hey" or "hi" to still work
        search_query = extracted_query if extracted_query else raw_message
        
        # ALWAYS perform the search first - no rigid rejections
        tool_result = await _run_tool(
            "search_uploads",
            query=search_query,
            user_id=request.user_id,
            limit=10,
        )
        
        # Check if search returned no results - then provide helpful guidance
        if tool_result:
            metadata = tool_result.get("metadata", {})
            upload_results = metadata.get("upload_results", [])
            
            if len(upload_results) == 0:
                logger.info(f"🔍 Search returned 0 results for '{raw_message}'")
                
                # Check if the message looks like casual chat (for friendlier guidance)
                casual_patterns = ["hey", "hi", "hello", "yo", "sup", "how are you", "what's up", "thanks", "ok", "yes", "no"]
                looks_like_chat = msg_lower in casual_patterns or any(msg_lower.startswith(p) for p in casual_patterns)
                
                if looks_like_chat:
                    guidance = (
                        f"I couldn't find any files matching '{raw_message}'. "
                        "If you're trying to chat, turn off **File Search** in the ✶ menu! 💬"
                    )
                else:
                    guidance = (
                        f"I couldn't find any files matching '{raw_message}'. "
                        "Could you try different keywords or be more specific?"
                    )
                
                return {
                    "response": guidance,
                    "metadata": {"no_results_found": True, "original_query": raw_message}
                }
        
        return tool_result

    # Activity suggestion triggers
    activity_keywords = [
        "suggest an activity",
        "suggest some activities",
        "activity suggestion",
        "activity suggestions",
        "what could i do",
        "what should i do",
        "something to do",
        "things to do",
    ]
    if "activity" in text or any(kw in text for kw in activity_keywords):
        ctx = request.context or {}
        return await _run_tool(
            "recommend_cognitive_activity",
            user_id=request.user_id,
            current_mood=ctx.get("current_mood"),
            available_duration=ctx.get("available_duration"),
            carer_available=ctx.get("carer_available", True),
            group_setting=ctx.get("group_setting", False),
            specific_request=request.message,
            max_recommendations=ctx.get("max_recommendations", 3),
        )

    # Wellbeing summary triggers - expanded keywords
    wellbeing_keywords = [
        "wellbeing summary",
        "well-being summary",
        "wellbeing report",
        "how have i been",
        "how am i doing",
        "how have i been doing",
        "how am i doing lately",
        "summary of my week",
        "my weekly summary",
        "what have i been up to",
        "give me a summary",
        "my progress",
        "how's my week",
        "how was my week",
        "show my stats",
        "my statistics",
        "my engagement",
        "how engaged have i been",
        "my activity summary",
    ]
    if any(kw in text for kw in wellbeing_keywords):
        return await _run_tool(
            "generate_wellbeing_report",
            user_id=request.user_id,
            report_type="weekly",
        )

    # Skip tool handling for system users (e.g., memory detector)
    if request.user_id and request.user_id.startswith("system_"):
        return None

    # Detect file search intent when button is OFF - guide user to enable it
    # This prevents the LLM from making up responses about files not existing
    file_search_phrases = [
        "find my", "show my", "get my", "where is my", "where's my",
        "search my", "locate my", "look for my", "looking for my",
        "my file", "my photo", "my image", "my picture", "my upload",
        "my files", "my photos", "my images", "my pictures", "my uploads",
        "find the file", "find the photo", "find the image", "find the picture",
    ]
    
    # Check for file/photo/image words combined with possessive or action words
    has_file_word = any(w in text for w in ["file", "photo", "image", "picture", "upload", "pic"])
    has_find_action = any(w in text for w in ["find", "show", "get", "where", "search", "locate", "look"])
    has_possessive = "my " in text or "my\n" in text
    
    # Exclude questions about HOW to do things (those should go to RAG)
    is_how_question = text.startswith("how") or "how do i" in text or "how can i" in text
    
    # If user is clearly trying to find their files but button is OFF
    if not is_how_question and (
        any(phrase in text for phrase in file_search_phrases) or
        (has_file_word and has_find_action and has_possessive)
    ):
        logger.info(f"🔍 File search intent detected but button is OFF - guiding user")
        guidance_response = (
            "To search your files, please enable the **File Search** button at the top of the chat "
            "(the green button near BETA). Click it so it says 'File Search ON', then ask me again "
            "and I'll find your file! 🔍"
        )
        return {
            "response": guidance_response,
            "metadata": {"guided_file_search": True}
        }

    return None


async def _extract_upload_search_query(user_message: str) -> str:
    """
    Use LLM to extract the actual search terms from a user's upload search request.
    
    Examples:
    - "Show me my recent uploaded files and photos" -> "" (empty = get all recent)
    - "Can you find my upload with 'faith' and 'ai generated' tags?" -> "faith ai generated"
    - "Find my vacation photos from last summer" -> "vacation summer"
    """
    try:
        llm_manager = getattr(app.state, "llm_manager", None)
        if llm_manager is None:
            llm_manager = LLMManager()
            app.state.llm_manager = llm_manager
        
        system_prompt = """You are a semantic search query extractor. Extract keywords that could match file names, tags, or descriptions.

CRITICAL RULES:
1. Extract meaningful nouns and descriptive words from the user's request
2. Include SYNONYMS and related words that might match files (e.g., "mom" → "mom mother", "sun" → "sun sunset")
3. DO NOT respond conversationally - output ONLY keywords or NONE
4. Return keywords on a single line, space-separated
5. Maximum 6 words total

EXAMPLES:
Input: "find my file with a beautiful sun" -> Output: sun sunset sunshine
Input: "show me my mom" -> Output: mom mother
Input: "find pictures of my dog" -> Output: dog puppy pet
Input: "beach vacation photos" -> Output: beach vacation
Input: "faith" -> Output: faith
Input: "find the sunset image" -> Output: sunset sun
Input: "show my family photos" -> Output: family
Input: "find my cat playing" -> Output: cat
Input: "hey" -> Output: NONE
Input: "how are you" -> Output: NONE"""

        extraction_prompt = {
            "system_prompt": system_prompt,
            "user_input": f"Extract keywords from: {user_message}",
            "max_tokens": 50,
        }
        
        response = await llm_manager.generate(prompt=extraction_prompt)
        extracted = response.content.strip().strip('"').strip("'")
        
        # Handle the NONE marker for "get all recent" queries
        if extracted.upper() == "NONE" or not extracted:
            return ""
        
        # Detect if LLM returned a conversational response instead of keywords
        conversational_indicators = [
            "😊", "!", "?", "how can i", "help you", "hi!", "hey!", "hello!",
            "what would you like", "i can help", "let me know", "sorry",
            "cannot", "i'm here", "i am here", "great to"
        ]
        extracted_lower = extracted.lower()
        if any(ind in extracted_lower for ind in conversational_indicators):
            logger.warning(f"⚠️ LLM returned conversational response instead of keywords: '{extracted}'")
            return ""
        
        # If the response is too long, it's probably not keywords
        if len(extracted) > 50 or len(extracted.split()) > 5:
            logger.warning(f"⚠️ LLM returned too many words, treating as no keywords: '{extracted}'")
            return ""
        
        return extracted
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract search query, using empty string: {e}")
        return ""

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/reme/photo-reminiscence-images")
async def photo_reminiscence_images(request: PhotoReminiscenceImageRequest):
    """Generate one or more images for the Photo Reminiscence activity.

    This is a thin proxy over the cognitive module's `/images/photo-reminiscence`
    endpoint so that frontends can call the core agent without talking directly
    to module services.
    """
    if not request.answers:
        raise HTTPException(status_code=400, detail="At least one answer is required")

    try:
        async with ModuleClient() as module_client:
            result = await module_client.generate_photo_reminiscence_images(
                user_id=request.user_id,
                activity_title=request.activity_title,
                answers=request.answers,
                n=request.n or 1,
                account_tier=request.account_tier or "free",
            )

        # Module client returns a small status envelope so callers can
        # distinguish between generic failures and rate limiting.
        if isinstance(result, dict):
            status_val = result.get("status")

            # Rate-limited: surface as HTTP 429 with structured JSON detail so
            # the gateway and frontends can show a clear cooldown message.
            if status_val == "rate_limited":
                detail = result.get("detail") or {"message": "Image generation temporarily rate limited"}
                raise HTTPException(status_code=429, detail=detail)

            # Generic error: map through the underlying status_code if present
            # so observability remains intact, falling back to 502.
            if status_val == "error":
                message = result.get("message") or "Image generation failed"
                status_code = int(result.get("status_code") or 502)
                raise HTTPException(status_code=status_code, detail=message)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Photo reminiscence images endpoint error: %s\n%s",
            e,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating images",
        )

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    # Debug: Log all incoming requests
    logger.info(f"🔍 INCOMING REQUEST: user_id={request.user_id}, message_length={len(request.message)}")
    try:
        logger.info(f"🔍 Step 1: Initializing LLM Manager...")
        llm_manager = getattr(app.state, "llm_manager", None)
        if llm_manager is None:
            logger.info("ℹ️ Global LLMManager not initialized; creating on-demand instance...")
            llm_manager = LLMManager()
            app.state.llm_manager = llm_manager
        logger.info(f"✅ Step 1: LLM Manager initialized")
        
        logger.info(f"🔍 Step 2: Initializing Context Builder...")
        context_builder = ContextBuilder()
        logger.info(f"✅ Step 2: Context Builder initialized")
        safety_chain = getattr(app.state, "safety_chain", None)
        if safety_chain is None:
            try:
                safety_chain = SafetyChain()
                app.state.safety_chain = safety_chain
            except Exception as e:
                logger.warning(f"SafetyChain initialization failed: {e}")
                safety_chain = None
        
        # Prepare user memories (ELR) if gateway provided them (only for authenticated users)
        user_memories = request.context.get("memory_context", []) if request.context else []
        wallet_context = request.context.get("wallet") if request.context else None
        if safety_chain is not None:
            try:
                await safety_chain.filter_input(request.message, request.user_id)
            except Exception as e:
                logger.warning(f"Safety input filter failed: {e}")

        # Optionally handle specific journeys via module tools (Epic 1)
        tool_payload = await _maybe_handle_with_tools(request, safety_chain=safety_chain)
        if tool_payload is not None:
            session_id = request.session_id or "new-session"
            payload = {
                "response": tool_payload["response"],
                "metadata": tool_payload.get("metadata", {}),
                "session_id": session_id,
            }
            logger.info("✅ Responding via module tool instead of LLM")
            return payload

        if safety_chain is not None and user_memories:
            try:
                allowed_memories = await safety_chain.check_consent(
                    user_id=request.user_id,
                    data_type="elr_memories",
                    context={
                        "source": "dev_api_chat",
                        "memory_items": len(user_memories),
                    },
                )
            except Exception as e:
                logger.warning(f"Consent check for memory retrieval failed: {e}")
                allowed_memories = False

            if not allowed_memories:
                logger.info(
                    "ELR memories will not be used in context due to consent or policy settings"
                )
                user_memories = []

        # ProjectKB search (independent of ELR, available to all users)
        proj_docs: List[Dict[str, Any]] = []
        kb = getattr(app.state, "project_kb", None)
        if kb is not None:
            try:
                msg = request.message.strip()
                msg_lower = msg.lower()
                msg_len = len(msg)

                # Only hit ProjectKB when the user is clearly asking about
                # ReMeLife/LUKi/platform concepts. This prevents care- and
                # docs-heavy content from hijacking casual chat (e.g. "tell me a joke").
                platform_keywords = [
                    "reme", "remelife", "remecare", "retegrid", "remegrid", "care2earn",
                    "luki", "lukitoken", "caps", "cap ", "token", "tokens", "wallet",
                    "nft", "genesis", "forum", "market", "dashboard", "elr",
                    "electronic life record", "carefi",
                    # Referral & invite flows  ensure these always use ProjectKB
                    "referral", "referr", "referral link", "invite friends", "community builder",
                    # Rewards & tokenomics language
                    "care action points", "care points", "registration bonus", "referral rewards",
                    "referral system", "referral plan", "passive earnings", "passive income",
                    "three level referral", "3-level referral",
                    "tri-token", "tri token model", "tokenomics", "cad20", "care to earn",
                    "data2earn", "data to earn", "careocracy", "universal basic income", "rubi",
                    # DeFi / staking modules
                    "staking", "ragency", "ragency defi", "community nft market",
                    # Wallet navigation & UI labels
                    "my apps", "explore more", "transaction history", "carefi hub",
                    "your referral link", "copy link button",
                    # Community / programs
                    "vip club", "luki vip club", "luki rewards", "luki rewards program",
                    "rewards program", "community builder referral",
                    "community builder referral program", "charity launch pad",
                    # DAO / governance / foundation
                    "dao", "remelife foundation", "foundation",
                    # NFTs & avatar ecosystem
                    "genesis luki", "genesis luki nft", "luki nft", "luki's friends",
                    "friends collection", "holder raffles", "holder benefits", "play with luki",
                    # Technical / infrastructure
                    "convex lattice", "convex solutions", "remegrid convex", "carefi defi",
                    "electronic life records", "elr data", "ai for elr", "carefi services",
                    # Chat interface features - "how do I" questions
                    "how do i upload", "how to upload", "upload a file", "upload an image",
                    "upload a photo", "my uploads", "file upload", "image upload",
                    "how do i generate", "generate an image", "generate image", "create image",
                    "make an image", "ai image", "image generation",
                    "how do i play", "play a game", "start a game", "games available",
                    "word garden", "memory tiles", "luki arcade", "luki runner",
                    "how do i start", "start an activity", "activities", "cognitive activity",
                    "photo reminiscence", "life story", "life stories",
                    "how do i encrypt", "encrypt my", "encryption", "wallet encryption",
                    "protect my data", "secure my memories",
                    "star button", "star menu", "features menu", "plus button",
                    "where is the menu", "how do i access", "how do i find",
                    # ReMeLife website features - settings, subscription, account
                    "how do i upgrade", "upgrade my account", "subscription", "plus plan", "pro plan",
                    "how do i change", "change my username", "change my password", "change theme",
                    "change profile picture", "profile picture", "avatar",
                    "how do i delete", "delete my account", "delete account", "danger zone",
                    "settings", "account settings", "notifications", "sounds",
                    "where is my", "where can i find", "how to find",
                ]

                if msg and any(kw in msg_lower for kw in platform_keywords):
                    # CRITICAL: Smart top_k selection based on query complexity
                    # Keyword-based boost ensures complete information for critical topics
                    if any(keyword in msg_lower for keyword in ['caps', 'cap', 'earn', 'reward', 'token', 'reme', 'luki']):
                        top_k = 10  # Comprehensive coverage for tokenomics queries
                    elif msg_len <= 80:
                        top_k = 5  # Standard coverage for short queries
                    else:
                        top_k = 8  # Enhanced coverage for longer queries

                    logger.info(f"ProjectKB search: msg_len={msg_len}, top_k={top_k}, query='{msg[:50]}...'")
                    proj_docs = kb.search(msg, top_k=top_k)
                else:
                    logger.info("ProjectKB: skipping search for non-platform or empty query")
                    proj_docs = []
            except Exception as e:
                logger.warning(f"ProjectKB search failed: {e}")

        # Inject a canonical ELR definition when the user is explicitly asking about ELR,
        # so answers are consistent and do not rely purely on model improvisation.
        try:
            msg_lower_for_elr = request.message.strip().lower()
            if "elr" in msg_lower_for_elr or "electronic life record" in msg_lower_for_elr:
                canonical_elr = (
                    "ELR stands for Electronic Life Record. It is ReMeLife’s secure data layer that stores and "
                    "organises the memories and life events you choose to share. It is not a separate app or "
                    "screen; it runs behind the scenes so that, when you give permission through your privacy "
                    "settings, LUKi can recall past details to make conversations feel more continuous and personal."
                )
                proj_docs = [{"content": canonical_elr}] + (proj_docs or [])
        except Exception as e:
            logger.warning(f"ELR canonical definition injection failed: {e}")

        logger.info(
            f"🔍 Step 3: Building context with {len(proj_docs)} project docs and {len(user_memories)} user memories...")
        
        # Log actual memory content for debugging
        if user_memories:
            logger.info("📦 User memories content:")
            for idx, mem in enumerate(user_memories[:3]):  # Log first 3 memories for debugging
                logger.info(f"  Memory {idx}: {mem.get('content', '')[:100]}...")

        # CRITICAL FIX: Keep project docs and user memories SEPARATE
        # Project docs are for knowledge, user memories are personal data
        # We'll pass them separately to the context builder

        # Determine personality mode / persona for this request
        persona_id = request.persona_id
        if persona_id is None and request.context:
            persona_id = request.context.get("persona_id") or request.context.get("personality_mode")
        personality_mode = persona_id or "default"
        logger.info(f"🎭 Persona selection: persona_id={persona_id!r}, personality_mode={personality_mode!r}")

        # Extract world day context if provided (for "today's world day" awareness)
        world_day_context = None
        if request.context:
            world_day_context = request.context.get("world_day")
            if world_day_context:
                logger.info(f"🌍 World day context: {world_day_context.get('name', 'unknown')}")

        context_result = await context_builder.build(
            user_input=request.message,
            user_id=request.user_id,
            conversation_history=request.context.get("conversation_history", []) if request.context else [],
            memory_context=user_memories,  # ONLY user memories here
            knowledge_context=proj_docs,    # Project knowledge separate
            wallet_context=wallet_context,
            personality_mode=personality_mode,
            world_day_context=world_day_context,  # World day awareness
            client_tag=request.client_tag,  # Widget mode detection
        )
        logger.info(f"✅ Step 3: Context built successfully (client_tag={request.client_tag})")
        
        # Log what's in the context result
        if context_result.get("final_prompt", {}).get("retrieval_context"):
            logger.info(f"📦 Retrieval context passed to LLM: {context_result['final_prompt']['retrieval_context'][:200]}...")

        logger.info(f"🔍 Step 4: Generating LLM response...")
        response = await llm_manager.generate(prompt=context_result["final_prompt"])
        if safety_chain is not None:
            try:
                await safety_chain.filter_output(response.content, request.user_id)
            except Exception as e:
                logger.warning(f"Safety output filter failed: {e}")
        logger.info(f"✅ Step 4: LLM response generated successfully")
        # Always include a session_id for gateway compatibility
        session_id = request.session_id or "new-session"
        metadata = response.metadata or {}
        if safety_chain is not None:
            try:
                metadata["safety_metrics"] = safety_chain.get_safety_metrics()
            except Exception as e:
                logger.warning(f"Failed to get safety metrics: {e}")
        payload = {"response": response.content, "metadata": metadata, "session_id": session_id}
        logger.info(f"🚀 Returning response to gateway | chars={len(response.content)}")
        return payload

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            llm_manager = getattr(app.state, "llm_manager", None)
            if llm_manager is None:
                llm_manager = LLMManager()
                app.state.llm_manager = llm_manager
            context_builder = ContextBuilder()
            # ELR user memories provided by gateway (authenticated users only)
            user_memories = request.context.get("memory_context", []) if request.context else []
            wallet_context = request.context.get("wallet") if request.context else None
            safety_chain = getattr(app.state, "safety_chain", None)
            if safety_chain is not None and user_memories:
                try:
                    allowed_memories = await safety_chain.check_consent(
                        user_id=request.user_id,
                        data_type="elr_memories",
                        context={
                            "source": "dev_api_chat_stream",
                            "memory_items": len(user_memories),
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Consent check for memory retrieval failed (stream): %s", e
                    )
                    allowed_memories = False

                if not allowed_memories:
                    logger.info(
                        "ELR memories will not be used in streaming context due to consent or policy settings"
                    )
                    user_memories = []

            # Optionally handle specific journeys via module tools (Epic 1)
            tool_payload = await _maybe_handle_with_tools(request, safety_chain=safety_chain)
            if tool_payload is not None:
                response_text = tool_payload["response"]
                metadata = tool_payload.get("metadata", {})
                logger.info("✅ Streaming response via module tool instead of LLM")
                # Stream as a single chunk followed by done marker
                yield f"data: {json.dumps({'token': response_text})}\n\n"
                yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"
                return

            # Project KB search
            proj_docs: List[Dict[str, Any]] = []
            kb = getattr(app.state, "project_kb", None)
            if kb is not None:
                try:
                    msg = request.message.strip()
                    msg_len = len(msg)
                    msg_lower = msg.lower()
                    # Identity questions like "who are you" should be answered
                    # from the ACTIVE PERSONA, not via platform docs.
                    if msg_len > 12 and not _is_identity_question(msg_lower):
                        platform_keywords = [
                            "reme", "remelife", "remecare", "retegrid", "remegrid", "care2earn",
                            "luki", "lukitoken", "caps", "cap ", "token", "tokens", "wallet",
                            "nft", "genesis", "forum", "market", "dashboard", "elr",
                            "electronic life record", "carefi",
                            # Referral & invite flows – ensure these always use ProjectKB
                            "referral", "referr", "referral link", "invite friends", "community builder",
                            # Rewards & tokenomics language
                            "care action points", "care points", "registration bonus", "referral rewards",
                            "referral system", "referral plan", "passive earnings", "passive income",
                            "three level referral", "3-level referral",
                            "tri-token", "tri token model", "tokenomics", "cad20", "care to earn",
                            "data2earn", "data to earn", "careocracy", "universal basic income", "rubi",
                            # DeFi / staking modules
                            "staking", "ragency", "ragency defi", "community nft market",
                            # Wallet navigation & UI labels
                            "my apps", "explore more", "transaction history", "carefi hub",
                            "your referral link", "copy link button",
                            # Community / programs
                            "vip club", "luki vip club", "luki rewards", "luki rewards program",
                            "rewards program", "community builder referral",
                            "community builder referral program", "charity launch pad",
                            # DAO / governance / foundation
                            "dao", "remelife foundation", "foundation",
                            # NFTs & avatar ecosystem
                            "genesis luki", "genesis luki nft", "luki nft", "luki's friends",
                            "friends collection", "holder raffles", "holder benefits", "play with luki",
                            # Technical / infrastructure
                            "convex lattice", "convex solutions", "remegrid convex", "carefi defi",
                            "electronic life records", "elr data", "ai for elr", "carefi services",
                        ]

                        if any(kw in msg_lower for kw in platform_keywords):
                            # CRITICAL: Smart top_k selection based on query complexity
                            # Keyword-based boost ensures complete information for critical topics
                            if any(keyword in msg_lower for keyword in ['caps', 'cap', 'earn', 'reward', 'token', 'reme', 'luki']):
                                top_k = 10  # Comprehensive coverage for tokenomics queries
                            elif msg_len <= 80:
                                top_k = 5  # Standard coverage for short queries
                            else:
                                top_k = 8  # Enhanced coverage for longer queries

                            logger.info(f"ProjectKB search (stream): msg_len={msg_len}, top_k={top_k}, query='{msg[:50]}...'")
                            proj_docs = kb.search(msg, top_k=top_k)
                        else:
                            logger.info("ProjectKB (stream): skipping search for non-platform query")
                            proj_docs = []
                    else:
                        proj_docs = []
                except Exception as e:
                    logger.warning(f"ProjectKB search failed (stream): {e}")

            persona_id = request.persona_id
            if persona_id is None and request.context:
                persona_id = request.context.get("persona_id") or request.context.get("personality_mode")
            personality_mode = persona_id or "default"
            logger.info(f"🎭 Persona selection (stream): persona_id={persona_id!r}, personality_mode={personality_mode!r}")

            # Extract world day context if provided (for "today's world day" awareness)
            world_day_context = None
            if request.context:
                world_day_context = request.context.get("world_day")
                if world_day_context:
                    logger.info(f"🌍 World day context (stream): {world_day_context.get('name', 'unknown')}")

            context_result = await context_builder.build(
                user_input=request.message,
                user_id=request.user_id,
                conversation_history=request.context.get("conversation_history", []) if request.context else [],
                memory_context=user_memories,
                knowledge_context=proj_docs,
                wallet_context=wallet_context,
                personality_mode=personality_mode,
                world_day_context=world_day_context,  # World day awareness
                client_tag=request.client_tag,  # Widget mode detection
            )
            logger.info(f"🔧 Stream context built (client_tag={request.client_tag})")

            async for token in llm_manager.generate_stream(prompt=context_result["final_prompt"]):
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
