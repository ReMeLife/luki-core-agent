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

try:
    from .config import settings
    from .llm_backends import LLMManager
    from .context_builder import ContextBuilder
    from .memory.memory_service_client import MemoryServiceClient
    from .project_kb import ProjectKB
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

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

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
        
        # Prepare user memories (ELR) if gateway provided them (only for authenticated users)
        user_memories = request.context.get("memory_context", []) if request.context else []

        # ProjectKB search (independent of ELR, available to all users)
        proj_docs: List[Dict[str, Any]] = []
        kb = getattr(app.state, "project_kb", None)
        if kb is not None:
            try:
                msg_len = len(request.message.strip())
                if msg_len > 12:
                    top_k = 3 if msg_len <= 80 else 5
                    logger.info(f"ProjectKB search: msg_len={msg_len}, top_k={top_k}")
                    proj_docs = kb.search(request.message, top_k=top_k)
                else:
                    proj_docs = []
            except Exception as e:
                logger.warning(f"ProjectKB search failed: {e}")

        logger.info(
            f"🔍 Step 3: Building context with {len(proj_docs)} project docs and {len(user_memories)} user memories..."
        )
        
        # Log actual memory content for debugging
        if user_memories:
            logger.info(f"📦 User memories content:")
            for idx, mem in enumerate(user_memories[:3]):  # Log first 3
                logger.info(f"  Memory {idx}: {mem.get('content', '')[:100]}...")

        # CRITICAL FIX: Keep project docs and user memories SEPARATE
        # Project docs are for knowledge, user memories are personal data
        # We'll pass them separately to the context builder
        
        context_result = await context_builder.build(
            user_input=request.message,
            user_id=request.user_id,
            conversation_history=request.context.get("conversation_history", []) if request.context else [],
            memory_context=user_memories,  # ONLY user memories here
            knowledge_context=proj_docs     # Project knowledge separate
        )
        logger.info(f"✅ Step 3: Context built successfully")
        
        # Log what's in the context result
        if context_result.get("final_prompt", {}).get("retrieval_context"):
            logger.info(f"📦 Retrieval context passed to LLM: {context_result['final_prompt']['retrieval_context'][:200]}...")

        logger.info(f"🔍 Step 4: Generating LLM response...")
        response = await llm_manager.generate(prompt=context_result["final_prompt"])
        logger.info(f"✅ Step 4: LLM response generated successfully")
        # Always include a session_id for gateway compatibility
        session_id = request.session_id or "new-session"
        payload = {"response": response.content, "metadata": response.metadata, "session_id": session_id}
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

            # Project KB search
            proj_docs: List[Dict[str, Any]] = []
            kb = getattr(app.state, "project_kb", None)
            if kb is not None:
                try:
                    msg_len = len(request.message.strip())
                    if msg_len > 12:
                        top_k = 3 if msg_len <= 80 else 5
                        logger.info(f"ProjectKB search (stream): msg_len={msg_len}, top_k={top_k}")
                        proj_docs = kb.search(request.message, top_k=top_k)
                    else:
                        proj_docs = []
                except Exception as e:
                    logger.warning(f"ProjectKB search failed (stream): {e}")

            context_result = await context_builder.build(
                user_input=request.message,
                user_id=request.user_id,
                conversation_history=request.context.get("conversation_history", []) if request.context else [],
                memory_context=(proj_docs or []) + (user_memories or [])
            )

            async for token in llm_manager.generate_stream(prompt=context_result["final_prompt"]):
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
