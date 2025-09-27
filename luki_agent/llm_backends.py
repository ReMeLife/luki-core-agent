"""LLM Backend Implementations for the LUKi Agent."""

import asyncio
import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .config import settings, get_model_config
from .schemas import LUKiResponse, LUKiMinimalResponse
from .prompt_registry import prompt_registry

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
        print(f"✅ Together AI backend initialized: {self.model_name} with max_tokens={self.max_tokens}")

    async def generate(self, prompt: Dict[str, Any], **kwargs) -> ModelResponse:
        system_content = f"{prompt['system_prompt']}\n{prompt['retrieval_context']}\n{prompt['conversation_history']}"
        
        # Debug: Log system content length
        print(f"🔍 System content length: {len(system_content)} characters (~{len(system_content)//4} tokens)")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt['user_input']}
        ]
        # Choose schema mode (minimal for speed by default)
        schema_mode = os.getenv("LUKI_SCHEMA_MODE", "minimal").lower()
        use_minimal = schema_mode == "minimal"

        # Dynamic max_tokens based on input length and schema mode
        user_len = len(prompt['user_input'].strip())
        if use_minimal:
            # More generous caps to reduce truncation while keeping speed reasonable
            if user_len <= 120:
                dyn_max = 1024
            elif user_len <= 300:
                dyn_max = 2048
            else:
                dyn_max = 3072
            response_model = LUKiMinimalResponse
        else:
            if user_len <= 120:
                dyn_max = 2048
            elif user_len <= 300:
                dyn_max = 4096
            else:
                dyn_max = 8192
            response_model = LUKiResponse

        # Build parameters with our settings taking absolute precedence
        final_params = {
            "model": self.model_name,
            "messages": messages,
            "response_model": response_model,
            "temperature": self.temperature,
            "max_tokens": dyn_max,
        }
        # Add any additional kwargs that don't conflict with our core settings
        for key, value in kwargs.items():
            if key not in final_params:
                final_params[key] = value

        # Debug logging to track token settings
        print(f"🔍 API call parameters: max_tokens={final_params.get('max_tokens')}, model={final_params.get('model')}")

        # Fast-path for trivial inputs: skip heavy structured call
        # Treat very short inputs as trivial regardless of context content
        user_len = len(prompt['user_input'].strip())
        is_trivial = user_len <= 6
        if is_trivial:
            print("⚡ Trivial input detected, using fast fallback path")
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
            # Minimal schema does not have 'thought'
            content = getattr(luki_response, "final_response", "")
            metadata: Dict[str, Any] = {}
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
                    # Heuristic: high utilization vs dyn_max and no terminal punctuation
                    def looks_truncated(txt: str) -> bool:
                        trimmed = (txt or "").rstrip()
                        if not trimmed:
                            return False
                        terminal = (".", "!", "?", "\u201d", '"', "'", ")", "]")
                        if trimmed.endswith(terminal):
                            return False
                        # Approx token estimate
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
                        # Token budget for continuation
                        cont_tokens = 768 if user_len <= 200 else 1024
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

            return ModelResponse(content=content, model=self.model_name, metadata=metadata)
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
                        "If unsure, say you don't know."
                    )
                try:
                    persona_core = prompt_registry.load_prompt("persona_luki", "v1")
                except Exception:
                    persona_core = ""
                rc = prompt.get('retrieval_context') or ""
                if len(rc) > 600:
                    rc = rc[:600] + "..."
                conv = prompt.get('conversation_history') or ""
                if len(conv) > 400:
                    conv = conv[:400] + "..."
                fallback_system = (
                    f"{minimal_core}\n\n"
                    f"{persona_core}\n\n"
                    f"Relevant Context:\n{rc}\n\n"
                    f"Recent Conversation:\n{conv}\n\n"
                    "Instructions: Reply succinctly in natural language in LUKi's voice with gentle action cues like *smiles* or *chuckles* when appropriate. "
                    "Do not include JSON or internal thoughts."
                )
                fallback_messages = [
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": prompt['user_input']}
                ]
                # Increase fallback tokens for non-trivial inputs to reduce truncation
                if user_len <= 60:
                    fb_tokens = min(self.fast_fallback_tokens, 768)
                elif user_len <= 200:
                    fb_tokens = min(max(self.fast_fallback_tokens, 768), 1536)
                else:
                    fb_tokens = min(max(self.fast_fallback_tokens, 1024), 2048)
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
        system_content = f"{prompt['system_prompt']}\n{prompt['retrieval_context']}\n{prompt['conversation_history']}"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt['user_input']}
        ]
        # Build parameters with our settings taking absolute precedence
        schema_mode = os.getenv("LUKI_SCHEMA_MODE", "minimal").lower()
        use_minimal = schema_mode == "minimal"
        user_len = len(prompt['user_input'].strip())
        if use_minimal:
            if user_len <= 80:
                dyn_max = 512
            elif user_len <= 200:
                dyn_max = 1024
            else:
                dyn_max = 1536
            stream_model = LUKiMinimalResponse
        else:
            if user_len <= 120:
                dyn_max = 2048
            elif user_len <= 300:
                dyn_max = 4096
            else:
                dyn_max = 8192
            stream_model = LUKiResponse

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
                    yield partial.final_response
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
                        "You are LUKi, the ReMeLife assistant. Follow platform facts accurately."
                    )
                try:
                    persona_core = prompt_registry.load_prompt("persona_luki", "v1")
                except Exception:
                    persona_core = ""
                rc = prompt.get('retrieval_context') or ""
                if len(rc) > 600:
                    rc = rc[:600] + "..."
                conv = prompt.get('conversation_history') or ""
                if len(conv) > 400:
                    conv = conv[:400] + "..."
                fb_system = (
                    f"{minimal_core}\n\n{persona_core}\n\n"
                    f"Relevant Context:\n{rc}\n\n"
                    f"Recent Conversation:\n{conv}\n\n"
                    "Instructions: Reply succinctly in LUKi's voice with gentle action cues like *smiles* or *chuckles* when appropriate. "
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
                # Yield content in a couple of chunks to emulate streaming
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
