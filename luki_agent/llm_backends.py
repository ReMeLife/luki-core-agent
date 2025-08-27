"""
LLM Backend Implementations for LUKi Agent

Supports multiple model backends:
- OpenAI GPT models (GPT-3.5, GPT-4)
- Local LLaMA models via transformers
- Hosted LLaMA models via API
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import json

from .config import settings, get_model_config


@dataclass
class ModelResponse:
    """Response from LLM model"""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the model"""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.api_key = config.get("api_key")
        self.organization = config.get("organization")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Import OpenAI client
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization
            )
        except ImportError:
            raise ImportError("openai package is required for OpenAI backend")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using OpenAI API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.get("max_tokens", 2048),
                temperature=temperature or self.config.get("temperature", 0.7),
                stop=stop_sequences,
                **kwargs
            )
            
            choice = response.choices[0]
            
            return ModelResponse(
                content=choice.message.content or "",
                usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI generation error: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.get("max_tokens", 2048),
                temperature=temperature or self.config.get("temperature", 0.7),
                stop=stop_sequences,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {e}")
    
    async def close(self):
        """Close OpenAI client"""
        if hasattr(self.client, 'close'):
            await self.client.close()


class LocalLLaMABackend(LLMBackend):
    """Local LLaMA model backend using transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path")
        self.device = config.get("device", "auto")
        
        if not self.model_path:
            raise ValueError("Model path is required for local LLaMA backend")
        
        # Import required libraries
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.torch = torch
            self.tokenizer = None
            self.model = None
            self._load_model()
            
        except ImportError:
            raise ImportError("torch and transformers are required for local LLaMA backend")
    
    def _load_model(self):
        """Load the local model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch.float16 if self.torch.cuda.is_available() else self.torch.float32,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using local model"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens,
                temperature,
                stop_sequences,
                kwargs
            )
            return response
            
        except Exception as e:
            raise RuntimeError(f"Local model generation error: {e}")
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop_sequences: Optional[List[str]],
        kwargs: Dict[str, Any]
    ) -> ModelResponse:
        """Synchronous generation for thread pool"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded. Check dependencies and model path.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        if self.torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.get("max_tokens", 2048),
                temperature=temperature or self.config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in response_text:
                    response_text = response_text.split(stop_seq)[0]
                    break
        
        return ModelResponse(
            content=response_text.strip(),
            model=self.model_path,
            finish_reason="stop",
            metadata={"input_tokens": input_length, "output_tokens": len(generated_tokens)}
        )
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response (simplified implementation)"""
        # For now, generate full response and yield in chunks
        # TODO: Implement proper streaming generation
        response = await self.generate(prompt, max_tokens, temperature, stop_sequences, **kwargs)
        
        # Yield response in chunks to simulate streaming
        content = response.content
        chunk_size = 10  # Characters per chunk
        
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    async def close(self):
        """Clean up model resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()


class TogetherAIBackend(LLMBackend):
    """Together AI LLaMA 3.3 70B backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        self.api_key = config.get("api_key")
        self.base_url = "https://api.together.xyz"
        
        if not self.api_key:
            raise ValueError("Together AI API key is required")
        
        # Use httpx for API calls
        try:
            import httpx
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=120.0  # Longer timeout for large models
            )
            print(f"✅ Together AI backend initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("httpx is required for Together AI backend")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Together AI API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
                "temperature": temperature or self.config.get("temperature", 0.7),
                "stop": stop_sequences or [],
                "top_p": kwargs.get("top_p", 0.9),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                **{k: v for k, v in kwargs.items() if k not in ["top_p", "repetition_penalty"]}
            }
            
            response = await self.client.post("/v1/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise RuntimeError("No choices returned from Together AI API")
            
            choice = data["choices"][0]
            
            return ModelResponse(
                content=choice.get("text", "").strip(),
                usage=data.get("usage"),
                model=data.get("model", self.model_name),
                finish_reason=choice.get("finish_reason"),
                metadata={
                    "response_id": data.get("id"),
                    "provider": "together_ai",
                    "model_used": self.model_name
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            raise RuntimeError(f"Together AI API error ({e.response.status_code}): {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Together AI generation error: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using Together AI API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
                "temperature": temperature or self.config.get("temperature", 0.7),
                "stop": stop_sequences or [],
                "stream": True,
                "top_p": kwargs.get("top_p", 0.9),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                **{k: v for k, v in kwargs.items() if k not in ["top_p", "repetition_penalty", "stream"]}
            }
            
            async with self.client.stream("POST", "/v1/completions", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                text = data["choices"][0].get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            raise RuntimeError(f"Together AI streaming error ({e.response.status_code}): {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Together AI streaming error: {e}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class HostedLLaMABackend(LLMBackend):
    """Generic hosted LLaMA backend (e.g., Replicate, other providers)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "meta-llama/Llama-2-70b-chat-hf")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.together.xyz")
        
        if not self.api_key:
            raise ValueError("API key is required for hosted LLaMA backend")
        
        # Use httpx for API calls
        try:
            import httpx
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
        except ImportError:
            raise ImportError("httpx is required for hosted LLaMA backend")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using hosted API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
                "temperature": temperature or self.config.get("temperature", 0.7),
                "stop": stop_sequences or [],
                **kwargs
            }
            
            response = await self.client.post("/v1/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            choice = data["choices"][0]
            
            return ModelResponse(
                content=choice["text"],
                usage=data.get("usage"),
                model=data.get("model"),
                finish_reason=choice.get("finish_reason"),
                metadata={"response_id": data.get("id")}
            )
            
        except Exception as e:
            raise RuntimeError(f"Hosted LLaMA generation error: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using hosted API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
                "temperature": temperature or self.config.get("temperature", 0.7),
                "stop": stop_sequences or [],
                "stream": True,
                **kwargs
            }
            
            async with self.client.stream("POST", "/v1/completions", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                text = data["choices"][0].get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise RuntimeError(f"Hosted LLaMA streaming error: {e}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class LLMManager:
    """Manager for LLM backends"""
    
    def __init__(self):
        self.backend: Optional[LLMBackend] = None
        self.config = get_model_config()
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate backend based on configuration"""
        backend_type = settings.model_backend
        
        if backend_type == "openai":
            self.backend = OpenAIBackend(self.config)
        elif backend_type == "llama3_local":
            self.backend = LocalLLaMABackend(self.config)
        elif backend_type == "llama3_hosted":
            self.backend = HostedLLaMABackend(self.config)
        elif backend_type == "together_ai":
            self.backend = TogetherAIBackend(self.config)
        else:
            raise ValueError(f"Unknown model backend: {backend_type}")
        
        print(f"🧠 LLM Manager initialized with backend: {backend_type}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using configured backend"""
        if not self.backend:
            raise RuntimeError("No backend initialized")
        
        return await self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            **kwargs
        )
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using configured backend"""
        if not self.backend:
            raise RuntimeError("No backend initialized")
        
        # Check if backend has generate_stream method
        if hasattr(self.backend, 'generate_stream'):
            try:
                stream_generator = self.backend.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                    **kwargs
                )
                async for chunk in stream_generator:
                    yield chunk
            except Exception as e:
                # If streaming fails, fallback to regular generation
                response = await self.backend.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                    **kwargs
                )
                yield response.content
        else:
            # Fallback to non-streaming generation
            response = await self.backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                **kwargs
            )
            yield response.content
    
    async def close(self):
        """Close backend resources"""
        if self.backend:
            await self.backend.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass
