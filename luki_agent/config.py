"""
Configuration management for LUKi Core Agent

Handles environment variables, feature flags, model routing, and service URLs.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class LukiAgentSettings(BaseSettings):
    """Configuration settings for LUKi Core Agent"""
    
    # Service Configuration
    service_name: str = Field(default="luki-core-agent", env="SERVICE_NAME")
    service_version: str = Field(default="0.1.0", env="SERVICE_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=9000, env="PORT")
    
    # Model Configuration
    model_backend: str = Field(default="openai", env="LUKI_MODEL_BACKEND")  # openai, llama3_local, llama3_hosted
    model_name: str = Field(default="gpt-3.5-turbo", env="MODEL_NAME")
    model_temperature: float = Field(default=0.7, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    
    # Local Model Configuration
    local_model_path: Optional[str] = Field(default=None, env="LOCAL_MODEL_PATH")
    device: str = Field(default="auto", env="DEVICE")  # auto, cpu, cuda
    
    # Context Configuration
    max_context_tokens: int = Field(default=2048, env="MAX_CONTEXT_TOKENS")
    retrieval_top_k: int = Field(default=6, env="RETRIEVAL_TOP_K")
    conversation_buffer_size: int = Field(default=20, env="CONVERSATION_BUFFER_SIZE")
    
    # Memory Service Configuration
    memory_service_url: str = Field(default="http://localhost:8002", env="MEMORY_SERVICE_URL")
    memory_service_timeout: int = Field(default=30, env="MEMORY_SERVICE_TIMEOUT")
    
    # Authentication
    modules_token: Optional[str] = Field(default=None, env="MODULES_TOKEN")
    internal_api_key: Optional[str] = Field(default=None, env="INTERNAL_API_KEY")
    
    # Safety and Compliance
    enable_safety_filter: bool = Field(default=True, env="ENABLE_SAFETY_FILTER")
    enable_pii_redaction: bool = Field(default=True, env="ENABLE_PII_REDACTION")
    enable_consent_checking: bool = Field(default=True, env="ENABLE_CONSENT_CHECKING")
    
    # Logging and Telemetry
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Redis Configuration (for caching and session storage)
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Feature Flags
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    enable_tool_use: bool = Field(default=True, env="ENABLE_TOOL_USE")
    enable_memory_updates: bool = Field(default=True, env="ENABLE_MEMORY_UPDATES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = LukiAgentSettings()


def get_model_config() -> dict:
    """Get model configuration based on selected backend"""
    base_config = {
        "temperature": settings.model_temperature,
        "max_tokens": settings.max_tokens,
    }
    
    if settings.model_backend == "openai":
        return {
            **base_config,
            "model_name": settings.model_name,
            "api_key": settings.openai_api_key,
            "organization": settings.openai_organization,
        }
    elif settings.model_backend == "llama3_local":
        return {
            **base_config,
            "model_path": settings.local_model_path,
            "device": settings.device,
        }
    elif settings.model_backend == "llama3_hosted":
        return {
            **base_config,
            "model_name": settings.model_name,
            "api_key": settings.openai_api_key,  # Many hosted services use OpenAI-compatible APIs
        }
    else:
        raise ValueError(f"Unknown model backend: {settings.model_backend}")


def get_safety_config() -> dict:
    """Get safety and compliance configuration"""
    return {
        "enable_safety_filter": settings.enable_safety_filter,
        "enable_pii_redaction": settings.enable_pii_redaction,
        "enable_consent_checking": settings.enable_consent_checking,
    }


def get_context_config() -> dict:
    """Get context building configuration"""
    return {
        "max_context_tokens": settings.max_context_tokens,
        "retrieval_top_k": settings.retrieval_top_k,
        "conversation_buffer_size": settings.conversation_buffer_size,
    }
