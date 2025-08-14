"""Configuration management for LUKi Agent

Handles environment variables, model settings, service URLs, and feature flags.
Uses Pydantic for validation and type safety.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class LukiAgentSettings(BaseSettings):
    """Configuration settings for LUKi Agent"""
    
    # Service Configuration
    service_name: str = "luki-core-agent"
    service_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 9000
    
    # Model Configuration
    model_backend: str = "llama3_hosted"  # openai, llama3_local, llama3_hosted
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    model_temperature: float = 0.7
    max_tokens: int = 2048
    
    # Hosted LLaMA Configuration (Together AI)
    hosted_api_key: Optional[str] = None  # Set via LUKI_HOSTED_API_KEY environment variable
    hosted_base_url: str = "https://api.together.xyz"
    
    # OpenAI Configuration (fallback only)
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    
    # Local Model Configuration
    local_model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    
    # Context Configuration
    max_context_tokens: int = 2048
    retrieval_top_k: int = 6
    conversation_buffer_size: int = 20
    
    # Memory Service Configuration
    memory_service_url: str = "http://localhost:8002"
    memory_service_timeout: int = 30
    
    # Authentication
    modules_token: Optional[str] = None
    internal_api_key: Optional[str] = None
    
    # Safety and Compliance
    enable_safety_filter: bool = True
    enable_pii_redaction: bool = True
    enable_consent_checking: bool = True
    
    # Logging and Telemetry
    log_level: str = "INFO"
    enable_tracing: bool = True
    jaeger_endpoint: Optional[str] = None
    
    # Redis Configuration (for caching and session storage)
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Feature Flags
    enable_streaming: bool = True
    enable_tool_use: bool = True
    enable_memory_updates: bool = True
    
    class Config:
        case_sensitive = False
        env_prefix = "LUKI_"


# Global settings instance
settings = LukiAgentSettings()


def get_context_config() -> dict:
    """Get context-specific configuration"""
    return {
        "max_context_tokens": settings.max_context_tokens,
        "retrieval_top_k": settings.retrieval_top_k,
        "conversation_buffer_size": settings.conversation_buffer_size,
        "enable_memory_updates": settings.enable_memory_updates,
    }


def get_model_config() -> dict:
    """Get model-specific configuration"""
    return {
        "backend": settings.model_backend,
        "model_name": settings.model_name,
        "temperature": settings.model_temperature,
        "max_tokens": settings.max_tokens,
        "device": settings.device,
        # Hosted LLaMA configuration
        "api_key": settings.hosted_api_key,
        "base_url": settings.hosted_base_url,
        # OpenAI configuration (fallback)
        "openai_api_key": settings.openai_api_key,
        "openai_organization": settings.openai_organization,
    }


def get_safety_config() -> dict:
    """Get safety and compliance configuration"""
    return {
        "enable_safety_filter": settings.enable_safety_filter,
        "enable_pii_redaction": settings.enable_pii_redaction,
        "enable_consent_checking": settings.enable_consent_checking,
    }
