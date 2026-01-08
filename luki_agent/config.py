"""Configuration management for LUKi Agent with validation and health checks."""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values"""
    pass


class AppSettings:
    # Service Info
    service_name: str = "luki-core-agent"
    environment: str = os.getenv("ENVIRONMENT", "production")

    # Server
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", 8000))

    # Default model backend to use
    default_backend: str = "together"

    # Model Backend Configurations
    model_config: Dict[str, Any] = {
        "together": {
            "provider": "together",
            # Qwen3-Next-80B-A3B-Instruct: Optimized for instruction following and formatting
            "model_name": os.getenv("LUKI_PRIMARY_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
            "api_key": os.getenv("TOGETHER_API_KEY")
        },
        "local_llama": {
            "provider": "local_llama",
            "model_name": os.getenv("LOCAL_MODEL_PATH", "path/to/local/model"),
            "model_path": os.getenv("LOCAL_MODEL_PATH", "path/to/local/model"),
            "device": "auto"
        }
    }

    # General Model Parameters
    model_temperature: float = 0.7  # Qwen3-Next recommended: 0.7
    max_tokens: int = 32768  # Use model's full capacity
    schema_mode: str = os.getenv("LUKI_SCHEMA_MODE", "minimal").lower()
    structured_timeout: int = int(os.getenv("LUKI_STRUCTURED_TIMEOUT", "20"))
    structured_timeout_long: int = int(os.getenv("LUKI_STRUCTURED_TIMEOUT_LONG", "35"))
    autocontinue_enabled: bool = os.getenv("LUKI_AUTOCONTINUE", "true").lower() == "true"

    # Context & Memory
    conversation_buffer_size: int = 10
    retrieval_top_k: int = 5
    memory_service_timeout: int = 20

    # Module Service URLs
    memory_service_url: str = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8002")
    cognitive_service_url: str = os.getenv("LUKI_COGNITIVE_SERVICE_URL", "http://localhost:8101")
    engagement_service_url: str = os.getenv("LUKI_ENGAGEMENT_SERVICE_URL", "http://localhost:8102")
    security_service_url: str = os.getenv("LUKI_SECURITY_SERVICE_URL", "http://localhost:8103")
    reporting_service_url: str = os.getenv("LUKI_REPORTING_SERVICE_URL", "http://localhost:8104")
    
    # Observability settings
    enable_metrics: bool = os.getenv("LUKI_ENABLE_METRICS", "true").lower() == "true"
    enable_tracing: bool = os.getenv("LUKI_ENABLE_TRACING", "false").lower() == "true"
    log_level: str = os.getenv("LUKI_LOG_LEVEL", "INFO").upper()
    
    # Resilience settings
    enable_retry: bool = os.getenv("LUKI_ENABLE_RETRY", "true").lower() == "true"
    max_retry_attempts: int = int(os.getenv("LUKI_MAX_RETRY_ATTEMPTS", "3"))
    enable_circuit_breaker: bool = os.getenv("LUKI_ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"
    
    def validate(self) -> bool:
        """
        Validate configuration and log warnings for missing optional settings
        
        Returns:
            True if configuration is valid
        
        Raises:
            ConfigurationError: If critical configuration is missing
        """
        errors = []
        warnings = []
        
        # Check critical model configuration
        backend_config = self.model_config.get(self.default_backend)
        if not backend_config:
            errors.append(f"No configuration for default backend: {self.default_backend}")
        elif not backend_config.get("api_key"):
            if self.environment == "production":
                errors.append(f"API key not configured for {self.default_backend} backend")
            else:
                warnings.append(f"API key not configured for {self.default_backend} backend")
        
        # Check service URLs are valid
        service_urls = {
            "memory_service": self.memory_service_url,
            "cognitive_service": self.cognitive_service_url,
            "security_service": self.security_service_url
        }
        
        for service_name, url in service_urls.items():
            if not url or url == "http://localhost:0":
                warnings.append(f"{service_name} URL not properly configured: {url}")
        
        # Check timeout values are reasonable
        if self.structured_timeout < 5:
            warnings.append(f"structured_timeout ({self.structured_timeout}s) is very low, may cause timeouts")
        
        if self.memory_service_timeout < 5:
            warnings.append(f"memory_service_timeout ({self.memory_service_timeout}s) is very low")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # Raise errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(error_msg)
        
        logger.info(f"Configuration validated successfully for environment: {self.environment}")
        return True

settings = AppSettings()

def get_model_config(backend_name: str) -> Dict[str, Any]:
    """Retrieve and enrich the configuration for a specific model backend."""
    backend_conf = settings.model_config.get(backend_name)
    if not backend_conf:
        raise ValueError(f"No configuration found for model backend: '{backend_name}'")
    
    backend_conf['temperature'] = settings.model_temperature
    backend_conf['max_tokens'] = settings.max_tokens
    return backend_conf


def get_schema_mode() -> str:
    return settings.schema_mode


def is_autocontinue_enabled() -> bool:
    return settings.autocontinue_enabled
