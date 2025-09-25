"""Configuration management for LUKi Agent."""

import os
from typing import Dict, Any

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
            # Prefer a faster model by default; allow override via env
            "model_name": os.getenv("LUKI_PRIMARY_MODEL", "openai/gpt-oss-20b"),
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
    model_temperature: float = 0.7
    max_tokens: int = 32768  # Use model's full capacity

    # Context & Memory
    conversation_buffer_size: int = 10
    retrieval_top_k: int = 5
    memory_service_timeout: int = 20

    # Module Service URLs
    memory_service_url: str = os.getenv("MEMORY_SERVICE_URL", "https://luki-memory-service-production.up.railway.app/")
    cognitive_service_url: str = os.getenv("LUKI_COGNITIVE_SERVICE_URL", "https://luki-modules-cognitive-production.up.railway.app")
    engagement_service_url: str = os.getenv("LUKI_ENGAGEMENT_SERVICE_URL", "https://luki-modules-engagement-production.up.railway.app")
    security_service_url: str = os.getenv("LUKI_SECURITY_SERVICE_URL", "https://luki-security-privacy-production.up.railway.app")
    reporting_service_url: str = os.getenv("LUKI_REPORTING_SERVICE_URL", "https://luki-modules-reporting-production.up.railway.app")

settings = AppSettings()

def get_model_config(backend_name: str) -> Dict[str, Any]:
    """Retrieve and enrich the configuration for a specific model backend."""
    backend_conf = settings.model_config.get(backend_name)
    if not backend_conf:
        raise ValueError(f"No configuration found for model backend: '{backend_name}'")
    
    backend_conf['temperature'] = settings.model_temperature
    backend_conf['max_tokens'] = settings.max_tokens
    return backend_conf
