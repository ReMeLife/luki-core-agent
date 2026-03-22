# luki-core-agent

> **This repository is archived.** Active development continues in a private repository with expanded persona management, improved safety detection, and streaming architecture. This public version reflects the ReMeLife integration era and is no longer maintained.

AI agent framework: dialogue orchestration, tool routing, safety filtering, and LLM backend integration.

## Privacy Notice

Proprietary components have been removed from git history for public release:
- Persona definitions and personality frameworks
- Domain-specific knowledge bases and prompt templates
- Internal test scenarios with API keys

The core framework (orchestration, safety chain, tool routing, context building) remains intact. The service requires these proprietary modules (`prompts/`, `_context/`, `knowledge/`) to fully function.

## What It Does

- Orchestrates multi-turn conversations with configurable personas
- Routes tool calls to downstream modules (cognitive, engagement, reporting)
- Filters content through a safety chain (pattern detection, PII awareness)
- Manages LLM backends with fallback support (Together AI, OpenAI)
- Builds context from ELR memories, conversation history, and persona prompts
- Streams responses via SSE

## Stack

- **Framework:** FastAPI
- **LLM:** Together AI (primary), OpenAI (fallback)
- **Prompts:** Jinja2 templates
- **Safety:** Configurable content filtering and pattern detection
- **Deployment:** Docker on Railway

## Structure

```
luki_agent/
├── dev_api.py            # FastAPI server (streaming + non-streaming chat)
├── config.py             # Feature flags, model routing, service URLs
├── llm_backends.py       # LLM provider abstraction (Together AI, OpenAI)
├── safety_chain.py       # Content filtering and safety compliance
├── context_builder.py    # Context assembly (ELR + history + persona)
├── context_optimizer.py  # Context caching and optimization
├── module_client.py      # HTTP client for downstream modules
├── project_kb.py         # Project knowledge base retrieval
├── prompt_registry.py    # Prompt template management
├── schemas.py            # Pydantic response models
├── resilience.py         # Retry + circuit breaker for HTTP calls
├── tools/
│   ├── registry.py       # Tool registration and routing
│   └── web_search.py     # Tavily web search integration
├── memory/
│   ├── memory_client.py        # ChromaDB memory client
│   ├── memory_service_client.py # Memory service HTTP integration
│   ├── retriever.py            # Semantic search retrieval
│   └── session_store.py        # Ephemeral session memory
└── features/
    └── tiers.py          # Subscription tier logic
```

## Prerequisites

This service requires running instances of:

- **luki-memory-service** — ELR memory storage (required for context building)
- **Together AI API key** — primary LLM provider
- Proprietary modules (`prompts/`, `_context/`) are required at runtime but not included in this public release

## Setup

```bash
git clone git@github.com:ReMeLife/luki-core-agent.git
cd luki-core-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```bash
export TOGETHER_API_KEY=your_together_key
export MEMORY_SERVICE_URL=http://localhost:8002
uvicorn luki_agent.dev_api:app --reload --port 9000
```

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat` | Non-streaming chat |
| POST | `/v1/chat/stream` | SSE streaming chat |
| GET | `/health` | Service health |

## Safety

The safety chain runs pattern-based detection on user input. Rules are defined in `safety_chain.py` and enforce content boundaries before sending context to external LLMs.

## License

Apache License 2.0. Copyright 2025 Singularities Ltd / ReMeLife. See [LICENSE](LICENSE).
