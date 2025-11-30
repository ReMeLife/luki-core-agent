# luki-core-agent  
*Open-source AI agent framework: dialogue management, tool orchestration & safety layers*

---

## Privacy & Proprietary Content Notice

This repository contains the core open-source architecture for an AI agent system. The following proprietary components have been completely removed from git history for public release:

### Removed Files:
- **Proprietary Knowledge**: `knowledge_glossary.py` - Business terminology and domain definitions
- **Conversation Logic**: `avatar_personality.py`, `personality_templates.py` - Proprietary personality frameworks
- **Test Scenarios**: All root-level test files containing API keys and proprietary integration scenarios
- **Configuration**: `.env` files and environment templates with sensitive API configurations
- **Database Files**: `engagement.db` and other local data stores
- **Documentation**: Internal development notes and proprietary specifications

### What Remains:
- Core agent orchestration framework (`agent_core.py`, `context_builder.py`)
- LLM backend integration (`llm_backends.py`)
- Memory service client interfaces
- Tool registry and routing framework
- Safety and content filtering chains
- Basic conversation chain structure

The framework architecture is fully functional for building AI agents, but requires implementing your own personality definitions, knowledge bases, and conversation templates.

---

## 1. Overview  
`luki-core-agent` provides the **agentic logic framework** for AI assistants:  
- System & persona prompt management, dialogue state machines, tool-routing policies  
- Safety, compliance and content filtering  
- Evaluation harnesses for prompt/chain quality  
- Integration framework for downstream modules and platform APIs

This framework can be adapted for various AI assistant use cases while maintaining safety and compliance standards.

---

## 2. Core Responsibilities  
- **Conversation Orchestration:** Manage multi-turn context, memory retrieval, and goal decomposition  
- **Tool/Skill Routing:** Decide when to call external tools and services  
- **Safety & Guardrails:** Content filtering, PII redaction, consent enforcement, safety compliance  
- **Prompt Management:** Versioned system/instruction prompts, response formats, evaluation prompts  
- **LLM Backend Integration:** Wrapper for various language models with fallback support  
- **Session Memory Interface:** Hooks into memory services for context retrieval and caching  
- **Evaluation & Telemetry:** Auto-evaluation scripts, metrics logging, trace IDs, performance monitoring

---

## 3. Tech Stack  
- **Framework:** Custom agent orchestration with FastAPI  
- **Models:** ChatGPT OSS 20B/120B (hybrid) via Together AI, with fallback support  
- **Prompt Templating:** Jinja2 templates with structured JSON response schemas  
- **Safety Filters:** Configurable content filters and PII detection  
- **Tracing & Eval:** OpenTelemetry integration, custom evaluation scripts  
- **Server:** FastAPI for development and production endpoints

---

## 4. Repository Structure  
~~~text
luki-core-agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-full.txt          # complete dependencies
├── requirements-minimal.txt       # minimal dependencies
├── requirements-railway.txt       # railway deployment dependencies
├── .env                          # environment variables 
├── .railwayignore               # railway deployment exclusions
├── .dockerignore                # docker build exclusions
├── Dockerfile                   # container build configuration
├── railway.toml                 # railway deployment configuration
├── Procfile                     # process definitions
├── standalone_api.py            # standalone API server
├── full_startup.py              # full system startup script
├── engagement.db                # local engagement database 
├── prompts/                     # Jinja2 prompt templates 
│   ├── persona_luki_v1.j2
│   ├── safety_rules_v1.j2
│   ├── system_core_min_v1.j2
│   ├── system_core_text_v1.j2
│   └── system_core_v1.j2
├── luki_agent/
│   ├── __init__.py
│   ├── config.py                # feature flags, model routing, rate limits
│   ├── main.py                  # main entry point
│   ├── dev_api.py               # development FastAPI server
│   ├── minimal_api.py           # minimal API interface
│   ├── llm_backends.py          # LLM integration (Together AI, local)
│   ├── safety_chain.py          # content filtering and safety compliance
│   ├── context_builder.py       # context assembly and management
│   ├── module_client.py         # client for LUKi modules
│   ├── project_kb.py            # project knowledge base integration
│   ├── prompt_registry.py       # prompt template management
│   ├── prompts_system.py        # system prompt definitions
│   ├── schemas.py               # Pydantic response schemas
│   ├── knowledge_glossary.py    # domain knowledge and terminology
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── conversation_chain.py # main conversation orchestrator
│   │   ├── user_context_detector.py # user context analysis
│   │   └── [additional chain files]
│   ├── tools/
│   │   ├── __init__.py
│   │   └── registry.py          # tool registration and routing
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_client.py     # memory service integration
│   │   ├── memory_service_client.py # enhanced memory client
│   │   ├── retriever.py         # vector and KV retrieval
│   │   └── session_store.py     # ephemeral session memory
│   └── knowledge/
│       └── project_glossary.py  # project-specific knowledge base
├── scripts/                     # deployment and utility scripts
└── tests/                       # comprehensive test suites
~~~

**Note:** This structure shows the core framework that remains after sanitization. Proprietary components including personality definitions, business terminology, conversation templates, comprehensive test scenarios, and prompt engineering files have been removed from the public release.

---

## 5. Quick Start (Development)

~~~powershell
git clone git@github.com:REMELife/luki-core-agent.git
cd luki-core-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

Set environment variables (create `.env` file or set in PowerShell):

<<<<<<< HEAD
~~~bash
<<<<<<< HEAD
export LUKI_MODEL_BACKEND=llama33_together   # or openai
export OPENAI_API_KEY=sk-...             # only if using fallback
=======
export LUKI_MODEL_BACKEND=together_ai     # or openai
export TOGETHER_API_KEY=your_together_key # for LLaMA 3.3 access
export OPENAI_API_KEY=your_openai_key     # only if using fallback
>>>>>>> develop
export MEMORY_API_URL=http://localhost:8002
export MODULES_TOKEN=dev123               # auth to call public modules
=======
~~~powershell
$env:LUKI_MODEL_BACKEND="together"         # Together AI backend (default)
$env:TOGETHER_API_KEY="your_together_key"   # required for Together AI models
$env:LUKI_PRIMARY_MODEL="openai/gpt-oss-20b" # ChatGPT OSS 20B model
$env:MEMORY_SERVICE_URL="http://localhost:8002" 
$env:MODULES_TOKEN="dev123"                # auth to call public modules
>>>>>>> develop
~~~

Run dev server (optional):

~~~powershell
# Development server
uvicorn luki_agent.dev_api:app --reload --port 9000

# Or use the standalone API
python standalone_api.py

# Or full startup script
python full_startup.py
~~~

Chat locally:

~~~python
from luki_agent.dev_api import app
import httpx

# Chat with the agent via API
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:9000/chat",
        json={
            "user_id": "user_123",
            "message": "Hello, how can you help me?"
        }
    )
    print(response.json())
~~~

**Note:** The conversation chain and personality system require implementing your own business logic using the provided stub files as templates.

**Note:** Evaluation datasets and comprehensive test scenarios have been removed as they contained proprietary data. You can create your own evaluation datasets following the framework patterns in the remaining code.

---

## 6. Prompt & Tool Versioning  
- Keep **all prompt files in `prompts/`**, no hard-coded strings.  
- Each change => bump prompt version (e.g., `system_v3.j2`).  
- Prompts use Jinja2 templating with structured JSON response schemas.  
- Tools are registered in `tools/registry.py` with versioned contracts.  
- Context files in `_context/` provide domain knowledge and personality framework.

---

## 7. Safety, Compliance & Consent  
- Redact PII before sending context to external LLMs.  
- Respect user consent scopes (ELR segments) in `tool_router.py`.  
- Audit logs (trace IDs) stored via telemetry hooks.  
- Add new safety rules in `safety_chain.py`; accompany with tests.

---

## 8. Testing & CI  
- **Unit tests:** Basic framework functionality testing  
- **Integration tests:** Limited to core framework components  
- **Custom Implementation:** You'll need to create your own test scenarios for proprietary logic  

Run available tests:  
~~~powershell
pytest -q
~~~

**Note:** Comprehensive integration tests containing proprietary scenarios have been removed. The remaining test structure provides a foundation for implementing your own test suites.

---

## 9. Roadmap  
- Multi-agent subteams (planner, critic, executor)  
- Enhanced structured output with improved JSON schemas  
- RLHF-lite loop using user thumbs-up/down  
- Realtime streaming support (server-sent events)  
- Fine-grained AB tests on prompt variants  
- Integration with ChatGPT OSS 120B for complex reasoning tasks

---

## 10. Contributing  
- Create a feature branch: `feature/<ticket>-<short-desc>`  
- Never commit raw user data or secrets.  
- Any prompt changes must include an eval run result.  
- Code review by at least 1 core maintainer.  
See `CONTRIBUTING.md` for full rules.

---

## 11. License  
**Apache License 2.0**  
Copyright 2025 Singularities Ltd / ReMeLife.  
See `LICENSE` for full terms.

---

**This is the brain. Adapt it.**
