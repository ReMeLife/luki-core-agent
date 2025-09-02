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
- **Framework:** LangChain (core routing), plus custom policy layer  
- **Models:** Configurable LLM backends with fallback support  
- **Prompt Templating:** Jinja2 / LangChain PromptTemplates  
- **Safety Filters:** Configurable content filters and PII detection  
- **Tracing & Eval:** OpenTelemetry integration, custom evaluation scripts  
- **Server:** FastAPI for development endpoints

---

## 4. Repository Structure  
~~~text
luki-core-agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── env.example                    # environment variable template
├── luki_agent/
│   ├── __init__.py
│   ├── config.py                  # feature flags, model routing, rate limits
│   │
│   ├── # Core Agent Framework
│   ├── agent_core.py              # main agent orchestration logic
│   ├── llm_backends.py            # LLM integration and fallback support
│   ├── dev_api.py                 # development FastAPI server
│   ├── safety_chain.py            # content filtering and safety compliance
│   │
│   ├── # Context Management
│   ├── context_builder.py         # context assembly and management
│   ├── enhanced_context_builder.py # advanced context processing
│   ├── smart_context_builder.py   # optimized context handling
│   ├── fast_query_handler.py      # quick response processing
│   │
│   ├── # Conversation Logic
│   ├── chains/
│   │   ├── conversation_chain.py  # main conversation orchestrator
│   │   └── user_context_detector.py # user context analysis
│   │
│   ├── # Tool Integration
│   ├── tools/
│   │   ├── __init__.py
│   │   └── registry.py            # tool registration and routing
│   │
│   ├── # Memory Services
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_client.py       # memory service integration
│   │   ├── memory_service_client.py # enhanced memory client
│   │   ├── retriever.py           # vector and KV retrieval
│   │   └── session_store.py       # ephemeral session memory
│   │
│   └── # Knowledge Base
│       └── knowledge/
│           └── project_glossary.py # project-specific knowledge base
├── scripts/
│   └── run_dev_server.py          # development server launcher
└── tests/
    └── # Basic test framework structure
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

~~~powershell
$env:LUKI_MODEL_BACKEND="configurable"     # or openai
$env:OPENAI_API_KEY="your_openai_key"      # only if using fallback
$env:MEMORY_API_URL="http://localhost:8002"
$env:MODULES_TOKEN="dev123"                # auth to call public modules
~~~

Run dev server (optional):

~~~powershell
uvicorn luki_agent.dev_api:app --reload --port 9000
~~~

Chat locally:

~~~python
from luki_agent.agent_core import LukiAgent

# Initialize with your own configuration
agent = LukiAgent()
response = await agent.process_message(
    user_id="user_123", 
    message="Hello, how can you help me?"
)
print(response)
~~~

**Note:** The conversation chain and personality system require implementing your own business logic using the provided stub files as templates.

**Note:** Evaluation datasets and comprehensive test scenarios have been removed as they contained proprietary data. You can create your own evaluation datasets following the framework patterns in the remaining code.

---

## 6. Prompt & Tool Versioning  
- Keep **all prompt files in `prompts/`**, no hard-coded strings.  
- Each change => bump prompt version (e.g., `system_v3.j2`).  
- Use `sync_prompts.py` to push/pull from secure S3/Git-crypt store.  
- Tools are registered in `tools/registry.py` with versioned contracts.

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
- Structured output enforcement (json schema tool calls)  
- RLHF-lite loop using user thumbs-up/down  
- Realtime streaming support (server-sent events)  
- Fine-grained AB tests on prompt variants

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
