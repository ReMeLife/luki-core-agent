# luki-core-agent  
*Primary LUKi brain: dialogue manager, tool orchestration, prompt packs & safety layers*  
**PRIVATE / PROPRIETARY – Do NOT expose or mirror externally**

---

## 1. Overview  
`luki-core-agent` contains the **agentic logic** that powers LUKi:  
- System & persona prompts, dialogue state machines, tool-routing policies  
- Safety, compliance and redaction filters  
- Evaluation harnesses for prompt/chain quality  
- Integration glue to call downstream modules (cognitive, engagement, reporting) and platform APIs

This repo is **closed-source IP**. Anything here (prompts, heuristics, fine‑tune adapters) must not leave our infra.

---

## 2. Core Responsibilities  
- **Conversation Orchestration:** Manage multi-turn context, memory retrieval, and goal decomposition.  
- **Tool/Skill Routing:** Decide when to call `recommend_activity`, `generate_wellbeing_report`, ELR search, etc.  
- **Safety & Guardrails:** Redact PII, enforce consent scopes, block unsafe content, comply with care guidelines.  
- **Prompt Pack Management:** Versioned system/instruction prompts, response formats, eval prompts.  
- **LLM Backends:** Wrapper to selected base model(s) (e.g., LLaMA‑3 + OpenAI fallback).  
- **Session Memory Interface:** Hooks into memory-service (vector + KV) for ELR retrieval and summary caching.  
- **Eval & Telemetry:** Auto-eval scripts (BLEU/ROUGE for reports, human ratings logs, trace IDs, latency metrics).

---

## 3. Tech Stack  
- **Framework:** LangChain (core routing), plus custom policy layer  
- **Models:** LLaMA‑3 (local/hosted), OpenAI GPT-* fallback (via feature flag)  
- **Prompt Templating:** Jinja2 / LangChain PromptTemplates  
- **Safety Filters:** regex + PII detectors, OpenAI moderation or local classifier fallback  
- **Tracing & Eval:** LangSmith / OpenTelemetry, custom eval scripts  
- **Server:** FastAPI for internal dev endpoints (optional); production served via `luki-api-gateway` repo

---

## 4. Repository Structure  
~~~text
luki_core_agent/
├── README.md
├── pyproject.toml
├── luki_agent/
│   ├── __init__.py
│   ├── config.py                  # feature flags, model routing, rate limits
│   ├── prompts/
│   │   ├── system/                # system persona, safety system prompts
│   │   ├── tools/                 # tool-call specific templates
│   │   ├── eval/                  # eval and grading prompts
│   │   └── fragments/             # reusable blocks
│   ├── chains/
│   │   ├── conversation_chain.py  # main orchestrator
│   │   ├── tool_router.py         # policy-based routing
│   │   ├── safety_chain.py        # redaction, refusal handling
│   │   └── summariser_chain.py    # memory/ELR summarisation
│   ├── tools/
│   │   ├── registry.py            # register public module tools
│   │   └── wrappers/              # http clients for other repos/APIs
│   ├── memory/
│   │   ├── retriever.py           # vector+kv retriever
│   │   └── session_store.py       # ephemeral session memory
│   ├── eval/
│   │   ├── eval_runner.py         # batch eval jobs
│   │   ├── metrics.py
│   │   └── datasets/              # synthetic eval sets
│   └── telemetry/
│       ├── tracing.py
│       └── logging.py
├── scripts/
│   ├── run_dev_server.sh
│   ├── batch_eval.sh
│   └── sync_prompts.py
└── tests/
    ├── unit/
    └── integration/
~~~

---

## 5. Quick Start (Internal Dev)

~~~bash
git clone git@github.com:REMELife/luki-core-agent.git
cd luki-core-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Optional local model
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
~~~

Set env vars:

~~~bash
export LUKI_MODEL_BACKEND=llama3_local   # or openai
export OPENAI_API_KEY=sk-...             # only if using fallback
export MEMORY_API_URL=http://localhost:8002
export MODULES_TOKEN=dev123              # auth to call public modules
~~~

Run dev server (optional):

~~~bash
uvicorn luki_agent.dev_api:app --reload --port 9000
~~~

Chat locally:

~~~python
from luki_agent.chains.conversation_chain import LukiConversation

agent = LukiConversation()
resp = agent.chat(user_id="user_123", message="Any ideas for a calming activity?")
print(resp.text)
print(resp.tool_calls)  # if any
~~~

Batch eval:

~~~bash
python -m luki_agent.eval.eval_runner --dataset data/eval/basic.jsonl --out results/basic_eval.jsonl
python -m luki_agent.eval.metrics results/basic_eval.jsonl
~~~

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
- **Unit tests:** prompt formatting, router policies, safety regex.  
- **Integration tests:** fake memory + mock tool servers.  
- **Eval harness:** use synthetic dialogues to regression-test style & safety.  
Run:  
~~~bash
pytest -q
python -m luki_agent.eval.eval_runner ...
~~~

---

## 9. Roadmap  
- Multi-agent subteams (planner, critic, executor)  
- Structured output enforcement (json schema tool calls)  
- RLHF-lite loop using user thumbs-up/down  
- Realtime streaming support (server-sent events)  
- Fine-grained AB tests on prompt variants

---

## 10. Contributing (Internal Only)  
- Create a feature branch: `feature/<ticket>-<short-desc>`  
- Never commit raw user data or secrets.  
- Any prompt changes must include an eval run result.  
- Code review by at least 1 core maintainer.  
See `CONTRIBUTING.md` for full rules.

---

## 11. License  
**Proprietary – All Rights Reserved**  
Copyright © 2025 Singularities Ltd / ReMeLife.  
Unauthorized copying, modification, distribution, or disclosure is strictly prohibited.

---

**This is the brain. Guard it.**
