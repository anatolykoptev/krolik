<p align="center">
  <img src="https://raw.githubusercontent.com/anatolykoptev/krolik/main/.assets/logo.png" alt="Krolik" width="120" />
</p>

<h1 align="center">Krolik 🐰</h1>

<p align="center">
  <strong>Proactive AI agent with long-term memory and dynamic LLM routing</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#cli-reference">CLI</a> •
  <a href="#development">Development</a>
</p>

---

## What is Krolik?

Krolik is an autonomous AI agent that runs 24/7, connects to your chat apps (Telegram, WhatsApp, Feishu), remembers everything through vector-based long-term memory ([memU](https://github.com/nevamind-ai/memu)), and proactively reaches out based on schedules, reminders, and context.

Unlike simple chatbot wrappers, Krolik:
- **Routes tasks intelligently** across dozens of LLM models using cost-aware 5-tier scoring
- **Remembers context** across conversations with semantic memory retrieval
- **Acts proactively** — cron jobs, heartbeats, and memory-driven suggestions
- **Runs tools** — filesystem, shell, web search, MCP servers, subagents

## Quick Start

```bash
# Clone and install
git clone https://github.com/anatolykoptev/krolik.git
cd krolik
pip install -e .

# Configure API key (at least one provider required)
mkdir -p ~/.krolik
cat > ~/.krolik/.env << 'EOF'
KROLIK_PROVIDERS__GEMINI__API_KEY=your-gemini-key
EOF

# Interactive CLI mode
krolik agent -m "Hello!"

# Full gateway (Telegram, cron, heartbeat)
krolik gateway
```

## Features

### 🧠 Long-Term Memory (memU)

Semantic vector memory with intent-aware retrieval. Krolik decides *before* each query whether memory retrieval would help, rewrites queries for better recall, and filters low-relevance results.

```
User: "What was that restaurant I liked?"
Krolik: [recalls from memory] "You mentioned loving Sushi Nakazawa last month."
```

- **Categories**: facts, preferences, tasks, conversations
- **Proactive suggestions**: daily digests, deadline reminders, learning goal nudges
- **Dual backend**: memU service (vector search) with file-based fallback

### 🔀 Dynamic LLM Router

Automatic model selection from 200+ models via OpenRouter API discovery:

| Tier | Cost | Use Case | Example Models |
|------|------|----------|----------------|
| **Free** | $0 | Casual chat, simple Q&A | Hermes 3 405B, Llama 3 |
| **Cheap** | <$0.05/1M | Everyday tasks | Gemini 2.0 Flash, DeepSeek |
| **Standard** | <$0.15/1M | Code, analysis | Gemini 2.5 Flash, Claude Sonnet |
| **Premium** | ≥$0.15/1M | Complex reasoning | Claude Opus, GPT-4o |
| **Research** | Any | Deep research | Perplexity Sonar Deep Research |

- **Bilingual scoring** — EN + RU keyword detection for task classification
- **Composite scoring** — priority (40%) + success rate (30%) + speed + latency
- **Persistent outcomes** — tracks which models succeed at which tasks
- **Cascade fallback** — automatic retry with next-best model on failure

### 🔌 Multi-Provider Gateway

Async HTTP gateway with streaming, retries, and fallback chains:

- **Providers**: Gemini, OpenRouter, Anthropic, CLIProxyAPI (free local OAuth)
- **Features**: exponential backoff, rate limit handling, per-provider stats
- **Streaming**: SSE-based token streaming for real-time responses

### 📱 Chat Channels

| Channel | Transport | Features |
|---------|-----------|----------|
| **Telegram** | Long polling | Text, photos, voice transcription, documents |
| **WhatsApp** | WebSocket bridge | Text, media |
| **Feishu/Lark** | HTTP webhook | Text, cards |
| **CLI** | stdin/stdout | Interactive & single-message modes |

### ⏰ Proactive Scheduling

- **Cron jobs** — schedule agent tasks with cron expressions or intervals
- **Heartbeat** — periodic wake-up reads `HEARTBEAT.md` for pending tasks
- **Memory digest** — daily summary of preferences, goals, deadlines
- **Reminder triggers** — proactive nudges based on stored memories

### 🛠️ Tool System

21 built-in tools registered on startup:

| Category | Tools |
|----------|-------|
| **Memory** | remember, recall, search_memory |
| **Workflow** | create_workflow, list_workflows, run_workflow |
| **LLM** | llm_call, coding_agent, list_models, discover_models |
| **System** | cli_proxy, agent_connect |
| **MCP** | Dynamic tools from connected MCP servers |
| **Skills** | github, weather, tmux, summarize (extensible via SKILL.md) |

### 🧩 MCP Integration

Connect to any [Model Context Protocol](https://modelcontextprotocol.io/) server:

```env
KROLIK_MCP__SERVERS__MEMOS__URL=http://localhost:8001/sse
KROLIK_MCP__SERVERS__GDRIVE__URL=http://localhost:3002/sse
KROLIK_MCP__AUTO_CONNECT=memos,gdrive
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         krolik gateway                           │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│   Telegram   │   WhatsApp   │    Feishu    │        CLI          │
│   Channel    │   Channel    │   Channel    │      Channel        │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬───────────┘
       │              │              │                 │
       ▼              ▼              ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Message Bus                               │
│                   (async inbound/outbound)                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Loop                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │   Context    │  │    Tools     │  │     Session Manager     │ │
│  │   Builder    │  │  (21 tools)  │  │   (JSONL persistence)   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────────────────┘ │
│         │                │                                       │
│  ┌──────▼──────┐  ┌──────▼───────┐  ┌─────────────────────────┐ │
│  │   Skills    │  │  Subagent    │  │     MCP Manager         │ │
│  │   Loader    │  │  Spawner     │  │  (dynamic tool import)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│   LLM Gateway    │ │   memU   │ │   Cron Service   │
│  ┌────────────┐  │ │  Memory  │ │  + Heartbeat     │
│  │  Router    │  │ │  Store   │ │                   │
│  │ (5-tier)   │  │ ├──────────┤ │  • cron jobs      │
│  ├────────────┤  │ │ Intent-  │ │  • daily digest   │
│  │  Models    │  │ │ Aware    │ │  • reminders      │
│  │ (registry) │  │ │ Retrieval│ │                   │
│  ├────────────┤  │ ├──────────┤ └───────────────────┘
│  │ Providers: │  │ │ Proactive│
│  │ • Gemini   │  │ │ Suggest  │
│  │ • OR       │  │ └──────────┘
│  │ • Anthropic│  │
│  │ • CLIProxy │  │
│  └────────────┘  │
└──────────────────┘
```

## Project Structure

```
krolik/
├── agent/         # Core agent loop, context builder, skills, subagents
├── bus/           # Async message bus (inbound/outbound queues)
├── channels/      # Telegram, WhatsApp, Feishu integrations
├── cli/           # Typer CLI commands (gateway, agent, cron, env, status)
├── config/        # Pydantic config schema + env/JSON loader
├── cron/          # Scheduled task service with cron/interval/one-shot
├── heartbeat/     # Periodic agent wake-up via HEARTBEAT.md
├── llm/           # LLM gateway, dynamic model registry, 5-tier router
├── mcp/           # Model Context Protocol client + tool wrapper
├── memory/        # memU integration (store, intent-aware, proactive, cron)
├── providers/     # LiteLLM multi-provider adapter
├── session/       # Conversation history (JSONL persistence)
├── skills/        # Bundled skills: github, weather, tmux, summarize
├── tools/         # Agent tools: filesystem, shell, web, CLI proxy, workflow
└── utils/         # Helpers (paths, dates, filenames)
```

## Configuration

### Environment Variables

Config is loaded from env vars with `KROLIK_` prefix. Legacy `NANOBOT_` prefix is auto-migrated.

```bash
# Required: at least one LLM provider
KROLIK_PROVIDERS__GEMINI__API_KEY=AIza...
KROLIK_PROVIDERS__OPENROUTER__API_KEY=sk-or-...

# Optional: chat channels
KROLIK_CHANNELS__TELEGRAM__ENABLED=true
KROLIK_CHANNELS__TELEGRAM__TOKEN=123456:ABC...

# Optional: memory
KROLIK_MEMORY__MEMU_URL=http://localhost:8000

# Optional: MCP servers
KROLIK_MCP__SERVERS__MEMOS__URL=http://localhost:8001/sse
```

### Config Files

| File | Location | Purpose |
|------|----------|---------|
| `.env` | `~/.krolik/.env` | API keys and secrets |
| `config.json` | `~/.krolik/config.json` | Full config (JSON, lower priority than env) |
| `AGENTS.md` | `~/.krolik/workspace/AGENTS.md` | Agent personality and instructions |
| `SOUL.md` | `~/.krolik/workspace/SOUL.md` | Core identity |
| `MEMORY.md` | `~/.krolik/workspace/memory/MEMORY.md` | Long-term memory file |
| `HEARTBEAT.md` | `~/.krolik/workspace/HEARTBEAT.md` | Pending tasks for heartbeat |

### Default Model

`google/gemini-3-flash-preview` — fast, capable, cost-effective.

Override via `KROLIK_AGENTS__DEFAULTS__MODEL=your/preferred-model`.

## CLI Reference

```bash
krolik onboard          # Initialize config, workspace, .env template
krolik gateway          # Start full gateway (channels + cron + heartbeat)
krolik agent -m "Hi"    # Single message mode
krolik agent            # Interactive REPL

krolik channels status  # Show channel configuration
krolik channels login   # Link WhatsApp via QR code

krolik cron list        # List scheduled jobs
krolik cron add ...     # Add a cron/interval/one-shot job
krolik cron run <id>    # Manually trigger a job

krolik env --show       # Show loaded env vars and API key status
krolik env --edit       # Open .env in $EDITOR
krolik status           # Show overall system status
```

## Development

### Setup

```bash
git clone https://github.com/anatolykoptev/krolik.git
cd krolik
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Tests

```bash
pytest                  # 145 tests
pytest tests/llm/       # LLM gateway, router, tools (96 tests)
pytest tests/memory/    # memU client, store, tools
pytest tests/mcp/       # MCP client, tool wrapper
```

### Optional: memU

```bash
pip install -e ".[memu]"   # Requires Python 3.13 + Rust toolchain
# Or run memU as a service:
cd /path/to/memU && uvicorn memu.app.service:app --port 8000
```

## License

MIT
