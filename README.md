# Krolik 🐰

Proactive AI bot with memU memory and dynamic LLM routing.

## What's Krolik?

A single-package AI assistant that combines:
- **Agent Loop** — tool-calling LLM agent with subagent spawning
- **LLM Gateway** — async multi-provider gateway with dynamic model discovery from OpenRouter API
- **Smart Router** — 5-tier (free/cheap/standard/premium/research) task routing with bilingual EN+RU scoring
- **memU Memory** — vector-based long-term memory with intent-aware retrieval
- **Multi-Channel** — Telegram, WhatsApp, Feishu delivery
- **Proactive Scheduling** — cron-based jobs, heartbeat, memory-driven suggestions

## Quick Start

```bash
pip install -e .

# Configure (at least one LLM provider key)
export NANOBOT_PROVIDERS__GEMINI__API_KEY=your-key
# Optional: Telegram
export NANOBOT_CHANNELS__TELEGRAM__TOKEN=your-bot-token

krolik gateway
```

## Project Structure

```
krolik/
├── agent/       # Core agent loop, context builder, skills, subagents
├── bus/         # Async message bus (inbound/outbound queues)
├── channels/    # Telegram, WhatsApp, Feishu integrations
├── cli/         # Typer CLI commands
├── config/      # Pydantic config schema + loader
├── cron/        # Scheduled task service
├── heartbeat/   # Periodic agent wake-up
├── llm/         # LLM gateway, dynamic model registry, task router
├── mcp/         # Model Context Protocol client
├── memory/      # memU-integrated memory (store, intent-aware, proactive)
├── providers/   # LiteLLM multi-provider adapter
├── session/     # Conversation history
├── skills/      # Bundled skills (github, weather, tmux, etc.)
├── tools/       # All agent tools (filesystem, shell, web, CLI proxy, workflow)
└── utils/       # Helpers
```

## Configuration

Config: `~/.krolik/config.json` or env vars with `NANOBOT_` prefix.
API keys: `~/.krolik/.env`

Default model: `google/gemini-3-flash-preview`

## License

MIT
