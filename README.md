# Krolik 🐰

Proactive AI bot with memU memory integration — a hard-fork from nanobot.

> Based on [nanobot](https://github.com/HKUDS/nanobot), an ultra-lightweight personal AI assistant.

## What's Krolik?

Krolik combines:
- **nanobot** — proactive cron-based scheduling, skills system, multi-channel delivery
- **memU** — advanced vector-based long-term memory with intent-aware retrieval

The result: a bot that not only responds to you, but proactively suggests actions based on your history, preferences, and goals.

## Quick Start

```bash
# Install
pip install -e .

# Configure
export NANOBOT_PROVIDERS__OPENROUTER__API_KEY=your-key
export NANOBOT_CHANNELS__TELEGRAM__TOKEN=your-bot-token

# Run
krolik start
```

## Features

- 🔮 **Proactive Scheduling** — Cron-based jobs that initiate conversations
- 🧠 **Semantic Memory** — memU-powered long-term memory with vector search
- 🎯 **Intent-Aware** — Pre-retrieval decisions for smart context fetching
- 🛠️ **Skills System** — Extensible via SKILL.md files
- 📱 **Multi-Channel** — Telegram, WhatsApp, Feishu support

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      Krolik                         │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │   Cron   │  │  Agent   │  │  memU Memory      │  │
│  │  Service │──│  Loop    │──│  • memorize()     │  │
│  └──────────┘  └──────────┘  │  • retrieve()     │  │
└─────────────────────────────────────────────────────┘
```

## Development

See [NANOBOT_README.md](NANOBOT_README.md) for original nanobot documentation.

## License

MIT
