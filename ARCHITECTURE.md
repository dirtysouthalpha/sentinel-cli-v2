# Sentinel CLI V2 — Architecture & Build Plan

## Current State (verified 2026-09-07)
- 25 files, 36 passing tests
- ReAct agent loop with session persistence
- Plugin system (filesystem, shell, memory builtins)
- 4 providers: Claude, Ollama (llama-swap), OpenAI, Echo
- Hashline surgical edit format (OMP grammar)
- Model routing (haiku/sonnet/opus)
- Cost tracking + strategic compaction
- Neuralis brain API integration

## Design Philosophy
"OMP but better" means:
- **Memory-first**: Every conversation, every decision, every outcome is remembered. The system learns.
- **Audit-everything**: Every tool call, model choice, and cost is logged and reviewable.
- **Grind mode**: Autonomous execution with human gate, like the brain's "grind" instruction (n.128577).
- **Plugin ecosystem**: Tools come from a marketplace, not hardcoded.
- **Streaming UX**: Real-time output, not blocking calls.

## Layer 1 — Memory Foundation
```
src/llm/memory/
├── __init__.py          # MemoryBackend ABC + resolver
├── neuralis.py          # Brain API :8000 (already exists as memory.py)
├── cache.py             # Local SQLite cache with TTL
├── mnemopi.py           # Memory plugins (onRemember/onRecall/onConsolidate)
└── consolidation.py     # Periodic background consolidation
```

## Layer 2 — Tool Ecosystem
```
src/llm/tools/
├── __init__.py          # ToolRegistry + ToolExecutor (exists)
├── filesystem.py        # Read, write, edit, hashline (exists in plugins)
├── shell.py             # Shell execution with sandbox (exists in plugins)
├── browser.py           # Headless browser (fleet fetch)
├── memory_tool.py       # Recall/remember/search
├── mcp_client.py        # MCP protocol client for external tool servers
└── plugin_marketplace.py # Install/uninstall from ~/.sentinel/plugins/
```

## Layer 3 — Agent Intelligence
```
src/llm/agent/
├── __init__.py
├── react.py             # ReAct loop (exists, rename)
├── grind.py             # Autonomous grind mode
├── planner.py           # Multi-step task decomposition
└── recovery.py          # Error recovery, retry, fallback
```

## Layer 4 — UX & Audit
```
src/cli/
├── __init__.py
├── repl.py              # Interactive REPL with history, completion
├── stream.py            # Streaming output handler
├── audit.py             # Audit log viewer
└── commands/            # CLI subcommands
    ├── sessions.py
    ├── costs.py
    ├── plugins.py
    ├── brain.py
    └── grind.py

src/hooks/
├── cost_tracker.py      # (exists)
├── context_monitor.py   # (exists)
└── audit_hook.py        # Every tool call logged with input/output/cost
```

## Layer 5 — Configuration
```
~/.sentinel/
├── config.yaml          # User config (providers, defaults, plugins)
├── sessions/             # Session storage
├── plugins/              # Installed plugins
├── audit/                # Audit logs
└── cache/                # Memory cache
```

## Implementation Order
1. Memory cache layer (local SQLite + brain backend)
2. Grind mode (autonomous execution)
3. Streaming REPL
4. Audit hook
5. MCP client
6. Plugin marketplace
7. Full audit at end
