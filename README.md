# Sentinel CLI V2

[![CI](https://github.com/dirtysouthalpha/sentinel-cli-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/dirtysouthalpha/sentinel-cli-v2/actions/workflows/ci.yml)

Harness — like OMP, but better. Model-agnostic coding agent with hashline edits, cost tracking, and model routing.

## Architecture

```
src/llm/
├── types.py          # Frozen dataclasses: LLMInput, ToolDefinition, LLMOutput
├── providers/
│   ├── __init__.py   # Provider resolver (.llm.env / env var / default)
│   ├── sentinel_claude.py
│   ├── sentinel_ollama.py  # llama.cpp :8083, llama-swap :9090, ollama
│   └── echo.py       # Testing
├── agent.py          # ReAct loop + hashline edit format (from OMP pi-edit)
├── router.py         # Model routing (haiku/sonnet/opus heuristic)
├── tools.py          # ToolRegistry + Executor
└── memory.py         # Neuralis brain integration (:8000)

src/hooks/
├── cost_tracker.py   # Read transcript JSONL, sum real usage per model
└── context_monitor.py # Strategic compaction + scope/tool-loop warnings

sc.py                 # CLI entry point
tests/
└── test_basic.py     # 4 passing tests (hashline, router, monitor)
```

## Quick Start

```bash
# Interactive mode
python sc.py

# Single shot
python sc.py --model claude-sonnet-4-5 "implement user auth"

# Auto-route (haiku/sonnet/opus based on task)
python sc.py --route "refactor the api layer"

# Local model (NUKE B60)
python sc.py --provider local-b60-8083 --model qwen3.8-27b
```

## Features

- **Hashline edit format** — surgical edits without reading whole files (from OMP `pi-edit`)
- **Model routing** — haiku for boilerplate, sonnet for impl, opus for review
- **Cost tracking** — reads transcript JSONL, sums real usage (not estimates)
- **Strategic compaction** — dual-signal (context + tool-count), fires at phase boundaries
- **Neuralis memory** — recall/remember via brain API :8000
- **Provider-agnostic** — Claude, Ollama, llama-swap, B60 all wired in

## Tests

```bash
python -m pytest tests/ -v
# 4 passed
```
# sync test 1788888957
# sync test 1788888968
