"""Cost tracker hook — reads transcript JSONL, sums real usage per model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class UsageEntry:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Tracks LLM usage and costs from transcript files."""
    
    transcript_dir: Path = field(default_factory=lambda: Path.home() / ".sentinel" / "transcripts")
    metrics_file: Path = field(default_factory=lambda: Path.home() / ".sentinel" / "metrics" / "costs.jsonl")
    
    def __post_init__(self):
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def record(self, entry: UsageEntry) -> None:
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({
                "ts": entry.timestamp,
                "model": entry.model,
                "input": entry.input_tokens,
                "output": entry.output_tokens,
                "cost": entry.cost_usd,
            }) + "\n")
    
    def summarize(self, session_id: str | None = None) -> dict:
        """Summarize total cost. If session_id given, filter to that session."""
        total = 0.0
        total_input = 0
        total_output = 0
        by_model: dict[str, float] = {}
        
        if not self.metrics_file.exists():
            return {"total": 0, "input": 0, "output": 0, "by_model": {}}
        
        with open(self.metrics_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if session_id and entry.get("session_id") != session_id:
                    continue
                total += entry.get("cost", 0)
                total_input += entry.get("input", 0)
                total_output += entry.get("output", 0)
                model = entry.get("model", "unknown")
                by_model[model] = by_model.get(model, 0) + entry.get("cost", 0)
        
        return {
            "total": round(total, 4),
            "input": total_input,
            "output": total_output,
            "by_model": {k: round(v, 4) for k, v in by_model.items()},
        }
    
    @classmethod
    def parse_transcript(cls, path: Path) -> list[UsageEntry]:
        """Parse a Claude Code or Sentinel transcript JSONL."""
        entries = []
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or obj.get("cost") or {}
                if usage.get("total_cost_usd") or usage.get("input_tokens"):
                    entries.append(UsageEntry(
                        timestamp=obj.get("timestamp", datetime.now().isoformat()),
                        model=obj.get("model", "unknown"),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cost_usd=usage.get("total_cost_usd", 0),
                    ))
        return entries
