"""ReAct agent loop — the brain of Sentinel CLI V2.

Uses Session for persistence, PluginManager for extensible tools,
and supports streaming output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from ..types import LLMInput, LLMOutput, ToolDefinition, ToolResult, FinishReason, Message, Role
from ..tools import ToolExecutor
from ..providers import resolve_provider, LLMProvider
from ..session import Session, SessionStore
from ..plugins import PluginManager, load_builtin_plugins
from ..hashline import parse_patch, apply_patch, Hunk
from ..logger import get_logger, EventCategory
from ..retry import ProviderRetry
from ...hooks.cost_tracker import CostTracker, UsageEntry
from ...hooks.context_monitor import ContextMonitor, StrategicCompact


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-5"
    max_iterations: int = 25
    system_prompt: str = "You are Sentinel CLI, a coding assistant."
    provider_override: str | None = None
    session_id: str | None = None
    enable_plugins: bool = True
    enable_cost_tracking: bool = True
    enable_compaction: bool = True
    stream: bool = False
    dry_run: bool = False


class ReActAgent:
    """Reasoning + Acting agent loop with session persistence."""

    def __init__(self, config: AgentConfig, executor: ToolExecutor | None = None):
        self.config = config
        self.provider: LLMProvider = resolve_provider(config.provider_override)
        
        # Session
        store = SessionStore()
        if config.session_id:
            self.session = store.load(config.session_id) or Session(session_id=config.session_id)
        else:
            self.session = Session()
        self.session.state.max_iterations = config.max_iterations
        
        # Plugin manager (must be built before executor)
        self.plugin_manager: PluginManager | None = None
        if config.enable_plugins:
            self.plugin_manager = PluginManager()
            # Load built-in plugins
            for name, plugin in load_builtin_plugins().items():
                self.plugin_manager._plugins[name] = plugin
            # Load MCP server tools (from ~/.sentinel/config.yaml mcp_servers)
            try:
                from ..config import SentinelConfig
                from ..mcp import create_mcp_plugin
                sentinel_cfg = SentinelConfig.load()
                if sentinel_cfg.mcp_servers:
                    mcp_plugin = create_mcp_plugin(sentinel_cfg.mcp_servers)
                    if mcp_plugin.tools:
                        self.plugin_manager._plugins["mcp"] = mcp_plugin
            except Exception:
                pass  # MCP is optional — never block agent startup
            # Load user plugins
            self.plugin_manager.load_all()
        
        # Tool executor
        if executor:
            self.executor = executor
        else:
            self.executor = self._build_executor()
        
        # Hooks
        self.cost_tracker = CostTracker() if config.enable_cost_tracking else None
        self.context_monitor = ContextMonitor() if config.enable_compaction else None
        self.compactor = StrategicCompact() if config.enable_compaction else None
        
        # Logger and retry
        self.logger = get_logger()
        self.retry = ProviderRetry()

    def _build_executor(self) -> ToolExecutor:
        """Build tool executor with all registered tools."""
        from ..tools import ToolRegistry
        registry = ToolRegistry()
        
        # Register tools from plugin manager
        if self.plugin_manager:
            tools, handlers = self.plugin_manager.get_all_tools()
            for tool_def in tools:
                handler = handlers.get(tool_def.name)
                if handler:
                    registry.register(tool_def, handler)
        
        return ToolExecutor(registry)

    def run(self, user_input: str) -> str:
        """Run one turn of the agent loop."""
        self.session.add_message(Message(role=Role.USER, content=user_input))
        self.logger.info(
            "turn_start",
            category=EventCategory.AGENT,
            session_id=self.session.session_id,
            input_length=len(user_input),
        )
        
        for i in range(self.config.max_iterations):
            self.session.state.iteration_count = i + 1
            
            # Build input
            llm_input = LLMInput(
                messages=tuple(self.session.messages),
                model=self.config.model,
                system_prompt=self.config.system_prompt,
                tools=tuple(self.executor.registry.list()),
            )
            
            # Generate with retry + circuit breaker
            output = self._generate_with_retry(llm_input)
            
            # Track usage
            if self.cost_tracker:
                self._track_usage(output.usage)
            self.session.add_usage(output.usage)
            
            # Append assistant response
            self.session.add_message(output.message)
            
            # Check for tool calls
            if output.finish_reason != FinishReason.TOOL_USE or not output.message.tool_calls:
                self._save_session()
                return output.message.content
            
            # Execute tool calls
            for tc in output.message.tool_calls:
                result = self.executor.execute(tc.name, tc.input)
                self.session.add_tool_call()
                
                # Check for tool loops
                if self.context_monitor:
                    loop_warning = self.context_monitor.check_tool_loop(tc.name)
                    if loop_warning:
                        # Inject warning into conversation
                        self.session.add_message(Message(
                            role=Role.SYSTEM,
                            content=f"⚠️ {loop_warning}",
                        ))
                
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=tc.id,
                )
                self.session.add_message(tool_msg)
            
            # Check for compaction
            if self.compactor and self.context_monitor:
                total_tokens = self.session.total_usage.total_tokens
                if self.compactor.should_compact(total_tokens, len(output.message.tool_calls)):
                    self._compact()
        
        self._save_session()
        return self.session.messages[-1].content if self.session.messages else ""

    async def astream(self, user_input: str) -> AsyncIterator[str]:
        """Run one turn with streaming output."""
        self.session.add_message(Message(role=Role.USER, content=user_input))
        
        for i in range(self.config.max_iterations):
            self.session.state.iteration_count = i + 1
            
            llm_input = LLMInput(
                messages=tuple(self.session.messages),
                model=self.config.model,
                system_prompt=self.config.system_prompt,
                tools=tuple(self.executor.registry.list()),
            )
            
            output = self.provider.generate(llm_input)
            
            if self.cost_tracker:
                self._track_usage(output.usage)
            self.session.add_usage(output.usage)
            self.session.add_message(output.message)
            
            # Yield text content
            if output.message.content:
                yield output.message.content
            
            if output.finish_reason != FinishReason.TOOL_USE or not output.message.tool_calls:
                self._save_session()
                return
            
            for tc in output.message.tool_calls:
                result = self.executor.execute(tc.name, tc.input)
                self.session.add_tool_call()
                
                if self.context_monitor:
                    loop_warning = self.context_monitor.check_tool_loop(tc.name)
                    if loop_warning:
                        self.session.add_message(Message(
                            role=Role.SYSTEM,
                            content=f"⚠️ {loop_warning}",
                        ))
                
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.content,
                    name=tc.id,
                )
                self.session.add_message(tool_msg)
                
                # Yield tool result
                yield f"\n[Tool: {tc.name}]\n{result.content}\n"
            
            if self.compactor and self.context_monitor:
                total_tokens = self.session.total_usage.total_tokens
                if self.compactor.should_compact(total_tokens, len(output.message.tool_calls)):
                    self._compact()
        
        self._save_session()

    def _generate_with_retry(self, llm_input: LLMInput) -> LLMOutput:
        """Generate with circuit breaker and retry logic."""
        provider_name = type(self.provider).__name__
        
        if not self.retry.can_use(provider_name):
            self.logger.warn(
                "circuit_open",
                category=EventCategory.PROVIDER,
                provider=provider_name,
            )
            raise RuntimeError(f"Circuit breaker open for {provider_name}")
        
        try:
            output = self.provider.generate(llm_input)
            self.retry.record_success(provider_name)
            return output
        except Exception as e:
            self.retry.record_failure(provider_name)
            self.logger.error(
                "provider_error",
                category=EventCategory.PROVIDER,
                provider=provider_name,
                error=str(e),
            )
            raise

    def _track_usage(self, usage: Any) -> None:
        """Record usage to cost tracker."""
        if not self.cost_tracker:
            return
        entry = UsageEntry(
            timestamp=self.session.updated_at,
            model=usage.model or self.config.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=usage.cost_usd,
        )
        self.cost_tracker.record(entry)

    def _compact(self) -> None:
        """Compact the session — summarize and reset context."""
        # Simple compaction: keep last N messages
        keep = 10
        if len(self.session.messages) > keep:
            self.session.messages = self.session.messages[-keep:]
            self.session.state.compacted = True
            if self.compactor:
                self.compactor.reset()

    def _save_session(self) -> None:
        """Persist session to disk."""
        store = SessionStore()
        store.save(self.session)

    def apply_hashline(self, patch_text: str) -> dict[str, Any]:
        """Apply a hashline patch to files."""
        hunks = parse_patch(patch_text)
        results = apply_patch(hunks, dry_run=self.config.dry_run)
        return {
            "results": [
                {
                    "path": r.path,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ]
        }
