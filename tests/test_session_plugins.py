"""Test session persistence and plugin system."""

import tempfile
from pathlib import Path

from src.llm.session import Session, SessionStore
from src.llm.plugins import (
    Plugin, PluginManager, load_builtin_plugins, ToolDefinition,
    create_filesystem_plugin, create_shell_plugin, create_memory_plugin,
)
from src.llm.types import Message, Role


def test_session_create():
    """Session creates with defaults."""
    s = Session()
    assert s.session_id
    assert len(s.messages) == 0
    assert s.total_usage.cost_usd == 0


def test_session_add_message():
    """Messages can be added."""
    s = Session()
    s.add_message(Message(role=Role.USER, content="hello"))
    assert len(s.messages) == 1
    assert s.messages[0].content == "hello"


def test_session_add_usage():
    """Usage accumulates."""
    s = Session()
    from src.llm.types import UsageCost
    s.add_usage(UsageCost(prompt_tokens=100, completion_tokens=50, cost_usd=0.01, provider="claude", model="sonnet"))
    s.add_usage(UsageCost(prompt_tokens=200, completion_tokens=100, cost_usd=0.02, provider="claude", model="sonnet"))
    assert s.total_usage.prompt_tokens == 300
    assert s.total_usage.completion_tokens == 150
    assert s.total_usage.cost_usd == 0.03


def test_session_to_dict():
    """Session serializes to dict."""
    s = Session()
    s.add_message(Message(role=Role.USER, content="hi"))
    d = s.to_dict()
    assert "session_id" in d
    assert "state" in d
    assert len(d["state"]["messages"]) == 1


def test_session_from_dict():
    """Session deserializes from dict."""
    s = Session()
    s.add_message(Message(role=Role.USER, content="hi"))
    d = s.to_dict()
    s2 = Session.from_dict(d)
    assert s2.session_id == s.session_id
    assert len(s2.messages) == 1


def test_session_store_save_load(tmp_path):
    """SessionStore saves and loads sessions."""
    store = SessionStore(base_dir=tmp_path)
    s = Session(session_id="test-123")
    s.add_message(Message(role=Role.USER, content="hello"))
    store.save(s)
    
    loaded = store.load("test-123")
    assert loaded is not None
    assert loaded.session_id == "test-123"
    assert len(loaded.messages) == 1


def test_session_store_list(tmp_path):
    """SessionStore lists sessions."""
    store = SessionStore(base_dir=tmp_path)
    s1 = Session(session_id="a1")
    s2 = Session(session_id="a2")
    store.save(s1)
    store.save(s2)
    
    sessions = store.list_sessions()
    assert len(sessions) == 2
    ids = {s["session_id"] for s in sessions}
    assert "a1" in ids
    assert "a2" in ids


def test_session_store_latest(tmp_path):
    """SessionStore returns latest session (by modification time)."""
    import time
    store = SessionStore(base_dir=tmp_path)
    s1 = Session(session_id="old")
    store.save(s1)
    time.sleep(0.01)  # Ensure different mtime
    s2 = Session(session_id="new")
    store.save(s2)
    
    latest = store.latest()
    assert latest is not None
    assert latest.session_id == "new"


def test_plugin_create():
    """Plugin creates with metadata."""
    p = Plugin(name="test", version="1.0.0", description="A test plugin")
    assert p.name == "test"
    assert p.version == "1.0.0"
    assert len(p.tools) == 0


def test_plugin_register_tool():
    """Plugin registers tools."""
    p = Plugin(name="test")
    
    def my_tool(x: str) -> str:
        return x
    
    td = ToolDefinition(name="my_tool", description="A tool", input_schema={})
    p.register_tool(td, my_tool)
    
    assert len(p.tools) == 1
    assert "my_tool" in p.handlers


def test_plugin_add_hook():
    """Plugin adds hooks."""
    p = Plugin(name="test")
    p.add_hook("on_start", lambda: None)
    p.add_hook("on_start", lambda: None)
    assert len(p.hooks["on_start"]) == 2


def test_plugin_manager_discover(tmp_path):
    """PluginManager discovers plugins."""
    # Create a test plugin
    plugin_dir = tmp_path / "test_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text("""
from src.llm.plugins import Plugin
def register(plugin: Plugin):
    plugin.register_tool(
        ToolDefinition(name="echo", description="Echo", input_schema={}),
        lambda x: x
    )
""")
    
    pm = PluginManager(plugin_dirs=[tmp_path])
    names = pm.discover()
    assert "test_plugin" in names


def test_plugin_manager_load_all(tmp_path):
    """PluginManager loads all plugins."""
    plugin_dir = tmp_path / "myplugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text("""
__version__ = "2.0.0"
from src.llm.plugins import Plugin, ToolDefinition
def register(plugin: Plugin):
    plugin.register_tool(
        ToolDefinition(name="greet", description="Greet", input_schema={}),
        lambda name: f"Hello, {name}!"
    )
""")
    
    pm = PluginManager(plugin_dirs=[tmp_path])
    count = pm.load_all()
    assert count == 1
    assert "myplugin" in pm.plugins
    assert pm.plugins["myplugin"].version == "2.0.0"


def test_builtin_filesystem_plugin():
    """Filesystem plugin has read, write, ls tools."""
    p = create_filesystem_plugin()
    tool_names = {t.name for t in p.tools}
    assert "read" in tool_names
    assert "write" in tool_names
    assert "ls" in tool_names


def test_builtin_shell_plugin():
    """Shell plugin has shell tool."""
    p = create_shell_plugin()
    tool_names = {t.name for t in p.tools}
    assert "shell" in tool_names


def test_builtin_memory_plugin():
    """Memory plugin has recall, remember, search tools."""
    p = create_memory_plugin()
    tool_names = {t.name for t in p.tools}
    assert "recall" in tool_names
    assert "remember" in tool_names
    assert "search" in tool_names


def test_load_builtin_plugins():
    """All built-in plugins load."""
    plugins = load_builtin_plugins()
    assert "filesystem" in plugins
    assert "shell" in plugins
    assert "memory" in plugins
    assert "web" in plugins


def test_plugin_get_all_tools():
    """PluginManager aggregates tools from all plugins."""
    pm = PluginManager()
    for name, plugin in load_builtin_plugins().items():
        pm._plugins[name] = plugin
    
    tools, handlers = pm.get_all_tools()
    tool_names = {t.name for t in tools}
    assert "read" in tool_names
    assert "shell" in tool_names
    assert "recall" in tool_names
    assert "fetch" in tool_names


def test_plugin_run_hooks():
    """PluginManager runs hooks."""
    pm = PluginManager()
    results = []
    
    p = Plugin(name="test")
    p.add_hook("test_event", lambda **kw: results.append(kw))
    pm._plugins["test"] = p
    
    pm.run_hooks("test_event", foo="bar")
    assert len(results) == 1
    assert results[0]["foo"] == "bar"


def test_session_compaction_flag():
    """Session tracks compaction state."""
    s = Session()
    assert s.state.compacted is False
    s.state.compacted = True
    assert s.state.compacted is True


def test_session_tool_call_count():
    """Session tracks tool calls."""
    s = Session()
    assert s.state.tool_call_count == 0
    s.add_tool_call()
    s.add_tool_call()
    assert s.state.tool_call_count == 2
