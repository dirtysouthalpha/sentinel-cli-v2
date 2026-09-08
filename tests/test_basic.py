"""Test hashline parser and patcher."""

import pytest
from src.llm.hashline import parse_patch, apply_patch, Hunk, format_patch

SAMPLE = """*** Begin Patch
PUT src/example.py
@@ def hello():
-    return "hi"
+    return "hello world"
*** End Patch
"""

SAMPLE_LID = """*** Begin Patch
PUT src/example.py
<def hello():
-    return "hi"
+    return "hello world"
*** End Patch
"""


def test_parse_patch():
    hunks = parse_patch(SAMPLE)
    assert len(hunks) == 1
    assert hunks[0].op == "PUT"
    assert hunks[0].path == "src/example.py"
    assert len(hunks[0].lines) > 0


def test_parse_patch_lid():
    hunks = parse_patch(SAMPLE_LID)
    assert len(hunks) == 1
    assert hunks[0].op == "PUT"


def test_parse_patch_rem():
    """REM operation removes a file."""
    patch = """*** Begin Patch
REM src/old.py
*** End Patch
"""
    hunks = parse_patch(patch)
    assert len(hunks) == 1
    assert hunks[0].op == "REM"
    assert hunks[0].path == "src/old.py"


def test_apply_patch(tmp_path):
    """Apply a hashline patch to a file."""
    f = tmp_path / "src" / "example.py"
    f.parent.mkdir()
    f.write_text('def hello():\n    return "hi"\n')
    
    hunks = parse_patch(SAMPLE)
    results = apply_patch(hunks, root=str(tmp_path))
    
    assert len(results) == 1
    assert results[0].success
    new_content = f.read_text()
    assert '"hello world"' in new_content
    assert '"hi"' not in new_content


def test_apply_patch_insert(tmp_path):
    """Apply a patch that inserts new content."""
    patch = """*** Begin Patch
PUT src/new.py
+def hello():
+    return "hello world"
*** End Patch
"""
    hunks = parse_patch(patch)
    results = apply_patch(hunks, root=str(tmp_path))
    
    assert len(results) == 1
    assert results[0].success
    new_content = (tmp_path / "src" / "new.py").read_text()
    assert '"hello world"' in new_content


def test_apply_patch_rem(tmp_path):
    """Apply a patch that removes a file."""
    f = tmp_path / "old.py"
    f.write_text("old content")
    
    patch = """*** Begin Patch
REM old.py
*** End Patch
"""
    hunks = parse_patch(patch)
    results = apply_patch(hunks, root=str(tmp_path))
    
    assert results[0].success
    assert not f.exists()


def test_apply_patch_dry_run(tmp_path):
    """Dry run does not modify files."""
    f = tmp_path / "test.py"
    f.write_text('old content\n')
    
    patch = """*** Begin Patch
PUT test.py
-old content
+new content
*** End Patch
"""
    hunks = parse_patch(patch)
    results = apply_patch(hunks, root=str(tmp_path), dry_run=True)
    
    assert results[0].success
    assert results[0].new_content == "new content\n"
    # File should be unchanged
    assert f.read_text() == "old content\n"


def test_format_patch():
    """Format hunks back to patch text."""
    hunks = [Hunk(op="PUT", path="test.py", lines=["+new line"])]
    text = format_patch(hunks)
    assert "*** Begin Patch" in text
    assert "PUT test.py" in text
    assert "+new line" in text
    assert "*** End Patch" in text


def test_router():
    from src.llm.router import ModelRouter
    router = ModelRouter()
    
    # Boilerplate -> haiku
    model, budget = router.route("generate boilerplate React component")
    assert "haiku" in model
    
    # Review -> opus
    model, budget = router.route("review this code for security issues")
    assert "opus" in model
    
    # Default -> sonnet
    model, budget = router.route("do something random")
    assert "sonnet" in model


def test_context_monitor():
    from src.hooks.context_monitor import ContextMonitor
    cm = ContextMonitor()
    
    # Tool loop
    for _ in range(6):
        result = cm.check_tool_loop("bash")
    assert result is not None
    assert "loop" in result.lower()
    
    # Context exhaustion
    result = cm.check_context_exhaustion(195_000, 200_000)
    assert result is not None
    assert "CRITICAL" in result
    
    # Scope drift (needs 10+ messages)
    messages = [
        {"content": "implement user authentication with JWT tokens"},
        {"content": "first we need to set up the database"},
        {"content": "then create the user model"},
        {"content": "add the login endpoint"},
        {"content": "now analyze the stock market trends for today"},
        {"content": "the weather forecast shows rain"},
        {"content": "quantum computing uses qubits"},
        {"content": "pizza is better than pasta"},
        {"content": "Mars has two moons called Phobos and Deimos"},
        {"content": "the ocean waves are calm today"},
    ]
    result = cm.check_scope_creep(messages)
    assert result is not None
    assert "drift" in result.lower()
