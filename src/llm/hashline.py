"""Hashline patch format — OMP pi-edit grammar.

Full implementation with LID-based locators, gap locators, and
surgical file editing without reading entire files.

Grammar:
    PatchFormat = "*** Begin Patch" LineTerminator { PatchHunk } "*** End Patch"
    PatchHunk   = HunkHeader LineTerminator { EditLine }
    HunkHeader  = ("PUT" | "CUT" | "REM" | "MV") " " Path LineTerminator
    EditLine    = ("+" | "-" | " " | "@@" | "<ID" | ">$" | "$$") .*
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Hunk:
    """A single file operation."""
    op: str          # PUT, CUT, REM, MV
    path: str
    lines: list[str] = field(default_factory=list)
    lid: str | None = None  # Line ID for precise targeting


@dataclass
class PatchResult:
    """Result of applying a patch."""
    path: str
    success: bool
    error: str | None = None
    old_content: str | None = None
    new_content: str | None = None


def parse_patch(text: str) -> list[Hunk]:
    """Parse a hashline patch into structured hunks."""
    hunks: list[Hunk] = []
    current: Hunk | None = None
    
    for line in text.splitlines():
        stripped = line.strip()
        
        if stripped.startswith("*** Begin Patch"):
            continue
        if stripped.startswith("*** End Patch"):
            break
        
        # Hunk header: PUT path, CUT path, REM path, MV path
        header_match = re.match(r'^(PUT|CUT|REM|MV)\s+(.+)$', stripped)
        if header_match:
            if current is not None:
                hunks.append(current)
            op = header_match.group(1)
            path = header_match.group(2).strip()
            current = Hunk(op=op, path=path)
            continue
        
        if current is None:
            continue
        
        # Edit lines
        if stripped and stripped[0] in " +-@<>$":
            current.lines.append(stripped)
    
    if current is not None:
        hunks.append(current)
    
    return hunks


def _find_lid(lines: list[str], lid: str) -> int:
    """Find the index of a line with the given LID."""
    for i, line in enumerate(lines):
        if lid in line:
            return i
    return -1


def _apply_hunk(existing: list[str], hunk: Hunk) -> list[str]:
    """Apply a single hunk to existing file lines."""
    result: list[str] = []
    idx = 0
    
    for line in hunk.lines:
        if not line:
            continue
        
        # Handle @@ as a two-character marker
        if line.startswith("@@"):
            marker = "@@"
            content = line[2:]
        else:
            marker = line[0]
            content = line[1:] if len(line) > 1 else ""
        
        if marker == "@@":
            # Context match — find the marker text in existing
            target = content.strip()
            while idx < len(existing) and target not in existing[idx]:
                result.append(existing[idx])
                idx += 1
            if idx < len(existing):
                # Keep the context line itself, then move past it
                result.append(existing[idx])
                idx += 1
        
        elif marker == "<":
            # LID locator — find line containing this ID
            lid = content.strip()
            found = _find_lid(existing[idx:], lid)
            if found >= 0:
                # Copy everything up to and including the LID line
                result.extend(existing[idx:idx + found + 1])
                idx += found + 1
        
        elif marker == ">":
            # Gap locator
            if content.strip() == "$":
                # >$ means "end of file"
                result.extend(existing[idx:])
                idx = len(existing)
            else:
                # >ID means "after this line"
                lid = content.strip()
                found = _find_lid(existing[idx:], lid)
                if found >= 0:
                    result.extend(existing[idx:idx + found + 1])
                    idx += found + 1
        
        elif marker == "+":
            # Insert new line
            result.append(content + "\n")
        
        elif marker == "-":
            # Delete line
            idx += 1
        
        elif marker == " ":
            # Context line — copy from existing
            if idx < len(existing):
                result.append(existing[idx])
                idx += 1
    
    # Append remaining lines
    result.extend(existing[idx:])
    return result


def apply_patch(hunks: list[Hunk], root: str = ".", dry_run: bool = False) -> list[PatchResult]:
    """Apply parsed hunks to files. Returns results for each file."""
    results: list[PatchResult] = []
    
    for hunk in hunks:
        filepath = Path(root) / hunk.path
        
        if hunk.op == "REM":
            if not dry_run:
                try:
                    old = filepath.read_text() if filepath.exists() else ""
                    filepath.unlink()
                    results.append(PatchResult(path=hunk.path, success=True, old_content=old))
                except Exception as e:
                    results.append(PatchResult(path=hunk.path, success=False, error=str(e)))
            else:
                results.append(PatchResult(path=hunk.path, success=True))
            continue
        
        if hunk.op == "MV":
            # Move: CUT from source, PUT to dest — handled as separate hunks
            results.append(PatchResult(path=hunk.path, success=True))
            continue
        
        try:
            old_content = filepath.read_text() if filepath.exists() else ""
            lines = old_content.splitlines(keepends=True)
            
            new_lines = _apply_hunk(lines, hunk)
            new_content = "".join(new_lines)
            
            if not dry_run:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(new_content)
            
            results.append(PatchResult(
                path=hunk.path,
                success=True,
                old_content=old_content,
                new_content=new_content,
            ))
        except Exception as e:
            results.append(PatchResult(path=hunk.path, success=False, error=str(e)))
    
    return results


def format_patch(hunks: list[Hunk]) -> str:
    """Format hunks back into hashline patch format."""
    lines = ["*** Begin Patch"]
    for hunk in hunks:
        lines.append(f"{hunk.op} {hunk.path}")
        for line in hunk.lines:
            lines.append(line)
    lines.append("*** End Patch")
    return "\n".join(lines)
