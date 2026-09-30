from __future__ import annotations

import os
import re
from pathlib import Path

from .types import ToolResponse


class AgentWorkspace:
    """File tools that cannot escape a single task directory."""

    def __init__(self, root: str | Path, *, max_file_bytes: int = 1_000_000) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes

    def resolve(self, relative: str, *, must_exist: bool = False) -> Path:
        if not relative or "\x00" in relative:
            raise ValueError("Invalid empty or NUL-containing path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes task workspace: {relative}")
        if must_exist and not path.exists():
            raise FileNotFoundError(relative)
        return path

    def list_files(self, pattern: str = "*", limit: int = 500) -> ToolResponse:
        files: list[str] = []
        for path in self.root.rglob(pattern):
            if path.is_file() and not any(part == ".git" for part in path.parts):
                files.append(str(path.relative_to(self.root)))
                if len(files) >= limit:
                    break
        return ToolResponse(True, "\n".join(sorted(files)), {"count": len(files)})

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> ToolResponse:
        resolved = self.resolve(path, must_exist=True)
        if not resolved.is_file():
            return ToolResponse(False, f"Not a file: {path}")
        size = resolved.stat().st_size
        if size > self.max_file_bytes:
            return ToolResponse(False, f"File exceeds {self.max_file_bytes} byte read limit")
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(start_line - 1, 0)
        end = min(max(end_line, start), len(lines))
        rendered = "\n".join(f"{index + 1:4d} | {lines[index]}" for index in range(start, end))
        return ToolResponse(True, rendered, {"lines": len(lines), "bytes": size})

    def search(self, query: str, glob: str = "*.py", limit: int = 100) -> ToolResponse:
        try:
            regex = re.compile(query)
        except re.error as exc:
            return ToolResponse(False, f"Invalid regex: {exc}")
        matches: list[str] = []
        for path in self.root.rglob(glob):
            if not path.is_file() or path.stat().st_size > self.max_file_bytes:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if regex.search(line):
                    matches.append(f"{path.relative_to(self.root)}:{line_number}:{line}")
                    if len(matches) >= limit:
                        return ToolResponse(True, "\n".join(matches), {"truncated": True})
        return ToolResponse(True, "\n".join(matches), {"count": len(matches)})

    def write_file(self, path: str, content: str, *, overwrite: bool = False) -> ToolResponse:
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            return ToolResponse(False, f"Content exceeds {self.max_file_bytes} byte write limit")
        resolved = self.resolve(path)
        if resolved.exists() and not overwrite:
            return ToolResponse(False, "File exists; use replace_text for modifications")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        descriptor = os.open(resolved, flags, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        return ToolResponse(True, f"Wrote {path}", {"bytes": len(encoded)})

    def replace_text(self, path: str, old: str, new: str) -> ToolResponse:
        resolved = self.resolve(path, must_exist=True)
        text = resolved.read_text(encoding="utf-8")
        occurrences = text.count(old)
        if occurrences != 1:
            return ToolResponse(
                False,
                f"Expected old text exactly once, found {occurrences}; no change made",
            )
        updated = text.replace(old, new, 1)
        if len(updated.encode("utf-8")) > self.max_file_bytes:
            return ToolResponse(False, "Updated file exceeds write limit")
        resolved.write_text(updated, encoding="utf-8")
        return ToolResponse(True, f"Patched {path}", {"before": len(text), "after": len(updated)})
