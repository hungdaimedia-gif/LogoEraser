"""Static AST code scanner to verify clean offline operation.

Guarantees that no networking libraries or forbidden shell interpreters
(socket, urllib, http, requests, webbrowser, powershell, wmic) are imported or invoked.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import pytest

BANNED_MODULES = {"socket", "urllib", "http", "requests", "webbrowser"}
BANNED_SUBSTRINGS = {"powershell", "wmic"}


def find_py_files(root: Path) -> list[Path]:
    py_files = []
    for p in root.rglob("*.py"):
        # Ignore virtualenv, build artifacts, caches, and the test scanner itself
        if p.name == "test_no_network.py":
            continue
        parts = [part.lower() for part in p.parts]
        if any(ign in parts for ign in (".venv", "venv", "__pycache__", ".git", ".pytest_cache")):
            continue
        py_files.append(p)
    return py_files


def test_no_network_and_no_forbidden_commands():
    project_root = Path(__file__).parent.parent
    py_files = find_py_files(project_root)
    assert len(py_files) > 0, "No python files found to audit!"

    violations: list[str] = []

    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            violations.append(f"{file_path}: Failed to parse AST ({e})")
            continue

        for node in ast.walk(tree):
            # Check import foo
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0].lower()
                    if root_pkg in BANNED_MODULES:
                        violations.append(
                            f"{file_path}:{node.lineno} - Banned import '{alias.name}'"
                        )

            # Check from foo import bar
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0].lower()
                    if root_pkg in BANNED_MODULES:
                        violations.append(
                            f"{file_path}:{node.lineno} - Banned import from '{node.module}'"
                        )

            # Check string constants for powershell/wmic
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_lower = node.value.lower()
                for banned_str in BANNED_SUBSTRINGS:
                    if banned_str in val_lower:
                        violations.append(
                            f"{file_path}:{node.lineno} - String contains forbidden word '{banned_str}'"
                        )

    assert not violations, "Offline clean-room audit failed with violations:\n" + "\n".join(violations)
