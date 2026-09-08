"""Pure path, license, data and static-interface preservation contracts."""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Any

from scripts.preservation.git import PreservationError


def validate_paths(paths: list[str]) -> None:
    """Reject traversal and NFC/casefold collisions in O(total path components)."""
    names: dict[str, str] = {}
    files: set[str] = set()
    directories: set[str] = set()
    for path in paths:
        parts = path.split("/")
        if (
            not path
            or len(path.encode()) > 4096
            or "\\" in path
            or any(part in ("", ".", "..") for part in parts)
            or any(ord(c) < 32 or ord(c) == 127 for c in path)
            or any(":" in part or part.endswith((" ", ".")) for part in parts)
        ):
            raise PreservationError("unsafe-path")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            key = unicodedata.normalize("NFC", prefix).casefold()
            if key in names and names[key] != prefix:
                raise PreservationError("path-collision")
            names[key] = prefix
            if index == len(parts):
                if key in files or key in directories:
                    raise PreservationError("path-collision")
                files.add(key)
            else:
                if key in files:
                    raise PreservationError("path-collision")
                directories.add(key)


def category(path: str) -> str:
    """Every file receives a capability class; unclassified assets stay visible."""
    if path.startswith(".github/"):
        return "workflow"
    if path.startswith(("tests/", "test/")) or "/tests/" in path:
        return "test"
    if path.startswith(("docs/assets/", "docs/evidence/", "reports/")):
        return "evidence"
    if path.startswith("docs/") or path.endswith(".md"):
        return "documentation"
    if path.startswith(("apps/", "console/", "web/")):
        return "gui-api"
    if path.startswith(("configs/", "config/")) or path.endswith((".toml", ".lock", ".yaml")):
        return "configuration"
    if path.startswith("scripts/"):
        return "cli-tooling"
    if path.endswith((".py", ".cpp", ".h", ".hpp", ".ts", ".tsx")):
        return "package"
    return "other-preserved"


def findings(path: str, content: bytes) -> list[str]:
    """Conservative deny rules; never emit matching payloads or claim full DLP."""
    result = []
    lowered = path.casefold()
    if any(part in {".venv", "node_modules", "__pycache__", ".env"} for part in lowered.split("/")):
        result.append("runtime-or-secret-path")
    if lowered.startswith(("data/raw/", "data/vendor/", "data/cache/", "data/processed/", "runs/")):
        result.append("raw-or-runtime-data")
    if PurePosixPath(lowered).suffix in {".pem", ".key", ".pkl", ".pickle", ".pt", ".h5"}:
        result.append("secret-or-model-artifact")
    if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        result.append("lfs-external-object")
    patterns = (
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b",
        rb"\bAKIA[0-9A-Z]{16}\b",
    )
    if any(re.search(pattern, content) for pattern in patterns):
        result.append("secret-marker")
    return result


def static_interfaces(path: str, content: bytes, node_limit: int) -> list[dict[str, Any]]:
    """Inventory syntactic public definitions, reexports and CLI/API declarations.

    Does not import source. Dynamic registration is explicitly unresolved; the
    subsequent migration must retain runtime contract tests, not infer parity.
    """
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(content, filename="committed-source")
        if sum(1 for _ in ast.walk(tree)) > node_limit:
            raise PreservationError("ast-node-limit")
    except (SyntaxError, UnicodeError, RecursionError, ValueError) as exc:
        raise PreservationError("invalid-or-oversized-python") from exc
    records: list[dict[str, Any]] = []

    def add(node: ast.AST, kind: str, name: str) -> None:
        records.append({"kind": kind, "name": name, "line": getattr(node, "lineno", 0)})

    def declarations(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    add(
                        node,
                        "class" if isinstance(node, ast.ClassDef) else "callable",
                        prefix + node.name,
                    )
                if isinstance(node, ast.ClassDef):
                    declarations(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if not name.startswith("_"):
                        add(node, "import-binding", prefix + name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and (
                        not target.id.startswith("_") or target.id == "__all__"
                    ):
                        add(node, "binding", prefix + target.id)
            elif isinstance(node, (ast.If, ast.Try)):
                declarations(node.body, prefix)
                declarations(node.orelse, prefix)
                if isinstance(node, ast.Try):
                    declarations(node.finalbody, prefix)
                    for handler in node.handlers:
                        declarations(handler.body, prefix)

    declarations(tree.body)
    verbs = {
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "route",
        "websocket",
        "add_api_route",
        "add_argument",
        "add_parser",
        "command",
        "option",
        "argument",
        "include_router",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            # Do not recursively unparse untrusted expressions: long valid
            # attribute chains can exceed Python's recursion limit. Only names
            # and attributes participate in static receiver classification.
            receiver_parts: list[str] = []
            if isinstance(node.func, ast.Attribute):
                receiver_node: ast.expr = node.func.value
                while isinstance(receiver_node, ast.Attribute):
                    receiver_parts.append(receiver_node.attr)
                    receiver_node = receiver_node.value
                if isinstance(receiver_node, ast.Name):
                    receiver_parts.append(receiver_node.id)
            receiver = ".".join(reversed(receiver_parts))
            route_receiver = receiver.split(".")[-1] in {"app", "api", "router"}
            cli_receiver = any(part in receiver for part in ("parser", "click", "typer"))
            if name in verbs and (route_receiver or cli_receiver):
                literals = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                record = {
                    "kind": "cli-api-declaration",
                    "name": name,
                    "line": node.lineno,
                    "literals": literals,
                    "dynamic": not bool(literals),
                }
                records.append(record)
    return sorted(records, key=lambda row: (row["line"], row["kind"], row["name"]))
