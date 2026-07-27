#!/usr/bin/env python3
"""Check that Python scopes close with ``####`` markers."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


DEFAULT_ROOTS = (Path("src"), Path("scripts"), Path("tests"), Path("examples"))
FUNCTION_NODES = (ast.AsyncFunctionDef, ast.FunctionDef)
REQUIRED_SCOPE_NODES = (
    *FUNCTION_NODES,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Match,
)
SCOPE_NODES = REQUIRED_SCOPE_NODES

def _read_text(path: Path) -> str:
    """Read source text without normalizing platform-specific newlines."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()
    ####
####


def _write_text(path: Path, text: str) -> None:
    """Write source text without translating platform-specific newlines."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
    ####
####


def _indentation(line: str) -> int:
    """Return the leading-space indentation of one source line."""
    return len(line) - len(line.lstrip(" "))
####


def _scope_marker_index(lines: list[str], node: ast.AST) -> int | None:
    """Return the line index of a scope's marker, if one is correctly placed."""
    end_line = getattr(node, "end_lineno", None)
    column = getattr(node, "col_offset", None)
    if end_line is None or column is None:
        return None
    ####
    for index, line in enumerate(lines[end_line:], start=end_line):
        if not line.strip():
            continue
        ####
        indentation = _indentation(line)
        if line.strip() == "####" and indentation == column:
            return index
        ####
        if indentation <= column:
            return None
        ####
    ####
    return None
####


def _is_stub_function(node: ast.AST) -> bool:
    """Return whether a function contains only documentation and a stub body."""
    if not isinstance(node, FUNCTION_NODES):
        return False
    ####
    return bool(node.body) and all(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and (isinstance(statement.value.value, str) or statement.value.value is Ellipsis)
        for statement in node.body
    )
####


def _parent_nodes(tree: ast.AST) -> dict[int, ast.AST]:
    """Return the immediate parent for every node in an AST."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
        ####
    ####
    return parents
####


def _is_elif_node(
        node: ast.If,
        parents: dict[int, ast.AST],
        lines: list[str],
) -> bool:
    """Return whether an ``if`` node is part of its parent's branch chain."""
    parent = parents.get(id(node))
    return (
            isinstance(parent, ast.If)
            and parent.orelse
            and parent.orelse[0] is node
            and lines[node.lineno - 1].lstrip().startswith("elif")
    )
####


def _scope_marker_insert_index(lines: list[str], node: ast.AST) -> int:
    """Return the insertion point after any already-closed child scopes."""
    end_line = getattr(node, "end_lineno", None)
    column = getattr(node, "col_offset", None)
    if end_line is None or column is None:
        raise ValueError("scope node has no source location")
    ####
    index = end_line
    while index < len(lines):
        line = lines[index]
        indentation = _indentation(line)
        if indentation > column:
            index += 1
            continue
        ####
        break
    ####
    return index
####


def _normalize_marker_spacing(lines: list[str]) -> bool:
    """Move closing markers above blank separators so they touch their scope."""
    changed = False
    index = 0
    while index < len(lines):
        if lines[index].strip() != "####":
            index += 1
            continue
        ####
        marker_index = index
        while marker_index > 0 and not lines[marker_index - 1].strip():
            marker_index -= 1
        ####
        if marker_index != index:
            marker = lines.pop(index)
            lines.insert(marker_index, marker)
            changed = True
        ####
        index += 1
    ####
    return changed
####


def _python_files(roots: list[Path]) -> list[Path]:
    """Expand files and directories into a deterministic Python file list."""
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.add(root)
        elif root.is_dir():
            files.update(root.rglob("*.py"))
        ####
    ####
    return sorted(files)
####


def _scope_marker_analysis(path: Path) -> tuple[list[str], list[ast.AST], list[int]]:
    """Return parse diagnostics, missing scopes, and misplaced marker lines."""
    source = _read_text(path)
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: cannot parse Python: {error.msg}"], [], []
    ####
    lines = source.splitlines()
    missing: list[ast.AST] = []
    valid_markers: set[int] = set()
    parents = _parent_nodes(tree)
    for node in ast.walk(tree):
        if not isinstance(node, SCOPE_NODES):
            continue
        ####
        if isinstance(node, ast.If) and _is_elif_node(node, parents, lines):
            continue
        ####
        if _is_stub_function(node):
            continue
        ####
        marker_index = _scope_marker_index(lines, node)
        if marker_index is None and isinstance(node, REQUIRED_SCOPE_NODES):
            missing.append(node)
        elif marker_index is not None:
            valid_markers.add(marker_index)
        ####
    ####
    marker_lines = [index for index, line in enumerate(lines) if line.strip() == "####"]
    misplaced = [index for index in marker_lines if index not in valid_markers]
    return [], missing, misplaced
####


def check_scope_markers(roots: list[Path]) -> list[str]:
    """Return diagnostics for missing or misplaced closing markers."""
    diagnostics: list[str] = []
    for path in _python_files(roots):
        parse_diagnostics, missing, misplaced = _scope_marker_analysis(path)
        diagnostics.extend(parse_diagnostics)
        entries: list[tuple[int, str]] = []
        for node in missing:
            if isinstance(node, ast.ClassDef):
                description = f"class {node.name!r}"
            elif isinstance(node, FUNCTION_NODES):
                description = f"function {node.name!r}"
            elif isinstance(node, ast.If):
                description = "if/elif/else chain"
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                description = "for/else loop"
            elif isinstance(node, ast.While):
                description = "while loop"
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                description = "with statement"
            elif isinstance(node, (ast.Try, ast.TryStar)):
                description = "try/except/else/finally block"
            elif isinstance(node, ast.Match):
                description = "match statement"
            else:
                description = "compound statement"
            ####
            entries.append((node.lineno, f"{path}:{node.lineno}: {description} must close with ####"))
        ####
        entries.extend((index + 1, f"{path}:{index + 1}: unexpected #### marker") for index in misplaced)
        diagnostics.extend(message for _, message in sorted(entries))
    ####
    return diagnostics
####


def fix_scope_markers(roots: list[Path]) -> list[Path]:
    """Remove misplaced markers, insert missing markers, and return changed files."""
    changed: list[Path] = []
    for path in _python_files(roots):
        source = _read_text(path)
        parse_diagnostics, missing, misplaced = _scope_marker_analysis(path)
        if parse_diagnostics:
            continue
        ####
        lines = source.splitlines(keepends=True)
        newline = "\r\n" if "\r\n" in source else "\n"
        for index in reversed(misplaced):
            lines.pop(index)
        ####
        removed_markers = bool(misplaced)
        if removed_markers:
            updated_source = "".join(lines)
            _write_text(path, updated_source)
            parse_diagnostics, missing, _ = _scope_marker_analysis(path)
            if parse_diagnostics:
                continue
            ####
            lines = updated_source.splitlines(keepends=True)
        ####
        insertions: dict[int, list[tuple[int, str]]] = {}
        for node in missing:
            column = getattr(node, "col_offset")
            index = _scope_marker_insert_index(lines, node)
            insertions.setdefault(index, []).append((column, " " * column + "####" + newline))
        ####
        for index in sorted(insertions, reverse=True):
            additions = [line for _, line in sorted(insertions[index], reverse=True)]
            lines[index:index] = additions
        ####
        spacing_changed = _normalize_marker_spacing(lines)
        changed_markers = removed_markers or bool(insertions) or spacing_changed
        if not changed_markers:
            continue
        ####
        _write_text(path, "".join(lines))
        changed.append(path)
    ####
    return changed
####


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="remove misplaced and insert missing scope markers")
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    args = parser.parse_args()
    if args.fix:
        changed = fix_scope_markers(args.paths)
        for path in changed:
            print(f"Fixed scope markers: {path}")
        ####
    ####
    diagnostics = check_scope_markers(args.paths)
    if diagnostics:
        print("Scope-marker check failed:")
        print("\n".join(diagnostics))
        return 1
    ####
    files = _python_files(args.paths)
    print(f"Scope-marker check passed ({len(files)} Python files).")
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
