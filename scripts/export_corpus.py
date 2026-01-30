#!/usr/bin/env python3
"""Export HF-GTC corpus to Arrow/Parquet format for alimentar distribution.

This script extracts all Python modules from the hf_gtc package and exports
them to a structured Parquet file for distribution via HuggingFace Hub.

Usage:
    python scripts/export_corpus.py --output hf_gtc_corpus.parquet
    python scripts/export_corpus.py --output corpus.parquet --coverage coverage.json
"""

from __future__ import annotations

import argparse
import ast
import doctest
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Require pyarrow for Parquet export
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    msg = "pyarrow is required for corpus export: uv pip install pyarrow"
    raise ImportError(msg) from e


@dataclass
class FunctionInfo:
    """Metadata for an extracted function."""

    name: str
    signature: str
    docstring: str
    line_number: int


@dataclass
class DoctestInfo:
    """Metadata for an extracted doctest."""

    source: str
    expected: str
    line_number: int


@dataclass
class ModuleInfo:
    """Extracted metadata for a Python module."""

    module_path: str
    category: str
    content: str
    functions: list[FunctionInfo] = field(default_factory=list)
    doctests: list[DoctestInfo] = field(default_factory=list)
    coverage: float = 0.0
    test_count: int = 0


def get_category_from_path(module_path: str) -> str:
    """Extract category from module path.

    Args:
        module_path: Relative path like "training/lora.py"

    Returns:
        Category name (e.g., "training")

    Examples:
        >>> get_category_from_path("training/lora.py")
        'training'
        >>> get_category_from_path("hub/cards.py")
        'hub'
    """
    parts = Path(module_path).parts
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def extract_function_signature(node: ast.FunctionDef) -> str:
    """Extract function signature as a string.

    Args:
        node: AST FunctionDef node

    Returns:
        Function signature string

    Examples:
        >>> import ast
        >>> code = "def foo(x: int, y: str = 'bar') -> bool: pass"
        >>> tree = ast.parse(code)
        >>> func = tree.body[0]
        >>> sig = extract_function_signature(func)
        >>> "x: int" in sig
        True
    """
    args_parts = []

    # Process regular arguments
    defaults_offset = len(node.args.args) - len(node.args.defaults)
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"

        # Check for default value
        default_idx = i - defaults_offset
        if default_idx >= 0:
            default = node.args.defaults[default_idx]
            arg_str += f" = {ast.unparse(default)}"

        args_parts.append(arg_str)

    # Process *args
    if node.args.vararg:
        vararg = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg += f": {ast.unparse(node.args.vararg.annotation)}"
        args_parts.append(vararg)
    elif node.args.kwonlyargs:
        args_parts.append("*")

    # Process keyword-only arguments
    kw_defaults_dict = {
        i: d for i, d in enumerate(node.args.kw_defaults) if d is not None
    }
    for i, kwarg in enumerate(node.args.kwonlyargs):
        kw_str = kwarg.arg
        if kwarg.annotation:
            kw_str += f": {ast.unparse(kwarg.annotation)}"
        if i in kw_defaults_dict:
            kw_str += f" = {ast.unparse(kw_defaults_dict[i])}"
        args_parts.append(kw_str)

    # Process **kwargs
    if node.args.kwarg:
        kwarg = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
        args_parts.append(kwarg)

    # Build signature
    args_str = ", ".join(args_parts)
    sig = f"def {node.name}({args_str})"

    if node.returns:
        sig += f" -> {ast.unparse(node.returns)}"

    return sig


def extract_functions(tree: ast.Module) -> list[FunctionInfo]:
    """Extract all function definitions from an AST.

    Args:
        tree: Parsed AST module

    Returns:
        List of FunctionInfo objects

    Examples:
        >>> import ast
        >>> code = '''
        ... def foo(x: int) -> int:
        ...     \"\"\"Docstring.\"\"\"
        ...     return x
        ... '''
        >>> tree = ast.parse(code)
        >>> funcs = extract_functions(tree)
        >>> len(funcs)
        1
        >>> funcs[0].name
        'foo'
    """
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions
            if node.name.startswith("_"):
                continue

            docstring = ast.get_docstring(node) or ""
            signature = extract_function_signature(node)

            functions.append(
                FunctionInfo(
                    name=node.name,
                    signature=signature,
                    docstring=docstring,
                    line_number=node.lineno,
                )
            )

    return functions


def extract_doctests(content: str) -> list[DoctestInfo]:
    """Extract all doctests from module content.

    Parses the AST to find all docstrings, then extracts doctests from each.

    Args:
        content: Python source code

    Returns:
        List of DoctestInfo objects

    Examples:
        >>> code = '''
        ... def foo():
        ...     \"\"\"Example.
        ...
        ...     >>> 1 + 1
        ...     2
        ...     \"\"\"
        ...     pass
        ... '''
        >>> tests = extract_doctests(code)
        >>> len(tests)
        1
        >>> tests[0].source.strip()
        '1 + 1'
        >>> tests[0].expected.strip()
        '2'
    """
    parser = doctest.DocTestParser()
    doctests = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return doctests

    # Collect all docstrings from the AST
    docstrings = []

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docstrings.append((module_doc, 1))

    # Function and class docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append((doc, node.lineno))

    # Extract examples from each docstring
    for docstring, base_lineno in docstrings:
        try:
            examples = parser.get_examples(docstring)
            for ex in examples:
                doctests.append(
                    DoctestInfo(
                        source=ex.source,
                        expected=ex.want,
                        line_number=base_lineno + ex.lineno,
                    )
                )
        except Exception:
            # Skip malformed doctests
            pass

    return doctests


def iter_modules(src_dir: Path) -> Iterator[tuple[str, str]]:
    """Iterate over all Python modules in source directory.

    Args:
        src_dir: Path to hf_gtc source directory

    Yields:
        Tuples of (relative_path, content)

    Examples:
        >>> from pathlib import Path
        >>> # This would iterate real modules in the package
        >>> # For testing we just verify the function exists
        >>> callable(iter_modules)
        True
    """
    for py_file in src_dir.rglob("*.py"):
        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue

        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue

        relative_path = py_file.relative_to(src_dir)
        content = py_file.read_text(encoding="utf-8")

        yield str(relative_path), content


def load_coverage(coverage_file: Path | None) -> dict[str, float]:
    """Load coverage data from pytest-cov JSON output.

    Args:
        coverage_file: Path to coverage.json, or None

    Returns:
        Dictionary mapping module paths to coverage percentages

    Examples:
        >>> load_coverage(None)
        {}
    """
    if coverage_file is None or not coverage_file.exists():
        return {}

    try:
        data = json.loads(coverage_file.read_text())
        coverage_data = {}

        # Handle pytest-cov JSON format
        if "files" in data:
            for file_path, file_data in data["files"].items():
                if "summary" in file_data:
                    pct = file_data["summary"].get("percent_covered", 0.0)
                    # Normalize path
                    rel_path = file_path.replace("src/hf_gtc/", "")
                    coverage_data[rel_path] = pct

        return coverage_data
    except Exception:
        return {}


def extract_module(module_path: str, content: str, coverage: float = 0.0) -> ModuleInfo:
    """Extract all metadata from a Python module.

    Args:
        module_path: Relative path to module
        content: Module source code
        coverage: Test coverage percentage

    Returns:
        ModuleInfo with extracted data

    Examples:
        >>> code = '''
        ... def foo(x: int) -> int:
        ...     \"\"\"Add one.
        ...
        ...     >>> foo(1)
        ...     2
        ...     \"\"\"
        ...     return x + 1
        ... '''
        >>> info = extract_module("test/module.py", code, 95.0)
        >>> info.category
        'test'
        >>> len(info.functions)
        1
    """
    category = get_category_from_path(module_path)

    try:
        tree = ast.parse(content)
        functions = extract_functions(tree)
    except SyntaxError:
        functions = []

    doctests = extract_doctests(content)

    return ModuleInfo(
        module_path=module_path,
        category=category,
        content=content,
        functions=functions,
        doctests=doctests,
        coverage=coverage,
        test_count=len(doctests),
    )


def modules_to_arrow(modules: list[ModuleInfo]) -> pa.Table:
    """Convert module info list to Arrow table.

    Args:
        modules: List of ModuleInfo objects

    Returns:
        PyArrow Table with corpus schema

    Examples:
        >>> info = ModuleInfo(
        ...     module_path="test/m.py",
        ...     category="test",
        ...     content="# code",
        ...     functions=[],
        ...     doctests=[],
        ...     coverage=95.0,
        ...     test_count=0,
        ... )
        >>> table = modules_to_arrow([info])
        >>> table.num_rows
        1
    """
    # Build arrays for each column
    module_paths = []
    categories = []
    contents = []
    function_arrays = []
    doctest_arrays = []
    coverages = []
    test_counts = []

    for mod in modules:
        module_paths.append(mod.module_path)
        categories.append(mod.category)
        contents.append(mod.content)
        coverages.append(mod.coverage)
        test_counts.append(mod.test_count)

        # Convert functions to struct array
        func_structs = []
        for fn in mod.functions:
            func_structs.append(
                {
                    "name": fn.name,
                    "signature": fn.signature,
                    "docstring": fn.docstring,
                    "line_number": fn.line_number,
                }
            )
        function_arrays.append(func_structs)

        # Convert doctests to struct array
        doctest_structs = []
        for dt in mod.doctests:
            doctest_structs.append(
                {
                    "source": dt.source,
                    "expected": dt.expected,
                    "line_number": dt.line_number,
                }
            )
        doctest_arrays.append(doctest_structs)

    # Define schema matching spec 4.6.3
    function_type = pa.struct(
        [
            ("name", pa.string()),
            ("signature", pa.string()),
            ("docstring", pa.string()),
            ("line_number", pa.int32()),
        ]
    )

    doctest_type = pa.struct(
        [
            ("source", pa.string()),
            ("expected", pa.string()),
            ("line_number", pa.int32()),
        ]
    )

    schema = pa.schema(
        [
            ("module", pa.string()),
            ("category", pa.string()),
            ("content", pa.string()),
            ("functions", pa.list_(function_type)),
            ("doctests", pa.list_(doctest_type)),
            ("coverage", pa.float32()),
            ("test_count", pa.int32()),
        ]
    )

    # Create table
    return pa.table(
        {
            "module": module_paths,
            "category": categories,
            "content": contents,
            "functions": function_arrays,
            "doctests": doctest_arrays,
            "coverage": coverages,
            "test_count": test_counts,
        },
        schema=schema,
    )


def export_corpus(
    src_dir: Path,
    output_file: Path,
    coverage_file: Path | None = None,
) -> int:
    """Export corpus to Parquet file.

    Args:
        src_dir: Path to hf_gtc source directory
        output_file: Output Parquet file path
        coverage_file: Optional coverage.json path

    Returns:
        Number of modules exported

    Examples:
        >>> # Integration test would use real paths
        >>> callable(export_corpus)
        True
    """
    # Load coverage data
    coverage_data = load_coverage(coverage_file)

    # Extract all modules
    modules = []
    for module_path, content in iter_modules(src_dir):
        coverage = coverage_data.get(module_path, 0.0)
        info = extract_module(module_path, content, coverage)
        modules.append(info)
        print(f"  Extracted: {module_path} ({len(info.functions)} functions)")

    if not modules:
        print("No modules found to export", file=sys.stderr)
        return 0

    # Convert to Arrow and write Parquet
    table = modules_to_arrow(modules)
    pq.write_table(table, output_file, compression="snappy")

    print(f"\nExported {len(modules)} modules to {output_file}")
    print(f"  Total functions: {sum(len(m.functions) for m in modules)}")
    print(f"  Total doctests: {sum(len(m.doctests) for m in modules)}")

    return len(modules)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export HF-GTC corpus to Parquet format"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("hf_gtc_corpus.parquet"),
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("src/hf_gtc"),
        help="Source directory containing hf_gtc package",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=None,
        help="Path to coverage.json from pytest-cov",
    )

    args = parser.parse_args()

    if not args.src_dir.exists():
        print(f"Error: Source directory not found: {args.src_dir}", file=sys.stderr)
        return 1

    print(f"Exporting corpus from {args.src_dir}")
    count = export_corpus(args.src_dir, args.output, args.coverage)

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
