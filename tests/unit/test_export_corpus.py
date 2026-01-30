"""Tests for corpus export script.

Tests cover AST extraction, doctest parsing, Arrow/Parquet conversion,
and end-to-end export functionality.
"""

from __future__ import annotations

import ast
import json

# Import module under test
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from export_corpus import (
    DoctestInfo,
    FunctionInfo,
    ModuleInfo,
    extract_doctests,
    extract_function_signature,
    extract_functions,
    extract_module,
    get_category_from_path,
    iter_modules,
    load_coverage,
    modules_to_arrow,
)


class TestGetCategoryFromPath:
    """Tests for get_category_from_path function."""

    def test_training_category(self) -> None:
        """Test training module path extraction."""
        assert get_category_from_path("training/lora.py") == "training"

    def test_hub_category(self) -> None:
        """Test hub module path extraction."""
        assert get_category_from_path("hub/cards.py") == "hub"

    def test_inference_category(self) -> None:
        """Test inference module path extraction."""
        assert get_category_from_path("inference/batch.py") == "inference"

    def test_preprocessing_category(self) -> None:
        """Test preprocessing module path extraction."""
        result = get_category_from_path("preprocessing/tokenization.py")
        assert result == "preprocessing"

    def test_deployment_category(self) -> None:
        """Test deployment module path extraction."""
        assert get_category_from_path("deployment/gguf.py") == "deployment"

    def test_evaluation_category(self) -> None:
        """Test evaluation module path extraction."""
        assert get_category_from_path("evaluation/metrics.py") == "evaluation"

    def test_nested_path(self) -> None:
        """Test deeply nested path."""
        assert get_category_from_path("training/advanced/qlora.py") == "training"

    def test_single_file(self) -> None:
        """Test single file without subdirectory."""
        assert get_category_from_path("module.py") == "module.py"


class TestExtractFunctionSignature:
    """Tests for extract_function_signature function."""

    def test_simple_function(self) -> None:
        """Test simple function with no args."""
        code = "def foo(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert sig == "def foo()"

    def test_typed_args(self) -> None:
        """Test function with typed arguments."""
        code = "def foo(x: int, y: str) -> bool: pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert sig == "def foo(x: int, y: str) -> bool"

    def test_default_values(self) -> None:
        """Test function with default values."""
        code = "def foo(x: int = 10) -> int: pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "x: int = 10" in sig

    def test_string_default(self) -> None:
        """Test function with string default."""
        code = "def foo(name: str = 'default') -> str: pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "name: str" in sig
        assert "'default'" in sig

    def test_args_kwargs(self) -> None:
        """Test function with *args and **kwargs."""
        code = "def foo(*args, **kwargs): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "*args" in sig
        assert "**kwargs" in sig

    def test_typed_args_kwargs(self) -> None:
        """Test function with typed *args and **kwargs."""
        code = "def foo(*args: int, **kwargs: str): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "*args: int" in sig
        assert "**kwargs: str" in sig

    def test_keyword_only_args(self) -> None:
        """Test function with keyword-only arguments."""
        code = "def foo(*, x: int, y: str = 'a'): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "x: int" in sig
        assert "y: str" in sig

    def test_complex_return_type(self) -> None:
        """Test function with complex return type."""
        code = "def foo() -> list[dict[str, int]]: pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert "-> list[dict[str, int]]" in sig

    def test_no_return_type(self) -> None:
        """Test function without return type annotation."""
        code = "def foo(x: int): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = extract_function_signature(func)
        assert sig == "def foo(x: int)"
        assert "->" not in sig


class TestExtractFunctions:
    """Tests for extract_functions function."""

    def test_single_function(self) -> None:
        """Test extracting a single function."""
        code = '''
def foo(x: int) -> int:
    """Docstring."""
    return x
'''
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 1
        assert funcs[0].name == "foo"
        assert funcs[0].docstring == "Docstring."
        assert funcs[0].line_number == 2

    def test_multiple_functions(self) -> None:
        """Test extracting multiple functions."""
        code = '''
def foo(): pass
def bar(): pass
def baz(): pass
'''
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 3
        names = [f.name for f in funcs]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_skips_private_functions(self) -> None:
        """Test that private functions are skipped."""
        code = '''
def public(): pass
def _private(): pass
def __dunder__(): pass
'''
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 1
        assert funcs[0].name == "public"

    def test_nested_functions_extracted(self) -> None:
        """Test that nested functions are also extracted."""
        code = '''
def outer():
    def inner():
        pass
'''
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        # ast.walk finds both outer and inner
        names = [f.name for f in funcs]
        assert "outer" in names
        assert "inner" in names

    def test_method_in_class(self) -> None:
        """Test extracting methods from classes."""
        code = '''
class Foo:
    def method(self) -> None:
        """Method doc."""
        pass
'''
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        assert len(funcs) == 1
        assert funcs[0].name == "method"

    def test_empty_module(self) -> None:
        """Test empty module returns empty list."""
        code = "# Just a comment"
        tree = ast.parse(code)
        funcs = extract_functions(tree)
        assert funcs == []


class TestExtractDoctests:
    """Tests for extract_doctests function."""

    def test_simple_doctest(self) -> None:
        """Test extracting simple doctest."""
        code = '''
def foo():
    """Example.

    >>> 1 + 1
    2
    """
    pass
'''
        tests = extract_doctests(code)
        assert len(tests) == 1
        assert tests[0].source.strip() == "1 + 1"
        assert tests[0].expected.strip() == "2"

    def test_multiple_doctests(self) -> None:
        """Test extracting multiple doctests."""
        code = '''
def foo():
    """Examples.

    >>> 1 + 1
    2
    >>> 2 * 3
    6
    """
    pass
'''
        tests = extract_doctests(code)
        assert len(tests) == 2

    def test_multiline_doctest(self) -> None:
        """Test extracting multiline doctest."""
        code = '''
def foo():
    """Example.

    >>> x = 1
    >>> y = 2
    >>> x + y
    3
    """
    pass
'''
        tests = extract_doctests(code)
        assert len(tests) >= 1

    def test_doctest_with_exception(self) -> None:
        """Test doctest expecting exception."""
        code = '''
def foo():
    """Example.

    >>> raise ValueError("oops")
    Traceback (most recent call last):
    ValueError: oops
    """
    pass
'''
        tests = extract_doctests(code)
        assert len(tests) == 1

    def test_no_doctests(self) -> None:
        """Test module with no doctests."""
        code = '''
def foo():
    """Just documentation, no examples."""
    pass
'''
        tests = extract_doctests(code)
        assert tests == []

    def test_invalid_syntax(self) -> None:
        """Test handling of invalid Python syntax."""
        code = "this is not valid python {"
        tests = extract_doctests(code)
        assert tests == []


class TestLoadCoverage:
    """Tests for load_coverage function."""

    def test_none_file(self) -> None:
        """Test with None file path."""
        assert load_coverage(None) == {}

    def test_nonexistent_file(self) -> None:
        """Test with nonexistent file."""
        assert load_coverage(Path("/nonexistent/coverage.json")) == {}

    def test_valid_coverage_file(self) -> None:
        """Test loading valid coverage JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "files": {
                        "src/hf_gtc/training/lora.py": {
                            "summary": {"percent_covered": 95.5}
                        },
                        "src/hf_gtc/hub/cards.py": {
                            "summary": {"percent_covered": 100.0}
                        },
                    }
                },
                f,
            )
            f.flush()

            coverage = load_coverage(Path(f.name))
            assert pytest.approx(coverage["training/lora.py"], rel=0.01) == 95.5
            assert pytest.approx(coverage["hub/cards.py"], rel=0.01) == 100.0

    def test_malformed_json(self) -> None:
        """Test handling malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()

            coverage = load_coverage(Path(f.name))
            assert coverage == {}


class TestExtractModule:
    """Tests for extract_module function."""

    def test_basic_module(self) -> None:
        """Test extracting basic module metadata."""
        code = '''
def foo(x: int) -> int:
    """Add one.

    >>> foo(1)
    2
    """
    return x + 1
'''
        info = extract_module("training/lora.py", code, 95.0)
        assert info.module_path == "training/lora.py"
        assert info.category == "training"
        assert info.content == code
        assert len(info.functions) == 1
        assert len(info.doctests) == 1
        assert pytest.approx(info.coverage, rel=0.01) == 95.0

    def test_module_with_syntax_error(self) -> None:
        """Test module with syntax error still captures content."""
        code = "def foo( invalid syntax {"
        info = extract_module("test/module.py", code, 0.0)
        assert info.module_path == "test/module.py"
        assert info.content == code
        assert info.functions == []

    def test_empty_module(self) -> None:
        """Test empty module."""
        code = "# Just a comment"
        info = extract_module("test/empty.py", code, 0.0)
        assert info.functions == []
        assert info.doctests == []


class TestModulesToArrow:
    """Tests for modules_to_arrow function."""

    def test_single_module(self) -> None:
        """Test converting single module to Arrow."""
        info = ModuleInfo(
            module_path="test/module.py",
            category="test",
            content="# code",
            functions=[
                FunctionInfo(
                    name="foo",
                    signature="def foo()",
                    docstring="Doc",
                    line_number=1,
                )
            ],
            doctests=[
                DoctestInfo(
                    source="1 + 1",
                    expected="2",
                    line_number=5,
                )
            ],
            coverage=95.0,
            test_count=1,
        )
        table = modules_to_arrow([info])
        assert table.num_rows == 1
        assert table.num_columns == 7

    def test_multiple_modules(self) -> None:
        """Test converting multiple modules to Arrow."""
        modules = [
            ModuleInfo(
                module_path=f"test/module{i}.py",
                category="test",
                content=f"# code {i}",
                functions=[],
                doctests=[],
                coverage=float(i * 10),
                test_count=i,
            )
            for i in range(5)
        ]
        table = modules_to_arrow(modules)
        assert table.num_rows == 5

    def test_schema_columns(self) -> None:
        """Test Arrow table has correct schema columns."""
        info = ModuleInfo(
            module_path="test/m.py",
            category="test",
            content="",
            functions=[],
            doctests=[],
            coverage=0.0,
            test_count=0,
        )
        table = modules_to_arrow([info])

        column_names = table.column_names
        assert "module" in column_names
        assert "category" in column_names
        assert "content" in column_names
        assert "functions" in column_names
        assert "doctests" in column_names
        assert "coverage" in column_names
        assert "test_count" in column_names

    def test_empty_list(self) -> None:
        """Test converting empty module list."""
        table = modules_to_arrow([])
        assert table.num_rows == 0


class TestIterModules:
    """Tests for iter_modules function."""

    def test_finds_python_files(self) -> None:
        """Test that Python files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir)
            (src_dir / "module1.py").write_text("# code 1")
            (src_dir / "module2.py").write_text("# code 2")

            modules = list(iter_modules(src_dir))
            assert len(modules) == 2

    def test_skips_init_files(self) -> None:
        """Test that __init__.py files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir)
            (src_dir / "__init__.py").write_text("# init")
            (src_dir / "module.py").write_text("# code")

            modules = list(iter_modules(src_dir))
            assert len(modules) == 1
            assert modules[0][0] == "module.py"

    def test_recursive_search(self) -> None:
        """Test recursive file search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir)
            subdir = src_dir / "subdir"
            subdir.mkdir()
            (src_dir / "top.py").write_text("# top")
            (subdir / "nested.py").write_text("# nested")

            modules = list(iter_modules(src_dir))
            paths = [m[0] for m in modules]
            assert any("top.py" in p for p in paths)
            assert any("nested.py" in p for p in paths)

    def test_skips_pycache(self) -> None:
        """Test that __pycache__ directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir)
            pycache = src_dir / "__pycache__"
            pycache.mkdir()
            (pycache / "module.cpython-311.pyc").write_text("# bytecode")
            (src_dir / "real.py").write_text("# real code")

            modules = list(iter_modules(src_dir))
            assert len(modules) == 1
            assert modules[0][0] == "real.py"


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_function_info_creation(self) -> None:
        """Test FunctionInfo dataclass creation."""
        info = FunctionInfo(
            name="test",
            signature="def test()",
            docstring="Test function.",
            line_number=10,
        )
        assert info.name == "test"
        assert info.signature == "def test()"
        assert info.docstring == "Test function."
        assert info.line_number == 10

    def test_doctest_info_creation(self) -> None:
        """Test DoctestInfo dataclass creation."""
        info = DoctestInfo(
            source="1 + 1",
            expected="2",
            line_number=5,
        )
        assert info.source == "1 + 1"
        assert info.expected == "2"
        assert info.line_number == 5

    def test_module_info_defaults(self) -> None:
        """Test ModuleInfo default values."""
        info = ModuleInfo(
            module_path="test.py",
            category="test",
            content="",
        )
        assert info.functions == []
        assert info.doctests == []
        assert pytest.approx(info.coverage, abs=0.01) == 0.0
        assert info.test_count == 0
