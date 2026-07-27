from pathlib import Path

from scripts.check_scope_markers import check_scope_markers, fix_scope_markers


def test_scope_marker_checker_accepts_nested_function_and_class_markers(tmp_path: Path) -> None:
    source = tmp_path / "marked.py"
    source.write_text(
        "class Example:\n"
        "    def method(self) -> None:\n"
        "        def nested() -> None:\n"
        "            pass\n"
        "        ####\n"
        "    ####\n"
        "####\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == []
####

def test_scope_marker_checker_reports_missing_marker(tmp_path: Path) -> None:
    source = tmp_path / "unmarked.py"
    source.write_text("def example() -> None:\n    pass\n", encoding="utf-8")

    diagnostics = check_scope_markers([source])

    assert diagnostics == [f"{source}:1: function 'example' must close with ####"]
####

def test_scope_marker_checker_ignores_docstring_and_ellipsis_stubs(tmp_path: Path) -> None:
    source = tmp_path / "stubs.py"
    source.write_text(
        "def ellipsis_stub() -> None: ...\n"
        "def docstring_stub() -> None:\n"
        "    \"\"\"Documented stub.\"\"\"\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == []
####


def test_scope_marker_checker_closes_if_elif_else_chain_once(tmp_path: Path) -> None:
    source = tmp_path / "marked.py"
    source.write_text(
        "def example(value: int) -> int:\n"
        "    if value > 0:\n"
        "        return 1\n"
        "    elif value < 0:\n"
        "        return -1\n"
        "    else:\n"
        "        return 0\n"
        "    ####\n"
        "####\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == []
####


def test_scope_marker_checker_requires_loop_and_with_markers(tmp_path: Path) -> None:
    source = tmp_path / "marked.py"
    source.write_text(
        "def example() -> None:\n"
        "    for value in range(2):\n"
        "        print(value)\n"
        "    ####\n"
        "    with open('file') as stream:\n"
        "        print(stream.read())\n"
        "    ####\n"
        "####\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == []
####


def test_scope_marker_checker_closes_for_else_and_try_chains_once(tmp_path: Path) -> None:
    source = tmp_path / "marked.py"
    source.write_text(
        "def example(values: list[int]) -> None:\n"
        "    for value in values:\n"
        "        print(value)\n"
        "    else:\n"
        "        print('empty')\n"
        "    ####\n"
        "    try:\n"
        "        print('try')\n"
        "    except OSError:\n"
        "        print('except')\n"
        "    else:\n"
        "        print('else')\n"
        "    finally:\n"
        "        print('finally')\n"
        "    ####\n"
        "    try:\n"
        "        print('try-star')\n"
        "    except* OSError:\n"
        "        print('except-star')\n"
        "    ####\n"
        "####\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == []
####


def test_scope_marker_checker_reports_unexpected_marker(tmp_path: Path) -> None:
    source = tmp_path / "misplaced.py"
    source.write_text("####\n\n", encoding="utf-8")

    assert check_scope_markers([source]) == [f"{source}:1: unexpected #### marker"]
####


def test_scope_marker_checker_rejects_marker_after_stub_function(tmp_path: Path) -> None:
    source = tmp_path / "stub.py"
    source.write_text(
        "def documented() -> None:\n"
        "    \"\"\"Documentation only.\"\"\"\n"
        "    ####\n",
        encoding="utf-8",
    )

    assert check_scope_markers([source]) == [f"{source}:3: unexpected #### marker"]
####


def test_scope_marker_fixer_removes_unexpected_markers(tmp_path: Path) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        "####\n"
        "def documented() -> None:\n"
        "    \"\"\"Documentation only.\"\"\"\n"
        "    ####\n"
        "def example() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert fix_scope_markers([source]) == [source]
    assert source.read_text(encoding="utf-8") == (
        "def documented() -> None:\n"
        "    \"\"\"Documentation only.\"\"\"\n"
        "def example() -> None:\n"
        "    pass\n"
        "####\n"
    )
    assert check_scope_markers([source]) == []
####


def test_scope_marker_fixer_closes_if_elif_else_chain_after_final_branch(
        tmp_path: Path,
) -> None:
    source = tmp_path / "unmarked.py"
    source.write_text(
        "def example(value: int) -> int:\n"
        "    if value > 0: return 1\n"
        "    elif value < 0: return -1\n"
        "    else: return 0\n"
        "####\n",
        encoding="utf-8",
    )

    assert fix_scope_markers([source]) == [source]
    assert check_scope_markers([source]) == []
    assert source.read_text(encoding="utf-8") == (
        "def example(value: int) -> int:\n"
        "    if value > 0: return 1\n"
        "    elif value < 0: return -1\n"
        "    else: return 0\n"
        "    ####\n"
        "####\n"
    )
####


def test_scope_marker_fixer_inserts_markers_for_nested_scopes(tmp_path: Path) -> None:
    source = tmp_path / "unmarked.py"
    source.write_text(
        "class Example:\n"
        "    def method(self) -> None:\n"
        "        def nested() -> None:\n"
        "            pass\n"
        "        pass\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert fix_scope_markers([source]) == [source]
    assert check_scope_markers([source]) == []
    assert source.read_text(encoding="utf-8") == (
        "class Example:\n"
        "    def method(self) -> None:\n"
        "        def nested() -> None:\n"
        "            pass\n"
        "        ####\n"
        "        pass\n"
        "    ####\n"
        "    pass\n"
        "####\n"
    )
####


def test_scope_marker_fixer_inserts_markers_after_with_and_try_chains(
        tmp_path: Path,
) -> None:
    source = tmp_path / "unmarked.py"
    source.write_text(
        "def example() -> None:\n"
        "    for value in range(2):\n"
        "        print(value)\n"
        "    else:\n"
        "        print('empty')\n"
        "    with open('file') as stream:\n"
        "        print(stream.read())\n"
        "    try:\n"
        "        print('try')\n"
        "    except OSError:\n"
        "        print('except')\n"
        "    else:\n"
        "        print('else')\n"
        "    finally:\n"
        "        print('finally')\n",
        encoding="utf-8",
    )

    assert fix_scope_markers([source]) == [source]
    assert check_scope_markers([source]) == []
    assert source.read_text(encoding="utf-8") == (
        "def example() -> None:\n"
        "    for value in range(2):\n"
        "        print(value)\n"
        "    else:\n"
        "        print('empty')\n"
        "    ####\n"
        "    with open('file') as stream:\n"
        "        print(stream.read())\n"
        "    ####\n"
        "    try:\n"
        "        print('try')\n"
        "    except OSError:\n"
        "        print('except')\n"
        "    else:\n"
        "        print('else')\n"
        "    finally:\n"
        "        print('finally')\n"
        "    ####\n"
        "####\n"
    )
####

def test_scope_marker_fixer_preserves_windows_newlines(tmp_path: Path) -> None:
    source = tmp_path / "unmarked.py"
    source.write_bytes(b"def example() -> None:\r\n    pass\r\n")

    assert fix_scope_markers([source]) == [source]
    assert source.read_bytes() == b"def example() -> None:\r\n    pass\r\n####\r\n"
####


def test_scope_marker_fixer_moves_marker_before_blank_separator(tmp_path: Path) -> None:
    source = tmp_path / "spaced.py"
    source.write_text(
        "def example() -> None:\n"
        "    pass\n"
        "\n"
        "####\n",
        encoding="utf-8",
    )

    assert fix_scope_markers([source]) == [source]
    assert source.read_text(encoding="utf-8") == (
        "def example() -> None:\n"
        "    pass\n"
        "####\n"
        "\n"
    )
####
