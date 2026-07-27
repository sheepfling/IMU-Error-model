from pathlib import Path

from scripts.check_markdown import link_target_error
from scripts.format_markdown import format_markdown_files, format_markdown_text


def test_markdown_formatter_normalizes_spacing_and_final_newline() -> None:
    source = "# Heading\ntext\n\n\n- item\n\n\n"

    assert format_markdown_text(source) == "# Heading\n\ntext\n\n- item\n"
####

def test_markdown_formatter_preserves_fenced_code_content() -> None:
    source = "# Example\n\n```python  \n# keep this line  \nvalue = 1\n```\n"

    assert format_markdown_text(source) == "# Example\n\n```python\n# keep this line  \nvalue = 1\n```\n"
####


def test_markdown_formatter_preserves_windows_newlines(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_bytes(b"# Heading\r\ntext\r\n\r\n\r\n- item\r\n")

    assert format_markdown_files([source]) == [source]
    assert source.read_bytes() == b"# Heading\r\n\r\ntext\r\n\r\n- item\r\n"
####


def test_markdown_links_use_portable_forward_slashes() -> None:
    assert link_target_error("docs\\guide.md") == "link uses platform-specific backslash separators"
    assert link_target_error("docs/guide.md") is None
####
