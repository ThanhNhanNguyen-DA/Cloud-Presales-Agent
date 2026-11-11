"""Regression coverage for HTML line breaks in LLM Markdown tables."""

from html.parser import HTMLParser

import pytest
from markdown_it import MarkdownIt

from ui.answer_rendering import format_answer_markdown


@pytest.mark.parametrize("tag", ["<br>", "<br/>", "<br />", "<BR>", "<BR />"])
def test_table_breaks_render_as_breaks_without_splitting_the_table(tag):
    answer = f"| Dịch vụ | Mô tả |\n| --- | --- |\n| RDS | **Triển khai nhanh**{tag}• Sao lưu tự động |"
    prepared = format_answer_markdown(answer)
    rendered = MarkdownIt("commonmark").enable("table").render(prepared)
    assert rendered.count("<table>") == 1 and rendered.count("<td>") == 2
    assert "<strong>Triển khai nhanh</strong><br>• Sao lưu tự động" in rendered
    assert "&lt;br" not in rendered


@pytest.mark.parametrize("literal", [
    "Ví dụ: `<br />` hoặc `<img src=x>`.",
    "Ví dụ: ``<br> và `backtick` bên trong``.",
    "```html\n<br/>\n<img src=x>\n```",
    "````markdown\n```html\n<br>\n```\n````",
    "~~~html\n<br />\n~~~",
    "```html\n<br>\n<img src=x>",  # an incomplete streamed code block
    "Đoạn code:\n\n    <br/>\n    <img src=x>",
    r"Giữ thẻ nguyên văn: \<br>.",
])
def test_code_and_escaped_tags_stay_literal(literal):
    assert format_answer_markdown(literal) == literal


def test_regular_markdown_links_math_and_entities_are_preserved():
    answer = (
        "> **Ghi chú**\n\n- [Nguồn](https://example.com?a=1&b=2)\n"
        "- <https://example.com> và <sales@example.com>\n\n"
        "Chi phí $x < y$, RAM > 4 GB &amp; CPU."
    )
    assert format_answer_markdown(answer) == answer


def test_other_html_is_displayed_as_text_not_enabled_with_breaks():
    answer = '<style>body{display:none}</style><br><img src=x onerror="alert(1)">'
    prepared = format_answer_markdown(answer)
    rendered = MarkdownIt("commonmark").render(prepared)

    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.names = []

        def handle_starttag(self, tag, attrs):
            self.names.append(tag)

    tags = Tags()
    tags.feed(rendered)
    assert tags.names == ["p", "br"]
    assert "&lt;style&gt;" in prepared and "&lt;img" in prepared


def test_common_indentation_cannot_bypass_html_filtering():
    prepared = format_answer_markdown('    <img src=x onerror="alert(1)">\n    Text<br/>More')
    assert "<img" not in prepared and "Text<br>More" in prepared


@pytest.mark.parametrize("answer", [
    '- Mục mô tả\n    <img src=x onerror="alert(1)">',
    '<img src="`x`" onerror="alert(1)">',
    '<div\n```html\n<img src=x>\n```\n>',
])
def test_non_code_html_is_escaped_in_lists_and_around_backticks(answer):
    prepared = format_answer_markdown(answer)
    assert "<img" not in prepared and "&lt;img" in prepared
