"""Prepare LLM Markdown for Streamlit, allowing only HTML line breaks."""

from __future__ import annotations

import html
import re
import textwrap

from markdown_it import MarkdownIt

_BREAK = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
_TOKEN = re.compile(
    r"(?P<literal>\\[\\`<>]|(?<!`)(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`)|"
    r"<(?:https?://|mailto:)[^\s<>]+>|<[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}>)|"
    r"(?P<markup><!--[\s\S]*?(?:-->|$)|<![^>]*>|<\?[\s\S]*?\?>|"
    r"</?[A-Za-z][\w:-]*(?=[\s/>])(?:[^>\"']|\"[^\"]*\"|'[^']*')*>|"
    r"</?[A-Za-z][\w:-]*(?=[\s/>]|$))",
    re.IGNORECASE,
)
_MARKDOWN = MarkdownIt("commonmark").enable("table")


def _format_prose(text: str) -> str:
    def token(match: re.Match) -> str:
        tag = match.group("markup")
        if tag is None:
            return match.group()
        return "<br>" if _BREAK.fullmatch(tag) else html.escape(tag, quote=False)

    return _TOKEN.sub(token, text)


def format_answer_markdown(text: str) -> str:
    """Render <br> variants in tables/prose while preserving Markdown code verbatim.

    Use the result with st.markdown(..., unsafe_allow_html=True). Other HTML
    tags are escaped. The original answer stays unchanged in history/feedback.
    """
    # Match Streamlit's clean_text before recognising indented code. Otherwise
    # Streamlit could later dedent a protected line into live HTML.
    text = textwrap.dedent(text).strip()
    # Source maps distinguish real code blocks from indented list continuations.
    code_lines = set()
    for token in _MARKDOWN.parse(text):
        if token.type in {"fence", "code_block"} and token.map:
            code_lines.update(range(*token.map))
    result, prose = [], []

    def flush() -> None:
        result.append(_format_prose("".join(prose)))
        prose.clear()

    for index, line in enumerate(text.splitlines(keepends=True)):
        if index in code_lines:
            flush()
            result.append(line)
        else:
            prose.append(line)
    flush()
    return "".join(result)
