#!/usr/bin/env python3
"""Lint markdown posts for unescaped literal `$` that get rendered as LaTeX.

The blog renders markdown via remark-math + rehype-katex, so a pair of `$`
on the same paragraph is interpreted as inline math. Currency like `$1
million ... $25,000 in API credits` therefore renders as one mangled math
expression.

The fix is to write `\\$` in markdown when the `$` is not math.

This linter flags only the currency-shaped pattern (a `$` immediately
followed by a digit and then a non-math character such as whitespace, a
comma, or a period) and ignores legitimate inline math like `$1/\\sqrt{d}$`
or `$10^6$`.

Opt-out: add `<!-- lint-allow-dollar -->` on the same line.

Exit codes
----------
0 - no violations
1 - one or more violations (paths printed to stderr in `file:line:col` form)

Usage
-----
    python scripts/lint_markdown_dollars.py [PATH ...]

If no paths are given, scans `posts/**/*.md` and `posts/**/*.mdx` under the
repo root (auto-detected as the script's parent's parent).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Currency-shaped: `$` followed immediately by a digit, optional digits /
# commas / periods, then a non-math character (whitespace, end of line, or a
# letter). Math expressions never have a plain space inside; they use
# `\space` or other LaTeX commands.
CURRENCY_PATTERN = re.compile(r"(?<!\\)\$\d[\d,.]*(?=\s|[a-zA-Z]|$)")

OPT_OUT_MARKER = "<!-- lint-allow-dollar -->"

FRONTMATTER_FENCE = "---"
CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def iter_lintable_lines(text: str):
    """Yield (line_number, line) skipping frontmatter and fenced code blocks."""
    lines = text.splitlines()
    i = 0
    n = len(lines)

    # Skip leading YAML frontmatter.
    if n > 0 and lines[0].strip() == FRONTMATTER_FENCE:
        i = 1
        while i < n and lines[i].strip() != FRONTMATTER_FENCE:
            i += 1
        i += 1  # past closing ---

    in_code = False
    while i < n:
        line = lines[i]
        if CODE_FENCE_RE.match(line):
            in_code = not in_code
            i += 1
            continue
        if not in_code:
            yield i + 1, line
        i += 1


def lint_file(path: Path) -> list[str]:
    """Return a list of human-readable violation strings for one file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: could not read ({exc})"]

    violations: list[str] = []
    for lineno, line in iter_lintable_lines(text):
        if OPT_OUT_MARKER in line:
            continue
        for match in CURRENCY_PATTERN.finditer(line):
            col = match.start() + 1
            snippet = line[max(0, col - 5) : col + 25].rstrip()
            violations.append(
                f"{path}:{lineno}:{col}: literal '$' before digit "
                f"(use '\\$' to avoid LaTeX rendering): ...{snippet}..."
            )
    return violations


def discover_default_paths(repo_root: Path) -> list[Path]:
    posts = repo_root / "posts"
    if not posts.exists():
        return []
    return sorted(
        list(posts.rglob("*.md")) + list(posts.rglob("*.mdx")),
    )


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    args = [Path(a) for a in argv[1:]]
    if not args:
        args = discover_default_paths(repo_root)

    violations: list[str] = []
    for path in args:
        if path.is_dir():
            for sub in sorted(list(path.rglob("*.md")) + list(path.rglob("*.mdx"))):
                violations.extend(lint_file(sub))
        elif path.suffix in {".md", ".mdx"}:
            violations.extend(lint_file(path))

    if violations:
        print("Unescaped literal '$' before digit (would render as LaTeX):", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        print(
            "\nFix by writing '\\$' (e.g. '\\$25,000'). "
            "If the '$' is intentional math, add '<!-- lint-allow-dollar -->' on the same line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
