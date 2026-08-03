from __future__ import annotations

import os
import re
import shutil
import sys

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX
    termios = None
    tty = None

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def _wants_ansi() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(sys.stdout.isatty() and sys.stdout.encoding)
    except (ValueError, AttributeError):
        return False


USE_ANSI = _wants_ansi()


def _uses_utf8() -> bool:
    locale = (os.environ.get("LC_ALL") or os.environ.get("LC_CTYPE") or "").lower()
    if "utf-8" in locale or "utf8" in locale:
        return True
    encoding = getattr(sys.stdout, "encoding", "") or ""
    return "utf" in encoding.lower()


USE_UTF8 = _uses_utf8()

_BOX = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "h": "─",
    "v": "│",
    "lt": "├",
    "rt": "┤",
} if USE_UTF8 else {
    "tl": "+",
    "tr": "+",
    "bl": "+",
    "br": "+",
    "h": "-",
    "v": "|",
    "lt": "+",
    "rt": "+",
}

def color(code: int, text: str) -> str:
    if not USE_ANSI:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def bold(text: str) -> str:
    return color(1, text)


def dim(text: str) -> str:
    return color(2, text)


def red(text: str) -> str:
    return color(31, text)


def green(text: str) -> str:
    return color(32, text)


def yellow(text: str) -> str:
    return color(33, text)


def blue(text: str) -> str:
    return color(34, text)


def magenta(text: str) -> str:
    return color(35, str(text))


def cyan(text: str) -> str:
    return color(36, str(text))


def pass_mark() -> str:
    return green("✓") if USE_UTF8 else green("OK")


def fail_mark() -> str:
    return red("✗") if USE_UTF8 else red("FAIL")


def warn_mark() -> str:
    return yellow("!") if USE_UTF8 else yellow("WARN")


def info_mark() -> str:
    return cyan("i") if USE_UTF8 else cyan("i")


def bullet() -> str:
    return "•" if USE_UTF8 else "*"


def visible_length(text: str) -> int:
    return len(ANSI_PATTERN.sub("", text))


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - visible_length(text))


def width() -> int:
    try:
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    except (OSError, ValueError):
        columns = 80
    return max(40, min(columns, 120))


def _wrap(line: str, max_width: int) -> list[str]:
    if visible_length(line) <= max_width:
        return [line]
    words = line.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if visible_length(word) > max_width:
            if current:
                lines.append(current)
            for start in range(0, len(word), max_width):
                lines.append(word[start : start + max_width])
            current = ""
            continue
        candidate = f"{current} {word}".strip()
        if visible_length(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _frame(title: str | None, body: str, *, title_color=None) -> str:
    terminal = width()
    h = _BOX["h"]
    style = title_color or bold
    if title:
        title_text = f" {style(title)} "
        mid = terminal - len(_BOX["tl"]) - visible_length(title_text) - len(_BOX["tr"])
        if mid > 0:
            header = f"{_BOX['tl']}{h * mid}{title_text}{_BOX['tr']}"
        else:
            header = f"{_BOX['tl']}{h * max(0, terminal - 2)}{_BOX['tr']}"
    else:
        header = f"{_BOX['tl']}{h * max(0, terminal - 2)}{_BOX['tr']}"
    if not body:
        body_lines: list[str] = []
    else:
        body_lines = body.splitlines()
    inner = max(1, terminal - 2)
    lines: list[str] = []
    for line in body_lines:
        indent_match = re.match(r"^[ \t]*", line)
        indent = indent_match.group(0) if indent_match else ""
        content = line[len(indent):]
        wrapped = _wrap(content, inner - len(indent))
        for index, part in enumerate(wrapped):
            text = indent if index > 0 else ""
            lines.append(
                f"{_BOX['v']}{text}{part}{' ' * max(0, inner - len(text) - visible_length(part))}{_BOX['v']}"
            )
    footer = f"{_BOX['bl']}{h * max(0, terminal - 2)}{_BOX['br']}"
    return "\n".join([header, *lines, footer])


def box(title: str | None = None, body: str = "", *, title_color=None) -> str:
    return _frame(title, body, title_color=title_color)


def hr(title: str | None = None) -> str:
    terminal = width()
    h = _BOX["h"]
    if title:
        rendered = bold(title)
        middle = f" {rendered} "
        left = max(1, (terminal - visible_length(middle)) // 2)
        right = max(1, terminal - visible_length(middle) - left)
        return f"{h * left}{middle}{h * right}"
    return h * max(4, terminal)


def kv(label: str, value: str, *, value_color=None) -> str:
    display = (value_color or (lambda text: text))(value)
    flattened = " ".join(display.splitlines())
    return f"  {cyan(label + ':') if USE_ANSI else label + ':'} {flattened}"


def _truncate(text: str, limit: int) -> str:
    if visible_length(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _truncate_segments(segments: list[tuple], limit: int) -> str:
    """Join styled (style, text) segments, truncating across styles safely."""
    total = sum(len(text) for _style, text in segments)
    if total <= limit:
        return "".join(style(text) if style else text for style, text in segments)
    parts: list[str] = []
    remaining = limit - 1
    for style, text in segments:
        if remaining <= 0:
            break
        take = min(len(text), remaining)
        parts.append(style(text[:take]) if style else text[:take])
        remaining -= take
    parts.append("…")
    return "".join(parts)


def table(
    headers: list[str],
    rows: list[list[str]],
    *,
    align_right: set[int] | None = None,
    max_widths: list[int] | None = None,
) -> str:
    if not rows:
        return dim("  no rows")
    align_right = align_right or set()
    max_widths = max_widths or []
    widths = [visible_length(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            limit = max_widths[index] if index < len(max_widths) else None
            rendered = _truncate(cell, limit) if limit else cell
            widths[index] = max(widths[index], visible_length(rendered))
    def render_row(row: list[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            limit = max_widths[index] if index < len(max_widths) else None
            rendered = _truncate(cell, limit) if limit else cell
            if index in align_right:
                cells.append(" " * (widths[index] - visible_length(rendered)) + rendered)
            else:
                cells.append(pad(rendered, widths[index]))
        return "  " + "  ".join(cells)
    h = "  " + "  ".join(
        pad(bold(cell), widths[index]) for index, cell in enumerate(headers)
    )
    rule = "  " + "-" * sum(widths) + "-" * (2 * (len(widths) - 1))
    body = "\n".join(render_row(row) for row in rows)
    return "\n".join([h, rule, body])


def ask(
    label: str,
    *,
    default: str | None = None,
    validate=None,
    hint: str | None = None,
) -> str:
    suffix = ""
    if default is not None and str(default) != "":
        suffix = f" [{dim(str(default)) if USE_ANSI else str(default)}]"
    while True:
        try:
            raw = input(f"{cyan(bold(label)) if USE_ANSI else label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        value = raw or (default or "")
        if validate is not None:
            error = validate(value)
            if error:
                print(f"  {fail_mark()} {error}")
                if hint:
                    print(f"  {dim(hint)}")
                continue
        return value


def confirm(label: str, *, default: bool = False) -> bool:
    choice = "Y/n" if default else "y/N"
    suffix = f" [{dim(choice) if USE_ANSI else choice}]"
    while True:
        try:
            raw = input(f"{cyan(bold(label)) if USE_ANSI else label}{suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print(f"  {fail_mark()} Answer yes or no.")


def pause(label: str = "Press Enter to return to the menu") -> None:
    print()
    print(dim(label))
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print()


def _read_key(fd: int) -> str:
    first = os.read(fd, 1)
    if not first:
        return "QUIT"
    if first == b"\x1b":
        second = os.read(fd, 1)
        if second in (b"[", b"O"):
            third = os.read(fd, 1)
            return {
                "A": "UP",
                "B": "DOWN",
                "C": "RIGHT",
                "D": "LEFT",
                "H": "HOME",
                "F": "END",
            }.get(third.decode("ascii", "ignore"), "UNKNOWN")
        return "ESC"
    character = first.decode("ascii", "ignore")
    if character in ("\r", "\n"):
        return "ENTER"
    if character in ("\x03", "\x04"):
        return "QUIT"
    if character in ("q", "Q"):
        return "QUIT"
    return character


def key_hint() -> str:
    if USE_UTF8:
        return "Use ↑/↓ to move, Enter to select, q to quit"
    return "Use up/down to move, Enter to select, q to quit"


def _menu_item(label: str, description: str, *, selected: bool) -> str:
    marker = "▸" if USE_UTF8 else ">"
    rail = "│" if USE_UTF8 else "|"
    segments: list[tuple] = []
    if selected:
        if USE_ANSI:
            segments.append((cyan, marker))
            segments.append((None, f" {rail} "))
        else:
            segments.append((None, f"{marker} {rail} "))
    else:
        segments.append((None, f"  {rail} "))
    segments.append((bold if (selected and USE_ANSI) else None, label))
    if description:
        segments.append((dim if USE_ANSI else None, f"  {description}"))
    return _truncate_segments(segments, width())


def _menu_window_end(first: int, available: int, total: int) -> int:
    """Last visible index (exclusive) starting at `first` within `available` rows."""
    if total <= 0:
        return 0

    def fit(remaining: int, first: int) -> int:
        end = first
        while end < total and remaining >= 1:
            remaining -= 1
            end += 1
        return end

    end = fit(available - (1 if first > 0 else 0), first)
    if end < total:
        end = fit(available - (1 if first > 0 else 0) - 1, first)
    return end


def select(
    options: list[tuple[str, str]],
    *,
    title: str | None = None,
    header: str | None = None,
    hint: str | None = None,
    initial: int = 0,
) -> int:
    """Arrow-key menu with scrolling. Returns the chosen index, or -1 when cancelled."""
    if termios is None or tty is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return -1
    if not options:
        return -1
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    cursor = max(0, min(initial, len(options) - 1))
    if hint is None:
        hint = key_hint()

    def rows() -> int:
        try:
            size = shutil.get_terminal_size(fallback=(80, 24))
        except (OSError, ValueError):
            size = os.terminal_size((80, 24))
        return max(6, size.lines)

    def available() -> int:
        chrome = 1 if title else 0
        if header:
            chrome += 1 + len(header.splitlines())
        chrome += 1  # blank line before the options
        return rows() - chrome - 1  # one-row bottom margin

    def render(first: int) -> list[str]:
        end = _menu_window_end(first, available(), len(options))
        lines: list[str] = []
        if title:
            lines.append(bold(title))
        if header:
            lines.append("")
            lines.extend(header.splitlines())
        lines.append("")
        if first > 0:
            lines.append(dim(f"↑ {first} more"))
        for index in range(first, end):
            label, description = options[index]
            lines.append(_menu_item(label, description, selected=index == cursor))
        if end < len(options):
            lines.append(dim(f"↓ {len(options) - end} more"))
        return lines

    def recenter(first: int) -> int:
        while True:
            end = _menu_window_end(first, available(), len(options))
            if cursor < first:
                first = cursor
            elif cursor >= end:
                first += 1
            else:
                return first

    printed = 0

    def draw(first: int) -> None:
        nonlocal printed
        if printed == 0:
            sys.stdout.write("\x1b[2J\x1b[H")
        else:
            sys.stdout.write(f"\x1b[{printed}A\x1b[J")
        output: list[str] = []
        if hint:
            output.append(dim(hint))
        output.extend(render(first))
        for line in output:
            sys.stdout.write(line + "\r\n")
        printed = len(output)
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        first = 0
        draw(first)
        while True:
            key = _read_key(fd)
            if key == "UP":
                cursor = max(0, cursor - 1)
            elif key == "DOWN":
                cursor = min(len(options) - 1, cursor + 1)
            elif key in ("HOME", "LEFT"):
                cursor = 0
            elif key in ("END", "RIGHT"):
                cursor = len(options) - 1
            elif key == "ENTER":
                return cursor
            elif key == "QUIT":
                return -1
            else:
                continue
            first = recenter(first)
            draw(first)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\r\n")
        sys.stdout.flush()
