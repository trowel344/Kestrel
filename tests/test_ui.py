from kestrel import ui


def test_visible_length_strips_ansi():
    assert ui.visible_length("\x1b[31mred\x1b[0m") == 3
    assert ui.visible_length("plain") == 5


def test_pad_pads_visible_width():
    text = "\x1b[31mabc\x1b[0m"
    assert ui.visible_length(ui.pad(text, 8)) == 8


def test_width_clamped():
    value = ui.width()
    assert 40 <= value <= 120


def test_wrap_long_word():
    wrapped = ui._wrap("abcdefghij", 4)
    assert wrapped == ["abcd", "efgh", "ij"]


def test_wrap_keeps_short_line():
    assert ui._wrap("hello", 10) == ["hello"]


def test_wrap_splits_words_within_width():
    assert ui._wrap("one two three four", 8) == ["one two", "three", "four"]


def test_box_contains_frame_characters():
    box = ui.box("Title", "some body text")
    assert "\n" in box
    assert box.startswith(("┌", "+"))
    assert box.endswith(("┘", "+"))
    assert "Title" in box
    assert "some body text" in box


def test_box_empty_body():
    box = ui.box("Only title")
    assert "Only title" in box
    assert "\n" in box


def test_hr_symmetry():
    rendered = ui.hr("Section")
    assert "Section" in rendered


def test_kv_flattens_newlines():
    line = ui.kv("Score", "10\nout of 10")
    assert "\n" not in line


def test_truncate_with_ellipsis():
    assert ui._truncate("hello world", 8) == "hello w…"


def test_truncate_noop_within_limit():
    assert ui._truncate("hello", 10) == "hello"


def test_truncate_segments_respects_limit():
    segments = [(ui.bold, "abc"), (ui.dim, "def")]
    result = ui._truncate_segments(segments, 4)
    assert ui.visible_length(result) <= 4
    assert result.endswith("…")


def test_truncate_segments_short():
    segments = [(None, "abc"), (None, "def")]
    assert ui._truncate_segments(segments, 10) == "abcdef"


def test_table_basic():
    rendered = ui.table(["Name", "Size"], [["a", "1"], ["bb", "22"]])
    assert "Name" in rendered
    assert "bb" in rendered


def test_table_empty_returns_dimmed():
    assert "no rows" in ui.table(["Name"], [])


def test_table_truncates_cells():
    rendered = ui.table(["Name"], [["a very very long name"]], max_widths=[6])
    assert "…" in rendered


def test_menu_item_selected():
    item = ui._menu_item("Run", "desc", selected=True)
    assert "Run" in item
    assert "desc" in item


def test_menu_window_end_respects_available():
    # One row is reserved for the "↓ N more" indicator when not at the end.
    assert ui._menu_window_end(0, 3, 10) == 2
    assert ui._menu_window_end(0, 3, 2) == 2


def test_confirm_returns_default_on_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    assert ui.confirm("Proceed?", default=False) is False
    assert ui.confirm("Proceed?", default=True) is True


def test_ask_returns_default_on_empty(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    assert ui.ask("Name", default="bob") == "bob"


def test_select_returns_neg_one_non_tty(monkeypatch):
    class NonTty:
        def isatty(self):
            return False

        def fileno(self):
            return 0

    monkeypatch.setattr("sys.stdin", NonTty())
    monkeypatch.setattr("sys.stdout", NonTty())
    assert ui.select([("a", "d")], initial=0) == -1


def test_marks_are_strings():
    assert isinstance(ui.pass_mark(), str)
    assert isinstance(ui.fail_mark(), str)
    assert isinstance(ui.warn_mark(), str)


def test_bullet_is_string():
    assert isinstance(ui.bullet(), str)


def test_color_noop_when_no_ansi(monkeypatch):
    monkeypatch.setattr(ui, "USE_ANSI", False)
    assert ui.color(31, "x") == "x"
    assert ui.bold("x") == "x"


def test_key_hint_is_string():
    assert isinstance(ui.key_hint(), str)
