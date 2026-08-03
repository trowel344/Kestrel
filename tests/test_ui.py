import unittest
from unittest import mock

from kestrel import ui


class UiStyleTests(unittest.TestCase):
    def test_color_is_disabled_without_ansi(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            self.assertEqual(ui.color(31, "x"), "x")
            self.assertEqual(ui.red("x"), "x")

    def test_color_wraps_text_with_escape_codes(self):
        with mock.patch.object(ui, "USE_ANSI", True):
            self.assertEqual(ui.color(31, "x"), "\x1b[31mx\x1b[0m")

    def test_visible_length_strips_ansi(self):
        with mock.patch.object(ui, "USE_ANSI", True):
            self.assertEqual(ui.visible_length(ui.red("abc")), 3)

    def test_no_color_env_disables_ansi(self):
        with mock.patch.dict("os.environ", {"NO_COLOR": "1"}, clear=True), mock.patch.object(
            ui, "sys"
        ) as fake_sys:
            fake_sys.stdout.isatty.return_value = True
            fake_sys.stdout.encoding = "utf-8"
            self.assertFalse(ui._wants_ansi())

    def test_kv_collapses_embedded_newlines(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            rendered = ui.kv("version", "line one\nline two")
            self.assertIn("line one line two", rendered)
            self.assertNotIn("\n", rendered)


class UiBoxTests(unittest.TestCase):
    ASCII_BOX = {
        "tl": "+", "tr": "+", "bl": "+", "br": "+",
        "h": "-", "v": "|", "lt": "+", "rt": "+",
    }

    def test_box_keeps_lines_within_frame(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "USE_UTF8", False
        ), mock.patch.object(ui, "_BOX", self.ASCII_BOX), mock.patch.object(
            ui, "width", return_value=40
        ):
            rendered = ui.box("Title", "body line")
            self.assertTrue(rendered.startswith("+"))
            self.assertTrue(rendered.endswith("+"))
            self.assertIn("|body line", rendered)
            for line in rendered.splitlines():
                self.assertTrue(line.startswith("+") or line.startswith("|"))

    def test_box_wraps_long_lines(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "USE_UTF8", False
        ), mock.patch.object(ui, "_BOX", self.ASCII_BOX), mock.patch.object(
            ui, "width", return_value=30
        ):
            rendered = ui.box(None, "word " * 20)
            lines = rendered.splitlines()
            self.assertEqual(len(lines), 6)
            for line in lines:
                self.assertTrue(line.startswith("+") or line.startswith("|"))

    def test_wrap_hard_splits_overlong_words(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            pieces = ui._wrap("/" + "x" * 50, 20)
            self.assertGreater(len(pieces), 1)
            for piece in pieces:
                self.assertLessEqual(ui.visible_length(piece), 20)

    def test_hr_contains_title(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "width", return_value=40
        ):
            rendered = ui.hr("Section")
            self.assertIn("Section", rendered)
            self.assertEqual(len(rendered), 40)


class UiTableTests(unittest.TestCase):
    def test_table_aligns_columns_and_pads(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            rendered = ui.table(
                ["name", "size"],
                [["a", "1 GiB"], ["bbbb", "2 GiB"]],
                align_right={1},
            )
            lines = rendered.splitlines()
            self.assertIn("name", lines[0])
            self.assertIn("bbbb", lines[-1])

    def test_table_truncates_wide_cells(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            rendered = ui.table(
                ["name"],
                [["a" * 80]],
                max_widths=[10],
            )
            self.assertIn("a" * 9 + "…", rendered)

    def test_table_empty_returns_placeholder(self):
        with mock.patch.object(ui, "USE_ANSI", False):
            self.assertEqual(ui.table(["h"], []), "  no rows")


class UiMenuTests(unittest.TestCase):
    def test_read_key_arrow_up(self):
        with mock.patch("kestrel.ui.os.read", side_effect=[b"\x1b", b"[", b"A"]):
            self.assertEqual(ui._read_key(0), "UP")

    def test_read_key_arrow_down(self):
        with mock.patch("kestrel.ui.os.read", side_effect=[b"\x1b", b"O", b"B"]):
            self.assertEqual(ui._read_key(0), "DOWN")

    def test_read_key_enter(self):
        with mock.patch("kestrel.ui.os.read", return_value=b"\r"):
            self.assertEqual(ui._read_key(0), "ENTER")

    def test_read_key_quit(self):
        with mock.patch("kestrel.ui.os.read", return_value=b"q"):
            self.assertEqual(ui._read_key(0), "QUIT")

    def test_menu_item_marks_selected(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "USE_UTF8", False
        ):
            self.assertIn(">", ui._menu_item("Chat", "desc", selected=True))
            self.assertNotIn(">", ui._menu_item("Chat", "desc", selected=False))

    def test_menu_item_keeps_vertical_rail(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "USE_UTF8", False
        ):
            selected = ui._menu_item("Chat", "desc", selected=True)
            unselected = ui._menu_item("Chat", "desc", selected=False)
            self.assertIn("| Chat", selected)
            self.assertIn("| Chat", unselected)

    def test_menu_item_truncates_to_terminal_width(self):
        with mock.patch.object(ui, "USE_ANSI", False), mock.patch.object(
            ui, "width", return_value=40
        ):
            rendered = ui._menu_item("Chat", "x" * 100, selected=False)
            self.assertEqual(ui.visible_length(rendered), 40)
            self.assertTrue(rendered.endswith("…"))

    def test_menu_window_end_fits_without_scroll(self):
        self.assertEqual(ui._menu_window_end(0, 5, 3), 3)

    def test_menu_window_end_reserves_down_indicator(self):
        self.assertEqual(ui._menu_window_end(0, 5, 10), 4)

    def test_menu_window_end_reserves_up_and_down_indicators(self):
        self.assertEqual(ui._menu_window_end(5, 5, 10), 8)

    def test_truncate_segments_joins_when_fits(self):
        segments = [(ui.bold, "abc"), (ui.dim, "def")]
        with mock.patch.object(ui, "USE_ANSI", True):
            self.assertEqual(ui._truncate_segments(segments, 20), "\x1b[1mabc\x1b[0m\x1b[2mdef\x1b[0m")

    def test_truncate_segments_keeps_styles_when_trimming(self):
        with mock.patch.object(ui, "USE_ANSI", True):
            rendered = ui._truncate_segments([(ui.bold, "abcdef"), (ui.dim, "gh")], 5)
            self.assertEqual(rendered, "\x1b[1mabcd\x1b[0m…")

    def test_key_hint_uses_arrows_with_utf8(self):
        with mock.patch.object(ui, "USE_UTF8", True):
            self.assertIn("↑/↓", ui.key_hint())

    def test_key_hint_uses_ascii_without_utf8(self):
        with mock.patch.object(ui, "USE_UTF8", False):
            self.assertNotIn("↑", ui.key_hint())

    def test_select_returns_minus_one_without_tty(self):
        with mock.patch("kestrel.ui.sys") as fake_sys:
            fake_sys.stdin.isatty.return_value = False
            fake_sys.stdout.isatty.return_value = False
            self.assertEqual(ui.select([("a", "")]), -1)

    def test_select_returns_minus_one_for_empty_options(self):
        with mock.patch("kestrel.ui.sys") as fake_sys:
            fake_sys.stdin.isatty.return_value = True
            fake_sys.stdout.isatty.return_value = True
            self.assertEqual(ui.select([]), -1)


if __name__ == "__main__":
    unittest.main()
