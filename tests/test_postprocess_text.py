"""Tests for _postprocess_text — the pure text pipeline in pedal_dictation."""

import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Mock every hardware/GUI import so the module loads in a headless environment
# ---------------------------------------------------------------------------
for _mod in [
    "sounddevice", "pyperclip", "pystray",
    "pynput", "pynput.keyboard",
    "PIL", "PIL.Image", "PIL.ImageDraw",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# pynput.keyboard needs Key and Controller attributes
import pynput.keyboard as _pk
_pk.Key = object()
_pk.Controller = lambda: None
_pk.Listener = lambda **_: None
_pk.KeyCode = type("KeyCode", (), {"from_vk": staticmethod(lambda v: None)})()

sys.path.insert(0, ".")
import pedal_dictation as pd  # noqa: E402  (must come after mocks)


class TestPostprocessText(unittest.TestCase):

    def test_trailing_space_on_normal_text(self):
        text, send_enter = pd._postprocess_text("Hello world")
        self.assertTrue(text.endswith(" "), f"Expected trailing space, got {repr(text)}")
        self.assertFalse(send_enter)

    def test_trailing_space_preserved_through_sentence(self):
        text, send_enter = pd._postprocess_text("The meeting is at three pm.")
        self.assertTrue(text.endswith(" "), f"Expected trailing space, got {repr(text)}")

    def test_consecutive_recordings_have_separator(self):
        t1, _ = pd._postprocess_text("First sentence.")
        t2, _ = pd._postprocess_text("Second sentence.")
        combined = t1 + t2
        # There must be whitespace between the two sentences
        self.assertIn(" S", combined, f"No space between recordings: {repr(combined)}")

    def test_sentence_boundary_space_fix(self):
        # Raw whisper sometimes produces "End.Start" without space — should be fixed
        text, _ = pd._postprocess_text("End.Start of next")
        self.assertIn("End. Start", text)

    def test_send_enter_command_stripped(self):
        text, send_enter = pd._postprocess_text("Submit the form. Send it.")
        self.assertTrue(send_enter)
        self.assertNotRegex(text, r'(?i)send it')
        self.assertTrue(text.endswith(" "), f"Expected trailing space, got {repr(text)}")

    def test_only_send_enter_returns_empty_text(self):
        text, send_enter = pd._postprocess_text("Send it.")
        self.assertEqual(text, "")
        self.assertTrue(send_enter)

    def test_empty_input_returns_empty(self):
        text, send_enter = pd._postprocess_text("")
        self.assertEqual(text, "")
        self.assertFalse(send_enter)


if __name__ == "__main__":
    unittest.main()
