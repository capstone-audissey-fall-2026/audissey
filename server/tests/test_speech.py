from __future__ import annotations

import unittest

from pydantic import ValidationError

from v2a_inspect_server.inference.speech import _speech_speed
from v2a_inspect_server.models.speech import KokoroGenerateSpeechRequest


class SpeechInferenceTest(unittest.TestCase):
    def test_speed_is_bounded(self) -> None:
        self.assertEqual(_speech_speed("hello", None), 1.0)
        self.assertEqual(_speech_speed("x", 10.0), 0.5)
        self.assertEqual(_speech_speed("x" * 120, 1.0), 2.0)

    def test_request_rejects_blank_text(self) -> None:
        with self.assertRaises(ValidationError):
            KokoroGenerateSpeechRequest(text="   ")


if __name__ == "__main__":
    unittest.main()
