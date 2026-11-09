from __future__ import annotations

import unittest
from unittest.mock import patch

from v2a_inspect.audio_generation.client import generate_audio_for_item


class AudioRoutingTest(unittest.TestCase):
    @patch("v2a_inspect.audio_generation.client.generate_speech_kokoro")
    def test_speech_uses_spoken_text(self, generate_speech) -> None:
        generate_speech.return_value = "speech.wav"

        result = generate_audio_for_item(
            kind="speech",
            description="A calm close-mic voice",
            spoken_text="Hello there",
            out_path="speech.wav",
            duration=1.0,
            generation_model="tts",
        )

        self.assertEqual(result, "speech.wav")
        generate_speech.assert_called_once_with("Hello there", "speech.wav", 1.0, None)

    @patch("v2a_inspect.audio_generation.client.generate_speech_kokoro")
    def test_speech_falls_back_to_description(self, generate_speech) -> None:
        generate_speech.return_value = "speech.wav"

        result = generate_audio_for_item(
            kind="speech",
            description="A calm close-mic voice",
            out_path="speech.wav",
            duration=1.0,
            generation_model="t2a",
        )

        self.assertEqual(result, "speech.wav")
        generate_speech.assert_called_once_with(
            "A calm close-mic voice", "speech.wav", 1.0, None
        )

    @patch("v2a_inspect.audio_generation.client.generate_sfx_elevenlabs")
    def test_non_synced_sfx_uses_t2a(self, generate_sfx) -> None:
        generate_sfx.return_value = "sfx.wav"

        result = generate_audio_for_item(
            kind="sfx",
            description="A distant dog bark",
            out_path="sfx.wav",
            duration=1.0,
            generation_model="t2a",
        )

        self.assertEqual(result, "sfx.wav")
        generate_sfx.assert_called_once()

    @patch("v2a_inspect.audio_generation.client.generate_sfx_elevenlabs")
    def test_v2a_does_not_fall_back_to_t2a(self, generate_sfx) -> None:
        result = generate_audio_for_item(
            kind="sfx",
            description="A visible impact",
            out_path="impact.wav",
            duration=0.2,
            generation_model="v2a",
        )

        self.assertIsNone(result)
        generate_sfx.assert_not_called()

    @patch("v2a_inspect.audio_generation.client.generate_speech_kokoro")
    def test_tts_rejects_non_speech(self, generate_speech) -> None:
        result = generate_audio_for_item(
            kind="sfx",
            description="A visible impact",
            out_path="impact.wav",
            duration=0.2,
            generation_model="tts",
        )

        self.assertIsNone(result)
        generate_speech.assert_not_called()


if __name__ == "__main__":
    unittest.main()
