import os
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from audyssey.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = Settings()
        self.assertEqual(
            (settings.planner, settings.separator, settings.filter),
            ("stub", "stub", "stub"),
        )
        self.assertEqual(
            (
                settings.max_plan_rounds,
                settings.max_separation_retries,
                settings.max_duration_s,
            ),
            (2, 2, 30),
        )

    def test_env_override(self):
        with patch.dict(
            os.environ, AUDYSSEY_PLANNER="real", AUDYSSEY_MAX_PLAN_ROUNDS="3"
        ):
            settings = Settings()
        self.assertEqual((settings.planner, settings.max_plan_rounds), ("real", 3))

    def test_rejects_unknown_mode(self):
        with patch.dict(os.environ, AUDYSSEY_SEPARATOR="gpu"):
            with self.assertRaises(ValidationError):
                Settings()

    def test_remote_modes_and_urls_from_env(self):
        with patch.dict(os.environ, {
            "AUDYSSEY_SEPARATOR": "remote",
            "AUDYSSEY_SEPARATOR_URL": "http://gpu:8700",
            "AUDYSSEY_TOOL_TOKEN": "t",
            "AUDYSSEY_PLANNER_BASE_URL": "http://llm/v1",
            "AUDYSSEY_PLANNER_MODEL": "m",
        }, clear=True):
            settings = Settings()
        self.assertEqual(settings.separator, "remote")
        self.assertEqual(settings.separator_url, "http://gpu:8700")
        self.assertEqual(settings.tool_token, "t")
        self.assertEqual(settings.planner_base_url, "http://llm/v1")
        self.assertEqual(settings.planner_model, "m")
        self.assertEqual(settings.filter, "stub")
        self.assertEqual(settings.remote_timeout_s, 600)

    def test_remote_timeout_must_be_positive(self):
        with patch.dict(os.environ, AUDYSSEY_REMOTE_TIMEOUT_S="0"):
            with self.assertRaises(ValidationError):
                Settings()

    def test_sam_audio_settings_from_env(self):
        with patch.dict(os.environ, {
            "AUDYSSEY_SAM_AUDIO_DEVICE": "cpu",
            "AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES": "3",
            "AUDYSSEY_SAM_AUDIO_PREDICT_SPANS": "true",
            "AUDYSSEY_JUDGE_PASS_THRESHOLD": "0.7",
        }, clear=True):
            settings = Settings()
        self.assertEqual(settings.sam_audio_device, "cpu")
        self.assertEqual(settings.sam_audio_reranking_candidates, 3)
        self.assertTrue(settings.sam_audio_predict_spans)
        self.assertEqual(settings.judge_pass_threshold, 0.7)
        with self.assertRaises(ValidationError):
            Settings(sam_audio_reranking_candidates=0)

    def test_grounding_defaults_and_environment(self):
        defaults = Settings()
        self.assertEqual((defaults.grounder, defaults.grounding_url, defaults.sam3_model, defaults.sam3_device),
                         ('stub', '', 'facebook/sam3.1', 'cuda'))
        with patch.dict(os.environ, AUDYSSEY_GROUNDER='remote', AUDYSSEY_GROUNDING_URL='http://grounding:8700', AUDYSSEY_SAM3_DEVICE='cpu', AUDYSSEY_SAM3_MODEL='local/model'):
            settings = Settings()
        self.assertEqual((settings.grounder, settings.grounding_url, settings.sam3_device, settings.sam3_model),
                         ('remote', 'http://grounding:8700', 'cpu', 'local/model'))
        with self.assertRaises(ValidationError):
            Settings(grounder='unknown')
