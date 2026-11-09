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
