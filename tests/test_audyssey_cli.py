import unittest
from click import unstyle
from typer.testing import CliRunner
from audyssey.cli import app


class CliTests(unittest.TestCase):
    def test_cli_help(self):
        result = CliRunner().invoke(app, ["serve", "--help"], color=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--port", unstyle(result.output))
