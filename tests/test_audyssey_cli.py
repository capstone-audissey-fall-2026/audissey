import unittest
from click import unstyle
from typer.testing import CliRunner
from audyssey.cli import app


class CliTests(unittest.TestCase):
    def test_cli_help(self):
        result = CliRunner().invoke(app, ["serve", "--help"], color=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--port", unstyle(result.output))

    def test_tool_server_help(self):
        result = CliRunner().invoke(app, ["tool-server", "--help"], color=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--tool", unstyle(result.output))
        self.assertIn("--impl", unstyle(result.output))

    def test_tool_server_refuses_public_host_without_token(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN=""), patch("audyssey.cli.uvicorn.run") as run:
            result = CliRunner().invoke(app, ["tool-server", "--tool", "separation", "--host", "0.0.0.0"])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("AUDYSSEY_TOOL_TOKEN is required when binding to a non-loopback host", unstyle(result.output))
        run.assert_not_called()

    def test_tool_server_accepts_loopback_or_authenticated_host(self):
        import os
        from unittest.mock import patch
        for host, token in [("127.0.0.1", ""), ("localhost", ""), ("::1", ""), ("0.0.0.0", "t")]:
            with self.subTest(host=host), patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN=token), patch("audyssey.cli.uvicorn.run") as run:
                result = CliRunner().invoke(app, ["tool-server", "--tool", "filtering", "--host", host])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(run.call_args.kwargs["host"], host)

    def test_tool_server_accepts_repeated_tool_option(self):
        import os
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN='t'), patch('audyssey.cli.uvicorn.run') as run:
            result = CliRunner().invoke(app, ['tool-server', '--tool', 'separation', '--tool', 'filtering', '--host', '0.0.0.0'])
        self.assertEqual(result.exit_code, 0, result.output)
        with TestClient(run.call_args.args[0]) as client:
            self.assertEqual(client.get('/health').json(), {'tools': ['separation', 'filtering'], 'impl': 'stub'})
            self.assertEqual({r.path for r in client.app.routes} & {'/separate', '/judge'}, {'/separate', '/judge'})

    def test_tool_server_requires_a_tool(self):
        result = CliRunner().invoke(app, ['tool-server'])
        self.assertEqual(result.exit_code, 2)

    def test_tool_server_rejects_unknown_tool(self):
        result = CliRunner().invoke(app, ['tool-server', '--tool', 'unknown'])
        self.assertEqual(result.exit_code, 2)

    def test_tool_server_accepts_grounding(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        with patch('audyssey.cli.uvicorn.run') as run:
            result = CliRunner().invoke(app, ['tool-server', '--tool', 'grounding'])
        self.assertEqual(result.exit_code, 0, result.output)
        with TestClient(run.call_args.args[0]) as client:
            self.assertEqual(client.get('/health').json()['tools'], ['grounding'])
