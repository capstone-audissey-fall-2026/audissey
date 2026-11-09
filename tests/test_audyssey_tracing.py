import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langsmith import Client
from langsmith.run_helpers import is_traceable_function, tracing_context

from audyssey import graph
from audyssey.contracts import Drop
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import load_tools


class TracingTests(unittest.TestCase):
    def test_tools_and_command_are_traceable(self):
        for mode in ['stub', 'real', 'remote']:
            settings = Settings(separator=mode, filter=mode, planner='real' if mode == 'real' else 'stub', separator_url='http://gpu', filter_url='http://gpu')
            tools = load_tools(settings)
            for function in [tools.plan, tools.validate_plan, tools.separate, tools.judge, graph.handle_command]:
                with self.subTest(mode=mode, function=function.__name__):
                    self.assertTrue(is_traceable_function(function))

    def test_trace_inputs_exclude_job_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(work_dir=Path(tmp), tool_token='secret-tool-token', planner_api_key='secret-planner-key')
            ctx = graph.Context(load_tools(settings), settings, JobStore(settings.work_dir))
            job, _ = ctx.store.create('clip.mp4', b'private-video-bytes')
            client = Mock(spec=Client)
            command = Drop(source_id='residual')
            with tracing_context(enabled=True, client=client):
                graph.handle_command(job, command, ctx)
            client.create_run.assert_called_once()
            recorded = client.create_run.call_args.kwargs
            self.assertEqual(recorded['name'], 'command')
            self.assertEqual(recorded['inputs'], {'job_id': job.id, 'command': {'type': 'drop', 'source_id': 'residual'}})
            self.assertFalse(job.residual.keep)
            for secret in ['secret-tool-token', 'secret-planner-key', 'private-video-bytes']:
                self.assertNotIn(secret, str(client.mock_calls))
