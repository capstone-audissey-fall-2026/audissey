import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import uvicorn

from audyssey import graph
from audyssey.contracts import Render
from audyssey.media import read_wav
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import load_tools
from audyssey.toolserver import create_tool_app
from tests.audyssey_media import make_video


class RemoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        cls.urls = []
        for tool in ['separation', 'filtering', 'grounding', 'generation']:
            listener = socket.socket()
            listener.bind(('127.0.0.1', 0))
            server = uvicorn.Server(uvicorn.Config(create_tool_app([tool], 'stub', 't'), log_level='error', ws='none'))
            thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
            thread.start()
            cls.servers.append((server, thread, listener))
            cls.urls.append(f'http://127.0.0.1:{listener.getsockname()[1]}')
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            if not server.started:
                cls.tearDownClass()
                raise RuntimeError('tool server did not start')

    @classmethod
    def tearDownClass(cls):
        for server, thread, listener in cls.servers:
            server.should_exit = True
            thread.join(timeout=5)
            listener.close()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = make_video(self.root / 'clip.mp4')
        env = {'AUDYSSEY_SEPARATOR': 'remote', 'AUDYSSEY_FILTER': 'remote', 'AUDYSSEY_SEPARATOR_URL': self.urls[0], 'AUDYSSEY_FILTER_URL': self.urls[1], 'AUDYSSEY_TOOL_TOKEN': 't'}
        self.env = patch.dict(os.environ, env)
        self.env.start()
        self.addCleanup(self.env.stop)

    def analyze(self):
        settings = Settings(work_dir=self.root / 'jobs')
        ctx = graph.Context(load_tools(settings), settings, JobStore(settings.work_dir))
        job, directory = ctx.store.create('clip.mp4', self.video.read_bytes())
        return graph.analyze(job, ctx), ctx, directory

    def test_remote_e2e_matches_stub(self):
        job, ctx, directory = self.analyze()
        self.assertEqual(job.state, 'ready', job.error)
        self.assertEqual(len(job.sources), 3)
        self.assertTrue(all(s.filter.passed for s in job.sources))
        self.assertEqual(job.sources[0].attempts, 2)
        graph.handle_command(job, Render(), ctx)
        np.testing.assert_allclose(read_wav(directory / job.outputs.mix_wav), read_wav(directory / job.audio_path), atol=1e-6)

    def assert_failed_sources(self, job, expected):
        self.assertEqual(job.state, 'ready')
        self.assertEqual([s.status for s in job.sources], ['failed'] * 3)
        messages = [s.message for s in job.steps if s.status == 'failed']
        self.assertEqual(len(messages), 3)
        for message in messages:
            self.assertIn(expected, message)
        return messages

    def test_wrong_token_fails_only_sources(self):
        token = 'secret-token-value'
        with patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN=token):
            job, _, _ = self.analyze()
        for message in self.assert_failed_sources(job, '401'):
            self.assertNotIn(token, message)

    def test_unreachable_server_fails_sources(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            url = f'http://127.0.0.1:{listener.getsockname()[1]}'
        with patch.dict(os.environ, AUDYSSEY_SEPARATOR_URL=url):
            job, _, _ = self.analyze()
        self.assert_failed_sources(job, url)

    def test_remote_without_url_raises(self):
        for tool in ['separator', 'filter']:
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, f'AUDYSSEY_{tool.upper()}_URL is required for remote mode'):
                load_tools(Settings(**{tool: 'remote', f'{tool}_url': ''}))

    def test_http_errors_are_sanitized(self):
        from audyssey.tools.remote import post
        token = 'secret-token-value'
        for failure in [httpx.Response(500, text=token), httpx.ReadTimeout(token), httpx.ConnectError(token)]:
            kwargs = {'return_value': failure} if isinstance(failure, httpx.Response) else {'side_effect': failure}
            with self.subTest(failure=type(failure).__name__), patch('httpx.post', **kwargs):
                with self.assertRaises(RuntimeError) as caught:
                    post(self.urls[0], '/separate', {}, {}, token, 1)
                self.assertIn(self.urls[0], str(caught.exception))
                self.assertNotIn(token, str(caught.exception))
                self.assertIn('500' if isinstance(failure, httpx.Response) else type(failure).__name__, str(caught.exception))

    def test_error_urls_exclude_credentials_and_query(self):
        from audyssey.tools.remote import post
        url = 'http://user:secret-url-password@gpu:8700?api_key=secret-query-key'
        for failure in [httpx.Response(401), httpx.ConnectError('secret-url-password')]:
            kwargs = {'return_value': failure} if isinstance(failure, httpx.Response) else {'side_effect': failure}
            with self.subTest(failure=type(failure).__name__), patch('httpx.post', **kwargs):
                with self.assertRaises(RuntimeError) as caught:
                    post(url, '/separate', {}, {}, 'secret-tool-token', 1)
                message = str(caught.exception)
                self.assertIn('http://gpu:8700', message)
                for secret in ['secret-url-password', 'secret-query-key', 'secret-tool-token']:
                    self.assertNotIn(secret, message)

    def test_grounding_remote_matches_direct_calls(self):
        from audyssey.contracts import VisualHint
        from audyssey.tools.grounding import remote, stub
        from tests.test_audyssey_grounding import frames, video_info
        hint = VisualHint(time=1, box=(0, 0, .5, 1), description='left')
        with patch.dict(os.environ, AUDYSSEY_GROUNDER='remote', AUDYSSEY_GROUNDING_URL=self.urls[2]):
            tools = load_tools(Settings())
            self.assertEqual(tools.locate(self.video, hint), stub.locate(self.video, hint))
            ghost = hint.model_copy(update={'description': 'ghost'})
            self.assertFalse(remote.locate(self.video, ghost).found)
            out = tools.mask_video(self.video, hint, self.root / 'remote/mask.mp4')
            direct = stub.mask_video(self.video, hint, self.root / 'direct.mp4')
            self.assertEqual(video_info(out), video_info(direct))
            np.testing.assert_array_equal(frames(out), frames(direct))
            with patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN=''):
                for call in [lambda: remote.locate(self.video, hint), lambda: remote.mask_video(self.video, hint, out)]:
                    with self.assertRaisesRegex(RuntimeError, '401'):
                        call()

    def test_generation_remote_matches_stub(self):
        from audyssey.contracts import SourcePlan
        from audyssey.tools.generation import stub
        plan = SourcePlan(name="sword_clash", kind="sfx", description="clashing swords")
        sibling = SourcePlan(name="rain", kind="sfx", description="rain falling")
        with patch.dict(os.environ, AUDYSSEY_GENERATOR="remote", AUDYSSEY_GENERATOR_URL=self.urls[3]):
            tools = load_tools(Settings())
            for seed in [42, 43]:
                remote = tools.generate(self.video, plan, self.root / "remote/track.wav", seed)
                direct = stub.generate(self.video, plan, self.root / "direct.wav", seed)
                np.testing.assert_array_equal(read_wav(remote), read_wav(direct))
                for reference in [None, direct]:
                    self.assertEqual(tools.score_generation(remote, plan, [sibling], reference), stub.score_generation(direct, plan, [sibling], reference))
            with patch.dict(os.environ, AUDYSSEY_TOOL_TOKEN="wrong-secret"):
                for call in [lambda: tools.generate(self.video, plan, self.root / "fail.wav", 43), lambda: tools.score_generation(direct, plan, [], None)]:
                    with self.assertRaisesRegex(RuntimeError, "401") as caught:
                        call()
                    self.assertNotIn("wrong-secret", str(caught.exception))
