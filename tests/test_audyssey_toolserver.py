import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient

from audyssey.contracts import FilterResult, SourcePlan, Span
from audyssey.media import write_wav, read_wav
from audyssey.tools.separation.stub import separate
from audyssey.toolserver import create_tool_app


class ToolServerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.mono = write_wav(self.root / 'mono.wav', np.ones((48000, 1)))
        self.plan = SourcePlan(name='character_a', kind='dialogue', description='voice', spans=[Span(start=0, end=0.5)])
        self.files = {'mono': ('../../escape.wav', self.mono.read_bytes(), 'audio/wav'), 'video': ('../../escape.mp4', b'video', 'video/mp4')}
        self.data = {'plan': self.plan.model_dump_json(), 'attempt': '1'}

    def client(self, tool='separation', token=''):
        client = TestClient(create_tool_app([tool], 'stub', token))
        self.addCleanup(client.close)
        return client

    def test_health(self):
        self.assertEqual(self.client(token='t').get('/health').json(), {'tools': ['separation'], 'impl': 'stub'})

    def test_separate_stub_matches_direct_call(self):
        response = self.client().post('/separate', files=self.files, data=self.data)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'audio/wav')
        direct = separate(self.mono, self.plan, Path('video'), self.root / 'direct.wav', 1)
        rate, data = wavfile.read(io.BytesIO(response.content))
        self.assertEqual(rate, 48000)
        np.testing.assert_allclose(data.reshape(-1, 1), read_wav(direct))

    def test_judge_stub_returns_filter_result(self):
        response = self.client('filtering').post('/judge', files={'mono': self.files['mono'], 'track': self.files['mono']}, data={'plan': self.plan.model_dump_json()})
        self.assertEqual(response.status_code, 200, response.text)
        result = FilterResult.model_validate(response.json())
        self.assertFalse(result.passed)
        self.assertTrue(result.suggestion.description.endswith(' (refined)'))

    def test_token_required(self):
        client = self.client(token='t')
        for headers, status in [({}, 401), ({'Authorization': 'Bearer x'}, 401), ({'Authorization': 'Bearer t'}, 200)]:
            self.assertEqual(client.post('/separate', files=self.files, data=self.data, headers=headers).status_code, status)

    def test_wrong_endpoint_for_tool(self):
        self.assertEqual(self.client().post('/judge').status_code, 404)
        self.assertEqual(self.client('filtering').post('/separate').status_code, 404)

    def test_bad_plan_json_is_422(self):
        for plan in ['{', '{}']:
            self.assertEqual(self.client().post('/separate', files=self.files, data={**self.data, 'plan': plan}).status_code, 422)

    def test_upload_names_are_not_used(self):
        def inspect(mono, plan, video, out, attempt):
            self.assertEqual(mono.name, 'mono.wav')
            self.assertEqual(video.name, 'video.mp4')
            self.assertEqual(mono.parent, video.parent)
            self.assertEqual(out.parent, mono.parent)
            self.assertEqual(video.read_bytes(), b'video')
            return separate(mono, plan, video, out, attempt)
        with patch('audyssey.tools.separation.stub.separate', inspect):
            self.assertEqual(self.client().post('/separate', files=self.files, data=self.data).status_code, 200)

    def test_separate_without_video_passes_none(self):
        def inspect(mono, plan, video, out, attempt):
            self.assertIsNone(video)
            return separate(mono, plan, video, out, attempt)
        with patch('audyssey.tools.separation.stub.separate', inspect):
            response = self.client().post('/separate', files={'mono': self.files['mono']}, data=self.data)
        self.assertEqual(response.status_code, 200)

    def test_multi_tool_app_serves_both(self):
        with TestClient(create_tool_app(['separation', 'filtering'], 'stub', 't')) as client:
            self.assertEqual(client.get('/health').json(), {'tools': ['separation', 'filtering'], 'impl': 'stub'})
            headers = {'Authorization': 'Bearer t'}
            response = client.post('/separate', files=self.files, data=self.data, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post('/judge', files={'mono': self.files['mono'], 'track': self.files['mono']}, data={'plan': self.plan.model_dump_json()}, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(FilterResult.model_validate(response.json()).passed)
            self.assertEqual(client.post('/judge').status_code, 401)

    def test_grounding_endpoints_validate_hint_and_authenticate(self):
        client = self.client('grounding', 't')
        self.assertEqual(client.get('/health').json(), {'tools': ['grounding'], 'impl': 'stub'})
        for endpoint in ['/locate', '/mask']:
            self.assertEqual(client.post(endpoint).status_code, 401)
            for hint in ['{', '{}', '{"time":0,"box":[0,0,2,1],"description":"a"}']:
                response = client.post(endpoint, files={'video': self.files['video']}, data={'hint': hint}, headers={'Authorization': 'Bearer t'})
                self.assertEqual(response.status_code, 422, response.text)
