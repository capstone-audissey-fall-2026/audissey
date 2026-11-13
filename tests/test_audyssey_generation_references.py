import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from audyssey import graph
from audyssey.contracts import FilterResult, Source, SourcePlan
from audyssey.media import read_wav, write_wav
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import load_tools
from audyssey.tools.generation import remote, stub
from audyssey.toolserver import create_tool_app


class GenerationReferencesTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan = SourcePlan(name='bell', kind='sfx', description='a bell ringing')
        self.wav = write_wav(self.root / 'ref.wav', np.ones(100))

    def test_graph_supplies_only_separated_stems(self):
        store = JobStore(self.root / 'jobs')
        job, directory = store.create('clip.mp4', b'video')
        generated = Source(id='s1', plan=self.plan, origin='generated', replaces='s2')
        job.sources = [
            generated,
            Source(id='s2', plan=self.plan, track_path='tracks/s2.wav', keep=False),
            Source(id='s3', plan=self.plan, track_path='tracks/s3.wav'),
            Source(id='s4', plan=self.plan, track_path='tracks/s4.wav', origin='generated'),
            Source(id='s5', plan=self.plan),
        ]
        score = Mock(return_value=FilterResult(passed=True))
        generate = lambda video, plan, out, seed: write_wav(out, np.zeros(100))
        ctx = graph.Context(replace(load_tools(Settings()), generate=generate, score_generation=score), Settings(), store)
        graph.generate_source(job, generated, ctx)
        self.assertEqual(generated.status, 'done')
        self.assertEqual(score.call_args.args[4], [directory / 'tracks/s2.wav', directory / 'tracks/s3.wav'])

    def test_server_preserves_each_reference_and_rejects_oversize(self):
        observed = []
        def score(track, plan, siblings, reference, references):
            self.assertEqual(read_wav(reference).shape, (100, 1))
            observed.extend(float(read_wav(p)[0, 0]) for p in references)
            return FilterResult(passed=True, scores={'ref_margin': -1})
        with patch.object(stub, 'score_generation', side_effect=score), TestClient(create_tool_app(['generation'], 'stub', '')) as client:
            response = client.post('/score', files=[('track', ('track.wav', self.wav.read_bytes())), ('reference', ('ref.wav', self.wav.read_bytes())), ('references', ('../../ref.wav', self.wav.read_bytes())), ('references', ('../../ref.wav', self.wav.read_bytes()))], data={'plan': self.plan.model_dump_json(), 'siblings': '[]'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['scores']['ref_margin'], -1)
        self.assertEqual(observed, [1, 1])
        with patch.dict(os.environ, {'AUDYSSEY_MAX_UPLOAD_MB': str(10 / 2**20)}), TestClient(create_tool_app(['generation'], 'stub', '')) as client:
            response = client.post('/score', files=[('track', ('track.wav', b'x')), ('references', ('ref.wav', b'x' * 20))], data={'plan': self.plan.model_dump_json(), 'siblings': '[]'})
            self.assertEqual(response.status_code, 413, response.text)

    def test_remote_sends_repeated_reference_uploads(self):
        import httpx
        first = write_wav(self.root / 'first.wav', np.ones(100))
        second = write_wav(self.root / 'second.wav', np.ones(100))
        with patch.object(remote, 'post', return_value=httpx.Response(200, json={'passed': True})) as post:
            result = remote.score_generation(self.wav, self.plan, [], self.wav, [first, second])
        self.assertTrue(result.passed)
        uploads = post.call_args.args[2]
        self.assertEqual([name for name, _ in uploads], ['track', 'reference', 'references', 'references'])
        self.assertTrue(all(value[1] == self.wav.read_bytes() for _, value in uploads))

    def test_remote_does_not_compare_target_reference_to_itself(self):
        import httpx
        other = write_wav(self.root / 'other.wav', np.zeros(100))
        with patch.object(remote, 'post', return_value=httpx.Response(200, json={'passed': True})) as post:
            remote.score_generation(self.wav, self.plan, [], self.wav, [self.wav, other])
        competitors = [value for name, value in post.call_args.args[2] if name == 'references']
        self.assertEqual(len(competitors), 1)
        self.assertEqual(competitors[0][1], other.read_bytes())

    def test_remote_targetsel_matches_direct_scoring(self):
        from audyssey.tools.generation import real
        from tests.selva_fakes import runtime
        fake = runtime()
        fake.clap.get_audio_embedding_from_data.side_effect = lambda x: np.array([[1., 0.]]) if x[0, 0] else np.array([[0., 1.]])
        other = write_wav(self.root / 'other.wav', np.zeros(100))
        real._load_clap.cache_clear()
        self.addCleanup(real._load_clap.cache_clear)
        with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}), patch.dict('sys.modules', fake.modules), TestClient(create_tool_app(['generation'], 'real', '')) as client:
            def post(url, path, files, data, token, timeout):
                response = client.post(path, files=files, data=data)
                self.assertEqual(response.status_code, 200, response.text)
                return response
            direct = real.score_generation(self.wav, self.plan, [], self.wav, [self.wav, other])
            with patch.object(remote, 'post', side_effect=post):
                transported = remote.score_generation(self.wav, self.plan, [], self.wav, [self.wav, other])
            self.assertEqual(direct.scores['ref_margin'], 1)
            self.assertEqual(transported, direct)
            with patch.object(remote, 'post', side_effect=post):
                single = remote.score_generation(self.wav, self.plan, [], self.wav, [self.wav])
            self.assertNotIn('ref_margin', single.scores)
