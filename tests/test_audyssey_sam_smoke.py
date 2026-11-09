import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from audyssey.contracts import FilterResult
from tests.audyssey_media import make_video


class SamSmokeTests(unittest.TestCase):
    def test_smoke_reports_loading_inference_and_scores(self):
        smoke = importlib.import_module('scripts.sam_audio_smoke')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_video(root / 'input.mp4')
            out = root / 'out'
            cuda = Mock()
            cuda.max_memory_allocated.return_value = 2_000_000_000
            torch = SimpleNamespace(cuda=cuda, __version__='test-torch', version=SimpleNamespace(cuda='12.8'))
            output = io.StringIO()
            with patch.dict(sys.modules, torch=torch), patch.dict(os.environ, HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='0'), patch.object(smoke.separation, '_load'), patch.object(smoke.filtering, '_load'), patch.object(smoke.separation, 'separate', return_value=out / 'track.wav') as separate, patch.object(smoke.filtering, 'judge', return_value=FilterResult(passed=True, scores={'overall': 0.7})) as judge, patch.object(sys, 'argv', ['smoke', str(video), '--description', 'voice', '--span', '0', '1', '--out', str(out)]), contextlib.redirect_stdout(output):
                smoke.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report['torch'], 'test-torch')
            self.assertEqual(report['cuda'], '12.8')
            self.assertEqual(report['peak_memory_gb'], 2.0)
            self.assertEqual(report['judge']['scores'], {'overall': 0.7})
            self.assertEqual(report['track'], str(out / 'track.wav'))
            for field in ('separation_load_s', 'separation_s', 'judge_load_s', 'judge_s'):
                self.assertGreaterEqual(report[field], 0)
            plan = separate.call_args.args[1]
            self.assertEqual(plan.description, 'voice')
            self.assertEqual([(s.start, s.end) for s in plan.spans], [(0, 1)])
            self.assertEqual(separate.call_args.args[4], 1)
            self.assertEqual(judge.call_args.args[1], out / 'track.wav')
            cuda.synchronize.assert_called()
