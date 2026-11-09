import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audyssey.contracts import SourcePlan
from audyssey.media import write_wav
from audyssey.tools.filtering import real
from tests.sam_audio_fakes import Tensor, fake_runtime


class SamJudgeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.mono = write_wav(self.root / 'mono.wav', np.ones((100, 1)))
        self.track = write_wav(self.root / 'track.wav', np.full((100, 1), 0.5))
        self.plan = SourcePlan(name='voice', kind='dialogue', description='a voice')
        self.torch, self.sam, self.model, self.processor, self.batch = fake_runtime(
            'SAMAudioJudgeModel', 'SAMAudioJudgeProcessor'
        )
        self.batch.to.return_value = {'input_values': 'input', 'separated_values': 'separated', 'input_ids': 'text'}
        self.model.return_value = SimpleNamespace(
            overall=Tensor([[0.7]]), recall=Tensor([[0.8]]), precision=Tensor([[0.6]]), faithfulness=None
        )
        env = patch.dict(os.environ, {'AUDYSSEY_SAM_AUDIO_DEVICE': 'cpu'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        modules = patch.dict(sys.modules, torch=self.torch, sam_audio=self.sam)
        modules.start()
        self.addCleanup(modules.stop)
        if hasattr(real, '_load'):
            real._load.cache_clear()
            self.addCleanup(real._load.cache_clear)

    def judge(self):
        return real.judge(self.mono, self.track, self.plan)

    def test_judge_maps_scores_and_threshold(self):
        result = self.judge()
        self.assertTrue(result.passed)
        self.assertEqual(result.scores, {'overall': 0.7, 'recall': 0.8, 'precision': 0.6})
        self.assertEqual(result.reason, '')
        self.assertIsNone(result.suggestion)
        self.model.return_value.overall = Tensor([[0.3]])
        result = self.judge()
        self.assertFalse(result.passed)
        self.assertIn('0.30', result.reason)
        self.assertIn('0.5', result.reason)
        self.assertIsNone(result.suggestion)
        with patch.dict(os.environ, AUDYSSEY_JUDGE_PASS_THRESHOLD='0.3'):
            self.assertTrue(self.judge().passed)
        self.sam.SAMAudioJudgeModel.from_pretrained.assert_called_once_with('facebook/sam-audio-judge')
        self.sam.SAMAudioJudgeProcessor.from_pretrained.assert_called_once_with('facebook/sam-audio-judge')

    def test_judge_passes_description_and_both_audios(self):
        self.judge()
        args = self.processor.call_args.kwargs
        self.assertEqual(args['text'], ['a voice'])
        np.testing.assert_array_equal(args['input_audio'][0].data, np.ones((1, 100)))
        np.testing.assert_array_equal(args['separated_audio'][0].data, np.full((1, 100), 0.5))
        self.assertEqual(args['sampling_rate'], 48000)
        self.batch.to.assert_called_once_with('cpu')
        self.model.assert_called_once_with(input_values='input', separated_values='separated', input_ids='text')

    def test_missing_gpu_env_message(self):
        with patch.dict(sys.modules, torch=None, sam_audio=None):
            with self.assertRaisesRegex(RuntimeError, 'SAM Audio is not installed: install the gpu/ environment'):
                self.judge()
