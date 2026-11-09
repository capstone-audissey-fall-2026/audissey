import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from audyssey.contracts import SourcePlan, Span
from audyssey.media import read_wav, write_wav
from audyssey.tools.separation import real
from tests.sam_audio_fakes import Tensor, fake_runtime


class SamSeparationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.mono = write_wav(self.root / 'mono.wav', np.ones((100, 1)))
        self.plan = SourcePlan(name='voice', kind='dialogue', description='a voice')
        self.torch, self.sam, self.model, self.processor, self.batch = fake_runtime(
            'SAMAudio', 'SAMAudioProcessor'
        )
        self.model.separate.return_value = SimpleNamespace(target=[Tensor(np.full((1, 100), 0.5))])
        env = patch.dict(os.environ, {'AUDYSSEY_SAM_AUDIO_DEVICE': 'cpu'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        modules = patch.dict(sys.modules, torch=self.torch, sam_audio=self.sam)
        modules.start()
        self.addCleanup(modules.stop)
        if hasattr(real, '_load'):
            real._load.cache_clear()
            self.addCleanup(real._load.cache_clear)

    def separate(self, attempt=1):
        return real.separate(self.mono, self.plan, Path('unused.mp4'), self.root / 'track.wav', attempt)

    def test_separate_passes_description_and_span_anchors(self):
        self.plan.spans = [Span(start=0, end=0.001), Span(start=0.001, end=0.002)]
        self.separate()
        args = self.processor.call_args.kwargs
        self.assertEqual(args['descriptions'], ['a voice'])
        self.assertEqual(args['anchors'], [[['+', 0.0, 0.001], ['+', 0.001, 0.002]]])
        self.assertNotIn('masked_videos', args)
        np.testing.assert_array_equal(args['audios'][0].data, np.ones((1, 100)))
        self.batch.to.assert_called_once_with('cpu')
        self.model.separate.assert_called_once_with(self.batch, reranking_candidates=1, predict_spans=False)
        np.testing.assert_array_equal(read_wav(self.root / 'track.wav'), np.full((100, 1), 0.5))

    def test_separate_without_spans_uses_no_anchors(self):
        with patch.dict(os.environ, AUDYSSEY_SAM_AUDIO_PREDICT_SPANS='true', AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES='3'):
            self.separate()
        self.assertIsNone(self.processor.call_args.kwargs['anchors'])
        self.model.separate.assert_called_once_with(self.batch, reranking_candidates=3, predict_spans=True)

    def test_separate_writes_same_length_mono_48k(self):
        for length in (80, 120):
            with self.subTest(length=length):
                self.model.separate.return_value = SimpleNamespace(target=[Tensor(np.full((1, length), 0.5))])
                rate, data = wavfile.read(self.separate())
                self.assertEqual(rate, 48000)
                self.assertEqual(data.shape, (100,))
                self.assertEqual(read_wav(self.root / "track.wav").shape, (100, 1))
                self.assertEqual(data.dtype, np.float32)
                np.testing.assert_array_equal(data[:min(length, 100)], 0.5)
                if length < 100:
                    np.testing.assert_array_equal(data[length:], 0)

    def test_attempt_sets_seed(self):
        self.separate(1)
        self.separate(2)
        self.assertEqual([c.args[0] for c in self.torch.manual_seed.call_args_list], [1, 2])

    def test_model_loaded_once(self):
        self.separate()
        self.separate()
        self.sam.SAMAudio.from_pretrained.assert_called_once_with('facebook/sam-audio-large-tv')
        self.sam.SAMAudioProcessor.from_pretrained.assert_called_once_with('facebook/sam-audio-large-tv')
        self.model.to.assert_called_once_with('cpu')
        self.model.eval.assert_called_once_with()

    def test_unused_rankers_and_span_predictor_dropped_before_device(self):
        moved = []
        self.model.to.side_effect = lambda device: moved.append(
            (self.model.visual_ranker, self.model.text_ranker, hasattr(self.model, 'span_predictor'))
        ) or self.model
        self.separate()
        self.assertEqual(moved, [(None, None, False)])

    def test_rankers_and_span_predictor_kept_when_used(self):
        self.model.to.side_effect = None
        ranker, predictor = object(), object()
        self.model.visual_ranker = self.model.text_ranker = ranker
        self.model.span_predictor = predictor
        with patch.dict(os.environ, AUDYSSEY_SAM_AUDIO_PREDICT_SPANS='true', AUDYSSEY_SAM_AUDIO_RERANKING_CANDIDATES='3'):
            self.separate()
        self.assertIs(self.model.visual_ranker, ranker)
        self.assertIs(self.model.text_ranker, ranker)
        self.assertIs(self.model.span_predictor, predictor)

    def test_missing_gpu_env_message(self):
        with patch.dict(sys.modules, torch=None, sam_audio=None):
            with self.assertRaisesRegex(RuntimeError, 'SAM Audio is not installed: install the gpu/ environment'):
                self.separate()
