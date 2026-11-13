import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from audyssey.contracts import SourcePlan, Span
from audyssey.media import read_wav, write_wav
from tests.generation_fakes import runtime


class SelvaGenerationTests(unittest.TestCase):
    def setUp(self):
        from audyssey.tools.generation import real
        self.real = real
        offline = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
        offline.start()
        self.addCleanup(offline.stop)
        real._load.cache_clear()
        real._load_clap.cache_clear()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = SourcePlan(name='bell', kind='sfx', description='a bell ringing', spans=[Span(start=0, end=0.1)])
        self.sibling = SourcePlan(name='rain', kind='ambience', description='rain falling')
        self.addCleanup(real._load.cache_clear)
        self.addCleanup(real._load_clap.cache_clear)

    def test_windows_short_video(self):
        self.assertEqual(self.real.windows(9.8, 15, 8), [(0, 9.8)])
        self.assertEqual(self.real.windows(15, 15, 8), [(0, 15)])

    def test_windows_cover_long_video(self):
        self.assertEqual(self.real.windows(20, 15, 8), [(0, 8), (7, 15), (12, 20)])
        self.assertEqual(self.real.windows(15.1, 15, 8), [(0, 8), (7, 15), (7.1, 15.1)])

    def test_windows_reject_invalid_settings(self):
        for args in [(20, 10, 1), (20, 10, 0), (20, 10, -2), (20, 0, 8), (float('nan'), 10, 8)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.real.windows(*args)

    def test_crossfade_join_length_and_continuity(self):
        spans = [(0, 8), (7, 15), (12, 20)]
        chunks = [np.ones(800, dtype=np.float32) for _ in spans]
        joined = self.real.crossfade_join(chunks, spans, 100)
        self.assertEqual(joined.shape, (2000,))
        np.testing.assert_allclose(joined, 1)
        joined = self.real.crossfade_join([np.zeros(8), np.ones(8)], [(0, 8), (4, 12)], 1)
        np.testing.assert_allclose(joined[4:8], [0, 1/3, 2/3, 1], atol=1e-7)

    def run_generate(self, fake, duration=10, seed=43):
        with patch.dict('sys.modules', fake.modules), patch.object(self.real, 'probe', return_value=(duration, 0)):
            return self.real.generate(Path('video.mp4'), self.plan, self.root / 'track.wav', seed)

    def test_loads_internal_generator_and_encoder(self):
        fake = runtime()
        self.run_generate(fake)
        self.run_generate(fake)
        requests = {(c.kwargs['repo_id'], c.kwargs['filename']) for c in fake.hub.hf_hub_download.call_args_list}
        self.assertEqual(requests, {
            ('jnwnlee/SelVA', 'weights/video_enc_sup_5.pth'),
            ('jnwnlee/SelVA', 'ext_weights/v1-44.pth'),
            ('jnwnlee/SelVA', 'ext_weights/synchformer_state_dict.pth'),
            ('sogang-audyssey/internal-model', 'generator/S_O_s43/model_ema_step1000.pth'),
            ('sogang-audyssey/internal-model', 'encoder/S_O/best.pt'),
        })
        self.assertTrue(all(c.kwargs['local_files_only'] for c in fake.hub.hf_hub_download.call_args_list))
        self.assertEqual(len(fake.hub.hf_hub_download.call_args_list), 5)
        self.assertEqual(fake.events, [('base', '/cache/weights/video_enc_sup_5.pth'), ('sync', '/cache/encoder/S_O/best.pt', False)])
        fake.generator.load_weights.assert_called_once_with('/cache/generator/S_O_s43/model_ema_step1000.pth')

    def test_generate_uses_eval_settings(self):
        fake = runtime()
        out = self.run_generate(fake)
        args, kwargs = fake.evaluation.generate.call_args
        self.assertIsNone(args[0])
        self.assertEqual(args[2], ['a bell ringing'])
        self.assertEqual(kwargs['cfg_strength'], 4.5)
        self.assertIsNone(kwargs.get('negative_text'))
        fake.modules['selva.model.flow_matching'].FlowMatching.assert_called_once_with(min_sigma=0, inference_mode='euler', num_steps=25)
        fake.torch.Generator.return_value.manual_seed.assert_called_once_with(43)
        fake.encoder.to.assert_called_once_with('cuda', 'bf16')
        data = read_wav(out)
        self.assertEqual(data.shape, (480000, 1))
        np.testing.assert_allclose(data[100:47000], 1, atol=0.002)
        np.testing.assert_array_equal(data[48000:], 0)

    def test_fifteen_seconds_uses_one_pass_by_default(self):
        fake = runtime(15)
        out = self.run_generate(fake, 15)
        self.assertEqual(read_wav(out).shape, (720000, 1))
        self.assertEqual(fake.evaluation.generate.call_count, 1)
        self.assertEqual(fake.evaluation.generate.call_args.args[1].shape[1], 375)

    def test_long_video_output_length(self):
        fake = runtime(20)
        out = self.run_generate(fake, 20)
        self.assertEqual(read_wav(out).shape, (960000, 1))
        calls = fake.evaluation.generate.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual([c.args[1].data[0, 0, 0, 0, 0] for c in calls], [0, 175, 300])
        self.assertEqual([c.args[1].shape[1] for c in calls], [200, 200, 200])

    def test_generate_pads_very_short_video_frames(self):
        fake = runtime(0.2)
        out = self.run_generate(fake, 0.2)
        self.assertEqual(read_wav(out).shape, (9600, 1))
        self.assertEqual(fake.evaluation.generate.call_args.args[1].shape[1], 16)

    def score(self, fake, siblings=None):
        track = write_wav(self.root / 'generated.wav', np.zeros(48000))
        with patch.dict('sys.modules', fake.modules):
            return self.real.score_generation(track, self.plan, [self.sibling] if siblings is None else siblings)

    def test_score_text_targetsel(self):
        fake = runtime()
        result = self.score(fake)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.scores['text_margin'], .8)
        fake.clap.get_text_embedding.assert_called_once_with(['a bell ringing', 'rain falling'])
        fake.clap.get_text_embedding.side_effect = lambda text: np.array([[0., 1.], [1., 0.]])
        result = self.score(fake)
        self.assertFalse(result.passed)
        self.assertEqual(result.scores['text_correct'], 0)
        self.assertEqual(result.scores['text_margin'], -1)
        self.assertEqual(result.reason, 'closer to sibling: rain')

    def test_score_averages_fixed_windows_and_is_repeatable(self):
        fake = runtime()
        track = write_wav(self.root / 'long.wav', np.linspace(0, .75, 15 * 48000))
        mono = read_wav(track).mean(axis=1)
        seen = []

        def embed(x):
            seen.append(x.copy())
            # One normalized embedding per 10 s window, as CLAP returns.
            return np.array([[1., 0.], [0., 1.]])[: len(x)]

        fake.clap.get_audio_embedding_from_data.side_effect = embed
        with patch.dict('sys.modules', fake.modules):
            first = self.real.score_generation(track, self.plan, [self.sibling])
            second = self.real.score_generation(track, self.plan, [self.sibling])
        self.assertEqual(first, second)
        self.assertTrue(first.passed)
        self.assertAlmostEqual(first.scores['text_margin'], .4 / 2 ** .5)
        self.assertEqual(len(seen), 2)
        for batch in seen:
            np.testing.assert_array_equal(batch, np.stack([mono[:480000], mono[240000:720000]]))

    def test_score_short_tracks_use_one_window(self):
        fake = runtime()
        for samples in (1, 48000, 480000):
            with self.subTest(samples=samples):
                fake.clap.get_audio_embedding_from_data.reset_mock()
                track = write_wav(self.root / 'short.wav', np.full(samples, .25))
                with patch.dict('sys.modules', fake.modules):
                    result = self.real.score_generation(track, self.plan, [self.sibling])
                self.assertTrue(result.passed)
                fake.clap.get_audio_embedding_from_data.assert_called_once()
                np.testing.assert_array_equal(
                    fake.clap.get_audio_embedding_from_data.call_args.kwargs['x'],
                    read_wav(track).mean(axis=1)[None, :],
                )

    def test_score_positive_margin_below_threshold_fails(self):
        with patch.dict(os.environ, {'AUDYSSEY_GENERATION_MARGIN_THRESHOLD': '0.9'}):
            result = self.score(runtime())
        self.assertFalse(result.passed)
        self.assertIn('threshold', result.reason)

    def test_score_without_siblings_passes(self):
        fake = runtime()
        with patch.dict(os.environ, {'AUDYSSEY_GENERATION_MARGIN_THRESHOLD': '100'}):
            result = self.score(fake, siblings=[])
        self.assertTrue(result.passed)
        self.assertEqual(result.scores, {'text_correct': 1})
        # CLAP's tokenizer squeezes a single text into the wrong shape; never call it.
        fake.clap.get_text_embedding.assert_not_called()
        fake.clap.get_audio_embedding_from_data.assert_not_called()

    def test_loading_requires_offline_mode_before_importing_models(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}), self.assertRaisesRegex(RuntimeError, "HF_HUB_OFFLINE=1"):
            self.real._load()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}), self.assertRaisesRegex(RuntimeError, "HF_HUB_OFFLINE=1"):
            self.real._load_clap()

    def test_clap_checkpoint_is_cached_and_reused(self):
        fake = runtime()
        self.score(fake)
        self.score(fake)
        fake.hub.hf_hub_download.assert_called_once_with(repo_id="lukewys/laion_clap", filename="630k-best.pt", local_files_only=True)
        fake.clap.load_ckpt.assert_called_once_with('/cache/630k-best.pt')
        fake.modules['laion_clap'].CLAP_Module.assert_called_once_with(enable_fusion=False, device='cuda')
        # 630k-best.pt pickles numpy scalars; allow exactly those under weights_only loading.
        (allowed,), _ = fake.torch.serialization.safe_globals.call_args
        self.assertIn('numpy.core.multiarray.scalar', [g[1] for g in allowed if isinstance(g, tuple)])

    def test_generate_matches_sync_segments_at_fractional_duration(self):
        fake = runtime(9.28)
        out = self.run_generate(fake, 9.28)
        self.assertEqual(read_wav(out).shape, (445440, 1))
        self.assertEqual(fake.evaluation.generate.call_args.args[1].shape[1], 232)
        fake.encoder.update_seq_lengths.assert_called_once_with(video_seq_len=224)
        self.assertEqual(fake.generator.update_seq_lengths.call_args.args[2], 224)
