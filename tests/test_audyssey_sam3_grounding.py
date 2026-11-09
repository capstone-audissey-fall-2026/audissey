import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audyssey.contracts import VisualHint
from tests.audyssey_media import make_video
from tests.test_audyssey_grounding import frames, video_info


class Sam3GroundingTests(unittest.TestCase):
    def setUp(self):
        from audyssey.tools.grounding import real
        self.real = real
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = make_video(self.root / 'input.mp4')
        self.hint = VisualHint(time=1, box=(.1, .2, .6, .8), description='man')
        self.state = {'num_frames': 30}
        self.model = Mock()
        self.model.init_state.side_effect = lambda **kwargs: self.state.copy()
        self.outputs = self.output()
        self.model.add_prompt.side_effect = lambda **kwargs: (kwargs['frame_idx'], self.outputs)
        self.model.propagate_in_video.return_value = iter(())
        self.builder = Mock(return_value=SimpleNamespace(model=self.model))
        self.hub = Mock(return_value='/existing/cache/sam3.1_multiplex.pt')
        self.torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda **kwargs: nullcontext(), bfloat16='bf16', cuda=SimpleNamespace(device=lambda device: nullcontext()), autocast_increment_nesting=lambda: 0, autocast_decrement_nesting=lambda: 0)
        modules = {'torch': self.torch, 'sam3.model_builder': SimpleNamespace(build_sam3_multiplex_video_predictor=self.builder), 'huggingface_hub': SimpleNamespace(hf_hub_download=self.hub)}
        def import_module(name):
            if name not in modules:
                raise ModuleNotFoundError(name)
            return modules[name]
        patcher = patch.object(real, 'import_module', side_effect=import_module)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, AUDYSSEY_SAM3_DEVICE='cuda:0', AUDYSSEY_SAM3_MODEL='facebook/sam3.1', AUDYSSEY_SAM3_TRACK_FPS='0')
        env.start()
        self.addCleanup(env.stop)
        real._load.cache_clear()
        self.addCleanup(real._load.cache_clear)

    def output(self):
        masks = np.zeros((2, 120, 160), dtype=bool)
        masks[0, :, 100:] = True
        masks[1, :, :80] = True
        return {'out_obj_ids': np.array([4, 8]), 'out_probs': np.array([.6, .9]), 'out_boxes_xywh': np.array([[.625, 0, .375, 1], [0, 0, .5, 1]]), 'out_binary_masks': masks}

    def test_locate_returns_normalized_best_box(self):
        result = self.real.locate(self.video, self.hint)
        self.assertTrue(result.found)
        self.assertEqual(result.box, (0, 0, .5, 1))
        self.assertAlmostEqual(result.score, .9)
        args = self.model.add_prompt.call_args.kwargs
        self.assertEqual(args['frame_idx'], 10)
        self.assertEqual(args['text_str'], 'man')
        np.testing.assert_allclose(args['boxes_xywh'], [[.1, .2, .5, .6]])
        self.assertEqual(args['box_labels'], [1])
        self.assertEqual(args['inference_state'], {})

    def test_locate_not_found(self):
        self.outputs = {'out_obj_ids': np.array([]), 'out_probs': np.array([])}
        result = self.real.locate(self.video, self.hint)
        self.assertFalse(result.found)
        self.assertIsNone(result.box)

    def test_mask_video_writes_same_shape_with_black_outside(self):
        def propagate(**kwargs):
            indices = range(10, 30) if not kwargs['reverse'] else range(10, -1, -1)
            for index in indices:
                if index != 20:
                    yield index, self.output()
        self.model.propagate_in_video.side_effect = propagate
        out = self.real.mask_video(self.video, self.hint, self.root / 'masks/s1.mp4')
        self.assertEqual(video_info(out), video_info(self.video))
        data = frames(out)
        self.assertTrue(np.all(data[:, :, 80:] == 0))
        self.assertTrue(np.all(data[20] == 0))
        self.assertGreater(data[0, :, :80].max(), 0)
        self.assertGreater(data[-1, :, :80].max(), 0)
        self.assertEqual([c.kwargs['reverse'] for c in self.model.propagate_in_video.call_args_list], [False, True])

    def test_mask_video_tracks_at_reduced_fps(self):
        # SAM3 keeps per-frame state on the GPU; tracking a 1080p60 clip at full rate
        # does not fit 24 GB, so the masked video is produced at AUDYSSEY_SAM3_TRACK_FPS.
        self.model.init_state.side_effect = lambda **kwargs: {'num_frames': len(frames(Path(kwargs['resource_path'])))}
        self.model.propagate_in_video.side_effect = lambda **kwargs: ((i, self.output()) for i in range(len(frames(Path(self.model.init_state.call_args.kwargs['resource_path'])))))
        source_fps = video_info(self.video)['r_frame_rate']
        with patch.dict(os.environ, AUDYSSEY_SAM3_TRACK_FPS='5'):
            out = self.real.mask_video(self.video, self.hint, self.root / 'masks/low.mp4')
        self.assertNotEqual(self.model.init_state.call_args.kwargs['resource_path'], str(self.video))
        self.assertEqual(video_info(out)['r_frame_rate'], '5/1')
        self.assertNotEqual(source_fps, '5/1')
        data = frames(out)
        self.assertTrue(np.all(data[:, :, 80:] == 0))
        self.assertGreater(data[0, :, :80].max(), 0)

    def test_model_loaded_once(self):
        self.real.locate(self.video, self.hint)
        self.real.locate(self.video, self.hint)
        self.builder.assert_called_once_with(checkpoint_path='/existing/cache/sam3.1_multiplex.pt', use_fa3=False, compile=False, warm_up=False)
        self.hub.assert_called_once_with(repo_id='facebook/sam3.1', filename='sam3.1_multiplex.pt', local_files_only=True)

    def test_missing_gpu_env_message(self):
        with patch.object(self.real, 'import_module', side_effect=ModuleNotFoundError('sam3')):
            with self.assertRaisesRegex(RuntimeError, 'SAM3 is not installed: install the gpu/ environment'):
                self.real.locate(self.video, self.hint)

    def test_prompt_failure_releases_state(self):
        state = {'num_frames': 30, 'features': object()}
        self.model.init_state.return_value = state
        self.model.init_state.side_effect = None
        self.model.add_prompt.side_effect = RuntimeError('prompt failed')
        with self.assertRaisesRegex(RuntimeError, 'prompt failed'):
            self.real.locate(self.video, self.hint)
        self.assertEqual(state, {})

    def test_propagation_flushes_final_buffer(self):
        def propagate(**kwargs):
            if kwargs.get('is_last_batch', False):
                yield (29 if not kwargs['reverse'] else 0), self.output()
        self.model.propagate_in_video.side_effect = propagate
        data = frames(self.real.mask_video(self.video, self.hint, self.root / 'mask.mp4'))
        self.assertGreater(data[0].max(), 0)
        self.assertGreater(data[-1].max(), 0)

    def test_builder_autocast_is_restored_on_success_and_failure(self):
        from contextlib import contextmanager
        for fails in (False, True):
            with self.subTest(fails=fails):
                self.real._load.cache_clear()
                state = {'depth': 0, 'enabled': False}
                def increment():
                    state['depth'] += 1
                    return state['depth']
                def decrement():
                    state['depth'] -= 1
                    return state['depth']
                @contextmanager
                def autocast(**kwargs):
                    previous = state['enabled']
                    state['enabled'] = kwargs.get('enabled', True)
                    increment()
                    try:
                        yield
                    finally:
                        state['enabled'] = previous
                        decrement()
                def build(**kwargs):
                    state['enabled'] = True
                    increment()
                    if fails:
                        raise RuntimeError('load failed')
                    increment()
                    return SimpleNamespace(model=self.model)
                self.torch.autocast = autocast
                self.torch.autocast_increment_nesting = increment
                self.torch.autocast_decrement_nesting = decrement
                self.builder.side_effect = build
                if fails:
                    with self.assertRaisesRegex(RuntimeError, 'load failed'):
                        self.real._load()
                else:
                    self.real._load()
                self.assertEqual(state, {'depth': 0, 'enabled': False})

    def variable_video(self):
        import subprocess
        path = self.root / 'vfr.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=10:duration=3', '-vf', r'select=eq(n\,0)+eq(n\,1)+gt(n\,15)', '-vsync', 'vfr', '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p', str(path)], check=True, capture_output=True)
        self.state['num_frames'] = 16
        return path

    def timestamps(self, video):
        import json
        import subprocess
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)], check=True, capture_output=True)
        return [float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']]

    def test_variable_timestamp_selects_actual_hint_frame(self):
        self.real.locate(self.variable_video(), self.hint.model_copy(update={'time': 1.6}))
        self.assertEqual(self.model.add_prompt.call_args.kwargs['frame_idx'], 2)

    def test_mask_preserves_variable_frame_timestamps(self):
        video = self.variable_video()
        self.model.propagate_in_video.side_effect = lambda **kwargs: ((i, self.output()) for i in range(16))
        out = self.real.mask_video(video, self.hint.model_copy(update={'time': 0}), self.root / 'vfr-mask.mp4')
        self.assertEqual(video_info(out), video_info(video))
        self.assertEqual(self.timestamps(out), self.timestamps(video))
        np.testing.assert_array_equal(frames(out)[:, :, :80], frames(video)[:, :, :80])

    def test_mask_supports_ubuntu_2204_ffmpeg_options(self):
        import subprocess
        from audyssey.tools.grounding import stub
        original = subprocess.run
        def older_ffmpeg(args, **kwargs):
            if '-fps_mode' in args:
                raise subprocess.CalledProcessError(1, args, stderr='Unrecognized option fps_mode')
            return original(args, **kwargs)
        with patch.object(self.real.subprocess, 'run', side_effect=older_ffmpeg):
            for tool in (self.real, stub):
                out = tool.mask_video(self.video, self.hint, self.root / f'{tool.__name__}.mp4')
                self.assertEqual(video_info(out), video_info(self.video))
