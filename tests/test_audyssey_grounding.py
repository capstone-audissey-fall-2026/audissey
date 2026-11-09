import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from langsmith.run_helpers import is_traceable_function

from audyssey.contracts import VisualHint
from audyssey.settings import Settings
from audyssey.tools import load_tools
from tests.audyssey_media import make_video


def video_info(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(path)], check=True, capture_output=True)
    return json.loads(result.stdout)['streams'][0]


def frames(path):
    info = video_info(path)
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True)
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, info['height'], info['width'], 3)


class GroundingTests(unittest.TestCase):
    def test_stub_locate_finds_unless_ghost(self):
        from audyssey.tools.grounding import stub
        hint = VisualHint(time=1, box=(0, 0, 0.5, 1), description='man')
        result = stub.locate(Path('unused'), hint)
        self.assertTrue(result.found)
        self.assertEqual(result.box, hint.box)
        self.assertFalse(stub.locate(Path('unused'), hint.model_copy(update={'description': 'ghost'})).found)

    def test_stub_mask_video_keeps_frames_and_blackens_outside(self):
        from audyssey.tools.grounding import stub
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_video(root / 'input.mp4')
            out = root / 'masks' / 's1.mp4'
            hint = VisualHint(time=1, box=(0, 0, 0.5, 1), description='left')
            self.assertEqual(stub.mask_video(video, hint, out), out)
            self.assertEqual(video_info(video), video_info(out))
            data = frames(out)
            self.assertTrue(np.all(data[:, :, 80:] == 0))
            self.assertGreater(data[:, :, :80].max(), 0)

    def test_loader_grounding_is_traced_and_remote_requires_url(self):
        tools = load_tools(Settings())
        self.assertTrue(is_traceable_function(tools.locate))
        self.assertTrue(is_traceable_function(tools.mask_video))
        with self.assertRaisesRegex(ValueError, 'AUDYSSEY_GROUNDING_URL is required'):
            load_tools(Settings(grounder='remote'))

    def test_stub_mask_preserves_variable_frame_count(self):
        from audyssey.tools.grounding import stub
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'vfr.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=10:duration=3', '-vf', r'select=eq(n\,0)+eq(n\,1)+gt(n\,15)', '-vsync', 'vfr', '-c:v', 'mpeg4', '-pix_fmt', 'yuv420p', str(video)], check=True, capture_output=True)
            hint = VisualHint(time=0, box=(0, 0, .5, 1), description='left')
            out = stub.mask_video(video, hint, root / 'mask.mp4')
            self.assertEqual(video_info(out), video_info(video))
