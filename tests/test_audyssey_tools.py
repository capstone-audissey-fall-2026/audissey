import tempfile
import unittest
from pathlib import Path
import numpy as np
from audyssey.media import write_wav, read_wav
from audyssey.tools.planning import stub as planning
from audyssey.tools.filtering import stub as filtering
from audyssey.tools.separation import stub as separation


class ToolsTests(unittest.TestCase):
    def test_stub_plan_loop(self):
        plans = planning.plan(Path("video"), Path("audio"), 3, [])
        issues = planning.validate_plan(Path("video"), Path("audio"), plans)
        self.assertEqual([i.source_name for i in issues], ["ghost_sound"])
        plans = planning.plan(Path("video"), Path("audio"), 3, issues)
        self.assertEqual(len(plans), 3)
        self.assertEqual(
            planning.validate_plan(Path("video"), Path("audio"), plans), []
        )

    def test_stub_judge_loop(self):
        plan = planning.plan(Path("video"), Path("audio"), 3, [])[0]
        result = filtering.judge(Path("mono"), Path("track"), plan)
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.suggestion)
        self.assertTrue(
            filtering.judge(Path("mono"), Path("track"), result.suggestion).passed
        )

    def test_stub_separate_keeps_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mono = write_wav(root / "mono.wav", np.ones((144000, 1), dtype=np.float32))
            plan = planning.plan(Path("video"), mono, 3, [])[0]
            out = separation.separate(mono, plan, Path("video"), root / "track.wav", 1)
            data = read_wav(out)
            self.assertEqual(data.shape, (144000, 1))
            self.assertTrue(np.all(data[:43200] == 1))
            self.assertTrue(np.all(data[43200:] == 0))
