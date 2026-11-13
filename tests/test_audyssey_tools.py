import inspect
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

    def test_load_tools_defaults_to_stub(self):
        from audyssey.tools import load_tools
        from audyssey.settings import Settings

        tools = load_tools(Settings())
        self.assertIs(inspect.unwrap(tools.plan), planning.plan)
        self.assertIs(inspect.unwrap(tools.validate_plan), planning.validate_plan)
        self.assertIs(inspect.unwrap(tools.separate), separation.separate)
        self.assertIs(inspect.unwrap(tools.judge), filtering.judge)

    def test_load_tools_real_uses_planner(self):
        from audyssey.tools import load_tools
        from audyssey.settings import Settings

        tools = load_tools(Settings(planner="real"))
        from audyssey.tools.planning import real

        self.assertIs(inspect.unwrap(tools.plan), real.plan)
        self.assertIs(inspect.unwrap(tools.validate_plan), real.validate_plan)

    def test_stub_plans_include_visual_hints(self):
        plans = {p.name: p for p in planning.plan(Path('v'), Path('a'), 3, [])}
        hint = plans['character_a'].visual_hint
        self.assertIsNotNone(hint)
        self.assertEqual((hint.time, hint.box, hint.description), (1.0, (0, 0, .5, 1), 'man on the left'))
        self.assertEqual(plans['ghost_sound'].visual_hint.description, 'ghost')
        self.assertIsNone(plans['character_b'].visual_hint)

    def test_stub_generate_length_matches_video(self):
        from audyssey.tools import load_tools
        from audyssey.settings import Settings
        from tests.audyssey_media import make_video
        from audyssey.media import probe
        from langsmith.run_helpers import is_traceable_function
        tools = load_tools(Settings())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_video(root / "silent.mp4", seconds=0.3, audio=False)
            plan = planning.plan(video, Path("unused"), 0.3, [])[2]
            # Generation fills the clip even when the planner asks for no spans.
            plan.spans = []
            out = tools.generate(video, plan, root / "track.wav", 43)
            self.assertEqual(read_wav(out).shape, (round(probe(video)[0] * 48000), 1))
            self.assertGreater(float(np.max(np.abs(read_wav(out)))), 0.15)
            self.assertTrue(is_traceable_function(tools.generate))
            self.assertTrue(is_traceable_function(tools.score_generation))

    def test_stub_score_fails_first_seed_then_passes(self):
        from audyssey.tools import load_tools
        from audyssey.settings import Settings
        from tests.audyssey_media import make_video
        settings = Settings()
        tools = load_tools(settings)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = make_video(root / "silent.mp4", audio=False)
            plan = planning.plan(video, Path("unused"), 3, [])[2]
            sibling = plan.model_copy(update={"name": "other", "description": "rain"})
            for seed, passed in [(settings.selva_seed, False), (settings.selva_seed + 1, True)]:
                out = tools.generate(video, plan, root / "track.wav", seed)
                result = tools.score_generation(out, plan, [sibling])
                self.assertEqual(result.passed, passed)
                self.assertEqual(result.scores["text_correct"], float(passed))
                self.assertGreater(result.scores["text_margin"], 0)
                self.assertEqual(result.reason, "" if passed else "stub: first seed")
                self.assertNotIn("text_margin", tools.score_generation(out, plan, []).scores)
            other = tools.generate(video, sibling, root / "other.wav", settings.selva_seed)
            self.assertTrue(tools.score_generation(other, sibling, [plan]).passed)

    def test_remote_generator_requires_url(self):
        from audyssey.tools import load_tools
        from audyssey.settings import Settings
        with self.assertRaisesRegex(ValueError, "AUDYSSEY_GENERATOR_URL is required for remote mode"):
            load_tools(Settings(generator="remote", generator_url=""))
