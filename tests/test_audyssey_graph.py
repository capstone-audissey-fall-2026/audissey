import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from tests.audyssey_media import make_video
from audyssey import graph
from audyssey.contracts import (
    SourcePlan,
    Source,
    Span,
    Drop,
    Keep,
    SetGain,
    SetPitch,
    Render,
    Reseparate,
    AddSource,
)
from audyssey.media import read_wav, extract_audio
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.tools import load_tools


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        video = make_video(self.root / "test.mp4")
        self.ctx = graph.Context(
            load_tools(Settings()), Settings(), JobStore(self.root / "jobs")
        )
        self.job, self.directory = self.ctx.store.create("test.mp4", video.read_bytes())

    def test_check_plans_merges_and_flags(self):
        plan = SourcePlan(
            name="a", kind="sfx", description="sound", spans=[Span(start=0, end=1)]
        )
        other = plan.model_copy(update={"spans": [Span(start=1, end=4)]})
        plans, issues = graph.check_plans([plan, other], 3)
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(plans[0].spans), 2)
        self.assertEqual(issues[0].kind, "span_out_of_range")
        self.assertEqual(len(plan.spans), 1)
        self.assertEqual(
            graph.check_plans([plan.model_copy(update={"description": " "})], 3)[1][
                0
            ].kind,
            "empty_description",
        )

    def test_process_source_retries_and_isolates_failure(self):
        extract_audio(self.directory / "input.mp4", self.directory / "mono.wav", True)
        self.job.mono_path = "mono.wav"
        source = Source(
            id="s1", plan=self.ctx.tools.plan(Path("v"), Path("a"), 3, [])[0]
        )
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(source.attempts, 2)
        self.assertTrue(source.filter.passed)

        def fail(*args):
            raise RuntimeError("broken tool")

        self.ctx.tools = replace(self.ctx.tools, separate=fail)
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(source.status, "failed")
        self.assertIsNone(source.track_path)

    def test_analyze_runs_both_validation_loops(self):
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready")
        self.assertEqual(len(job.sources), 3)
        self.assertNotIn("ghost_sound", [s.plan.name for s in job.sources])
        self.assertEqual(next(s for s in job.steps if s.key == "plan").round, 2)
        a = job.sources[0]
        self.assertEqual(a.attempts, 2)
        self.assertTrue(a.filter.passed)
        self.assertTrue(a.plan.description.endswith(" (refined)"))
        self.assertEqual([s.attempts for s in job.sources[1:]], [1, 1])
        self.assertTrue((self.directory / job.residual.path).is_file())

    def test_plan_round_limit_warns(self):
        self.ctx.settings.max_plan_rounds = 1
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(len(job.plan_issues), 1)
        self.assertEqual(
            next(s.status for s in job.steps if s.key == "plan"), "warning"
        )
        self.assertIn("ghost_sound", [s.plan.name for s in job.sources])

    def test_tool_error_marks_only_that_source_failed(self):
        original = self.ctx.tools.separate

        def separate(mono, plan, video, out, attempt):
            if plan.name == "sword_clash":
                raise RuntimeError("separation failed")
            return original(mono, plan, video, out, attempt)

        self.ctx.tools = replace(self.ctx.tools, separate=separate)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready")
        self.assertEqual([s.status for s in job.sources], ["done", "done", "failed"])

    def test_plan_exception_marks_job_failed(self):
        def fail(*args):
            raise RuntimeError("planner unavailable")

        self.ctx.tools = replace(self.ctx.tools, plan=fail)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "failed")
        self.assertEqual(job.error, "planner unavailable")
        self.assertEqual(self.ctx.store.load(job.id).state, "failed")
