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
        self.ctx.settings.visual_prompt = True
        self.ctx.settings.max_plan_rounds = 1
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(
            [(issue.source_name, issue.kind) for issue in job.plan_issues],
            [("ghost_sound", "not_visible_in_frame"), ("ghost_sound", "visual_hint_not_found")],
        )
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

    def test_analyze_then_untouched_render_equals_original(self):
        job = graph.analyze(self.job, self.ctx)
        graph.handle_command(job, Render(), self.ctx)
        np.testing.assert_allclose(
            read_wav(self.directory / job.outputs.mix_wav),
            read_wav(self.directory / "audio.wav"),
            atol=1e-4,
        )

    def test_commands_update_values(self):
        job = graph.analyze(self.job, self.ctx)
        for cmd in [
            Drop(source_id="s1"),
            SetGain(source_id="s2", gain=0.5),
            Keep(source_id="residual"),
            SetPitch(source_id="s2", semitones=5),
        ]:
            graph.handle_command(job, cmd, self.ctx)
        self.assertFalse(job.sources[0].keep)
        self.assertEqual(job.sources[1].gain, 0.5)
        self.assertEqual(job.sources[1].pitch_semitones, 5)
        self.assertTrue(job.residual.keep)
        with self.assertRaises(ValueError):
            graph.handle_command(
                job, SetPitch(source_id="residual", semitones=1), self.ctx
            )

    def test_reseparate_and_add_source(self):
        job = graph.analyze(self.job, self.ctx)
        before = (self.directory / "residual.wav").stat().st_mtime_ns
        graph.handle_command(
            job, Reseparate(source_id="s2", description="new"), self.ctx
        )
        self.assertEqual(job.sources[1].plan.description, "new")
        self.assertEqual(job.sources[1].attempts, 1)
        self.assertNotEqual(
            (self.directory / "residual.wav").stat().st_mtime_ns, before
        )
        graph.handle_command(
            job,
            AddSource(plan=SourcePlan(name="new", kind="sfx", description="sound")),
            self.ctx,
        )
        self.assertEqual(job.sources[-1].id, "s4")
        with self.assertRaises(ValueError):
            graph.handle_command(
                job,
                AddSource(
                    plan=SourcePlan(
                        name="bad",
                        kind="sfx",
                        description="sound",
                        spans=[Span(start=0, end=4)],
                    )
                ),
                self.ctx,
            )

    def test_unknown_source_id(self):
        job = graph.analyze(self.job, self.ctx)
        snapshot = job.model_dump_json()
        version = self.ctx.store.load(job.id).version
        with self.assertRaises(KeyError):
            graph.handle_command(job, Drop(source_id="s99"), self.ctx)
        self.assertEqual(job.model_dump_json(), snapshot)
        self.assertEqual(self.ctx.store.load(job.id).version, version)

    def test_reseparate_rejects_invalid_span_without_mutation(self):
        job = graph.analyze(self.job, self.ctx)
        snapshot = job.model_dump_json()
        with self.assertRaises(ValueError):
            graph.handle_command(
                job, Reseparate(source_id="s2", spans=[Span(start=0, end=4)]), self.ctx
            )
        self.assertEqual(job.model_dump_json(), snapshot)

    def test_edit_during_planning_survives_replan(self):
        original = self.ctx.tools.validate_plan

        def validate(video, audio, plans):
            if any(p.name == "ghost_sound" for p in plans):
                graph.handle_command(self.job, Drop(source_id="s1"), self.ctx)
                graph.handle_command(
                    self.job, SetGain(source_id="s2", gain=0.5), self.ctx
                )
            return original(video, audio, plans)

        self.ctx.tools = replace(self.ctx.tools, validate_plan=validate)
        job = graph.analyze(self.job, self.ctx)
        self.assertFalse(job.sources[0].keep)
        self.assertEqual(job.sources[1].gain, 0.5)

    def test_hint_not_found_triggers_replan(self):
        self.ctx.settings.visual_prompt = True
        from audyssey.contracts import GroundResult, VisualHint
        feedback = []
        hint = VisualHint(time=1, box=(0, 0, .5, 1), description='ghost')
        def plan(video, audio, duration, issues):
            feedback.append(issues)
            return [SourcePlan(name='a', kind='sfx', description='sound', visual_hint=hint if not issues else None)]
        self.ctx.tools = replace(self.ctx.tools, plan=plan, validate_plan=lambda *args: [], locate=lambda *args: GroundResult(found=False))
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, 'ready')
        self.assertEqual(len(feedback), 2)
        self.assertEqual(feedback[1][0].kind, 'visual_hint_not_found')
        self.assertEqual(feedback[1][0].source_name, 'a')

    def test_locate_exception_becomes_issue(self):
        self.ctx.settings.visual_prompt = True
        from audyssey.contracts import VisualHint
        hint = VisualHint(time=1, box=(0, 0, .5, 1), description='a')
        def fail(*args):
            raise RuntimeError('grounder unavailable')
        self.ctx.settings.max_plan_rounds = 1
        self.ctx.tools = replace(self.ctx.tools, plan=lambda *args: [SourcePlan(name='a', kind='sfx', description='sound', visual_hint=hint)], validate_plan=lambda *args: [], locate=fail)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, 'ready')
        self.assertEqual(job.plan_issues[0].kind, 'visual_hint_not_found')
        self.assertIn('grounder unavailable', job.plan_issues[0].message)

    def test_visual_prompt_masks_video_only_with_hint(self):
        from audyssey.contracts import VisualHint
        from unittest.mock import Mock
        self.ctx.settings.visual_prompt = True
        extract_audio(self.directory / 'input.mp4', self.directory / 'mono.wav', True)
        self.job.mono_path = 'mono.wav'
        mask = Mock(side_effect=lambda video, hint, out: out)
        separate = Mock(wraps=self.ctx.tools.separate)
        self.ctx.tools = replace(self.ctx.tools, mask_video=mask, separate=separate)
        source = Source(id='s1', plan=SourcePlan(name='a', kind='sfx', description='sound'))
        graph.process_source(self.job, source, self.ctx)
        mask.assert_not_called()
        self.assertIsNone(separate.call_args.args[2])
        source.plan.visual_hint = VisualHint(time=1, box=(0, 0, .5, 1), description='a')
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(separate.call_args.args[2], self.directory / 'masks/s1.mp4')
        mask.assert_called_once_with(self.directory / 'input.mp4', source.plan.visual_hint, self.directory / 'masks/s1.mp4')

    def test_visual_prompt_off_ignores_hints(self):
        from audyssey.contracts import VisualHint
        from unittest.mock import Mock
        hint = VisualHint(time=1, box=(0, 0, .5, 1), description='a')
        locate, mask = Mock(), Mock()
        separate = Mock(wraps=self.ctx.tools.separate)
        self.ctx.tools = replace(
            self.ctx.tools,
            plan=lambda *args: [SourcePlan(name='a', kind='sfx', description='sound', visual_hint=hint)],
            locate=locate, mask_video=mask, separate=separate,
        )
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, 'ready', job.error)
        locate.assert_not_called()
        mask.assert_not_called()
        self.assertIsNone(separate.call_args.args[2])

    def test_mask_reused_across_retries_with_same_hint(self):
        from audyssey.contracts import FilterResult, VisualHint
        from unittest.mock import Mock
        self.ctx.settings.visual_prompt = True
        extract_audio(self.directory / 'input.mp4', self.directory / 'mono.wav', True)
        self.job.mono_path = 'mono.wav'
        mask = Mock(side_effect=lambda video, hint, out: out)
        judge = Mock(return_value=FilterResult(passed=False, reason='low'))
        self.ctx.tools = replace(self.ctx.tools, mask_video=mask, judge=judge)
        hint = VisualHint(time=1, box=(0, 0, .5, 1), description='a')
        source = Source(id='s1', plan=SourcePlan(name='a', kind='sfx', description='sound', visual_hint=hint))
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(source.attempts, self.ctx.settings.max_separation_retries + 1)
        mask.assert_called_once()
        moved = hint.model_copy(update={'time': 2})
        judge.return_value = FilterResult(passed=False, suggestion=source.plan.model_copy(update={'visual_hint': moved}))
        source.attempts = 0
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual([c.args[1] for c in mask.call_args_list], [hint, hint, moved])

    def test_mask_failure_fails_only_that_source(self):
        self.ctx.settings.visual_prompt = True
        from audyssey.contracts import VisualHint
        original = self.ctx.tools.plan
        def plan(*args):
            plans = original(*args)
            plans[0].visual_hint = VisualHint(time=1, box=(0, 0, .5, 1), description='a')
            return plans
        def fail(*args):
            raise RuntimeError('mask failed')
        self.ctx.tools = replace(self.ctx.tools, plan=plan, mask_video=fail)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, 'ready')
        self.assertEqual([s.status for s in job.sources], ['failed', 'done', 'done'])

    def test_preprocess_marks_silent_without_audio_stream(self):
        import json
        make_video(self.directory / self.job.video_path, seconds=1, audio=False)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertTrue(job.silent)
        for filename, channels in [("audio.wav", 2), ("mono.wav", 1)]:
            data = read_wav(self.directory / filename)
            self.assertEqual(data.shape, (48000, channels))
            np.testing.assert_array_equal(data, 0)
        self.assertEqual(json.loads((self.directory / "peaks.json").read_text()), [0] * 800)
        self.assertIsNone(job.residual.path)

    def test_preprocess_marks_silent_for_near_silent_audio(self):
        import subprocess
        out = self.directory / "quiet.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.directory / self.job.video_path),
                        "-af", "volume=0.0025", "-c:v", "copy", str(out)], check=True, capture_output=True)
        self.job.video_path = "quiet.mp4"
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertTrue(job.silent)
        self.assertIsNone(job.residual.path)

    def test_preprocess_keeps_normal_audio(self):
        job = graph.analyze(self.job, self.ctx)
        self.assertFalse(job.silent)
        self.assertTrue((self.directory / job.residual.path).is_file())

    def generation_setup(self, results):
        from unittest.mock import Mock
        from audyssey.media import write_wav
        source = Source(id="s1", origin="generated", plan=SourcePlan(name="rain", kind="sfx", description="rain"))
        self.job.sources = [source]
        generate = Mock(side_effect=lambda video, plan, out, seed: write_wav(out, np.full((100, 1), seed / 100)))
        score = Mock(side_effect=results)
        self.ctx.tools = replace(self.ctx.tools, generate=generate, score_generation=score)
        return source, generate, score

    def test_generation_retries_then_keeps_best_margin(self):
        from audyssey.contracts import FilterResult
        source, _, _ = self.generation_setup([FilterResult(passed=False, scores={"text_margin": margin}) for margin in [0.1, 0.3, 0.2]])
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(source.status, "done")
        self.assertEqual(source.filter.scores["text_margin"], 0.3)
        np.testing.assert_allclose(read_wav(self.directory / source.track_path), 0.43)
        self.assertEqual([p.name for p in (self.directory / "tracks").iterdir()], ["s1.wav"])
        message = next(s.message for s in self.job.steps if s.key == "score:s1")
        for n in [1, 2, 3]:
            self.assertIn(f"attempt {n}:", message)

    def test_generation_seed_counts_attempts(self):
        from audyssey.contracts import FilterResult
        source, generate, _ = self.generation_setup([FilterResult(passed=False)] * 6)
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual([c.args[3] for c in generate.call_args_list], [42, 43, 44])
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual([c.args[3] for c in generate.call_args_list], [42, 43, 44, 45, 46, 47])
        self.assertEqual(source.attempts, 6)

    def test_generation_stops_on_pass(self):
        from audyssey.contracts import FilterResult
        source, generate, _ = self.generation_setup([FilterResult(passed=True)])
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(source.status, "done")
        self.assertTrue(source.filter.passed)

    def test_generation_no_margin_keeps_passing_attempt(self):
        from audyssey.contracts import FilterResult
        source, _, _ = self.generation_setup([FilterResult(passed=False), FilterResult(passed=True)])
        graph.process_source(self.job, source, self.ctx)
        self.assertTrue(source.filter.passed)
        np.testing.assert_allclose(read_wav(self.directory / source.track_path), 0.43)

    def test_generation_siblings_and_reference(self):
        from audyssey.contracts import FilterResult
        source, _, score = self.generation_setup([FilterResult(passed=True)])
        original = Source(id="s2", plan=SourcePlan(name="original", kind="sfx", description="rain"), track_path="original.wav")
        other = Source(id="s3", plan=SourcePlan(name="other", kind="music", description="music"))
        dialogue = Source(id="s4", plan=SourcePlan(name="voice", kind="dialogue", description="voice"))
        source.replaces = original.id
        self.job.sources += [original, other, dialogue]
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(score.call_args.args[2], [other.plan])
        self.assertEqual(score.call_args.args[3], self.directory / "original.wav")

    def test_generation_error_isolated_and_attempt_files_removed(self):
        from unittest.mock import Mock
        source, _, _ = self.generation_setup([])
        self.ctx.tools = replace(self.ctx.tools, score_generation=Mock(side_effect=RuntimeError("score failed")))
        graph.process_source(self.job, source, self.ctx)
        self.assertEqual(source.status, "failed")
        self.assertIsNone(source.track_path)
        self.assertEqual(list((self.directory / "tracks").glob("*.attempt*.wav")), [])
        self.assertEqual(next(s.status for s in self.job.steps if s.key == "score:s1"), "failed")

    def test_silent_job_generates_non_dialogue_sources(self):
        from unittest.mock import Mock
        make_video(self.directory / self.job.video_path, seconds=1, audio=False)
        separate = Mock(side_effect=AssertionError("silent job must not separate"))
        self.ctx.tools = replace(self.ctx.tools, separate=separate)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertTrue(all(s.origin == "generated" for s in job.sources))
        for source in job.sources:
            if source.plan.kind == "dialogue":
                self.assertEqual(source.status, "skipped")
                self.assertIsNone(source.track_path)
                self.assertIn("dialogue is not generated", next(s.message for s in job.steps if s.key == f"generate:{source.id}"))
            else:
                self.assertEqual(source.status, "done")
                self.assertTrue(source.filter.passed)
        separate.assert_not_called()
        self.assertIsNone(job.residual.path)
        graph.handle_command(job, Render(), self.ctx)
        self.assertEqual(read_wav(self.directory / job.outputs.mix_wav).shape, (48000, 2))
        self.assertTrue((self.directory / job.outputs.video_mp4).is_file())

    def test_silent_job_with_only_dialogue_is_ready(self):
        make_video(self.directory / self.job.video_path, seconds=1, audio=False)
        self.ctx.tools = replace(self.ctx.tools, plan=lambda *args: [SourcePlan(name="voice", kind="dialogue", description="voice")])
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertEqual(job.sources[0].status, "skipped")
        graph.handle_command(job, Render(), self.ctx)
        self.assertEqual(job.outputs.stems, [])
        np.testing.assert_array_equal(read_wav(self.directory / job.outputs.mix_wav), 0)

    def test_refresh_residual_ignores_generated_sources(self):
        from audyssey.media import write_wav
        job = graph.analyze(self.job, self.ctx)
        original_residual = read_wav(self.directory / job.residual.path)
        write_wav(self.directory / "generated.wav", np.ones((len(original_residual), 1)))
        job.sources.append(Source(id="s4", origin="generated", plan=SourcePlan(name="extra", kind="sfx", description="extra"), track_path="generated.wav"))
        job.sources[0].keep = False
        graph.refresh_residual(job, self.ctx)
        np.testing.assert_array_equal(read_wav(self.directory / job.residual.path), original_residual)

    def test_sound_job_unchanged(self):
        from unittest.mock import Mock
        generate, score = Mock(), Mock()
        self.ctx.tools = replace(self.ctx.tools, generate=generate, score_generation=score)
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertTrue(all(s.origin == "separated" for s in job.sources))
        generate.assert_not_called()
        score.assert_not_called()

    def test_generate_replaces_separated_source(self):
        from audyssey.contracts import Generate
        job = graph.analyze(self.job, self.ctx)
        original = job.sources[2]
        path = original.track_path
        graph.handle_command(job, Generate(source_id=original.id), self.ctx)
        generated = job.sources[-1]
        self.assertEqual((generated.origin, generated.replaces, generated.plan.name), ("generated", original.id, original.plan.name + " (generated)"))
        self.assertFalse(original.keep)
        self.assertEqual(original.track_path, path)
        self.assertEqual(generated.status, "done")

    def test_generate_twice_reuses_replacement(self):
        from audyssey.contracts import Generate
        from unittest.mock import Mock
        job = graph.analyze(self.job, self.ctx)
        generate = Mock(wraps=self.ctx.tools.generate)
        self.ctx.tools = replace(self.ctx.tools, generate=generate)
        graph.handle_command(job, Generate(source_id="s3"), self.ctx)
        replacement = job.sources[-1]
        graph.handle_command(job, Generate(source_id="s3"), self.ctx)
        self.assertEqual(len(job.sources), 4)
        self.assertIs(job.sources[-1], replacement)
        self.assertEqual([c.args[3] for c in generate.call_args_list], [42, 43])
        self.assertEqual(replacement.attempts, 2)

    def test_generate_regenerates_generated_source(self):
        from audyssey.contracts import Generate
        job = graph.analyze(self.job, self.ctx)
        graph.handle_command(job, Generate(source_id="s3"), self.ctx)
        source = job.sources[-1]
        previous = source.attempts
        graph.handle_command(job, Generate(source_id=source.id, description="new sound"), self.ctx)
        self.assertEqual(source.plan.description, "new sound")
        self.assertGreater(source.attempts, previous)

    def test_generate_does_not_change_residual(self):
        from audyssey.contracts import Generate
        job = graph.analyze(self.job, self.ctx)
        residual = self.directory / job.residual.path
        before, modified = residual.read_bytes(), residual.stat().st_mtime_ns
        graph.handle_command(job, Generate(source_id="s3"), self.ctx)
        graph.handle_command(job, Keep(source_id="s3"), self.ctx)
        self.assertEqual(residual.read_bytes(), before)
        self.assertEqual(residual.stat().st_mtime_ns, modified)
        self.assertTrue(job.sources[2].keep and job.sources[-1].keep)
        graph.handle_command(job, Render(), self.ctx)
        original = read_wav(self.directory / job.audio_path)
        generated = read_wav(self.directory / job.sources[-1].track_path)
        expected = original.copy()
        expected[:len(generated)] += generated
        np.testing.assert_allclose(read_wav(self.directory / job.outputs.mix_wav), expected, atol=1e-6)

    def test_generation_commands_reject_invalid_targets_without_mutation(self):
        from audyssey.contracts import Generate
        job = graph.analyze(self.job, self.ctx)
        commands = [Generate(source_id="s1"), Generate(source_id="residual"),
                    AddSource(method="generate", plan=SourcePlan(name="voice", kind="dialogue", description="voice"))]
        for command in commands:
            snapshot = job.model_dump_json()
            with self.subTest(command=command), self.assertRaises(ValueError):
                graph.handle_command(job, command, self.ctx)
            self.assertEqual(job.model_dump_json(), snapshot)
        graph.handle_command(job, Generate(source_id="s3"), self.ctx)
        snapshot = job.model_dump_json()
        with self.assertRaises(ValueError):
            graph.handle_command(job, Reseparate(source_id=job.sources[-1].id), self.ctx)
        self.assertEqual(job.model_dump_json(), snapshot)
        job.silent = True
        for command in [Reseparate(source_id="s3"), AddSource(plan=SourcePlan(name="new", kind="sfx", description="sound"))]:
            snapshot = job.model_dump_json()
            with self.assertRaises(ValueError):
                graph.handle_command(job, command, self.ctx)
            self.assertEqual(job.model_dump_json(), snapshot)

    def test_add_source_generate(self):
        from unittest.mock import Mock
        job = graph.analyze(self.job, self.ctx)
        before = (self.directory / job.residual.path).read_bytes()
        separate = Mock(side_effect=AssertionError("generation must not separate"))
        self.ctx.tools = replace(self.ctx.tools, separate=separate)
        graph.handle_command(job, AddSource(method="generate", plan=SourcePlan(name="extra", kind="sfx", description="rain")), self.ctx)
        self.assertEqual(job.sources[-1].origin, "generated")
        self.assertEqual(job.sources[-1].status, "done")
        self.assertIsNone(job.sources[-1].replaces)
        separate.assert_not_called()
        self.assertEqual((self.directory / job.residual.path).read_bytes(), before)

    def test_generate_rejects_empty_description_without_mutation(self):
        from audyssey.contracts import Generate
        job = graph.analyze(self.job, self.ctx)
        snapshot = job.model_dump_json()
        with self.assertRaises(ValueError):
            graph.handle_command(job, Generate(source_id="s3", description=" "), self.ctx)
        self.assertEqual(job.model_dump_json(), snapshot)

    def test_preprocess_keeps_audible_opposite_polarity_stereo(self):
        import subprocess
        from audyssey.media import write_wav
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
        stereo = np.column_stack((tone, -tone))
        audio = write_wav(self.root / "opposite.wav", stereo)
        video = self.directory / "opposite.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.directory / self.job.video_path),
                        "-i", str(audio), "-map", "0:v", "-map", "1:a", "-t", "1",
                        "-c:v", "copy", "-c:a", "pcm_f32le", str(video)], check=True, capture_output=True)
        self.job.video_path = video.name
        job = graph.analyze(self.job, self.ctx)
        self.assertEqual(job.state, "ready", job.error)
        self.assertFalse(job.silent)
        np.testing.assert_array_equal(read_wav(self.directory / job.mono_path), 0)
        np.testing.assert_allclose(read_wav(self.directory / job.audio_path), stereo, atol=1e-7)
        self.assertIsNotNone(job.residual.path)
        self.assertTrue(all(s.origin == "separated" for s in job.sources))
