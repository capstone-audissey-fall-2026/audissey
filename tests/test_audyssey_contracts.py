import unittest

from pydantic import ValidationError
from audyssey.contracts import (
    Span,
    VisualHint,
    SourcePlan,
    Source,
    Residual,
    Job,
    SetGain,
    Render,
    parse_command,
)


class ContractsTests(unittest.TestCase):
    def test_span_rejects_end_before_start(self):
        for start, end in [(2, 1), (1, 1), (-1, 2)]:
            with self.assertRaises(ValidationError):
                Span(start=start, end=end)

    def test_visual_hint_box_must_be_normalized(self):
        with self.assertRaises(ValidationError):
            VisualHint(time=0, box=(0, 0, 1.5, 1), description="a")

    def test_job_roundtrip_json(self):
        job = Job(
            id="a",
            video_path="input.mp4",
            sources=[
                Source(
                    id="s1", plan=SourcePlan(name="a", kind="sfx", description="sound")
                )
            ],
            residual=Residual(path="residual.wav"),
        )
        self.assertEqual(Job.model_validate_json(job.model_dump_json()), job)

    def test_parse_command_dispatches_on_type(self):
        self.assertIsInstance(
            parse_command(dict(type="set_gain", source_id="s1", gain=0.5)), SetGain
        )
        self.assertEqual(parse_command(dict(type="render")), Render())

    def test_command_bounds(self):
        for data in [
            dict(type="set_gain", source_id="s1", gain=2.5),
            dict(type="set_pitch", source_id="s1", semitones=13),
            dict(type="unknown"),
            dict(type="render", extra=1),
        ]:
            with self.assertRaises(ValidationError):
                parse_command(data)

    def test_find_source(self):
        source = Source(id="s1", plan=SourcePlan(name="a", kind="sfx", description="a"))
        job = Job(id="a", video_path="input.mp4", sources=[source])
        self.assertEqual(job.find_source("s1"), source)
        self.assertIsNone(job.find_source("s99"))

    def test_ground_result_roundtrip_and_bounds(self):
        from audyssey.contracts import GroundResult
        result = GroundResult(found=True, box=(0, 0.2, 1, 0.8), score=0.9)
        self.assertEqual(GroundResult.model_validate_json(result.model_dump_json()), result)
        self.assertIsNone(GroundResult(found=False).box)
        for box in [(-0.1, 0, 1, 1), (0.8, 0, 0.2, 1)]:
            with self.assertRaises(ValidationError):
                GroundResult(found=True, box=box)

    def test_generate_command_parses(self):
        from audyssey.contracts import Generate
        command = parse_command({"type": "generate", "source_id": "s1"})
        self.assertIsInstance(command, Generate)
        self.assertIsNone(command.description)

    def test_add_source_method_defaults_to_separate(self):
        command = parse_command({"type": "add_source", "plan": {"name": "new", "kind": "sfx", "description": "sound"}})
        self.assertEqual(command.method, "separate")
        with self.assertRaises(ValidationError):
            parse_command({"type": "add_source", "method": "unknown", "plan": command.plan.model_dump()})

    def test_source_origin_defaults_to_separated(self):
        source = Source(id="s1", plan=SourcePlan(name="a", kind="sfx", description="sound"))
        self.assertEqual(source.origin, "separated")
        self.assertIsNone(source.replaces)
        generated = Source.model_validate({**source.model_dump(), "origin": "generated", "replaces": "s2", "status": "skipped"})
        self.assertEqual((generated.origin, generated.replaces, generated.status), ("generated", "s2", "skipped"))

    def test_old_job_json_still_loads(self):
        job = Job.model_validate_json('{"id":"old","video_path":"input.mp4","sources":[{"id":"s1","plan":{"name":"a","kind":"sfx","description":"sound"}}]}')
        self.assertFalse(job.silent)
        self.assertEqual(job.sources[0].origin, "separated")
        self.assertIsNone(job.sources[0].replaces)
