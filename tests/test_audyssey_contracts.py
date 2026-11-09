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
