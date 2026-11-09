import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2

from audyssey.contracts import PlanIssue, SourcePlan
from audyssey.tools.planning import real
from audyssey.tools.planning.llm import chat_client


class PlannerRealTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.image = self.root / "frame.jpg"
        self.image.write_bytes(b"\xff\xd8test-jpeg")
        self.video, self.audio = self.root / "video.mp4", self.root / "audio.wav"
        self.video.write_bytes(b"private-video-bytes")
        self.audio.write_bytes(b"private-audio-bytes")
        self.requests = []
        self.responses = []
        env = patch.dict(os.environ, {
            "AUDYSSEY_PLANNER_BASE_URL": "http://test.invalid/v1",
            "AUDYSSEY_PLANNER_MODEL": "fake-model",
            "AUDYSSEY_PLANNER_API_KEY": "fake-key",
            "AUDYSSEY_PLANNER_FRAMES": "2",
            "LANGSMITH_TRACING": "false",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        frames = patch("audyssey.media.extract_frames", return_value=[
            (0.25, self.image), (1.25, self.image),
        ])
        frames.start()
        self.addCleanup(frames.stop)
        # The real SDK serializes the request; only the network is replaced.
        client = patch("audyssey.tools.planning.llm.chat_client", side_effect=lambda settings: chat_client(
            settings, transport=httpx2.MockTransport(self.respond)
        ))
        client.start()
        self.addCleanup(client.stop)

    def respond(self, request):
        self.requests.append(json.loads(request.content))
        content = self.responses.pop(0)
        return httpx2.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "fake-model", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
        })

    def source(self, **updates):
        return {"name": "person", "kind": "dialogue", "description": "a person speaking", **updates}

    def plan(self, sources, feedback=None):
        self.responses.append(json.dumps({"sources": sources}))
        return real.plan(self.video, self.audio, 2, feedback or [])

    def test_plan_parses_sources_and_clamps(self):
        result = self.plan([self.source(
            spans=[{"start": -1, "end": 5}, {"start": 3, "end": 4}, {"start": 1, "end": 0}],
            visual_hint={"time": 5, "box": [-0.1, 0.2, 1.5, 0.8], "description": "person"},
        )])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SourcePlan)
        self.assertEqual([(s.start, s.end) for s in result[0].spans], [(0, 2)])
        self.assertEqual(result[0].visual_hint.box, (0, 0.2, 1, 0.8))
        self.assertEqual(result[0].visual_hint.time, 2)
        result = self.plan([self.source(visual_hint={
            "time": 0, "box": [2, 0, 3, 1], "description": "invalid box",
        })])
        self.assertIsNone(result[0].visual_hint)

    def test_plan_retries_once_on_bad_json(self):
        self.responses.append("broken")
        self.assertEqual(self.plan([self.source()])[0].name, "person")
        self.assertEqual(len(self.requests), 2)
        self.requests.clear()
        for bad in ["x" * 600, '{}', '{"sources": null}', '[]', None]:
            with self.subTest(bad=bad):
                self.responses = [bad, bad]
                with self.assertRaisesRegex(RuntimeError, "planner returned invalid JSON") as caught:
                    real.plan(self.video, self.audio, 2, [])
                self.assertLessEqual(len(str(caught.exception)), 532)
        self.assertEqual(len(self.requests), 10)

    def test_plan_drops_invalid_items_and_errors_when_none_left(self):
        invalid = [None, 3, {}, self.source(kind="unknown"), self.source(spans="bad"), self.source(spans=[{"start": float("nan"), "end": 1}])]
        self.assertEqual([s.name for s in self.plan([*invalid, self.source()])], ["person"])
        for sources in [invalid, []]:
            with self.assertRaisesRegex(RuntimeError, "no valid sources"):
                self.plan(sources)

    def test_plan_includes_feedback_and_frame_times_in_prompt(self):
        feedback = [PlanIssue(source_name="ghost", kind="missing", message="remove ghost exactly")]
        self.plan([self.source()], feedback)
        request = self.requests[0]
        self.assertEqual(request["model"], "fake-model")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        parts = request["messages"][1]["content"]
        text = "\n".join(part["text"] for part in parts if part["type"] == "text")
        self.assertIn(json.dumps([feedback[0].model_dump()]), text)
        for expected in ["0.25", "1.25", "2"]:
            self.assertIn(expected, text)

    def test_plan_sends_images_not_video_or_audio(self):
        self.plan([self.source()])
        parts = self.requests[0]["messages"][1]["content"]
        images = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        for url in images:
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",")[1]), self.image.read_bytes())
        wire = json.dumps(self.requests)
        for data in [self.video.read_bytes(), self.audio.read_bytes()]:
            self.assertNotIn(data.decode(), wire)
            self.assertNotIn(base64.b64encode(data).decode(), wire)
