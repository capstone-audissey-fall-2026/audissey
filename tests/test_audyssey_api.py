import json
import shutil
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import httpx
import uvicorn
from fastapi.testclient import TestClient
from tests.audyssey_media import make_video
from audyssey.api import create_app
from audyssey.settings import Settings


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = make_video(self.root / "clip.mp4")
        self.app = create_app(Settings(work_dir=self.root / "jobs"))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def upload(self):
        response = self.client.post(
            "/api/jobs",
            files={"file": ("clip.mp4", self.video.read_bytes(), "video/mp4")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ready(self):
        job_id = self.upload()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["state"] != "running":
                self.assertEqual(job["state"], "ready", job)
                return job
            time.sleep(0.1)
        self.fail("analysis timed out")

    def test_upload_runs_analysis_to_ready(self):
        self.assertEqual(len(self.ready()["sources"]), 3)
        self.assertEqual(len(self.client.get("/api/jobs").json()), 1)

    def test_upload_rejects_no_audio(self):
        silent = make_video(self.root / "silent.mp4", audio=False)
        response = self.client.post(
            "/api/jobs", files={"file": ("silent.mp4", silent.read_bytes())}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("audio stream", response.json()["detail"])
        self.assertEqual(self.client.get("/api/jobs").json(), [])

    def test_upload_rejects_too_long(self):
        self.app.state.ctx.settings.max_duration_s = 2
        response = self.client.post(
            "/api/jobs", files={"file": ("clip.mp4", self.video.read_bytes())}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("maximum is 2s", response.json()["detail"])
        self.assertEqual(self.client.get("/api/jobs").json(), [])

    def test_files_traversal_404(self):
        job = self.ready()
        for path in ["..%2F..%2Fetc%2Fpasswd", "missing.wav"]:
            self.assertEqual(
                self.client.get(f"/api/jobs/{job['id']}/files/{path}").status_code, 404
            )
        self.assertGreater(
            len(self.client.get(f"/api/jobs/{job['id']}/files/audio.wav").content), 0
        )
        self.assertEqual(self.client.get("/api/jobs/000000000000").status_code, 404)

    def test_events_first_message_is_job(self):
        job = self.ready()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(self.app, log_level="error"))
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            thread.start()
            try:
                deadline = time.monotonic() + 5
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.01)
                with httpx.stream(
                    "GET",
                    f"http://127.0.0.1:{port}/api/jobs/{job['id']}/events",
                    timeout=5,
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("text/event-stream", response.headers["content-type"])
                    line = next(response.iter_lines())
                    self.assertEqual(
                        json.loads(line.removeprefix("data: "))["id"], job["id"]
                    )
            finally:
                server.should_exit = True
                thread.join(timeout=5)
