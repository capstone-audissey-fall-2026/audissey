import tempfile
import unittest
from pathlib import Path
from audyssey.store import JobStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = JobStore(Path(self.tmp.name))

    def test_create_and_load(self):
        job, directory = self.store.create("clip.mp4", b"video")
        self.assertEqual(job.video_path, "input.mp4")
        self.assertEqual((directory / job.video_path).read_bytes(), b"video")
        self.assertEqual(self.store.load(job.id), job)

    def test_save_increments_version(self):
        job, _ = self.store.create("clip.mp4", b"video")
        version = job.version
        self.store.save(job)
        self.assertEqual(self.store.load(job.id).version, version + 1)

    def test_list_newest_first(self):
        a, _ = self.store.create("a.mp4", b"a")
        b, _ = self.store.create("b.mp4", b"b")
        self.assertEqual([j.id for j in self.store.list()], [b.id, a.id])
        self.store.save(a)
        self.assertEqual(self.store.list()[0].id, a.id)

    def test_resolve_rejects_traversal(self):
        job, directory = self.store.create("clip.mp4", b"video")
        (directory / "escape").symlink_to("/etc/passwd")
        for rel in ["../x", "/etc/passwd", "missing", "escape"]:
            with self.subTest(rel=rel), self.assertRaises(KeyError):
                self.store.resolve(job.id, rel)
        self.assertEqual(
            self.store.resolve(job.id, "input.mp4"), directory / "input.mp4"
        )

    def test_load_rejects_bad_id(self):
        with self.assertRaises(KeyError):
            self.store.load("../..")
