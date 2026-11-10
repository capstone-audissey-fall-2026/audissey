import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from uuid import uuid4

from audyssey.contracts import Job

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def create(self, filename: str, data: bytes) -> tuple[Job, Path]:
        job_id = uuid4().hex[:12]
        directory = self.dir(job_id)
        directory.mkdir()
        # The stored name decides the served Content-Type; never keep e.g. ".html".
        suffix = Path(filename).suffix.lower()
        video_path = "input" + (suffix if suffix in VIDEO_SUFFIXES else ".mp4")
        (directory / video_path).write_bytes(data)
        job = self.save(Job(id=job_id, video_path=video_path))
        return job, directory

    def dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            raise KeyError(job_id)
        directory = (self.root / job_id).resolve()
        if directory.parent != self.root:
            raise KeyError(job_id)
        return directory

    def save(self, job: Job) -> Job:
        with self._lock:
            directory = self.dir(job.id)
            job.version += 1
            with NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
                stream.write(job.model_dump_json())
                temporary = Path(stream.name)
            temporary.replace(directory / "job.json")
        return job

    def load(self, job_id: str) -> Job:
        try:
            return Job.model_validate_json((self.dir(job_id) / "job.json").read_text())
        except FileNotFoundError as exc:
            raise KeyError(job_id) from exc

    def list(self) -> list[Job]:
        paths = sorted(
            self.root.glob("*/job.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        return [self.load(path.parent.name) for path in paths]

    def resolve(self, job_id: str, rel: str) -> Path:
        directory = self.dir(job_id)
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise KeyError(rel)
        return path
