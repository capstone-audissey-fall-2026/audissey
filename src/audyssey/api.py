import asyncio
from importlib.resources import files
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from audyssey.contracts import AddSource, Job, Render, Reseparate, parse_command
from audyssey.graph import Context, analyze, handle_command
from audyssey.media import MediaError, check_upload
from audyssey.settings import Settings
from audyssey.store import JobStore
from audyssey.uploads import save_upload
from audyssey.tools import Tools, load_tools


def create_app(settings: Settings | None = None, tools: Tools | None = None) -> FastAPI:
    settings = settings or Settings()
    ctx = Context(tools or load_tools(settings), settings, JobStore(settings.work_dir))
    app = FastAPI()
    app.state.ctx = ctx
    locks: dict[str, Lock] = {}
    jobs: dict[str, Job] = {}
    app.state.locks, app.state.jobs = locks, jobs

    def load(job_id: str) -> Job:
        try:
            return ctx.store.load(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc

    @app.post("/api/jobs", status_code=201)
    def upload(file: UploadFile) -> Job:
        with TemporaryDirectory() as tmp:
            limit = int(settings.max_upload_mb * 2**20)
            path = save_upload(file, Path(tmp) / "upload", limit)
            try:
                duration = check_upload(path, settings.max_duration_s)
            except MediaError as exc:
                raise HTTPException(400, str(exc)) from exc
            data = path.read_bytes()
        job, _ = ctx.store.create(file.filename or "video.mp4", data)
        job.duration = duration
        ctx.store.save(job)
        jobs[job.id] = job
        lock = locks.setdefault(job.id, Lock())
        lock.acquire()
        snapshot = job.model_copy(deep=True)

        # Release the lock before the final state is saved, so a client that sees
        # ready/failed can run an execution command at once.
        Thread(target=analyze, args=(job, ctx, lock.release), daemon=True).start()
        return snapshot

    @app.get("/api/jobs")
    def list_jobs() -> list[Job]:
        return ctx.store.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        return load(job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request) -> StreamingResponse:
        initial = load(job_id)

        async def stream():
            job = initial
            version = job.version
            yield f"data: {job.model_dump_json()}\n\n"
            # ponytail: polling store, push from save() if latency matters
            while not await request.is_disconnected():
                await asyncio.sleep(0.25)
                job = load(job_id)
                if job.version != version:
                    version = job.version
                    yield f"data: {job.model_dump_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/jobs/{job_id}/commands")
    def command(job_id: str, data: dict = Body(...)) -> Job:
        snapshot = load(job_id)
        try:
            cmd = parse_command(data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Share the live job so edits made during analysis survive later saves.
        job = jobs.setdefault(job_id, snapshot)
        lock = locks.setdefault(job_id, Lock())
        execution = isinstance(cmd, (Reseparate, AddSource, Render))
        if execution and not lock.acquire(blocking=False):
            raise HTTPException(409, "job is busy")
        try:
            return handle_command(job, cmd, ctx).model_copy(deep=True)
        except KeyError as exc:
            raise HTTPException(404, "source not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if execution:
                lock.release()

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def get_file(job_id: str, path: str) -> FileResponse:
        try:
            return FileResponse(ctx.store.resolve(job_id, path))
        except KeyError as exc:
            raise HTTPException(404, "file not found") from exc

    @app.get("/")
    def root() -> HTMLResponse:
        return HTMLResponse((files("audyssey.devui") / "index.html").read_text())

    return app
