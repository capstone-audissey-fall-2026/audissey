from importlib import import_module
from pathlib import Path
from secrets import compare_digest
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from audyssey.contracts import FilterResult, SourcePlan

# ponytail: one GPU model, one request at a time; add a queue or batching if throughput matters
_model_lock = Lock()


def create_tool_app(
    tools: list[Literal["separation", "filtering"]],
    impl: Literal["stub", "real"],
    token: str,
) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token and request.url.path != "/health":
            supplied = request.headers.get("authorization", "").encode()
            if not compare_digest(supplied, f"Bearer {token}".encode()):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str | list[Literal["separation", "filtering"]]]:
        return {"tools": tools, "impl": impl}

    def parse_plan(raw: str) -> SourcePlan:
        try:
            return SourcePlan.model_validate_json(raw)
        except ValidationError:
            raise HTTPException(422, "invalid source plan") from None

    if "separation" in tools:
        separation = import_module(f"audyssey.tools.separation.{impl}")

        @app.post("/separate")
        def separate(
            mono: UploadFile,
            video: UploadFile,
            plan: Annotated[str, Form()],
            attempt: Annotated[int, Form()],
        ) -> Response:
            source_plan = parse_plan(plan)
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                mono_path, video_path = root / "mono.wav", root / "video.mp4"
                mono_path.write_bytes(mono.file.read())
                video_path.write_bytes(video.file.read())
                with _model_lock:
                    out = separation.separate(
                        mono_path, source_plan, video_path, root / "track.wav", attempt
                    )
                return Response(out.read_bytes(), media_type="audio/wav")

    if "filtering" in tools:
        filtering = import_module(f"audyssey.tools.filtering.{impl}")

        @app.post("/judge")
        def judge(
            mono: UploadFile, track: UploadFile, plan: Annotated[str, Form()]
        ) -> FilterResult:
            source_plan = parse_plan(plan)
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                mono_path, track_path = root / "mono.wav", root / "track.wav"
                mono_path.write_bytes(mono.file.read())
                track_path.write_bytes(track.file.read())
                with _model_lock:
                    return filtering.judge(mono_path, track_path, source_plan)

    return app
