from importlib import import_module
from pathlib import Path
from secrets import compare_digest
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter, ValidationError

from audyssey.contracts import FilterResult, GroundResult, SourcePlan, VisualHint
from audyssey.settings import Settings
from audyssey.uploads import save_upload

# ponytail: one GPU model, one request at a time; add a queue or batching if throughput matters
_model_lock = Lock()


def create_tool_app(
    tools: list[Literal["separation", "filtering", "grounding", "generation"]],
    impl: Literal["stub", "real"],
    token: str,
) -> FastAPI:
    app = FastAPI()
    limit = int(Settings().max_upload_mb * 2**20)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token and request.url.path != "/health":
            supplied = request.headers.get("authorization", "").encode()
            if not compare_digest(supplied, f"Bearer {token}".encode()):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[
        str, str | list[Literal["separation", "filtering", "grounding", "generation"]]
    ]:
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
            plan: Annotated[str, Form()],
            attempt: Annotated[int, Form()],
            video: UploadFile | None = None,
        ) -> Response:
            source_plan = parse_plan(plan)
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                mono_path, video_path = root / "mono.wav", None
                save_upload(mono, mono_path, limit)
                if video is not None:
                    video_path = save_upload(video, root / "video.mp4", limit)
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
                save_upload(mono, mono_path, limit)
                save_upload(track, track_path, limit)
                with _model_lock:
                    return filtering.judge(mono_path, track_path, source_plan)

    if "grounding" in tools:
        grounding = import_module(f"audyssey.tools.grounding.{impl}")

        def parse_hint(raw: str) -> VisualHint:
            try:
                return VisualHint.model_validate_json(raw)
            except ValidationError:
                raise HTTPException(422, "invalid visual hint") from None

        @app.post("/locate")
        def locate(video: UploadFile, hint: Annotated[str, Form()]) -> GroundResult:
            visual_hint = parse_hint(hint)
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "video.mp4"
                save_upload(video, path, limit)
                with _model_lock:
                    return grounding.locate(path, visual_hint)

        @app.post("/mask")
        def mask(video: UploadFile, hint: Annotated[str, Form()]) -> Response:
            visual_hint = parse_hint(hint)
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "video.mp4"
                save_upload(video, path, limit)
                with _model_lock:
                    out = grounding.mask_video(
                        path, visual_hint, Path(tmp) / "mask.mp4"
                    )
                return Response(out.read_bytes(), media_type="video/mp4")

    if "generation" in tools:
        generation = import_module(f"audyssey.tools.generation.{impl}")

        @app.post("/generate")
        def generate(
            video: UploadFile,
            plan: Annotated[str, Form()],
            seed: Annotated[int, Form()],
        ) -> Response:
            source_plan = parse_plan(plan)
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                video_path = save_upload(video, root / "video.mp4", limit)
                with _model_lock:
                    out = generation.generate(
                        video_path, source_plan, root / "track.wav", seed
                    )
                return Response(out.read_bytes(), media_type="audio/wav")

        @app.post("/score")
        def score(
            track: UploadFile,
            plan: Annotated[str, Form()],
            siblings: Annotated[str, Form()],
            reference: UploadFile | None = None,
        ) -> FilterResult:
            source_plan = parse_plan(plan)
            try:
                sibling_plans = TypeAdapter(list[SourcePlan]).validate_json(siblings)
            except ValidationError:
                raise HTTPException(422, "invalid sibling plans") from None
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                track_path = save_upload(track, root / "track.wav", limit)
                reference_path = (
                    save_upload(reference, root / "reference.wav", limit)
                    if reference is not None
                    else None
                )
                with _model_lock:
                    return generation.score_generation(
                        track_path, source_plan, sibling_plans, reference_path
                    )

    return app
