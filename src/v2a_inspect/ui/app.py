from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .auth import install_auth
from .routes import create_router
from .store import VideoAssetStore


def create_app() -> FastAPI:
    store = VideoAssetStore()
    app = FastAPI(title="v2a-inspect-ui")
    install_auth(app)
    app.state.video_asset_store = store
    app.include_router(create_router(store))

    dist_dir = _frontend_dist_dir()
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        index_path = dist_dir / "index.html"
        if not index_path.exists():
            return FileResponse(_dev_index_path())
        return FileResponse(index_path)

    @app.get("/favicon.ico")
    async def favicon():
        from fastapi.responses import Response

        return Response(status_code=204)

    return app


def _frontend_dist_dir() -> Path:
    configured_dir = os.getenv("V2A_INSPECT_UI_STATIC_DIR")
    if configured_dir:
        return Path(configured_dir)

    packaged_dir = Path("/app/web/dist")
    if packaged_dir.exists():
        return packaged_dir

    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _dev_index_path() -> Path:
    return Path(__file__).with_name("dev_index.html")


app = create_app()
