from __future__ import annotations

import os
from typing import Annotated

import typer
import uvicorn


def ui(
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind the UI server to."),
    ] = os.getenv("V2A_INSPECT_UI_HOST", "127.0.0.1"),
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind the UI server to."),
    ] = int(os.getenv("V2A_INSPECT_UI_PORT", "8501")),
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Reload the UI server on code changes."),
    ] = False,
) -> None:
    """Run the v2a-inspect UI server."""

    uvicorn.run(
        "v2a_inspect.ui.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    typer.run(ui)


if __name__ == "__main__":
    main()
