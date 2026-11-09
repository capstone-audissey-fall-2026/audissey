from typing import Annotated, Literal

import typer
import uvicorn

from audyssey.api import create_app
from audyssey.settings import Settings
from audyssey.toolserver import create_tool_app

app = typer.Typer()


@app.callback()
def callback() -> None:
    """Audyssey audio editing skeleton."""


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8600) -> None:
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def tool_server(
    tool: Annotated[Literal["separation", "filtering"], typer.Option()],
    impl: Annotated[Literal["stub", "real"], typer.Option()] = "stub",
    host: str = "127.0.0.1",
    port: int = 8700,
) -> None:
    token = Settings().tool_token
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        typer.echo(
            "AUDYSSEY_TOOL_TOKEN is required when binding to a non-loopback host"
        )
        raise typer.Exit(1)
    uvicorn.run(create_tool_app(tool, impl, token), host=host, port=port)


def main() -> None:
    app()
