import typer
import uvicorn

from audyssey.api import create_app

app = typer.Typer()


@app.callback()
def callback() -> None:
    """Audyssey audio editing skeleton."""


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8600) -> None:
    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    app()
