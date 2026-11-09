from __future__ import annotations

import html
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


@dataclass
class AuthSettings:
    password: str | None
    cookie_name: str = "v2a_inspect_auth"
    max_age_seconds: int = 604800

    @classmethod
    def from_env(cls) -> AuthSettings:
        return cls(
            password=_read_setting("V2A_INSPECT_UI_PASSWORD"),
            cookie_name=os.getenv("V2A_INSPECT_AUTH_COOKIE", "v2a_inspect_auth"),
            max_age_seconds=int(
                os.getenv("V2A_INSPECT_AUTH_MAX_AGE_SECONDS", "604800")
            ),
        )


@dataclass
class AuthState:
    settings: AuthSettings
    sessions: set[str] = field(default_factory=set)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.password)

    def create_session(self) -> str:
        token = secrets.token_urlsafe(32)
        self.sessions.add(token)
        return token

    def clear_session(self, token: str | None) -> None:
        if token:
            self.sessions.discard(token)

    def valid_request(self, request: Request) -> bool:
        if not self.enabled:
            return True
        session = request.cookies.get(self.settings.cookie_name)
        if session in self.sessions:
            return True
        bearer = _bearer_token(request.headers.get("authorization"))
        if self.valid_password(bearer):
            return True
        return self.valid_password(request.query_params.get("token"))

    def valid_password(self, value: str | None) -> bool:
        password = self.settings.password
        if not value or password is None:
            return False
        return secrets.compare_digest(value, password)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, auth_state: AuthState) -> None:
        super().__init__(app)
        self._auth_state = auth_state

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if _is_public_path(request.url.path) or self._auth_state.valid_request(request):
            return await call_next(request)
        if _wants_api_response(request.url.path):
            return PlainTextResponse("Unauthorized", status_code=401)
        return RedirectResponse(url="/login", status_code=303)


def install_auth(app: FastAPI) -> None:
    auth_state = AuthState(AuthSettings.from_env())
    app.state.auth_state = auth_state
    app.add_middleware(AuthMiddleware, auth_state=auth_state)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login")
    async def login_page(request: Request) -> Response:
        if auth_state.valid_request(request):
            return _redirect_home()
        return HTMLResponse(_login_html())

    @app.post("/login")
    async def login(password: str = Form()) -> Response:
        if not auth_state.enabled:
            return _redirect_home()
        if not auth_state.valid_password(password):
            return HTMLResponse(_login_html("Invalid password"), status_code=401)
        response = _redirect_home()
        response.set_cookie(
            auth_state.settings.cookie_name,
            auth_state.create_session(),
            httponly=True,
            max_age=auth_state.settings.max_age_seconds,
            samesite="lax",
        )
        return response

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        auth_state.clear_session(request.cookies.get(auth_state.settings.cookie_name))
        response = _redirect_home()
        response.delete_cookie(auth_state.settings.cookie_name)
        return response


def _read_setting(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    secret_path = Path("/run/secrets") / name
    if secret_path.exists():
        secret_value = secret_path.read_text(encoding="utf-8").strip()
        if secret_value:
            return secret_value
    return None


def _bearer_token(value: str | None) -> str | None:
    if value is None:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token


def _is_public_path(path: str) -> bool:
    return path in {"/login", "/healthz"} or path.startswith("/favicon")


def _wants_api_response(path: str) -> bool:
    return path.startswith("/api/") or path == "/events"


def _redirect_home() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=303)


def _login_html(error: str | None = None) -> str:
    error_html = "" if error is None else f'<p class="error">{html.escape(error)}</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2A Inspect Login</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #111827;
      color: #e5e7eb;
      font-family: Inter, system-ui, sans-serif;
    }}
    form {{
      width: min(360px, calc(100vw - 32px));
      display: grid;
      gap: 12px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    label {{ display: grid; gap: 6px; color: #cbd5e1; }}
    input, button {{
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
    }}
    input {{ background: #020617; color: #f8fafc; }}
    button {{ background: #2563eb; color: white; cursor: pointer; }}
    .error {{ color: #fca5a5; margin: 0; }}
  </style>
</head>
<body>
  <form method="post" action="/login">
    <h1>V2A Inspect</h1>
    {error_html}
    <label>
      Password
      <input name="password" type="password" autocomplete="current-password" autofocus>
    </label>
    <button type="submit">Log in</button>
  </form>
</body>
</html>"""
