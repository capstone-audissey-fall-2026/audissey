import asyncio
from typing import Any, Optional

import httpx

from ..config.settings import settings


class ClientError(Exception):
    """Custom exception for client errors."""

    pass


RETRY_STATUS_CODES = {502, 503, 504}


class BaseClient:
    """Base client for interacting with the inference server."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or settings.server_url).rstrip("/")
        self.timeout = timeout or settings.timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an HTTP request and handle errors."""
        if not self._client:
            raise RuntimeError("Client must be used as an async context manager")

        method_name = method.upper()
        url = f"{self.base_url}{endpoint}"
        max_retries = max(0, settings.max_retries)
        for attempt in range(max_retries + 1):
            try:
                _rewind_files(files)
                response = await self._client.request(
                    method_name,
                    url,
                    params=params,
                    json=json,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if not _should_retry_http_status(
                    exc.response.status_code, attempt, max_retries
                ):
                    raise _client_error_from_status(method_name, endpoint, exc) from exc
                await _sleep_before_retry(attempt)
            except httpx.RequestError as exc:
                if attempt >= max_retries:
                    raise ClientError(
                        f"{method_name} {endpoint} failed with "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                await _sleep_before_retry(attempt)

        raise RuntimeError("unreachable retry loop exit")


def _should_retry_http_status(status_code: int, attempt: int, max_retries: int) -> bool:
    return status_code in RETRY_STATUS_CODES and attempt < max_retries


async def _sleep_before_retry(attempt: int) -> None:
    backoff = max(0.0, settings.retry_backoff_seconds)
    if backoff == 0:
        return
    await asyncio.sleep(backoff * (2**attempt))


def _client_error_from_status(
    method_name: str, endpoint: str, exc: httpx.HTTPStatusError
) -> ClientError:
    try:
        error_detail = exc.response.json()
    except Exception:
        error_detail = exc.response.text
    return ClientError(
        f"{method_name} {endpoint} failed with "
        f"HTTP {exc.response.status_code}: {error_detail}"
    )


def _rewind_files(files: Optional[dict[str, Any]]) -> None:
    if files is None:
        return
    for value in files.values():
        if isinstance(value, tuple) and len(value) >= 2:
            file_obj = value[1]
            if _is_seekable_binary(file_obj):
                file_obj.seek(0)


def _is_seekable_binary(value: object) -> bool:
    return hasattr(value, "seek") and callable(value.seek)
