import httpx


def post(
    url: str,
    path: str,
    files: dict[str, tuple[str, bytes, str]],
    data: dict[str, str],
    token: str,
    timeout: float,
) -> httpx.Response:
    endpoint = url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.post(
            endpoint, files=files, data=data, headers=headers, timeout=timeout
        )
    except httpx.RequestError as exc:
        # Transport messages and response bodies may contain credentials.
        raise RuntimeError(
            f"tool server {endpoint} failed: {type(exc).__name__}"
        ) from None
    if not response.is_success:
        raise RuntimeError(f"tool server {endpoint} failed: {response.status_code}")
    return response
