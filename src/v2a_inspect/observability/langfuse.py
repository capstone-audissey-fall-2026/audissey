from __future__ import annotations

from importlib import import_module
import logging
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from v2a_inspect.config import settings

logger = logging.getLogger(__name__)

_UNINITIALIZED = object()
_langfuse_client: Any | None | object = _UNINITIALIZED


def is_langfuse_enabled() -> bool:
    return (
        settings.langfuse_public_key is not None
        and settings.langfuse_secret_key is not None
    )


def get_langfuse_client() -> Any | None:
    global _langfuse_client

    if _langfuse_client is not _UNINITIALIZED:
        return _langfuse_client

    if not is_langfuse_enabled():
        _langfuse_client = None
        return None

    try:
        langfuse_module = import_module("langfuse")
    except ImportError as exc:
        raise RuntimeError(
            "Install v2a-inspect[observability] to use Langfuse tracing."
        ) from exc

    langfuse_class = langfuse_module.Langfuse
    _langfuse_client = langfuse_class(
        public_key=settings.langfuse_public_key.get_secret_value()
        if settings.langfuse_public_key is not None
        else None,
        secret_key=settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key is not None
        else None,
        base_url=settings.langfuse_base_url,
    )
    return _langfuse_client


def create_langfuse_handler() -> Any | None:
    try:
        if get_langfuse_client() is None:
            return None
        langfuse_langchain_module = import_module("langfuse.langchain")
        callback_handler_class = langfuse_langchain_module.CallbackHandler
        handler = callback_handler_class()
        if hasattr(handler, "raise_error"):
            handler.raise_error = False
        return handler
    except Exception as exc:  # noqa: BLE001 - tracing must not break the pipeline.
        logger.warning("Langfuse tracing disabled: %s", exc)
        return None


def build_langchain_config(
    *,
    run_name: str,
    tags: list[str],
    metadata: dict[str, Any],
) -> RunnableConfig | None:
    handler = create_langfuse_handler()
    if handler is None:
        return None

    trace_metadata = dict(metadata)
    trace_metadata["langfuse_environment"] = settings.langfuse_environment
    if settings.langfuse_release is not None:
        trace_metadata["langfuse_release"] = settings.langfuse_release

    return cast(
        RunnableConfig,
        {
            "callbacks": [handler],
            "run_name": run_name,
            "tags": tags,
            "metadata": trace_metadata,
        },
    )


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()
