from __future__ import annotations

from threading import Lock
from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from v2a_inspect.config import Settings, settings

T = TypeVar("T")


class ChatModelManager:
    """Construct and cache shared LangChain chat models by capacity tier."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._models: dict[str, BaseChatModel] = {}
        self._lock = Lock()

    @property
    def small(self) -> BaseChatModel:
        return self._get("small", self.settings.llm_small_model)

    @property
    def medium(self) -> BaseChatModel:
        return self._get("medium", self.settings.llm_medium_model)

    @property
    def large(self) -> BaseChatModel:
        return self._get("large", self.settings.llm_large_model)

    def clear_cache(self) -> None:
        with self._lock:
            self._models.clear()

    def _get(self, tier: str, model: str) -> BaseChatModel:
        with self._lock:
            if tier not in self._models:
                self._models[tier] = self._build(tier, model)
            return self._models[tier]

    def _build(self, tier: str, model: str) -> BaseChatModel:
        api_key = self._api_key()
        if self.settings.llm_base_url is not None:
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, object] = {
                "model": model,
                "base_url": self.settings.llm_base_url,
                "api_key": SecretStr(api_key or "unused"),
                "temperature": self.settings.llm_temperature,
                "max_retries": self.settings.llm_max_retries,
                "timeout": self.settings.llm_timeout_seconds,
                "use_responses_api": False,
            }
            return ChatOpenAI(**cast(Any, kwargs))

        if api_key is None:
            raise ValueError(
                "Set V2A_INSPECT_LLM_API_KEY, V2A_LLM_API_KEY, GEMINI_API_KEY, "
                "or API_KEY to build Gemini chat models."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, object] = {
            "model": model,
            "api_key": SecretStr(api_key),
            "temperature": self.settings.llm_temperature,
            "retries": self.settings.llm_max_retries,
            "request_timeout": self.settings.llm_timeout_seconds,
        }
        thinking_level = self._thinking_level(tier)
        thinking_budget = self._thinking_budget(tier)
        if thinking_level is not None:
            kwargs["thinking_level"] = thinking_level
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        return ChatGoogleGenerativeAI(**cast(Any, kwargs))

    def _api_key(self) -> str | None:
        if self.settings.llm_api_key is None:
            return None
        return self.settings.llm_api_key.get_secret_value()

    def _thinking_level(self, tier: str) -> str | None:
        if tier == "small":
            return _specific_or_default(
                self.settings.llm_small_thinking_level,
                self.settings.llm_thinking_level,
            )
        if tier == "medium":
            return _specific_or_default(
                self.settings.llm_medium_thinking_level,
                self.settings.llm_thinking_level,
            )
        if tier == "large":
            return _specific_or_default(
                self.settings.llm_large_thinking_level,
                self.settings.llm_thinking_level,
            )
        return None

    def _thinking_budget(self, tier: str) -> int | None:
        if tier == "small":
            return _specific_or_default(
                self.settings.llm_small_thinking_budget,
                self.settings.llm_thinking_budget,
            )
        if tier == "medium":
            return _specific_or_default(
                self.settings.llm_medium_thinking_budget,
                self.settings.llm_thinking_budget,
            )
        if tier == "large":
            return _specific_or_default(
                self.settings.llm_large_thinking_budget,
                self.settings.llm_thinking_budget,
            )
        return None


def _specific_or_default(value: T | None, default: T | None) -> T | None:
    if value is None:
        return default
    return value


model_manager = ChatModelManager()
