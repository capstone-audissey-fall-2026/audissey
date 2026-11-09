from langsmith.wrappers import wrap_openai
from openai import OpenAI

from audyssey.settings import Settings


def chat_client(settings: Settings) -> OpenAI:
    """Return a traced client; callers pass settings.planner_model to create()."""
    if not settings.planner_base_url.strip():
        raise ValueError("AUDYSSEY_PLANNER_BASE_URL is required")
    return wrap_openai(
        OpenAI(
            base_url=settings.planner_base_url,
            api_key=settings.planner_api_key or "unused",
        )
    )
