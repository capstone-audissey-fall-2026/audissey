from langsmith.wrappers import wrap_openai
from openai import APIStatusError, OpenAI

from audyssey.settings import Settings


class PlannerClient(OpenAI):
    def _make_status_error(
        self, err_msg: str, *, body: object, response
    ) -> APIStatusError:
        # Providers can echo API keys in error bodies; sanitize before tracing.
        return super()._make_status_error(
            f"planner server failed: {response.status_code}",
            body=None,
            response=response,
        )


def chat_client(settings: Settings) -> OpenAI:
    """Return a traced client; callers pass settings.planner_model to create()."""
    if not settings.planner_base_url.strip():
        raise ValueError("AUDYSSEY_PLANNER_BASE_URL is required")
    return wrap_openai(
        PlannerClient(
            base_url=settings.planner_base_url,
            api_key=settings.planner_api_key or "unused",
        )
    )
