import unittest

from audyssey.settings import Settings
from audyssey.tools.planning.llm import chat_client


class LlmTests(unittest.TestCase):
    def test_chat_client_uses_settings(self):
        settings = Settings(planner_base_url='http://llm/v1', planner_model='m', planner_api_key='secret-planner-key')
        with chat_client(settings) as client:
            self.assertEqual(str(client.base_url), 'http://llm/v1/')
            self.assertEqual(client.api_key, 'secret-planner-key')
            from langsmith import Client
            from langsmith.run_helpers import tracing_context
            from openai.types.chat import ChatCompletion
            from unittest.mock import Mock, patch
            completion = ChatCompletion.model_validate({
                "id": "test", "object": "chat.completion", "created": 0,
                "model": "m", "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            })
            trace_client = Mock(spec=Client)
            with patch.object(client, "request", return_value=completion), tracing_context(enabled=True, client=trace_client):
                result = client.chat.completions.create(model=settings.planner_model, messages=[{"role": "user", "content": "hello"}])
            self.assertEqual(result.usage.total_tokens, 5)
            trace_client.create_run.assert_called_once()
            self.assertEqual(trace_client.create_run.call_args.kwargs["run_type"], "llm")
            self.assertNotIn("secret-planner-key", str(trace_client.mock_calls))

    def test_chat_client_requires_base_url(self):
        with self.assertRaises(ValueError):
            chat_client(Settings(planner_base_url=''))

    def test_chat_client_allows_keyless_local_server(self):
        with chat_client(Settings(planner_base_url='http://localhost:8000/v1', planner_api_key='')) as client:
            self.assertEqual(client.api_key, 'unused')

    def test_provider_errors_exclude_keys_from_exceptions_and_traces(self):
        from langsmith import Client
        from langsmith.run_helpers import tracing_context
        from openai import AuthenticationError
        import httpx2 as sdk_httpx
        from unittest.mock import Mock
        key = 'secret-planner-key'
        settings = Settings(planner_base_url='http://llm/v1', planner_api_key=key)
        trace_client = Mock(spec=Client)
        def respond(request):
            return sdk_httpx.Response(401, json={'error': {'message': 'Invalid API key: ' + key}})
        with chat_client(settings, transport=sdk_httpx.MockTransport(respond)) as client:
            with tracing_context(enabled=True, client=trace_client):
                with self.assertRaises(AuthenticationError) as caught:
                    client.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'hello'}])
            self.assertEqual(caught.exception.status_code, 401)
            self.assertNotIn(key, str(caught.exception))
            self.assertIn('Invalid API key: ***', str(caught.exception))
            trace_client.update_run.assert_called_once()
            self.assertNotIn(key, str(trace_client.mock_calls))
