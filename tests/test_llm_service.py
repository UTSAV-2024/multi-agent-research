# ==========================================
# LLM SERVICE TESTS
# ==========================================
#
# Tests for run_agent() with the Groq
# client wrapper (sync SDK via
# asyncio.to_thread).
#
# ==========================================

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock


class TestRunAgent:
    """Tests for run_agent()."""

    # ---------------------------------------------------------------
    # HELPER
    # ---------------------------------------------------------------

    def _run(self, coro):
        """Run an async test."""
        return asyncio.run(coro)

    def _make_mock_client(self):
        """Create a mock Groq client with a sync mock chat.completions.create.

        Groq SDK is synchronous, so the mock must be a regular MagicMock
        (not AsyncMock) because asyncio.to_thread calls it in a thread
        pool and expects a normal synchronous return value.
        """
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Hello, world!  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(return_value=mock_response)
        mock_client.chat.completions.create = mock_create
        return mock_client, mock_create, mock_response

    # ---------------------------------------------------------------
    # TESTS: SUCCESS PATH
    # ---------------------------------------------------------------

    @patch("app.services.llm_service._get_client")
    def test_returns_stripped_text(self, mock_get_client):
        """run_agent returns the response content, stripped."""
        from app.services.llm_service import run_agent

        mock_client, mock_create, mock_response = self._make_mock_client()
        mock_response.choices[0].message.content = "  Hello, world!  "
        mock_get_client.return_value = mock_client

        result = self._run(run_agent(
            system_prompt="You are a helper.",
            user_prompt="Say hello.",
            max_tokens=100,
        ))

        assert result == "Hello, world!"

    @patch("app.services.llm_service._get_client")
    def test_passes_correct_model(self, mock_get_client):
        """run_agent passes settings.MODEL_NAME to create()."""
        from app.services.llm_service import run_agent
        from app.config.settings import settings

        mock_client, mock_create, _ = self._make_mock_client()
        mock_get_client.return_value = mock_client

        self._run(run_agent("sys", "hello"))

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == settings.MODEL_NAME

    @patch("app.services.llm_service._get_client")
    def test_passes_messages(self, mock_get_client):
        """run_agent passes system + user messages."""
        from app.services.llm_service import run_agent

        mock_client, mock_create, _ = self._make_mock_client()
        mock_get_client.return_value = mock_client

        self._run(run_agent("Be helpful.", "Hello!"))

        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    @patch("app.services.llm_service._get_client")
    def test_passes_temperature(self, mock_get_client):
        """Temperature from settings is passed to create()."""
        from app.services.llm_service import run_agent
        from app.config.settings import settings

        mock_client, mock_create, _ = self._make_mock_client()
        mock_get_client.return_value = mock_client

        self._run(run_agent("sys", "hi"))

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == settings.TEMPERATURE

    @patch("app.services.llm_service._get_client")
    def test_passes_max_tokens(self, mock_get_client):
        """max_tokens is passed to create()."""
        from app.services.llm_service import run_agent

        mock_client, mock_create, _ = self._make_mock_client()
        mock_get_client.return_value = mock_client

        self._run(run_agent("sys", "hi", max_tokens=500))

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    # ---------------------------------------------------------------
    # TESTS: RETRY BEHAVIOR
    # ---------------------------------------------------------------

    @patch("app.services.llm_service._get_client")
    def test_retry_on_exception(self, mock_get_client):
        """Retries when the API call raises an exception."""
        from app.services.llm_service import run_agent

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(
            side_effect=[RuntimeError("API error"), mock_response]
        )
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        result = self._run(run_agent("sys", "hi"))

        assert result == "OK"
        assert mock_create.call_count == 2

    @patch("app.services.llm_service._get_client")
    def test_retry_on_timeout(self, mock_get_client):
        """Retries on asyncio.TimeoutError."""
        from app.services.llm_service import run_agent

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(
            side_effect=[asyncio.TimeoutError(), mock_response]
        )
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        result = self._run(run_agent("sys", "hi"))

        assert result == "OK"
        assert mock_create.call_count == 2

    @patch("app.services.llm_service._get_client")
    def test_max_retries_exceeded_returns_fallback(self, mock_get_client):
        """When all retries fail, returns empty JSON fallback."""
        from app.services.llm_service import run_agent
        from app.config.settings import settings

        mock_client = MagicMock()
        mock_create = MagicMock(
            side_effect=RuntimeError("Persistent error")
        )
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        result = self._run(run_agent("sys", "hi"))

        assert mock_create.call_count == settings.MAX_RETRIES
        assert "facts" in result

    @patch("app.services.llm_service._get_client")
    def test_semaphore_serializes_calls(self, mock_get_client):
        """The semaphore prevents concurrent LLM calls."""
        from app.services.llm_service import run_agent

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "done"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(return_value=mock_response)
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        async def run_concurrent():
            async with asyncio.TaskGroup() as tg:
                t1 = tg.create_task(run_agent("sys", "a"))
                t2 = tg.create_task(run_agent("sys", "b"))
            return t1.result(), t2.result()

        r1, r2 = asyncio.run(run_concurrent())
        assert r1 == "done"
        assert r2 == "done"
        assert mock_create.call_count == 2

    # ---------------------------------------------------------------
    # TESTS: TIMEOUT WRAPPING
    # ---------------------------------------------------------------

    @patch("app.services.llm_service._get_client")
    def test_call_wrapped_in_wait_for(self, mock_get_client):
        """The Groq call is wrapped in asyncio.wait_for (60s timeout)."""
        from app.services.llm_service import run_agent

        mock_client, mock_create, _ = self._make_mock_client()
        mock_get_client.return_value = mock_client

        self._run(run_agent("sys", "hi"))

        mock_create.assert_called_once()

    # ---------------------------------------------------------------
    # TESTS: MALFORMED RESPONSES (JSON COMPATIBILITY)
    # ---------------------------------------------------------------

    @patch("app.services.llm_service._get_client")
    def test_handles_empty_response(self, mock_get_client):
        """Empty response content is handled gracefully."""
        from app.services.llm_service import run_agent

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(return_value=mock_response)
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        result = self._run(run_agent("sys", "hi"))
        assert result == ""

    @patch("app.services.llm_service._get_client")
    def test_preserves_malformed_content(self, mock_get_client):
        """Malformed LLM output is passed through for downstream JSON parsing."""
        from app.services.llm_service import run_agent

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "```json\n{\"key\": \"value\"}\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_create = MagicMock(return_value=mock_response)
        mock_client.chat.completions.create = mock_create
        mock_get_client.return_value = mock_client

        result = self._run(run_agent("sys", "hi"))
        assert "```json" in result
