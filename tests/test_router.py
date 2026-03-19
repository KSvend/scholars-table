import pytest
from unittest.mock import patch, MagicMock
from orchestrator.router import LLMRouter


class TestLLMRouter:
    def test_init_free_tier(self):
        router = LLMRouter(tier="free")
        assert router.tier == "free"
        assert router.model == "meta-llama/Llama-3.3-70B-Instruct"

    def test_init_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            LLMRouter(tier="quantum")

    @patch("orchestrator.router.InferenceClient")
    def test_generate_calls_client(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        mock_client_class.return_value = mock_client

        router = LLMRouter(tier="free")
        result = router.generate(
            system_prompt="You are a scholar.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Test response"
        mock_client.chat_completion.assert_called_once()

    @patch("orchestrator.router.InferenceClient")
    def test_generate_with_fallback_on_error(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = [
            RuntimeError("Rate limited"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback response"))]),
        ]
        mock_client_class.return_value = mock_client

        router = LLMRouter(tier="free")
        result = router.generate(
            system_prompt="You are a scholar.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Fallback response"
        assert mock_client.chat_completion.call_count == 2
        # Verify fallback used the fallback model
        second_call_kwargs = mock_client.chat_completion.call_args_list[1]
        assert second_call_kwargs[1]["model"] == "meta-llama/Llama-3.1-8B-Instruct"
