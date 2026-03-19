import pytest
from unittest.mock import MagicMock, patch
from orchestrator.modes import PrivateConsultation


class TestPrivateConsultation:
    def test_init(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        mode = PrivateConsultation(
            scholar_id="peacegrave",
            engine=mock_engine,
            router=mock_router,
        )
        assert mode.scholar_id == "peacegrave"
        assert mode.history == []

    def test_send_message_returns_response(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "You are Peacegrave."
        mock_router = MagicMock()
        mock_router.generate.return_value = "Let me map the violence triangle here..."

        mode = PrivateConsultation(
            scholar_id="peacegrave",
            engine=mock_engine,
            router=mock_router,
        )

        response = mode.send_message("What do you think about the conflict in Sudan?")
        assert response == "Let me map the violence triangle here..."
        assert len(mode.history) == 2  # user msg + assistant msg

    def test_history_accumulates(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_router = MagicMock()
        mock_router.generate.side_effect = ["Response 1", "Response 2"]

        mode = PrivateConsultation(
            scholar_id="peacegrave",
            engine=mock_engine,
            router=mock_router,
        )

        mode.send_message("First question")
        mode.send_message("Follow-up")
        assert len(mode.history) == 4  # 2 user + 2 assistant

    def test_reset_clears_history(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_router = MagicMock()
        mock_router.generate.return_value = "Response"

        mode = PrivateConsultation(
            scholar_id="peacegrave",
            engine=mock_engine,
            router=mock_router,
        )

        mode.send_message("Question")
        mode.reset()
        assert mode.history == []

    def test_system_prompt_uses_private_mode(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Full prompt"
        mock_router = MagicMock()
        mock_router.generate.return_value = "Response"

        mode = PrivateConsultation(
            scholar_id="peacegrave",
            engine=mock_engine,
            router=mock_router,
        )

        mode.send_message("Hello")
        mock_engine.build_system_prompt.assert_called_with("peacegrave", mode="private")
