import json
import pytest
from unittest.mock import MagicMock, patch
from orchestrator.modes import PrivateConsultation, PanelDiscussion, FreeDebate


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


class TestPanelDiscussion:
    def test_init(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm", "silencio"],
            engine=mock_engine,
            router=mock_router,
        )
        assert panel.scholar_ids == ["peacegrave", "ironhelm", "silencio"]
        assert panel.phase == "waiting"
        assert panel.conversation == []

    def test_initial_round_generates_all_responses(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = ["Peace response", "Realist response"]

        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
        )
        responses = list(panel.start_discussion("What about Sudan?"))
        assert len(responses) == 2
        assert "Peace response" in responses[0]["content"]
        assert "Realist response" in responses[1]["content"]
        assert panel.phase == "rebuttals"

    def test_initial_round_skips_failed_scholar(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = [Exception("API down"), "Realist response"]

        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
        )
        responses = list(panel.start_discussion("Question"))
        scholar_responses = [r for r in responses if "unavailable" not in r["content"]]
        assert len(scholar_responses) == 1

    def test_user_interjection(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {"peacegrave": "P"}
        mock_router = MagicMock()
        mock_router.generate.return_value = "Response"

        panel = PanelDiscussion(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        list(panel.start_discussion("Initial question"))
        panel.add_interjection("But what about power?")
        assert any(m["role"] == "user" for m in panel.conversation)

    def test_reset(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        panel = PanelDiscussion(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        panel.conversation = [{"role": "user", "content": "test"}]
        panel.reset()
        assert panel.conversation == []
        assert panel.phase == "waiting"


class TestFreeDebate:
    def test_init(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=10,
        )
        assert debate.max_exchanges == 10
        assert debate.exchange_count == 0
        assert debate.running is False

    def test_opening_round(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = ["Opening 1", "Opening 2"]

        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=10,
        )
        responses = list(debate.start("What is peace?"))
        assert len(responses) == 2
        assert debate.exchange_count == 2
        assert debate.running is True

    def test_next_turn_uses_judge(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "P", "ironhelm": "I",
        }
        mock_router = MagicMock()
        mock_router.generate.return_value = "Next response"

        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=20,
        )
        debate.running = True
        debate.exchange_count = 2
        debate.conversation = [
            {"role": "user", "content": "Topic"},
            {"role": "assistant", "content": "R1"},
            {"role": "assistant", "content": "R2"},
        ]
        debate.turn_counts = {"peacegrave": 1, "ironhelm": 1}

        with patch.object(debate.judge, 'select_next_speaker', return_value="peacegrave"):
            result = debate.next_turn()
            assert result is not None
            assert debate.exchange_count == 3

    def test_stops_at_max_exchanges(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "P"
        mock_engine.get_scholar_names.return_value = {"peacegrave": "P"}
        mock_router = MagicMock()
        mock_router.generate.return_value = "R"

        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=2,
        )
        debate.running = True
        debate.exchange_count = 2
        debate.conversation = [{"role": "assistant", "content": "x"}] * 2
        debate.turn_counts = {"peacegrave": 2}

        with patch.object(debate.judge, 'select_next_speaker', return_value="peacegrave"):
            result = debate.next_turn()
            assert result is None
            assert debate.running is False

    def test_interjection(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        debate.running = True
        debate.add_interjection("What about gender?")
        assert debate.conversation[-1]["role"] == "user"

    def test_stop(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        debate.running = True
        debate.stop()
        assert debate.running is False
