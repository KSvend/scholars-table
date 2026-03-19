import pytest
import json
from unittest.mock import MagicMock, patch
from orchestrator.judge import Judge


class TestAnalyzeTensions:
    def test_returns_tension_tuples(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "tensions": [
                {"responder": "silencio", "target": "ironhelm", "summary": "Colonial power critique vs realist power analysis", "score": 0.9},
                {"responder": "flickerstone", "target": "ironhelm", "summary": "Constructed vs fixed interests", "score": 0.7},
            ]
        })
        judge = Judge(router=mock_router)
        responses = {
            "peacegrave": "Structural violence analysis...",
            "ironhelm": "Power dynamics dictate...",
            "silencio": "Colonial legacies persist...",
            "flickerstone": "These identities are constructed...",
        }
        tensions = judge.analyze_tensions(responses, "How should we understand the Sudan conflict?")
        assert len(tensions) == 2
        assert tensions[0]["responder"] == "silencio"
        assert tensions[0]["target"] == "ironhelm"
        assert tensions[0]["score"] > tensions[1]["score"]

    def test_handles_malformed_json(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "not valid json"
        judge = Judge(router=mock_router)
        tensions = judge.analyze_tensions({"a": "response"}, "question")
        assert tensions == []


class TestSelectNextSpeaker:
    def test_selects_speaker(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "next_speaker": "silencio",
            "reason": "Post-colonial lens most relevant to last point"
        })
        judge = Judge(router=mock_router)
        speaker = judge.select_next_speaker(
            scholars=["peacegrave", "ironhelm", "silencio", "flickerstone"],
            conversation=[{"role": "assistant", "content": "[Ironhelm] Power is what matters."}],
            turn_counts={"peacegrave": 2, "ironhelm": 2, "silencio": 1, "flickerstone": 2},
            scholar_names={"peacegrave": "Prof Peacegrave", "ironhelm": "Col Ironhelm",
                          "silencio": "Dr Silencio", "flickerstone": "Dr Flickerstone"},
        )
        assert speaker == "silencio"

    def test_falls_back_to_least_spoken_on_error(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "broken"
        judge = Judge(router=mock_router)
        speaker = judge.select_next_speaker(
            scholars=["peacegrave", "ironhelm", "silencio"],
            conversation=[],
            turn_counts={"peacegrave": 3, "ironhelm": 2, "silencio": 1},
            scholar_names={"peacegrave": "P", "ironhelm": "I", "silencio": "S"},
        )
        assert speaker == "silencio"


class TestCheckConvergence:
    def test_detects_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "new_concepts": ["none significant"],
            "converged": True
        })
        judge = Judge(router=mock_router)
        result = judge.check_convergence([
            {"role": "assistant", "content": "[Peacegrave] Same point again..."},
            {"role": "assistant", "content": "[Ironhelm] I agree for once..."},
            {"role": "assistant", "content": "[Silencio] We seem to be circling..."},
        ])
        assert result is True

    def test_no_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "new_concepts": ["deterrence theory", "hybrid peace", "epistemic justice"],
            "converged": False
        })
        judge = Judge(router=mock_router)
        result = judge.check_convergence([
            {"role": "assistant", "content": "New point 1"},
            {"role": "assistant", "content": "New point 2"},
            {"role": "assistant", "content": "New point 3"},
        ])
        assert result is False

    def test_handles_error_as_no_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "broken"
        judge = Judge(router=mock_router)
        result = judge.check_convergence([{"role": "assistant", "content": "x"}] * 3)
        assert result is False
