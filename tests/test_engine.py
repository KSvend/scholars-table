# tests/test_engine.py
import pytest
import os
from scholars.engine import ScholarEngine


class TestScholarEngine:
    def test_load_scholars(self):
        engine = ScholarEngine()
        assert "peacegrave" in engine.scholars
        assert "ironhelm" in engine.scholars

    def test_get_scholar_names(self):
        engine = ScholarEngine()
        names = engine.get_scholar_names()
        assert "Professor Galthorn Peacegrave" in names.values()

    def test_build_system_prompt_mode1(self):
        engine = ScholarEngine()
        prompt = engine.build_system_prompt("peacegrave", mode="private")
        # Mode 1 includes all layers
        assert "Peacegrave" in prompt
        assert "violence triangle" in prompt.lower() or "structural violence" in prompt.lower()
        assert "signature_phrases" in prompt.lower() or "question is not whether" in prompt.lower()
        # Should include relationship info in full mode
        assert "ironhelm" in prompt.lower() or "Ironhelm" in prompt

    def test_build_system_prompt_multi_mode(self):
        engine = ScholarEngine()
        prompt = engine.build_system_prompt("peacegrave", mode="multi")
        # Multi mode has core + intellectual but is shorter than private
        assert "Peacegrave" in prompt
        private_prompt = engine.build_system_prompt("peacegrave", mode="private")
        assert len(prompt) < len(private_prompt)

    def test_build_system_prompt_invalid_scholar_raises(self):
        engine = ScholarEngine()
        with pytest.raises(KeyError):
            engine.build_system_prompt("nonexistent", mode="private")

    def test_build_system_prompt_with_responding_to(self):
        engine = ScholarEngine()
        prompt = engine.build_system_prompt(
            "peacegrave", mode="multi", responding_to="ironhelm"
        )
        # Should inject relationship dynamic
        assert "power-maximizer" in prompt.lower() or "morally bankrupt" in prompt.lower() or "ironhelm" in prompt.lower()
