# tests/test_persona_loader.py
import pytest
import os
import tempfile
import yaml
from scholars.persona_loader import load_persona, validate_persona, load_all_personas


MINIMAL_VALID_PERSONA = {
    "name": "Professor Test Scholar",
    "school": "Test School",
    "title": "Professor",
    "personality": {
        "background": "A test scholar.",
        "emotional_patterns": {
            "passionate_about": ["testing"],
            "frustrated_by": ["bugs"],
            "curious_about": ["code"],
        },
        "humor_style": "dry",
        "when_wrong": "concedes",
        "when_challenged": "debates calmly",
    },
    "intellectual": {
        "core_concepts": ["unit testing", "integration testing"],
        "reasoning_pattern": "Start with the test, then implement.",
        "blind_spots": ["over-engineering"],
        "internal_tensions": ["mocks vs real deps"],
        "changed_mind_about": ["TDD strictness"],
        "borrowed_concepts": [
            {
                "concept": "property testing",
                "from_tradition": "Haskell",
                "attitude": "open",
            }
        ],
    },
    "relationships": {
        "ironhelm": {
            "stance": "respectful_rival",
            "dynamic": "Disagrees on method but respects rigor",
            "triggers": ["power", "realism"],
            "common_ground": ["Both value evidence"],
        }
    },
    "rhetorical": {
        "sentence_style": "concise and direct",
        "opening_move": "Let me frame this differently.",
        "argument_method": "deductive",
        "signature_phrases": ["The evidence suggests..."],
        "citation_style": "conversational",
    },
    "key_thinkers": ["Karl Popper"],
}


class TestValidatePersona:
    def test_valid_persona_passes(self):
        errors = validate_persona(MINIMAL_VALID_PERSONA)
        assert errors == []

    def test_missing_name_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        del persona["name"]
        errors = validate_persona(persona)
        assert any("name" in e for e in errors)

    def test_missing_school_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        del persona["school"]
        errors = validate_persona(persona)
        assert any("school" in e for e in errors)

    def test_missing_personality_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        del persona["personality"]
        errors = validate_persona(persona)
        assert any("personality" in e for e in errors)

    def test_missing_intellectual_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        del persona["intellectual"]
        errors = validate_persona(persona)
        assert any("intellectual" in e for e in errors)

    def test_missing_rhetorical_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        del persona["rhetorical"]
        errors = validate_persona(persona)
        assert any("rhetorical" in e for e in errors)

    def test_empty_core_concepts_fails(self):
        persona = {**MINIMAL_VALID_PERSONA}
        persona["intellectual"] = {**persona["intellectual"], "core_concepts": []}
        errors = validate_persona(persona)
        assert any("core_concepts" in e for e in errors)


class TestLoadPersona:
    def test_load_valid_yaml(self, tmp_path):
        persona_file = tmp_path / "test_scholar.yaml"
        persona_file.write_text(yaml.dump(MINIMAL_VALID_PERSONA))
        persona = load_persona(str(persona_file))
        assert persona["name"] == "Professor Test Scholar"
        assert persona["school"] == "Test School"

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_persona("/nonexistent/path.yaml")

    def test_load_invalid_schema_raises(self, tmp_path):
        persona_file = tmp_path / "bad.yaml"
        persona_file.write_text(yaml.dump({"name": "Only Name"}))
        with pytest.raises(ValueError):
            load_persona(str(persona_file))


class TestLoadAllPersonas:
    def test_loads_all_yaml_files(self, tmp_path):
        for name in ["scholar_a", "scholar_b"]:
            p = {**MINIMAL_VALID_PERSONA, "name": f"Prof {name}"}
            (tmp_path / f"{name}.yaml").write_text(yaml.dump(p))
        personas = load_all_personas(str(tmp_path))
        assert len(personas) == 2
        assert "scholar_a" in personas
        assert "scholar_b" in personas
